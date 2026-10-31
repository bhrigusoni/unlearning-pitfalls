"""Item-level paired analysis of the localization x SAM grid.

Reads only the JSON that evaluate.py and relearn.py already write, never torch, so
this runs on a laptop rather than the GPU box.

Scope of inference: one seed means one model per cell of the grid, so there is no
method-level error term. Every test here pairs over ITEMS (16 forget athletes, 256
held-out athletes) and answers "is this advantage consistent across these items for
these particular checkpoints". It does not answer "would this reappear at another
seed". Reporting any of it as "significantly better method" is pseudo-replication.
"""
import argparse
import csv
import itertools
import math
import random
from pathlib import Path

from prepare_data import (ALL_METHODS, LEVELS, READOUT_STEP, arm_dir, attack_dir, attack_tag, config, level_step,
                          read_json, stage_cfg, write_json)

FIELD = 'normalized_target_probability'
ARMS = ('base',) + ALL_METHODS
# Both axes oriented so that higher is better, which keeps the headline scatter's
# best corner at top-right and makes every contrast a one-directional claim.
AXES = ('collateral', 'resistance')
CELLS = {'neither': 'ul', 'flu': 'ul_flu', 'sam': 'ul_sam', 'both': 'ul_flu_sam'}
PAIRS = (('ul_flu_sam', 'ul_flu'), ('ul_flu_sam', 'ul_sam'), ('ul_flu_sam', 'ul'),
         ('ul_flu', 'ul_rand'), ('ul_flu_sam', 'ul_rand_sam'), ('ul_flu', 'ul'), ('ul_sam', 'ul'))
# Reviewer controls, a separate Holm family so the seven PAIRS keep their adjustment.
# FLU against random layers drawn outside FLU (randx) and also depth-matched (randd);
# randx against the original rand asks whether the shared layers changed the result.
CONTROL_PAIRS = (('ul_flu', 'ul_randx'), ('ul_flu_sam', 'ul_randx_sam'),
                 ('ul_flu', 'ul_randd'), ('ul_flu_sam', 'ul_randd_sam'),
                 ('ul_randx', 'ul_rand'), ('ul_randx_sam', 'ul_rand_sam'))
FAMILIES = (('main', PAIRS), ('control', CONTROL_PAIRS))
# The base model must keep at least this retain and forget accuracy at READOUT_STEP
# for the attack to count as an attack rather than a divergence.
VALID_ACCURACY = 0.95


def load_records(cfg, arm, split, step=None):
    """{athlete: record} for one arm/split, pre-attack or at a relearning step."""
    path = (arm_dir(cfg, arm) / 'evaluation.json' if step is None
            else attack_dir(cfg, arm) / f'evaluation_{step:04d}.json')
    return {r['athlete']: r for r in read_json(path)[split]['records']}


def item_auc(cfg, arm, split='forget', field=FIELD):
    """Per-athlete mean of `field` over the relearning trajectory.

    Trapezoidal integral divided by the final step, matching run_trial.aggregate's
    recovery_accuracy_auc so the per-item and per-arm numbers are comparable.
    """
    steps = [c['step'] for c in read_json(attack_dir(cfg, arm) / 'curve.json')]
    series = {step: load_records(cfg, arm, split, step) for step in steps}
    return {athlete: sum((b - a) * (series[a][athlete][field] + series[b][athlete][field]) / 2
                         for a, b in zip(steps, steps[1:])) / max(1, steps[-1])
            for athlete in series[steps[0]]}


def outcome(cfg, arm, axis):
    """Per-athlete outcome on one axis, always oriented higher-is-better."""
    if axis == 'collateral':
        # Pre-attack forget_heldout is the only probe disjoint from the forget set and
        # absent from both loss terms. retain is inside the loss; post-attack
        # forget_heldout is the attack's own training data.
        return {a: r[FIELD] for a, r in load_records(cfg, arm, 'forget_heldout').items()}
    if axis == 'resistance':
        return {a: 1.0 - v for a, v in item_auc(cfg, arm, 'forget').items()}
    raise ValueError(f'Unknown axis: {axis}')


def differences(left, right):
    """Paired left-minus-right, keyed on athlete so misalignment cannot pass silently."""
    if left.keys() != right.keys():
        missing = set(left) ^ set(right)
        raise ValueError(f'Item sets differ ({len(missing)} unmatched, e.g. {sorted(missing)[:3]}).')
    return [left[a] - right[a] for a in sorted(left)]


def binomial_p(k, n):
    """Exact two-sided binomial test of k successes in n against p = 1/2."""
    if not n:
        return 1.0
    extreme = max(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(extreme, n + 1)) / 2 ** n)


def sign_test(values):
    """Exact two-sided binomial test on the signs. Assumption-free fallback."""
    values = [v for v in values if v]
    positive = sum(v > 0 for v in values)
    return {'n': len(values), 'positive': positive, 'p': binomial_p(positive, len(values))}


def mcnemar(left, right):
    """Exact McNemar test on paired booleans keyed by athlete (same facts, two arms)."""
    if left.keys() != right.keys():
        raise ValueError('McNemar needs the same items in both arms.')
    only_left = sum(left[a] and not right[a] for a in left)
    only_right = sum(right[a] and not left[a] for a in left)
    return {'only_left': only_left, 'only_right': only_right, 'p': binomial_p(only_left, only_left + only_right)}


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact test on k1/n1 vs k2/n2: sums tables no likelier than the one seen.

    Unpaired, so it ignores that both arms are scored on the same facts; reported only
    for comparison with earlier numbers. McNemar is the paired test.
    """
    total, k = n1 + n2, k1 + k2
    def probability(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(total, k)
    observed = probability(k1)
    return min(1.0, sum(probability(x) for x in range(max(0, k - n2), min(k, n1) + 1)
                        if probability(x) <= observed * (1 + 1e-9)))


def holm(ps):
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def permutation_test(values, draws=20000, seed=0):
    """Two-sided paired sign-flip permutation test on the mean difference.

    Exact by enumerating all 2**n sign patterns when n <= 20, which covers the 16-item
    forget split; Monte Carlo above that. Assumes only that each difference is
    symmetric about zero under the null, and makes no normality assumption.
    """
    values = [v for v in values if v]
    n = len(values)
    if not n:
        return {'n': 0, 'mean': 0.0, 'p': 1.0, 'exact': True}
    observed = abs(sum(values)) - 1e-12
    if n <= 20:
        hits = sum(abs(sum(s * v for s, v in zip(signs, values))) >= observed
                   for signs in itertools.product((1, -1), repeat=n))
        return {'n': n, 'mean': sum(values) / n, 'p': hits / 2 ** n, 'exact': True}
    rng = random.Random(seed)
    hits = sum(abs(sum(v if rng.random() < 0.5 else -v for v in values)) >= observed
               for _ in range(draws))
    return {'n': n, 'mean': sum(values) / n, 'p': (hits + 1) / (draws + 1), 'exact': False}


def bootstrap_ci(values, draws=10000, alpha=0.05, seed=0):
    """Percentile bootstrap over items. Item-level uncertainty, NOT seed-level."""
    values = list(values)
    if not values:
        return {'mean': None, 'low': None, 'high': None}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws))
    return {'mean': sum(values) / len(values),
            'low': means[int(alpha / 2 * draws)], 'high': means[int((1 - alpha / 2) * draws) - 1]}


def available(cfg):
    """Trained arms that reached and were evaluated at this level; the others have no model to compare.

    An arm trained with --level forget can pass the steps or loss level on the way
    without being evaluated there.
    """
    return [arm for arm in ALL_METHODS
            if level_step(cfg, arm) is not None and (arm_dir(cfg, arm) / 'evaluation.json').exists()]


def attacked(cfg):
    """Arms at this level that this attack has run on (a sweep may cover only some)."""
    return [arm for arm in available(cfg) if (attack_dir(cfg, arm) / 'curve.json').exists()]


def gate_forgetting(cfg, floor=0.05):
    """Has every arm fully forgotten: 0 correct and every item below `floor`?

    Reported, not enforced. At the steps and loss levels unmatched forgetting is part
    of what is being compared, so a failed gate is a caveat on each contrast rather
    than a reason to withhold it. The forget level passes it by construction.
    """
    rows, passed = [], True
    for arm in available(cfg):
        records = load_records(cfg, arm, 'forget')
        values = [r[FIELD] for r in records.values()]
        correct = sum(r['correct'] for r in records.values())
        ok = correct == 0 and max(values) < floor
        passed = passed and ok
        rows.append({'arm': arm, 'count': len(values), 'correct': correct,
                     'mean_normalized': sum(values) / len(values),
                     'max_normalized': max(values), 'passes_floor': ok})
    return {'floor': floor, 'chance': 1 / 3, 'passed': passed, 'arms': rows}


def attack_works(cfg):
    """The base arm must recover under the attack, or the resistance axis is void."""
    curve = read_json(attack_dir(cfg, 'base') / 'curve.json')
    start, end = curve[0]['forget'], curve[-1]['forget']
    return {'base_forget_accuracy_start': start['accuracy'], 'base_forget_accuracy_end': end['accuracy'],
            'base_normalized_start': start['mean_normalized_target_probability'],
            'base_normalized_end': end['mean_normalized_target_probability'],
            'recovers': end['mean_normalized_target_probability'] >= start['mean_normalized_target_probability']}


def known(cfg, arm, step):
    """{athlete: still answered correctly} on the forget set at one attack step (0 = pre-attack)."""
    return {a: r['correct'] for a, r in load_records(cfg, arm, 'forget', step).items()}


def readout(cfg, arms):
    """Counts at the pre-attack and early readout steps, paired tests, and attack validity.

    Recovery AUC over the whole attack mixes recovery with the damage the attack does
    after ~10 steps, so the headline is facts known at READOUT_STEP. The attack is valid
    only if the base model keeps its retain and forget accuracy at that step; a diverged
    attack drives every arm to one-class answers and makes all arms look alike.
    """
    if not (attack_dir(cfg, 'base') / 'curve.json').exists():
        return {'skipped': 'the base model was not attacked'}
    rows = []
    for arm in ('base',) + tuple(arms):
        curve = {c['step']: c for c in read_json(attack_dir(cfg, arm) / 'curve.json')}
        if READOUT_STEP not in curve:
            return {'skipped': f'no evaluation at step {READOUT_STEP}'}
        rows.append({'arm': arm, 'count': curve[0]['forget']['count'],
                     'forget_known_pre': round(curve[0]['forget']['accuracy'] * curve[0]['forget']['count']),
                     'forget_known_readout': round(curve[READOUT_STEP]['forget']['accuracy']
                                                   * curve[READOUT_STEP]['forget']['count']),
                     'retain_pre': curve[0]['retain']['accuracy'],
                     'retain_readout': curve[READOUT_STEP]['retain']['accuracy'],
                     'heldout_readout': curve[READOUT_STEP]['forget_heldout']['accuracy']})
    base = rows[0]
    valid = (base['retain_readout'] >= VALID_ACCURACY
             and base['forget_known_readout'] >= VALID_ACCURACY * base['count'])
    tests = []
    for name, pairs in FAMILIES:
        for left, right in pairs:
            if left not in arms or right not in arms:
                continue
            for step in (0, READOUT_STEP):
                l, r = known(cfg, left, step), known(cfg, right, step)
                tests.append({'family': name, 'left': left, 'right': right, 'step': step,
                              'left_known': sum(l.values()), 'right_known': sum(r.values()), 'n': len(l),
                              **{f'mcnemar_{k}': v for k, v in mcnemar(l, r).items()},
                              'fisher_p': fisher(sum(l.values()), len(l), sum(r.values()), len(r)),
                              'mcnemar_p_holm': None})
    # Holm over the readout-step tests of each family only; pre-attack tests describe
    # the starting point.
    for name, _ in FAMILIES:
        family = [t for t in tests if t['step'] == READOUT_STEP and t['family'] == name]
        for test, adjusted in zip(family, holm([t['mcnemar_p'] for t in family])):
            test['mcnemar_p_holm'] = adjusted
    return {'step': READOUT_STEP, 'valid_accuracy': VALID_ACCURACY, 'attack_valid': valid,
            'arms': rows, 'tests': tests}


def runner_up(cfg, arms):
    """Is the true sport still the runner-up on the facts an arm got wrong (pre-attack)?

    Conditional null: once the true sport is not top-1, it is 2nd or 3rd of three, so
    under no residual knowledge it is runner-up with probability 1/2. A uniform-over-3
    null (1/3) is wrong here, because rank 1 is excluded by conditioning on the error.
    """
    rows = []
    for arm in arms:
        k = n = 0
        for record in load_records(cfg, arm, 'forget').values():
            if record['correct']:
                continue
            # normalized_sport_probabilities is underflow-free; older files only have the raw ones.
            scores = record.get('normalized_sport_probabilities', record['sport_probabilities'])
            n += 1
            k += sorted(scores, key=scores.get, reverse=True).index(record['sport']) == 1
        rows.append({'arm': arm, 'wrong': n, 'true_sport_runner_up': k, 'p_conditional_half': binomial_p(k, n)})
    return rows


def contrast(cfg, left, right, axis):
    values = differences(outcome(cfg, left, axis), outcome(cfg, right, axis))
    return {'left': left, 'right': right, 'axis': axis,
            'permutation': permutation_test(values), 'sign': sign_test(values),
            'difference_ci': bootstrap_ci(values)}


def interaction(cfg, axis, arms):
    """Point estimates only: one model per cell leaves no error term for the 2x2.

    The per-item c_i is a valid item-level test of the same quantity but is a
    difference of differences over four correlated measurements, so its power is
    materially worse than a simple paired contrast. Report it, do not build on it.
    """
    if not set(CELLS.values()) <= set(arms):
        return {'axis': axis, 'skipped': 'a 2x2 cell never reached this level'}
    per_item = {name: outcome(cfg, arm, axis) for name, arm in CELLS.items()}
    means = {name: sum(v.values()) / len(v) for name, v in per_item.items()}
    athletes = sorted(per_item['both'])
    c = [per_item['both'][a] - per_item['flu'][a] - per_item['sam'][a] + per_item['neither'][a]
         for a in athletes]
    return {'axis': axis, 'means': means,
            'delta_flu': means['flu'] - means['neither'],
            'delta_sam': means['sam'] - means['neither'],
            'delta_both': means['both'] - means['neither'],
            'interaction': means['both'] - means['flu'] - means['sam'] + means['neither'],
            'per_item_interaction': permutation_test(c)}


def pareto(report, delta=0.02):
    """Conjunction claim, tested as an intersection-union test: max(p) <= alpha.

    ul_flu_sam dominates a comparator when it is better on the axis where that
    comparator is weak and no worse than `delta` on the axis where it is strong. No
    multiplicity correction is needed for a conjunction.
    """
    claims = []
    for comparator, weak, strong in (('ul_flu', 'resistance', 'collateral'),
                                     ('ul_sam', 'collateral', 'resistance')):
        better = next((c for c in report if c['left'] == 'ul_flu_sam' and c['right'] == comparator
                       and c['axis'] == weak), None)
        equal = next((c for c in report if c['left'] == 'ul_flu_sam' and c['right'] == comparator
                      and c['axis'] == strong), None)
        if better is None or equal is None:
            return {'skipped': f'ul_flu_sam or {comparator} never reached this level'}
        claims.append({'comparator': comparator, 'better_on': weak, 'noninferior_on': strong,
                       'improvement': better['permutation']['mean'], 'p': better['permutation']['p'],
                       'shortfall': equal['permutation']['mean'],
                       'noninferior': equal['difference_ci']['low'] is not None
                       and equal['difference_ci']['low'] > -delta})
    return {'delta': delta, 'claims': claims,
            'conjunction_p': max(c['p'] for c in claims),
            'dominates': all(c['noninferior'] for c in claims)}


def analyze(cfg):
    out = Path(cfg['output'])
    tag = f"{cfg['level']}_{attack_tag(cfg)}"
    if not (attack_dir(cfg, 'base') / 'curve.json').exists():
        print(f'[{tag}] skipped: the base model was not attacked')
        return None
    arms = attacked(cfg)
    gate = gate_forgetting(cfg)
    attack = attack_works(cfg)
    contrasts = [{'family': name, **contrast(cfg, left, right, axis)}
                 for axis in AXES for name, pairs in FAMILIES for left, right in pairs
                 if left in arms and right in arms]
    early = readout(cfg, arms)
    ranks = runner_up(cfg, arms)
    figure = [{'arm': arm, 'axis': axis, **bootstrap_ci(outcome(cfg, arm, axis).values())}
              for arm in ('base',) + tuple(arms) for axis in AXES]
    report = {'inference_scope': 'item-level only; one seed means no method-level error term',
              'level': cfg['level'], 'level_steps': {arm: level_step(cfg, arm) for arm in ALL_METHODS},
              'attack_tag': attack_tag(cfg), 'attack_settings': cfg['relearn'],
              'forgetting_gate': gate, 'attack_check': attack, 'readout': early, 'runner_up': ranks,
              'interactions': [interaction(cfg, axis, arms) for axis in AXES],
              'contrasts': contrasts, 'pareto': pareto(contrasts),
              'trainable_parameters': {arm: read_json(out / arm / 'settings.json')['trainable_parameters']
                                       for arm in ALL_METHODS if (out / arm / 'settings.json').exists()}}
    write_json(out / f'analysis_{tag}.json', report)
    rows = [{'family': c['family'], 'left': c['left'], 'right': c['right'], 'axis': c['axis'], 'n': c['permutation']['n'],
             'mean_difference': c['permutation']['mean'], 'ci_low': c['difference_ci']['low'],
             'ci_high': c['difference_ci']['high'], 'permutation_p': c['permutation']['p'],
             'exact': c['permutation']['exact'], 'sign_positive': c['sign']['positive'],
             'sign_p': c['sign']['p']} for c in contrasts]
    for path, table in ((out / f'analysis_{tag}.csv', rows), (out / f'figure_data_{tag}.csv', figure),
                        (out / f'readout_{tag}.csv', early.get('tests', [])),
                        (out / f'readout_arms_{tag}.csv', early.get('arms', [])),
                        (out / f'runner_up_{tag}.csv', ranks)):
        if not table:
            continue
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(f'[{tag}] arms: {", ".join(arms)}; forgetting gate: {"PASS" if gate["passed"] else "FAIL"}; '
          f'attack recovers base: {attack["recovers"]}')
    if not gate['passed']:
        print('Caveat: forgetting is not matched across arms at this level.')
    if 'skipped' in early:
        print(f'Readout skipped: {early["skipped"]}')
    else:
        if not early['attack_valid']:
            print(f'WARNING: base model fell below {VALID_ACCURACY} at step {READOUT_STEP}; '
                  'the attack diverged and no readout contrast means anything.')
        for row in early['arms']:
            print(f'{row["arm"]:12} known pre={row["forget_known_pre"]}/{row["count"]} '
                  f'step{READOUT_STEP}={row["forget_known_readout"]}/{row["count"]} '
                  f'retain step{READOUT_STEP}={row["retain_readout"]:.3f}')
        for t in early['tests']:
            if t['step'] == READOUT_STEP:
                print(f'{t["family"]:7} {t["left"]:12} vs {t["right"]:12} step{READOUT_STEP} {t["left_known"]} vs {t["right_known"]} '
                      f'McNemar {t["mcnemar_only_left"]}:{t["mcnemar_only_right"]} p={t["mcnemar_p"]:.4g} '
                      f'(Holm {t["mcnemar_p_holm"]:.4g})')
    for row in rows:
        print(f'{row["left"]:12} - {row["right"]:12} {row["axis"]:11} '
              f'mean={row["mean_difference"]:+.4f} p={row["permutation_p"]:.4g} '
              f'signs={row["sign_positive"]}/{row["n"]}')
    print('Wrote ' + ', '.join(str(out / f'{name}_{tag}.{kind}')
                                 for name, kind in (('analysis', 'json'), ('analysis', 'csv'),
                                                    ('figure_data', 'csv'), ('readout', 'csv'),
                                                    ('readout_arms', 'csv'), ('runner_up', 'csv'))))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    cfg = config(parser.parse_args().config)
    for level in LEVELS:
        for attack in cfg['relearn']:
            analyze(stage_cfg(cfg, level, attack))
