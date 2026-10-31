"""Pool the step-10 readout over seeds: one config per seed, same grid.

Each seed draws its own forget/retain/held-out facts, relabel targets, batch order,
FLU probe folds and random layers, so a seed is an independent replication of the
whole pipeline. Two tests per pair of arms, per (level, attack):

- pooled McNemar: discordant facts summed over seeds (exact, stratified by seed,
  since McNemar only uses discordant pairs). More facts, still item-level.
- seed sign test: in how many seeds the left arm knows fewer facts than the right.
  Two-sided p < 0.05 needs at least 6 seeds all one way; with 3 seeds the best possible
  p is 0.25, so report it as a direction count (e.g. 3/3), with pooled McNemar as the test.

Seeds where the base model fails the attack-validity check, or where either arm never
reached the level, are left out of that pair and counted in `seeds_used`.
Reads JSON only, so it runs on the laptop.

Usage: python analyze_seeds.py --configs config_sports16_relabel.yaml config_sports16_relabel_seed1.yaml ...
"""
import argparse
import csv
from pathlib import Path

import analyze
from analyze import FAMILIES, attacked, binomial_p, holm, known, readout
from prepare_data import LEVELS, config, stage_cfg, write_json


def pooled(cfgs):
    """Pooled readout tests for one (level, attack) across seed configs."""
    step = analyze.READOUT_STEP
    per_seed = []
    for cfg in cfgs:
        arms = attacked(cfg)
        early = readout(cfg, arms)
        per_seed.append({'seed': cfg['seed'], 'arms': arms,
                         'valid': 'skipped' not in early and early['attack_valid']})
    rows = []
    for family, left, right in [(name, l, r) for name, pairs in FAMILIES for l, r in pairs]:
        row = {'family': family, 'left': left, 'right': right, 'seeds_used': 0, 'n': 0, 'left_known_pre': 0, 'right_known_pre': 0,
               'left_known': 0, 'right_known': 0, 'only_left': 0, 'only_right': 0,
               'seeds_left_fewer': 0, 'seeds_right_fewer': 0, 'seeds_tied': 0}
        for cfg, seed in zip(cfgs, per_seed):
            if not seed['valid'] or left not in seed['arms'] or right not in seed['arms']:
                continue
            l0, r0 = known(cfg, left, 0), known(cfg, right, 0)
            l, r = known(cfg, left, step), known(cfg, right, step)
            row['seeds_used'] += 1
            row['n'] += len(l)
            row['left_known_pre'] += sum(l0.values())
            row['right_known_pre'] += sum(r0.values())
            row['left_known'] += sum(l.values())
            row['right_known'] += sum(r.values())
            row['only_left'] += sum(l[a] and not r[a] for a in l)
            row['only_right'] += sum(r[a] and not l[a] for a in l)
            diff = sum(l.values()) - sum(r.values())
            row['seeds_left_fewer' if diff < 0 else 'seeds_right_fewer' if diff > 0 else 'seeds_tied'] += 1
        if not row['seeds_used']:
            continue
        row['mcnemar_p'] = binomial_p(row['only_left'], row['only_left'] + row['only_right'])
        row['seed_sign_p'] = binomial_p(row['seeds_left_fewer'], row['seeds_left_fewer'] + row['seeds_right_fewer'])
        rows.append(row)
    # Holm within each (level, attack, family), as analyze.py does for one seed.
    for name, _ in FAMILIES:
        family = [r for r in rows if r['family'] == name]
        for key in ('mcnemar_p', 'seed_sign_p'):
            for row, adjusted in zip(family, holm([r[key] for r in family])):
                row[f'{key}_holm'] = adjusted
    return {'step': step, 'seeds': per_seed, 'tests': rows}


def main(paths):
    cfgs = [config(p) for p in paths]
    seeds = [c['seed'] for c in cfgs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Duplicate seeds: {seeds}')
    out = Path(cfgs[0]['output']).parent / f"seeds_{Path(cfgs[0]['output']).name}"
    report, table = {'seeds': seeds, 'outputs': [c['output'] for c in cfgs]}, []
    for level in LEVELS:
        for attack in cfgs[0]['relearn']:
            tag = f'{level}_{attack}'
            result = pooled([stage_cfg(c, level, attack) for c in cfgs])
            report[tag] = result
            table += [{'level': level, 'attack': attack, **row} for row in result['tests']]
            invalid = [s['seed'] for s in result['seeds'] if not s['valid']]
            print(f'[{tag}] seeds {seeds}' + (f'; attack invalid or no readout, dropped: {invalid}' if invalid else ''))
            for r in result['tests']:
                print(f'{r["family"]:7} {r["left"]:12} vs {r["right"]:12} known step{result["step"]} {r["left_known"]} vs '
                      f'{r["right_known"]} of {r["n"]} | McNemar {r["only_left"]}:{r["only_right"]} '
                      f'p={r["mcnemar_p"]:.4g} (Holm {r["mcnemar_p_holm"]:.4g}) | seeds fewer '
                      f'{r["seeds_left_fewer"]}:{r["seeds_right_fewer"]} tie {r["seeds_tied"]} '
                      f'p={r["seed_sign_p"]:.4g} (Holm {r["seed_sign_p_holm"]:.4g})')
    write_json(out.with_suffix('.json'), report)
    if table:
        with out.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(f'Wrote {out}.json, {out}.csv')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='+', required=True, help='one config per seed, same grid')
    main(parser.parse_args().configs)
