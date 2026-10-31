"""Reviewer requests answerable from the saved results (no GPU): confidence intervals,
readout-step and validity-threshold sensitivity, pre-attack depth at equal forgetting."""
import json, math
from pathlib import Path
import pandas as pd
from scipy.stats import beta, binomtest

R = Path(__file__).resolve().parent.parent / 'results'
RUNS = {'s16_relabel_42': 'sports16_relabel', 's16_relabel_10042': 'sports16_relabel_seed10042',
        's16_relabel_20042': 'sports16_relabel_seed20042', 's64_relabel_42': 'sports64_relabel'}
S16 = ['s16_relabel_42', 's16_relabel_10042', 's16_relabel_20042']
ATTACKS = ['split_r8', 'split_r512_a16', 'heldout_r512_a16']
PAIRS = [('ul_flu', 'ul_rand'), ('ul_flu_sam', 'ul_rand_sam')]
Z = 1.959964


def rd(p):
    return json.loads(Path(p).read_text())


def entry(run, arm, attack, step, level='forget'):
    folder = R / RUNS[run] / ('base' if arm == 'base' else f'{arm}/{level}') / 'relearning' / attack
    p = folder / f'evaluation_{step:04d}.json'
    return rd(p) if p.exists() else None


def known(run, arm, attack, step):
    e = entry(run, arm, attack, step)
    return None if e is None else {r['athlete']: bool(r['correct']) for r in e['forget']['records']}


def base_ok(run, attack, step, threshold=0.95):
    e = entry(run, 'base', attack, step)
    return e['retain']['summary']['accuracy'] >= threshold and e['forget']['summary']['accuracy'] >= threshold


def base_acc(run, attack, step):
    e = entry(run, 'base', attack, step)
    return e['retain']['summary']['accuracy'], e['forget']['summary']['accuracy']


def wilson(x, n):
    p = x / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def paired_ci(a, b, c, d):
    """Newcombe (1998, method 10) 95% CI for p_left - p_right on paired binary data.
    a both known, b left only, c right only, d neither."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    phi = 0.0 if den == 0 else (max(num - n / 2, 0) if num > 0 else num) / math.sqrt(den)
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return p1 - p2 - dl, p1 - p2 + du


def exact_ci(b, c, n):
    """95% CI for (b - c) / n conditional on the b + c discordant facts: Clopper-Pearson on
    b / (b + c), mapped to the difference. Dual to the exact McNemar test used in the paper."""
    m = b + c
    if m == 0:
        return 0.0, 0.0
    lo = 0.0 if b == 0 else beta.ppf(0.025, b, m - b + 1)
    hi = 1.0 if b == m else beta.ppf(0.975, b + 1, m - b)
    return m / n * (2 * lo - 1), m / n * (2 * hi - 1)


def compare(runs, left, right, attack, step):
    a = b = c = d = 0
    for run in runs:
        kl, kr = known(run, left, attack, step), known(run, right, attack, step)
        if kl is None or kr is None:
            return None
        for x in kl:
            a += kl[x] and kr[x]; b += kl[x] and not kr[x]; c += kr[x] and not kl[x]; d += not kl[x] and not kr[x]
    n = a + b + c + d
    p = 1.0 if b + c == 0 else binomtest(b, b + c, .5).pvalue
    lo, hi = exact_ci(b, c, n)
    nlo, nhi = paired_ci(a, b, c, d)
    return dict(left=a + b, right=a + c, n=n, only_left=b, only_right=c, p=p, diff=(b - c) / n, lo=lo, hi=hi,
                nlo=nlo, nhi=nhi)


def fmt(r):
    return (f"{r['left']} vs {r['right']} of {r['n']} ({r['only_left']}:{r['only_right']}) p={r['p']:.3g} "
            f"diff={r['diff']:+.2f} Newcombe [{r['nlo']:+.2f}, {r['nhi']:+.2f}] exact-cond [{r['lo']:+.2f}, {r['hi']:+.2f}]")


print('== 1. Paired differences at step 10, equal forgetting, 95% CIs: Newcombe (quoted in paper) and exact conditional')
for attack in ATTACKS:
    valid = [run for run in S16 if base_ok(run, attack, 10)]
    for left, right in PAIRS:
        for run in S16 + ['s64_relabel_42']:
            r = compare([run], left, right, attack, 10)
            flag = '' if base_ok(run, attack, 10) else ' (attack invalid)'
            print(f'{attack:16} {left}/{right} {run:18} {fmt(r)}{flag}')
        if len(valid) > 1:
            print(f'{attack:16} {left}/{right} pooled valid {valid}: {fmt(compare(valid, left, right, attack, 10))}')

print('\n== 2. Readout-step sensitivity (FLU vs Random, equal forgetting); base validity at that step')
for attack in ATTACKS:
    for step in (5, 10, 20, 30):
        cells = []
        for run in S16:
            r = compare([run], 'ul_flu', 'ul_rand', attack, step)
            if r is None:
                cells.append(f'{run[-5:]}: not scored')
                continue
            ret, fg = base_acc(run, attack, step)
            cells.append(f"{run[-5:]}: {r['left']}v{r['right']} p={r['p']:.2g} base ret {ret:.2f} fg {fg:.2f}")
        valid = [run for run in S16 if entry(run, 'base', attack, step) and base_ok(run, attack, step)]
        pooled = compare(valid, 'ul_flu', 'ul_rand', attack, step) if valid else None
        r64 = compare(['s64_relabel_42'], 'ul_flu', 'ul_rand', attack, step) or dict(left='-', right='-', p=float('nan'))
        print(f'{attack:16} step {step:3} | ' + ' | '.join(cells)
              + f" || pooled valid {[v[-5:] for v in valid]}: {fmt(pooled) if pooled else '-'}"
              + f" || S64 {r64['left']}v{r64['right']} p={r64['p']:.2g}")

print('\n== 3. Validity-threshold sensitivity at step 10 (FLU vs Random, pooled over admitted seeds)')
for attack in ATTACKS:
    for threshold in (0.90, 0.93, 0.95, 0.97):
        valid = [run for run in S16 if base_ok(run, attack, 10, threshold)]
        cells = [f'{left}/{right} {fmt(compare(valid, left, right, attack, 10))}' for left, right in PAIRS] if valid else ['-']
        print(f'{attack:16} threshold {threshold:.2f} seeds {[v[-5:] for v in valid]} | ' + ' | '.join(cells))

print('\n== 4. Pre-attack depth at equal forgetting: mean true-sport log-odds, step reached')
d = pd.read_csv('all.csv')
d = d[(d.level == 'forget') & d.reached & d.run.isin(list(RUNS))]
print(d.pivot(index='arm', columns='run', values='mean_logodds').round(1).to_string())
print(d.pivot(index='arm', columns='run', values='step').to_string())
