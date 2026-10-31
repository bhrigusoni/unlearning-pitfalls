"""Paired tests, P1 correlation, SAM depth, runner-up for the final v3 runs."""
import json, math
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr, binomtest, wilcoxon

R = Path(__file__).resolve().parent.parent / 'results'
RUNS = {'s16_relabel_42': 'sports16_relabel', 's16_relabel_10042': 'sports16_relabel_seed10042',
        's16_relabel_20042': 'sports16_relabel_seed20042', 's16_npo_42': 'sports16_npo',
        's64_relabel_42': 'sports64_relabel'}
ATTACKS = ['split_r8', 'split_r512_a16', 'heldout_r512_a16']
d = pd.read_csv('all.csv'); d = d[d.reached]
base = json.load(open('base.json'))


def known(run, arm, level, attack, step):
    e = json.loads((R / RUNS[run] / arm / level / 'relearning' / attack / f'evaluation_{step:04d}.json').read_text())
    return {r['athlete']: r['correct'] for r in e['forget']['records']}


def mc(run, a, b, level, attack, step=10):
    ka, kb = known(run, a, level, attack, step), known(run, b, level, attack, step)
    oa = sum(ka[x] and not kb[x] for x in ka); ob = sum(kb[x] and not ka[x] for x in ka)
    p = 1.0 if oa + ob == 0 else binomtest(oa, oa + ob, .5).pvalue
    return sum(ka.values()), sum(kb.values()), oa, ob, p


print('== Key contrasts at step 10 (known a vs b, discordant a:b, McNemar p); * = base attack invalid')
PAIRS = [('ul_flu', 'ul_rand'), ('ul_flu_sam', 'ul_rand_sam'), ('ul_flu_sam', 'ul_flu'), ('ul', 'ul_flu'), ('ul_sam', 'ul')]
for run in RUNS:
    for level in ['steps', 'loss', 'forget']:
        for attack in ATTACKS:
            v = '' if base[f'{run}|{attack}']['valid'] else '*'
            cells = []
            for a, b in PAIRS:
                if ((d.run == run) & (d.level == level) & d.arm.isin([a, b])).sum() < 2:
                    cells.append(f'{a}/{b}: n/a'); continue
                ka, kb, oa, ob, p = mc(run, a, b, level, attack)
                cells.append(f'{a}/{b} {ka}v{kb} ({oa}:{ob}) p={p:.3g}')
            print(f'{run:18} {level:6} {attack:16}{v:1} | ' + ' | '.join(cells))

print('\n== Pooled forget-level FLU vs Random over seeds (valid seeds only / all seeds)')
for attack in ATTACKS:
    for a, b in [('ul_flu', 'ul_rand'), ('ul_flu_sam', 'ul_rand_sam'), ('ul_flu_sam', 'ul_flu'), ('ul', 'ul_flu')]:
        for only_valid in (True, False):
            tot = [0, 0, 0, 0, 0]; used = []
            for run in ['s16_relabel_42', 's16_relabel_10042', 's16_relabel_20042']:
                if only_valid and not base[f'{run}|{attack}']['valid']:
                    continue
                ka, kb, oa, ob, _ = mc(run, a, b, 'forget', attack)
                tot = [tot[0] + ka, tot[1] + kb, tot[2] + oa, tot[3] + ob, tot[4] + 16]; used.append(run[-5:])
            if not used: continue
            p = 1.0 if tot[2] + tot[3] == 0 else binomtest(tot[2], tot[2] + tot[3], .5).pvalue
            print(f'{attack:16} {a}/{b} {"valid" if only_valid else "all  "} seeds={used} {tot[0]}v{tot[1]} of {tot[4]} ({tot[2]}:{tot[3]}) p={p:.3g}')

print('\n== P1: Spearman(recovery AUC, pre-attack mean log-odds) and (AUC, step-10 known)')
for attack in ATTACKS:
    rows = d[(d.run != 's16_npo_42')].dropna(subset=[f'{attack}_auc'])
    for lvl in ['steps', 'all']:
        x = rows if lvl == 'all' else rows[rows.level == lvl]
        r1 = spearmanr(x[f'{attack}_auc'], x.mean_logodds); r2 = spearmanr(x[f'{attack}_auc'], x.max_norm)
        r3 = spearmanr(x[f'{attack}_k10'] / x.n, x.mean_logodds)
        print(f'{attack:16} {lvl:5} n={len(x)}  AUC~logodds rho={r1.statistic:.2f} p={r1.pvalue:.2g} | AUC~max_norm rho={r2.statistic:.2f} | k10~logodds rho={r3.statistic:.2f} p={r3.pvalue:.2g}')
    # per run, steps level (unmatched)
    for run in RUNS:
        x = rows[(rows.run == run)]
        if len(x) > 4:
            r = spearmanr(x[f'{attack}_auc'], x.mean_logodds)
            print(f'   {run} all levels n={len(x)} rho={r.statistic:.2f} p={r.pvalue:.2g}')

print('\n== P3: retain at step 10 and step 100 (split_r8), min over arms; base')
for run in RUNS:
    x = d[d.run == run]
    for attack in ATTACKS:
        print(f'{run:18} {attack:16} base ret10={base[f"{run}|{attack}"]["ret10"]:.3f} k10={base[f"{run}|{attack}"]["k10"]}/{base[f"{run}|{attack}"]["n"]} | arms ret10 min={x[f"{attack}_ret10"].min():.2f} ret100 min={x[f"{attack}_ret100"].min():.2f} median ret100={x[f"{attack}_ret100"].median():.2f}')

print('\n== P2: heldout attack train loss at step 10, per arm (forget level)')
x = d[d.level == 'forget'][['run', 'arm', 'heldout_r512_a16_tl10', 'split_r8_tl10', 'heldout']]
print(x.to_string())

print('\n== SAM depth: mean log-odds at steps level, and steps to forget level; pairs (no SAM, SAM)')
lo_ns, lo_s = [], []
for run in ['s16_relabel_42', 's16_relabel_10042', 's16_relabel_20042', 's64_relabel_42']:
    for a in ['ul', 'ul_flu', 'ul_rand']:
        g = d[(d.run == run)]
        s = g[(g.arm == a) & (g.level == 'steps')].iloc[0]; t = g[(g.arm == a + '_sam') & (g.level == 'steps')].iloc[0]
        fs = g[(g.arm == a) & (g.level == 'forget')]; ft = g[(g.arm == a + '_sam') & (g.level == 'forget')]
        lo_ns.append(s.mean_logodds); lo_s.append(t.mean_logodds)
        print(f'{run:18} {a:8} logodds {s.mean_logodds:7.2f} vs SAM {t.mean_logodds:7.2f} | pre-known {s.pre_known:.0f} vs {t.pre_known:.0f} | forget-level step {fs.step.iloc[0] if len(fs) else None} vs {ft.step.iloc[0] if len(ft) else None}')
print('Wilcoxon paired (12 pairs) no-SAM vs SAM log-odds:', wilcoxon(lo_ns, lo_s), sum(a < b for a, b in zip(lo_ns, lo_s)), 'of', len(lo_s), 'SAM shallower')

print('\n== Runner-up (true sport ranked 2nd among wrong facts), conditional null 1/2, pre-attack, forget level')
for run in RUNS:
    p = R / RUNS[run] / 'runner_up_forget_split_r8.csv'
    if p.exists():
        print(run); print(pd.read_csv(p).to_string(index=False))
