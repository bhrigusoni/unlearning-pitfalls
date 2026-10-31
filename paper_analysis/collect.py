"""Collect the final v3 results (../results) into flat tables."""
import csv, json, math, sys
from pathlib import Path
from scipy.stats import spearmanr, binomtest, wilcoxon

R = Path(__file__).resolve().parent.parent / 'results'
RUNS = {'s16_relabel_42': 'sports16_relabel', 's16_relabel_10042': 'sports16_relabel_seed10042',
        's16_relabel_20042': 'sports16_relabel_seed20042', 's16_npo_42': 'sports16_npo',
        's64_relabel_42': 'sports64_relabel'}
ARMS = ['ul', 'ul_flu', 'ul_sam', 'ul_flu_sam', 'ul_rand', 'ul_rand_sam']
LEVELS = ['steps', 'loss', 'forget']
ATTACKS = ['split_r8', 'split_r512_a16', 'heldout_r512_a16']


def rd(p):
    return json.loads(Path(p).read_text())


def curve(run, arm, level, attack):
    p = R / run / ('base' if arm == 'base' else f'{arm}/{level}') / 'relearning' / attack / 'curve.json'
    return {e['step']: e for e in rd(p)} if p.exists() else None


def records(entry, split='forget'):
    return entry[split].get('records')


def mcnemar(a, b):
    n = a + b
    return 1.0 if n == 0 else binomtest(a, n, 0.5).pvalue


out = []
for key, run in RUNS.items():
    for arm in ARMS:
        lv = R / run / arm / 'levels.json'
        reached = rd(lv)['reached'] if lv.exists() else {}
        for level in LEVELS:
            if level not in reached:
                out.append(dict(run=key, arm=arm, level=level, reached=False))
                continue
            ev = R / run / arm / level / 'evaluation.json'
            e = rd(ev) if ev.exists() else None
            row = dict(run=key, arm=arm, level=level, reached=True, step=reached[level]['step'],
                       max_norm=reached[level]['forget_max_normalized'])
            if e:
                f = e['forget']['summary']
                row.update(pre_known=round(f['accuracy'] * f['count']), n=f['count'],
                           mean_norm=f['mean_normalized_target_probability'],
                           mass=f['mean_sport_probability_mass'],
                           largest_class=f['largest_predicted_class_fraction'],
                           retain=e['retain']['summary']['accuracy'],
                           heldout=e['forget_heldout']['summary']['accuracy'])
                recs = e['forget']['records']
                # log-odds of true sport within closed set
                lo = []
                for r in recs:
                    p = r.get('normalized_target_probability')
                    if p is not None:
                        p = min(max(p, 1e-30), 1 - 1e-12)
                        lo.append(math.log(p / (1 - p)))
                row['mean_logodds'] = sum(lo) / len(lo) if lo else None
            for attack in ATTACKS:
                c = curve(run, arm, level, attack)
                if not c:
                    continue
                s10 = c.get(10)
                row[f'{attack}_k10'] = round(s10['forget']['accuracy'] * s10['forget']['count'])
                row[f'{attack}_ret10'] = s10['retain']['accuracy']
                last = c[max(c)]
                row[f'{attack}_k100'] = round(last['forget']['accuracy'] * last['forget']['count'])
                row[f'{attack}_ret100'] = last['retain']['accuracy']
                row[f'{attack}_auc'] = sum(c[s]['forget']['accuracy'] for s in sorted(c)) / len(c)
                row[f'{attack}_tl10'] = s10.get('train_loss')
            out.append(row)

# base validity per run/attack
base = {}
for key, run in RUNS.items():
    for attack in ATTACKS:
        c = curve(run, 'base', None, attack)
        if c:
            s = c[10]
            base[(key, attack)] = dict(k10=round(s['forget']['accuracy'] * s['forget']['count']),
                                       n=s['forget']['count'], ret10=s['retain']['accuracy'],
                                       valid=s['retain']['accuracy'] >= .95 and s['forget']['accuracy'] >= .95,
                                       tl10=s.get('train_loss'))

fields = sorted({k for r in out for k in r}, key=lambda k: (k not in ('run', 'arm', 'level', 'reached', 'step'), k))
with open(sys.argv[1], 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(out)
json.dump({f'{k[0]}|{k[1]}': v for k, v in base.items()}, open(sys.argv[2], 'w'), indent=1)
print('rows', len(out))
