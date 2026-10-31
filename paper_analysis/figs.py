"""Paper figures from all.csv (collect.py) and base.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

OUT = 'figures/'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5})
d = pd.read_csv('all.csv'); d = d[d.reached & (d.run != 's16_npo_42')]
base = json.load(open('base.json'))

# Figure 1: recovery AUC vs pre-attack depth, split_r8; colour + marker by level.
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for level, colour, marker, name in [('steps', BLUE, 'o', 'same steps'), ('loss', ORANGE, 's', 'below loss'),
                                    ('forget', AQUA, '^', 'same forgetting')]:
    x = d[d.level == level]
    rho = spearmanr(x.split_r8_auc, x.mean_logodds).statistic
    ax.scatter(x.mean_logodds, x.split_r8_auc, s=16, c=colour, marker=marker, edgecolors='white', linewidths=.6,
               label=f'{name} ($\\rho$={rho:.2f})', zorder=3)
ax.set_xlabel('pre-attack mean log-odds of the true sport')
ax.set_ylabel('recovery AUC (rank-8 split attack)')
ax.legend(frameon=False, fontsize=7, loc='upper left', handletextpad=.2)
fig.tight_layout(); fig.savefig(OUT + 'p1_depth.pdf'); fig.savefig(OUT + 'p1_depth.png', dpi=300)

# Figure 2: FLU vs Random, facts known at attack step 10, same-forgetting level.
rows = [('s16_relabel_42', 'S16 seed 1'), ('s16_relabel_10042', 'S16 seed 2'),
        ('s16_relabel_20042', 'S16 seed 3'), ('s64_relabel_42', 'S64 seed 1')]
attacks = [('split_r8', 'split, r8'), ('split_r512_a16', 'split, r512'), ('heldout_r512_a16', 'held-out, r512')]
fig, axes = plt.subplots(1, 3, figsize=(7, 1.9), sharey=True)
for ax, (attack, title) in zip(axes, attacks):
    for i, (run, label) in enumerate(rows):
        x = d[(d.run == run) & (d.level == 'forget')].set_index('arm')
        n = x.n.iloc[0]
        f, r = x.loc['ul_flu', f'{attack}_k10'] / n, x.loc['ul_rand', f'{attack}_k10'] / n
        valid = base[f'{run}|{attack}']['valid']
        ax.plot([f, r], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([r], [i], s=30, c=ORANGE, marker='s', zorder=3, label='Random' if i == 0 else None,
                   alpha=1 if valid else .35, edgecolors='white', linewidths=.6)
        ax.scatter([f], [i], s=30, c=BLUE, marker='o', zorder=3, label='FLU' if i == 0 else None,
                   alpha=1 if valid else .35, edgecolors='white', linewidths=.6)
        if not valid:
            ax.text(1.04, i, 'invalid', va='center', fontsize=6, color=MUTED)
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xlim(-.02, 1.2); ax.set_xticks([0, .5, 1])
    ax.set_xlabel('fraction known, attack step 10')
axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[1] for r in rows]); axes[0].invert_yaxis()
axes[0].legend(frameon=False, fontsize=7, loc='lower left', handletextpad=.1)
fig.tight_layout(); fig.savefig(OUT + 'flu_vs_random.pdf'); fig.savefig(OUT + 'flu_vs_random.png', dpi=300)
print('ok')

# Figure 3: attack curves (P3) and the attack on the original model (P4).
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'results' / 'sports16_relabel'
VIOLET, RED, YELLOW, GREY = '#4a3aa7', '#e34948', '#eda100', '#8a8980'
def curve(p):
    return json.loads((R / p).read_text())
arms = [('ul', 'Whole', BLUE, '-'), ('ul_sam', 'Whole + SAM', BLUE, '--'),
        ('ul_flu', 'FLU', ORANGE, '-'), ('ul_flu_sam', 'FLU + SAM', ORANGE, '--'),
        ('ul_rand', 'Random', AQUA, '-'), ('ul_rand_sam', 'Random + SAM', AQUA, '--')]
fig, axes = plt.subplots(1, 3, figsize=(7, 2.35))
base = curve('base/relearning/split_r8/curve.json')
for ax, key, title in [(axes[0], 'forget', '(a) forget facts known'), (axes[1], 'retain', '(b) retain accuracy')]:
    ax.plot([e['step'] for e in base], [e[key]['accuracy'] for e in base], color=GREY, lw=1.2, ls=':', label='Original')
    for arm, name, colour, ls in arms:
        c = curve(f'{arm}/forget/relearning/split_r8/curve.json')
        ax.plot([e['step'] for e in c], [e[key]['accuracy'] for e in c], color=colour, ls=ls, lw=1.2, label=name)
    ax.axvline(10, color=MUTED, lw=.6, ls=':')
    ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel('attack step'); ax.set_ylim(-.03, 1.05)
h, l = axes[1].get_legend_handles_labels()
axes[0].set_ylabel('fraction / accuracy')
ax = axes[2]
for p, name, colour, marker in [('base/relearning/split_r512/curve.json', r'$\alpha$=1024', RED, 'o'),
                                ('base/relearning/split_r512_a128/fold0/curve.json', r'$\alpha$=128', YELLOW, 's'),
                                ('base/relearning/split_r512_a16/curve.json', r'$\alpha$=16', VIOLET, '^')]:
    c = curve(p)
    ax.plot([e['step'] for e in c], [e['retain']['accuracy'] for e in c], color=colour, lw=1.2,
            marker=marker, markersize=3, markevery=max(1, len(c) // 10), label=name)
ax.axhline(1 / 3, color=MUTED, lw=.6, ls=':'); ax.text(98, .36, 'chance', ha='right', fontsize=6, color=MUTED)
ax.set_title('(c) original model, rank-512 attack', fontsize=8, color=INK)
ax.set_xlabel('attack step'); ax.set_ylabel('retain accuracy'); ax.set_ylim(-.03, 1.05)
ax.legend(frameon=False, fontsize=6, loc='center right', bbox_to_anchor=(1, .68))
fig.tight_layout(rect=(0, .1, 1, 1))
fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower left', bbox_to_anchor=(.03, 0), ncol=7, handlelength=2, columnspacing=1.2)
fig.savefig(OUT + 'attack_curves.png', dpi=300)
print('fig3 ok')
