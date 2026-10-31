"""Attack curves: forget, retain and held-out accuracy vs attack step, one line per arm.

One PNG per (level, attack) in <output>/plots/. Reads only curve.json, so it runs on
a laptop. Needs matplotlib (not used anywhere else).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_data import ALL_METHODS, LEVELS, READOUT_STEP, attack_dir, config, level_step, read_json, stage_cfg

# Fixed order: each arm keeps its colour and marker in every figure.
COLORS = dict(zip(ALL_METHODS, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
                                '#7b4fd6', '#9c6b3e', '#00a3b4', '#c4314b')))
MARKERS = dict(zip(ALL_METHODS, 'osD^vPXhp*'))
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'


def plot(cfg):
    heldout_trained = cfg['relearn'].get('data', 'forget_heldout') == 'forget_heldout'
    panels = (('forget', 'Forget set (unseen by attack)'), ('retain', 'Retain set'),
              ('forget_heldout', 'Held-out set' + (' (attack trains on it)' if heldout_trained else '')))
    arms = [arm for arm in ('base',) + ALL_METHODS if level_step(cfg, arm) is not None
            and (attack_dir(cfg, arm) / 'curve.json').exists()]
    if len(arms) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (split, title) in zip(axes, panels):
        for arm in arms:
            curve = read_json(attack_dir(cfg, arm) / 'curve.json')
            steps = [c['step'] for c in curve]
            values = [c[split]['accuracy'] for c in curve]
            if arm == 'base':
                ax.plot(steps, values, color=MUTED, linewidth=2, linestyle='--', label='base (not unlearned)')
            else:
                ax.plot(steps, values, color=COLORS[arm], linewidth=2, marker=MARKERS[arm], markersize=6,
                        markevery=max(1, len(steps) // 10), label=f'{arm} (step {level_step(cfg, arm)})')
        ax.axvline(READOUT_STEP, color=MUTED, linewidth=1, linestyle=':')
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('Attack step', color=MUTED)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel('Accuracy', color=MUTED)
    axes[0].text(READOUT_STEP + 1, 0.97, f'step {READOUT_STEP}', color=MUTED, fontsize=9, va='top')
    axes[-1].legend(frameon=False, fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5),
                    title='Arm (unlearning step)', title_fontsize=9)
    fig.suptitle(f"{Path(cfg['output']).name} | level: {cfg['level']} | attack: {cfg['relearn']['tag']}",
                 color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    path = Path(cfg['output']) / 'plots' / f"{cfg['level']}_{cfg['relearn']['tag']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    cfg = config(parser.parse_args().config)
    for level in LEVELS:
        for attack in cfg['relearn']:
            path = plot(stage_cfg(cfg, level, attack))
            print(path or f'{level}/{attack}: no curves yet')
