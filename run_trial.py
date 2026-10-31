"""Run each stage in a fresh process to release GPU memory; resume checkpoints."""
import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

from prepare_data import (ALL_METHODS, LEVELS, RANDOM_LOCALIZATIONS, READOUT_STEP, arm_dir, attack_dir, attack_tag,
                          config, level_step, mask_tag, read_json, stage_cfg, write_json)

from train import METHODS


def preflight(cfg):
    import importlib.metadata
    import torch
    from prepare_data import read_rows
    if not Path(cfg['dataset']).is_file():
        raise FileNotFoundError(cfg['dataset'])
    rows = read_rows(cfg['dataset'])
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('This machine has no CUDA GPU; use --self-test for CPU validation.')
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'packages': {p: importlib.metadata.version(p) for p in ('transformers', 'peft', 'accelerate')},
            'dataset_unique_rows': len(rows)}
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0), gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
    print(json.dumps(info, indent=2))
    write_json(Path(cfg['output']) / 'environment.json', info)


def aggregate(cfg):
    """One table per (level, attack); an arm that never reached the level is left out."""
    out = Path(cfg['output'])
    tag = attack_tag(cfg)
    table = []
    for method in ('base',) + ALL_METHODS:
        # Control arms and partial attack sets (e.g. an attack-strength sweep) are left out.
        if level_step(cfg, method) is None or not (attack_dir(cfg, method) / 'curve.json').exists():
            continue
        report = read_json(arm_dir(cfg, method) / 'evaluation.json')
        curve = read_json(attack_dir(cfg, method) / 'curve.json')
        settings = out / method / 'settings.json'
        readout = next((c for c in curve if c['step'] == READOUT_STEP), None)
        first, last = curve[0], curve[-1]
        area = sum((b['step'] - a['step']) * (a['forget']['accuracy'] + b['forget']['accuracy']) / 2
                   for a, b in zip(curve, curve[1:])) / max(1, last['step'])
        table.append({'method': method, 'level': cfg['level'], 'attack_tag': tag,
                      'forget_accuracy': report['forget']['summary']['accuracy'],
                      'forget_probability': report['forget']['summary']['mean_target_probability'],
                      'retain_accuracy': report['retain']['summary']['accuracy'],
                      # Pre-attack forget_heldout is the only probe absent from both loss
                      # terms, so it is the collateral-damage measure; retain is in-loss.
                      'forget_heldout_accuracy': report['forget_heldout']['summary']['accuracy'],
                      'forget_normalized': report['forget']['summary']['mean_normalized_target_probability'],
                      'retain_normalized': report['retain']['summary']['mean_normalized_target_probability'],
                      'forget_heldout_normalized': report['forget_heldout']['summary']['mean_normalized_target_probability'],
                      'trainable_parameters': read_json(settings)['trainable_parameters'] if settings.exists() else None,
                      # The training step whose weights were evaluated and attacked.
                      'level_step': level_step(cfg, method),
                      'attack_final_train_loss': last.get('train_loss'),
                      # forget_split attacks only: accuracy on the facts the attack trained on.
                      'forget_trained_accuracy_after_attack': last.get('forget_trained', {}).get('accuracy'),
                      f'forget_accuracy_step{READOUT_STEP}': readout['forget']['accuracy'] if readout else None,
                      f'retain_accuracy_step{READOUT_STEP}': readout['retain']['accuracy'] if readout else None,
                      'retain_accuracy_after_attack': last['retain']['accuracy'],
                      'forget_accuracy_after_attack': last['forget']['accuracy'],
                      'forget_accuracy_recovery': last['forget']['accuracy'] - first['forget']['accuracy'],
                      'recovery_accuracy_auc': area,
                      'largest_forget_prediction_class': report['forget']['summary']['largest_predicted_class_fraction']})
    # Named after the attack, so several attack settings coexist in one output folder.
    name = f"comparison_{cfg['level']}_{tag}"
    write_json(out / f'{name}.json', table)
    with (out / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='required for everything except --self-test')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--method', nargs='+', choices=('base',) + ALL_METHODS,
                        help='run only these conditions (default: the six main arms, plus base and comparison); '
                             'base here attacks the base model too')
    parser.add_argument('--attack', nargs='+',
                        help='run only these attacks, by name under relearn (default: all)')
    parser.add_argument('--level', nargs='+', choices=LEVELS,
                        help='evaluate and attack only these levels, and stop training once they are saved '
                             '(default: all three)')
    parser.add_argument('--attack-only', action='store_true',
                        help='reuse the trained arms and evaluations and run only relearn.py')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.self_test:
        subprocess.run([sys.executable, str(root / 'test_trial.py')], check=True)
        return
    if not args.config:
        parser.error('--config is required')
    cfg = config(args.config)
    preflight(cfg)
    if args.preflight:
        return
    # Baseline evaluation is a required experimental stage and runs before any
    # localization or unlearning, establishing what the base model knows.
    with_base = not args.method or 'base' in args.method
    methods = [m for m in args.method or METHODS if m != 'base']
    levels = args.level or list(LEVELS)
    attacks = args.attack or list(cfg['relearn'])
    unknown = set(attacks) - set(cfg['relearn'])
    if unknown:
        raise ValueError(f'No such attack in the config: {sorted(unknown)}')
    stages = []
    if not args.attack_only:
        stages.append(('prepare_data.py', []))
        if not args.method:
            stages.append(('evaluate.py', ['--method', 'base']))
        # Every random mask copies the FLU mask's layer count, so it needs flu.py first.
        tags = {mask_tag(m) for m in methods} - {None}
        if tags:
            stages.append(('flu.py', []))
        stages += [('flu.py', ['--localization', RANDOM_LOCALIZATIONS[t]]) for t in sorted(tags - {'flu'})]
        # One training run per arm yields every level's checkpoint.
        stop = ['--levels', *levels] if args.level else []
        stages += [('train.py', ['--method', m, *stop]) for m in methods]
        stages += [('evaluate.py', ['--method', m, '--level', level]) for level in levels for m in methods]
    for attack in attacks:
        if with_base:
            stages.append(('relearn.py', ['--method', 'base', '--attack', attack]))
        stages += [('relearn.py', ['--method', m, '--level', level, '--attack', attack])
                   for level in levels for m in methods]
    logs = Path(cfg['output']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for script, options in stages:
        # --method X --level Y --attack Z -> relearn_X_Y_Z.log
        label = '_'.join([script.removesuffix('.py')] + options[1::2])
        command = [sys.executable, '-u', str(root / script), '--config', str(Path(args.config).resolve()), *options]
        print('Running', label, flush=True)
        with (logs / (label + '.log')).open('a', encoding='utf-8') as stream:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace') as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait() != 0:
                    raise RuntimeError(f'{label} failed; see {logs / (label + ".log")}. Re-run to resume.')
    if args.method:
        # comparison tables need every arm; run without --method to build them (completed stages resume).
        print('Complete:', ', '.join(str(Path(cfg['output']) / m) for m in methods))
        return
    for level in levels:
        for attack in attacks:
            aggregate(stage_cfg(cfg, level, attack))
    print(f'Complete: comparison_<level>_<attack>.csv in {cfg["output"]}')


if __name__ == '__main__':
    main()
