"""Reviewer GPU runs for one ~5 GPU-hour session, most important first.

  python run_today.py --list          # show the steps
  python run_today.py                 # run them all, within --hours
  python run_today.py --only 1 2      # run selected steps

1. Seed 3: random layers drawn outside the FLU set (ul_randx), timed: the calibration.
2. Sports-64: the same.
3. Seed 2: the same (seed 2 is excluded from the main analysis, so it comes last).
4. SAM gradient-norm check on seed 1 (sam_gradcheck.py): ul_flu_sam, then ul_sam.
5. Attack strength on seed 1 (config_sports16_relabel_alpha.yaml): alpha 8/32/16/64,
   30 steps, on base, ul_flu and ul_rand. Seed 1's random layers do not overlap FLU.
6. Depth-matched random layers outside FLU (ul_randd): seed 3, then Sports-64.
7. If time is left: the SAM versions of step 1-2 (ul_randx_sam), which cost ~2x.

Every new arm is trained only to the 'forget' (equal-forgetting) level and attacked
with the main attack, split_r512_a16. Its resume state is kept, so a later run without
--level saves the other levels. Each step is resumable: after a crash or a stop, run
the same command again and finished stages are skipped.

Budget: no step that trains starts after --stop-training-at hours, and nothing starts
after --hours. analyze.py runs after each step (JSON only, a minute or two).
Seed numbers: paper seed 1/2/3 = 44/10042/20042 (Sports-16 Relabel; NPO and Sports-64 use 42); check
that this matches your Table I before running.
"""
import argparse
import sys
import time
from pathlib import Path

import run_sports16 as grid
from run_sports16 import run, say, seed_config

ROOT = Path(__file__).resolve().parent
grid.LOG = ROOT / 'results' / f'run_today_{time.strftime("%Y%m%d_%H%M%S")}.log'
SEED_1, SEED_2, SEED_3 = 44, 10042, 20042
S16 = 'config_sports16_relabel.yaml'
S64 = 'config_sports64_relabel.yaml'
ALPHA = 'config_sports16_relabel_alpha.yaml'
MAIN_ATTACK = 'split_r512_a16'


def control(cfg, arm):
    """Train one control arm to the forget level, evaluate it and run the main attack."""
    return [('run_trial.py', '--config', cfg, '--method', arm, '--level', 'forget', '--attack', MAIN_ATTACK),
            ('analyze.py', '--config', cfg)]


def steps():
    """(label, trains, commands); configs for other seeds are generated as run_sports16.py does."""
    s16 = {seed: seed_config(S16, seed) for seed in (SEED_1, SEED_2, SEED_3)}
    gradcheck = [('sam_gradcheck.py', '--config', s16[SEED_1], '--method', arm, *level)
                 for arm in ('ul_flu_sam', 'ul_sam') for level in ((), ('--level', 'forget'))]
    return [
        ('seed 3: random layers outside FLU (ul_randx)', True, control(s16[SEED_3], 'ul_randx')),
        ('Sports-64: random layers outside FLU (ul_randx)', True, control(S64, 'ul_randx')),
        ('seed 2: random layers outside FLU (ul_randx)', True, control(s16[SEED_2], 'ul_randx')),
        ('seed 1: SAM gradient-norm check (ul_flu_sam, ul_sam)', False, gradcheck),
        ('seed 1: attack strength, alpha 8/32/16/64', False,
         [('run_trial.py', '--config', ALPHA, '--method', 'base', 'ul_flu', 'ul_rand', '--level', 'forget',
           '--attack-only'), ('analyze.py', '--config', ALPHA)]),
        ('seed 3 + Sports-64: depth-matched random layers outside FLU (ul_randd)', True,
         control(s16[SEED_3], 'ul_randd') + control(S64, 'ul_randd')),
        ('seed 3 + Sports-64: SAM versions (ul_randx_sam)', True,
         control(s16[SEED_3], 'ul_randx_sam') + control(S64, 'ul_randx_sam')),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hours', type=float, default=5.0, help='start nothing after this (default 5)')
    parser.add_argument('--stop-training-at', type=float, default=4.0,
                        help='start no training step after this many hours (default 4)')
    parser.add_argument('--only', type=int, nargs='+', help='step numbers to run (default: all)')
    parser.add_argument('--list', action='store_true', help='print the steps and exit')
    args = parser.parse_args()
    plan = steps()
    if args.list:
        for number, (label, trains, commands) in enumerate(plan, 1):
            print(f'{number}. {label}' + (' [trains]' if trains else ''))
            for command in commands:
                print('     python', ' '.join(command))
        return
    start, report = time.time(), []
    say(f'start, log: {grid.LOG}')
    for number, (label, trains, commands) in enumerate(plan, 1):
        if args.only and number not in args.only:
            continue
        elapsed = (time.time() - start) / 3600
        limit = args.stop_training_at if trains else args.hours
        if elapsed >= limit:
            report.append(f'{number}. SKIPPED ({elapsed:.2f} h >= {limit} h): {label}')
            say(report[-1])
            continue
        say(f'step {number} at {elapsed:.2f} h: {label}')
        began = time.time()
        # A failed step (e.g. out of memory on ul_sam) does not stop the steps after it.
        ok = all([run(*command, required=False) for command in commands])
        report.append(f'{number}. {"done" if ok else "FAILED"} in {(time.time() - began) / 3600:.2f} h: {label}')
        say(report[-1])
    say(f'finished at {(time.time() - start) / 3600:.2f} h')
    for line in report:
        say(line)
    if any('FAILED' in line for line in report):
        sys.exit(1)


if __name__ == '__main__':
    main()
