"""Full grid for each config in turn (default: Sports-16 Relabel then NPO), then stats and plots.
Works on Windows and Linux.

Usage (from anywhere):  python unlearning_v3/run_sports16.py
Sports-64:              python unlearning_v3/run_sports16.py config_sports64_relabel.yaml config_sports64_npo.yaml
More seeds:             python unlearning_v3/run_sports16.py config_sports16_relabel.yaml:42,10042,20042
  Space seeds by 10000: the code derives sub-seeds as seed + 1..7000 (+ step), so close
  seeds (42, 43) would share data samples and batch orders.
  A config without ":seeds" runs its own seed. Each extra seed gets a generated
  config_<name>_seed<N>.yaml and output folder <output>_seed<N>; the config's own seed
  keeps its folder, so finished runs are reused. analyze_seeds.py then pools the seeds.
Re-run the same command after a crash: finished stages are skipped, training resumes.
Uses the Python that runs this file, so activate the right environment first.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / 'results').mkdir(exist_ok=True)
CONFIGS = (sys.argv[1:] if __name__ == '__main__' else None) or ['config_sports16_relabel.yaml',
                                                                  'config_sports16_npo.yaml']
LOG = ROOT / 'results' / f'run_grid_{time.strftime("%Y%m%d_%H%M%S")}.log'


def say(message):
    line = f'== {time.strftime("%Y-%m-%d %H:%M:%S")} {message}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def run(*args, required=True):
    """Run one script, copying its output to the console and the log."""
    with subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
        with LOG.open('a', encoding='utf-8') as stream:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
        code = process.wait()
    if code and required:
        say(f'FAILED: {" ".join(args)} (exit {code}). Fix it, then re-run this script to resume.')
        sys.exit(code)
    return code == 0


def run_config(cfg):
    say(f'{cfg}: preflight')
    run('run_trial.py', '--config', cfg, '--preflight')
    say(f'{cfg}: prepare, FLU, train, evaluate, attacks')
    run('run_trial.py', '--config', cfg)
    say(f'{cfg}: analysis and plots')
    if not run('analyze.py', '--config', cfg, required=False):
        say(f'analysis failed for {cfg}; run it later on the laptop')
    if not run('plot.py', '--config', cfg, required=False):
        say(f'plots skipped for {cfg} (pip install matplotlib)')


def seed_config(name, seed):
    """Copy of config `name` with another seed and its own output folder; comments kept."""
    path = ROOT / name
    text = path.read_text(encoding='utf-8')
    own = int(re.search(r'^seed:\s*(\d+)', text, re.M).group(1))
    if seed == own:
        return name
    text, n_seed = re.subn(r'^seed:\s*\d+', f'seed: {seed}', text, flags=re.M)
    text, n_out = re.subn(r'^(output:\s*\S+)', rf'\g<1>_seed{seed}', text, flags=re.M)
    if (n_seed, n_out) != (1, 1):
        raise ValueError(f'{name}: expected one top-level seed: and output: line')
    target = path.with_name(f'{path.stem}_seed{seed}{path.suffix}')
    target.write_text(text, encoding='utf-8')
    return target.name


if __name__ == '__main__':  # run_today.py imports run, say and seed_config from here
    say(f'start, log: {LOG}')
    run('run_trial.py', '--self-test')
    for entry in CONFIGS:
        # "name.yaml:42,10042" -> seed list; fullmatch so a Windows "C:\..." path still works.
        match = re.fullmatch(r'(.+\.yaml):([\d,]+)', entry)
        name, seeds = match.groups() if match else (entry, '')
        group = [seed_config(name, int(s)) for s in seeds.split(',') if s] if seeds else [name]
        for cfg in group:
            run_config(cfg)
        if len(group) > 1:
            say(f'{name}: pooled analysis over {len(group)} seeds')
            if not run('analyze_seeds.py', '--configs', *group, required=False):
                say(f'pooled analysis failed for {name}; run analyze_seeds.py later on the laptop')
    say('done. Results: the output folder named in each config')
