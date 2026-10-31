#!/usr/bin/env bash
# Full grid (6 arms x 3 levels x 3 attacks) for each config given, then stats and plots.
# Default: Sports-16 Relabel then NPO. Sports-64:
#   bash run_sports16.sh config_sports64_relabel.yaml config_sports64_npo.yaml
# Re-run the same command after a crash: finished stages are skipped, training resumes.
# Usage: bash run_sports16.sh            (PYTHON=/path/to/python bash run_sports16.sh)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
mkdir -p results
LOG="results/run_grid_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== $(date) start, log: $LOG"
"$PYTHON" run_trial.py --self-test

CONFIGS=("$@")
[ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(config_sports16_relabel.yaml config_sports16_npo.yaml)
for cfg in "${CONFIGS[@]}"; do
    echo "== $(date) $cfg: preflight"
    "$PYTHON" run_trial.py --config "$cfg" --preflight
    echo "== $(date) $cfg: prepare, FLU, train, evaluate, attacks"
    "$PYTHON" run_trial.py --config "$cfg"
    echo "== $(date) $cfg: analysis and plots"
    "$PYTHON" analyze.py --config "$cfg" || echo "analysis failed for $cfg; run it later on the laptop"
    "$PYTHON" plot.py --config "$cfg" || echo "plots skipped for $cfg (pip install matplotlib)"
done

echo "== $(date) done. Results: the output folder named in each config"
