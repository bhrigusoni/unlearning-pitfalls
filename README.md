# Pitfalls in Measuring Robust Unlearning

Code, configurations and evaluation outputs for

> B. Soni, Y. S. Dadwhal and T. Datta, "Pitfalls in Measuring Robust Unlearning: A Controlled
> Re-test of Where-to-Unlearn on Llama-3-8B," ICon INDIA 2026.

The pipeline unlearns 16 or 64 Sports Facts from Llama-3-8B with six arms (Whole model, FLU,
Random; each with and without SAM), saves three forgetting levels per arm (same steps, below a
loss value, same forgetting), attacks every level with LoRA relearning attacks, and analyses the
facts recovered at attack step 10 with exact McNemar tests.

## Layout

| Path | What |
|---|---|
| `prepare_data.py` | config loading, data splits, model loading, shared helpers |
| `flu.py` | FLU probe localization, size-matched random layers, controls outside FLU (`ul_randx`, depth-matched `ul_randd`) |
| `train.py`, `sam.py` | unlearning (Relabel, NPO), SAM step, level checkpoints |
| `evaluate.py` | closed-set and open-vocabulary accuracy, sport mass, collapse diagnostics |
| `relearn.py` | LoRA relearning attacks (split and held-out) |
| `run_trial.py` | one experiment end to end (`--method`, `--level`, `--attack`, `--preflight`, `--self-test`) |
| `run_sports16.py` | the full grid for one or more configs and seeds |
| `run_today.py` | the reviewer runs: random layers outside FLU, depth-matched, SAM versions, attack strength, SAM gradient check |
| `analyze.py`, `analyze_seeds.py`, `plot.py` | step-10 readout, McNemar + Holm, pooled seeds, curves |
| `sam_gradcheck.py` | one-step SAM gradient check (forget vs. retain share of the update) |
| `test_trial.py` | offline test on a tiny random Llama (CPU, no download) |
| `config_*.yaml` | the configurations used in the paper |
| `paper_analysis/` | scripts that produce the paper's tables, statistics and figures from `results/` |
| `results/` | evaluation outputs (download separately, see below) |

## Install

Python 3.11. Install PyTorch with the CUDA wheel for your system first, then:

```bash
pip install -r requirements.txt
```

Llama-3-8B is gated: accept its licence on Hugging Face and run `huggingface-cli login`. The
configs pin the model revision used in the paper. One RTX A6000 (48 GB) ran everything.

Offline check (no GPU, no download):

```bash
python test_trial.py
```

## Data

The Sports Facts CSV is not included; see `data/README.md`.

## Running

```bash
python run_sports16.py config_sports16_relabel.yaml:44,10042,20042 config_sports16_npo.yaml
python run_sports16.py config_sports64_relabel.yaml config_sports64_npo.yaml
python run_today.py
```

Runs resume after a crash; finished stages are skipped. Seeds: the paper's Sports-16 Relabel
seeds 1, 2, 3 are 44, 10042 and 20042; Sports-16 NPO and both Sports-64 experiments use 42.
Seeds must be at least 10000 apart, because sub-seeds are derived as seed + 1..7000.

Approximate cost on one RTX A6000: Sports-16 Relabel 4.5 h, Sports-16 NPO 16.5 h per seed,
Sports-64 NPO 16.7 h, reviewer runs 4.2 h; about 75 GPU-hours in total.

## Results

`results.tar.gz` (attached to the GitHub release, about 150 MB packed, 1.2 GB unpacked) holds
every output except model weights. Unpack it in the repository root to get `results/`:

```bash
tar -xzf results.tar.gz
```

Each
experiment folder (`results/sports16_relabel`, `..._seed10042`, `..._seed20042`,
`sports16_npo`, `sports64_relabel`, `sports64_npo`) contains:

- `split.json` (the exact forget, retain and held-out facts), `mask.json` (FLU layers and probe
  accuracies), `mask_random*.json` (random layers);
- per arm: `settings.json`, `training.json`, `levels.json`, `level_checks.json`, and per level
  `evaluation.json` (pre-attack, per fact) and `relearning/<attack>/curve.json` (every
  evaluated attack step);
- per-fact attack evaluations `evaluation_NNNN.json` for attack steps 0, 5, 10, 20, 30 and
  100 only (the steps the paper reads); the other steps are summarized in `curve.json`;
- `analysis_*`, `readout_*`, `runner_up_*` CSV/JSON from `analyze.py`, and run logs.

Model weights (about 520 GB) are not released. The output folders of the Sports-16 layer
controls (`ul_randx`, `ul_randd`, `ul_randx_sam` in seeds 2 and 3), the attack-strength sweep
and the SAM gradient check are not in the archive; their results are in
`results/run_today_20261030_100940.log`.

Attack names: `split_r8`, `split_r512_a16`, `heldout_r512_a16` are the paper's attacks;
`split_r512` (alpha 1024) and `split_r512_a128` are the diverging attacks of pitfall P4;
`split_r512_a{8,16,32,64}_s30` is the attack-strength sweep.

## Reproducing the paper's numbers

```bash
cd paper_analysis
python collect.py all.csv base.json   # flat table from results/
python tests.py                        # McNemar / Holm tests, conditional nulls
python reviewer_stats.py               # CIs, readout-step and threshold sensitivity, log-odds
python figs.py                         # Figures 1-3 into figures/
```

`tests_out.txt` and `reviewer_stats_out.txt` are the outputs quoted in the paper. Table I comes
from the `mask*.json` files, Table III from `levels.json` and `reviewer_stats.py` (section 4),
Table IV from `readout_arms_forget_split_r8.csv`. The layer controls, the attack-strength sweep
and the SAM gradient check are in `results/run_today_20261030_100940.log` (the `control` lines
of `[forget_split_r512_a16]`, the `[forget_split_r512_a*_s30]` blocks, and the step-0
`Summary` blocks of step 4 for the SAM gradient check).

## Licence

Code: MIT (see `LICENSE`). Llama-3 is under Meta's licence; the dataset under its own terms.

## Citation

See `CITATION.cff`.
