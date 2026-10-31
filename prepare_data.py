"""Dataset preparation and small shared I/O/model helpers. No download on import."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def without(cfg, *sections):
    """cfg minus sections a stage never reads, so tuning them keeps that stage's outputs valid."""
    return {k: v for k, v in cfg.items() if k not in sections}


FORGET_OBJECTIVES = ('relabel', 'npo', 'uniform', 'unlikelihood')
# Unlearning levels, each a checkpoint of the SAME training run (constant learning
# rate, so stopping early equals a separate shorter run):
# steps: train.levels.steps exactly; loss: first check where the full-set forget loss
# is below train.levels.loss; forget: first check with 0 forget facts correct and every
# fact's normalized true-sport probability below train.levels.forget_probability.
LEVELS = ('steps', 'loss', 'forget')
# The 2x2 of localization x SAM comes first so a partial run already answers the
# headline claim; the random-localization controls follow. They live here, not in
# train.py, so analyze.py can name the arms without importing torch.
METHODS = ('ul', 'ul_flu', 'ul_sam', 'ul_flu_sam', 'ul_rand', 'ul_rand_sam')
# Reviewer controls (2026-10-30), run only when named with --method, so a default run
# still trains the six arms above. randx: random layers drawn only from outside the FLU
# set. randd: the same, depth-matched to FLU (as many layers per depth quarter).
CONTROL_METHODS = ('ul_randx', 'ul_randx_sam', 'ul_randd', 'ul_randd_sam')
ALL_METHODS = METHODS + CONTROL_METHODS
MASKS = {'flu': 'mask.json', 'rand': 'mask_random.json',
         'randx': 'mask_random_complement.json', 'randd': 'mask_random_depth.json'}
# flu.py --localization name for each random mask.
RANDOM_LOCALIZATIONS = {'rand': 'random', 'randx': 'random_complement', 'randd': 'random_depth'}


def mask_tag(method):
    """The MASKS key in an arm's name ('ul_randx_sam' -> 'randx'), or None for unmasked arms.

    Matched on whole name parts, not substrings: 'rand' is a substring of 'randx'.
    """
    return next((part for part in method.split('_') if part in MASKS), None)
FLU_SELECTIONS = ('all_layers', 'band')
# What the relearning attack trains on. forget_heldout: disjoint athletes the model
# still knows (indirect; its gradient mostly repairs collateral damage).
# forget_split: k-1 folds of the forget facts themselves, scored on the unseen fold.
ATTACK_DATA = ('forget_heldout', 'forget_split')
ATTACK_TARGETS = {'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'mlp': ['gate_proj', 'up_proj', 'down_proj'],
                  'all': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
# Harness switches, never experimental settings: they are excluded from every
# fingerprint, so turning a check off does not invalidate the artifacts it guards.
# `level` is set per stage by stage_cfg, not in the YAML; relearn.py adds it to the
# attack fingerprint itself, because each level is a different starting model.
HARNESS_KEYS = ('strict_identity', 'level')
# The early attack readout: after ~10 steps the attack itself starts to damage the
# model, so later forget accuracy partly measures that damage rather than recovery.
READOUT_STEP = 10


def config(path='config.yaml'):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg.setdefault('strict_identity', False)
    for key in ('dataset', 'output'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg['batch_size'] != 4:
        raise ValueError('This trial requires real batch_size=4.')
    if cfg.get('eval_batch_size', cfg['batch_size']) < 1:
        raise ValueError('eval_batch_size must be positive.')
    if cfg['splits']['forget'] < 16:
        raise ValueError('The forget set needs at least 16 facts.')
    if min(cfg['splits']['retain'], cfg['splits']['forget_heldout']) < cfg['batch_size']:
        raise ValueError('Retain and heldout splits must each fill a batch.')
    if not 0 <= cfg['splits']['minimum_probability'] <= 1:
        raise ValueError('minimum_probability must be between zero and one.')
    if cfg['dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Supported dtypes are float32 and bfloat16; FP16 requires a different scaler path.')
    if not cfg['relearn']:
        raise ValueError('relearn must name at least one attack.')
    for name, attack in cfg['relearn'].items():
        if attack['rank'] < 1 or attack['alpha'] <= 0:
            raise ValueError(f'{name}: LoRA rank and alpha must be positive.')
        if attack['steps'] < 1 or attack['learning_rate'] <= 0:
            raise ValueError(f'Invalid attack settings for {name}.')
        if attack['evaluate_every'] < 1 or attack.get('checkpoint_every', 1) < 1:
            raise ValueError(f'{name}: evaluation/checkpoint intervals must be positive.')
        if attack.get('data', 'forget_heldout') not in ATTACK_DATA:
            raise ValueError(f'{name}: data must be one of {ATTACK_DATA}.')
        if attack.get('targets', 'attention') not in ATTACK_TARGETS:
            raise ValueError(f'{name}: targets must be one of {tuple(ATTACK_TARGETS)}.')
        if not 2 <= attack.get('folds', 2) <= cfg['splits']['forget'] // cfg['batch_size']:
            raise ValueError(f'{name}: folds must leave at least one full batch of forget facts per fold.')
    flu = cfg['flu']
    flu.setdefault('selection', 'all_layers')
    flu.setdefault('gain_threshold', 0.0)
    if flu['selection'] not in FLU_SELECTIONS:
        raise ValueError(f"flu.selection must be one of {FLU_SELECTIONS}.")
    if flu['selection'] == 'band' and not 0 < flu.get('localization_budget_fraction', 0) <= 1:
        raise ValueError('localization_budget_fraction must be in (0, 1] for selection: band.')
    if flu['selection'] == 'all_layers' and not 0 <= flu['gain_threshold'] < 1:
        raise ValueError('gain_threshold must be in [0, 1); 0 keeps every layer the probe rises on.')
    train = cfg['train']
    if train['steps'] < 1 or train['learning_rate'] <= 0 or train['checkpoint_every'] < 1:
        raise ValueError('Invalid train settings.')
    levels = train['levels']
    if levels['check_every'] < 1 or not 1 <= levels['steps'] <= train['steps']:
        raise ValueError('levels.check_every must be positive and levels.steps within train.steps.')
    if levels['loss'] <= 0 or not 0 < levels['forget_probability'] < 1:
        raise ValueError('levels.loss must be positive and levels.forget_probability in (0, 1).')
    if train['forget_objective'] == 'npo' and train.get('npo_beta', 0) <= 0:
        raise ValueError('npo needs a positive train.npo_beta.')
    if train.get('select_batch_size', cfg['batch_size']) < 1:
        raise ValueError('select_batch_size must be positive.')
    if not 0 < train.get('min_learning_rate', train['learning_rate']) <= train['learning_rate']:
        raise ValueError('min_learning_rate must be positive and at most learning_rate (the peak).')
    if not 0 <= train.get('warmup_steps', 0) < train['steps']:
        raise ValueError('warmup_steps must be in [0, steps).')
    if cfg['train']['max_grad_norm'] <= 0:
        raise ValueError('Clipping norm must be positive.')
    if 'sam_rho' in train and not {'sam_rho_forget', 'sam_rho_retain'} <= train.keys():
        raise ValueError('v2 splits sam_rho into sam_rho_forget and sam_rho_retain; set both.')
    if train['sam_rho_forget'] <= 0:
        raise ValueError('sam_rho_forget must be positive; the forget term is always flattened.')
    if train['sam_rho_retain'] < 0:
        raise ValueError('sam_rho_retain must be non-negative; 0 evaluates retain at the unperturbed weights.')
    if min(cfg['train']['forget_weight'], cfg['train']['retain_weight']) <= 0:
        raise ValueError('Both loss weights must be positive.')
    if cfg['train']['forget_objective'] not in FORGET_OBJECTIVES:
        raise ValueError(f'forget_objective must be one of {FORGET_OBJECTIVES}.')
    return cfg


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(cfg, source=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if cfg['device'].startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Run the trial on the remote GPU.')
        if cfg['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('GPU does not support BF16.')
    seed_all(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        source or cfg['model'], revision=None if source else cfg['revision'],
        torch_dtype=getattr(torch, cfg['dtype']), low_cpu_mem_usage=True,
        attn_implementation='sdpa', device_map={'': cfg['device']},
    )
    model.config.use_cache = False
    model.eval()
    # Deterministic closure: no dropout differences between the two SAM passes.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model, tokenizer


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        raw = list(csv.DictReader(stream))
    if not raw or not {'athlete', 'sport', 'prompt'} <= raw[0].keys():
        raise ValueError('CSV requires athlete, sport, prompt columns.')
    rows, seen = [], {}
    for row in raw:
        athlete, sport, prompt = row['athlete'].strip(), row['sport'].strip(), row['prompt']
        if not athlete or not sport or athlete not in prompt:
            raise ValueError(f'Invalid record: {athlete!r}')
        value = {'athlete': athlete, 'sport': sport, 'prompt': prompt}
        if athlete in seen:
            if seen[athlete] != value:
                raise ValueError(f'Conflicting records for {athlete}; resolve before splitting.')
            continue
        seen[athlete] = value
        rows.append(value)
    if sorted({r['sport'] for r in rows}) != ['baseball', 'basketball', 'football']:
        raise ValueError('Expected baseball, basketball and football.')
    return rows


def target_ids(tokenizer, sports):
    result = {}
    for sport in sports:
        ids = tokenizer.encode(' ' + sport, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'{sport!r} is not a single token for this tokenizer.')
        result[sport] = ids[0]
    return result


def encode(rows, tokenizer, cfg, offsets=False):
    import torch
    encoded = tokenizer([r['prompt'] for r in rows], padding=True, truncation=False,
                        return_tensors='pt', return_offsets_mapping=offsets)
    if encoded['input_ids'].shape[1] > cfg['max_length']:
        raise ValueError('Prompt exceeds max_length; increase it rather than truncate the athlete.')
    span = encoded.pop('offset_mapping', None)
    ids = target_ids(tokenizer, sorted({r['sport'] for r in rows}))
    # Verify target encoding at the actual prompt boundary, including leading space.
    for row in rows:
        prompt_ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        full = tokenizer.encode(row['prompt'] + ' ' + row['sport'], add_special_tokens=False)
        if full != prompt_ids + [ids[row['sport']]]:
            raise ValueError('Prompt/answer token boundary is not a single appended token.')
    encoded = {key: value.to(cfg['device']) for key, value in encoded.items()}
    targets = torch.tensor([ids[r['sport']] for r in rows], device=cfg['device'])
    return encoded, targets, span


def next_logits(model, inputs):
    import torch
    output = model(**inputs, use_cache=False).logits
    positions = inputs['attention_mask'].sum(1) - 1
    return output[torch.arange(len(positions), device=output.device), positions].float()


def balanced_take(rows, size, seed):
    sports = sorted({r['sport'] for r in rows})
    if len(sports) != 3 or size < 3:
        raise ValueError('Each split must represent all three sports.')
    rng = random.Random(seed)
    rng.shuffle(sports)
    chosen = []
    for i, sport in enumerate(sports):
        pool = sorted((r for r in rows if r['sport'] == sport), key=lambda r: r['athlete'])
        rng.shuffle(pool)
        count = size // 3 + (i < size % 3)
        if len(pool) < count:
            raise ValueError(f'Need {count} {sport} facts, have {len(pool)}.')
        chosen.extend(pool[:count])
    rng.shuffle(chosen)
    return chosen


def batches(rows, size=4):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def step_batch(rows, step, seed, size=4):
    if len(rows) < size:
        raise ValueError('Not enough examples for a full batch.')
    per_epoch = (len(rows) + size - 1) // size
    epoch, slot = divmod(step, per_epoch)
    order = list(rows)
    random.Random(seed + epoch).shuffle(order)
    start = slot * size
    return [order[(start + j) % len(order)] for j in range(size)]


def require(cfg, ok, message):
    """Enforce an identity fingerprint, or, under strict_identity: false, warn and continue.

    Every check this guards compares a recorded fingerprint against the current
    settings. Structural checks -- tensor shapes, optimizer scope, missing files --
    are never routed through here and still raise, because reusing a checkpoint whose
    shapes disagree is a bug rather than a deliberate reuse.
    """
    if ok:
        return
    if cfg.get('strict_identity', False):
        raise ValueError(message)
    print(f'WARNING (strict_identity: false): {message}', flush=True)


def load_split(cfg):
    payload = read_json(Path(cfg['output']) / 'split.json')
    # The split, base evaluation and masks never read train/relearn settings.
    require(cfg, payload['config_hash'] == fingerprint(without(cfg, 'train', 'relearn', *HARNESS_KEYS)),
            'Configuration changed since the split was prepared; the saved split is being reused.')
    require(cfg, payload['dataset_sha256'] == hashlib.sha256(Path(cfg['dataset']).read_bytes()).hexdigest(),
            'Dataset changed since preparation.')
    return payload


def attack_tag(cfg):
    """Folder name for one attack setting, so several can share an unlearned arm."""
    settings = cfg['relearn']
    if settings.get('tag'):
        return str(settings['tag'])
    tag = 'lr{learning_rate:g}_r{rank}_a{alpha}_s{steps}'.format(**settings)
    # Suffixes only for non-default attacks, so earlier attack folders keep their names.
    if settings.get('data', 'forget_heldout') == 'forget_split':
        tag += f"_split{settings.get('folds', 2)}"
    if settings.get('targets', 'attention') != 'attention':
        tag += '_' + settings['targets']
    return tag


def forget_folds(rows, folds, seed):
    """Partition the forget facts into `folds` sport-stratified folds.

    Each sport is shuffled and dealt round-robin with one counter running across
    sports, so fold sizes differ by at most one and every fold mixes sports.
    Deterministic in the seed, so every arm is attacked on identical folds.
    """
    rng = random.Random(seed)
    result, dealt = [[] for _ in range(folds)], 0
    for sport in sorted({r['sport'] for r in rows}):
        pool = sorted((r for r in rows if r['sport'] == sport), key=lambda r: r['athlete'])
        rng.shuffle(pool)
        for row in pool:
            result[dealt % folds].append(row)
            dealt += 1
    return result


def stage_cfg(cfg, level=None, attack=None):
    """cfg for one stage: the unlearning level it reads and the one attack it runs.

    config() returns every attack under relearn; the stage code sees just one, flat,
    with its YAML name as the tag, so every attack lands in its own folder.
    """
    result = dict(cfg)
    if level is not None:
        result['level'] = level
    if attack is not None:
        result['relearn'] = {**cfg['relearn'][attack], 'tag': attack}
    return result


def arm_dir(cfg, method):
    """Where an arm's evaluation and attacks live: one folder per level, except base."""
    out = Path(cfg['output']) / method
    return out if method == 'base' else out / cfg['level']


def level_step(cfg, method):
    """The training step saved for cfg['level'], or None if the arm never reached it."""
    if method == 'base':
        return 0
    path = Path(cfg['output']) / method / 'levels.json'
    return read_json(path)['reached'].get(cfg['level'], {}).get('step') if path.exists() else None


def attack_dir(cfg, method):
    return arm_dir(cfg, method) / 'relearning' / attack_tag(cfg)


def prepare(cfg):
    import torch
    from evaluate import evaluate_rows
    out = Path(cfg['output'])
    if (out / 'split.json').exists():
        return load_split(cfg)
    rows = read_rows(cfg['dataset'])
    model, tokenizer = load_model(cfg)
    report = evaluate_rows(model, tokenizer, rows, cfg)
    write_json(out / 'base_all.json', report)
    known = [row for row, rec in zip(rows, report['records'])
             if rec['correct'] and rec['target_probability'] >= cfg['splits']['minimum_probability']]
    forget = balanced_take(known, cfg['splits']['forget'], cfg['seed'])
    # NPO's reference model is the base model; its log P(true sport) on the forget
    # facts is all NPO needs from it, so it is stored here once and no second 8B copy
    # has to sit in memory during training.
    reference = {row['athlete']: -rec['target_loss'] for row, rec in zip(rows, report['records'])}
    forget = [dict(r, reference_logprob=reference[r['athlete']]) for r in forget]
    used = {r['athlete'] for r in forget}
    remaining = [r for r in rows if r['athlete'] not in used]
    heldout = balanced_take(remaining, cfg['splits']['forget_heldout'], cfg['seed'] + 1)
    used.update(r['athlete'] for r in heldout)
    retain = balanced_take([r for r in remaining if r['athlete'] not in used],
                           cfg['splits']['retain'], cfg['seed'] + 2)
    result = {'config_hash': fingerprint(without(cfg, 'train', 'relearn', *HARNESS_KEYS)), 'model_commit': getattr(model.config, '_commit_hash', None),
              'dataset_sha256': hashlib.sha256(Path(cfg['dataset']).read_bytes()).hexdigest(),
              'forget': forget, 'retain': retain, 'forget_heldout': heldout}
    write_json(out / 'split.json', result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Prepared splits: forget={len(forget)}, retain={len(retain)}, heldout={len(heldout)}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--inspect', action='store_true', help='Inspect CSV without loading a model.')
    args = parser.parse_args()
    cfg = config(args.config)
    if args.inspect:
        from collections import Counter
        rows = read_rows(cfg['dataset'])
        print(json.dumps({'unique_athletes': len(rows), 'sports': dict(Counter(r['sport'] for r in rows))}, indent=2))
    else:
        prepare(cfg)
