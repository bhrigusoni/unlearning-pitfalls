"""Six unlearning arms sharing losses, batches, SGD and checkpoints.

One run per arm saves a checkpoint for each unlearning level (prepare_data.LEVELS):
the model evaluated and attacked at level L is the weights at the step L was reached.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from evaluate import evaluate_rows
from flu import apply_mask
from prepare_data import (ALL_METHODS, FORGET_OBJECTIVES, HARNESS_KEYS, LEVELS, MASKS, METHODS, batches, config,
                          encode, fingerprint, level_step, load_model, mask_tag, without,
                          load_split, next_logits, read_json, require, seed_all, step_batch,
                          target_ids, write_json)
from sam import MasterSGD, SAM, gradient_norm, put_gradients, take_gradients

# METHODS/MASKS are defined in prepare_data and re-exported here, so the many
# `from train import METHODS` call sites keep working.
# Console cadence only; every step is still recorded in training.json.
LOG_EVERY = 10


SPORTS = ('baseball', 'basketball', 'football')


def wrong_sport(athlete, true_sport, seed):
    """The single wrong sport this athlete is relabelled to.

    Deterministic in (seed, athlete) so the target is identical across steps,
    checkpoint resumes and all four methods. A target that moved between steps
    would never converge.

    Chosen at random among the two wrong sports rather than by a fixed rule:
    a fixed map (baseball -> basketball, always) is trivially invertible, so the
    fact would still be fully recoverable from the unlearned model.
    """
    others = [sport for sport in SPORTS if sport != true_sport]
    if len(others) != len(SPORTS) - 1:
        raise ValueError(f'{true_sport!r} is not one of {SPORTS}.')
    digest = hashlib.sha256(f'{seed}:{athlete}'.encode()).digest()
    return others[digest[0] % len(others)]


def uniform(logits, candidates):
    """Entropy maximization over the closed answer set: KL(uniform || q) + log 3.

    Floor is log 3 at q = (1/3, 1/3, 1/3). The gradient on each candidate logit is
    q_j - 1/3, which sums to zero over the set, so shifting all three down together
    is a null direction and cannot satisfy the loss.
    """
    return -logits.float()[:, candidates].log_softmax(1).mean()


def unlikelihood(logits, targets):
    """Legacy target-only objective, -log(1 - p_y), over the full vocabulary.

    Kept only to reproduce its documented failure: it depends on p_y alone, so it is
    satisfied by scaling all three sport logits down together, which preserves their
    ranking. Its gradient on the target logit is exactly p_y, so it also
    self-extinguishes once p_y is small.
    """
    others = logits.float().scatter(1, targets[:, None], -torch.inf)
    return (torch.logsumexp(logits.float(), 1) - torch.logsumexp(others, 1)).mean()


def npo(logits, targets, reference, beta):
    """Negative preference optimization: -(2/beta) log sigmoid(-beta * log(p / p_ref)).

    p is P(true sport) under the model, p_ref under the base model (stored in the
    split). Equals (2/beta) log 2 at the start, falls toward 0 as p drops below p_ref,
    and its weight on each fact shrinks once that fact is forgotten.
    """
    logp = logits.float().log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    return -(2 / beta) * torch.nn.functional.logsigmoid(-beta * (logp - reference)).mean()


def objective(model, tokenizer, rows, cfg, label):
    """One objective's unweighted loss at the model's current weights."""
    inputs, targets, _ = encode(rows, tokenizer, cfg)
    logits = next_logits(model, inputs)
    objective = cfg['train']['forget_objective']
    if label == 'retain':
        loss = torch.nn.functional.cross_entropy(logits, targets)
    elif objective == 'relabel':
        # Naming which sport must end up on top is what makes this unsatisfiable by
        # scaling all three sport logits down together.
        ids = target_ids(tokenizer, SPORTS)
        relabelled = [ids[wrong_sport(row['athlete'], row['sport'], cfg['seed'])] for row in rows]
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(relabelled, device=logits.device))
    elif objective == 'npo':
        reference = torch.tensor([row['reference_logprob'] for row in rows], device=logits.device)
        loss = npo(logits, targets, reference, cfg['train']['npo_beta'])
    elif objective == 'uniform':
        ids = target_ids(tokenizer, SPORTS)
        loss = uniform(logits, [ids[sport] for sport in SPORTS])
    else:
        loss = unlikelihood(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite {label} loss.')
    return loss


def loss_backward(model, tokenizer, rows, cfg, label):
    """Compute one objective at the model's current weights and backpropagate it."""
    loss = objective(model, tokenizer, rows, cfg, label)
    (cfg['train'][label + '_weight'] * loss).backward()
    return loss.detach().item()


@torch.no_grad()
def selection_loss(model, tokenizer, split, cfg):
    """The training objective over the FULL forget and retain sets, for the level checks.

    The per-step training loss comes from one small batch, so levels are judged on
    this instead. Always the clean (unperturbed) loss, so SAM and non-SAM arms are
    judged on the same quantity. forget_heldout is deliberately absent: it is the
    attack's data and the collateral-damage probe, and must not steer which model is kept.
    Forward-only, so it runs at select_batch_size rather than the training batch of 4;
    each prompt is scored independently, so the batch size does not change the loss.
    """
    was_training = model.training
    model.eval()
    size = cfg['train'].get('select_batch_size', cfg['batch_size'])
    try:
        losses = {}
        for label in ('forget', 'retain'):
            rows = split[label]
            losses[label + '_loss'] = sum(objective(model, tokenizer, batch, cfg, label).item() * len(batch)
                                          for batch in batches(rows, size)) / len(rows)
    finally:
        model.train(was_training)
    losses['total_loss'] = sum(cfg['train'][label + '_weight'] * losses[label + '_loss']
                               for label in ('forget', 'retain'))
    return losses


def sam_step(model, tokenizer, rows, cfg, label, sam, optimizer, losses, step):
    """One SAM pass pair for a single objective, leaving its perturbed gradient behind.

    The clean gradient sets the ascent direction and is then discarded; what survives
    in .grad is the gradient at theta + delta, which is the term this objective
    contributes to the update. Any gradient present on entry is overwritten, so the
    caller parks anything it still needs.
    """
    seed_all(cfg['seed'] + step + (0 if label == 'forget' else 7000))
    optimizer.zero_grad(set_to_none=True)
    losses[label + '_loss'] = loss_backward(model, tokenizer, rows, cfg, label)
    try:
        radius = sam.perturb()
        if radius == 0:
            raise RuntimeError(f'SAM {label} perturbation vanished at this precision. '
                               'Adjust rho/dtype in a new run.')
        optimizer.zero_grad(set_to_none=True)
        seed_all(cfg['seed'] + step + (0 if label == 'forget' else 7000))
        losses['perturbed_' + label + '_loss'] = loss_backward(model, tokenizer, rows, cfg, label)
    finally:
        sam.restore()
    return radius


def learning_rate(cfg, step):
    """Linear warmup to learning_rate, then cosine decay to min_learning_rate.

    A pure function of the step, so a resumed run replays the same schedule, and
    every arm gets the identical schedule. Without min_learning_rate/warmup_steps
    it is the old constant rate.
    """
    train = cfg['train']
    peak = train['learning_rate']
    floor = train.get('min_learning_rate', peak)
    warmup = train.get('warmup_steps', 0)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, train['steps'] - 1 - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def mask_path(cfg, method):
    """The localization mask this arm trains inside, or None for the all-parameter arms."""
    tag = mask_tag(method)
    return Path(cfg['output']) / MASKS[tag] if tag else None


def setup_scope(model, cfg, method):
    if method not in ALL_METHODS:
        raise ValueError(f'Unknown method: {method}')
    path = mask_path(cfg, method)
    if path is None:
        model.requires_grad_(True)
    else:
        mask = read_json(path)
        require(cfg, mask['split_hash'] == fingerprint(load_split(cfg)),
                f'{path.name} does not match current split.')
        apply_mask(model, mask)
    return [p for p in model.parameters() if p.requires_grad]


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def identity(cfg, method, split):
    path = mask_path(cfg, method)
    mask_hash = fingerprint(read_json(path)) if path is not None else None
    return fingerprint({'config': without(cfg, 'relearn', *HARNESS_KEYS), 'method': method,
                        'split': split, 'mask_hash': mask_hash})


def validate_model(cfg, model, split):
    actual = getattr(model.config, '_commit_hash', None)
    require(cfg, split.get('model_commit') == actual,
            'Base model revision changed since preparation. Pin revision and prepare a new run.')


def restore_checkpoint(cfg, model, optimizer, path, expected):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(cfg, payload['identity'] == expected,
            f'{path} was written under different settings, method or data, and is being reused as-is.')
    # Structural, never relaxed: load_state_dict refuses a checkpoint whose optimizer
    # scope or tensor shapes disagree with the model now being trained.
    optimizer.load_state_dict(payload['optimizer'])
    with torch.no_grad():
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if 'master' not in optimizer.state[parameter]:
                    raise ValueError('Missing optimizer master weight.')
                parameter.copy_(optimizer.state[parameter]['master'])
    return payload['step']


def weights_path(out, step):
    return out / 'weights' / f'step_{step:04d}.pt'


@torch.no_grad()
def level_check(model, tokenizer, split, cfg):
    """Full-set clean losses plus the forget-set gate, the quantities the levels test."""
    check = selection_loss(model, tokenizer, split, cfg)
    records = evaluate_rows(model, tokenizer, split['forget'], cfg)['records']
    check['forget_correct'] = sum(r['correct'] for r in records)
    check['forget_max_normalized'] = max(r['normalized_target_probability'] for r in records)
    return check


def levels_met(cfg, step, check):
    """Which levels this step satisfies. Each is saved only the first time it holds."""
    levels = cfg['train']['levels']
    return {'steps': step == levels['steps'],
            'loss': check['forget_loss'] < levels['loss'],
            'forget': check['forget_correct'] == 0
                      and check['forget_max_normalized'] < levels['forget_probability']}


@torch.no_grad()
def restore_weights(model, path):
    # mmap keeps a full-model weights file (~16 GB BF16) out of host RAM while loading.
    weights = torch.load(path, map_location='cpu', weights_only=True, mmap=True)['weights']
    named = dict(model.named_parameters())
    scope = {name for name, p in named.items() if p.requires_grad}
    # Structural, never relaxed: the saved scope must be exactly this arm's scope.
    if set(weights) != scope:
        raise ValueError(f'{path} holds a different trainable scope than this arm.')
    for name, value in weights.items():
        named[name].copy_(value)


def load_trained(cfg, method):
    split = load_split(cfg)
    model, tokenizer = load_model(cfg)
    validate_model(cfg, model, split)
    if method != 'base':
        out = Path(cfg['output']) / method
        setup_scope(model, cfg, method)  # restore_weights checks the saved scope against it
        require(cfg, read_json(out / 'levels.json')['identity'] == identity(cfg, method, split),
                f'{out} was trained under different settings and is being reused as-is.')
        step = level_step(cfg, method)
        if step is None:
            raise ValueError(f'{method} never reached level {cfg["level"]}; there is no model to load.')
        restore_weights(model, weights_path(out, step))
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def train(cfg, method, needed=LEVELS):
    """Train one arm until every level in `needed` is saved.

    `needed` short of all LEVELS stops early to save GPU time: the resume state is kept
    and complete.json is not written, so a later run without --levels resumes the
    same deterministic run and saves the remaining levels.
    """
    split = load_split(cfg)
    out = Path(cfg['output']) / method
    out.mkdir(parents=True, exist_ok=True)
    expected = identity(cfg, method, split)
    done_path = out / 'complete.json'
    if done_path.exists():
        require(cfg, read_json(done_path)['identity'] == expected,
                f'Completed run in {out} has different settings and is being reused rather than retrained; '
                'delete it or choose a new output directory to train it again.')
        print(f'{method}: already complete', flush=True)
        return
    levels_path, checks_path = out / 'levels.json', out / 'level_checks.json'
    if levels_path.exists() and set(needed) <= set(read_json(levels_path)['reached']):
        print(f'{method}: levels {list(needed)} already saved; resume state kept for the rest', flush=True)
        return
    model, tokenizer = load_model(cfg)
    validate_model(cfg, model, split)
    parameters = setup_scope(model, cfg, method)
    # Non-reentrant recomputation supports gradients through frozen early layers.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    optimizer = MasterSGD(parameters, cfg['train']['learning_rate'])
    # Two radii, two SAM objects: the forget term is always flattened in a SAM arm,
    # the retain term only when sam_rho_retain is positive.
    sam_forget = SAM(parameters, cfg['train']['sam_rho_forget']) if 'sam' in method else None
    sam_retain = (SAM(parameters, cfg['train']['sam_rho_retain'])
                  if sam_forget and cfg['train']['sam_rho_retain'] > 0 else None)
    checkpoint = out / 'checkpoint.pt'
    start = restore_checkpoint(cfg, model, optimizer, checkpoint, expected) if checkpoint.exists() else 0
    # Resume logs to the last committed checkpoint, avoiding duplicate steps.
    log_path = out / 'training.json'
    logs = [r for r in read_json(log_path) if r['step'] <= start] if log_path.exists() else []
    # Level checkpoints. A resume keeps only what was reached by the resumed step; the
    # deterministic replay re-finds anything later and rewrites its weights file.
    check_every = cfg['train']['levels']['check_every']
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    reached ={level: r for level, r in read_json(levels_path)['reached'].items()
               if r['step'] <= start} if levels_path.exists() else {}
    checks = [r for r in read_json(checks_path) if r['step'] <= start] if checks_path.exists() else []
    def save_levels():
        write_json(levels_path, {'identity': expected, 'reached': reached})
        write_json(checks_path, checks)
    trainable = sum(p.numel() for p in parameters)
    write_json(out / 'settings.json', {'identity': expected, 'method': method, 'trainable_parameters': trainable,
                                      'optimizer': 'SGD with FP32 CPU master weights', 'config': cfg})
    for step in range(start, cfg['train']['steps']):
        seed_all(cfg['seed'] + step)
        forget = step_batch(split['forget'], step, cfg['seed'],
                            cfg['train'].get('forget_batch_size', cfg['batch_size']))
        retain = step_batch(split['retain'], step, cfg['seed'] + 1000,
                            cfg['train'].get('retain_batch_size', cfg['batch_size']))
        optimizer.zero_grad(set_to_none=True)
        losses, radii = {}, {'forget': None, 'retain': None}
        if sam_forget:
            # Each objective's perturbation comes from that objective's gradient alone,
            # at its own radius. Between the two the finished forget gradient is parked
            # so the retain ascent direction is the retain gradient and nothing else;
            # without that the retain direction would be the sum of both terms.
            radii['forget'] = sam_step(model, tokenizer, forget, cfg, 'forget', sam_forget,
                                       optimizer, losses, step)
            if sam_retain:
                parked = take_gradients(parameters)
                try:
                    radii['retain'] = sam_step(model, tokenizer, retain, cfg, 'retain', sam_retain,
                                               optimizer, losses, step)
                finally:
                    put_gradients(parked)
            else:
                # sam_rho_retain: 0 -- the retain loss is evaluated at the unperturbed
                # parameters, which is exactly v1's behaviour.
                seed_all(cfg['seed'] + step + 7000)
                losses['retain_loss'] = loss_backward(model, tokenizer, retain, cfg, 'retain')
        else:
            losses['forget_loss'] = loss_backward(model, tokenizer, forget, cfg, 'forget')
            losses['retain_loss'] = loss_backward(model, tokenizer, retain, cfg, 'retain')
        norm = gradient_norm(parameters)
        clip_scale = min(1.0, cfg['train']['max_grad_norm'] / (norm.item() + 1e-12))
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_scale)
        lr = learning_rate(cfg, step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {'step': step + 1, **losses, 'gradient_norm': norm.item(),
                  'effective_sam_radius_forget': radii['forget'],
                  'effective_sam_radius_retain': radii['retain'], 'learning_rate': lr}
        logs.append(record)
        if (step + 1) % LOG_EVERY == 0 or step + 1 == cfg['train']['steps']:
            print(method, json.dumps(record), flush=True)
        finished = False
        if (step + 1) % check_every == 0 or step + 1 == cfg['train']['levels']['steps']:
            check = level_check(model, tokenizer, split, cfg)
            new = [level for level, met in levels_met(cfg, step + 1, check).items()
                   if met and level not in reached]
            if new:
                atomic_torch_save(weights_path(out, step + 1), {'identity': expected, 'step': step + 1,
                                  'weights': {name: p.detach() for name, p in named}})
                reached.update({level: {'step': step + 1, **check} for level in new})
            checks.append({'step': step + 1, **check, 'new_levels': new})
            print(method, 'levels', json.dumps(checks[-1]), flush=True)
            # Nothing left to save once every needed level is in hand.
            finished = set(needed) <= set(reached)
        if (step + 1) % cfg['train']['checkpoint_every'] == 0 or finished:
            atomic_torch_save(checkpoint, {'identity': expected, 'step': step + 1, 'optimizer': optimizer.state_dict()})
            write_json(log_path, logs)
            save_levels()
        if finished:
            break
    write_json(log_path, logs)
    save_levels()
    stopped = logs[-1]['step'] if logs else start
    if len(reached) < len(LEVELS) and stopped < cfg['train']['steps']:
        print(f'{method}: stopped at step {stopped} with levels {sorted(reached)}; resume state kept, '
              'run again without --levels to save the rest', flush=True)
        return
    write_json(done_path, {'identity': expected, 'step': logs[-1]['step'] if logs else start,
                           'trainable_parameters': trainable,
                           'level_steps': {level: reached.get(level, {}).get('step') for level in LEVELS}})
    # The resume state is FP32 masters (32 GB for a full-model arm) and nothing reads
    # it once the arm is complete; the level weights are what later stages load.
    checkpoint.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=ALL_METHODS, required=True)
    parser.add_argument('--levels', nargs='+', choices=LEVELS, default=list(LEVELS),
                        help='stop once these levels are saved (default: all); the rest can be resumed later')
    args = parser.parse_args()
    train(config(args.config), args.method, args.levels)
