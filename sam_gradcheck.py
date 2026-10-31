"""One-step SAM gradient check: forget vs retain, with and without the perturbation.

Tests the Section VI explanation (the perturbed retain loss) instead of leaving it
untested. On the arm's own training batches and trainable scope, each objective is
measured the way train.sam_step builds it: the clean gradient at theta, then the
gradient at theta + delta, delta = rho * g/||g|| from that objective's own gradient
at its own radius. Nothing is trained and no weights are written.

Reported per batch and averaged: clean and perturbed loss, gradient norm (weighted by
forget_weight / retain_weight, i.e. as it enters the update), the cosine between clean
and perturbed gradient, the forget-retain cosine, and each objective's share of the
summed update, share_f = <g_f, g_f + g_r> / ||g_f + g_r||^2 (share_f + share_r = 1),
without SAM (clean) and with it (perturbed). If the perturbed retain gradient takes a
larger share of the clipped update, forgetting per step slows, which is the mechanism
to confirm or rule out.

  python sam_gradcheck.py --config config_sports16_relabel.yaml --method ul_flu_sam
  python sam_gradcheck.py --config config_sports16_relabel.yaml --method ul_flu_sam --level forget

Without --level the arm's scope is measured at the base weights (step 0); with it, at
the weights saved for that level. Output: <output>/<arm>/<level or start>/sam_gradcheck.json.
Memory: three gradient snapshots plus SAM's weight snapshot sit on the CPU in BF16.
For the FLU and random arms that is small; for ul_sam (full model) it is ~64 GB of RAM,
about what ul_sam training already needs.
"""
import argparse
import json
from pathlib import Path

import torch

from prepare_data import ALL_METHODS, LEVELS, arm_dir, config, level_step, load_model, load_split, seed_all, \
    stage_cfg, step_batch, write_json
from sam import SAM, gradient_norm, take_gradients
from train import load_trained, loss_backward, setup_scope, validate_model


@torch.no_grad()
def dot(params, snapshot):
    """<current .grad, snapshot> over the scope; snapshot from take_gradients."""
    if len(snapshot) != sum(p.grad is not None for p in params):
        raise ValueError('Gradient scope changed between passes.')
    return sum((p.grad.float() * s.to(p.device, torch.float32)).sum().item() for p, s in snapshot)


@torch.no_grad()
def snapshot_dot(a, b):
    """<a, b> for two take_gradients snapshots of the same scope."""
    return sum((x.float() * y.float()).sum().item() for (_, x), (_, y) in zip(a, b))


def snapshot(params):
    """Copy .grad to the CPU without removing it (SAM.perturb still needs it)."""
    return [(p, p.grad.detach().cpu().clone()) for p in params if p.grad is not None]


def zero(params):
    for p in params:
        p.grad = None


def measure(model, tokenizer, params, rows, cfg, label, rho, seed):
    """Clean and perturbed pass for one objective; returns stats and both gradients (CPU)."""
    zero(params)
    seed_all(seed)
    stats = {'loss': loss_backward(model, tokenizer, rows, cfg, label)}
    clean = snapshot(params)
    stats['norm'] = gradient_norm(params).item()
    if not rho:
        # sam_rho_retain: 0 -- training evaluates retain at theta, so clean is what it uses.
        zero(params)
        return stats, clean, clean
    sam = SAM(params, rho)
    try:
        stats['radius'] = sam.perturb()
        zero(params)
        seed_all(seed)
        stats['perturbed_loss'] = loss_backward(model, tokenizer, rows, cfg, label)
        stats['perturbed_norm'] = gradient_norm(params).item()
        stats['clean_perturbed_cosine'] = dot(params, clean) / (stats['norm'] * stats['perturbed_norm'])
    finally:
        sam.restore()
    return stats, clean, take_gradients(params)


def shares(norm_f, norm_r, cross):
    """Forget and retain share of the summed update, and its norm, from norms and <g_f, g_r>."""
    total = norm_f ** 2 + norm_r ** 2 + 2 * cross
    return {'forget_share': (norm_f ** 2 + cross) / total, 'retain_share': (norm_r ** 2 + cross) / total,
            'update_norm': total ** 0.5, 'forget_retain_cosine': cross / (norm_f * norm_r)}


def check(cfg, method, level, batches):
    split = load_split(cfg)
    if level:
        cfg = stage_cfg(cfg, level)
        model, tokenizer = load_trained(cfg, method)
        first = level_step(cfg, method)
    else:
        model, tokenizer = load_model(cfg)
        validate_model(cfg, model, split)
        first = 0
    params = setup_scope(model, cfg, method)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    train = cfg['train']
    rows = []
    # The batches this arm trains on next, with train.py's seeds.
    for step in range(first, first + batches):
        forget = step_batch(split['forget'], step, cfg['seed'], train.get('forget_batch_size', cfg['batch_size']))
        retain = step_batch(split['retain'], step, cfg['seed'] + 1000, train.get('retain_batch_size', cfg['batch_size']))
        f, f_clean, f_pert = measure(model, tokenizer, params, forget, cfg, 'forget',
                                     train['sam_rho_forget'], cfg['seed'] + step)
        r, r_clean, r_pert = measure(model, tokenizer, params, retain, cfg, 'retain',
                                     train['sam_rho_retain'], cfg['seed'] + step + 7000)
        row = {'step': step, 'forget': f, 'retain': r,
               'clean': shares(f['norm'], r['norm'], snapshot_dot(f_clean, r_clean)),
               'sam': shares(f.get('perturbed_norm', f['norm']), r.get('perturbed_norm', r['norm']),
                             snapshot_dot(f_pert, r_pert))}
        rows.append(row)
        print(method, level or 'start', json.dumps(row), flush=True)
        del f_clean, f_pert, r_clean, r_pert
    zero(params)
    def mean(*keys):
        values = []
        for row in rows:
            value = row
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        return sum(values) / len(values) if None not in values else None
    summary = {f'{part}_{key}': mean(part, key)
               for part in ('forget', 'retain')
               for key in ('loss', 'perturbed_loss', 'norm', 'perturbed_norm', 'clean_perturbed_cosine')}
    summary.update({f'{kind}_{key}': mean(kind, key) for kind in ('clean', 'sam')
                    for key in ('forget_share', 'retain_share', 'update_norm', 'forget_retain_cosine')})
    report = {'method': method, 'level': level or 'start', 'starting_step': first,
              'weights': 'base model' if not level else f'level {level} (training step {first})',
              'rho_forget': train['sam_rho_forget'], 'rho_retain': train['sam_rho_retain'],
              'loss_weights': {'forget': train['forget_weight'], 'retain': train['retain_weight']},
              'note': 'norms are of the weighted gradients, as they enter the update before clipping',
              'summary': summary, 'batches': rows}
    out = (arm_dir(cfg, method) if level else Path(cfg['output']) / method / 'start') / 'sam_gradcheck.json'
    write_json(out, report)
    print('Summary:', json.dumps(summary, indent=2))
    print('Wrote', out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=ALL_METHODS, default='ul_flu_sam')
    parser.add_argument('--level', choices=LEVELS, help='omit to measure at the base weights')
    parser.add_argument('--batches', type=int, default=4, help='training batches to average over')
    args = parser.parse_args()
    check(config(args.config), args.method, args.level, args.batches)
