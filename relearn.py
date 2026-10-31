"""LoRA relearning attack on an unlearned checkpoint.

relearn.data picks what the attack trains on:
- forget_heldout (v1 default): disjoint athletes the model still knows. Indirect
  recovery only; an arm with little collateral damage gives it almost no gradient,
  so a flat curve there is a weak attack, not a robust model.
- forget_split: the forget facts themselves, in relearn.folds folds. Each fold's
  attack trains on the other folds and is scored only on the fold it never saw, so
  recovery on that fold is knowledge coming back, not the attack teaching it.
  Folds are merged into one curve.json/evaluation_*.json per step, with the same
  schema as forget_heldout, so aggregation and analysis read either unchanged.

relearn.targets picks the LoRA modules: attention, mlp or all. The unlearning edits
of the FLU arms live in MLPs, so an attention-only attacker cannot reach them.

Each attack is one named entry under relearn in the config and writes to
<output>/<method>/<level>/relearning/<name>/ (base: <output>/base/relearning/<name>/).

Only the adapter config is kept, not the adapter: at rank 512 on every projection the
adapter is ~1.3B parameters (~5 GB), once per fold, arm, level and attack.
"""
import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from evaluate import evaluate_model, summarize
from prepare_data import (ALL_METHODS, ATTACK_TARGETS, LEVELS, attack_dir, attack_tag, config, encode, fingerprint,
                          forget_folds, level_step, load_split, next_logits, read_json, require, seed_all,
                          stage_cfg, step_batch, write_json)
from train import METHODS, atomic_torch_save, identity, load_trained, weights_path


def attack(cfg, method, split, rows, out, expected):
    """One LoRA attack trained on `rows`, evaluated on every split, written to `out`."""
    stale = f'Relearning settings or model changed since {out} was written; reusing it as-is.'
    if (out / 'complete.json').exists():
        require(cfg, read_json(out / 'complete.json')['identity'] == expected, stale)
        return
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_trained(cfg, method)
    seed_all(cfg['seed'] + 2000)
    settings = cfg['relearn']
    lora = LoraConfig(
        task_type='CAUSAL_LM', r=settings['rank'], lora_alpha=settings['alpha'], lora_dropout=0.0,
        target_modules=ATTACK_TARGETS[settings.get('targets', 'attention')], bias='none',
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=0.0)
    path = out / 'checkpoint.pt'
    start, curve = 0, []
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(cfg, payload['identity'] == expected, stale)
        set_peft_model_state_dict(model, payload['adapter'])
        optimizer.load_state_dict(payload['optimizer'])
        start, curve = payload['step'], payload['curve']
    def record(step, loss):
        report = evaluate_model(model, tokenizer, split, cfg)
        write_json(out / f'evaluation_{step:04d}.json', report)
        # train_loss near zero from the start means the attack has nothing to learn.
        curve.append({'step': step, 'train_loss': loss, **{name: val['summary'] for name, val in report.items()}})
        write_json(out / 'curve.json', curve)
    if not curve:
        record(0, None)
    for step in range(start, settings['steps']):
        model.train()
        seed_all(cfg['seed'] + 2000 + step)
        batch = step_batch(rows, step, cfg['seed'] + 3000)
        inputs, targets, _ = encode(batch, tokenizer, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(next_logits(model, inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite relearning loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % settings['evaluate_every'] == 0 or step + 1 == settings['steps']:
            print(f'{method} relearning {out.name} {step + 1}: loss={loss.item():.6f}', flush=True)
            record(step + 1, loss.item())
        # Resume state is decoupled from evaluation: at rank 512 it is ~16 GB (adapter
        # plus Adam), too big to write every step.
        if (step + 1) % settings.get('checkpoint_every', settings['evaluate_every']) == 0:
            atomic_torch_save(path, {'identity': expected, 'step': step + 1,
                                    'adapter': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                                    'optimizer': optimizer.state_dict(), 'curve': curve})
    lora.save_pretrained(out / 'adapter')
    write_json(out / 'complete.json', {'identity': expected, 'steps': settings['steps'],
                                     'trainable_parameters': sum(p.numel() for p in parameters),
                                     'trained_on': [r['athlete'] for r in rows]})
    path.unlink(missing_ok=True)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_folds(out, split, folds):
    """One evaluation per step in which every forget fact is scored by the fold that never saw it.

    'forget' is the unseen-fold records only (16 in total, one per fact); what each
    attack trained on goes to 'forget_trained' as a check that the attack learned.
    retain and forget_heldout are pooled over folds, so their counts are folds x size.
    """
    fold_of = {r['athlete']: k for k, fold in enumerate(folds) for r in fold}
    curves = [read_json(out / f'fold{k}' / 'curve.json') for k in range(len(folds))]
    merged = []
    for entries in zip(*curves):
        step = entries[0]['step']
        reports = [read_json(out / f'fold{k}' / f'evaluation_{step:04d}.json') for k in range(len(folds))]
        records = {'forget': [], 'forget_trained': [], 'retain': [], 'forget_heldout': []}
        for k, report in enumerate(reports):
            for record in report['forget']['records']:
                records['forget' if fold_of[record['athlete']] == k else 'forget_trained'].append(record)
            records['retain'] += report['retain']['records']
            records['forget_heldout'] += report['forget_heldout']['records']
        order = {r['athlete']: i for i, r in enumerate(split['forget'])}
        for name in ('forget', 'forget_trained'):
            records[name].sort(key=lambda r: order[r['athlete']])
        report = {name: {'summary': summarize(rows), 'records': rows} for name, rows in records.items()}
        write_json(out / f'evaluation_{step:04d}.json', report)
        losses = [e['train_loss'] for e in entries if e['train_loss'] is not None]
        merged.append({'step': step, 'train_loss': sum(losses) / len(losses) if losses else None,
                       **{name: value['summary'] for name, value in report.items()}})
    write_json(out / 'curve.json', merged)


def relearn(cfg, method):
    if level_step(cfg, method) is None:
        print(f'{method}: never reached level {cfg["level"]}; nothing to attack', flush=True)
        return
    split = load_split(cfg)
    out = attack_dir(cfg, method)
    # The attack depends on the model it starts from (arm AND level) plus its own
    # settings. Recorded for provenance; under strict_identity: false a mismatch warns.
    model_identity = (fingerprint({'arm': identity(cfg, method, split), 'level': cfg['level']})
                      if method != 'base' else fingerprint(split))
    expected = fingerprint({'model': model_identity, 'relearn': cfg['relearn']})
    if (out / 'complete.json').exists():
        require(cfg, read_json(out / 'complete.json')['identity'] == expected,
                f'Relearning settings or model changed since {out} was written; reusing it as-is.')
        print(f'{method} relearning {attack_tag(cfg)}: already complete', flush=True)
        return
    settings = cfg['relearn']
    data = settings.get('data', 'forget_heldout')
    if data == 'forget_heldout':
        attack(cfg, method, split, split['forget_heldout'], out, expected)
        description = 'LoRA on disjoint athletes; indirect recovery, not direct fact relearning'
        folds = None
    else:
        # Same folds for every arm, so arms are compared on identical attacks.
        folds = forget_folds(split['forget'], settings.get('folds', 2), cfg['seed'] + 4000)
        for k, held in enumerate(folds):
            unseen = {r['athlete'] for r in held}
            attack(cfg, method, split, [r for r in split['forget'] if r['athlete'] not in unseen],
                   out / f'fold{k}', expected)
        merge_folds(out, split, folds)
        description = (f'LoRA on {len(folds) - 1} of {len(folds)} folds of the forget facts; '
                       'each fact scored only by the attack that never trained on it')
    trained = Path(cfg['output']) / method
    start = 'base model' if method == 'base' else str(weights_path(trained, level_step(cfg, method)))
    write_json(out / 'complete.json', {
        'identity': expected, 'steps': settings['steps'], 'attack': description,
        'targets': ATTACK_TARGETS[settings.get('targets', 'attention')],
        'folds': [[r['athlete'] for r in fold] for fold in folds] if folds else None,
        'tag': attack_tag(cfg), 'settings': settings,
        'starting_method_checkpoint': start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=('base',) + ALL_METHODS, required=True)
    parser.add_argument('--level', choices=LEVELS, help='required for every method except base')
    parser.add_argument('--attack', required=True, help='name of one attack under relearn in the config')
    args = parser.parse_args()
    relearn(stage_cfg(config(args.config), args.level, args.attack), args.method)
