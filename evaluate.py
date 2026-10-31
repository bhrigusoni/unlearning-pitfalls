"""Next-token factual metrics, vocabulary confidence and class-collapse diagnostics."""
import argparse
import math
from collections import Counter

import torch

from prepare_data import (LEVELS, arm_dir, batches, config, encode, level_step, load_split, next_logits,
                          stage_cfg, target_ids, write_json)


def summarize(records):
    if not records:
        raise ValueError('Cannot evaluate an empty split.')
    sports = ['baseball', 'basketball', 'football']
    counts = Counter(r['predicted'] for r in records)
    count = len(records)
    return {'count': count, 'accuracy': sum(r['correct'] for r in records) / count,
            'mean_target_probability': sum(r['target_probability'] for r in records) / count,
            'mean_target_loss': sum(r['target_loss'] for r in records) / count,
            # Headline forgetting metric: raw target probability and vocabulary top-1
            # both move under a uniform downshift of every sport logit, which removes
            # no knowledge. This one only moves if the closed-set ranking changes.
            'mean_normalized_target_probability':
                sum(r['normalized_target_probability'] for r in records) / count,
            'vocabulary_top1_accuracy': sum(r['vocabulary_correct'] for r in records) / count,
            'mean_sport_probability_mass': sum(sum(r['sport_probabilities'].values()) for r in records) / count,
            'prediction_counts': {s: counts[s] for s in sports},
            'largest_predicted_class_fraction': max(counts.values()) / count,
            'prediction_entropy': -sum((n / count) * math.log(n / count) for n in counts.values()),
            'per_sport_accuracy': {s: sum(r['correct'] for r in records if r['sport'] == s) /
                                  max(1, sum(r['sport'] == s for r in records)) for s in sports},
            'confusion_matrix': {s: {p: sum(r['sport'] == s and r['predicted'] == p for r in records)
                                    for p in sports} for s in sports}}


@torch.no_grad()
def evaluate_rows(model, tokenizer, rows, cfg):
    was_training = model.training
    model.eval()
    sports = ['baseball', 'basketball', 'football']
    ids = target_ids(tokenizer, sports)
    records = []
    try:
        # Forward-only, and each prompt is scored at its own last token, so the batch
        # size does not change any number; a larger one only makes per-step attack
        # evaluation affordable.
        for batch in batches(rows, cfg.get('eval_batch_size', cfg['batch_size'])):
            inputs, targets, _ = encode(batch, tokenizer, cfg)
            logits = next_logits(model, inputs)
            logp = logits.log_softmax(-1)
            # Closed-set probabilities come from a softmax over the three sport logits,
            # never from dividing vocabulary probabilities: NPO drives all three below
            # exp(-103), where they underflow to 0 in FP32 and the ratio is 0/0.
            closed = logits[:, [ids[sport] for sport in sports]].softmax(-1)
            for i, row in enumerate(batch):
                probabilities = {sport: logp[i, ids[sport]].exp().item() for sport in sports}
                normalized = dict(zip(sports, closed[i].tolist()))
                predicted = max(normalized, key=normalized.get)
                top = logits[i].argmax().item()
                records.append({'athlete': row['athlete'], 'sport': row['sport'], 'predicted': predicted,
                                'correct': predicted == row['sport'], 'target_probability': probabilities[row['sport']],
                                'target_loss': -logp[i, targets[i]].item(), 'sport_probabilities': probabilities,
                                'normalized_sport_probabilities': normalized,
                                'normalized_target_probability': normalized[row['sport']],
                                'vocabulary_top1_token': top, 'vocabulary_correct': top == targets[i].item()})
    finally:
        model.train(was_training)
    return {'summary': summarize(records), 'records': records}


def evaluate_model(model, tokenizer, split, cfg):
    return {name: evaluate_rows(model, tokenizer, split[name], cfg)
            for name in ('forget', 'retain', 'forget_heldout')}


def evaluate(cfg, method):
    from train import load_trained
    if level_step(cfg, method) is None:
        print(f'{method}: never reached level {cfg["level"]}; nothing to evaluate', flush=True)
        return None
    split = load_split(cfg)
    model, tokenizer = load_trained(cfg, method)
    report = evaluate_model(model, tokenizer, split, cfg)
    write_json(arm_dir(cfg, method) / 'evaluation.json', report)
    print({name: value['summary']['accuracy'] for name, value in report.items()}, flush=True)
    return report


if __name__ == '__main__':
    from prepare_data import ALL_METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=('base',) + ALL_METHODS, default='base')
    parser.add_argument('--level', choices=LEVELS, help='required for every method except base')
    args = parser.parse_args()
    evaluate(stage_cfg(config(args.config), args.level), args.method)
