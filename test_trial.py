"""Offline CPU checks using a tiny random Llama, never the real model weights."""
import copy
import csv
import math
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from evaluate import evaluate, evaluate_rows
from flu import (apply_mask, depth_bin, layer_features, localize, localize_random, mlps,
                 probe_accuracy, probe_folds, random_layers, select_all_layers, select_band, select_components)
from analyze import analyze, binomial_p, fisher, holm, mcnemar
from analyze_seeds import pooled
from plot import plot
from prepare_data import (ALL_METHODS, CONTROL_METHODS, MASKS, arm_dir, attack_dir, attack_tag, balanced_take,
                          encode, forget_folds, mask_tag, prepare, read_json, read_rows, stage_cfg, step_batch,
                          target_ids)
from relearn import relearn
from run_trial import aggregate
from sam import MasterSGD, SAM, put_gradients, take_gradients
from train import (METHODS, SPORTS, learning_rate, levels_met, load_trained, mask_path, npo,
                   selection_loss, train, unlikelihood, uniform, weights_path, wrong_sport)


def check_random_controls():
    """Random-layer controls on a 32-layer model: count, overlap, depth and reproducibility."""
    flu = [0, 1, 2, 3, 4, 13, 15]  # early-heavy, as in Table I
    # The original control is unchanged: same calls on the same rng as before.
    old = random.Random(42 + 4000)
    assert random_layers(flu, 32, False, random.Random(42 + 4000), 'random') == sorted(old.sample(range(32), 7))
    for seed in range(200):
        outside = random_layers(flu, 32, False, random.Random(seed), 'random_complement')
        assert len(outside) == 7 and not set(outside) & set(flu), 'complement draw must avoid FLU'
        band = random_layers([10, 11, 12], 32, True, random.Random(seed), 'random_complement')
        assert band == list(range(band[0], band[0] + 3)) and not set(band) & {10, 11, 12}
        depth = random_layers(flu, 32, False, random.Random(seed), 'random_depth')
        assert len(depth) == len(set(depth)) == 7 and not set(depth) & set(flu)
        # Quarter 0 holds 5 FLU layers and 3 free ones: all 3 taken, 2 spill into quarter 1;
        # quarter 1 also gets its own 2 (layers 13 and 15), so 4 there; none deeper.
        assert [sum(depth_bin(i, 32) == b for i in depth) for b in range(4)] == [3, 4, 0, 0], depth
    assert random_layers(flu, 32, False, random.Random(1), 'random_depth') == \
        random_layers(flu, 32, False, random.Random(1), 'random_depth'), 'draws must be deterministic'
    try:
        random_layers([0, 1], 3, False, random.Random(0), 'random_complement')
    except ValueError:
        pass
    else:
        raise AssertionError('too few layers outside FLU must raise')
    assert mask_tag('ul_rand_sam') == 'rand' and mask_tag('ul_randx') == 'randx'
    assert mask_tag('ul_randd_sam') == 'randd' and mask_tag('ul_flu') == 'flu' and mask_tag('ul_sam') is None
    assert set(METHODS) < set(ALL_METHODS) and all(mask_tag(m) in MASKS for m in CONTROL_METHODS)
    print('PASS: random-layer controls (original, outside FLU, depth-matched) and arm-name parsing')


def check_math():
    names = [f'Athlete {i}' for i in range(200)]
    for true_sport in SPORTS:
        assigned = [wrong_sport(name, true_sport, 42) for name in names]
        assert all(a != true_sport for a in assigned), 'must never relabel to the true sport'
        assert all(a in SPORTS for a in assigned)
        # Both wrong sports must actually get used, or the map is invertible.
        assert len(set(assigned)) == 2, 'relabelling collapsed onto one sport'
    assert wrong_sport('Theo Epstein', 'baseball', 42) == wrong_sport('Theo Epstein', 'baseball', 42),         'target must be stable across steps and resumes'
    assert any(wrong_sport(n, 'baseball', 42) != wrong_sport(n, 'baseball', 7) for n in names),         'assignment must depend on the seed'
    # Uniform: floors at log 3, flattens rather than shrinks, and a uniform shift of
    # all three candidate logits is a null direction.
    flat = torch.zeros(1, 3, requires_grad=True)
    assert abs(uniform(flat, [0, 1, 2]).item() - math.log(3)) < 1e-6, 'uniform floor must be log 3'
    skewed = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    uniform(skewed, [0, 1, 2]).backward()
    assert skewed.grad[0, 0] > 0 and skewed.grad[0, 2] < 0
    assert abs(skewed.grad[0].sum()) < 1e-6, 'uniform shift must have zero gradient'
    # NPO: (2/beta) log 2 when the model equals the reference, falls as P(true) drops
    # below the reference, and its gradient pushes the true logit down.
    logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    reference = logits.detach().log_softmax(-1)[:, 0]
    start = npo(logits, torch.tensor([0]), reference, 0.1)
    assert abs(start.item() - 2 / 0.1 * math.log(2)) < 1e-5, 'NPO must start at (2/beta) log 2'
    start.backward()
    assert logits.grad[0, 0] > 0, 'NPO must push the true-answer logit down'
    lower = torch.tensor([[-3.0, 0.0, -1.0]])
    assert npo(lower, torch.tensor([0]), reference, 0.1) < start, 'NPO must fall as P(true) drops'
    # Levels: steps only at exactly levels.steps; loss below its bar; forget needs both
    # zero correct and every fact under the probability bar.
    bars = {'train': {'levels': {'steps': 20, 'loss': 0.1, 'forget_probability': 0.05}}}
    met = levels_met(bars, 20, {'forget_loss': 0.05, 'forget_correct': 0, 'forget_max_normalized': 0.01})
    assert met == {'steps': True, 'loss': True, 'forget': True}
    met = levels_met(bars, 30, {'forget_loss': 0.2, 'forget_correct': 1, 'forget_max_normalized': 0.01})
    assert met == {'steps': False, 'loss': False, 'forget': False}
    assert not levels_met(bars, 30, {'forget_loss': 0.2, 'forget_correct': 0,
                                     'forget_max_normalized': 0.06})['forget']
    check_random_controls()
    legacy = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    unlikelihood(legacy, torch.tensor([0])).backward()
    assert legacy.grad[0, 0] > 0, 'unlikelihood must push the target logit down'
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])
    sam = SAM([p], 0.5)
    original = p.detach().clone()
    radius = sam.perturb()
    assert abs(radius - 0.5) < 1e-6 and torch.allclose(p, torch.tensor([3.3, 4.4]))
    try:
        raise RuntimeError('second-pass failure')
    except RuntimeError:
        sam.restore()
    assert torch.equal(p, original)
    bf_sam = torch.nn.Parameter(torch.tensor([0.01, 1.0], dtype=torch.bfloat16))
    bf_sam.grad = torch.ones_like(bf_sam)
    original_bf = bf_sam.detach().clone()
    bf_perturb = SAM([bf_sam], 0.1)
    bf_perturb.perturb()
    bf_perturb.restore()
    assert torch.equal(bf_sam, original_bf), 'BF16 SAM restore must be bit-exact'
    bf = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    opt = MasterSGD([bf], 0.001)
    for _ in range(10):
        bf.grad = torch.ones_like(bf)
        opt.step()
    assert bf.item() < 1, 'FP32 master must retain updates smaller than BF16 resolution'
    saved = copy.deepcopy(opt.state_dict())
    resumed = MasterSGD([bf], 9)
    resumed.load_state_dict(saved)
    assert resumed.state[bf]['master'].dtype == torch.float32
    assert resumed.state[bf]['master'].device.type == 'cpu'
    assert torch.allclose(resumed.state[bf]['master'], torch.tensor([0.99]))
    # v1's budget-capped band: the contiguous run with the largest rise, two layers fit.
    accuracies = [0.34, 0.90, 0.92, 0.35]
    selected, gains, used = select_band(accuracies, 20, [10, 10, 10, 10])
    assert selected == [1, 2] and used == 20
    assert gains[1] > gains[0] and gains[3] < 0
    assert select_band(accuracies, 1, [10, 10, 10, 10])[0] == [1], 'Budget below one MLP keeps exactly one'
    # Rank order would take layers 1 and 3 and skip 2; the band keeps the region whole.
    assert select_band([0.3, 0.8, 0.3, 0.8, 0.8], 30, [10] * 5)[0] == [1, 2, 3], 'FLU band must not skip layers'
    # v2's rule: every rising layer, no budget, and a falling layer in the middle is
    # dropped rather than bridged, which is what separates it from the band.
    rising = [0.34, 0.90, 0.60, 0.95]
    selected, gains, used = select_all_layers(rising, [10] * 4)
    assert selected == [0, 1, 3] and used == 30, 'all_layers must keep every rising layer and skip the fall'
    assert gains[2] < 0 and used == 10 * len(selected)
    assert select_all_layers(rising, [10] * 4, threshold=0.1)[0] == [1, 3], 'gain_threshold must prune small rises'
    assert select_all_layers([0.2, 0.1, 0.05], [10] * 3)[0] == [2], 'never return an empty edit scope'
    # Dispatch reads flu.selection, and only band reads the budget fraction.
    assert select_components({'flu': {'selection': 'all_layers', 'gain_threshold': 0.0}},
                             rising, [10] * 4, 40)[0] == [0, 1, 3]
    assert select_components({'flu': {'selection': 'band', 'localization_budget_fraction': 0.5}},
                             rising, [10] * 4, 40)[0] == [0, 1]
    # Parking a gradient: the next objective sees a clean slate, and the parked term
    # is added back rather than replacing what accumulated while it was away.
    parked_param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parked_param.grad = torch.tensor([0.5, -0.5])
    parked = take_gradients([parked_param])
    assert parked_param.grad is None, 'parking must clear .grad so the next SAM direction is unmixed'
    parked_param.grad = torch.tensor([1.0, 1.0])
    put_gradients(parked)
    assert torch.allclose(parked_param.grad, torch.tensor([1.5, 0.5])), 'parked gradient must be added back'
    put_gradients(take_gradients([torch.nn.Parameter(torch.zeros(2))]))  # No gradient: no-op, no error.
    # Shared probe folds: stratified, deterministic, every athlete validated exactly once.
    labels = torch.arange(3).repeat_interleave(20)
    folds = probe_folds(labels, 0)
    assert sorted(torch.cat(folds).tolist()) == list(range(60))
    assert all(len(labels[f].unique()) == 3 for f in folds), 'Each fold must contain every sport'
    assert all(torch.equal(a, b) for a, b in zip(folds, probe_folds(labels, 0))), 'Folds must be identical for every layer'
    # Learning-rate schedule: warmup ramps up, peak after warmup, floor on the last step,
    # and old configs without the new keys keep a constant rate.
    sched = {'train': {'learning_rate': 0.01, 'min_learning_rate': 0.001, 'warmup_steps': 4, 'steps': 20}}
    rates = [learning_rate(sched, s) for s in range(20)]
    assert abs(rates[0] - 0.0025) < 1e-12 and abs(rates[3] - 0.01) < 1e-12 and abs(rates[4] - 0.01) < 1e-12, \
        'warmup must reach the peak'
    assert all(a >= b for a, b in zip(rates[4:], rates[5:])), 'decay must be monotone'
    assert abs(rates[-1] - 0.001) < 1e-12, 'last step must reach the floor'
    constant = {'train': {'learning_rate': 0.01, 'steps': 20}}
    assert all(learning_rate(constant, s) == 0.01 for s in range(20)), 'no schedule keys must mean constant lr'
    # Attack folds: a sport-stratified partition, identical for every arm.
    facts = [{'athlete': f'a{i}', 'sport': SPORTS[i % 3]} for i in range(16)]
    folds = forget_folds(facts, 2, 7)
    assert sorted(r['athlete'] for fold in folds for r in fold) == sorted(r['athlete'] for r in facts)
    assert [len(f) for f in folds] == [8, 8] and all(len({r['sport'] for r in f}) == 3 for f in folds)
    assert forget_folds(facts, 2, 7) == folds, 'folds must be deterministic in the seed'
    # Default attacks keep their v1 folder names; only new settings add suffixes.
    base_attack = {'relearn': {'learning_rate': 1e-4, 'rank': 8, 'alpha': 16, 'steps': 100}}
    assert attack_tag(base_attack) == 'lr0.0001_r8_a16_s100'
    base_attack['relearn'].update(data='forget_split', folds=2, targets='mlp')
    assert attack_tag(base_attack) == 'lr0.0001_r8_a16_s100_split2_mlp'
    # Exact tests, against numbers reported earlier for the same counts.
    assert abs(binomial_p(11, 16) - 0.2101) < 1e-4, 'conditional runner-up null: 11/16 at 1/2'
    assert abs(fisher(0, 16, 10, 16) - 0.000248) < 1e-6, 'Fisher 0/16 vs 10/16 (scipy: 0.000248)'
    assert fisher(5, 16, 5, 16) == 1.0
    assert mcnemar({'a': False, 'b': False}, {'a': True, 'b': False}) == {'only_left': 0, 'only_right': 1, 'p': 1.0}
    flips = {f'x{i}': i < 5 for i in range(16)}
    assert abs(mcnemar({a: False for a in flips}, flips)['p'] - 0.0625) < 1e-12, 'McNemar 0:5'
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    print('PASS: UL sign/stability, NPO start/sign, level rules, SAM radius/restoration, FP32 optimizer, '
          'all-layer and banded MLP selection, gradient parking, lr schedule, attack folds and tags, '
          'exact binomial/McNemar/Fisher/Holm')


def fixture(root):
    sports = ['baseball', 'basketball', 'football']
    rows = [{'athlete': f'player{i}', 'sport': sports[i % 3],
             'prompt': f'Fact: Tiger Woods plays the sport of golf\nFact: player{i} plays the sport of'} for i in range(60)]
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2}
    words = ['Fact', ':', 'Tiger', 'Woods', 'plays', 'the', 'sport', 'of', 'golf'] + sports + [r['athlete'] for r in rows]
    for word in words:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=24,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=128, pad_token_id=1, eos_token_id=2))
    source = root / 'tiny_model'
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    dataset = root / 'sports.csv'
    with dataset.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [rows[0]])
    cfg = {'model': str(source), 'revision': 'main', 'device': 'cpu', 'dtype': 'float32',
           'seed': 42, 'dataset': str(dataset), 'output': str(root / 'results'), 'batch_size': 4,
           'eval_batch_size': 16, 'max_length': 128,
           'strict_identity': False,
           'splits': {'forget': 16, 'retain': 12, 'forget_heldout': 12, 'minimum_probability': 0.5},
           'flu': {'selection': 'all_layers', 'gain_threshold': 0.0, 'localization_budget_fraction': 0.15},
           'train': {'forget_objective': 'relabel',
                     'steps': 2, 'learning_rate': 0.01, 'min_learning_rate': 0.001, 'warmup_steps': 0,
                     'forget_weight': 1.0, 'retain_weight': 1.0,
                     'sam_rho_forget': 0.05, 'sam_rho_retain': 0.02,
                     'max_grad_norm': 1.0, 'checkpoint_every': 1, 'select_batch_size': 8,
                     'forget_batch_size': 16, 'retain_batch_size': 8, 'npo_beta': 0.1,
                     # loss is reached at the first check, steps at step 2, and a random
                     # model never forgets, so the forget level exercises "never reached".
                     'levels': {'check_every': 1, 'steps': 2, 'loss': 100.0, 'forget_probability': 0.05}},
           'relearn': {'a': {'steps': 2, 'evaluate_every': 1, 'checkpoint_every': 1, 'learning_rate': 0.01,
                             'rank': 2, 'alpha': 4}}}
    return cfg, rows, model, tokenizer


def run_sam_sequence(cfg, directory, fake, name, rho_retain):
    """Record every objective pass of one SAM step at the stated retain radius."""
    import train as train_module
    cfg = copy.deepcopy(cfg)
    cfg['output'] = str(Path(directory) / name)
    cfg['train']['steps'] = 1
    cfg['train']['levels']['steps'] = 1
    cfg['train']['sam_rho_retain'] = rho_retain
    with patch('evaluate.evaluate_rows', return_value=fake):
        prepare(cfg)
    localize(cfg)
    calls, actual = [], train_module.loss_backward
    def recording(model, tokenizer, rows, cfg_, label):
        weight = next(p for p in model.parameters() if p.requires_grad).detach().clone()
        grads_before = [p.grad.detach().clone() if p.grad is not None else None
                        for p in model.parameters() if p.requires_grad]
        value = actual(model, tokenizer, rows, cfg_, label)
        calls.append({'label': label, 'weight': weight, 'grads_before': grads_before})
        return value
    with patch.object(train_module, 'loss_backward', recording):
        train_module.train(cfg, 'ul_sam')
    return cfg, calls


def check_sam_sequence(cfg, directory, fake):
    """Each objective's delta comes from that objective's own gradient, at its own radius."""
    single, calls = run_sam_sequence(cfg, directory, fake, 'sam_forget_only', 0)
    # sam_rho_retain: 0 is v1's sequence exactly -- retain evaluated at theta.
    assert [c['label'] for c in calls] == ['forget', 'forget', 'retain'], 'Wrong SAM objective order'
    assert all(g is None for g in calls[0]['grads_before']), 'Delta must be built from a clean forget gradient'
    assert all(g is None for g in calls[1]['grads_before']), 'Perturbed pass must start from zeroed gradients'
    assert not torch.equal(calls[0]['weight'], calls[1]['weight']), 'Second forget pass must run at theta + delta'
    assert torch.equal(calls[0]['weight'], calls[2]['weight']), 'Retain loss must be evaluated at unperturbed theta'
    assert any(g is not None for g in calls[2]['grads_before']), 'Retain gradient must accumulate onto the SAM forget gradient'
    log = read_json(Path(single['output']) / 'ul_sam' / 'training.json')[0]
    assert log['effective_sam_radius_retain'] is None and log['effective_sam_radius_forget'] > 0
    assert 'perturbed_retain_loss' not in log

    dual, calls = run_sam_sequence(cfg, directory, fake, 'sam_dual', 0.02)
    assert [c['label'] for c in calls] == ['forget', 'forget', 'retain', 'retain'], 'Wrong dual-SAM objective order'
    # The point of parking: the retain ascent direction is the retain gradient alone,
    # not the retain gradient plus the forget term already sitting in .grad.
    assert all(g is None for g in calls[2]['grads_before']), 'Retain delta must be built from a clean retain gradient'
    assert all(g is None for g in calls[3]['grads_before']), 'Perturbed retain pass must start from zeroed gradients'
    assert torch.equal(calls[0]['weight'], calls[2]['weight']), 'Both clean passes must run at the same theta'
    assert not torch.equal(calls[2]['weight'], calls[3]['weight']), 'Second retain pass must run at theta + delta'
    assert not torch.equal(calls[1]['weight'], calls[3]['weight']), 'The two deltas must differ'
    log = read_json(Path(dual['output']) / 'ul_sam' / 'training.json')[0]
    radii = (log['effective_sam_radius_forget'], log['effective_sam_radius_retain'])
    assert all(r > 0 for r in radii) and abs(radii[0] / radii[1] - 0.05 / 0.02) < 0.05, \
        f'Each objective must be perturbed at its own configured radius, got {radii}'
    assert 'perturbed_retain_loss' in log and 'perturbed_forget_loss' in log
    # The retain radius must reach the update; if the parked/restored gradient dance
    # dropped the retain pass, the two runs would land on identical weights.
    a, _ = load_trained(stage_cfg(dual, 'steps'), 'ul_sam')
    b, _ = load_trained(stage_cfg(single, 'steps'), 'ul_sam')
    assert not all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())), \
        'A retain radius must change the update; otherwise the retain SAM pass is being discarded'
    print('PASS: per-objective SAM deltas at separate radii; forget term parked and restored intact')


def check_pipeline():
    with tempfile.TemporaryDirectory() as directory:
        cfg, rows, model, tokenizer = fixture(Path(directory))
        assert len(read_rows(cfg['dataset'])) == 60
        chosen = balanced_take(rows, 16, 42)
        assert sorted(sum(r['sport'] == s for r in chosen) for s in ['baseball', 'basketball', 'football']) == [5, 5, 6]
        assert balanced_take(rows, 16, 42) == chosen
        assert len({r['athlete'] for r in step_batch(rows, 0, 42)}) == 4
        assert step_batch(rows, 0, 42) != step_batch(rows, 1, 42)
        # Random tiny model has no factual knowledge. Stub ONLY preparation's
        # known-fact gate; all localization/training/evaluation below is real.
        fake = {'records': [{'correct': True, 'target_probability': 0.8, 'target_loss': 0.22} for _ in rows]}
        with patch('evaluate.evaluate_rows', return_value=fake):
            split = prepare(cfg)
        names = [r['athlete'] for key in ('forget', 'retain', 'forget_heldout') for r in split[key]]
        assert len(names) == len(set(names)) == 40
        assert all(r['reference_logprob'] == -0.22 for r in split['forget']), 'NPO reference must be stored'
        batch = split['forget'][:4]
        inputs, targets, _ = encode(batch, tokenizer, cfg)
        assert inputs['input_ids'].shape[0] == 4
        model.requires_grad_(False)
        features = layer_features(model, inputs)
        assert len(features) == 2 and all(f.shape == (4, 16) and torch.isfinite(f).all() for f in features)
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        # Evaluation batch size must not change any number.
        small, large = (evaluate_rows(model, tokenizer, split['retain'], {**cfg, 'eval_batch_size': n})
                        for n in (4, 12))
        assert all(abs(a['target_probability'] - b['target_probability']) < 1e-5
                   for a, b in zip(small['records'], large['records']))
        # NPO pushes every sport logit far below the rest of the vocabulary. Vocabulary
        # probabilities then underflow to 0 (the old 0/0 crash); closed-set ones must not.
        ids = target_ids(tokenizer, SPORTS)
        def buried(model_, inputs_):
            logits = torch.zeros(inputs_['input_ids'].shape[0], model.config.vocab_size)
            for k, sport in enumerate(SPORTS):
                logits[:, ids[sport]] = -300.0 - k
            return logits
        with patch('evaluate.next_logits', buried):
            records = evaluate_rows(model, tokenizer, split['forget'][:4], cfg)['records']
        assert all(sum(r['sport_probabilities'].values()) == 0 for r in records), 'must reproduce the underflow'
        assert all(abs(sum(r['normalized_sport_probabilities'].values()) - 1) < 1e-6
                   and r['predicted'] == SPORTS[0] for r in records), 'closed-set probabilities must survive it'
        separable = torch.cat([torch.zeros(20, 4), torch.ones(20, 4), 2 * torch.ones(20, 4)])
        sports = torch.arange(3).repeat_interleave(20)
        assert probe_accuracy(separable, sports, probe_folds(sports, 0), 0, steps=100) > 0.9
        mask = localize(cfg)
        assert mask['localization_unit'] == 'mlp_component', 'FLU must localize MLP components, not neurons'
        assert mask['selected_mlp_layers'] and set(mask['selected_mlp_layers']) <= {0, 1}
        rand = localize_random(cfg)
        assert rand['parameter_count'] == mask['parameter_count'], 'random arm must match the FLU budget exactly'
        before = {n: p.clone() for n, p in model.named_parameters()}
        attack_a = stage_cfg(cfg, attack='a')
        evaluate(cfg, 'base')
        relearn(attack_a, 'base')
        for method in METHODS:
            if method == 'ul':
                calls = 0
                actual = MasterSGD.step
                def interrupted(opt, closure=None):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError('simulated interruption')
                    return actual(opt, closure)
                try:
                    with patch.object(MasterSGD, 'step', interrupted):
                        train(cfg, method)
                except RuntimeError as exc:
                    assert str(exc) == 'simulated interruption'
            train(cfg, method)
            out = Path(cfg['output']) / method
            reached = read_json(out / 'levels.json')['reached']
            assert reached['loss']['step'] == 1 and reached['steps']['step'] == 2 and 'forget' not in reached
            assert read_json(out / 'complete.json')['level_steps'] == {'steps': 2, 'loss': 1, 'forget': None}
            assert not (out / 'checkpoint.pt').exists(), 'resume state must be deleted once complete'
            assert [r['step'] for r in read_json(out / 'level_checks.json')] == [1, 2]
            trained, tok = load_trained(stage_cfg(cfg, 'steps'), method)
            early, _ = load_trained(stage_cfg(cfg, 'loss'), method)
            assert not all(torch.equal(a, b) for a, b in zip(trained.parameters(), early.parameters())), \
                'each level must load the weights of the step it was reached'
            if method == 'ul':
                # Resumed full-model result equals an uninterrupted run, at both levels.
                other = copy.deepcopy(cfg)
                other['output'] = str(Path(directory) / 'uninterrupted')
                with patch('evaluate.evaluate_rows', return_value=fake):
                    prepare(other)
                train(other, method)
                for level, resumed in (('steps', trained), ('loss', early)):
                    clean, _ = load_trained(stage_cfg(other, level), method)
                    assert all(torch.equal(a, b) for a, b in zip(resumed.parameters(), clean.parameters()))
            saved = torch.load(weights_path(out, 2), weights_only=True)['weights']
            assert saved and all(torch.equal(p, saved[n]) for n, p in trained.named_parameters() if n in saved)
            scope = mask_path(cfg, method)
            if scope is not None:
                editable = tuple(f'model.layers.{i}.mlp.' for i in read_json(scope)['selected_mlp_layers'])
                changed = False
                for name, p in trained.named_parameters():
                    if name.startswith(editable):
                        changed |= not torch.equal(p, before[name])
                        continue
                    assert torch.equal(p, before[name]), f'Frozen weight changed: {name}'
                assert changed, 'Selected MLP parameters were never updated'
            for level in ('steps', 'loss'):
                evaluate(stage_cfg(cfg, level), method)
                relearn(stage_cfg(cfg, level, 'a'), method)
                curve = read_json(attack_dir(stage_cfg(cfg, level, 'a'), method) / 'curve.json')
                assert [r['step'] for r in curve] == [0, 1, 2]
                # Adding zero-initialized LoRA must preserve pre-attack behavior.
                report = read_json(arm_dir(stage_cfg(cfg, level), method) / 'evaluation.json')
                assert abs(curve[0]['forget']['mean_target_probability']
                           - report['forget']['summary']['mean_target_probability']) < 1e-6
            # A level the arm never reached is skipped, not crashed on.
            assert evaluate(stage_cfg(cfg, 'forget'), method) is None
            relearn(stage_cfg(cfg, 'forget', 'a'), method)
            assert not arm_dir(stage_cfg(cfg, 'forget'), method).exists()
            train(cfg, method)  # Idempotent completed-stage validation.
        a_dir = attack_dir(stage_cfg(cfg, 'steps', 'a'), 'ul')
        assert not (a_dir / 'checkpoint.pt').exists(), 'attack resume state must be deleted once complete'
        assert (a_dir / 'adapter' / 'adapter_config.json').exists()
        assert not list((a_dir / 'adapter').glob('*.safetensors')), 'the adapter weights must not be kept'
        # NPO trains end to end (its math is checked in check_math).
        npo_cfg = copy.deepcopy(cfg)
        npo_cfg['output'] = str(Path(directory) / 'npo')
        npo_cfg['train']['forget_objective'] = 'npo'
        with patch('evaluate.evaluate_rows', return_value=fake):
            prepare(npo_cfg)
        train(npo_cfg, 'ul')
        assert all(math.isfinite(r['forget_loss']) for r in read_json(Path(npo_cfg['output']) / 'ul' / 'training.json'))
        # Once every level is saved, training stops early and records where it stopped.
        import train as train_module
        stop_cfg = copy.deepcopy(cfg)
        stop_cfg['output'] = str(Path(directory) / 'early_stop')
        stop_cfg['train']['steps'] = 4
        with patch('evaluate.evaluate_rows', return_value=fake):
            prepare(stop_cfg)
        always = lambda cfg_, step, check: {'steps': step == 2, 'loss': True, 'forget': True}
        with patch.object(train_module, 'levels_met', always):
            train(stop_cfg, 'ul')
        stopped = read_json(Path(stop_cfg['output']) / 'ul' / 'complete.json')
        assert stopped['step'] == 2 and stopped['level_steps'] == {'steps': 2, 'loss': 1, 'forget': 1}
        # --levels loss stops once that level is saved, keeps the resume state and writes
        # no complete.json; a later full run resumes and matches an uninterrupted one.
        part_cfg = copy.deepcopy(stop_cfg)
        part_cfg['output'] = str(Path(directory) / 'partial')
        with patch('evaluate.evaluate_rows', return_value=fake):
            prepare(part_cfg)
        part = Path(part_cfg['output']) / 'ul'
        with patch.object(train_module, 'levels_met', always):
            train(part_cfg, 'ul', ('loss',))
            assert not (part / 'complete.json').exists() and (part / 'checkpoint.pt').exists()
            assert set(read_json(part / 'levels.json')['reached']) == {'loss', 'forget'}
            train(part_cfg, 'ul', ('loss',))  # Already saved: returns without training.
            assert [r['step'] for r in read_json(part / 'training.json')] == [1]
            train(part_cfg, 'ul')
        assert read_json(part / 'complete.json')['level_steps'] == stopped['level_steps']
        assert not (part / 'checkpoint.pt').exists()
        for level in ('steps', 'loss'):
            a, _ = load_trained(stage_cfg(part_cfg, level), 'ul')
            b, _ = load_trained(stage_cfg(stop_cfg, level), 'ul')
            assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())), \
                'a resumed partial run must equal an uninterrupted one'
        check_sam_sequence(cfg, directory, fake)
        for level in ('steps', 'loss', 'forget'):
            aggregate(stage_cfg(cfg, level, 'a'))
        rows_steps = read_json(Path(cfg['output']) / 'comparison_steps_a.json')
        assert len(rows_steps) == 1 + len(METHODS) and rows_steps[1]['level_step'] == 2
        assert len(read_json(Path(cfg['output']) / 'comparison_forget_a.json')) == 1, 'only base at forget'
        # A second named attack reuses the same unlearned checkpoints in its own folder.
        tuned = copy.deepcopy(cfg)
        tuned['relearn']['b'] = {**cfg['relearn']['a'], 'learning_rate': 0.02}
        first = attack_dir(stage_cfg(cfg, 'steps', 'a'), 'ul_flu')
        second = attack_dir(stage_cfg(tuned, 'steps', 'b'), 'ul_flu')
        assert first != second
        preserved = read_json(first / 'curve.json')
        relearn(stage_cfg(tuned, 'steps', 'b'), 'ul_flu')
        assert [r['step'] for r in read_json(second / 'curve.json')] == [0, 1, 2]
        assert read_json(first / 'curve.json') == preserved, 'the second attack must not disturb the first'
        relearn(stage_cfg(tuned, 'steps', 'b'), 'ul_flu')  # Idempotent.
        # strict_identity: false warns and reuses a completed arm whose train settings
        # changed; strict_identity: true still refuses it.
        tuned['train']['learning_rate'] *= 2
        train(tuned, 'ul_flu')
        strict = copy.deepcopy(tuned)
        strict['strict_identity'] = True
        try:
            train(strict, 'ul_flu')
        except ValueError as exc:
            assert 'different settings' in str(exc), exc
        else:
            raise AssertionError('strict_identity: true must refuse a completed arm built differently')
        # Direct attack: train on one fold of the forget facts, score only the other,
        # with LoRA reaching the MLPs; merged files keep the schema aggregate/analyze read.
        cfg['relearn']['direct'] = {**cfg['relearn']['a'], 'data': 'forget_split', 'folds': 2, 'targets': 'all'}
        direct = stage_cfg(cfg, 'steps', 'direct')
        assert attack_tag(direct) == 'direct'
        for method in ('base',) + METHODS:
            relearn(direct, method)
        where = attack_dir(direct, 'ul_flu')
        folds = read_json(where / 'complete.json')['folds']
        assert sorted(sum(folds, [])) == sorted(r['athlete'] for r in split['forget'])
        merged = read_json(where / 'evaluation_0002.json')
        assert merged['forget']['summary']['count'] == 16 == merged['forget_trained']['summary']['count']
        for k, fold in enumerate(folds):
            assert not set(read_json(where / f'fold{k}' / 'complete.json')['trained_on']) & set(fold), \
                'a fold must never be trained on by the attack that scores it'
            own = {r['athlete']: r for r in read_json(where / f'fold{k}' / 'evaluation_0002.json')['forget']['records']}
            assert all(r == own[r['athlete']] for r in merged['forget']['records'] if r['athlete'] in fold), \
                'each fact must be scored by the attack that never saw it'
        targets = set(read_json(where / 'fold0' / 'adapter' / 'adapter_config.json')['target_modules'])
        assert {'q_proj', 'gate_proj', 'up_proj', 'down_proj'} <= targets
        assert read_json(where / 'curve.json')[-1]['train_loss'] is not None
        aggregate(direct)
        analyze(direct)
        assert read_json(Path(cfg['output']) / 'analysis_steps_direct.json')['readout'] ==             {'skipped': 'no evaluation at step 10'}, 'a 2-step attack has no step-10 readout'
        with patch('analyze.READOUT_STEP', 2):
            analyze(direct)
        report = read_json(Path(cfg['output']) / 'analysis_steps_direct.json')
        assert report['level'] == 'steps' and len(report['runner_up']) == len(METHODS)
        assert report['readout']['tests'] and 'attack_valid' in report['readout']
        assert all((t['mcnemar_p_holm'] is None) == (t['step'] == 0) for t in report['readout']['tests'])
        assert (Path(cfg['output']) / 'readout_steps_direct.csv').exists()
        # Pooling the same run as two "seeds" must exactly double every count.
        with patch('analyze.READOUT_STEP', 2), patch('analyze.VALID_ACCURACY', 0.0):
            single = {(t['left'], t['right']): t for t in report['readout']['tests'] if t['step'] == 2}
            pooled_tests = pooled([direct, {**direct, 'seed': direct['seed'] + 10000}])['tests']
        assert pooled_tests and len(pooled_tests) == len(single)
        for row in pooled_tests:
            one = single[(row['left'], row['right'])]
            assert row['seeds_used'] == 2 and row['n'] == 2 * one['n']
            assert (row['only_left'], row['only_right']) == (2 * one['mcnemar_only_left'], 2 * one['mcnemar_only_right'])
            assert row['seeds_left_fewer'] + row['seeds_right_fewer'] + row['seeds_tied'] == 2
        analyze(stage_cfg(cfg, 'forget', 'a'))  # No arm reached it: must not crash.
        assert plot(direct).exists() and plot(stage_cfg(cfg, 'forget', 'a')) is None
        assert read_json(first / 'curve.json') == preserved, 'the direct attack must not disturb earlier attacks'
        print(f'PASS: real tiny-Llama MLP probing, FLU + random localization, {len(METHODS)} training '
              f'paths, resume, three unlearning levels incl. an unreached one, early stop, NPO training, '
              f'evaluation, LoRA attacks per level, a second named attack, a fold-split MLP attack on '
              f'every arm, aggregation, analysis and plots')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(42)
    check_math()
    check_pipeline()
    print('ALL CHECKS PASSED (CPU; no downloaded model or research-result claim)')
