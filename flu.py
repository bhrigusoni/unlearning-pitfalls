"""Paper-A-style FLU localization at the MLP-component level.

The localization unit is a whole transformer MLP, never an individual neuron
and never a down_proj column. A linear probe is fitted on each layer's MLP
output over the closed sport answer set, every layer scored on the same
cross-validation folds. Every layer whose probe accuracy rises is
selected (flu.selection: all_layers), so the FLU region is whatever this model's
probe reports rather than a band cropped to a fixed parameter budget; v1's
budgeted contiguous band is still available as flu.selection: band.
"""
import argparse
import random
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from prepare_data import (MASKS, batches, config, encode, fingerprint, load_model,
                          load_split, read_json, require, write_json)

SPORTS = ('baseball', 'basketball', 'football')


def mlps(model):
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise ValueError('Expected a Llama-compatible model.model.layers architecture.')
    result = [layer.mlp for layer in model.model.layers]
    if not result:
        raise ValueError('Model contains no MLP components.')
    return result


@contextmanager
def hooks(handles):
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def layer_features(model, inputs):
    """MLP output at the final prompt position, one vector per layer."""
    modules = mlps(model)
    captured = {}
    def capture(i):
        def hook(module, args, output):
            captured[i] = output.detach().float()
        return hook
    handles = [m.register_forward_hook(capture(i)) for i, m in enumerate(modules)]
    with hooks(handles):
        model(**inputs, use_cache=False)
    positions = inputs['attention_mask'].sum(1) - 1
    index = torch.arange(len(positions), device=positions.device)
    return [captured[i][index, positions].cpu() for i in range(len(modules))]


def probe_folds(labels, seed, folds=5):
    """Stratified validation folds, built once and shared by every layer.

    Giving each layer its own random split would make accuracy[i] - accuracy[i-1]
    partly a difference between validation sets; with shared folds it is a
    difference between layers only. Each athlete is validated exactly once.
    """
    count = min(folds, len(labels))
    if count < 2:
        raise ValueError('Need at least two probe examples to cross-validate.')
    generator = torch.Generator().manual_seed(seed)
    assignment = torch.empty(len(labels), dtype=torch.long)
    dealt = 0
    for label in labels.unique():
        members = (labels == label).nonzero().flatten()
        members = members[torch.randperm(len(members), generator=generator)]
        # Continue the deal across classes so small classes don't all land in fold 0.
        assignment[members] = (torch.arange(len(members)) + dealt) % count
        dealt += len(members)
    return [(assignment == k).nonzero().flatten() for k in range(count)]


def probe_accuracy(features, labels, folds, seed, steps=300):
    """Cross-validated accuracy of a linear sport probe on one layer's features."""
    correct = 0
    for k, validate in enumerate(folds):
        train = torch.ones(len(labels), dtype=torch.bool)
        train[validate] = False
        mean, std = features[train].mean(0), features[train].std(0) + 1e-6
        x = (features - mean) / std
        torch.manual_seed(seed + k)
        probe = nn.Linear(features.shape[1], len(SPORTS))
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(probe(x[train]), labels[train]).backward()
            optimizer.step()
        with torch.no_grad():
            correct += (probe(x[validate]).argmax(1) == labels[validate]).sum().item()
    return correct / sum(len(v) for v in folds)


def localization_gains(accuracies):
    """Rise in probe accuracy at each layer, chance level before the first layer."""
    chance = 1.0 / len(SPORTS)
    return [a - (chance if i == 0 else accuracies[i - 1]) for i, a in enumerate(accuracies)]


def select_all_layers(accuracies, layer_parameters, threshold=0.0):
    """Every MLP the probe says carries sport information: gain > threshold.

    v2's default. The FLU region is read off the probe instead of being cropped to a
    parameter budget, so its size is a property of the model rather than a knob, and
    a region that is split across the depth is kept split rather than bridged. If no
    layer clears the threshold the single largest-gain layer is kept, because an
    empty edit scope is not a runnable arm.

    The consequence for the controls is deliberate: the Random arm matches this
    layer COUNT, so the two arms still edit the same number of parameters, but the
    parameter fraction is no longer fixed at 0.15 and must be read from the mask.
    """
    gains = localization_gains(accuracies)
    selected = [i for i, g in enumerate(gains) if g > threshold]
    if not selected:
        selected = [max(range(len(gains)), key=lambda i: (gains[i], -i))]
    return selected, gains, sum(layer_parameters[i] for i in selected)


def select_band(accuracies, budget_parameters, layer_parameters):
    """The contiguous band of MLP layers where sport information rises most.

    v1's rule, kept for selection: band. The band is as many consecutive layers as
    the budget allows (at least one), placed where its summed rise - accuracy at its
    last layer minus accuracy just before it - is largest; the earliest band wins
    ties. No layer inside the FLU region is skipped, and the budget keeps the
    comparison against Random/Causal-Tracing/All-MLPs fair.
    """
    gains = localization_gains(accuracies)
    if len(set(layer_parameters)) != 1:
        raise ValueError('MLP components differ in size; a contiguous band needs equal-sized layers.')
    width = max(1, min(len(gains), int(budget_parameters // layer_parameters[0])))
    start = max(range(len(gains) - width + 1), key=lambda s: (round(sum(gains[s:s + width]), 12), -s))
    return list(range(start, start + width)), gains, width * layer_parameters[0]


def select_components(cfg, accuracies, layer_parameters, total_parameters):
    """Dispatch to the configured selection rule."""
    settings = cfg['flu']
    if settings['selection'] == 'band':
        return select_band(accuracies, settings['localization_budget_fraction'] * total_parameters,
                           layer_parameters)
    return select_all_layers(accuracies, layer_parameters, settings['gain_threshold'])


def localize(cfg):
    split = load_split(cfg)
    mask_path = Path(cfg['output']) / 'mask.json'
    if mask_path.exists():
        result = read_json(mask_path)
        require(cfg, result['split_hash'] == fingerprint(split), 'Saved mask does not match the split.')
        # Not an identity fingerprint: a mask built under a different selection rule
        # answers a different question, and silently overwriting it would invalidate
        # arms already trained inside it.
        if result.get('configuration') != cfg['flu']:
            raise ValueError(f'{mask_path} was built with flu: {result.get("configuration")}, '
                             f'config.yaml now asks for {cfg["flu"]}. Delete the mask to re-localize, '
                             'or use a new output directory to keep both.')
        return result
    model, tokenizer = load_model(cfg)
    require(cfg, getattr(model.config, '_commit_hash', None) == split.get('model_commit'),
            'Base model revision changed since preparation.')
    model.requires_grad_(False)
    modules = mlps(model)
    # Probe on retain facts: they are disjoint from the forget set, so the
    # selected components are not fitted to the 16 examples being unlearned.
    rows = split['retain']
    collected = [[] for _ in modules]
    for batch in batches(rows, cfg['batch_size']):
        inputs, _, _ = encode(batch, tokenizer, cfg)
        for store, value in zip(collected, layer_features(model, inputs)):
            store.append(value)
    features = [torch.cat(store) for store in collected]
    labels = torch.tensor([SPORTS.index(r['sport']) for r in rows])
    folds = probe_folds(labels, cfg['seed'])
    accuracies = []
    for i, value in enumerate(features):
        accuracies.append(probe_accuracy(value, labels, folds, cfg['seed']))
        print(f'FLU probe layer {i}: accuracy={accuracies[-1]:.3f}', flush=True)
    layer_parameters = [sum(p.numel() for p in m.parameters()) for m in modules]
    total = sum(p.numel() for p in model.parameters())
    selected, gains, used = select_components(cfg, accuracies, layer_parameters, total)
    mode = cfg['flu']['selection']
    result = {'method': 'FLU', 'localization_unit': 'mlp_component',
              'split_hash': fingerprint(split), 'model_commit': getattr(model.config, '_commit_hash', None),
              'selected_mlp_layers': selected, 'probe_accuracy': accuracies, 'localization_scores': gains,
              'parameter_count': used, 'parameter_fraction': used / total,
              'budget_fraction': cfg['flu']['localization_budget_fraction'] if mode == 'band' else None,
              'candidate_layers': len(modules), 'probe_examples': len(rows),
              'contiguous': selected == list(range(selected[0], selected[0] + len(selected))),
              'selection': ('contiguous band with the largest summed rise' if mode == 'band' else
                            f'every MLP whose probe accuracy rises by more than {cfg["flu"]["gain_threshold"]}'),
              'probe_validation': f'{len(folds)}-fold stratified cross-validation, same folds for every layer',
              'configuration': cfg['flu']}
    write_json(mask_path, result)
    print(f'FLU selected MLP layers {selected} ({used / total:.4f} of parameters).', flush=True)
    return result


# Random controls: (mask file key, rng seed offset, name in mask.json). 'random' is the
# original control and keeps its offset, so existing mask_random.json files reproduce.
RANDOM_POOLS = {'random': ('rand', 4000, 'Random'),
                'random_complement': ('randx', 4100, 'Random (outside FLU)'),
                'random_depth': ('randd', 4200, 'Random (outside FLU, depth-matched)')}
# Depth bins for random_depth: quarters of the model (8 layers each for Llama-3-8B).
DEPTH_BINS = 4


def depth_bin(layer, layers):
    return layer * DEPTH_BINS // layers


def random_layers(flu_layers, layers, contiguous, rng, pool):
    """Layer indices for one random control, the same count as FLU.

    random: any layers, so the draw may overlap FLU's. random_complement: only layers
    FLU did not select, so the contrast cannot be pulled toward a null by shared
    layers. random_depth: also outside FLU, with as many layers in each depth quarter
    as FLU has there; a quarter with too few free layers takes the rest from the
    nearest quarter. Contiguity is matched for random and random_complement only;
    depth matching takes priority over it.
    """
    count = len(flu_layers)
    if pool == 'random':
        if contiguous:
            start = rng.randrange(layers - count + 1)
            return list(range(start, start + count))
        return sorted(rng.sample(range(layers), count))
    flu = set(flu_layers)
    free = [i for i in range(layers) if i not in flu]
    if len(free) < count:
        raise ValueError(f'Only {len(free)} layers lie outside FLU; cannot draw {count}.')
    if pool == 'random_complement':
        if contiguous:
            starts = [s for s in range(layers - count + 1) if not flu & set(range(s, s + count))]
            if not starts:
                raise ValueError(f'No contiguous band of {count} layers avoids FLU {sorted(flu)}.')
            start = rng.choice(starts)
            return list(range(start, start + count))
        return sorted(rng.sample(free, count))
    if pool != 'random_depth':
        raise ValueError(f'Unknown random pool: {pool}')
    chosen = set()
    for b in range(DEPTH_BINS):
        short = sum(depth_bin(i, layers) == b for i in flu)
        # Own quarter first, then the nearest ones, until this quarter's count is met.
        for distance in range(DEPTH_BINS):
            if not short:
                break
            near = [i for i in free if i not in chosen and abs(depth_bin(i, layers) - b) == distance]
            take = rng.sample(near, min(short, len(near)))
            chosen.update(take)
            short -= len(take)
    return sorted(chosen)


def localize_random(cfg, pool='random'):
    """A random set of MLPs of FLU's size: the control for WHICH layers FLU chose.

    The count comes from mask.json, so the editable parameter count is identical and
    any difference between the two arms is attributable to which layers are edited
    rather than how many. The shape of the control follows the shape of the FLU mask:
    a contiguous FLU band gets a contiguous random band, so spacing is matched too,
    while a scattered all_layers mask gets a uniform random subset, which is the only
    control that can match a scattered mask's count without fabricating a band.
    `pool` picks the control (RANDOM_POOLS, random_layers); each has its own mask file.
    """
    tag, offset, name = RANDOM_POOLS[pool]
    split = load_split(cfg)
    path = Path(cfg['output']) / MASKS[tag]
    if path.exists():
        result = read_json(path)
        require(cfg, result['split_hash'] == fingerprint(split), 'Saved random mask does not match the split.')
        return result
    flu = read_json(Path(cfg['output']) / 'mask.json')
    require(cfg, flu['split_hash'] == fingerprint(split),
            'FLU mask does not match the split; re-run localization first.')
    count = len(flu['selected_mlp_layers'])
    if flu['parameter_count'] % count:
        raise ValueError('MLP components differ in size; matching the budget by layer count is invalid.')
    rng = random.Random(cfg['seed'] + offset)
    selected = random_layers(flu['selected_mlp_layers'], flu['candidate_layers'],
                             flu.get('contiguous', True), rng, pool)
    result = dict(flu, method=name, selected_mlp_layers=selected,
                  probe_accuracy=None, localization_scores=None,
                  contiguous=selected == list(range(selected[0], selected[0] + count)))
    if pool != 'random':
        layers = flu['candidate_layers']
        result.update(random_pool=pool, flu_layers=flu['selected_mlp_layers'],
                      overlap_with_flu=sorted(set(selected) & set(flu['selected_mlp_layers'])))
        if pool == 'random_depth':
            per_bin = lambda chosen: [sum(depth_bin(i, layers) == b for i in chosen) for b in range(DEPTH_BINS)]
            result.update(depth_bins=DEPTH_BINS, flu_layers_per_bin=per_bin(flu['selected_mlp_layers']),
                          random_layers_per_bin=per_bin(selected))
    write_json(path, result)
    print(f'{name} selected MLP layers {selected} ({result["parameter_fraction"]:.4f} of parameters).', flush=True)
    return result


def apply_mask(model, mask):
    """Make exactly the selected MLP components trainable; freeze everything else."""
    model.requires_grad_(False)
    modules = mlps(model)
    selected = mask['selected_mlp_layers']
    if not selected or min(selected) < 0 or max(selected) >= len(modules):
        raise ValueError('Mask layer outside model.')
    for i in selected:
        modules[i].requires_grad_(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--localization', choices=('flu',) + tuple(RANDOM_POOLS), default='flu')
    args = parser.parse_args()
    cfg = config(args.config)
    localize(cfg) if args.localization == 'flu' else localize_random(cfg, args.localization)
