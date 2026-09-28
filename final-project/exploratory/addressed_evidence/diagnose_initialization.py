"""TRAIN-only initial branch units and exact zero-coefficient gradients.

No optimizer steps, validation files, development results, or TEST access.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from categorical_experiment import masked_episode
from exploratory.addressed_evidence.model import AddressedEvidenceModel, build_neighbors
from study import digest, write_json


def joint_loss(logits, context, targets, mask):
    rows, columns = torch.where(mask)
    counts = torch.bincount(rows, minlength=len(context))
    assert torch.all(counts > 0) and not context[mask].any()
    eligible = context == 0
    eligible[:, 0] = False
    denominator = torch.logsumexp(logits.masked_fill(~eligible[..., None], -torch.inf).flatten(1), 1)
    losses = denominator[rows] - logits[rows, columns, targets[rows, columns]-1]
    return (torch.zeros_like(denominator).scatter_add(0, rows, losses)/counts).mean()


def gradient_stats(values):
    array = values.detach().double()
    nonzero = array != 0
    return {'l2_norm': float(array.norm()), 'rms': float(array.square().mean().sqrt()),
            'max_abs': float(array.abs().max()), 'nonzero_fraction': float(nonzero.double().mean()),
            'fraction_abs_below_adam_epsilon_1e8': float((array.abs() < 1e-8).double().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT/'runs/joint-field-convergence-v1/2026/training-categories.npz')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'initialization-diagnostic-v2.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Refuse overwrite of diagnostic')
    torch.set_num_threads(1)
    with np.load(args.train, allow_pickle=False) as archive:
        matrix = archive['ratings']
    neighbors = build_neighbors(matrix, 64)
    model = AddressedEvidenceModel(matrix.shape[1], k=64, variant='pair', neighbors=neighbors)
    context, mask = masked_episode(matrix, 2026, 1001)
    indices = np.random.default_rng(20269101).choice(len(matrix), 64, replace=False)
    count = model.k*(model.k-1)/2
    multipliers = {'pair_mean': 1., 'pair_rms_count': math.sqrt(count),
                   'pair_rms_count_and_fixed_vote_scale': math.sqrt(count)/model.config['pair_vote_scale']}
    accumulated = {name: {key: torch.zeros_like(value) for key, value in model.named_parameters()}
                   for name in multipliers}
    moments = {name: np.zeros(6) for name in multipliers}
    losses = []
    for offset in range(0, len(indices), 16):
        chosen = indices[offset:offset+16]
        c, target, selected = (torch.as_tensor(value[chosen], dtype=dtype) for value, dtype in
                              ((context, torch.long), (matrix, torch.long), (mask, torch.bool)))
        out = model(c, return_diagnostics=True)
        d = out['diagnostics']
        base = model.bias[None] + d['direct']
        eligible = c == 0
        eligible[:, 0] = False
        direct_values = d['direct'][eligible].detach().double().numpy().ravel()
        for j, (name, scale) in enumerate(multipliers.items()):
            pair = d['distinct_pair_mean'] * scale
            logits = (base + model.pair_bound*model.pair_raw.tanh()[None]*pair) * model._candidate_mask[None]
            loss = joint_loss(logits, c, target, selected)
            losses.append(float(loss.detach()))
            gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=j < len(multipliers)-1)
            for (parameter, _), gradient in zip(model.named_parameters(), gradients):
                accumulated[name][parameter] += gradient.detach() / (len(indices)/16)
            pair_values = pair[eligible].detach().double().numpy().ravel()
            moments[name] += [len(pair_values), direct_values.sum(), pair_values.sum(),
                              np.sum(direct_values**2), np.sum(pair_values**2), np.sum(direct_values*pair_values)]
    result = {'stage': 'post_test_exploratory_train_only_initialization_diagnostic',
              'optimizer_steps': 0, 'validation_read': False, 'test_read': False,
              'sample_users': len(indices), 'sample_seed': 20269101,
              'episode': {'seed': 2026, 'epoch': 1001, 'context_fraction': .8},
              'model_config': model.config, 'parameters': sum(p.numel() for p in model.parameters()),
              'source_sha256': {'model.py': digest(Path(__file__).parent/'model.py'),
                                'diagnose_initialization.py': digest(Path(__file__)),
                                'categorical_experiment.py': digest(ROOT/'categorical_experiment.py')},
              'training_categories_sha256': digest(args.train), 'normalizations': {}}
    for name, (n, sx, sy, xx, yy, xy) in moments.items():
        covariance = xy/n - (sx/n)*(sy/n)
        variance_x, variance_y = xx/n-(sx/n)**2, yy/n-(sy/n)**2
        result['normalizations'][name] = {
            'multiplier_from_pair_mean': multipliers[name],
            'direct_rms': float(np.sqrt(xx/n)), 'pair_rms': float(np.sqrt(yy/n)),
            'pair_to_direct_rms': float(np.sqrt(yy/xx)),
            'direct_pair_covariance': float(covariance),
            'direct_pair_correlation': float(covariance/np.sqrt(variance_x*variance_y)),
            'parameter_gradients': {key: gradient_stats(value) for key, value in accumulated[name].items()}}
    assert all(torch.equal(accumulated['pair_mean'][key], accumulated[name][key])
               for key in ('potentials', 'bias') for name in multipliers)
    assert all(losses[j] == losses[j//3*3] for j in range(len(losses)))
    result['same_initial_outputs_loss_and_direct_gradients'] = True
    result['caution'] = 'Adam rescales gradients; small raw gradients alone do not prove starvation. Feature RMS determines bounded branch influence, and epsilon affects very small gradients.'
    write_json(args.out, result)
    for name, row in result['normalizations'].items():
        print(name, 'direct_rms', row['direct_rms'], 'pair_rms', row['pair_rms'],
              'pair_gradient_rms', row['parameter_gradients']['pair_raw']['rms'], flush=True)


if __name__ == '__main__':
    main()
