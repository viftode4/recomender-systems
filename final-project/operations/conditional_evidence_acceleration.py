"""Opt-in, parity-tested execution acceleration for the frozen evidence reader.

Importing this module changes nothing. ``install`` replaces only the runner's
two_pass_backward callable in the current process; it does not alter files,
checkpoint guards, model/configuration, optimizer, examples or candidate chunks.
An operational caller must separately bind this source digest to its migration
receipt. Existing scientific source hashes alone do not describe this override.

The frozen reader is deterministic and has no dropout or forward state changes.
We retain each chunk's grad-enabled outputs, obtain d(loss)/d(logits) from an
independent detached concatenation, then backpropagate the saved chunks in the
original order. Arithmetic and parameter-gradient accumulation are unchanged.
All chunk activations for ONE query stay resident until backward starts; they
are released chunk by chunk. The original has only one chunk's activations live.
Physical graph-build/edge/scramble counters therefore halve during training,
while the semantic examples and optimizer steps are unchanged. Timing and peak
memory must be reported as new measurements, never spliced into old receipts.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from exploratory.conditional_evidence import run_experiment as runner

import numpy as np
import torch

ALGORITHM = 'retained-chunk-forward-ordered-backward-v1'
_ORIGINAL = runner.two_pass_backward


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt():
    return {
        'algorithm': ALGORITHM,
        'accelerator_path': 'operations/conditional_evidence_acceleration.py',
        'accelerator_sha256': digest(__file__),
        'baseline_path': 'exploratory/conditional_evidence/run_experiment.py',
        'baseline_sha256': digest(runner.__file__),
        'changed_callable': 'two_pass_backward',
        'candidate_chunk_or_optimizer_changed': False,
        'activation_retention': 'all candidate chunks of one query; release in original backward order',
        'resource_counter_meaning': 'physical work: one build/forward per chunk instead of two',
        'checkpoint_schema_changed': False,
        'requires_separate_operational_binding': True,
    }


def one_forward_backward(model, make_chunks, loss_function, divisor=1.0):
    """Same ordered chunk gradients as the frozen two-pass implementation.

    Requires its original deterministic/no-mutable-forward-state reader contract.
    Does not pretend to detect stochastic readers: use the explicit parity suite
    before activating it with any new reader or numerical backend.
    """
    saved = [model(batch).reshape(-1) for batch in make_chunks()]
    first = torch.cat([scores.detach() for scores in saved])
    logits = first.clone().requires_grad_(True)
    loss = loss_function(logits)
    derivative, = torch.autograd.grad(loss, logits)
    offset = 0
    for index in range(len(saved)):
        scores = saved[index]
        count = len(scores)
        torch.autograd.backward(scores, derivative[offset:offset + count] / divisor)
        offset += count
        saved[index] = None  # Drop this graph's reference immediately after use.
    return float(loss.detach())


def install(*, expected_accelerator_sha256=None, expected_baseline_sha256=None):
    """Explicit, idempotent current-process activation with optional hash guards."""
    result = receipt()
    for field, expected in (('accelerator_sha256', expected_accelerator_sha256),
                            ('baseline_sha256', expected_baseline_sha256)):
        if expected is not None and result[field] != expected:
            raise ValueError(f'acceleration source guard differs: {field}')
    if runner.two_pass_backward not in (_ORIGINAL, one_forward_backward):
        raise ValueError('refusing to replace an unknown backward override')
    runner.two_pass_backward = one_forward_backward
    return result


def uninstall():
    """Restore only our own override; do not remove an unrelated patch."""
    if runner.two_pass_backward is one_forward_backward:
        runner.two_pass_backward = _ORIGINAL
    elif runner.two_pass_backward is not _ORIGINAL:
        raise ValueError('unknown backward override')


def _equal_state(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_equal_state(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(map(lambda pair: _equal_state(*pair), zip(first, second)))
    return first == second


def benchmark(args):
    """Small TRAIN-only timing/parity run, never a study or selection result."""
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 1 <= args.episodes <= 16:
        raise ValueError('this diagnostic is limited to 1..16 query episodes per arm/method')
    from exploratory.conditional_evidence import data, model as model_api
    sources = runner.source_hashes()
    own_receipt = receipt()
    runner.numerical_setup(args.seed)
    inputs = data.load_inputs(args.seed, source_root=args.source_root,
                             categorical_root=args.categorical_root, ratings=args.ratings,
                             load_meta=False)
    bank = model_api.EvidenceBank(inputs['categories'])
    rows = {}
    for arm in args.arms:
        outputs, states = {}, {}
        for method, backward in (('two_pass', _ORIGINAL), ('one_forward', one_forward_backward)):
            runner.numerical_setup(args.seed)
            model = runner.make_reader(model_api, arm, args.seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
            started = time.perf_counter()
            with patch.object(runner, 'two_pass_backward', backward):
                result = runner.train_epoch(model, bank, inputs, data, optimizer, 1, args.seed,
                                            arm, episode_limit=args.episodes)
            outputs[method] = {'training': result, 'elapsed_seconds': time.perf_counter() - started,
                               'process_lifetime_peak_memory_bytes': runner.peak_memory_bytes()}
            states[method] = {'model': {key: value.clone() for key, value in model.state_dict().items()},
                              'optimizer': optimizer.state_dict(), 'rng': runner.rng_state()}
        checks = {key: _equal_state(states['two_pass'][key], states['one_forward'][key])
                  for key in ('model', 'optimizer', 'rng')}
        checks['loss_exact'] = outputs['two_pass']['training']['mean_query_loss'] == outputs['one_forward']['training']['mean_query_loss']
        if not all(checks.values()):
            raise ValueError(f'exact TRAIN-only parity failed: {arm}: {checks}')
        baseline, accelerated = (outputs[key]['training']['seconds'] for key in ('two_pass', 'one_forward'))
        rows[arm] = {'methods': outputs, 'bitwise_parity': checks,
                     'observed_speedup': baseline / accelerated,
                     'memory_caveat': 'Cumulative process lifetime peak; sequential values are not isolated per-method peaks.'}
    runner.check_hashes(sources)
    if own_receipt != receipt():
        raise ValueError('accelerator changed during benchmark')
    result = {'status': 'complete', 'stage': 'nonstudy_train_only_acceleration_diagnostic',
              'seed': args.seed, 'episodes_per_arm_method': args.episodes,
              'meta_labels_read': False, 'development_read': False, 'original_test_read': False,
              'input_signature': inputs['signature'], 'source_sha256': sources,
              'acceleration': own_receipt, 'runtime': runner.runtime(), 'arms': rows}
    args.out.mkdir(parents=True)
    runner.atomic_json(args.out / 'benchmark.json', result)
    print(json.dumps({arm: {'speedup': row['observed_speedup'], 'parity': row['bitwise_parity']}
                      for arm, row in rows.items()}, indent=2))


def main():
    from exploratory.conditional_evidence import data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=data.DEFAULT_SOURCE)
    parser.add_argument('--categorical-root', type=Path, default=data.DEFAULT_CATEGORICAL)
    parser.add_argument('--ratings', type=Path, default=data.DEFAULT_RATINGS)
    parser.add_argument('--seed', type=int, choices=runner.SEEDS, default=2026)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--arms', nargs='+', choices=runner.ARMS, default=list(runner.ARMS))
    benchmark(parser.parse_args())


if __name__ == '__main__':
    main()
