"""Explicit post-v1 optimization-budget sensitivity; old experiment untouched."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np

import joint_field_experiment as base
from categorical_experiment import (DEFAULT_CONFIG, aggregate_diagnostics, array_digest,
    categorical_matrix, make_model, masked_episode, restore, snapshot, state_digest)
from joint_evaluation import runtime_signature, verify_choices
from joint_field_experiment import (VARIANTS, aggregate_result, baseline_logits, evaluate_logits,
    joint_contrast, joint_metrics, joint_probe_loss, predict_logits, ranking_scores, validate_logits)
from exception_experiment import load_source
from study import digest, partition_users, write_json


CHECKPOINTS = (10, 30, 60, 100, 200, 300, 400)
REPLAY_EPOCH = 100
SOURCE_FILES = tuple(dict.fromkeys(base.SOURCE_FILES + (
    'joint_field_convergence.py', 'JOINT_FIELD_CONVERGENCE_PROTOCOL.md',
    'joint_evaluation.py', 'categorical_evaluation.py', 'exception_evaluation.py')))


def protocol(seeds):
    result = base.protocol(seeds)
    result.update(study_kind='post_v1_convergence_sensitivity', epochs=max(CHECKPOINTS),
        checkpoints=list(CHECKPOINTS), replay_epoch=REPLAY_EPOCH,
        decision='Post-v1 meta curves still decreased at the 100-epoch boundary; reused development evidence prompted one final budget sensitivity, before TEST.',
        restart='Original seed and fresh optimizer at epoch1; exact v1 trajectory verification before exceeding epoch100.',
        continuation_limit='400 epochs is the final cap; no later model or budget expansion.',
        parallelism='At most three seed processes, one numerical thread each; each fits both variants sequentially.',
        runtime=runtime_signature())
    return result


def verify_prefix_checkpoint(model, logits, prior_folder, epoch, matrix, new_path):
    prior_path = prior_folder / f'epoch-{epoch}.pt'
    prior_model, _ = restore(prior_path)
    prior_state = state_digest(prior_model)
    if state_digest(model) != prior_state:
        raise AssertionError(f'V1 state replay mismatch at epoch {epoch}')
    prior_logits = predict_logits(prior_model, matrix)
    if not np.array_equal(logits, prior_logits):
        raise AssertionError(f'V1 raw-logit replay mismatch at epoch {epoch}')
    return {'state_exact': True, 'raw_logits_exact': True, 'state_sha256': prior_state,
            'raw_logits_array_sha256': array_digest(logits),
            'prior_checkpoint_sha256': digest(prior_path), 'new_checkpoint_sha256': digest(new_path)}


def train_variant(matrix, users, items, validation, fit_users, seed, variant, output, prior_folder):
    import torch
    before = time.monotonic()
    prior_trace = json.loads((prior_folder / 'training-trace.json').read_text())
    if len(prior_trace) != REPLAY_EPOCH:
        raise ValueError('Prior study must contain the complete original prefix budget')
    model = make_model(matrix.shape[1], seed, variant)
    initial = state_digest(model)
    prior_selection = json.loads((prior_folder / 'selection.json').read_text())
    if initial != prior_selection['initial_state_sha256']:
        raise AssertionError('V1 initial model state differs before epoch1')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trace, grid, best, verified = [], [], None, {}
    for epoch in range(1, max(CHECKPOINTS) + 1):
        model.train()
        context, mask = masked_episode(matrix, seed, epoch)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 977])).permutation(len(matrix))
        total = 0.
        for start in range(0, len(matrix), 16):
            batch = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            loss = joint_probe_loss(model, torch.as_tensor(context[batch]),
                torch.as_tensor(matrix[batch]), torch.as_tensor(mask[batch]))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        trace.append({'epoch': epoch, 'macro_train_joint_probe_nll': total / len(matrix),
                      'context_sha256': array_digest(context), 'probe_mask_sha256': array_digest(mask)})
        if epoch <= REPLAY_EPOCH and trace[-1] != prior_trace[epoch - 1]:
            raise AssertionError(f'V1 training trace mismatch at epoch {epoch}')
        if epoch in CHECKPOINTS:
            logits = predict_logits(model, matrix)
            nll = joint_metrics(logits, users, items, matrix, validation, fit_users)['macro_joint_nll']
            row = {'epoch': epoch, 'meta_fit_macro_joint_nll': nll}
            grid.append(row)
            path = output / f'epoch-{epoch}.pt'
            torch.save(snapshot(model, epoch, seed), path)
            if epoch <= REPLAY_EPOCH:
                verified[str(epoch)] = verify_prefix_checkpoint(model, logits, prior_folder, epoch, matrix, path)
            if best is None or nll < best['meta_fit_macro_joint_nll']:
                best = dict(row)
                np.savez_compressed(output / 'selected-logits.npz', users=np.asarray(users), items=np.asarray(items), logits=logits)
            print(f'{seed} convergence {variant} epoch {epoch}: meta-fit joint NLL {nll:.6f}', flush=True)
        if epoch == REPLAY_EPOCH:
            verification = {'restarted_from_same_seed': True, 'replay_epoch': REPLAY_EPOCH,
                'prefix_training_trace_exact': trace == prior_trace, 'prefix_epoch_count': len(trace),
                'prior_variant_directory': str(prior_folder.resolve()),
                'prior_manifest_sha256': digest(prior_folder.parent / 'manifest.json'),
                'prior_training_trace_sha256': digest(prior_folder / 'training-trace.json'),
                'checkpoints': verified}
            write_json(output / 'verification.json', verification)
    selected = output / f"epoch-{best['epoch']}.pt"
    best.update(checkpoint=selected.name, checkpoint_sha256=digest(selected),
        initial_state_sha256=initial, selection_metric='meta_fit_macro_joint_nll', selection_cohort='meta_fit',
        parameter_count=sum(p.numel() for p in model.parameters()), elapsed_seconds=time.monotonic() - before,
        prefix100_verification_sha256=digest(output / 'verification.json'))
    write_json(output / 'training-trace.json', trace)
    write_json(output / 'selection-grid.json', grid)
    write_json(output / 'selection.json', best)
    restored, _ = restore(selected)
    replay = predict_logits(restored, matrix)
    with np.load(output / 'selected-logits.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['logits'], replay):
            raise AssertionError('Selected checkpoint failed exact replay')
    best.update(logits_array_sha256=array_digest(replay), checkpoint_replay_exact=True)
    write_json(output / 'selection.json', best)
    return best


def run_seed(source, ratings, metadata, output, seed, prior):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    prior_manifest, _, _ = verify_choices(prior)
    prior_manifest_hash = digest(prior / 'manifest.json')
    root = Path(__file__).resolve().parent
    sources = {name: digest(root / name) for name in SOURCE_FILES}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {**protocol([seed]), 'source_sha256': sources,
        'prior_v1_manifest_sha256': prior_manifest_hash})
    users, items, train, valid, training, validation, _, _, upstream = load_source(source, ratings, metadata)
    if (upstream.get('validation_used_for_training', True)
            or upstream['split_sha256'] != prior_manifest['split_sha256']
            or upstream['data_sha256'] != prior_manifest['data_sha256']):
        raise ValueError('Source differs from v1 or assimilated validation')
    matrix = categorical_matrix(users, items, training)
    with np.load(prior / 'training-categories.npz', allow_pickle=False) as saved:
        if (saved['users'].tolist() != users or saved['items'].tolist() != items
                or not np.array_equal(saved['ratings'], matrix)):
            raise ValueError('Original TRAIN query data differs from v1')
    fit_users, dev_users = partition_users(users, seed)
    write_json(output / 'cohorts.json', {'meta_fit': sorted(fit_users), 'development': sorted(dev_users)})
    np.savez_compressed(output / 'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    write_json(output / 'catalog.json', {'padding_index': 0, 'items': items})
    for name in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(source / name, output / name)
    selections = {}
    for variant in VARIANTS:
        folder = output / variant
        folder.mkdir()
        selections[variant] = train_variant(matrix, users, items, validation, fit_users, seed,
                                            variant, folder, prior / variant)
    if len({value['initial_state_sha256'] for value in selections.values()}) != 1:
        raise AssertionError('Core initial states differ')
    write_json(output / 'selection.json', selections)
    evaluated, diagnostics = {}, {}
    for variant in VARIANTS:
        folder = output / variant
        with np.load(folder / 'selected-logits.npz', allow_pickle=False) as saved:
            logits = saved['logits']
        evaluated[variant] = evaluate_logits(logits, users, items, matrix, train, validation, dev_users)
        model, _ = restore(folder / selections[variant]['checkpoint'])
        diagnostics[variant] = aggregate_diagnostics(model, matrix)
        write_json(folder / 'diagnostics.json', diagnostics[variant])
        for adapter, scores in ranking_scores(logits).items():
            export = folder / adapter
            export.mkdir()
            np.savez_compressed(export / 'valid-scores.npz', users=np.asarray(users), items=np.asarray(items), scores=scores)
            for name in ('train.tsv', 'valid.tsv', 'ids.json'):
                shutil.copyfile(source / name, export / name)
            write_json(export / 'manifest.json', {'status': 'complete', 'model': 'JointCategoricalFieldConvergence',
                'variant': variant, 'ranking_adapter': adapter, 'seed': seed, 'validation_used_for_training': False,
                'test_read': False, 'test_evaluated': False, 'selection': selections[variant],
                'source_sha256': sources, 'data_sha256': upstream['data_sha256'], 'split_sha256': upstream['split_sha256'],
                'predictor_kind': 'static_train_fitted_full_catalog',
                'score_space': protocol([seed])['ranking_adapters'][adapter.removesuffix('_adapter')],
                'export_sha256': {'valid-scores.npz': digest(export / 'valid-scores.npz')}})
    for name, logits in baseline_logits(matrix).items():
        evaluated[name] = evaluate_logits(logits, users, items, matrix, train, validation, dev_users)
    comparisons = {f'adaptive_minus_{name}': joint_contrast(evaluated['adaptive'], value, seed + 419)
                   for name, value in evaluated.items() if name != 'adaptive'}
    if sources != {name: digest(root / name) for name in SOURCE_FILES}:
        raise RuntimeError('Source changed during convergence study')
    verify_choices(prior)
    if digest(prior / 'manifest.json') != prior_manifest_hash:
        raise RuntimeError('Prior v1 evidence changed during convergence study')
    write_json(output / 'metrics.json', evaluated)
    write_json(output / 'comparisons.json', comparisons)
    verification = {v: json.loads((output / v / 'verification.json').read_text()) for v in VARIANTS}
    result = {'models': {name: aggregate_result(value) for name, value in evaluated.items()},
        'selections': selections, 'comparisons': comparisons, 'diagnostics': diagnostics,
        'cohorts': {'meta_fit': len(fit_users), 'development': len(dev_users)},
        'prefix100_replay_exact': all(v['prefix_training_trace_exact'] for v in verification.values()),
        'elapsed_seconds': time.monotonic() - started}
    write_json(output / 'aggregates.json', result)
    write_json(output / 'manifest.json', {'status': 'complete', 'seed': seed, 'test_read': False,
        'test_evaluated': False, 'source_sha256': sources, 'runtime': runtime_signature(),
        'source_run': str(source.resolve()), 'source_manifest_sha256': digest(source / 'manifest.json'),
        'prior_v1_manifest_sha256': prior_manifest_hash, 'prefix100_verification': verification,
        'data_sha256': upstream['data_sha256'], 'split_sha256': upstream['split_sha256'],
        'train_categories_sha256': array_digest(matrix), 'protocol_sha256': digest(output / 'protocol.json'),
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return result


def curate(output, evidence, seeds):
    base.curate(output, evidence, seeds)
    provenance_path = evidence / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['study_kind'] = 'post_v1_convergence_sensitivity'
    for seed in seeds:
        manifest = json.loads((output / str(seed) / 'manifest.json').read_text())
        provenance['seeds'][str(seed)].update(
            prior_v1_manifest_sha256=manifest['prior_v1_manifest_sha256'],
            prefix100_verification={variant: {key: value for key, value in verification.items()
                if key != 'prior_variant_directory'}
                for variant, verification in manifest['prefix100_verification'].items()},
            runtime=manifest['runtime'])
    write_json(provenance_path, provenance)
    path = evidence / 'RESULTS.md'
    text = path.read_text().replace('# Joint recorded-item and rating likelihood', '# Post-v1 joint-field convergence sensitivity')
    text = text.replace('epochs 10, 30, 60 or 100', 'epochs 10, 30, 60, 100, 200, 300 or 400')
    text += '\nThis explicitly post-v1 extension was motivated by decreasing meta-fit loss at the original budget boundary. It restarts from the same seeds and verifies the complete first-100 training trajectory against v1. It reuses development data and is not independent confirmation. No further model or budget expansion is planned.\n'
    path.write_text(text)
    write_json(evidence / 'SHA256.json', {p.name: digest(p) for p in sorted(evidence.iterdir()) if p.is_file() and p.name != 'SHA256.json'})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--item-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Require distinct nonnegative seeds')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    write_json(args.out / 'protocol.json', {**protocol(args.seeds),
        'source_sha256': {name: digest(root / name) for name in SOURCE_FILES},
        'prior_v1_manifests_sha256': {str(s): digest(args.prior_root / str(s) / 'manifest.json') for s in args.seeds}})
    results = {}
    with ProcessPoolExecutor(max_workers=min(3, len(args.seeds)), mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_seed, args.source_root / f'{s}-EASE-1', args.ratings,
            args.item_metadata, args.out / str(s), s, args.prior_root / str(s)): s for s in args.seeds}
        for future in as_completed(futures):
            results[str(futures[future])] = future.result()
            write_json(args.out / 'summary.json', results)
    if args.evidence:
        curate(args.out, args.evidence, args.seeds)
    print(json.dumps({'status': 'complete', 'out': str(args.out), 'test_read': False}), flush=True)


if __name__ == '__main__':
    main()
