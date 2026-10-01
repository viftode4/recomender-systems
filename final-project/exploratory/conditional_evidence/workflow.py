"""Resume the approved study through fitting, gated retrieval and assessment.

Long training runs are ordinary local processes. All scientific output is
written by the same source-sealed program; progress files are not results.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import run_experiment as run
from . import data, evaluation, model, retrieval
import numpy as np
import torch


POLICIES = ('learned', 'next_nearest', 'random', 'no_action')
RETRIEVAL_FILES = {'utility.pt', 'crossfit.npz', 'fit.json', 'prediction-costs.json',
                   'feature-shift.json', *(f'{policy}.npy' for policy in POLICIES)}


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(args, stage, **values):
    payload = {'stage': stage, 'updated_utc': now(), 'pid': os.getpid(),
               'original_test_read': False, **values}
    # The runner creates its output directory exclusively when beginning a run.
    path = args.out.parent / (args.out.name + '-progress.json')
    run.atomic_json(path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def release_for_closed_gate(directory):
    """Only a verified, genuinely closed META gate can skip acquisition."""
    directory = Path(directory)
    seal = run.verify_global_seal(directory)
    screen = json.loads((directory / 'meta-screen.json').read_text())
    recomputed = evaluation.meta_screen(screen['per_seed_ndcg'], screen['raw'],
                                        screen['controls'], screen['references'])
    if recomputed != screen or screen['passed']:
        raise ValueError('Cannot skip a triggered or inconsistent retrieval gate')
    return {'status': 'ready', 'base_seal_sha256': run.digest(directory / 'SELECTIONS-FROZEN.json'),
            'meta_screen_sha256': seal['meta_screen_sha256'],
            'extension_status': 'not_triggered', 'extension_artifacts_sha256': {},
            'reason': 'Raw reader did not exceed both matched controls and the strongest meta reference.'}


def fit_fold(request, directory, source):
    """Fixed-budget refit with only other-fold records and durable Adam state."""
    directory = Path(directory)
    run.numerical_setup(request.seed)
    initialization_seed = int(request.selected_config['seed'])
    reader = run.make_reader(model, 'raw', initialization_seed)
    optimizer = torch.optim.Adam(reader.parameters(),
        lr=float(request.selected_config['learning_rate']), weight_decay=1e-4)
    inputs = {'categories': request.categories}
    bank = model.EvidenceBank(request.categories)
    guard = {'source_sha256': source, 'fold': request.fold,
             'training_categories_sha256': evaluation.array_digest(request.categories),
             'selected_config': request.selected_config, 'epochs': request.epochs,
             'initialization_seed': initialization_seed, 'training_seed': request.seed}
    latest = directory / 'latest.pt'
    history, completed = {'epochs': [], 'checkpoints': []}, 0
    if latest.exists():
        completed, history = run.restore_checkpoint(latest, reader, optimizer, guard)
    if completed > request.epochs:
        raise ValueError('Crossfit budget changed on resume')
    for epoch in range(completed + 1, request.epochs + 1):
        result = run.train_epoch(reader, bank, inputs, data, optimizer, epoch, request.seed, 'raw')
        history['epochs'].append(result)
        run.save_checkpoint(latest, reader, optimizer, epoch, guard, history)
        if epoch % 25 == 0 or epoch == request.epochs:
            print(json.dumps({'stage': 'crossfit', 'fold': request.fold,
                              'epoch': epoch, 'budget': request.epochs}), flush=True)
    if request.epochs == 0 and not latest.exists():
        run.save_checkpoint(latest, reader, optimizer, 0, guard, history)
    run.check_hashes(source)
    return retrieval.ReaderFit(reader, request.epochs,
        {'initialization_seed': initialization_seed, 'training_seed': request.seed,
         'checkpoint_sha256': run.digest(latest), 'epochs': history['epochs'],
         'held_fold_received': False})


def verify_retrieval_manifest(directory, seed, base_hash):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('seed') != seed
            or manifest.get('base_seal_sha256') != base_hash
            or manifest.get('development_read') is not False
            or manifest.get('policies') != list(POLICIES)
            or set(manifest.get('artifacts', {})) != RETRIEVAL_FILES):
        raise ValueError('Incomplete or mismatched retrieval manifest')
    for name, expected in manifest['artifacts'].items():
        run.verified_relative(directory, name, expected)
    return manifest


def feature_shift(summary, fitted, user_count):
    count = int(summary['count'])
    if count <= 0:
        raise ValueError('No deployment action features to compare')
    mean = np.asarray(summary['sum']) / count
    variance = np.maximum(0, np.asarray(summary['sum_squared']) / count - mean**2)
    return {'deployment_action_count': count, 'deployment_mean': mean.tolist(),
        'deployment_std': np.sqrt(variance).tolist(),
        'crossfit_mean': fitted.utility.mean.tolist(),
        'crossfit_normalization_scale': fitted.utility.scale.tolist(),
        'mean_shift_in_crossfit_scale': ((mean-fitted.utility.mean)/fitted.utility.scale).tolist(),
        'full_training_bank_rows': user_count, 'deployment_allowed_bank_rows': user_count-1,
        'crossfit_bank_rows': [fold['training_users'] for fold in fitted.crossfit.folds],
        'crossfit_context': '80% and 90% retained TRAIN history',
        'deployment_context': 'complete TRAIN history', 'outcome_labels_used': False,
        'limitation': 'Common initialization does not ensure aligned latent coordinates across readers.'}


def fit_retrieval(args):
    """Freeze utility models and all policy predictions before any DEV access."""
    seal = run.verify_global_seal(args.out)
    screen = json.loads((args.out / 'meta-screen.json').read_text())
    if not screen['passed']:
        return release_for_closed_gate(args.out)
    values = screen['per_seed_ndcg']
    gate = retrieval.meta_gate(
        {int(s): row['raw'] for s, row in values.items()},
        {name: {int(s): row[name] for s, row in values.items()} for name in screen['controls']},
        {name: {int(s): row[name] for s, row in values.items()} for name in screen['references']},
        stage='meta_fit')
    if not gate.enabled:
        raise ValueError('Retrieval and evaluation gate implementations disagree')
    artifacts, score_files = {}, {}
    source = seal['source_sha256']
    for seed in run.SEEDS:
        progress(args, 'retrieval', seed=seed)
        inputs = run.load_inputs(args, data, seed)
        selected = json.loads((args.out / str(seed) / 'selection.json').read_text())['raw']
        target = args.out / 'retrieval' / str(seed)
        complete = target / 'manifest.json'
        if complete.exists():
            manifest = verify_retrieval_manifest(target, seed,
                run.digest(args.out / 'SELECTIONS-FROZEN.json'))
        else:
            target.mkdir(parents=True, exist_ok=True)
            probes = [retrieval.Episode(ep['user_index'], ep['context'],
                                       ep['candidate_ids'][ep['target_mask']])
                      for ep in data.episodes(inputs, epoch=0,
                            retain_fraction=(.8, .9), seed=seed + 61001)]
            fitted = retrieval.fit_extension(inputs['categories'], probes, gate=gate,
                selected_config=selected['config'],
                selected_epochs=int(selected['selected']['epoch']), seed=seed,
                fit_reader=lambda request: fit_fold(request,
                    target / f'fold-{request.fold}', source))
            if fitted.utility is None:
                raise ValueError('A triggered extension did not produce a utility model')
            run.atomic_torch(target / 'utility.pt', fitted.utility.checkpoint())
            run.atomic_write(target / 'crossfit.npz', lambda stream: np.savez_compressed(
                stream, features=fitted.crossfit.features, reductions=fitted.crossfit.reductions))
            run.atomic_json(target / 'fit.json', fitted.protocol_summary())
            reader = run.make_reader(model, 'raw', seed)
            lr_index = run.LEARNING_RATES.index(selected['config']['learning_rate'])
            checkpoint_dir = args.out / str(seed) / run.config_id('raw', lr_index)
            record = json.loads((checkpoint_dir / 'result.json').read_text())
            optimizer = torch.optim.Adam(reader.parameters(),
                lr=selected['config']['learning_rate'], weight_decay=1e-4)
            run.restore_checkpoint(checkpoint_dir / selected['selected']['checkpoint'],
                                   reader, optimizer, record['guard'])
            bank = model.EvidenceBank(inputs['categories'])
            predictions = {policy: np.zeros(inputs['categories'].shape, np.float64)
                           for policy in POLICIES}
            costs = {policy: [] for policy in POLICIES}
            feature_totals = {'count': 0, 'sum': np.zeros(35), 'sum_squared': np.zeros(35)}
            for row, context in enumerate(inputs['categories']):
                for policy in POLICIES:
                    result = retrieval.deploy(reader, bank, context, row, fitted.utility,
                        policy=policy, seed=seed + row + 62001,
                        chunk_size=run.CANDIDATE_CHUNK)
                    scores = np.asarray(result['scores'], dtype=np.float64)
                    # Seen/PAD sentinels belong to the internal retrieval helper;
                    # the exported aligned matrices always contain finite values.
                    eligible = (context == 0)
                    eligible[0] = False
                    if not np.isfinite(scores[eligible]).all():
                        raise ValueError('Nonfinite retrieval score for eligible item')
                    predictions[policy][row, eligible] = scores[eligible]
                    costs[policy].append(result['costs'].to_dict())
                    if policy == 'learned':
                        for key in feature_totals:
                            feature_totals[key] += result['feature_summary'][key]
            for policy, scores in predictions.items():
                run.atomic_array(target / f'{policy}.npy', scores)
            run.atomic_json(target / 'prediction-costs.json', costs)
            run.atomic_json(target / 'feature-shift.json',
                            feature_shift(feature_totals, fitted, len(inputs['categories'])))
            manifest = {'base_seal_sha256': run.digest(args.out / 'SELECTIONS-FROZEN.json'),
                'seed': seed, 'status': 'complete', 'development_read': False,
                'policies': list(POLICIES),
                'artifacts': {name: run.digest(target / name) for name in sorted(RETRIEVAL_FILES)}}
            run.atomic_json(complete, manifest)
            verify_retrieval_manifest(target, seed, manifest['base_seal_sha256'])
        for name, expected in manifest['artifacts'].items():
            artifacts[str((target / name).relative_to(args.out))] = expected
        artifacts[str(complete.relative_to(args.out))] = run.digest(complete)
        score_files[str(seed)] = {policy: str((target / f'{policy}.npy').relative_to(args.out))
                                 for policy in POLICIES}
    run.check_hashes(source)
    return {'status': 'ready', 'base_seal_sha256': run.digest(args.out / 'SELECTIONS-FROZEN.json'),
            'meta_screen_sha256': seal['meta_screen_sha256'], 'extension_status': 'complete',
            'extension_artifacts_sha256': artifacts, 'extension_scores': score_files}


def assess_retrieval(args, release):
    if release['extension_status'] != 'complete':
        return None
    run.verify_assessment_release(args.out, args.release)
    result_path = args.out / 'retrieval-assessment.json'
    receipt_path = args.out / 'retrieval-assessment-complete.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('release_sha256') != run.digest(args.release)
                or receipt.get('result_sha256') != run.digest(result_path)):
            raise ValueError('Completed retrieval assessment changed')
        return json.loads(result_path.read_text())['public']
    public = {}
    for seed in run.SEEDS:
        inputs = run.load_inputs(args, data, seed)
        outcomes = data.load_development(inputs, args.out,
            verify_global_seal=lambda _: run.verify_assessment_release(args.out, args.release))
        public[str(seed)] = {}
        for policy, name in release['extension_scores'][str(seed)].items():
            scores = np.load(args.out / name, allow_pickle=False)
            result = evaluation.evaluate_scores(scores, inputs['users'], inputs['items'],
                inputs['categories'], outcomes['truth'], inputs['dev_users'],
                ratings=outcomes['ratings'])
            public[str(seed)][policy] = result['public']
    run.immutable_json(result_path, {'public': public,
        'release_sha256': run.digest(args.release), 'fresh_confirmation': False})
    run.immutable_json(receipt_path, {'release_sha256': run.digest(args.release),
                                    'result_sha256': run.digest(result_path)})
    return public


def compute_report(directory):
    """Aggregate resource counters and curves without exporting user records."""
    trajectories, references = {}, {}
    for seed in run.SEEDS:
        references[str(seed)] = json.loads((directory / str(seed) / 'reference-metadata.json').read_text())
        for arm in run.ARMS:
            for index, _ in enumerate(run.LEARNING_RATES):
                key = f'{seed}/{run.config_id(arm, index)}'
                record = json.loads((directory / key / 'result.json').read_text())
                history = record['history']
                totals = {}
                for epoch in history['epochs']:
                    for name, value in epoch['resources'].items():
                        totals[name] = totals.get(name, 0) + value
                trajectories[key] = {'config': record['config'],
                    'completed_epochs': record['completed_epoch'],
                    'parameters': history['parameter_count'],
                    'training_seconds': sum(row['seconds'] for row in history['epochs']),
                    'meta_evaluation_seconds': sum(row['resources']['seconds'] for row in history['checkpoints']),
                    'training_resources': totals,
                    'peak_process_memory_bytes': record['peak_process_memory_bytes'],
                    'training_curve': [{name: row[name] for name in ('epoch', 'mean_query_loss', 'seconds')}
                                       for row in history['epochs']],
                    'meta_curve': [{name: row[name] for name in ('epoch', 'meta_ndcg@10', 'resources')}
                                   for row in history['checkpoints']]}
    return {'trajectories': trajectories, 'reference_selection_budgets': references,
        'memory_scope': 'Process lifetime peak; spawned workers may execute multiple trajectories',
        'counter_scope': 'Actual graph construction and inference calls, including both gradient passes',
        'timing_scope': 'Per-process elapsed times; summing concurrent trajectories is not total wall time'}


def curate(args):
    """Publish aggregates only; per-user outcomes remain in ignored run files."""
    aggregate_path = args.out / 'aggregates.json'
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get('status') != 'complete' or aggregate.get('original_test_read') is not False:
        raise ValueError('A completed, verified exploratory assessment is required')
    run.verify_assessment_release(args.out, args.release)
    run.atomic_json(args.evidence / 'compute-and-curves.json', compute_report(args.out))
    roles = [*run.ARMS, 'EASEexpanded', 'categorical', 'SLIM', 'neighbor']
    lines = ['# Conditional evidence: exploratory results', '',
        'MovieLens 100K was used in earlier research. These results are exploratory;',
        'the three splits overlap and are not independent datasets. Original TEST was not opened.', '',
        '| Model | Mean nDCG@10 | Mean recall@10 | Mean MRR@10 |',
        '|---|---:|---:|---:|']
    for role in roles:
        means = [np.mean([row['models'][role]['all_observed'][metric]
                          for row in aggregate['seeds'].values()])
                 for metric in ('ndcg@10', 'recall@10', 'mrr@10')]
        lines.append(f'| {role} | {means[0]:.6f} | {means[1]:.6f} | {means[2]:.6f} |')
    gate = aggregate['development_gate']
    lines += ['', f"Strongest reference chosen on meta data: **{gate['strongest_meta_reference']}**.",
        f"The declared 10% substantial-result target was **{'met' if gate['substantial_target_met'] else 'not met'}**.",
        'This target does not establish first-ever novelty or a universal performance advantage.', '',
        'Raw versus scrambled compares equal architectures and preserved rating margins.',
        'Raw versus summary also changes rating information, global degree features and parameter count;',
        'that comparison alone cannot isolate the value of correspondence.',
        'The three-layer anonymous mean reader cannot distinguish every different graph arrangement.', '']
    extension = args.out / 'retrieval-assessment.json'
    if extension.exists():
        values = json.loads(extension.read_text())
        run.atomic_json(args.evidence / 'retrieval-aggregates.json', values)
        diagnostics = {str(seed): {name: json.loads((args.out / 'retrieval' / str(seed) / name).read_text())
                                  for name in ('fit.json', 'feature-shift.json', 'prediction-costs.json')}
                       for seed in run.SEEDS}
        # Per-query costs are summed before publication; no individual traces.
        for value in diagnostics.values():
            value['prediction-costs.json'] = {policy: {
                key: sum(row[key] for row in rows) for key in rows[0]}
                for policy, rows in value['prediction-costs.json'].items()}
        run.atomic_json(args.evidence / 'retrieval-diagnostics.json', diagnostics)
        lines += ['The one-donor extension was triggered by its frozen meta-only gate.', '',
                  '| Policy | Mean nDCG@10 |', '|---|---:|']
        for policy in POLICIES:
            mean = np.mean([row[policy]['all_observed']['ndcg@10']
                            for row in values['public'].values()])
            lines.append(f'| {policy} | {mean:.6f} |')
        lines += ['', 'The utility model uses out-of-fold TRAIN loss reductions. Its oracle is a',
            'privileged TRAIN diagnostic; learned latent coordinates can still differ across',
            'fold readers despite their common initialization. No oracle DEV scores are reported.', '']
    else:
        lines += ['The meta-only gate did not trigger additional retrieval; no acquisition model was fitted.', '']
    final_cap = json.loads((args.out / 'training-summary.json').read_text())['extension']['target_epochs']
    boundary = []
    for seed, row in aggregate['seeds'].items():
        for arm, selected in row['selections'].items():
            epoch = selected['selected']['epoch']
            if epoch == final_cap:
                boundary.append(f'{seed}/{arm}: {epoch}')
    if boundary:
        lines += ['Selected budget-boundary checkpoints: ' + ', '.join(boundary) + '.',
                  'These selections do not demonstrate convergence.', '']
    lines += ['See [protocol](../PROTOCOL.md), `aggregates.json`, `provenance.json`, and `SHA256.json`.',
              '`compute-and-curves.json` contains all training curves, resource counts and reference search budgets.',
              '`RESULTS-SHA256.json` also covers this report and the supplementary diagnostics.',
              'Raw records, predictions, training checkpoints and per-user diagnostics remain in ignored runs.', '']
    run.atomic_write(args.evidence / 'RESULTS.md', lambda stream: stream.write('\n'.join(lines).encode()))
    inventory = {path.name: run.digest(path) for path in args.evidence.iterdir()
                 if path.is_file() and path.name != 'RESULTS-SHA256.json'}
    # Keep the base export inventory immutable so a completed resume can verify it.
    run.atomic_json(args.evidence / 'RESULTS-SHA256.json', inventory)
    # Confirmation is explicitly distinct from completion of an exploratory run.
    confirmation = {'criterion_met': gate['substantial_target_met'],
        'status': 'requires_separate_reserved_replication' if gate['substantial_target_met'] else 'not_triggered',
        'fresh_confirmation_completed': False,
        'next_dataset': 'MovieLens 1M',
        'rule': 'fixed method, 70/10/20 per-user rating-independent split; no final refit'}
    run.atomic_json(args.out / 'confirmation-status.json', confirmation)
    return confirmation


def execute(args):
    args.out = Path(args.out).resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # OS locks survive neither crashes nor reboot, avoiding stale lock files.
    with (args.out.parent / (args.out.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another workflow already owns this study directory') from error
        return execute_locked(args)


def execute_locked(args):
    args.out = Path(args.out).resolve()
    args.release = args.out / 'ASSESSMENT-RELEASE.json'
    args.assessment_release = args.release
    run.numerical_setup(run.SEEDS[0])
    if not args.resume and args.out.exists():
        raise ValueError('Output already exists; explicitly resume or choose a fresh path')
    try:
        if not (args.out / 'training-summary.json').exists():
            progress(args, 'training', workers=args.workers)
            run.execute_train(args)
        if not (args.out / 'SELECTIONS-FROZEN.json').exists():
            progress(args, 'sealing')
            run.execute_seal(args)
        run.verify_global_seal(args.out)
        if not args.release.exists():
            release = fit_retrieval(args)
            run.atomic_json(args.release, release)
        release = run.verify_assessment_release(args.out, args.release)
        progress(args, 'assessment', extension_status=release['extension_status'])
        run.execute_assess(args)
        assess_retrieval(args, release)
        confirmation = curate(args)
        progress(args, 'assessment_complete',
                 extension_status=release['extension_status'],
                 results=str(args.out / 'aggregates.json'),
                 confirmation_status=confirmation['status'])
    except BaseException as error:
        progress(args, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                 error_type=type(error).__name__, message=str(error),
                 resume_command=f'python -m {run.PACKAGE}.workflow --out {args.out} --resume')
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--out', type=Path, default=run.ROOT / 'runs/conditional-evidence-v1')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--workers', type=int, choices=(1, 3, 6), default=6)
    result.add_argument('--source-root', type=Path, default=data.DEFAULT_SOURCE)
    result.add_argument('--categorical-root', type=Path, default=data.DEFAULT_CATEGORICAL)
    result.add_argument('--reference-root', type=Path, default=data.DEFAULT_CATEGORICAL)
    result.add_argument('--ratings', type=Path, default=data.DEFAULT_RATINGS)
    result.add_argument('--evidence', type=Path,
                        default=run.ROOT / 'exploratory/conditional_evidence/results-v1')
    return result


if __name__ == '__main__':
    execute(parser().parse_args())
