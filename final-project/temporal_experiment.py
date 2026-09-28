"""Bounded per-user chronological sensitivity study; validation only.

This changes RecBole's within-user order from RO to TO while preserving the
80/10/10 ratio and shared hyperparameter grids. It is NOT a global timestamp
cutoff: other users' training interactions may postdate a user's validation
interaction. Different held-out interactions prevent interpreting score changes
as improvements on an identical prediction task. MovieLens timestamps order
rating entries, not movie viewing; see Harper and Konstan (2015), section 3.2.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment import variant_settings
from study import digest, partition_users, read_pairs, write_json


ROOT = Path(__file__).resolve().parent
MODELS = ('ExactPop', 'Random', 'EASE', 'ItemKNN')
TIMESTAMP_SOURCE = 'https://files.grouplens.org/papers/harper-tiis2015.pdf'
TIMESTAMP_CAVEAT = ('MovieLens timestamps record rating entries, which may be submitted long after viewing '
                    'or backfilled in batches. This experiment orders rating activity and cannot establish '
                    'viewing order or changing taste.')


def read_json(path):
    return json.loads(path.read_text())


def verify_random_grid(selection, root, seed):
    """Check actual candidate manifests, rather than trusting selection prose."""
    for model in MODELS:
        expected = variant_settings(model, True)
        keys = sorted(set().union(*(settings.keys() for settings in expected)))
        actual = []
        for candidate in selection[model]['candidates']:
            path = root / Path(candidate['path']).name
            manifest = read_json(path/'manifest.json')
            settings = manifest['settings']
            if (manifest['model'] != model or settings['seed'] != seed
                    or settings['eval_args']['order'] != 'RO' or manifest['test_evaluated']
                    or manifest['validation_used_for_training']):
                raise ValueError('Random grid contains an incompatible source run')
            actual.append({key: settings[key] for key in keys})
        canonical = lambda rows: sorted(json.dumps(row, sort_keys=True) for row in rows)
        if canonical(actual) != canonical(expected):
            raise ValueError(f'Random control uses a different tuning grid for {model}')


def split_diagnostics(path):
    """Describe changes in observed validation targets without touching test."""
    train, valid = read_pairs(path/'train.tsv'), read_pairs(path/'valid.tsv')
    counts = Counter(item for _, item in train)
    valid_counts = sorted(counts.get(item, 0) for _, item in valid)
    return {'training_interactions': len(train), 'validation_interactions': len(valid),
            'validation_fraction_items_unseen_in_training': sum(count == 0 for count in valid_counts)/len(valid_counts),
            'validation_target_mean_training_frequency': sum(valid_counts)/len(valid_counts)}


def run_logged(command, log):
    environment = dict(os.environ)
    environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    with log.open('x') as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_study(paths, data, out, seed):
    run_logged([sys.executable, str(ROOT/'study.py'), '--runs', *map(str, paths),
                '--items', str(data/'ml-100k/ml-100k.item'), '--out', str(out),
                '--seed', str(seed), '--calibration-fraction', '.5'], out.with_suffix('.log'))


def comparable_rows(path):
    manifest = read_json(path/'manifest.json')
    selection = read_json(path/'selection.json')
    results = read_json(path/'results.json')
    labels = {source['manifest']['model']: name for name, source in manifest['sources'].items()}
    labels.update(selection['families'])
    labels.update(best_expert=selection['best_expert'], rrf='rrf', group_switch='group-switch')
    return {family: {'model': name, 'aggregate': results[name]['aggregate'], 'groups': results[name]['groups']}
            for family, name in labels.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--random-selection', type=Path, required=True,
                        help='Existing random-order expert-selection-SEED.json; reuse its selected four baselines')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.out.exists() or args.evidence.exists():
        parser.error('Refusing existing run or evidence directory')
    source_selection = read_json(args.random_selection)
    verify_random_grid(source_selection, args.random_selection.parent, args.seed)
    random_paths = []
    for model in MODELS:
        # Resolve by run basename beneath the given selection file for portability.
        path = args.random_selection.parent / Path(source_selection[model]['selected']['path']).name
        manifest = read_json(path/'manifest.json')
        if (manifest['settings']['seed'] != args.seed or manifest['settings']['eval_args']['order'] != 'RO'
                or manifest['test_evaluated'] or manifest['validation_used_for_training']):
            raise ValueError('Random control must match seed, use RO, and remain validation-only/fixed-budget')
        if model in ('EASE', 'ItemKNN'):
            candidates = source_selection[model]['candidates']
            if source_selection[model]['selected']['meta_fit_ndcg'] != max(c['meta_fit_ndcg'] for c in candidates):
                raise ValueError('Random expert not selected on meta-fit validation')
        random_paths.append(path.resolve())
    args.out.mkdir(parents=True)
    started = time.monotonic()
    data, output = args.data_path.resolve(), args.out.resolve()
    base = read_json(ROOT/'config.json')
    base['seed'] = args.seed
    base['eval_args']['order'] = 'TO'
    plan = {'status': 'started', 'seed': args.seed, 'models': MODELS, 'test_evaluated': False,
            'numerical_threads': 1,
            'grids': {m: variant_settings(m, True) for m in MODELS},
            'random_control_sources': [str(p) for p in random_paths],
            'selection': 'Individual hyperparameters selected by meta-fit NDCG@10; hybrid settings by disjoint development users.',
            'chronology': 'Per-user ordered 80/10/10 split; not a global timestamp cutoff.',
            'timestamp_semantics': {'caveat': TIMESTAMP_CAVEAT, 'source': TIMESTAMP_SOURCE},
            'confounds': 'Validation labels and candidate histories change across split protocols; absolute metric differences are sensitivity, not improvement.',
            'source_sha256': {name: digest(ROOT/name) for name in ('temporal_experiment.py', 'run.py', 'study.py', 'experiment.py')}}
    write_json(output/'plan.json', plan)
    selections, temporal_paths = {}, []
    for model in MODELS:
        candidates = []
        for index, settings in enumerate(variant_settings(model, True)):
            label = f'{args.seed}-{model}-{index}'
            config = output / f'{label}-config.json'
            write_json(config, {**base, **settings})
            path = output/label
            print(f'Temporal {label}: {settings}', flush=True)
            run_logged([sys.executable, str(ROOT/'run.py'), '--model', model, '--data-path', str(data),
                        '--config', str(config), '--out', str(path), '--fixed-epochs'], output/f'{label}.log')
            metrics = read_json(path/'valid-metrics.json')['per_user']
            fit_users, _ = partition_users(list(metrics), args.seed)
            score = sum(metrics[user]['ndcg@10'] for user in fit_users) / len(fit_users)
            candidates.append({'path': str(path), 'settings': settings, 'meta_fit_ndcg': score})
        winner = max(candidates, key=lambda candidate: candidate['meta_fit_ndcg'])
        selections[model] = {'selected': winner, 'candidates': candidates}
        temporal_paths.append(Path(winner['path']))
    write_json(output/f'expert-selection-{args.seed}.json', selections)
    random_manifest = read_json(random_paths[0]/'manifest.json')
    temporal_manifest = read_json(temporal_paths[0]/'manifest.json')
    if (random_manifest['data_sha256'] != temporal_manifest['data_sha256']
            or random_manifest['split_sizes'] != temporal_manifest['split_sizes']
            or read_json(random_paths[0]/'ids.json') != read_json(temporal_paths[0]/'ids.json')):
        raise ValueError('Control and temporal runs differ in dataset, split sizes, or internal IDs')
    print('Matched four-expert random and temporal hybrid studies', flush=True)
    run_study(temporal_paths, data, output/'temporal-study', args.seed)
    run_study(random_paths, data, output/'random-control-study', args.seed)
    random_cohorts = read_json(output/'random-control-study/cohorts.json')
    temporal_cohorts = read_json(output/'temporal-study/cohorts.json')
    if random_cohorts != temporal_cohorts:
        raise ValueError('Split sensitivity cohorts or activity groups differ')
    comparison = {'random_user_order': comparable_rows(output/'random-control-study'),
                  'chronological_user_order': comparable_rows(output/'temporal-study')}
    plan.update(status='complete', seconds=time.monotonic()-started,
                matched_cohorts={name: len(random_cohorts[name]) for name in ('meta_fit', 'development', 'calibration')})
    write_json(output/'plan.json', plan)
    args.evidence.mkdir(parents=True)
    write_json(args.evidence/'comparison.json', comparison)
    write_json(args.evidence/'MANIFEST.json', plan)
    write_json(args.evidence/'expert-selection.json', selections)
    write_json(args.evidence/'split-diagnostics.json', {
        'random_user_order': split_diagnostics(random_paths[0]),
        'chronological_user_order': split_diagnostics(temporal_paths[0])})
    for prefix in ('random-control', 'temporal'):
        path = output/f'{prefix}-study'
        for name in ('selection.json', 'coefficients.json', 'group-policy.json'):
            data = read_json(path/name)
            if name == 'group-policy.json' and data.get('independent_calibration'):
                policy = data['independent_calibration']
                policy['calibration_user_count'] = len(policy.pop('calibration_users'))
            write_json(args.evidence/f'{prefix}-{name}', data)
    rows = ['# Per-user chronology sensitivity', '',
            'Validation only. Both protocols use the same four experts, tuning grids, user cohorts and hybrid code.',
            'Chronological ordering is within each user; it does not prevent global-timeline leakage. Held-out labels change, so absolute differences measure split sensitivity.', '',
            f'{TIMESTAMP_CAVEAT} [Harper and Konstan (2015), section 3.2]({TIMESTAMP_SOURCE}).', '',
            '| Model family | Random NDCG@10 | Chronological NDCG@10 |', '|---|---:|---:|']
    for family in ('Random', 'ExactPop', 'EASE', 'ItemKNN', 'best_expert', 'static', 'constrained', 'item', 'context', 'user', 'rrf', 'group_switch'):
        rows.append(f"| {family} | {comparison['random_user_order'][family]['aggregate']['ndcg@10']:.4f} | {comparison['chronological_user_order'][family]['aggregate']['ndcg@10']:.4f} |")
    rows += ['', 'These are selected development results on one seed. They do not establish final-test gains, causality, or deployment realism.',
             'The matched four-expert control excludes BPR from both splits, so hybrid values need not match the full primary study.',
             'Runtime provenance is recorded in the source run manifests; this script makes no cross-platform reproducibility claim.']
    (args.evidence/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    write_json(args.evidence/'SHA256.json', {p.name: digest(p) for p in sorted(args.evidence.iterdir()) if p.is_file()})
    print(json.dumps({'evidence': str(args.evidence), 'seconds': plan['seconds'], 'test_evaluated': False}))


if __name__ == '__main__':
    main()
