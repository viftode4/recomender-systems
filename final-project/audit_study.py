"""Add training-defined taste groups and genre-calibration audits to a study.

The CLI reads recommendations and already computed validation or final metrics,
never test labels or test score files. Raw ratings are retained only for exact
training pairs after hashes establish the source dataset and interaction split.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from societal import calibration_diagnostics, group_diagnostics, training_taste_groups
from study import digest, load_genres, read_pairs, write_json


def audit_models(recommendations, results, user_ids, items, train_ratings,
                 genre_fraction, *, baseline=None, k=10):
    """Pure audit: training ratings plus recommendations and existing metrics.

    Recommendations/results use original user/item IDs. ``items`` is the same
    index-ordered list used by the genre matrix, including the padding entry.
    ``train_ratings`` contains (original user ID, integer item index, rating).
    No held-out rating is used to define a group or a calibration target.
    """
    if not recommendations or set(recommendations) != set(results):
        raise ValueError('Recommendation and metric models must match exactly')
    if k < 1 or len(set(items)) != len(items) or len(items) != len(genre_fraction):
        raise ValueError('Invalid cutoff or catalog/genre alignment')
    if baseline is not None and baseline not in results:
        raise ValueError('Unknown reference model')
    definitions = training_taste_groups(user_ids, train_ratings, genre_fraction)
    index = {item: i for i, item in enumerate(items)}
    reference = None if baseline is None else results[baseline]['per_user']
    metric = f'ndcg@{k}'
    audits = {}
    for model in sorted(recommendations):
        recs = recommendations[model]
        rows = results[model]['per_user']
        if not rows or set(recs) != set(rows) or set(rows) - set(user_ids):
            raise ValueError('Recommendation/metric users differ or are unknown')
        if reference is not None and set(rows) != set(reference):
            raise ValueError('Reference must use the same evaluation cohort')
        indexed = {}
        for user, selected in recs.items():
            if len(selected) != k or set(selected) - set(index):
                raise ValueError('Wrong cutoff or unknown recommended item')
            indexed[user] = [index[item] for item in selected]
        calibration = calibration_diagnostics(indexed, genre_fraction,
                                              definitions['profiles'], definitions['liked_profiles'])
        groups = {name: group_diagnostics(rows, membership, metric, reference)
                  for name, membership in definitions['groups'].items()}
        # JSD is a cost, so avoid labelling its minimum a worst-group utility.
        calibration_by_group = {}
        for name, membership in definitions['groups'].items():
            grouped = {}
            for group in sorted(set(membership[u] for u in rows)):
                group_users = [u for u in rows if membership[u] == group]
                keys = sorted(set().union(*(calibration['per_user'][u] for u in group_users)))
                grouped[group] = {'users': len(group_users), 'metrics': {key: {
                    'mean': float(np.mean([calibration['per_user'][u][key] for u in group_users
                                           if key in calibration['per_user'][u]])),
                    'users': sum(key in calibration['per_user'][u] for u in group_users)} for key in keys}}
            calibration_by_group[name] = grouped
        calibration['groups'] = calibration_by_group
        audits[model] = {'groups': groups, 'calibration': calibration}
    return {'training_groups': definitions, 'models': audits,
            'reference': baseline, 'metric': metric,
            'note': 'Descriptive audits of existing predictions; groups and calibration targets use training ratings only. Genre contradiction is a proxy, not emotional understanding.'}


def load_training_ratings(path, train_pairs, item_index):
    """Join the atomic raw ratings onto an exact set of training pairs only."""
    pairs = list(train_pairs)
    selected = set(pairs)
    if not selected or len(selected) != len(pairs):
        raise ValueError('Unique, nonempty training pairs required')
    found, ratings = set(), []
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        required = {'user_id:token', 'item_id:token', 'rating:float'}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError('Missing atomic rating columns')
        for row in reader:
            pair = row['user_id:token'], row['item_id:token']
            if pair not in selected:
                continue
            if pair in found:
                raise ValueError('Duplicate raw rating for a training pair')
            if pair[1] not in item_index:
                raise ValueError('Training item is absent from catalog')
            found.add(pair)
            rating = float(row['rating:float'])
            if not np.isfinite(rating):
                raise ValueError('Nonfinite training rating')
            ratings.append((pair[0], item_index[pair[1]], rating))
    if found != selected:
        raise ValueError('Raw data is missing a training rating')
    return ratings


def load_final_audit_inputs(frozen, final_results, data):
    """Read completed final predictions plus verified TRAIN-only context.

    No inference is imported or executed. Derived final metrics are opened, but
    test TSVs, held-out rating values and validation rating values are not read.
    """
    from compare_frozen import load_completed
    frozen, final_results, data = Path(frozen), Path(final_results), Path(data)
    bundle, results, _ = load_completed(frozen, final_results)
    final_manifest = json.loads((final_results / 'manifest.json').read_text())
    recommendations_path = final_results / 'recommendations.json'
    if digest(recommendations_path) != final_manifest.get('output_sha256', {}).get('recommendations.json'):
        raise ValueError('Final recommendations differ from completed manifest')
    train_path, archive_path = frozen / 'train.tsv', frozen / 'frozen.npz'
    for path in (train_path, archive_path):
        if digest(path) != bundle.get('artifacts_sha256', {}).get(path.name):
            raise ValueError('Frozen training/context artifact hash mismatch')
    signatures = [source['manifest']['data_sha256'] for source in bundle['sources'].values()]
    if not signatures or any(signature != signatures[0] for signature in signatures):
        raise ValueError('Frozen experts have inconsistent dataset signatures')
    metadata = signatures[0]
    files = {}
    for suffix in ('.inter', '.item'):
        names = [name for name in metadata if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError('Expected unique interaction and item metadata files')
        path = data / names[0]
        if digest(path) != metadata[names[0]]:
            raise ValueError('Raw data differs from frozen dataset')
        files[suffix] = path
    with np.load(archive_path, allow_pickle=False) as archive:
        users, items = archive['users'].tolist(), archive['items'].tolist()
    ratings = load_training_ratings(files['.inter'], read_pairs(train_path),
                                    {item: i for i, item in enumerate(items)})
    _, genres = load_genres(files['.item'], items)
    return {'users': users, 'items': items, 'ratings': ratings, 'genres': genres,
            'recommendations': json.loads(recommendations_path.read_text()), 'results': results,
            'baseline': bundle['selection']['best_expert'], 'k': bundle['k'],
            'inputs': [frozen / 'freeze.json', frozen / 'freeze.sha256', train_path, archive_path,
                       final_results / 'manifest.json', final_results / 'results.json',
                       recommendations_path, *files.values()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--study', type=Path)
    mode.add_argument('--final', type=Path, help='Completed final-evaluation directory; never a test TSV')
    parser.add_argument('--source-run', type=Path, help='Required with --study')
    parser.add_argument('--frozen', type=Path, help='Required with --final')
    parser.add_argument('--data', type=Path, required=True,
                        help='Atomic dataset directory containing *.inter and *.item')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', help='Existing model label; default study best expert')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Export only group counts and aggregate metrics, no individual-user data')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    if args.final:
        if args.frozen is None or args.source_run is not None:
            parser.error('--final requires --frozen and does not use --source-run')
        context = load_final_audit_inputs(args.frozen, args.final, args.data)
        users, items, ratings, genres = (context[key] for key in ('users', 'items', 'ratings', 'genres'))
        recommendations, results = context['recommendations'], context['results']
        baseline, k, inputs = context['baseline'], context['k'], context['inputs']
        if args.baseline is not None and args.baseline != baseline:
            parser.error('Final audit reference must be the validation-selected frozen best expert')
    else:
        if args.source_run is None or args.frozen is not None:
            parser.error('--study requires --source-run and does not use --frozen')
        study_manifest = json.loads((args.study / 'manifest.json').read_text())
        source_manifest = json.loads((args.source_run / 'manifest.json').read_text())
        if study_manifest.get('status') != 'complete' or source_manifest.get('status') != 'complete':
            raise ValueError('Complete source run and study required')
        # The audit need not open validation/test TSVs or score files.
        train_path = args.source_run / 'train.tsv'
        if digest(train_path) != source_manifest['split_sha256']['train']:
            raise ValueError('Training split differs from source manifest')
        if not any(source['manifest']['split_sha256'] == source_manifest['split_sha256']
                   and source['manifest']['data_sha256'] == source_manifest['data_sha256']
                   for source in study_manifest['sources'].values()):
            raise ValueError('Source dataset/split is unrelated to this study')
        files = {}
        for suffix in ('.inter', '.item'):
            names = [name for name in source_manifest['data_sha256'] if name.endswith(suffix)]
            if len(names) != 1:
                raise ValueError('Expected one atomic interaction and item metadata file')
            path = args.data / names[0]
            if digest(path) != source_manifest['data_sha256'][names[0]]:
                raise ValueError('Raw dataset differs from source manifest')
            files[suffix] = path
        ids = json.loads((args.source_run / 'ids.json').read_text())
        if ids.get('padding_index') != 0:
            raise ValueError('Expected padding at index zero')
        users, items = ids['users'][1:], ids['items']
        ratings = load_training_ratings(files['.inter'], read_pairs(train_path),
                                        {item: i for i, item in enumerate(items)})
        _, genres = load_genres(files['.item'], items)
        recommendations = json.loads((args.study / 'recommendations.json').read_text())
        results = json.loads((args.study / 'results.json').read_text())
        baseline = args.baseline
        selection_file = args.study / 'selection.json'
        if baseline is None and selection_file.exists():
            baseline = json.loads(selection_file.read_text()).get('best_expert')
        k = study_manifest['k']
        inputs = [train_path, args.source_run / 'manifest.json', args.source_run / 'ids.json',
                  args.study / 'manifest.json', args.study / 'recommendations.json',
                  args.study / 'results.json', *files.values()]
    audit = audit_models(recommendations, results, users, items, ratings, genres,
                         baseline=baseline, k=k)
    args.out.mkdir(parents=True, exist_ok=False)
    definitions = audit.pop('training_groups')
    group_counts = {name: dict(Counter(labels.values())) for name, labels in definitions['groups'].items()}
    if args.aggregate_only:
        for report in audit['models'].values():
            del report['calibration']['per_user']
        exported_definitions = {'group_counts': group_counts, 'thresholds': definitions['thresholds'],
                                'definition': definitions['definition']}
    else:
        exported_definitions = definitions
    write_json(args.out / 'training-taste-groups.json', exported_definitions)
    write_json(args.out / 'societal-audit.json', audit)
    provenance = {'status': 'complete', 'test_read': False, 'training_ratings': len(ratings),
                  'evaluation_phase': 'final' if args.final else 'development',
                  'test_derived_metrics_read': bool(args.final), 'test_interactions_opened': False,
                  'individual_user_data_exported': not args.aggregate_only,
                  'study_source': str(args.final or args.study),
                  'source_run': str(args.frozen or args.source_run),
                  'baseline': baseline, 'training_only_groups': True,
                  'inputs_sha256': {str(p): digest(p) for p in inputs},
                  'code_sha256': {p.name: digest(p) for p in [Path(__file__),
                      Path(__file__).with_name('societal.py'), Path(__file__).with_name('study.py'),
                      Path(__file__).with_name('compare_frozen.py')]},
                  'interpretation': 'No new performance estimate; descriptive audit of existing study predictions.'}
    write_json(args.out / 'manifest.json', provenance)
    lines = ['# Training-defined societal audit', '', audit['note'], '',
             'All metrics below reuse the existing evaluation cohort. No test interactions or held-out rating values were opened.', '',
             '| Training group definition | Membership counts across all training users |', '|---|---|']
    for name, counts in group_counts.items():
        lines.append(f'| {name} | ' + ', '.join(f'{g}: {n}' for g, n in sorted(counts.items())) + ' |')
    lines += ['', '| Model | Worst activity nDCG | Worst contradiction-group nDCG | All-history JSD | Liked-history JSD |',
              '|---|---:|---:|---:|---:|']
    for model, report in audit['models'].items():
        activity = report['groups']['activity']['worst_group_utility']
        contradiction = report['groups']['contradiction']['worst_group_utility']
        calibration = report['calibration']['aggregate']
        all_jsd = calibration['jsd_all_history']['mean']
        liked = calibration.get('jsd_liked_history', {}).get('mean')
        liked_text = 'n/a' if liked is None else f'{liked:.4f}'
        lines.append(f'| {model} | {activity:.4f} | {contradiction:.4f} | {all_jsd:.4f} | {liked_text} |')
    (args.out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    provenance['output_sha256'] = {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.name != 'manifest.json'}
    write_json(args.out / 'manifest.json', provenance)
    print(json.dumps({'out': str(args.out), 'models': len(audit['models']),
                      'training_ratings': len(ratings), 'test_read': False}))


if __name__ == '__main__':
    main()
