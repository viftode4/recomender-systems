"""Locked-reference comparisons on the field study's development users.

No fitting, hyperparameter search, test split access or model-source edits.
Output aggregates contain no user IDs; detailed metrics remain under runs/.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from exception_experiment import evaluate_ratings, load_source, recommendations
from exception_model import rating_matrices
from study import digest, grouped, partition_users, write_json


ENDPOINTS = ('all_observed', 'liked_ratings')
SOURCES = ('field_reference_comparison.py', 'exception_experiment.py', 'exception_model.py',
           'study.py', 'metrics.py', 'hybrid_constraints.py')


def identity_digest(users, items):
    return hashlib.sha256(json.dumps({'users': users, 'items': items}, sort_keys=True).encode()).hexdigest()


def verify_common(directory, reference, split_directory=None):
    split_directory = directory if split_directory is None else split_directory
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['status'] != 'complete' or manifest.get('test_evaluated', True)
            or manifest.get('test_read', False) or manifest.get('validation_used_for_training', False)):
        raise ValueError(f'Incomplete or label-exposed run: {directory}')
    if manifest['data_sha256'] != reference['data_sha256']:
        raise ValueError(f'Dataset signature mismatch: {directory}')
    for part in ('train', 'valid'):
        if (manifest['split_sha256'][part] != reference['split_sha256'][part]
                or digest(split_directory/f'{part}.tsv') != reference['split_sha256'][part]):
            raise ValueError(f'Split signature mismatch: {directory}/{part}')
    return manifest


def load_scores(directory, reference, users, items, ids_hash, shared_directory=None):
    manifest = verify_common(directory, reference, shared_directory)
    identity_directory = directory if shared_directory is None else shared_directory
    if digest(identity_directory/'ids.json') != ids_hash:
        raise ValueError(f'Internal ID mapping hash mismatch: {directory}')
    score_file = directory/'valid-scores.npz'
    score_hash = digest(score_file)
    pinned = manifest.get('export_sha256', {}).get('valid-scores.npz')
    if pinned is not None and pinned != score_hash:
        raise ValueError(f'Exported score hash mismatch: {directory}')
    with np.load(score_file, allow_pickle=False) as archive:
        actual_users, actual_items = archive['users'].tolist(), archive['items'].tolist()
        scores = archive['scores'].astype(float)
    if actual_users != users or actual_items != items or scores.shape != (len(users), len(items)):
        raise ValueError(f'Original score IDs differ: {directory}')
    if not np.isfinite(scores).all():
        raise ValueError(f'Nonfinite score: {directory}')
    provenance = {'manifest_sha256': digest(directory/'manifest.json'),
                  'score_file_sha256': score_hash, 'preexisting_export_hash_available': pinned is not None,
                  'ids_file_sha256': ids_hash, 'ordered_identity_sha256': identity_digest(users, items),
                  'score_space': manifest.get('score_space', 'original model full-catalog score')}
    return scores, manifest, provenance


def verify_field_seed(directory, reference, users, items, fit_users, dev_users):
    manifest = verify_common(directory, reference)
    cohorts = json.loads((directory/'cohorts.json').read_text())
    if set(cohorts['meta_fit']) != fit_users or set(cohorts['development']) != dev_users:
        raise ValueError(f'Field cohort mismatch: {directory}')
    if sorted(cohorts['meta_fit']+cohorts['development']) != sorted(users):
        raise ValueError(f'Duplicate, overlapping or missing field users: {directory}')
    catalog = json.loads((directory/'catalog.json').read_text())
    catalog_items = catalog['items'] if isinstance(catalog, dict) else catalog
    if catalog_items != items:
        raise ValueError(f'Field catalog mismatch: {directory}')
    for name, expected in manifest['output_sha256'].items():
        # Field output manifests list TRAIN/validation artifacts only. Refuse a
        # future schema that would make this audit open a test artifact.
        if 'test' in Path(name).name.lower():
            raise ValueError('Field manifest unexpectedly requests a test artifact')
        artifact = (directory/name).resolve()
        if not artifact.is_relative_to(directory.resolve()):
            raise ValueError('Field artifact escapes its run directory')
        if digest(artifact) != expected:
            raise ValueError(f'Field artifact hash mismatch: {directory/name}')
    return manifest


def verify_field_readout(directory, users, items, kind):
    """Independently check adapter algebra against the selected raw predictions."""
    filename, key = ('selected-probabilities.npz', 'probabilities') if kind == 'conditional' else ('selected-logits.npz', 'logits')
    with np.load(directory/filename, allow_pickle=False) as archive:
        if archive['users'].tolist() != users or archive['items'].tolist() != items:
            raise ValueError('Selected field prediction IDs differ')
        values = archive[key]
    if values.shape != (len(users), len(items), 5) or not np.isfinite(values).all():
        raise ValueError('Invalid selected categorical predictions')
    if kind == 'conditional':
        if np.any(values < 0) or not np.allclose(values.sum(axis=-1), 1., atol=2e-6, rtol=0):
            raise ValueError('Invalid conditional probabilities')
        expected = {directory: values[:, :, 3:].sum(axis=-1)}
    else:
        logits = values.astype(float)
        expected = {directory/'all_observed_adapter': logsumexp(logits, axis=-1),
                    directory/'liked_record_adapter': logsumexp(logits[:, :, 3:], axis=-1)}
    for path, scores in expected.items():
        with np.load(path/'valid-scores.npz', allow_pickle=False) as archive:
            if not np.array_equal(archive['scores'], scores):
                raise ValueError(f'Field adapter algebra differs: {path}')


def training_frequency(recs, chosen_users, counts, catalog):
    """Slot-weighted diagnostics on ALL development users, never a chosen cohort."""
    head_size = int(np.ceil(.2*len(catalog)))
    catalog_set = set(catalog)
    head = set(sorted(catalog, key=lambda item: (-counts.get(item, 0), item))[:head_size])
    recommended = [item for user in sorted(chosen_users) for item in recs[user]]
    frequency = np.asarray([counts.get(item, 0) for item in recommended], dtype=float)
    if not len(frequency) or any(item not in catalog_set for item in recommended):
        raise ValueError('Invalid recommendation-frequency input')
    return {'users': len(chosen_users), 'recommendation_slots': len(recommended),
            'mean_training_count': float(frequency.mean()), 'median_training_count': float(np.median(frequency)),
            'zero_training_count_slot_share': float(np.mean(frequency == 0)),
            'head_slot_share': float(np.mean([item in head for item in recommended])),
            'head_catalog_size': head_size, 'catalog_items': len(catalog),
            'head_definition': 'Top ceil(20% of nonpadding catalog) by observed-TRAIN count; original item-token lexical tie-break.',
            'training_count_definition': 'Every observed original TRAIN user-item pair, irrespective of rating category.'}


def reference_path(root, seed, model):
    choice_file = root/f'expert-selection-{seed}.json'
    choice = json.loads(choice_file.read_text())[model]['selected']
    if 'meta_fit_ndcg' not in choice or choice.get('selection_cohort', 'meta_fit') != 'meta_fit':
        raise ValueError('Reference was not selected by the recorded meta-fit criterion')
    path = Path(choice['path'])
    if not path.exists():
        path = root/path.name
    manifest = json.loads((path/'manifest.json').read_text())
    if any(manifest['settings'].get(key) != value for key, value in choice['settings'].items()):
        raise ValueError('Locked selection settings differ from fitted reference')
    return path, {'selection_file_sha256': digest(choice_file), 'settings': choice['settings'],
                  'selection_metric': 'meta-fit all-observed nDCG@10', 'selection_cohort': 'meta_fit',
                  'selection_value': choice['meta_fit_ndcg']}


def run_seed(args, seed):
    ease, ease_selection = reference_path(args.reference_root, seed, 'EASE')
    users, items, train, _, training, validation, _, _, reference = load_source(ease, args.ratings, args.item_metadata)
    fit_users, dev_users = partition_users(users, seed)
    field_cohorts = json.loads((args.cohort_root/str(seed)/'cohorts.json').read_text())
    if set(field_cohorts['meta_fit']) != fit_users or set(field_cohorts['development']) != dev_users:
        raise ValueError('Reference partition differs from recorded field cohorts')
    if len(field_cohorts['meta_fit']) != len(fit_users) or len(field_cohorts['development']) != len(dev_users):
        raise ValueError('Duplicate field cohort users')
    # Validate the field's copied pair files even in references-only mode while
    # field training is still running. The cohort root is never mutated.
    for part in ('train', 'valid'):
        if digest(args.cohort_root/str(seed)/f'{part}.tsv') != reference['split_sha256'][part]:
            raise ValueError('Reference/field pair identity mismatch')
    ids_hash = digest(ease/'ids.json')
    _, observed = rating_matrices(users, items, training)
    history, counts = grouped(train), Counter(item for _, item in train)
    positive_path = args.positive_root/str(seed)/'positive_ease-liked_ratings'
    if not positive_path.exists():
        positive_path = args.positive_root/str(seed)/'positive_ease-all_observed'
    slim_root = args.slim_root if seed == 2026 else args.slim_more_root
    slim, slim_selection = reference_path(slim_root, seed, 'SLIMElastic')
    rows, detailed, provenance = {}, {}, {}

    def evaluate_row(name, paths, selection, shared_directory=None):
        readouts, raw = {}, {}
        for endpoint, directory in paths.items():
            scores, manifest, signatures = load_scores(directory, reference, users, items, ids_hash, shared_directory)
            if selection is None:
                selection = manifest['selection']
            if selection.get('selection_cohort') != 'meta_fit':
                raise ValueError('Comparison requires a locked meta-fit selection')
            recs = recommendations(scores, users, items, observed, 10)
            if any(set(recs[u]) & history[u] for u in dev_users):
                raise AssertionError('Observed TRAIN item entered ranking')
            metrics = evaluate_ratings(recs, dev_users, validation, history, items[1:], counts, 10)
            readouts[endpoint] = {'metrics': metrics[endpoint]['aggregate'],
                                  'denominators': metrics['denominators'],
                                  'known_dislike_rate_per_slot': metrics['known_dislike_rate_per_slot'],
                                  'training_frequency': training_frequency(recs, dev_users, counts, items[1:]),
                                  'score_space': signatures['score_space']}
            raw[endpoint] = {'metrics': metrics, 'recommendations': {u: recs[u] for u in sorted(dev_users)}}
            provenance.setdefault(name, {})[endpoint] = signatures
        rows[name] = {'selection': selection, 'endpoints': readouts}
        detailed[name] = raw

    evaluate_row('EASE', dict.fromkeys(ENDPOINTS, ease), ease_selection)
    evaluate_row('SLIMElastic', dict.fromkeys(ENDPOINTS, slim), slim_selection)
    evaluate_row('PositiveEASE', dict.fromkeys(ENDPOINTS, positive_path), None)
    field_manifests = {}
    if args.conditional_root:
        directory = args.conditional_root/str(seed)
        field_manifests['conditional'] = verify_field_seed(directory, reference, users, items, fit_users, dev_users)
        selections = json.loads((directory/'selection.json').read_text())
        for variant in ('adaptive', 'fixed_flow', 'hard_clamp'):
            verify_field_readout(directory/variant, users, items, 'conditional')
            evaluate_row('Conditional-'+variant, dict.fromkeys(ENDPOINTS, directory/variant), selections[variant], directory)
    if args.joint_root:
        directory = args.joint_root/str(seed)
        field_manifests['joint'] = verify_field_seed(directory, reference, users, items, fit_users, dev_users)
        selections = json.loads((directory/'selection.json').read_text())
        for variant in ('adaptive', 'fixed_flow'):
            verify_field_readout(directory/variant, users, items, 'joint')
            evaluate_row('Joint-'+variant, {
                'all_observed': directory/variant/'all_observed_adapter',
                'liked_ratings': directory/variant/'liked_record_adapter'}, selections[variant])
    # Recommendation-dependent denominators differ; fixed label/cohort counts
    # must match exactly across every readout.
    fixed_keys = ('all_observed_users', 'liked_ratings_users', 'validation_observations',
                  'validation_likes', 'validation_dislikes', 'recommendation_slots')
    denominator_signatures = {tuple(result['denominators'][key] for key in fixed_keys)
                             for row in rows.values() for result in row['endpoints'].values()}
    if len(denominator_signatures) != 1:
        raise AssertionError('Comparison endpoints have different label/cohort denominators')
    out = args.out/str(seed)
    out.mkdir()
    write_json(out/'details.json', detailed)
    aggregate = {'seed': seed, 'rows': rows, 'provenance': provenance,
                 'cohorts': {'meta_fit': len(fit_users), 'development': len(dev_users)},
                 'ordered_identity_sha256': identity_digest(users, items),
                 'split_sha256': {part: reference['split_sha256'][part] for part in ('train', 'valid')},
                 'data_sha256': reference['data_sha256'],
                 'field_manifest_sha256': {kind: digest((args.conditional_root if kind == 'conditional' else args.joint_root)/str(seed)/'manifest.json')
                                            for kind in field_manifests},
                 'cohort_file_sha256': digest(args.cohort_root/str(seed)/'cohorts.json'),
                 'details_sha256': digest(out/'details.json')}
    write_json(out/'aggregates.json', aggregate)
    print(f'{seed}: matched {len(rows)} locked rows on {len(dev_users)} development users', flush=True)
    return aggregate


def curate(results, plan, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', plan)
    write_json(output/'aggregates.json', {'protocol': plan, 'seeds': results})
    seeds = list(results)
    models = list(results[seeds[0]]['rows'])
    if any(list(results[s]['rows']) != models for s in seeds):
        raise ValueError('Model rows differ across seeds')
    lines = ['# Locked-reference comparison for the field models', '',
             'Every row uses the field study\'s same 472 development users, original catalog, and full observed-TRAIN candidate mask. '
             'Liked metrics exclude only users without a held-out rating >=4; denominators are reported below. '
             'These are exploratory comparisons on reused validation users, with previously locked parameters and checkpoints. No new tuning or test access.', '',
             'EASE and SLIM were selected on meta-fit all-observed nDCG@10. PositiveEASE uses its recorded selection objective below. '
             'Conditional fields select categorical cross-entropy and rank by P(rating>=4). Joint fields select joint recorded-event likelihood; '
             'their all-observed endpoint ranks logsumexp of five categories, and their liked endpoint ranks logsumexp of categories4/5. '
             'Training objectives and tuning budgets differ, so this is a practical locked-reference comparison, not an isolated architecture ablation.', '']
    for endpoint in ENDPOINTS:
        lines += [f'## {endpoint} nDCG@10', '', '| Model | '+' | '.join(seeds)+' | Mean |',
                  '|---|'+'---:|'*(len(seeds)+1)]
        for name in models:
            values = [results[s]['rows'][name]['endpoints'][endpoint]['metrics']['ndcg@10'] for s in seeds]
            lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+f' | {np.mean(values):.4f} |')
        lines += ['', f'## {endpoint} recommendation frequency', '',
                  'Every frequency row uses all 472 development users and 4720 recommendation slots, including users without a held-out like.', '',
                  '| Model | Seed | Mean TRAIN count | Median TRAIN count | Zero-count slots | Head slots |',
                  '|---|---|---:|---:|---:|---:|']
        for name in models:
            for seed in seeds:
                f = results[seed]['rows'][name]['endpoints'][endpoint]['training_frequency']
                lines.append(f"| {name} | {seed} | {f['mean_training_count']:.2f} | {f['median_training_count']:.1f} | "
                             f"{f['zero_training_count_slot_share']:.2%} | {f['head_slot_share']:.2%} |")
        lines.append('')
    lines += ['## Denominators and selection', '',
              '| Seed | Development users | Users with validation likes | Validation observations | Validation likes | PositiveEASE selection |',
              '|---|---:|---:|---:|---:|---|']
    for seed in seeds:
        d = results[seed]['rows']['EASE']['endpoints']['all_observed']['denominators']
        selection = results[seed]['rows']['PositiveEASE']['selection']
        lines.append(f"| {seed} | {d['all_observed_users']} | {d['liked_ratings_users']} | {d['validation_observations']} | "
                     f"{d['validation_likes']} | {selection.get('selection_objective', selection.get('selection_metric'))}, ridge {selection.get('penalty')} |")
    lines += ['',
              'TRAIN count includes every observed training pair, irrespective of rating. The head is the top ceil(20% of the nonpadding catalog) '
              'by that count, with original item-token lexical tie breaking. Zero-count means no original training observation for that item. '
              'These are descriptive exposure diagnostics; they do not identify why a model ranks an item or establish a causal failure mechanism.', '',
              'All dataset, train/validation pair, internal-ID file, ordered score-ID and field-output hashes were checked. Older EASE/SLIM manifests '
              'did not pin prediction-file hashes originally; this comparison records their current score-file hashes and verifies their locked selection '
              'records and common data/ID mappings. Newer exports also verify their pre-existing score hashes. Raw per-user metrics and recommendation lists '
              'remain in ignored runs/; this directory contains aggregate-only evidence.', '',
              'Full Recall, Precision, MRR, HitRate, coverage, novelty and known-dislike measures are retained in `aggregates.json`, alongside explicit '
              'readout descriptions, selection settings and provenance. No results from the smaller 236-user main-study cohort are reused.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=Path('runs/research-v2'))
    parser.add_argument('--slim-root', type=Path, default=Path('runs/coverage-v1'))
    parser.add_argument('--slim-more-root', type=Path, default=Path('runs/coverage-v1-more'))
    parser.add_argument('--positive-root', type=Path, default=Path('runs/exception-v2'))
    parser.add_argument('--cohort-root', type=Path, default=Path('runs/adaptive-v1'))
    parser.add_argument('--conditional-root', type=Path)
    parser.add_argument('--joint-root', type=Path)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--item-metadata', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate seeds')
    if 'runs' not in args.out.parts:
        parser.error('Detailed output must be under ignored runs/')
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {name: digest(Path(__file__).parent/name) for name in SOURCES}
    plan = {'status': 'written_before_comparison', 'stage': 'validation_development', 'seeds': args.seeds,
            'test_read': False, 'test_evaluated': False, 'training_or_selection_performed': False,
            'references': ['EASE', 'SLIMElastic', 'PositiveEASE'],
            'field_tracks': [kind for kind, path in (('conditional', args.conditional_root), ('joint', args.joint_root)) if path],
            'metrics': ['all_observed', 'liked_ratings'], 'k': 10,
            'frequency_cohort': 'all development users, no outcome-conditioned subgroup',
            'frequency_diagnostics': ['slot mean training count', 'slot median training count', 'zero-training-count slot share', 'head slot share'],
            'head_definition': 'Top ceil(20% of nonpadding catalog) by observed-TRAIN count; original item-token lexical tie-break.',
            'source_sha256': hashes,
            'scope': 'Exploratory comparison of already selected models; reused validation users, unequal objectives and tuning budgets, no causal interpretation.'}
    write_json(args.out/'protocol.json', plan)
    results = {str(seed): run_seed(args, seed) for seed in args.seeds}
    if hashes != {name: digest(Path(__file__).parent/name) for name in SOURCES}:
        raise RuntimeError('Comparison source changed during evaluation')
    curate(results, plan, args.evidence)
    write_json(args.out/'manifest.json', {'status': 'complete', 'source_sha256': hashes,
        'test_read': False, 'test_evaluated': False,
        'evidence_sha256': {p.name: digest(p) for p in args.evidence.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
