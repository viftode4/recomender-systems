"""Fixed paired comparisons of sealed final field and strong-reference outputs.

Reads manifests, frozen metadata and completed per-user metrics only. It never
opens interaction files, rating values, model scores or checkpoints, and never
fits or selects a model. Output is aggregate-only and safe to share.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TRACKS = ('conditional', 'joint100', 'joint400')
REFERENCES = ('EASE', 'SLIMElastic', 'PositiveEASE')
ENDPOINTS = ('all_observed', 'liked_ratings')
ADAPTERS = {'all_observed': 'all_observed_adapter', 'liked_ratings': 'liked_record_adapter'}
DENOMINATORS = ('all_observed_users', 'liked_ratings_users', 'test_observations',
                'test_likes', 'test_dislikes', 'recommendation_slots')
LIMITATION = (
    'Descriptive paired user-bootstrap percentile intervals conditional on the '
    'frozen models and this MovieLens dataset. Bonferroni adjustment covers all '
    '24 predeclared nDCG comparisons within each seed, not across seeds; bootstrap '
    'coverage is approximate. Repeated splits share users and items. Cross-seed '
    'means have no pooled confidence interval. Objectives and tuning budgets '
    'differ across model families, so reference comparisons do not isolate '
    'architecture. No direct joint400-minus-joint100 contrast is included; '
    'comparing their reference intervals does not test a budget effect. '
    'No model is selected using these test outcomes.'
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load_completed(frozen, results, kind):
    """Verify hash chains without opening any split, score or checkpoint file."""
    frozen, results = Path(frozen), Path(results)
    if kind not in (*TRACKS, 'references'):
        raise ValueError('Unknown frozen experiment kind')
    freeze_hash = digest(frozen / 'manifest.json')
    if freeze_hash != (frozen / 'manifest.sha256').read_text().strip():
        raise ValueError('Frozen manifest hash mismatch')
    manifest = read_json(frozen / 'manifest.json')
    evaluated = read_json(results / 'manifest.json')
    marker = read_json(frozen / 'TEST-OPENED.json')
    if (manifest.get('status') != 'frozen' or manifest.get('test_read') is not False
            or manifest.get('test_evaluated') is not False):
        raise ValueError('Expected metadata frozen before test access')
    if (evaluated.get('status') != 'complete' or evaluated.get('test_read') is not True
            or evaluated.get('test_evaluated') is not True
            or evaluated.get('selection_after_test') is not False
            or evaluated.get('frozen_manifest_sha256') != freeze_hash
            or not isinstance(manifest.get('code_sha256'), dict) or not manifest['code_sha256']
            or evaluated.get('code_sha256') != manifest.get('code_sha256')):
        raise ValueError('Expected completed evaluation of these exact frozen choices')
    if (marker.get('status') != 'test_evaluated'
            or marker.get('evaluation_manifest_sha256') != digest(results / 'manifest.json')):
        raise ValueError('Completed test marker does not seal this evaluation manifest')
    seeds = manifest['seeds']
    if (not seeds or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Frozen seeds must be distinct integers')
    output = {}
    for seed in seeds:
        bundle_path = frozen / str(seed) / 'bundle.json'
        metrics_path = results / str(seed) / 'metrics.json'
        if digest(bundle_path) != manifest['bundle_sha256'][str(seed)]:
            raise ValueError('Frozen per-seed bundle hash mismatch')
        if digest(metrics_path) != evaluated.get('output_sha256', {}).get(f'{seed}/metrics.json'):
            raise ValueError('Completed per-seed metrics hash mismatch')
        bundle, metrics = read_json(bundle_path), read_json(metrics_path)
        if (bundle['seed'] != seed or bundle.get('test_read') is not False
                or bundle.get('test_evaluated') is not False or bundle['k'] != 10
                or bundle['expected_test_sha256'] != bundle['split_sha256']['test']):
            raise ValueError('Inconsistent frozen per-seed protocol')
        if set(metrics) != set(bundle['models']):
            raise ValueError('Final metrics must cover every frozen model exactly')
        required = set(REFERENCES) if kind == 'references' else {'adaptive', 'fixed_flow'}
        if not required.issubset(metrics) or (kind == 'references' and set(metrics) != required):
            raise ValueError('Required predeclared models are missing')
        if not required.issubset(bundle['selections']):
            raise ValueError('Required frozen model selections are missing')
        if any(bundle['selections'][name].get('selection_cohort') != 'meta_fit' for name in required):
            raise ValueError('Required models were not selected on the frozen meta-fit cohort')
        if (kind == 'references' and
                bundle['selections']['PositiveEASE'].get('selection_objective') != 'liked_ratings'):
            raise ValueError('PositiveEASE must be the predeclared liked-selected reference')
        if kind != 'references':
            protocol_path = frozen / str(seed) / 'protocol.json'
            if (digest(protocol_path) != bundle['source_protocol_sha256']
                    or digest(protocol_path) != bundle['payload_sha256']['protocol.json']):
                raise ValueError('Frozen field protocol hash mismatch')
            protocol = read_json(protocol_path)
            objective = ('categorical_cross_entropy' if kind == 'conditional'
                         else 'joint_recorded_item_rating_multinomial')
            epochs = 400 if kind == 'joint400' else 100
            if (protocol['training_objective'] != objective or protocol['epochs'] != epochs
                    or protocol['seeds'] != [seed]
                    or kind == 'joint400' and protocol.get('study_kind') != 'post_v1_convergence_sensitivity'):
                raise ValueError('Field track differs from the declared objective or epoch budget')
        output[seed] = {'bundle': bundle, 'metrics': metrics,
                        'provenance': {'bundle_sha256': digest(bundle_path),
                                       'metrics_sha256': digest(metrics_path)}}
    return {'kind': kind, 'seeds': output,
            'provenance': {'frozen_manifest_sha256': freeze_hash,
                           'evaluation_manifest_sha256': digest(results / 'manifest.json'),
                           'completed_marker_sha256': digest(frozen / 'TEST-OPENED.json')}}


def _ranking(row, kind, endpoint):
    if kind == 'references':
        return row
    if kind == 'conditional':
        return row['ranking']
    return row['ranking'][ADAPTERS[endpoint]]


def _validated_row(row, kind, endpoint, k):
    ranked = _ranking(row, kind, endpoint)
    if ranked.get('stage') != 'test':
        raise ValueError('Only explicitly test-stage ranking metrics are accepted')
    denominators = {name: ranked['denominators'][name] for name in DENOMINATORS}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in denominators.values()):
        raise ValueError('Ranking denominators must be nonnegative integers')
    if denominators['recommendation_slots'] != k * denominators['all_observed_users']:
        raise ValueError('Recommendation-slot denominator differs from frozen cutoff')
    all_rows = ranked['all_observed']['per_user']
    liked = ranked['liked_ratings']
    if liked is None:
        raise ValueError('At least two test users with liked ratings are needed')
    liked_rows = liked['per_user']
    if (len(all_rows) != denominators['all_observed_users']
            or len(liked_rows) != denominators['liked_ratings_users']
            or not set(liked_rows).issubset(all_rows)):
        raise ValueError('Ranking cohort does not match its stated denominator')
    selected = ranked[endpoint]
    users = sorted(selected['per_user'])
    if len(users) < 2:
        raise ValueError('At least two paired users are required per endpoint')
    aggregate = {}
    for metric in (f'ndcg@{k}', f'recall@{k}'):
        values = np.asarray([selected['per_user'][u][metric] for u in users], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError('Ranking metrics must be finite and bounded in [0,1]')
        mean = float(values.mean())
        if not np.isclose(mean, selected['aggregate'][metric], rtol=1e-12, atol=1e-12):
            raise ValueError('Ranking aggregate differs from its per-user values')
        aggregate[metric] = mean
    return users, selected['per_user'], aggregate, denominators


def _intervals(matrix, comparisons, *, resamples, seed):
    """Share resampled users within endpoint; Bonferroni needs no independence."""
    if not isinstance(resamples, int) or resamples < 20_000:
        raise ValueError('Use at least 20000 user-bootstrap draws')
    tail = .05 / (2 * comparisons)
    draws = max(resamples, int(np.ceil(20 / tail)))
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, matrix.shape[1]))
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        rows = rng.integers(0, len(matrix), (stop - start, len(matrix)))
        samples[start:stop] = matrix[rows].mean(axis=1)
    return {'pointwise': np.quantile(samples, [.025, .975], axis=0).T,
            'bonferroni': np.quantile(samples, [tail, 1-tail], axis=0).T,
            'draws': draws, 'adjusted_tail_probability': tail}


def _selection_summary(bundle, names):
    allowed = ('epoch', 'penalty', 'selection_objective', 'selection_metric',
               'selection_cohort')
    return {name: {key: bundle['selections'][name][key] for key in allowed
                   if key in bundle['selections'][name]} for name in names}


def compare_seed(experiments, seed, *, resamples=20_000, bootstrap_seed=2026):
    """Fixed 3 tracks x 4 pairs x 2 endpoints; returns no user identifiers."""
    if set(experiments) != set((*TRACKS, 'references')):
        raise ValueError('All three field experiments and locked references are required')
    rows = {name: experiment['seeds'][seed] for name, experiment in experiments.items()}
    reference_bundle = rows['references']['bundle']
    k = reference_bundle['k']
    if k != 10:
        raise ValueError('This predeclared report uses nDCG@10')
    for value in rows.values():
        bundle = value['bundle']
        for key in ('seed', 'k', 'data_sha256', 'split_sha256', 'expected_test_sha256'):
            if bundle[key] != reference_bundle[key]:
                raise ValueError(f'Experiments differ in frozen {key}')
    comparisons = {}
    denominator_reference = None
    endpoint_users = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        differences, keys = [], []
        cohort = None
        for track in TRACKS:
            candidate = rows[track]['metrics']['adaptive']
            for reference in ('fixed_flow', *REFERENCES):
                reference_kind = track if reference == 'fixed_flow' else 'references'
                other = rows[reference_kind]['metrics'][reference]
                users, candidate_per_user, candidate_aggregate, denominator = _validated_row(candidate, track, endpoint, k)
                other_users, reference_per_user, reference_aggregate, other_denominator = _validated_row(other, reference_kind, endpoint, k)
                if users != other_users or (cohort is not None and users != cohort):
                    raise ValueError('Every model must use identical paired users within an endpoint')
                cohort = users
                if (denominator != other_denominator or
                        denominator_reference is not None and denominator != denominator_reference):
                    raise ValueError('Label and recommendation denominators differ between models')
                denominator_reference = denominator
                delta = np.asarray([candidate_per_user[u]['ndcg@10'] - reference_per_user[u]['ndcg@10'] for u in users])
                key = f'{track}/{endpoint}/adaptive-minus-{reference}'
                comparisons[key] = {'track': track, 'endpoint': endpoint,
                    'candidate': 'adaptive', 'reference': reference, 'users': len(users),
                    'candidate_metrics': candidate_aggregate, 'reference_metrics': reference_aggregate,
                    'mean_ndcg_difference': float(delta.mean()),
                    'fraction_users_improved': float(np.mean(delta > 0)),
                    'fraction_users_harmed': float(np.mean(delta < 0)),
                    'fraction_users_tied': float(np.mean(delta == 0))}
                differences.append(delta)
                keys.append(key)
        endpoint_users[endpoint] = cohort
        intervals = _intervals(np.stack(differences, axis=1), 24, resamples=resamples,
                              seed=np.random.SeedSequence([bootstrap_seed, seed, endpoint_index]))
        for index, key in enumerate(keys):
            comparisons[key]['pointwise_interval_95'] = intervals['pointwise'][index].tolist()
            comparisons[key]['bonferroni_family_interval_95'] = intervals['bonferroni'][index].tolist()
    if not set(endpoint_users['liked_ratings']).issubset(endpoint_users['all_observed']):
        raise ValueError('Liked-record cohort must be a subset of all-observed cohort')
    return {'seed': seed, 'metric': 'ndcg@10', 'comparisons': comparisons,
            'denominators': denominator_reference,
            'selection': {name: _selection_summary(row['bundle'], REFERENCES if name == 'references'
                                                   else ('adaptive', 'fixed_flow')) for name, row in rows.items()},
            'bootstrap': {'confidence': .95, 'family_comparisons': 24,
                          'resamples_per_endpoint': intervals['draws'],
                          'adjusted_tail_probability': intervals['adjusted_tail_probability'],
                          'expected_draws_per_adjusted_tail': intervals['draws'] * intervals['adjusted_tail_probability'],
                          'paired_within_endpoint': True,
                          'cohort_note': 'All-observed and liked-record cohorts are resampled separately; shared user indices within each endpoint preserve pairing. Bonferroni does not require independent contrasts.'},
            'provenance': {name: row['provenance'] for name, row in rows.items()},
            'interpretation': LIMITATION}


def aggregate_reports(reports):
    if not reports or len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('Require nonempty reports with unique seeds')
    keys = list(reports[0]['comparisons'])
    if any(set(r['comparisons']) != set(keys) for r in reports):
        raise ValueError('Predeclared comparison families differ between seeds')
    comparisons = {}
    for key in keys:
        rows = [r['comparisons'][key] for r in reports]
        differences = [r['mean_ndcg_difference'] for r in rows]
        comparisons[key] = {'track': rows[0]['track'], 'endpoint': rows[0]['endpoint'],
            'reference': rows[0]['reference'], 'candidate': 'adaptive', 'seeds': len(rows),
            'mean_candidate_ndcg': float(np.mean([r['candidate_metrics']['ndcg@10'] for r in rows])),
            'mean_reference_ndcg': float(np.mean([r['reference_metrics']['ndcg@10'] for r in rows])),
            'mean_candidate_recall': float(np.mean([r['candidate_metrics']['recall@10'] for r in rows])),
            'mean_reference_recall': float(np.mean([r['reference_metrics']['recall@10'] for r in rows])),
            'mean_ndcg_difference': float(np.mean(differences)),
            'split_sd_ndcg_difference': float(np.std(differences, ddof=1)) if len(rows) > 1 else None}
    return {'seeds': [r['seed'] for r in reports], 'comparisons': comparisons,
            'cross_seed_confidence_interval': None,
            'recall_note': 'Recall means are descriptive secondary metrics, without interval-based claims.',
            'interpretation': LIMITATION}


def write_report(paths, out, *, resamples=20_000, bootstrap_seed=2026):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Refuse to overwrite comparison evidence')
    experiments = {kind: load_completed(*pair, kind) for kind, pair in paths.items()}
    if set(experiments) != set((*TRACKS, 'references')):
        raise ValueError('Provide all three predeclared tracks and the reference bundle')
    seeds = sorted(experiments['references']['seeds'])
    if any(set(value['seeds']) != set(seeds) for value in experiments.values()):
        raise ValueError('Frozen experiments must contain the same seed set')
    reports = [compare_seed(experiments, seed, resamples=resamples,
                            bootstrap_seed=bootstrap_seed) for seed in seeds]
    aggregate = aggregate_reports(reports)
    lines = ['# Frozen field models versus fixed flow and strong references', '',
             'Every comparison was fixed before test access: conditional field, joint field with a 100-epoch budget, '
             'and its separate 400-epoch convergence study, each against fixed flow, EASE, SLIMElastic and liked-selected PositiveEASE. '
             'All-observed relevance counts every test-rated item. Liked-record relevance counts only ratings at least four; '
             'users without a test like are excluded only from that endpoint.', '',
             'Conditional models use P(rating >= 4) for both endpoints. Joint models use total event mass for all-observed ranking '
             'and categories 4/5 event mass for liked-record ranking. References use their original frozen ranking. '
             'All rankings exclude the same TRAIN and validation items and retain frozen TRAIN-only model context.', '', LIMITATION, '',
             '| Track | Endpoint | Reference | Mean adaptive nDCG | Mean reference nDCG | Mean difference |',
             '|---|---|---|---:|---:|---:|']
    for row in aggregate['comparisons'].values():
        lines.append(f"| {row['track']} | {row['endpoint']} | {row['reference']} | {row['mean_candidate_ndcg']:.5f} | {row['mean_reference_ndcg']:.5f} | {row['mean_ndcg_difference']:+.5f} |")
    lines += ['', 'The means above weight the overlapping split seeds equally. No confidence interval is formed by treating these seeds as independent.', '',
              '| Seed | Track | Endpoint | Reference | Paired users | Difference | Adjusted 95% family interval |',
              '|---|---|---|---|---:|---:|---|']
    for report in reports:
        for row in report['comparisons'].values():
            low, high = row['bonferroni_family_interval_95']
            lines.append(f"| {report['seed']} | {row['track']} | {row['endpoint']} | {row['reference']} | {row['users']} | {row['mean_ndcg_difference']:+.5f} | [{low:+.5f}, {high:+.5f}] |")
    lines += ['', 'Unadjusted intervals, recall means, model choices, exact cohort/label denominators and input hashes are retained in JSON. '
              'No individual user IDs, recommendation lists, ratings or predictions are exported. '
              'This report reads completed metrics and sealed metadata only; it performs no test-label access, inference, refitting or selection.', '']
    out.mkdir(parents=True)
    write_json(out / 'aggregate.json', aggregate)
    write_json(out / 'per-seed-comparisons.json', reports)
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    write_json(out / 'manifest.json', {'status': 'complete', 'test_derived_metrics_read': True,
        'test_interactions_opened': False, 'inference_performed': False, 'selection_performed': False,
        'predeclared_tracks': list(TRACKS), 'predeclared_references': list(REFERENCES),
        'comparisons_per_seed': 24, 'bootstrap_seed': bootstrap_seed, 'seeds': seeds,
        'inputs': {name: value['provenance'] for name, value in experiments.items()},
        'code_sha256': digest(Path(__file__)),
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}})
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in (*TRACKS, 'references'):
        parser.add_argument('--' + kind, nargs=2, metavar=('FROZEN', 'RESULTS'), required=True, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    args = parser.parse_args()
    write_report({kind: getattr(args, kind) for kind in (*TRACKS, 'references')}, args.out,
                 resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)


if __name__ == '__main__':
    main()
