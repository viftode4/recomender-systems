"""Predeclared paired comparisons of completed, frozen evaluation artifacts.

This analysis never reads test interactions, generates predictions, or chooses a
test winner. Candidate/reference pairs come exclusively from freeze manifests.
Exported evidence contains aggregate metrics and intervals, never user IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LIMITATION = (
    'Descriptive paired percentile-bootstrap intervals conditional on the frozen '
    'models and this dataset; approximate simultaneous coverage is adjusted within '
    'each seed, not across seeds. Repeated splits share users and items, and one '
    'MovieLens dataset does not establish population-level or cross-dataset gains. '
    'No model or parameter is selected using these evaluation outcomes.'
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def predefined_comparisons(bundle):
    """Only frozen metadata determines the tested candidate/reference pairs."""
    selection = bundle['selection']
    families = selection['families']
    if 'constrained' not in families or 'context' not in families:
        raise ValueError('Frozen constrained and context families are required')
    reference = selection['best_expert']
    comparisons = {f'hybrid:{family}': {'candidate': candidate, 'reference': reference}
                   for family, candidate in sorted(families.items())}
    for fixed in ('rrf', 'group-switch'):
        if fixed in bundle['models']:
            comparisons[f'hybrid:{fixed}'] = {'candidate': fixed, 'reference': reference}
    comparisons['policy:group-utility-budget-exposure'] = {
        'candidate': 'group-utility-budget-exposure', 'reference': families['context']}
    for pair in comparisons.values():
        if pair['candidate'] not in bundle['models'] or pair['reference'] not in bundle['models']:
            raise ValueError('Frozen comparison references an absent model')
    return comparisons


def paired_bootstrap(differences, *, resamples=20000, confidence=.95, seed=2026):
    """Bootstrap user rows jointly; columns are all predeclared comparisons.

    Shared resampled row indices preserve candidate/reference pairing and the
    dependence between comparisons. Bonferroni-adjusted percentile endpoints
    remain approximate bootstrap bounds, not exact finite-sample guarantees.
    Memory stays bounded by processing at most 128 bootstrap draws at a time.
    """
    difference = np.asarray(differences, dtype=float)
    if difference.ndim == 1:
        difference = difference[:, None]
    if (difference.ndim != 2 or difference.shape[0] < 2 or difference.shape[1] < 1
            or not np.isfinite(difference).all()):
        raise ValueError('At least two paired users and finite comparisons required')
    if resamples < 20000 or not 0 < confidence < 1:
        raise ValueError('Require at least 20000 bootstrap draws and valid confidence')
    count = difference.shape[1]
    tail = (1 - confidence) / (2 * count)
    # Avoid reporting a far-tail quantile with almost no Monte Carlo resolution.
    resamples = max(int(resamples), int(np.ceil(20 / tail)))
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty((resamples, count))
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        bootstrapped[start:stop] = difference[indices].mean(axis=1)
    ordinary = np.quantile(bootstrapped, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
    simultaneous = np.quantile(bootstrapped, [tail, 1 - tail], axis=0)
    return {'mean_differences': difference.mean(axis=0).tolist(),
            'pointwise_intervals': ordinary.T.tolist(),
            'bonferroni_intervals': simultaneous.T.tolist(),
            'comparisons': count, 'users': len(difference), 'resamples': resamples,
            'confidence': confidence, 'bonferroni_tail_probability': tail,
            'expected_draws_per_adjusted_tail': resamples * tail}


def compare_one(bundle, results, *, resamples=20000, confidence=.95, seed=2026):
    """Analyze one completed seed; return aggregate-only comparison records."""
    comparisons = predefined_comparisons(bundle)
    metric = f"ndcg@{bundle['k']}"
    if bundle['selection']['metric'] != metric:
        raise ValueError('Frozen primary selection metric must match cutoff')
    if set(results) != set(bundle['models']):
        raise ValueError('Evaluation results must cover every frozen model exactly')
    users = None
    difference_columns = []
    records = {}
    for name, pair in comparisons.items():
        candidate, reference = results[pair['candidate']], results[pair['reference']]
        if users is None:
            users = sorted(candidate['per_user'])
        if set(users) != set(candidate['per_user']) or set(users) != set(reference['per_user']):
            raise ValueError('Every comparison must use the same paired users')
        candidate_values = np.asarray([candidate['per_user'][u][metric] for u in users], dtype=float)
        reference_values = np.asarray([reference['per_user'][u][metric] for u in users], dtype=float)
        for label, values, record in [('candidate', candidate_values, candidate), ('reference', reference_values, reference)]:
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f'{label} NDCG must be bounded in [0,1]')
            if not np.isclose(values.mean(), record['aggregate'][metric], atol=1e-12, rtol=1e-12):
                raise ValueError('Aggregate NDCG differs from paired user values')
        difference = candidate_values - reference_values
        difference_columns.append(difference)
        # Aggregate metrics contain no individual-user information.
        aggregate = {}
        for field, row in [('candidate_metrics', candidate['aggregate']), ('reference_metrics', reference['aggregate'])]:
            aggregate[field] = {}
            for key, value in row.items():
                if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value)):
                    raise ValueError('Aggregate metrics must be finite numbers or null')
                aggregate[field][key] = value
        records[name] = {**pair, **aggregate,
                         'fraction_users_improved': float(np.mean(difference > 0)),
                         'fraction_users_harmed': float(np.mean(difference < 0)),
                         'fraction_users_tied': float(np.mean(difference == 0))}
    intervals = paired_bootstrap(np.stack(difference_columns, axis=1), resamples=resamples,
                                 confidence=confidence, seed=seed)
    for index, name in enumerate(comparisons):
        records[name].update({'mean_ndcg_difference': intervals['mean_differences'][index],
                              'pointwise_interval': intervals['pointwise_intervals'][index],
                              'bonferroni_interval': intervals['bonferroni_intervals'][index]})
    return {'seed': int(bundle['seed']), 'metric': metric, 'comparisons': records,
            'bootstrap': {k: v for k, v in intervals.items()
                          if k not in ('mean_differences', 'pointwise_intervals', 'bonferroni_intervals')},
            'interpretation': LIMITATION}


def aggregate_comparisons(reports):
    """Descriptive means across splits, without treating seeds as independent."""
    if not reports or len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('Reports must contain distinct, nonempty split seeds')
    keys = list(reports[0]['comparisons'])
    if any(set(report['comparisons']) != set(keys) for report in reports):
        raise ValueError('All seeds must use the same predeclared comparison families')
    if len({r['metric'] for r in reports}) != 1:
        raise ValueError('Primary metric differs across seeds')
    metric = reports[0]['metric']
    output = {}
    for key in keys:
        records = [r['comparisons'][key] for r in reports]
        result = {}
        for field in ('candidate_metrics', 'reference_metrics'):
            if any(set(row[field]) != set(records[0][field]) for row in records):
                raise ValueError('Aggregate metric columns differ across seeds')
            result[f'mean_{field}'] = {}
            result[f'metric_seed_counts_{field}'] = {}
            for name in records[0][field]:
                values = [row[field][name] for row in records if row[field][name] is not None]
                result[f'mean_{field}'][name] = float(np.mean(values)) if values else None
                result[f'metric_seed_counts_{field}'][name] = len(values)
        differences = [row['mean_ndcg_difference'] for row in records]
        candidate_mean = result['mean_candidate_metrics'][metric]
        reference_mean = result['mean_reference_metrics'][metric]
        result.update({'seeds': len(records), 'mean_ndcg_difference': float(np.mean(differences)),
                       'split_sd_ndcg_difference': float(np.std(differences, ddof=1)) if len(records) > 1 else None,
                       'relative_ndcg_difference_percent': 100 * (candidate_mean / reference_mean - 1) if reference_mean > 0 else None,
                       'per_seed_models': {str(report['seed']): {
                           'candidate': report['comparisons'][key]['candidate'],
                           'reference': report['comparisons'][key]['reference']} for report in reports}})
        output[key] = result
    return {'metric': metric, 'seeds': [r['seed'] for r in reports],
            'comparisons': output, 'interpretation': LIMITATION,
            'cross_seed_confidence_interval': None,
            'cross_seed_interval_note': 'Not computed: the repeated splits are not independent samples.'}


def load_completed(freeze_path, result_path):
    """Verify input artifacts without importing inference or reading split data."""
    freeze_path, result_path = Path(freeze_path), Path(result_path)
    freeze_file, manifest_file = freeze_path / 'freeze.json', result_path / 'manifest.json'
    results_file = result_path / 'results.json'
    freeze_hash = digest(freeze_file)
    if freeze_hash != (freeze_path / 'freeze.sha256').read_text().strip():
        raise ValueError('Frozen manifest hash mismatch')
    bundle, manifest = read_json(freeze_file), read_json(manifest_file)
    if bundle.get('status') != 'frozen' or bundle.get('test_read') is not False:
        raise ValueError('Expected a manifest frozen before evaluation')
    if manifest.get('status') != 'complete' or manifest.get('test_model_selection') is not False:
        raise ValueError('Expected completed evaluation without test selection')
    if manifest.get('freeze_sha256') != freeze_hash:
        raise ValueError('Evaluation belongs to a different frozen bundle')
    if manifest.get('test_sha256') != bundle.get('expected_test_sha256'):
        raise ValueError('Evaluation split hash differs from frozen expectation')
    if manifest.get('k') != bundle['k']:
        raise ValueError('Evaluation cutoff differs from frozen cutoff')
    if manifest.get('output_sha256', {}).get('results.json') != digest(results_file):
        raise ValueError('Results hash differs from completed evaluation manifest')
    provenance = {'freeze_directory': freeze_path.name, 'evaluation_directory': result_path.name,
                  'freeze_sha256': freeze_hash, 'evaluation_manifest_sha256': digest(manifest_file),
                  'results_sha256': digest(results_file), 'test_interactions_opened': False}
    return bundle, read_json(results_file), provenance


def write_report(freeze_paths, result_paths, out, *, resamples=20000, confidence=.95,
                 seed=2026):
    """Create a new shareable evidence directory from fixed evaluation artifacts."""
    if not freeze_paths or len(freeze_paths) != len(result_paths):
        raise ValueError('Provide paired freeze/evaluation directories')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    reports, provenance = [], []
    for frozen, evaluated in zip(freeze_paths, result_paths):
        bundle, results, inputs = load_completed(frozen, evaluated)
        # Seed generation is independent of metrics and directory argument order.
        rng_seed = np.random.SeedSequence([seed, int(bundle['seed'])])
        reports.append(compare_one(bundle, results, resamples=resamples,
                                   confidence=confidence, seed=rng_seed))
        provenance.append(inputs)
    reports.sort(key=lambda r: r['seed'])
    aggregate = aggregate_comparisons(reports)
    out.mkdir(parents=True, exist_ok=False)
    def dump(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    dump(out / 'aggregate.json', aggregate)
    dump(out / 'per-seed-comparisons.json', reports)
    lines = ['# Frozen paired comparisons', '', LIMITATION, '',
             'Candidate and reference models were specified in freeze manifests. Rows are not ordered by evaluation performance.', '',
             f"Primary metric: {aggregate['metric']}. Splits: {', '.join(map(str, aggregate['seeds']))}.", '',
             '| Predeclared comparison | Mean candidate | Mean reference | Mean difference | Relative change |',
             '|---|---:|---:|---:|---:|']
    for name, result in aggregate['comparisons'].items():
        relative = result['relative_ndcg_difference_percent']
        relative_text = 'n/a' if relative is None else f'{relative:+.2f}%'
        lines.append(f"| {name} | {result['mean_candidate_metrics'][aggregate['metric']]:.5f} | "
                     f"{result['mean_reference_metrics'][aggregate['metric']]:.5f} | "
                     f"{result['mean_ndcg_difference']:+.5f} | {relative_text} |")
    lines += ['', f'Per-seed paired intervals use confidence {confidence:.1%}, with Bonferroni adjustment across all declared comparisons within each seed.', '',
              '| Split | Comparison | Candidate | Reference | Mean difference | Adjusted interval |',
              '|---|---|---|---|---:|---|']
    for report in reports:
        for name, result in report['comparisons'].items():
            low, high = result['bonferroni_interval']
            lines.append(f"| {report['seed']} | {name} | {result['candidate']} | {result['reference']} | "
                         f"{result['mean_ndcg_difference']:+.5f} | [{low:+.5f}, {high:+.5f}] |")
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    manifest = {'status': 'complete', 'inputs': provenance,
                'analysis_sha256': digest(Path(__file__)), 'numpy': np.__version__,
                'bootstrap_random_seed': seed, 'bootstrap_requested_resamples': resamples,
                'confidence': confidence, 'test_model_selection': False,
                'test_interactions_opened': False, 'individual_user_data_exported': False,
                'output_sha256': {p.name: digest(p) for p in sorted(out.iterdir())},
                'interpretation': LIMITATION}
    dump(out / 'manifest.json', manifest)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', type=Path, required=True)
    parser.add_argument('--freezes', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    result = write_report(args.freezes, args.results, args.out, resamples=args.bootstrap,
                          confidence=args.confidence, seed=args.seed)
    print(json.dumps({'out': str(args.out), 'splits': result['seeds'],
                      'comparisons': len(result['comparisons']), 'individual_user_data_exported': False}))


if __name__ == '__main__':
    main()
