"""Reproduce the meta-fit-only scale diagnostic motivating affine calibration.

Run with the project's NumPy/RecBole environment. This parses the saved training
and validation partitions, uses relevance labels only for meta-fit users, and
never reads test interactions or computes development/test metrics. Output is
aggregate-only: identifiers and sampled observation rows are not exported.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hybrid_constraints import fit_score_calibration, fit_sum_to_one_ridge
from study import digest, fit_ridge, grouped, load_runs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Refusing to overwrite a previous audit')
    study = args.study.resolve()
    manifest = json.loads((study/'manifest.json').read_text())
    assembly = json.loads((study/'assembly.json').read_text())
    cohorts = json.loads((study/'cohorts.json').read_text())
    selected = json.loads((study/'selection.json').read_text())
    coefficients = json.loads((study/'coefficients.json').read_text())
    if manifest['status'] != 'complete' or manifest['test_read']:
        raise ValueError('Requires a completed validation-only study')
    paths = [Path(path) for path in assembly['runs']]
    users, items, experts, train, valid, sources = load_runs(paths)
    if sources != manifest['sources']:
        raise ValueError('Expert sources differ from study provenance')
    fitting_users = set(cohorts['meta_fit'])
    if fitting_users & (set(cohorts['development']) | set(cohorts.get('calibration', []))):
        raise ValueError('Meta-fit users overlap held-out selection/calibration users')
    history, truth = grouped(train), grouped(valid)
    index = {item: i for i, item in enumerate(items)}
    eligible = [np.asarray([i for i, item in enumerate(items) if i and item not in history[user]])
                for user in users]
    rng = np.random.default_rng(manifest['seed'])
    rows, columns, labels = [], [], []
    ratio = manifest['negative_ratio']
    for u, user in enumerate(users):
        if user not in fitting_users:
            continue
        positive = np.asarray(sorted(index[item] for item in truth[user]))
        available = np.setdiff1d(eligible[u], positive)
        negative = rng.choice(available, min(len(available), len(positive)*ratio), replace=False)
        observations = np.concatenate([positive, negative])
        rows.extend([u]*len(observations))
        columns.extend(observations)
        labels.extend([1.]*len(positive) + [0.]*len(negative))
        # Replay the shared RNG consumed by study.py's pairwise observations.
        for _ in positive:
            rng.choice(available, min(len(available), ratio), replace=False)
    rows, columns, target = np.asarray(rows), np.asarray(columns), np.asarray(labels)
    names = [name for name in experts if sources[name]['manifest']['model'] != 'Random']
    features = []
    for name in names:
        scores = experts[name]
        means = np.asarray([scores[u, candidates].mean() for u, candidates in enumerate(eligible)])
        scales = np.asarray([max(scores[u, candidates].std(), 1e-8) for u, candidates in enumerate(eligible)])
        features.append((scores[rows, columns] - means[rows])/scales[rows])
    raw = np.stack(features, axis=1)
    slopes, offsets = fit_score_calibration(raw, target)
    calibrated = raw*slopes + offsets
    checks = {}
    for family, fit in [('static', fit_ridge), ('constrained', fit_sum_to_one_ridge)]:
        stored = coefficients[selected['families'][family]]
        weights, intercept = fit(raw, target, stored['penalty'])
        if (not np.allclose(weights, [stored['weights'][name] for name in names], atol=1e-12, rtol=1e-10)
                or not np.isclose(intercept, stored['intercept'], atol=1e-12, rtol=1e-10)):
            raise ValueError('Meta-fit observation reproduction differs from saved coefficients')
        checks[family] = True
    diagnostics = []
    for name, x, fit, penalties in [
            ('raw_sum_to_one', raw, fit_sum_to_one_ridge, [1e-6, .0001, .001, .01, .1, 1.]),
            ('affine_calibrated_sum_to_one', calibrated, fit_sum_to_one_ridge, [.001, .01, .1, 1.]),
            ('raw_unconstrained', raw, fit_ridge, [1.])]:
        for penalty in penalties:
            weights, intercept = fit(x, target, penalty)
            prediction = x@weights + intercept
            diagnostics.append({'variant': name, 'penalty': penalty,
                'meta_fit_mse': float(np.mean((prediction-target)**2)),
                'prediction_std': float(prediction.std()), 'weight_sum': float(weights.sum()),
                'negative_weights': int((weights < 0).sum()),
                'largest_weight_expert': names[int(weights.argmax())],
                'largest_weight': float(weights.max())})
    report = {'seed': manifest['seed'], 'study_name': study.name,
        'scope': 'Meta-fit regression diagnostic only; no development or test metric computed.',
        'motivation': 'Development inspection raised a scale-confounding concern for sum-to-one weights on z-scores. This exploratory diagnostic motivated retaining that baseline and adding monotone affine response calibration before constrained regression.',
        'cohort_users': len(fitting_users), 'observations': len(target), 'negative_ratio': ratio,
        'target_mean': float(target.mean()), 'target_std': float(target.std()),
        'expert_feature_std_range': [float(raw.std(0).min()), float(raw.std(0).max())],
        'calibration': {name: {'slope': float(a), 'offset': float(b)}
                        for name, a, b in zip(names, slopes, offsets)},
        'diagnostics': diagnostics, 'saved_coefficient_reproduction': checks,
        'study_artifact_sha256': {name: digest(study/name) for name in
            ('manifest.json', 'assembly.json', 'cohorts.json', 'selection.json', 'coefficients.json')},
        'original_study_source_sha256': manifest['study_source_sha256'],
        'audit_source_sha256': digest(Path(__file__)),
        'calibration_source_sha256': digest(ROOT/'hybrid_constraints.py'),
        'source_runs': {path.name: {'manifest_sha256': digest(path/'manifest.json'),
            'scores_sha256': sources[path.name]['scores_sha256'],
            'split_sha256': {part: sources[path.name]['manifest']['split_sha256'][part]
                             for part in ('train', 'valid')}} for path in paths},
        'limitations': ['Meta-fit loss is not ranking accuracy or generalization evidence.',
                       'Affine responses are unbounded and are not calibrated probabilities.',
                       'Sampled-negative prevalence affects the transform.',
                       'Calibration and combination coefficients reuse meta-fit labels; final evaluation remains necessary.'],
        'test_labels_read': False, 'development_metrics_computed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps({'out': str(args.out), 'coefficient_reproduction': checks,
                      'observations': len(target)}, indent=2))


if __name__ == '__main__':
    main()
