"""Check positive ElasticNet KKT residuals from a trusted local SLIM checkpoint.

Example (use the project's RecBole Python environment):
  python evidence/model-audits/audit_slim.py \
    --run runs/coverage-v1/2026-SLIMElastic-2 \
    --out evidence/model-audits/slim-seed2026.json

No validation/test labels or user/item identifiers appear in the output.
The checkpoint must be locally produced and trusted: torch.load uses pickle.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Refusing to overwrite an existing audit')
    manifest_path = args.run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    settings = manifest['settings']
    if manifest['model'] != 'SLIMElastic' or manifest['status'] != 'complete':
        raise ValueError('Requires a completed SLIMElastic run')
    if not settings['positive_only'] or not settings['hide_item']:
        raise ValueError('This audit checks the nonnegative, zero-diagonal variant only')
    if digest(args.run / 'train.tsv') != manifest['split_sha256']['train']:
        raise ValueError('Training split hash mismatch')
    checkpoint_path = args.run / manifest['checkpoint']
    if digest(checkpoint_path) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    state = checkpoint['other_parameter']
    x = state['interaction_matrix'].tocsr().astype(np.float64)
    weights = state['item_similarity'].toarray().astype(np.float64)
    alpha, ratio = settings['alpha'], settings['l1_ratio']
    # For each target j, minimize ||X b_j-X_j||^2/(2n)
    # + alpha*l1_ratio*sum(b_j) + alpha*(1-l1_ratio)*||b_j||^2/2,
    # subject to b_j >= 0 and b_j[j] = 0. The fixed diagonal is excluded.
    gram = (x.T @ x).toarray() / x.shape[0]
    gradient = gram @ weights - gram + alpha * (1-ratio) * weights + alpha * ratio
    residual = np.where(weights > 0, np.abs(gradient), np.maximum(-gradient, 0.))
    np.fill_diagonal(residual, 0.)
    residual[0, :], residual[:, 0] = 0., 0.
    item_maxima = residual.max(axis=0)[1:]
    report = {
        'model': 'SLIMElastic', 'seed': settings['seed'],
        'source_run_name': args.run.name,
        'run_manifest_sha256': digest(manifest_path),
        'training_split_sha256': manifest['split_sha256']['train'],
        'checkpoint_sha256': digest(checkpoint_path),
        'upstream_model_source': manifest['upstream_model_source'],
        'audit_source_sha256': digest(Path(__file__)),
        'settings': {key: settings[key] for key in ('alpha', 'l1_ratio', 'positive_only', 'hide_item')},
        'matrix_shape': list(x.shape), 'training_nonzero_count': int(x.nnz),
        'normalization_rows_including_padding': x.shape[0],
        'minimum_weight': float(weights.min()),
        'largest_absolute_diagonal': float(np.abs(np.diag(weights)).max()),
        'max_kkt_residual': float(residual.max()),
        'median_item_max_kkt_residual': float(np.median(item_maxima)),
        'p95_item_max_kkt_residual': float(np.quantile(item_maxima, .95)),
        'residual_definition': 'Positive coefficients: absolute gradient; zero coefficients: max(-gradient,0). Fixed diagonal and padding excluded.',
        'interpretation': 'Numerical stationarity diagnostic; not a convergence or optimality certificate. Instructor max_iter=100 and suppressed convergence warnings remain limitations.',
        'test_labels_read': False, 'validation_labels_read': False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('max_kkt_residual', 'median_item_max_kkt_residual', 'p95_item_max_kkt_residual')}, indent=2))


if __name__ == '__main__':
    main()
