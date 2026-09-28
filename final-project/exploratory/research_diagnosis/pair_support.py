"""Count TRAIN support for binary item conjunctions, without held-out labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exploratory.categorical_reconstruction.run_experiment import load_train


def summarize(source_root, ratings, seeds=(2026, 2027, 2028)):
    rows = []
    for seed in seeds:
        source = Path(source_root) / f'{seed}-EASE-1'
        users, items, train, matrix, manifest = load_train(source, Path(ratings))
        x = (matrix[:, 1:] > 0).astype(np.float64)
        counts = (x.T @ x)[np.triu_indices(x.shape[1], k=1)]
        positive = counts[counts > 0]
        if not len(positive):
            raise ValueError('No observed source pairs in this diagnostic population')
        rows.append({
            'seed': seed, 'users': len(users), 'items_without_padding': len(items)-1,
            'observed_training_pairs': len(train), 'density': float(x.mean()),
            'all_distinct_item_pairs': len(counts),
            'observed_distinct_item_pairs': len(positive),
            'observed_source_pair_support_quantiles': {
                str(q): float(np.quantile(positive, q))
                for q in [0, .25, .5, .75, .9, .99, 1]},
            'observed_source_pairs_supported_by_at_most': {
                str(k): {'count': int((positive <= k).sum()),
                         'fraction': float((positive <= k).mean())}
                for k in [1, 2, 5, 10]},
            'source_manifest_sha256': hashlib.sha256(
                (source / 'manifest.json').read_bytes()).hexdigest(),
            'train_sha256': manifest['split_sha256']['train'],
        })
    return {
        'stage': 'TRAIN-only descriptive diagnosis',
        'validation_read': False, 'test_read': False,
        'interpretation': 'Support counts measure available examples for conjunctions. '
                         'They do not establish a prediction ceiling or prove that more '
                         'users will improve accuracy. Rare pairs can share information '
                         'through other features.',
        'seeds': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('runs/research-v2'))
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = summarize(args.source_root, args.ratings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
