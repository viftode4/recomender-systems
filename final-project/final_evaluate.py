"""Evaluate a frozen bundle once on its hash-matched held-out split.

Never tunes, refits, changes transforms, or selects winners. All frozen models
are reported, including required reranking comparisons. Use only after the
complete experimental protocol and research variants have been finalized.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from freeze import load_frozen, recommend
from metrics import evaluate
from societal import discounted_exposure_metrics
from study import digest, grouped, jsd, read_pairs, write_json


def assess(recommendations, truth, history, bundle, arrays):
    """Accuracy and societal diagnostics with training-only group definitions."""
    k = bundle['k']
    users, items = arrays['users'].tolist(), arrays['items'].tolist()
    user_index, item_index = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    counts = {item: int(count) for item, count in zip(items[1:], arrays['training_counts'][1:])}
    result = evaluate(recommendations, truth, history, items[1:], counts, k)
    for user, recs in recommendations.items():
        u = user_index[user]
        selected = [item_index[item] for item in recs]
        binary = arrays['genre_binary'][selected]
        intersection = binary @ binary.T
        union = binary.sum(axis=1, keepdims=True) + binary.sum(axis=1)[None, :] - intersection
        distances = 1 - intersection / np.maximum(union, 1)
        head_fraction = float(arrays['head'][selected].mean())
        result['per_user'][user].update({
            'diversity': float(distances[np.triu_indices(k, 1)].mean()),
            'calibration_jsd': float(jsd(arrays['profiles'][u], arrays['genre_fraction'][selected].mean(axis=0))),
            'head_exposure': head_fraction,
            'popularity_jsd': float(jsd(arrays['popularity_profiles'][u], [1-head_fraction, head_fraction])),
        })
    for metric in ('diversity', 'calibration_jsd', 'head_exposure', 'popularity_jsd'):
        result['aggregate'][metric] = float(np.mean([row[metric] for row in result['per_user'].values()]))
    groups = {}
    for group in range(3):
        members = [u for u in truth if arrays['groups'][user_index[u]] == group]
        if members:
            groups[str(group)] = {'users': len(members), **{
                metric: float(np.mean([result['per_user'][u][metric] for u in members]))
                for metric in result['aggregate'] if metric != f'coverage@{k}'}}
    result['groups'] = groups
    means = [row[f'ndcg@{k}'] for row in groups.values()]
    result['aggregate']['activity_ndcg_gap'] = max(means) - min(means)
    result['aggregate']['worst_activity_ndcg'] = min(means)
    result['aggregate']['head_exposure_gap'] = abs(result['aggregate']['head_exposure'] - float(arrays['head'][1:].mean()))
    exposure_groups = {item: ('head' if arrays['head'][i] else 'tail') for i, item in enumerate(items) if i}
    exposure = discounted_exposure_metrics(recommendations, items[1:], exposure_groups)
    result['item_exposure'] = exposure
    result['aggregate'].update({'exposure_gini': exposure['gini'],
                                'exposure_entropy': exposure['normalized_entropy'],
                                'item_group_exposure_gap': exposure['group_mean_exposure_gap']})
    result['item_groups'] = {}
    for name, membership in [('head', arrays['head']), ('tail', ~arrays['head'])]:
        values = []
        for user, relevant in truth.items():
            positives = {item for item in relevant if membership[item_index[item]]}
            if positives:
                values.append(len(positives & set(recommendations[user])) / len(positives))
        value = float(np.mean(values)) if values else None
        result['aggregate'][f'{name}_recall'] = value
        result['item_groups'][name] = {'users_with_positives': len(values), 'macro_recall': value}
    return result


def evaluate_frozen(frozen, test_path, out):
    frozen, test_path, out = Path(frozen), Path(test_path), Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    started = time.monotonic()
    bundle, arrays = load_frozen(frozen)
    if digest(test_path) != bundle['expected_test_sha256']:
        raise ValueError('Held-out split does not match the predeclared split hash')
    pairs = read_pairs(test_path)
    train, valid = read_pairs(frozen / 'train.tsv'), read_pairs(frozen / 'valid.tsv')
    if len(pairs) != len(set(pairs)) or set(pairs) & (set(train) | set(valid)):
        raise ValueError('Duplicate held-out interactions or overlap with observed history')
    truth, history = grouped(pairs), grouped(train + valid)
    users, items = arrays['users'].tolist(), arrays['items'].tolist()
    if not truth or set(truth) - set(users) or {item for _, item in pairs} - set(items[1:]):
        raise ValueError('Held-out split has unknown users/items or is empty')
    cohort = [u for u, user in enumerate(users) if user in truth]
    # Reserve the output location before costly inference; failures remain visibly
    # incomplete rather than permitting silent replacement with a different run.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'started', 'freeze_sha256': digest(frozen / 'freeze.json'),
                'test_sha256': digest(test_path), 'test_read': True, 'k': bundle['k'],
                'protocol': bundle['protocol'], 'models': list(bundle['models']),
                'selection_metric': bundle['selection']['metric'],
                'test_model_selection': False, 'numpy': np.__version__}
    write_json(out / 'manifest.json', manifest)
    recommendations = recommend(bundle, arrays, 'test', cohort)
    results = {name: assess(recs, truth, history, bundle, arrays) for name, recs in recommendations.items()}
    write_json(out / 'recommendations.json', recommendations)
    write_json(out / 'results.json', results)
    context = bundle['selection']['families']['context']
    policy = results['group-utility-budget-exposure']
    write_json(out / 'group-policy-audit.json', {
        'reference': context, 'calibration': bundle['policy_calibration'],
        'test_retention': {group: row[f"ndcg@{bundle['k']}"] /
                           max(results[context]['groups'][group][f"ndcg@{bundle['k']}"], 1e-12)
                           for group, row in policy['groups'].items()},
        'note': 'The calibration constraint is not a held-out guarantee; no test adjustments were made.'})
    columns = [f"ndcg@{bundle['k']}", f"recall@{bundle['k']}", 'diversity',
               'calibration_jsd', 'head_exposure', 'worst_activity_ndcg']
    lines = ['# Frozen held-out evaluation', '',
             'All choices were frozen before this evaluation. Models are listed in frozen order; no test winner is selected.', '',
             '| Model | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
    for name in bundle['models']:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{results[name]["aggregate"][metric]:.4f}' for metric in columns) + ' |')
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    manifest.update(status='complete', users=len(truth), interactions=len(pairs), seconds=time.monotonic() - started)
    manifest['output_sha256'] = {path.name: digest(path) for path in sorted(out.iterdir()) if path.name != 'manifest.json'}
    write_json(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_frozen(args.frozen, args.test, args.out)
    print(json.dumps({'out': str(args.out), 'users': result['users'], 'models': len(result['models']),
                      'test_model_selection': False}, indent=2))


if __name__ == '__main__':
    main()
