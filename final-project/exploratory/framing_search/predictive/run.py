"""Finite nested TRAIN experiment with all-arm selection seal before assessment."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import evaluate, ranking_metrics
from exploratory.categorical_reconstruction.group_analysis import analyze_groups
from exploratory.framing_search.recording_audit.audit import (
    check_inputs, digest, load_genres, load_train_pairs, require_digest, save_json)

ROLES = ('true_group', 'shuffle_0', 'shuffle_1', 'shuffle_2', 'bag', 'EASE')
PAIR_LAMBDAS = (50., 250., 1000.)
BETAS = (.1, 1.)
EASE_LAMBDAS = (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)
SEED = 20260929
BOOTSTRAPS = 2000
SOURCE_FILES = ('exploratory/framing_search/predictive/run.py',
                'exploratory/framing_search/predictive/model.py',
                'exploratory/framing_search/predictive/PROTOCOL.md',
                'exploratory/framing_search/recording_audit/audit.py',
                'exploratory/categorical_reconstruction/group_analysis.py',
                'metrics.py', 'societal.py')


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def nested_split(pairs, seed=SEED):
    pairs = list(pairs)
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate original TRAIN pairs')
    users = defaultdict(list)
    for user, item in pairs:
        users[user].append(item)
    result = {name: [] for name in ('fit', 'development', 'assessment')}
    for user in sorted(users):
        def key(item):
            encoded = json.dumps([seed, user, item], separators=(',', ':')).encode()
            return hashlib.sha256(encoded).digest(), item
        ordered = sorted(users[user], key=key)
        holdout = max(1, len(ordered) // 10)
        nfit = len(ordered) - 2 * holdout
        if nfit < 1:
            raise ValueError('Every user requires at least three original TRAIN items')
        for name, selected in zip(result, (ordered[:nfit], ordered[nfit:nfit+holdout], ordered[nfit+holdout:])):
            result[name].extend((user, item) for item in selected)
    flattened = [pair for values in result.values() for pair in values]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(pairs):
        raise AssertionError('Nested split is not a disjoint partition')
    return result


def real_catalog_users(ids, train_pairs):
    users = [user for user in ids['users'] if user != '[PAD]']
    present = {user for user, _ in train_pairs}
    if len(set(users)) != len(users) or set(users) != present:
        raise ValueError('Real catalog users must match original TRAIN users')
    return users


def read_fit_timestamps(path, allowed):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = row['user_id:token'], row['item_id:token']
            if pair not in allowed:
                continue
            if pair in result:
                raise ValueError('Duplicate F timestamp')
            value = float(row['timestamp:float'])
            if not math.isfinite(value) or value != int(value):
                raise ValueError('Invalid F timestamp')
            result[pair] = int(value)
    if set(result) != set(allowed):
        raise ValueError('Missing F timestamps')
    return result


def read_assessment_values(path, allowed, run_dir):
    verify_barrier(run_dir)
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = row['user_id:token'], row['item_id:token']
            if pair not in allowed:
                continue
            if pair in result:
                raise ValueError('Duplicate A value')
            rating, timestamp = float(row['rating:float']), float(row['timestamp:float'])
            if (rating not in (1., 2., 3., 4., 5.) or not math.isfinite(timestamp)
                    or timestamp != int(timestamp)):
                raise ValueError('Invalid A value')
            result[pair] = rating, int(timestamp)
    if set(result) != set(allowed):
        raise ValueError('Missing A values')
    return result


def write_pairs(path, pairs):
    with Path(path).open('w') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(('user_id', 'item_id'))
        writer.writerows(pairs)


def grouped(pairs):
    result = defaultdict(set)
    for user, item in pairs:
        result[user].add(item)
    return dict(result)


def rank_scores(scores, observed, k=10):
    scores, observed = np.asarray(scores, float), np.asarray(observed)
    if scores.ndim != 2 or observed.shape != scores.shape or observed.dtype != bool:
        raise ValueError('Require scores and matching boolean F observations')
    if not np.isfinite(scores).all() or not 0 < k < scores.shape[1]:
        raise ValueError('Invalid scores or k')
    masked = scores.copy()
    masked[observed] = -np.inf
    masked[:, 0] = -np.inf
    if np.any(np.isfinite(masked).sum(axis=1) < k):
        raise ValueError('Too few unseen candidates')
    return np.argsort(-masked, axis=1, kind='stable')[:, :k]


def metric_rows(ranks, truth_sets):
    ranks = np.asarray(ranks)
    if ranks.ndim != 2 or len(ranks) != len(truth_sets):
        raise ValueError('Mismatched rank/truth rows')
    rows = [ranking_metrics(row.tolist(), truth, ranks.shape[1])
            for row, truth in zip(ranks, truth_sets)]
    return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}


def current_inputs_match(protocol):
    for entry in protocol['inputs'].values():
        require_digest(Path(entry['path']), entry['sha256'])


def validate_selection(candidates, choices):
    if set(choices) != set(ROLES):
        raise ValueError('All six roles must be selected')
    expected = [(role, penalty, beta) for role in ROLES for penalty, beta in candidate_specs(role)]
    if len(candidates) != len(expected):
        raise ValueError('Incomplete 39-candidate grid')
    for row, (role, penalty, beta) in zip(candidates, expected):
        if ((row.get('role'), row.get('lambda'), row.get('beta')) != (role, penalty, beta)
                or row.get('candidate_id') != candidate_id(role, penalty, beta)):
            raise ValueError('Candidate grid/order differs from declared specification')
        value = row['development']['aggregate']['ndcg@10']
        if not math.isfinite(value):
            raise ValueError('Nonfinite candidate selection metric')
    for role in ROLES:
        best = max((row for row in candidates if row['role'] == role),
                   key=lambda row: row['development']['aggregate']['ndcg@10'])
        choice = choices[role]
        if any(choice.get(key) != value for key, value in best.items()):
            raise ValueError('Selection is not the first exact development maximum')
        if choice.get('scores_file') != f'selected-scores/{role}.npy':
            raise ValueError('Unexpected selected score path')


def write_barrier(run_dir, roles=ROLES):
    if tuple(roles) != ROLES:
        raise ValueError('Cannot seal a subset or reordered role set')
    run_dir = Path(run_dir)
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    if protocol['source_sha256'] != source_hashes():
        raise ValueError('Source changed before seal')
    current_inputs_match(protocol)
    choices = json.loads((run_dir / 'selections.json').read_text())
    validate_selection(json.loads((run_dir / 'candidates.json').read_text()), choices)
    filenames = ['protocol.json', 'selections.json', 'candidates.json', 'nested-A.tsv',
                 'nested-F.tsv', 'nested-D.tsv', 'fit-context.npz', 'ids.json']
    filenames += [choices[role]['scores_file'] for role in ROLES]
    hashes = {name: digest(run_dir / name) for name in filenames}
    seal = {'roles': list(ROLES), 'files_sha256': hashes,
            'source_sha256': protocol['source_sha256'], 'assessment_evaluated': False}
    save_json(run_dir / 'SELECTIONS-FROZEN.json', seal)
    return seal


def verify_barrier(run_dir):
    run_dir = Path(run_dir)
    seal = json.loads((run_dir / 'SELECTIONS-FROZEN.json').read_text())
    if seal.get('roles') != list(ROLES) or seal.get('assessment_evaluated') is not False:
        raise ValueError('Incomplete or invalid all-arm selection barrier')
    if seal['source_sha256'] != source_hashes():
        raise ValueError('Source changed after seal')
    required = {'protocol.json', 'selections.json', 'candidates.json', 'nested-A.tsv',
                'nested-F.tsv', 'nested-D.tsv', 'fit-context.npz', 'ids.json'}
    if not required.issubset(seal['files_sha256']):
        raise ValueError('Incomplete required barrier files')
    for name, expected in seal['files_sha256'].items():
        require_digest(run_dir / name, expected)
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    if protocol['source_sha256'] != seal['source_sha256']:
        raise ValueError('Protocol source signature differs')
    choices = json.loads((run_dir / 'selections.json').read_text())
    validate_selection(json.loads((run_dir / 'candidates.json').read_text()), choices)
    for role in ROLES:
        path = choices[role]['scores_file']
        if seal['files_sha256'].get(path) != choices[role]['scores_file_sha256']:
            raise ValueError('Selected score is not sealed')
    expected_files = required | {choices[role]['scores_file'] for role in ROLES}
    if set(seal['files_sha256']) != expected_files:
        raise ValueError('Barrier file set differs')
    current_inputs_match(protocol)
    return seal


def candidate_specs(role):
    return [(penalty, 0.) for penalty in EASE_LAMBDAS] if role == 'EASE' else [
        (penalty, beta) for penalty in PAIR_LAMBDAS for beta in BETAS]


def candidate_id(role, penalty, beta):
    return f'{role}-lambda-{penalty:g}-beta-{beta:g}'


def public_evaluation(ranks, truth, users, items, history, genres, counts):
    eligible = [user for user in users if truth.get(user)]
    if not eligible:
        return {'users': 0, 'aggregate': None}, {}
    index = {user: row for row, user in enumerate(users)}
    recommendations = {user: [items[i] for i in ranks[index[user]]] for user in eligible}
    restricted = {user: truth[user] for user in eligible}
    detail = evaluate(recommendations, restricted, history, items[1:], counts, 10)
    result = analyze_groups(recommendations, restricted, history, items[1:], counts, genres,
                            k=10, group_users=users)
    return result, detail['per_user']


def paired_bootstrap(per_user, users):
    indices = np.random.default_rng(2026092901).integers(0, len(users), (BOOTSTRAPS, len(users)))
    result = {}
    for control in ('bag', 'mean_shuffle', 'EASE'):
        result['true_minus_' + control] = {}
        for metric in ('ndcg@10', 'recall@10'):
            true = np.array([per_user['true_group'][user][metric] for user in users])
            if control == 'mean_shuffle':
                other = np.mean([[per_user[f'shuffle_{r}'][user][metric] for user in users]
                                 for r in range(3)], axis=0)
            else:
                other = np.array([per_user[control][user][metric] for user in users])
            differences = true - other
            interval = np.quantile(differences[indices].mean(axis=1), [.025, .975])
            result['true_minus_' + control][metric] = {'users': len(users),
                'mean_delta': float(differences.mean()), 'ci95': interval.tolist(),
                'bootstrap_seed': 2026092901, 'replicates': BOOTSTRAPS,
                'interpretation': 'descriptive paired-user percentile interval in this explored dataset'}
    return result


def experiment(args):
    from exploratory.framing_search.predictive.model import prepare_features, fit
    started = time.monotonic()
    if args.run_dir.exists() or args.evidence.exists():
        raise ValueError('Refusing to overwrite experiment artifacts')
    sources = source_hashes()
    inputs, expected_rows = check_inputs(args.source, args.signature, args.train, args.ratings, args.metadata)
    signature = json.loads(args.signature.read_text())
    ids_path = args.source / 'ids.json'
    require_digest(ids_path, signature['ids_sha256'])
    inputs['source_catalog_ids'] = {'path': str(ids_path), 'sha256': digest(ids_path)}
    ids = json.loads(ids_path.read_text())
    items = ids['items']
    if (ids.get('padding_index') != 0 or items[0] != '[PAD]' or len(set(items)) != len(items)
            or len(items) != 1683):
        raise ValueError('Unexpected original full catalog')
    original = load_train_pairs(args.train)
    users = real_catalog_users(ids, original)
    ids = {**ids, 'users': users}
    if len(original) != expected_rows or set(grouped(original)) != set(users):
        raise ValueError('Original TRAIN dimensions differ')
    partition = nested_split(original)
    args.run_dir.mkdir(parents=True)
    for name, suffix in (('fit', 'F'), ('development', 'D'), ('assessment', 'A')):
        write_pairs(args.run_dir / f'nested-{suffix}.tsv', partition[name])
    split_counts = {name: len(pairs) for name, pairs in partition.items()}
    split_counts.update(users=len(users), catalog=len(items)-1)
    fit_pairs, development = set(partition['fit']), grouped(partition['development'])
    del partition
    save_json(args.run_dir / 'ids.json', ids)
    protocol = {'schema_version': 1, 'study': 'nested_original_train_exploratory',
        'original_validation_read': False, 'original_test_read': False, 'fresh_population_test': False,
        'roles': list(ROLES), 'pair_lambdas': list(PAIR_LAMBDAS), 'betas': list(BETAS),
        'ease_lambdas': list(EASE_LAMBDAS), 'candidate_count': 39,
        'nested_split_seed': SEED, 'inputs': inputs, 'source_sha256': sources,
        'input_shape_correction': 'Explicit [PAD] source user removed before split/value parsing; real user and item order preserved.',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'numerical_threads': 1},
        'split_counts': split_counts, 'command': sys.argv}
    save_json(args.run_dir / 'protocol.json', protocol)
    times = read_fit_timestamps(args.ratings, fit_pairs)
    ui, ii = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    binary = np.zeros((len(users), len(items)), dtype=bool)
    group_labels = np.zeros(binary.shape, dtype=np.int64)
    for (user, item), timestamp in times.items():
        binary[ui[user], ii[item]] = True
        group_labels[ui[user], ii[item]] = timestamp
    np.savez_compressed(args.run_dir / 'fit-context.npz', binary=binary, groups=group_labels)
    history, counts = grouped(fit_pairs), Counter(item for _, item in fit_pairs)
    genres = load_genres(args.metadata, set(items[1:]))
    prepared_true = prepare_features(binary, group_labels)
    prepared_bag = prepare_features(binary)
    prepared = {'true_group': prepared_true, 'bag': prepared_bag, 'EASE': prepared_bag}
    for r in range(3):
        rng = np.random.default_rng(2026092900 + r)
        shuffled = group_labels.copy()
        for row in range(len(users)):
            positions = np.flatnonzero(binary[row])
            shuffled[row, positions] = rng.permutation(shuffled[row, positions])
        own = prepare_features(binary, shuffled)
        if not np.isclose(own.normalizer, prepared_true.normalizer, rtol=0, atol=1e-12):
            raise AssertionError('Shuffle changed exact group pair mass')
        prepared[f'shuffle_{r}'] = own
    candidates, selections = [], {}
    (args.run_dir / 'selected-scores').mkdir()
    for role in ROLES:
        best = None
        for penalty, beta in candidate_specs(role):
            if source_hashes() != sources:
                raise ValueError('Frozen source changed during fits')
            tick = time.monotonic()
            model = fit(prepared[role], lambda_binary=penalty, pair_weight=beta)
            scores = np.asarray(model.predict(), dtype=np.float64)
            fit_seconds = time.monotonic() - tick
            ranks = rank_scores(scores, binary)
            recs = {user: [items[i] for i in ranks[row]] for row, user in enumerate(users)}
            measured = evaluate(recs, development, history, items[1:], counts, 10)
            row = {'role': role, 'candidate_id': candidate_id(role, penalty, beta),
                   'lambda': penalty, 'beta': beta, 'development': {'users': len(users),
                   'aggregate': measured['aggregate']}, 'fit_seconds': fit_seconds,
                   'normalizer': float(prepared[role].normalizer), 'diagnostics': model.diagnostics}
            candidates.append(row)
            value = measured['aggregate']['ndcg@10']
            if best is None or value > best['development']['aggregate']['ndcg@10']:
                best = row.copy()
                relative = f'selected-scores/{role}.npy'
                np.save(args.run_dir / relative, scores, allow_pickle=False)
                best.update(scores_file=relative, scores_file_sha256=digest(args.run_dir / relative))
            save_json(args.run_dir / 'candidates.json', candidates)
            print(json.dumps({'completed_fits': len(candidates), 'total_fits': 39, 'role': role,
                'candidate_id': row['candidate_id'], 'fit_seconds': fit_seconds,
                'development_ndcg': value}, sort_keys=True), flush=True)
        selections[role] = best
    if len(candidates) != 39:
        raise AssertionError('Incomplete fixed grid')
    save_json(args.run_dir / 'selections.json', selections)
    write_barrier(args.run_dir)
    verify_barrier(args.run_dir)
    selection_seconds = time.monotonic() - started
    assessment_pairs = load_train_pairs(args.run_dir / 'nested-A.tsv')
    assessment_values = read_assessment_values(args.ratings, assessment_pairs, args.run_dir)
    fit_times = defaultdict(set)
    for (user, _), timestamp in times.items():
        fit_times[user].add(timestamp)
    endpoints = {'all_recorded': grouped(assessment_pairs),
        'liked_ge4': grouped(pair for pair, (rating, _) in assessment_values.items() if rating >= 4),
        'timestamp_seen_in_fit': grouped(pair for pair, (_, timestamp) in assessment_values.items()
                                         if timestamp in fit_times[pair[0]]),
        'timestamp_unseen_in_fit': grouped(pair for pair, (_, timestamp) in assessment_values.items()
                                           if timestamp not in fit_times[pair[0]])}
    assessment, primary_per_user = {}, {}
    for role in ROLES:
        scores = np.load(args.run_dir / selections[role]['scores_file'], allow_pickle=False)
        ranks = rank_scores(scores, binary)
        assessment[role] = {}
        for endpoint, truth in endpoints.items():
            measured, individual = public_evaluation(ranks, truth, users, items, history, genres, counts)
            measured['endpoint'] = endpoint
            if 'definitions' in measured:
                measured['definitions']['relevance'] = endpoint
            measured['positive_pairs'] = sum(map(len, truth.values()))
            assessment[role][endpoint] = measured
            if endpoint == 'all_recorded':
                primary_per_user[role] = individual
        np.savez_compressed(args.run_dir / f'{role}-assessment.npz', ranks=ranks,
            **{key: np.array([primary_per_user[role][user][key] for user in users])
               for key in next(iter(primary_per_user[role].values()))})
    paired = paired_bootstrap(primary_per_user, users)
    ndcg = {role: assessment[role]['all_recorded']['aggregate']['ndcg@10'] for role in ROLES}
    shuffle_mean = float(np.mean([ndcg[f'shuffle_{r}'] for r in range(3)]))
    gain = ndcg['true_group'] / ndcg['EASE'] - 1
    success = {'relative_gain_over_ease': gain, 'threshold_relative_gain': .1,
               'beats_bag': ndcg['true_group'] > ndcg['bag'],
               'beats_shuffle_mean': ndcg['true_group'] > shuffle_mean,
               'mean_shuffle_ndcg': shuffle_mean,
               'declared_success': gain >= .1 and ndcg['true_group'] > ndcg['bag']
                                   and ndcg['true_group'] > shuffle_mean,
               'mechanism_support_descriptive_intervals': all(
                   paired['true_minus_' + role]['ndcg@10']['ci95'][0] > 0
                   for role in ('bag', 'mean_shuffle'))}
    verify_barrier(args.run_dir)
    result = {'schema_version': 1, 'status': 'complete', 'study': protocol['study'],
        'original_validation_read': False, 'original_test_read': False, 'fresh_population_test': False,
        'split_counts': split_counts, 'candidates': candidates, 'selected': selections,
        'assessment': assessment, 'paired_comparisons': paired, 'success': success,
        'timing': {'selection_seconds': selection_seconds, 'total_seconds': time.monotonic()-started},
        'limitations': ['Nested exploration after prior dataset use, not a fresh population test.',
            'Timestamp equality marks rating recording, not consumption or identified screen exposure.',
            'Timestamp subset endpoints overlap users and are not independent replications.',
            'Different families have declared unequal hyperparameter grid sizes.',
            'Paired intervals are descriptive and do not correct prior model exploration.']}
    save_json(args.run_dir / 'assessment-aggregates.json', result)
    manifest = {'status': 'complete', 'original_validation_read': False, 'original_test_read': False,
                'source_sha256': sources, 'selection_seal_sha256': digest(args.run_dir / 'SELECTIONS-FROZEN.json'),
                'aggregate_sha256': digest(args.run_dir / 'assessment-aggregates.json'),
                'protocol_sha256': digest(args.run_dir / 'protocol.json'),
                'runtime': protocol['runtime'], 'command': sys.argv,
                'raw_run_files_sha256': {str(path.relative_to(args.run_dir)): digest(path)
                    for path in sorted(args.run_dir.rglob('*')) if path.is_file()}}
    save_json(args.run_dir / 'manifest.json', manifest)
    args.evidence.mkdir(parents=True)
    save_json(args.evidence / 'aggregates.json', result)
    save_json(args.evidence / 'provenance.json', manifest)
    save_json(args.evidence / 'protocol.json', protocol)
    save_json(args.evidence / 'SHA256.json', {name: digest(args.evidence / name)
              for name in ('aggregates.json', 'provenance.json', 'protocol.json')})
    print(json.dumps({'status': 'complete', 'success': success, 'timing': result['timing']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'signature', 'train', 'ratings', 'metadata', 'run-dir', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    experiment(parser.parse_args())
