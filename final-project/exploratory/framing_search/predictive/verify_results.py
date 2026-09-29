"""Independent artifact/metric audit; no model search and no original VALID/TEST.

This script does not import the study's model, runner or metric implementations.
It checks selected saved predictions, selection choices and every reported
assessment metric using literal definitions. The one reference solve is the
selected EASE item-space closed form, not another hyperparameter search.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[3]
ROLES = ('true_group', 'shuffle_0', 'shuffle_1', 'shuffle_2', 'bag', 'EASE')
PAIR_GRID = [(a, b) for a in (50., 250., 1000.) for b in (.1, 1.)]
EASE_GRID = [(a, 0.) for a in (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def pairs(path):
    with Path(path).open() as stream:
        result = [(row['user_id'], row['item_id'])
                  for row in csv.DictReader(stream, delimiter='\t')]
    require(len(result) == len(set(result)), 'Duplicate pair IDs')
    return result


class Checks:
    def __init__(self):
        self.numeric = 0
        self.maximum_error = 0.

    def number(self, actual, expected, label, tolerance=3e-11):
        if expected is None:
            require(actual is None, label + ' should be empty')
            return
        error = abs(float(actual) - float(expected))
        require(math.isfinite(error) and error <= tolerance,
                f'{label}: {actual} differs from independent {expected}')
        self.numeric += 1
        self.maximum_error = max(self.maximum_error, error)

    def tree(self, actual, expected, label):
        for key, value in expected.items():
            require(key in actual, f'{label}: missing {key}')
            if isinstance(value, dict):
                self.tree(actual[key], value, label + '.' + key)
            elif isinstance(value, str):
                require(actual[key] == value, f'{label}.{key} differs')
            else:
                self.number(actual[key], value, label + '.' + key)


def verify_seals(directory):
    protocol = read_json(directory / 'protocol.json')
    manifest = read_json(directory / 'manifest.json')
    seal = read_json(directory / 'SELECTIONS-FROZEN.json')
    choices = read_json(directory / 'selections.json')
    require(manifest['status'] == 'complete', 'Run incomplete')
    require(seal['roles'] == list(ROLES) and seal['assessment_evaluated'] is False,
            'Invalid selection seal')
    require(set(choices) == set(ROLES), 'Incomplete choices')
    required = {'protocol.json', 'selections.json', 'candidates.json', 'nested-A.tsv',
                'nested-F.tsv', 'nested-D.tsv', 'fit-context.npz', 'ids.json'}
    required |= {f'selected-scores/{role}.npy' for role in ROLES}
    require(set(seal['files_sha256']) == required, 'Selection seal file set differs')
    for source, expected in protocol['source_sha256'].items():
        require(digest(ROOT / source) == expected, f'Source changed: {source}')
    require(protocol['source_sha256'] == manifest['source_sha256'] == seal['source_sha256'],
            'Source signatures disagree')
    for group in (seal['files_sha256'], manifest['raw_run_files_sha256']):
        for path, expected in group.items():
            require(digest(directory / path) == expected, f'Artifact hash differs: {path}')
    for record in protocol['inputs'].values():
        require(digest(record['path']) == record['sha256'], 'Input changed')
    for role, choice in choices.items():
        require(choice['scores_file'] == f'selected-scores/{role}.npy', 'Unexpected scores path')
        require(digest(directory / choice['scores_file']) == choice['scores_file_sha256'],
                'Selected score hash differs')
    require(digest(directory / 'SELECTIONS-FROZEN.json') == manifest['selection_seal_sha256'],
            'Selection seal changed')
    require(digest(directory / 'assessment-aggregates.json') == manifest['aggregate_sha256'],
            'Assessment aggregates changed')
    require(digest(directory / 'protocol.json') == manifest['protocol_sha256'],
            'Protocol changed')
    require(not protocol['original_validation_read'] and not protocol['original_test_read'],
            'Unexpected study scope')
    return protocol, choices


def rank_literal(scores, observed):
    require(scores.shape == observed.shape and np.isfinite(scores).all(), 'Invalid score matrix')
    rows = []
    for row, mask in zip(scores, observed):
        candidates = [item for item in range(1, len(row)) if not mask[item]]
        rows.append(sorted(candidates, key=lambda item: (-row[item], item))[:10])
    return np.asarray(rows, dtype=np.int64)


def literal_user_metrics(recs, truth, counts, total, catalog):
    require(len(recs) == 10 and len(set(recs)) == 10 and truth, 'Invalid evaluation row')
    hits = [position for position, item in enumerate(recs) if item in truth]
    dcg = sum(1 / math.log2(position + 2) for position in hits)
    ideal = sum(1 / math.log2(position + 2) for position in range(min(10, len(truth))))
    return {'precision@10': len(hits) / 10, 'recall@10': len(hits) / len(truth),
            'ndcg@10': dcg / ideal, 'mrr@10': 1 / (hits[0] + 1) if hits else 0.,
            'hit@10': float(bool(hits)),
            'novelty@10': sum(math.log2((total + catalog) / (counts[item] + 1))
                              for item in recs) / 10}


def jsd(first, second):
    middle = (first + second) / 2
    return .5 * sum(float(np.sum(values[values > 0]
                    * np.log2(values[values > 0] / middle[values > 0])))
                    for values in (first, second))


def evaluate_literal(ranks, truths, context):
    binary, counts, item_tokens, genres, fractions, profiles, activity, head = context
    eligible = [row for row, truth in enumerate(truths) if truth]
    if not eligible:
        return {'users': 0, 'aggregate': None}, {}
    selected = {row: list(map(int, ranks[row])) for row in eligible}
    per_user, exposed = {}, set()
    total, catalog = int(binary.sum()), binary.shape[1] - 1
    exposures = np.zeros(binary.shape[1], dtype=float)
    for row, recs in selected.items():
        require(not any(binary[row, item] for item in recs), 'Seen item recommended')
        values = literal_user_metrics(recs, truths[row], counts, total, catalog)
        distances = [1 - len(genres[a] & genres[b]) / len(genres[a] | genres[b])
                     for position, a in enumerate(recs) for b in recs[position + 1:]]
        history_head = float(binary[row, list(head)].sum() / binary[row].sum())
        exposure_head = len(set(recs) & head) / 10
        values.update(diversity=float(np.mean(distances)),
            calibration_jsd=jsd(profiles[row], np.mean(fractions[recs], axis=0)),
            head_exposure=exposure_head,
            popularity_jsd=jsd(np.array([1-history_head, history_head]),
                               np.array([1-exposure_head, exposure_head])))
        per_user[row] = values
        exposed.update(recs)
        for position, item in enumerate(recs):
            exposures[item] += 1 / math.log2(position + 2)
    aggregate = {key: float(np.mean([values[key] for values in per_user.values()]))
                 for key in next(iter(per_user.values()))}
    aggregate['coverage@10'] = len(exposed) / catalog
    user_groups = {}
    for group in ('sparse', 'medium', 'dense'):
        members = [row for row in eligible if activity[row] == group]
        user_groups[group] = {'users': len(members), 'status': 'evaluated' if members else 'empty'}
        user_groups[group].update({key: float(np.mean([per_user[row][key] for row in members]))
                                  if members else None for key in aggregate if key != 'coverage@10'})
    group_ndcgs = [g['ndcg@10'] for g in user_groups.values() if g['users']]
    aggregate['activity_ndcg_gap'] = max(group_ndcgs) - min(group_ndcgs)
    aggregate['worst_activity_ndcg'] = min(group_ndcgs)
    aggregate['head_exposure_gap'] = abs(aggregate['head_exposure'] - len(head) / catalog)
    values = exposures[1:] / len(eligible)
    proportions = values / values.sum()
    positive = proportions > 0
    entropy = -float(np.sum(proportions[positive] * np.log2(proportions[positive])))
    # Pairwise absolute difference definition, independent of the production sorted formula.
    gini = float(np.abs(values[:, None] - values[None, :]).sum()
                 / (2 * catalog * values.sum()))
    item_groups, exposure_groups = {}, {}
    for name, members in (('head', head), ('tail', set(range(1, catalog + 1)) - head)):
        conditional = []
        positive_pairs = hits = slots = 0
        for row, recs in selected.items():
            relevant = truths[row] & members
            found = len(set(recs) & relevant)
            positive_pairs += len(relevant)
            hits += found
            slots += len(set(recs) & members)
            if relevant:
                conditional.append(found / len(relevant))
        conditional_mean = float(np.mean(conditional)) if conditional else None
        exposure_groups[name] = {'items': len(members),
            'mean_item_exposure': float(np.mean(exposures[list(members)]) / len(eligible)),
            'fraction_total_exposure': float(exposures[list(members)].sum() / exposures.sum())}
        item_groups[name] = {'items': len(members), 'users_with_positives': len(conditional),
            'heldout_positive_pairs': positive_pairs, 'hits': hits, 'recall@10': conditional_mean,
            'exposure': slots / (len(eligible) * 10),
            'discounted_exposure': exposure_groups[name]['fraction_total_exposure'],
            'mean_item_exposure': exposure_groups[name]['mean_item_exposure']}
        aggregate[name + '_recall'] = conditional_mean
    gap = abs(exposure_groups['head']['mean_item_exposure']
              - exposure_groups['tail']['mean_item_exposure'])
    aggregate.update(exposure_gini=gini, exposure_entropy=entropy / math.log2(catalog),
                     item_group_exposure_gap=gap)
    exposure = {'gini': gini, 'entropy_bits': entropy, 'normalized_entropy': entropy / math.log2(catalog),
                'groups': exposure_groups, 'group_mean_exposure_gap': gap,
                'users': len(eligible), 'catalogue_items': catalog}
    return {'users': len(eligible), 'aggregate': aggregate, 'user_groups': user_groups,
            'item_groups': item_groups, 'item_exposure': exposure}, per_user


def audit(directory, output):
    directory, output = Path(directory), Path(output)
    require(not output.exists(), 'Refusing to overwrite audit receipt')
    started = time.monotonic()
    protocol, choices = verify_seals(directory)  # Before parsing any A values.
    checks = Checks()
    results = read_json(directory / 'assessment-aggregates.json')
    ids = read_json(directory / 'ids.json')
    users, items = ids['users'], ids['items']
    ui, ii = {u: row for row, u in enumerate(users)}, {i: col for col, i in enumerate(items)}
    require(len(users) == 943 and len(items) == 1683 and items[0] == '[PAD]', 'Unexpected IDs')
    original = pairs(protocol['inputs']['original_train_pairs']['path'])
    source_ids = read_json(protocol['inputs']['source_catalog_ids']['path'])
    require(source_ids['items'] == items, 'Catalog differs from original')
    require([u for u in source_ids['users'] if u != '[PAD]'] == users,
            'User order differs after explicit padding removal')
    partitions = {part: pairs(directory / f'nested-{part}.tsv') for part in ('F', 'D', 'A')}
    sets = {part: set(values) for part, values in partitions.items()}
    require(set.union(*sets.values()) == set(original), 'Nested split union differs')
    require(sum(map(len, sets.values())) == len(original), 'Nested splits overlap')
    expected = {'F': [], 'D': [], 'A': []}
    by_user = defaultdict(list)
    for user, item in original:
        by_user[user].append(item)
    for user in sorted(by_user):
        ordered = sorted(by_user[user], key=lambda item: (
            hashlib.sha256(json.dumps([20260929, user, item], separators=(',', ':')).encode()).hexdigest(), item))
        holdout = max(1, len(ordered) // 10)
        for part, selected in (('F', ordered[:-2*holdout]),
                               ('D', ordered[-2*holdout:-holdout]), ('A', ordered[-holdout:])):
            expected[part].extend((user, item) for item in selected)
    require(expected == partitions, 'Nested split is not the declared ID-only partition')
    with np.load(directory / 'fit-context.npz', allow_pickle=False) as archive:
        binary, timestamps = archive['binary'], archive['groups']
    require(binary.shape == (943, 1683) and binary.dtype == bool
            and timestamps.shape == binary.shape, 'Invalid F context shape')
    expected_binary = np.zeros(binary.shape, dtype=bool)
    for user, item in partitions['F']:
        expected_binary[ui[user], ii[item]] = True
    require(np.array_equal(binary, expected_binary), 'F context includes or omits records')
    fit_times, assessment_values = {}, {}
    with Path(protocol['inputs']['raw_interactions']['path']).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = row['user_id:token'], row['item_id:token']
            if pair in sets['F']:
                value = float(row['timestamp:float'])
                require(math.isfinite(value) and value == int(value), 'Invalid F time')
                require(pair not in fit_times, 'Duplicate F raw row')
                fit_times[pair] = int(value)
            elif pair in sets['A']:
                rating, value = float(row['rating:float']), float(row['timestamp:float'])
                require(rating in (1., 2., 3., 4., 5.) and math.isfinite(value)
                        and value == int(value), 'Invalid A value')
                require(pair not in assessment_values, 'Duplicate A raw row')
                assessment_values[pair] = rating, int(value)
            # Original VALID/TEST and nested D values are deliberately not interpreted.
    require(set(fit_times) == sets['F'] and set(assessment_values) == sets['A'], 'Missing raw rows')
    for (user, item), timestamp in fit_times.items():
        require(timestamps[ui[user], ii[item]] == timestamp, 'Context timestamp differs')
    require(np.all(timestamps[~binary] == 0), 'Unobserved timestamps in context')
    local_times = defaultdict(set)
    for (user, _), timestamp in fit_times.items():
        local_times[user].add(timestamp)
    endpoints = {name: [set() for _ in users] for name in
                 ('all_recorded', 'liked_ge4', 'timestamp_seen_in_fit', 'timestamp_unseen_in_fit')}
    for (user, item), (rating, timestamp) in assessment_values.items():
        row, col = ui[user], ii[item]
        endpoints['all_recorded'][row].add(col)
        if rating >= 4:
            endpoints['liked_ge4'][row].add(col)
        name = 'timestamp_seen_in_fit' if timestamp in local_times[user] else 'timestamp_unseen_in_fit'
        endpoints[name][row].add(col)
    development = [set() for _ in users]
    for user, item in partitions['D']:
        development[ui[user]].add(ii[item])
    genres = [set() for _ in items]
    with Path(protocol['inputs']['static_item_metadata']['path']).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            token = row['item_id:token']
            if token in ii:
                genres[ii[token]] = set(row['class:token_seq'].split())
    require(all(genres[1:]), 'Missing genres')
    vocabulary = sorted(set().union(*genres))
    fractions = np.zeros((len(items), len(vocabulary)))
    for item in range(1, len(items)):
        for col, genre in enumerate(vocabulary):
            fractions[item, col] = float(genre in genres[item]) / len(genres[item])
    lengths, counts = binary.sum(axis=1), binary.sum(axis=0)
    profiles = (binary.astype(float) @ fractions) / lengths[:, None]
    thresholds = np.quantile(lengths, [1/3, 2/3])
    activity = [ ('sparse', 'medium', 'dense')[sum(value >= thresholds)] for value in lengths]
    head = set(sorted(range(1, len(items)), key=lambda i: (-int(counts[i]), items[i]))
               [:math.ceil(.2*(len(items)-1))])
    context = (binary, counts, items, genres, fractions, profiles, activity, head)
    grouped_mass = sum(sum(n*(n-1)//2 for n in Counter(timestamps[row, binary[row]]).values())
                       for row in range(len(users)))
    bag_mass = sum(int(n)*(int(n)-1)//2 for n in lengths)
    normalizers = {role: (bag_mass if role in ('bag', 'EASE') else grouped_mass) / binary.sum()
                   for role in ROLES}
    candidates = read_json(directory / 'candidates.json')
    declared = [(role, penalty, beta) for role in ROLES
                for penalty, beta in (EASE_GRID if role == 'EASE' else PAIR_GRID)]
    require(len(candidates) == len(declared) == 39, 'Incomplete declared grid')
    for candidate, (role, penalty, beta) in zip(candidates, declared):
        require((candidate['role'], candidate['lambda'], candidate['beta']) == (role, penalty, beta),
                'Candidate grid or order differs')
        require(candidate['candidate_id'] == f'{role}-lambda-{penalty:g}-beta-{beta:g}',
                'Candidate identity differs')
        checks.number(candidate['normalizer'], normalizers[role], 'F normalizer')
    for role in ROLES:
        own = [row for row in candidates if row['role'] == role]
        values = [row['development']['aggregate']['ndcg@10'] for row in own]
        best = own[values.index(max(values))]
        require(all(choices[role][key] == value for key, value in best.items()),
                'Selection is not the first exact maximum: ' + role)
    all_primary = {}
    ease_error = None
    for role in ROLES:
        scores = np.load(directory / choices[role]['scores_file'], allow_pickle=False)
        ranks = rank_literal(scores, binary)
        selected_d, _ = evaluate_literal(ranks, development, context)
        for key, value in choices[role]['development']['aggregate'].items():
            checks.number(value, selected_d['aggregate'][key], role + '.selected_D.' + key)
        for endpoint, truths in endpoints.items():
            measured, per_user = evaluate_literal(ranks, truths, context)
            recorded = results['assessment'][role][endpoint]
            checks.tree(recorded, measured, role + '.' + endpoint)
            checks.number(recorded['positive_pairs'], sum(map(len, truths)), role + '.positive_pairs')
            if endpoint == 'all_recorded':
                all_primary[role] = per_user
        with np.load(directory / f'{role}-assessment.npz', allow_pickle=False) as archive:
            require(np.array_equal(archive['ranks'], ranks), 'Saved rank indices differ')
            for key in archive.files:
                if key != 'ranks':
                    expected_values = np.array([all_primary[role][row][key] for row in range(len(users))])
                    error = float(np.max(np.abs(archive[key] - expected_values)))
                    checks.number(error, 0., role + '.per_user.' + key)
        if role == 'EASE':
            design = binary.astype(float)
            matrix = design.T @ design + choices[role]['lambda'] * np.eye(len(items))
            inverse = linalg.cho_solve(linalg.cho_factor(matrix, lower=True), np.eye(len(items)))
            coefficients = -inverse / np.diag(inverse)[None, :]
            np.fill_diagonal(coefficients, 0.)
            direct = design @ coefficients
            ease_error = float(np.max(np.abs(scores - direct)))
            require(ease_error < 2e-9, 'Saved EASE scores differ from direct item-space reference')
        print(json.dumps({'audited_role': role, 'numeric_checks': checks.numeric}), flush=True)
    draws = np.random.default_rng(2026092901).integers(0, len(users), (2000, len(users)))
    for control in ('bag', 'mean_shuffle', 'EASE'):
        for metric in ('ndcg@10', 'recall@10'):
            true = np.array([all_primary['true_group'][row][metric] for row in range(len(users))])
            if control == 'mean_shuffle':
                other = np.mean([[all_primary[f'shuffle_{rep}'][row][metric]
                                  for row in range(len(users))] for rep in range(3)], axis=0)
            else:
                other = np.array([all_primary[control][row][metric] for row in range(len(users))])
            differences = true - other
            sampled = differences[draws].mean(axis=1)
            lower, upper = np.quantile(sampled, [.025, .975])
            recorded = results['paired_comparisons']['true_minus_' + control][metric]
            checks.number(recorded['mean_delta'], float(differences.mean()), 'paired delta')
            checks.number(recorded['ci95'][0], lower, 'paired lower')
            checks.number(recorded['ci95'][1], upper, 'paired upper')
    ndcg = {role: float(np.mean([row['ndcg@10'] for row in all_primary[role].values()]))
            for role in ROLES}
    shuffle_mean = sum(ndcg[f'shuffle_{rep}'] for rep in range(3)) / 3
    gain = ndcg['true_group'] / ndcg['EASE'] - 1
    checks.number(results['success']['relative_gain_over_ease'], gain, 'EASE relative gain')
    checks.number(results['success']['mean_shuffle_ndcg'], shuffle_mean, 'Mean shuffled nDCG')
    require(results['success']['declared_success'] == (gain >= .1
            and ndcg['true_group'] > ndcg['bag'] and ndcg['true_group'] > shuffle_mean),
            'Declared success flag differs')
    verify_seals(directory)
    receipt = {'status': 'PASS', 'original_validation_read': False, 'original_test_read': False,
        'fresh_population_test': False, 'new_model_search': False, 'grouped_models_refitted': False,
        'checks': {'declared_candidates_and_selection_choices': 39, 'selected_score_roles': 6,
                   'assessment_endpoints_per_role': 4, 'numeric_metric_checks': checks.numeric,
                   'maximum_metric_error': checks.maximum_error,
                   'selected_ease_item_space_maximum_score_error': ease_error,
                   'source_and_input_hashes': 'PASS', 'complete_selection_seal': 'PASS',
                   'raw_artifact_manifest': 'PASS', 'independent_nested_partition': 'PASS',
                   'fit_only_timestamp_context': 'PASS', 'paired_bootstrap_comparisons': 6},
        'limits': ['Candidate selection ordering checked for all 39 rows; only selected D metrics replayed.',
                   'Grouped saved scores evaluated but not refitted; independent primal solver tests cover their algebra.',
                   'No novelty, causal effect, untouched population or universal performance claim.'],
        'run_manifest_sha256': digest(directory / 'manifest.json'),
        'selection_seal_sha256': digest(directory / 'SELECTIONS-FROZEN.json'),
        'audit_source_sha256': digest(__file__), 'primary_ndcg': ndcg,
        'elapsed_seconds': time.monotonic()-started}
    output.mkdir(parents=True)
    write_json(output / 'audit.json', receipt)
    write_json(output / 'SHA256.json', {'audit.json': digest(output / 'audit.json')})
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.run, arguments.out)
