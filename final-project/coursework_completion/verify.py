"""Independent numerical and integrity audit of the bounded coursework study.

No scientific implementation imports, no TEST labels, and no additional tuning.
Only selected configurations are replayed after the all-seed selection seal passes.
Literal metric definitions are copied from our independent earlier audit, not from
the scientific runner/metric code. This script has its own source hash.
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

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
FAMILIES = ('mixed', 'meta', 'tuned_switch', 'tuned_rrf')
ROLES = ('EASE', 'SLIM', 'context', 'fixed_switch', 'fixed_rrf', *FAMILIES)
GRIDS = {'mixed': [[6, 2, 2], [4, 4, 2], [4, 2, 4]], 'meta': [.1, 1., 10., 100.],
         'tuned_switch': [2, 3, 4], 'tuned_rrf': [10, 60, 100]}
SELECTION_FILES = {'ids.json', 'cohorts.json', 'input-signature.json', 'candidate-grid.json',
                   'selections.json', 'predictions.npz', 'candidate-predictions.npz'}
SOURCES = {'coursework_completion/__init__.py', 'coursework_completion/models.py',
           'coursework_completion/run.py', 'coursework_completion/PROTOCOL.md', 'freeze.py',
           'study.py', 'metrics.py', 'societal.py', 'hybrid_constraints.py',
           'exploratory/categorical_reconstruction/group_analysis.py'}


def meta_level_literal(binary, genres, penalty):
    """Two independent augmented least-squares solutions; no normal-equation helper."""
    n, m = binary.shape
    g = genres.shape[1]
    ridge = math.sqrt(penalty) * np.eye(g)
    profiles = np.linalg.lstsq(np.vstack([genres, ridge]),
                               np.vstack([binary.T, np.zeros((g, n))]), rcond=None)[0].T
    factors = np.linalg.lstsq(np.vstack([profiles, ridge]),
                              np.vstack([binary, np.zeros((g, m))]), rcond=None)[0].T
    return profiles @ factors.T


def eligible_orders(scores, observed):
    return [sorted([j for j in range(1, scores.shape[1]) if not observed[u, j]],
                   key=lambda j: (-scores[u, j], j)) for u in range(len(scores))]


def rrf_literal(experts, observed, constant):
    result = np.zeros(observed.shape, dtype=float)
    for expert in experts:
        for user, ordered in enumerate(eligible_orders(expert, observed)):
            for rank, item in enumerate(ordered, 1):
                result[user, item] += 1. / (constant + rank)
    return result


def mixed_literal(experts, observed, quotas):
    orders = [eligible_orders(expert, observed) for expert in experts]
    rows = []
    for user in range(len(observed)):
        accepted, taken = [], [0] * len(quotas)
        while len(accepted) < 10:
            available = {e: next((j for j in orders[e][user] if j not in accepted), None)
                         for e in range(len(quotas)) if taken[e] < quotas[e]}
            available = {e: item for e, item in available.items() if item is not None}
            if not available:
                break
            priorities = {e: (len(accepted) + 1) * quotas[e] - 10 * taken[e] for e in available}
            winner = sorted(available, key=lambda e: (-priorities[e], e))[0]
            accepted.append(available[winner])
            taken[winner] += 1
        accepted += [j for j in orders[0][user] if j not in accepted][:10-len(accepted)]
        require(len(accepted) == 10, 'Literal mixed list exhausted')
        rows.append(accepted)
    return np.asarray(rows, dtype=int)


def context_literal(bundle, arrays):
    name = bundle['selection']['families']['context']
    coefficients = bundle['models'][name]['coefficients']
    weights = coefficients['weights']
    active = bundle['active_experts']
    order = bundle['expert_order']
    standardized = np.stack([(arrays['raw_scores'][order.index(expert)]
                              - arrays['score_mean'][e, :, None])
                             / arrays['score_scale'][e, :, None]
                             for e, expert in enumerate(active)])
    result = np.full(standardized.shape[1:], coefficients['intercept'], dtype=float)
    for e, expert in enumerate(active):
        z = standardized[e]
        result += z * weights[expert]
        result += z * arrays['activity'][:, None] * weights[expert + '*activity']
        result += z * arrays['entropy'][:, None] * weights[expert + '*genre_entropy']
        result += z * arrays['popularity'][None, :] * weights[expert + '*item_popularity']
    result += standardized.std(axis=0) * weights['expert_disagreement']
    return result

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


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


def basic_literal(ranks, truth, binary):
    counts = binary.sum(axis=0)
    total, catalog = int(binary.sum()), binary.shape[1] - 1
    values, exposed = {}, set()
    for row, relevant in enumerate(truth):
        if not relevant:
            continue
        recs = list(map(int, ranks[row]))
        require(all(0 < item < binary.shape[1] and not binary[row, item] for item in recs),
                'Ineligible recommendation')
        values[row] = literal_user_metrics(recs, relevant, counts, total, catalog)
        exposed.update(recs)
    aggregate = {key: float(np.mean([value[key] for value in values.values()]))
                 for key in next(iter(values.values()))}
    aggregate['coverage@10'] = len(exposed) / catalog
    return {'users': len(values), 'aggregate': aggregate}, values


def verify_integrity(directory, public):
    protocol = read_json(directory / 'protocol.json')
    manifest = read_json(directory / 'manifest.json')
    seal = read_json(directory / 'SELECTIONS-FROZEN.json')
    require(protocol['seeds'] == list(SEEDS) and protocol['roles'] == list(ROLES)
            and protocol['grids'] == GRIDS and protocol['candidate_count'] == 39,
            'Protocol scope differs')
    require(not protocol['original_test_read'] and not protocol['fresh_holdout']
            and not manifest['original_test_read'] and not manifest['fresh_holdout']
            and manifest['status'] == 'complete'
            and manifest['assessment_after_all_seed_selections'], 'Invalid declared stage')
    require(set(protocol['source_sha256']) == SOURCES
            and protocol['source_sha256'] == manifest['source_sha256'] == seal['source_sha256'],
            'Scientific source inventory changed')
    for name, expected in protocol['source_sha256'].items():
        require(digest(ROOT / name) == expected, 'Scientific source changed: ' + name)
    for name, expected in protocol['input_artifacts_sha256'].items():
        require(Path(name).name != 'test.tsv', 'Unexpected TEST input')
        require(digest(name) == expected, 'Source artifact changed: ' + name)
    require(seal['seeds'] == list(SEEDS) and seal['status'] == 'all_selections_frozen'
            and seal['assessment_labels_parsed'] is False
            and set(seal['selection_manifest_sha256']) == set(map(str, SEEDS)), 'Incomplete global seal')
    require(digest(directory / 'SELECTIONS-FROZEN.json') == manifest['selection_seal_sha256']
            == (directory / 'SELECTIONS-FROZEN.sha256').read_text().strip(), 'Global seal hash differs')
    require(digest(directory / 'protocol.json') == manifest['protocol_sha256'] == seal['protocol_sha256'],
            'Protocol seal differs')
    require(digest(directory / 'aggregates.json') == manifest['aggregate_sha256'], 'Aggregate hash differs')
    required = {'protocol.json', 'SELECTIONS-FROZEN.json', 'SELECTIONS-FROZEN.sha256',
                'ASSESSMENT-OPENED.json', 'aggregates.json'}
    for seed in SEEDS:
        required |= {f'{seed}/{name}' for name in SELECTION_FILES | {
            'selection-manifest.json', 'assessment-metrics.json', 'assessment-per-user.json'}}
        selection = read_json(directory / str(seed) / 'selection-manifest.json')
        require(digest(directory / str(seed) / 'selection-manifest.json')
                == seal['selection_manifest_sha256'][str(seed)], 'Seed selection seal changed')
        require(selection['seed'] == seed and selection['roles'] == list(ROLES)
                and selection['candidate_count'] == 13 and selection['assessment_labels_parsed'] is False
                and set(selection['artifact_sha256']) == SELECTION_FILES, 'Incomplete seed seal')
        for name, expected in selection['artifact_sha256'].items():
            require(digest(directory / str(seed) / name) == expected, 'Selected artifact changed')
    require(set(manifest['output_sha256']) == required, 'Completed output inventory differs')
    for name, expected in manifest['output_sha256'].items():
        require(digest(directory / name) == expected, 'Completed artifact hash differs: ' + name)
    opened = read_json(directory / 'ASSESSMENT-OPENED.json')
    require(opened['status'] == 'complete' and opened['all_seed_choices_frozen']
            and opened['selection_seal_sha256'] == manifest['selection_seal_sha256'], 'Assessment stage differs')
    public_hashes = read_json(public / 'SHA256.json')
    require(set(public_hashes) == {'aggregates.json', 'protocol.json', 'provenance.json'},
            'Public evidence inventory differs')
    for name, expected in public_hashes.items():
        raw_name = 'manifest.json' if name == 'provenance.json' else name
        require(digest(public / name) == expected == digest(directory / raw_name),
                'Public and private evidence differ: ' + name)
    return protocol, manifest, public_hashes


def read_truth(path, users, item_index):
    truth = [set() for _ in users]
    user_index = {user: row for row, user in enumerate(users)}
    with Path(path).open() as stream:
        for record in csv.DictReader(stream, delimiter='\t'):
            row, item = user_index[record['user_id']], item_index[record['item_id']]
            require(item > 0 and item not in truth[row], 'Invalid/duplicate relevance pair')
            truth[row].add(item)
    require(all(truth), 'Missing relevance user')
    return truth


def load_audit_data(directory, protocol, seed):
    source = Path(protocol['source_root']) / str(seed)
    bundle = read_json(source / 'freeze.json')
    ids = read_json(directory / str(seed) / 'ids.json')
    names = ('users', 'items', 'raw_scores', 'train_mask', 'score_mean', 'score_scale',
             'activity', 'entropy', 'popularity', 'raw_activity', 'genre_binary', 'genre_fraction', 'groups')
    with np.load(source / 'frozen.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in names}  # Never valid_mask.
    users, items = arrays['users'].tolist(), arrays['items'].tolist()
    require(ids == {'users': users, 'items': items} and len(users) == 943
            and len(items) == 1683 and items[0] == '[PAD]', 'Source/catalog IDs differ')
    ui, ii = {u: r for r, u in enumerate(users)}, {i: c for c, i in enumerate(items)}
    binary = np.zeros((len(users), len(items)), dtype=bool)
    with (source / 'train.tsv').open() as stream:
        for record in csv.DictReader(stream, delimiter='\t'):
            row, item = ui[record['user_id']], ii[record['item_id']]
            require(item > 0 and not binary[row, item], 'Invalid/duplicate TRAIN pair')
            binary[row, item] = True
    require(np.array_equal(binary, arrays['train_mask']), 'Frozen TRAIN mask differs from identities')
    cohorts = read_json(directory / str(seed) / 'cohorts.json')
    require(cohorts == read_json(source / 'provenance/cohorts.json'), 'Frozen cohorts differ')
    ordered = np.asarray(sorted(users))
    np.random.default_rng(seed).shuffle(ordered)
    meta = set(ordered[:len(users)//2])
    rest = np.asarray(sorted(ordered[len(users)//2:]))
    np.random.default_rng(seed + 917).shuffle(rest)
    expected = {'meta_fit': meta, 'calibration': set(rest[:len(rest)//2]),
                'development': set(rest[len(rest)//2:])}
    require(all(set(cohorts[name]) == population for name, population in expected.items()),
            'Cohorts do not match original ID-only partition')
    fractions = arrays['genre_fraction']
    genres = [set(np.flatnonzero(row)) for row in arrays['genre_binary']]
    require(all(genres[1:]) and not genres[0], 'Invalid source genres')
    require(np.allclose(fractions[1:].sum(axis=1), 1), 'Genre fractions differ')
    profiles = binary @ fractions / binary.sum(axis=1)[:, None]
    counts = binary.sum(axis=0)
    activity = np.array(['sparse', 'medium', 'dense'])[
        np.searchsorted(np.quantile(binary.sum(axis=1), [1/3, 2/3]), binary.sum(axis=1), side='right')]
    head = set(sorted(range(1, len(items)), key=lambda i: (-counts[i], items[i]))[:math.ceil(.2*(len(items)-1))])
    truth = read_truth(source / 'valid.tsv', users, ii)  # Called only after all seals passed.
    require(not any(binary[row, list(values)].any() for row, values in enumerate(truth)), 'Relevance overlaps TRAIN')
    contexts = (binary, counts, items, genres, fractions, profiles, activity, head)
    signature = read_json(directory / str(seed) / 'input-signature.json')
    expected_signature = {'freeze_sha256': digest(source / 'freeze.json'),
        'root_manifest_sha256': digest(Path(protocol['source_root']) / 'manifest.json'),
        'metadata_sha256': digest(protocol['items']),
        'users_sha256': hashlib.sha256(json.dumps(users).encode()).hexdigest(),
        'items_sha256': hashlib.sha256(json.dumps(items).encode()).hexdigest(),
        'train_sha256': digest(source / 'train.tsv'), 'valid_sha256': digest(source / 'valid.tsv')}
    require(signature == expected_signature, 'Shared source identity signatures differ')
    return bundle, arrays, cohorts, truth, contexts


def subset_truth(truth, users, allowed):
    allowed = set(allowed)
    return [values if user in allowed else set() for user, values in zip(users, truth)]


def compare_ranks(actual, expected, label):
    require(np.array_equal(actual, expected), label + ' rankings differ from independent computation')


def audit_seed(directory, protocol, seed, checks):
    folder = directory / str(seed)
    bundle, arrays, cohorts, truth, context = load_audit_data(directory, protocol, seed)
    users, items = arrays['users'].tolist(), arrays['items'].tolist()
    binary = arrays['train_mask']
    with np.load(folder / 'predictions.npz', allow_pickle=False) as archive:
        ranks = {name: archive[name] for name in archive.files}
    require(set(ranks) == set(ROLES), 'Missing assessment role')
    with np.load(folder / 'candidate-predictions.npz', allow_pickle=False) as archive:
        candidate_ranks = {name: archive[name] for name in archive.files}
    grid = read_json(folder / 'candidate-grid.json')
    choices = read_json(folder / 'selections.json')
    require(set(choices['families']) == set(FAMILIES) and set(choices['references']) == set(ROLES)-set(FAMILIES),
            'Family/reference choices differ')
    expected_ids = [f'{family}-{index}' for family in FAMILIES for index in range(len(GRIDS[family]))]
    require([row['candidate_id'] for row in grid] == expected_ids and set(candidate_ranks) == set(expected_ids),
            'Candidate grid differs')
    dev_truth = subset_truth(truth, users, cohorts['development'])
    for family in FAMILIES:
        pool = [row for row in grid if row['family'] == family]
        parameter = {'mixed': 'quotas', 'meta': 'lambda', 'tuned_switch': 'group_count', 'tuned_rrf': 'offset'}[family]
        require([row['settings'][parameter] for row in pool] == GRIDS[family], 'Fixed candidate parameters changed')
        for row in pool:
            measured, _ = basic_literal(candidate_ranks[row['candidate_id']], dev_truth, binary)
            checks.tree(row['development'], measured, f'{seed}.{row["candidate_id"]}.development')
        chosen = max(pool, key=lambda row: row['development']['aggregate']['ndcg@10'])
        require(choices['families'][family] == chosen, 'Selection is not the first exact development maximum')
        compare_ranks(ranks[family], candidate_ranks[chosen['candidate_id']], f'{seed}.{family}.selected')

    # Independently replay the selected settings, with no search or reselection.
    experts = dict(zip(bundle['expert_order'], arrays['raw_scores']))
    active = bundle['active_experts']
    aliases = {}
    for model in ('EASE', 'SLIMElastic', 'GenreContent', 'ExactPop'):
        found = [name for name in active if bundle['sources'][name]['manifest']['model'] == model]
        require(len(found) == 1, 'Ambiguous reference model')
        aliases[model] = found[0]
    base_ranks = {name: rank_literal(experts[name], binary) for name in active}
    compare_ranks(ranks['EASE'], base_ranks[aliases['EASE']], f'{seed}.EASE')
    compare_ranks(ranks['SLIM'], base_ranks[aliases['SLIMElastic']], f'{seed}.SLIM')
    compare_ranks(ranks['context'], rank_literal(context_literal(bundle, arrays), binary), f'{seed}.context')
    thresholds = np.quantile(binary.sum(axis=1), [1/3, 2/3])
    groups = np.searchsorted(thresholds, binary.sum(axis=1), side='right')
    require(np.array_equal(arrays['groups'], groups), 'Frozen fixed-switch groups differ')
    fixed_policy = bundle['models']['group-switch']['experts']
    fixed = np.stack([base_ranks[fixed_policy[str(group)]][row] for row, group in enumerate(groups)])
    compare_ranks(ranks['fixed_switch'], fixed, f'{seed}.fixed_switch')

    selected = choices['families']
    mix_names = [aliases[name] for name in ('EASE', 'GenreContent', 'ExactPop')]
    require(selected['mixed']['settings']['expert_order'] == mix_names, 'Mixed expert order changed')
    compare_ranks(ranks['mixed'], mixed_literal([experts[name] for name in mix_names], binary,
                                               selected['mixed']['settings']['quotas']), f'{seed}.mixed')
    meta_scores = np.zeros(binary.shape)
    meta_scores[:, 1:] = meta_level_literal(binary[:, 1:], arrays['genre_fraction'][1:],
                                           selected['meta']['settings']['lambda'])
    compare_ranks(ranks['meta'], rank_literal(meta_scores, binary), f'{seed}.meta')
    chosen_switch = selected['tuned_switch']['settings']
    group_count = chosen_switch['group_count']
    bounds = np.quantile(binary.sum(axis=1), np.arange(1, group_count)/group_count)
    checks.number(max(abs(np.asarray(chosen_switch['thresholds']) - bounds), default=0), 0, 'Switch thresholds')
    assignments = np.searchsorted(bounds, binary.sum(axis=1), side='right')
    meta_truth = subset_truth(truth, users, cohorts['meta_fit'])
    utilities = {name: basic_literal(base_ranks[name], meta_truth, binary)[1] for name in active}
    global_means = {name: np.mean([v['ndcg@10'] for v in utilities[name].values()]) for name in active}
    global_best = max(active, key=lambda name: global_means[name])
    require(chosen_switch['global_winner'] == global_best and chosen_switch['expert_order'] == active,
            'Global switching policy differs')
    switch_winners = {}
    for group in range(group_count):
        members = [row for row, user in enumerate(users) if user in set(cohorts['meta_fit']) and assignments[row] == group]
        means = {name: float(np.mean([utilities[name][row]['ndcg@10'] for row in members]))
                 if members else global_means[name] for name in active}
        switch_winners[str(group)] = max(active, key=lambda name: means[name])
        checks.tree(chosen_switch['meta_fit_group_means'][str(group)], means, f'{seed}.switch.group{group}')
        checks.number(chosen_switch['meta_fit_group_users'][str(group)], len(members), 'Switch fit group users')
    require(chosen_switch['winners'] == switch_winners, 'Switching winners differ')
    switched = np.stack([base_ranks[switch_winners[str(group)]][row] for row, group in enumerate(assignments)])
    compare_ranks(ranks['tuned_switch'], switched, f'{seed}.tuned_switch')
    offset = selected['tuned_rrf']['settings']['offset']
    require(selected['tuned_rrf']['settings']['expert_order'] == active, 'RRF expert order changed')
    for constant in sorted({60, offset}):
        result = rank_literal(rrf_literal([experts[name] for name in active], binary, constant), binary)
        if constant == 60:
            compare_ranks(ranks['fixed_rrf'], result, f'{seed}.fixed_rrf')
        if constant == offset:
            compare_ranks(ranks['tuned_rrf'], result, f'{seed}.tuned_rrf')

    assessment = read_json(folder / 'assessment-metrics.json')
    per_user = read_json(folder / 'assessment-per-user.json')
    assessed_truth = subset_truth(truth, users, cohorts['calibration'])
    independent_results = {}
    for role in ROLES:
        measured, _ = evaluate_literal(ranks[role], assessed_truth, context)
        checks.tree(assessment[role], measured, f'{seed}.{role}.assessment')
        checks.number(assessment[role]['positive_pairs'], sum(map(len, assessed_truth)), 'Assessment positive pairs')
        basic, values = basic_literal(ranks[role], assessed_truth, binary)
        require(set(per_user[role]) == set(cohorts['calibration']), 'Per-user cohort differs')
        for row, metrics in values.items():
            checks.tree(per_user[role][users[row]], metrics, f'{seed}.{role}.per_user')
        independent_results[role] = measured
    return {'seed': seed, 'assessment_users': len(cohorts['calibration']),
            'assessment_positive_pairs': sum(map(len, assessed_truth)),
            'selected_candidates': {family: selected[family]['candidate_id'] for family in FAMILIES},
            'selected_model_rankings_replayed': list(ROLES),
            'candidate_development_metrics_replayed': len(grid)}, independent_results


def audit(directory, public, output):
    directory, public, output = Path(directory), Path(public), Path(output)
    require(not output.exists(), 'Refusing to overwrite audit receipt')
    started = time.monotonic()
    protocol, manifest, public_hashes = verify_integrity(directory, public)
    aggregate = read_json(public / 'aggregates.json')
    require(aggregate['status'] == 'complete' and set(aggregate['seeds']) == set(map(str, SEEDS))
            and not aggregate['fresh_holdout'] and not aggregate['original_test_read'], 'Public scope differs')
    checks, summaries, measurements = Checks(), [], {}
    for seed in SEEDS:
        summary, measured = audit_seed(directory, protocol, seed, checks)
        summaries.append(summary)
        measurements[str(seed)] = measured
        reported = aggregate['seeds'][str(seed)]
        checks.tree(reported['assessment'], measured, f'{seed}.public assessment')
        require(reported['candidates'] == read_json(directory / str(seed) / 'candidate-grid.json'), 'Public grid differs')
        require(reported['selections'] == read_json(directory / str(seed) / 'selections.json')['families'],
                'Public choices differ')
        print(json.dumps({'seed': seed, 'status': 'independently verified', 'numeric_checks': checks.numeric}), flush=True)
    means = {role: {metric: float(np.mean([measurements[str(seed)][role]['aggregate'][metric] for seed in SEEDS]))
                   for metric in measurements[str(SEEDS[0])][role]['aggregate']} for role in ROLES}
    checks.tree(aggregate['mean_metrics'], means, 'Public descriptive seed means')
    verify_integrity(directory, public)  # Changes during replay also fail.
    receipt = {'schema_version': 1, 'status': 'passed', 'study': protocol['study'],
        'checks_completed': checks.numeric, 'maximum_absolute_error': checks.maximum_error,
        'scientific_public_artifacts': public_hashes, 'scientific_source_sha256': protocol['source_sha256'],
        'verifier_source_sha256': digest(Path(__file__)),
        'selection_barrier_sha256': manifest['selection_seal_sha256'],
        'assessment_manifest_sha256': digest(public / 'provenance.json'),
        'seeds': summaries, 'candidate_development_metrics_replayed': 39,
        'assessment_role_seed_pairs_replayed': 27, 'selected_model_equations_replayed': list(ROLES),
        'original_test_read': False, 'valid_mask_array_read': False,
        'fresh_holdout': False, 'seconds': time.monotonic() - started,
        'limitations': ['The 236 calibration users per seed were used in earlier policy fitting and exploration.',
            'Seeds reuse people and observations; means are descriptive and are not independent population replications.',
            'This audit replays saved candidate metrics and selected model equations; it does not repeat the hyperparameter search.',
            'Original source score models are treated as hash-verified frozen inputs; their training is not rerun.',
            'The four completion families are established coursework methods, not claimed novel architectures.']}
    output.mkdir(parents=True)
    write_json(output / 'audit.json', receipt)
    write_json(output / 'SHA256.json', {'audit.json': digest(output / 'audit.json'),
                                      'verify.py': digest(Path(__file__))})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.evidence, args.out)
    print(json.dumps({'status': result['status'], 'numeric_checks': result['checks_completed'],
                      'maximum_absolute_error': result['maximum_absolute_error'], 'seconds': result['seconds']}))
