"""Validation-only hybrid study. Never reads test labels or test scores."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from metrics import evaluate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pairs(path):
    with path.open() as stream:
        return [(r['user_id'], r['item_id']) for r in csv.DictReader(stream, delimiter='\t')]


def grouped(pairs):
    result = defaultdict(set)
    for u, i in pairs:
        result[u].add(i)
    return result


def partition_users(users, seed):
    """User-disjoint meta-fit and development cohorts; labels never select users."""
    if len(users) < 4:
        raise ValueError('Need at least four users')
    ordered = np.asarray(sorted(users))
    np.random.default_rng(seed).shuffle(ordered)
    cut = len(ordered) // 2
    return set(ordered[:cut]), set(ordered[cut:])


def load_runs(paths):
    reference = None
    experts, sources = {}, {}
    for path in paths:
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete source run: {path}')
        if manifest['test_evaluated']:
            raise ValueError('Use validation-only runs for this development study')
        for part in ('train', 'valid'):
            if digest(path / f'{part}.tsv') != manifest['split_sha256'][part]:
                raise ValueError('Split contents do not match source manifest')
        with np.load(path / 'valid-scores.npz', allow_pickle=False) as archive:
            users, items, scores = archive['users'].tolist(), archive['items'].tolist(), archive['scores'].astype(float)
        if len(set(users)) != len(users) or len(set(items)) != len(items):
            raise ValueError('Duplicate score IDs')
        if scores.shape != (len(users), len(items)) or not np.isfinite(scores).all():
            raise ValueError('Invalid score array')
        signature = (manifest['split_sha256'], manifest['data_sha256'], users, items)
        if reference is not None and signature != reference:
            raise ValueError('Experts have different split, dataset or score ID order')
        reference = signature
        label = path.name
        if label in experts:
            raise ValueError('Run directory names must be distinct')
        experts[label] = scores
        sources[label] = {'manifest': manifest, 'scores_sha256': digest(path / 'valid-scores.npz')}
    if len(experts) < 2:
        raise ValueError('Need at least two expert runs')
    train, valid = read_pairs(paths[0] / 'train.tsv'), read_pairs(paths[0] / 'valid.tsv')
    if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) & set(valid):
        raise ValueError('Invalid interaction partitions')
    if set(grouped(valid)) != set(users):
        raise ValueError('Score users differ from validation users')
    if set(i for _, i in train + valid) - set(items[1:]):
        raise ValueError('Interaction IDs missing from score catalog')
    return users, items, experts, train, valid, sources


def load_genres(path, items):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    mapping = {r['item_id:token']: set(r['class:token_seq'].split()) for r in rows}
    if any(not mapping.get(i) for i in items[1:]):
        raise ValueError('Missing item genres')
    vocabulary = sorted(set().union(*mapping.values()))
    binary = np.zeros((len(items), len(vocabulary)))
    for row, item in enumerate(items[1:], 1):
        binary[row] = [genre in mapping[item] for genre in vocabulary]
    fractional = binary / np.maximum(binary.sum(axis=1, keepdims=True), 1)
    return binary, fractional


def jsd(p, q):
    """Jensen-Shannon divergence in bits; last dimension is genre distribution."""
    p, q = np.asarray(p), np.asarray(q)
    middle = (p + q) / 2
    def kl(a):
        return np.sum(a * np.log2(np.maximum(a, 1e-15) / np.maximum(middle, 1e-15)), axis=-1)
    return (kl(p) + kl(q)) / 2


def normalize_scores(scores, candidates):
    """Per-user, per-expert z-score using eligible items only, without labels."""
    result = np.zeros_like(scores, dtype=float)
    for u, eligible in enumerate(candidates):
        values = scores[u, eligible]
        result[u, eligible] = (values - values.mean()) / max(values.std(), 1e-8)
    return result


def build_features(base, activity, entropy, popularity, variant):
    """Linear contextual weights, not probabilistic/convex mixture weights."""
    columns = [base]
    if variant in ('user', 'context'):
        columns += [base * activity[:, None, None], base * entropy[:, None, None]]
    if variant in ('item', 'context'):
        columns += [base * popularity[None, :, None]]
    if variant in ('disagreement', 'context'):
        columns += [np.std(base, axis=2, keepdims=True)]
    return np.concatenate(columns, axis=2)


def fit_ridge(features, y, penalty):
    """Minimize mean squared error + penalty * squared coefficient norm."""
    if penalty <= 0 or len(y) < 1:
        raise ValueError('Positive penalty and nonempty observations required')
    center_x, center_y = features.mean(axis=0), y.mean()
    x = features - center_x
    weights = np.linalg.solve(x.T @ x / len(y) + penalty * np.eye(x.shape[1]),
                              x.T @ (y - center_y) / len(y))
    return weights, float(center_y - center_x @ weights)


def ranking(scores, eligible, k):
    if len(eligible) < k:
        raise ValueError('Insufficient candidates')
    return eligible[np.argsort(-scores[eligible], kind='stable')[:k]]


def rrf(experts, candidates, constant=60):
    result = np.zeros_like(next(iter(experts.values())))
    for scores in experts.values():
        for u, eligible in enumerate(candidates):
            ranked = ranking(scores[u], eligible, len(eligible))
            result[u, ranked] += 1 / (constant + np.arange(1, len(ranked) + 1))
    return result


def rerank(scores, eligible, genre_binary, genre_fraction, profile, head, k,
           strength, mode, pool_size=100):
    """Greedy finite-pool reranker. strength=0 exactly preserves original top-k."""
    if mode not in ('diversity', 'calibration', 'exposure') or not 0 <= strength <= 1:
        raise ValueError('Invalid reranking objective/strength')
    pool = ranking(scores, eligible, min(pool_size, len(eligible)))
    if len(pool) < k:
        raise ValueError('Candidate pool smaller than k')
    relevance = scores[pool].copy()
    relevance = (relevance - relevance.min()) / max(np.ptp(relevance), 1e-8)
    selected = []
    selected_positions = []
    genre_sum = np.zeros(genre_fraction.shape[1])
    target_head = float(np.mean(head[1:]))
    for step in range(k):
        if mode == 'diversity':
            if selected:
                a, b = genre_binary[pool], genre_binary[selected]
                intersection = a @ b.T
                union = a.sum(axis=1, keepdims=True) + b.sum(axis=1)[None, :] - intersection
                bonus = 1 - (intersection / np.maximum(union, 1)).max(axis=1)
            else:
                bonus = np.ones(len(pool))
        elif mode == 'calibration':
            distributions = (genre_sum[None, :] + genre_fraction[pool]) / (step + 1)
            bonus = 1 - jsd(profile[None, :], distributions)
        else:
            head_fraction = (sum(head[i] for i in selected) + head[pool]) / (step + 1)
            bonus = 1 - np.abs(head_fraction - target_head)
        objective = (1 - strength) * relevance + strength * bonus
        objective[selected_positions] = -np.inf
        position = int(np.argmax(objective))
        selected_positions.append(position)
        selected.append(int(pool[position]))
        genre_sum += genre_fraction[pool[position]]
    return np.asarray(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--negative-ratio', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.negative_ratio < 1 or args.k < 2 or args.bootstrap < 1:
        parser.error('Require positive negative ratio/bootstrap count and k >= 2')
    start = time.monotonic()
    users, items, experts, train, valid, sources = load_runs(args.runs)
    shared_selection = any(v['manifest'].get('validation_used_for_training', True) for v in sources.values())
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out
    history, truth = grouped(train), grouped(valid)
    item_index = {item: i for i, item in enumerate(items)}
    counts = Counter(item for _, item in train)
    eligible = [np.asarray([i for i, item in enumerate(items) if i and item not in history[u]]) for u in users]
    expected_metadata = next(iter(sources.values()))['manifest']['data_sha256'].get(args.items.name)
    if expected_metadata != digest(args.items):
        raise ValueError('Metadata differs from the source-run dataset')
    binary, fractional = load_genres(args.items, items)
    profiles = np.asarray([fractional[[item_index[i] for i in history[u]]].mean(axis=0) for u in users])
    raw_activity = np.asarray([len(history[u]) for u in users])
    activity = np.log1p(raw_activity)
    activity = (activity - activity.mean()) / max(activity.std(), 1e-8)
    entropy = -np.sum(profiles * np.log2(np.maximum(profiles, 1e-15)), axis=1)
    entropy = (entropy - entropy.mean()) / max(entropy.std(), 1e-8)
    popularity = np.log1p([counts.get(i, 0) for i in items])
    popularity = (popularity - popularity[1:].mean()) / max(popularity[1:].std(), 1e-8)
    head = np.zeros(len(items), dtype=bool)
    head[np.argsort(-np.asarray([counts.get(i, 0) for i in items[1:]]), kind='stable')[:int(np.ceil(.2*(len(items)-1)))]+1] = True
    thresholds = np.quantile(raw_activity, [1/3, 2/3])
    groups = np.searchsorted(thresholds, raw_activity, side='right')
    fit_users, dev_users = partition_users(users, args.seed)
    fit_indices = [i for i, u in enumerate(users) if u in fit_users]
    dev_indices = [i for i, u in enumerate(users) if u in dev_users]
    write_json(out/'cohorts.json', {'meta_fit': sorted(fit_users), 'development': sorted(dev_users),
                                   'activity_thresholds': thresholds.tolist()})
    # Same observations/negative samples for all feature ablations and penalties.
    rng = np.random.default_rng(args.seed)
    observations, labels, pair_users, pair_positive, pair_negative = [], [], [], [], []
    for u in fit_indices:
        positive = np.asarray(sorted(item_index[i] for i in truth[users[u]]))
        available = np.setdiff1d(eligible[u], positive)
        negative = rng.choice(available, min(len(available), len(positive)*args.negative_ratio), replace=False)
        observations += [(u, int(i)) for i in np.concatenate([positive, negative])]
        labels += [1.] * len(positive) + [0.] * len(negative)
        for positive_item in positive:
            sampled = rng.choice(available, min(len(available), args.negative_ratio), replace=False)
            pair_users.extend([u]*len(sampled))
            pair_positive.extend([int(positive_item)]*len(sampled))
            pair_negative.extend(sampled.tolist())
    rows, cols = np.asarray(observations).T
    labels = np.asarray(labels)
    active_experts = {n: s for n, s in experts.items() if sources[n]['manifest']['model'] != 'Random'}
    if len(active_experts) < 2:
        raise ValueError('Need two non-random experts for fusion')
    base = np.stack([normalize_scores(s, eligible) for s in active_experts.values()], axis=2)
    models = dict(experts)
    models['rrf'] = rrf(active_experts, eligible)
    coefficients = {}
    penalties = [0.001, 0.01, 0.1, 1.0]
    feature_labels = list(active_experts)
    for variant in ('static', 'user', 'item', 'disagreement', 'context'):
        features = build_features(base, activity, entropy, popularity, variant)
        names = feature_labels.copy()
        if variant in ('user', 'context'):
            names += [f'{n}*activity' for n in feature_labels] + [f'{n}*genre_entropy' for n in feature_labels]
        if variant in ('item', 'context'):
            names += [f'{n}*item_popularity' for n in feature_labels]
        if variant in ('disagreement', 'context'):
            names += ['expert_disagreement']
        for penalty in penalties:
            label = f'{variant}-ridge-{penalty:g}'
            weights, bias = fit_ridge(features[rows, cols], labels, penalty)
            models[label] = features @ weights + bias
            coefficients[label] = {'weights': dict(zip(names, weights.tolist())), 'intercept': bias,
                                    'penalty': penalty, 'training_examples': len(labels)}
        if variant in ('static', 'context'):
            differences = features[pair_users, pair_positive] - features[pair_users, pair_negative]
            for penalty in penalties:
                label = f'{variant}-pairwise-{penalty:g}'
                # A score difference should approach 1 for each sampled positive/negative pair.
                # Intercept cancels; no centering is applied to pairwise feature differences.
                weights = np.linalg.solve(differences.T @ differences / len(differences) + penalty*np.eye(differences.shape[1]), differences.mean(axis=0))
                models[label] = features @ weights
                coefficients[label] = {'weights': dict(zip(names, weights.tolist())), 'intercept': 0.,
                                        'penalty': penalty, 'training_pairs': len(differences),
                                        'objective': 'mean squared positive-negative score margin error + L2'}
    write_json(out/'coefficients.json', coefficients)
    records = {}
    per_user = {}
    all_recs = {}

    def assess(label, rec_indices, cohort=dev_indices, save=True):
        recs = {users[u]: [items[i] for i in rec_indices[u]] for u in cohort}
        result = evaluate(recs, {users[u]: truth[users[u]] for u in cohort}, history, items[1:], counts, args.k)
        for u in cohort:
            selected = rec_indices[u]
            g = binary[selected]
            intersection = g @ g.T
            union = g.sum(axis=1, keepdims=True) + g.sum(axis=1)[None, :] - intersection
            distances = 1 - intersection / np.maximum(union, 1)
            result['per_user'][users[u]]['diversity'] = float(distances[np.triu_indices(args.k, 1)].mean())
            result['per_user'][users[u]]['calibration_jsd'] = float(jsd(profiles[u], fractional[selected].mean(axis=0)))
            result['per_user'][users[u]]['head_exposure'] = float(np.mean(head[selected]))
        for metric in ('diversity', 'calibration_jsd', 'head_exposure'):
            result['aggregate'][metric] = float(np.mean([v[metric] for v in result['per_user'].values()]))
        grouped_metrics = {}
        for group in range(3):
            members = [u for u in cohort if groups[u] == group]
            if members:
                grouped_metrics[str(group)] = {'users': len(members), **{
                    key: float(np.mean([result['per_user'][users[u]][key] for u in members]))
                    for key in result['aggregate'] if key != f'coverage@{args.k}'}}
        result['groups'] = grouped_metrics
        means = [v[f'ndcg@{args.k}'] for v in grouped_metrics.values()]
        result['aggregate']['activity_ndcg_gap'] = max(means)-min(means)
        result['aggregate']['worst_activity_ndcg'] = min(means)
        result['aggregate']['head_exposure_gap'] = abs(result['aggregate']['head_exposure']-float(head[1:].mean()))
        # Conditional item-group recall; exclude users without positives in that group.
        for group_name, membership in [('head', head), ('tail', ~head)]:
            values = []
            for u in cohort:
                positives = {i for i in truth[users[u]] if membership[item_index[i]]}
                if positives:
                    values.append(len(positives & set(recs[users[u]]))/len(positives))
            result['aggregate'][f'{group_name}_recall'] = float(np.mean(values)) if values else None
        if save:
            records[label] = result
            per_user[label] = result['per_user']
            all_recs[label] = recs
        return result

    rankings = {name: {u: ranking(scores[u], eligible[u], args.k) for u in range(len(users))}
                for name, scores in models.items()}
    for name, recs in rankings.items():
        assess(name, recs)
    metric = f'ndcg@{args.k}'
    # Switch expert by training-activity group using only meta-fit cohort labels.
    fit_results = {name: assess(name, rankings[name], fit_indices, save=False) for name in active_experts}
    switch = {}
    for group in range(3):
        candidates = [name for name in active_experts if str(group) in fit_results[name]['groups']]
        switch[group] = max(candidates, key=lambda n: fit_results[n]['groups'][str(group)][metric]) if candidates else next(iter(active_experts))
    rankings['group-switch'] = {u: rankings[switch[int(groups[u])]][u] for u in range(len(users))}
    assess('group-switch', rankings['group-switch'])
    write_json(out/'switching.json', switch)
    winners = {family: max([n for n in models if n.startswith(family+'-ridge')], key=lambda n: records[n]['aggregate'][metric])
               for family in ('static', 'user', 'item', 'disagreement', 'context')}
    for family in ('static-pairwise', 'context-pairwise'):
        winners[family] = max([n for n in models if n.startswith(family+'-')], key=lambda n: records[n]['aggregate'][metric])
    best_expert = max(active_experts, key=lambda n: records[n]['aggregate'][metric])
    # Rerank both strongest expert and contextual model to expose simple-baseline tradeoffs.
    for base_name in (best_expert, winners['context']):
        for mode in ('diversity', 'calibration', 'exposure'):
            for strength in (.2, .5, .8):
                label = f'{base_name}/{mode}-{strength}'
                recs = {u: rerank(models[base_name][u], eligible[u], binary, fractional,
                                  profiles[u], head, args.k, strength, mode) for u in dev_indices}
                assess(label, recs)
    # Contrast equal-rank hybrid before/after per-expert reranking with a fixed setting.
    # Promote each reranked top-k, then retain all remaining items in original order.
    # Both orderings use full candidate support; neither truncates experts to top-k.
    for mode in ('diversity', 'calibration', 'exposure'):
        before, after = {}, {}
        for u in dev_indices:
            fused = np.zeros(len(items))
            for name in active_experts:
                reordered = rerank(models[name][u], eligible[u], binary, fractional, profiles[u], head, args.k, .5, mode)
                original = ranking(models[name][u], eligible[u], len(eligible[u]))
                promoted = set(reordered)
                full_order = np.concatenate([reordered, [i for i in original if i not in promoted]])
                fused[full_order] += 1/(60+np.arange(1,len(full_order)+1))
            before[u] = ranking(fused, eligible[u], args.k)
            after[u] = rerank(models['rrf'][u], eligible[u], binary, fractional, profiles[u], head, args.k, .5, mode)
        assess(f'rerank-experts-then-rrf/{mode}', before)
        assess(f'rrf-then-rerank/{mode}', after)
    # Candidate feasibility is label-free: a top-k list cannot contain more tail
    # items than are available in its candidate pool.
    budget_diagnostics = []
    target_tail = args.k - int(round(float(head[1:].mean()) * args.k))
    for base_name in (best_expert, winners['context']):
        for pool_size in (20, 50, 100, 300, 1000):
            possible, floors = [], []
            for u in dev_indices:
                pool = ranking(models[base_name][u], eligible[u], min(pool_size, len(eligible[u])))
                tail_count = int(np.sum(~head[pool]))
                possible.append(tail_count >= target_tail)
                floors.append(max(0, args.k-tail_count)/args.k)
            budget_diagnostics.append({'model': base_name, 'pool': pool_size,
                'fraction_quota_feasible': float(np.mean(possible)),
                'mean_lower_bound_head_exposure': float(np.mean(floors))})
        # Exact smallest score-prefix containing the target number of tail items,
        # with a minimum pool of 100. No relevance labels influence expansion.
        adaptive, full = {}, {}
        sizes = []
        for u in dev_indices:
            order = ranking(models[base_name][u], eligible[u], len(eligible[u]))
            tails = np.flatnonzero(~head[order])
            minimum = int(tails[target_tail-1]+1) if len(tails) >= target_tail else len(order)
            size = min(len(order), max(100, minimum))
            sizes.append(size)
            adaptive[u] = rerank(models[base_name][u], eligible[u], binary, fractional,
                                  profiles[u], head, args.k, 1., 'exposure', size)
            full[u] = rerank(models[base_name][u], eligible[u], binary, fractional,
                              profiles[u], head, args.k, 1., 'exposure', len(order))
        assess(f'{base_name}/adaptive-pool-exposure', adaptive)
        assess(f'{base_name}/full-pool-exposure', full)
        write_json(out/f'{base_name}-candidate-budget.json', {
            'target_tail_items': target_tail, 'mean_pool': float(np.mean(sizes)),
            'median_pool': float(np.median(sizes)), 'max_pool': int(max(sizes)),
            'fraction_expanded': float(np.mean(np.asarray(sizes)>100)),
            'same_topk_as_full': float(np.mean([np.array_equal(adaptive[u], full[u]) for u in dev_indices]))})
    write_json(out/'candidate-feasibility.json', budget_diagnostics)
    # User-side utility protection: choose an exposure strength per activity group
    # on meta-fit users, requiring at least 95% of that group's reference NDCG.
    # Audit retention on development users; this is not a generalization guarantee.
    base_name = winners['context']
    strengths = (0., .2, .5, .8)
    policy_results, policy_recs = {}, {}
    for strength in strengths:
        recs = {u: rerank(models[base_name][u], eligible[u], binary, fractional,
                          profiles[u], head, args.k, strength, 'exposure') for u in range(len(users))}
        policy_recs[strength] = recs
        policy_results[strength] = assess('policy-fit', recs, fit_indices, save=False)
    policy = {}
    for group in range(3):
        key = str(group)
        if key not in policy_results[0.]['groups']:
            policy[group] = 0.
            continue
        baseline_ndcg = policy_results[0.]['groups'][key][metric]
        admissible = [a for a in strengths if policy_results[a]['groups'][key][metric] >= .95*baseline_ndcg]
        policy[group] = min(admissible, key=lambda a: (policy_results[a]['groups'][key]['head_exposure'], a))
    recs = {u: policy_recs[policy[int(groups[u])]][u] for u in dev_indices}
    policy_evaluation = assess('group-utility-budget-exposure', recs)
    retention = {g: values[metric]/max(records[base_name]['groups'][g][metric], 1e-12)
                 for g, values in policy_evaluation['groups'].items()}
    write_json(out/'group-policy.json', {'strengths': policy, 'fit_retention_floor': .95,
        'development_retention': retention,
        'note': 'Fitted using meta-fit labels, including labels used for hybrid fitting; development audit is essential.'})

    # Descriptive paired intervals only: same development cohort selected these models.
    interval_rng = np.random.default_rng(args.seed+1)
    intervals = {}
    for family, winner in winners.items():
        difference = np.asarray([per_user[winner][users[u]][metric]-per_user[best_expert][users[u]][metric] for u in dev_indices])
        draws = interval_rng.choice(difference, (args.bootstrap, len(difference)), replace=True).mean(axis=1)
        intervals[family] = {'candidate': winner, 'reference': best_expert,
                             'mean_difference': float(difference.mean()),
                             'descriptive_95_percent_interval': np.quantile(draws,[.025,.975]).tolist(),
                             'fraction_users_improved': float(np.mean(difference>0)),
                             'fraction_users_harmed': float(np.mean(difference<0))}
    write_json(out/'paired-comparisons.json', intervals)
    write_json(out/'results.json', records)
    write_json(out/'recommendations.json', all_recs)
    write_json(out/'selection.json', {'metric': metric, 'best_expert': best_expert, 'families': winners})
    write_json(out/'manifest.json', {'status': 'complete', 'sources': sources,
        'study_source_sha256': digest(Path(__file__)), 'metrics_source_sha256': digest(Path(__file__).with_name('metrics.py')),
        'metadata_sha256': digest(args.items), 'numpy': np.__version__, 'seed': args.seed,
        'k': args.k, 'negative_ratio': args.negative_ratio, 'meta_fit_users': len(fit_users), 'development_users': len(dev_users),
        'test_read': False, 'seconds': time.monotonic()-start,
        'limitations': ['Development results, not unbiased generalization estimates.',
                       f'Expert training used validation for selection: {shared_selection}.',
                       'Hybrid coefficients fit only meta-fit users; model choices use development users.',
                       'User cohorts share training interactions; not a user cold-start experiment.',
                       'Unobserved sampled pairs are assumed negative; some may be future positives.',
                       'Catalog exposure parity is a chosen diagnostic, not a universal fairness definition.',
                       'Rerank-before-RRF promotes top-k then preserves the full remaining ranking; both orderings retain full support.']})
    columns = [metric, f'mrr@{args.k}', 'diversity', 'calibration_jsd', 'head_exposure', 'worst_activity_ndcg']
    table = ['# Development experiment results', '',
             f'{len(fit_users)} users fit coefficients; {len(dev_users)} disjoint users compare methods. Test remains reserved.', '',
             f'Exploratory development results; expert training used validation for selection: {shared_selection}. Intervals are descriptive, not confirmatory.', '',
             '| Model | '+' | '.join(columns)+' |', '|---|'+'---:|'*len(columns)]
    for name, result in sorted(records.items(), key=lambda r:-r[1]['aggregate'][metric]):
        table.append('| '+name+' | '+' | '.join(f'{result["aggregate"][c]:.4f}' for c in columns)+' |')
    (out/'SUMMARY.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({'out': str(out), 'models': len(records), 'selected': winners, 'seconds': time.monotonic()-start}, indent=2))


if __name__ == '__main__':
    main()
