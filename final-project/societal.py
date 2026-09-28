"""Training-defined groups and independently calibrated reranking policies.

This module never loads an interaction file. Callers must pass training ratings
to ``training_taste_groups`` and reserve policy-calibration users before tuning
the underlying recommender. Calibration here means held-out policy fitting;
genre calibration is an unrelated distribution-matching diagnostic.
"""
from collections import defaultdict

import numpy as np

from study import jsd, ranking, rerank


def _distribution(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f'Invalid {name} distribution')
    if not np.isclose(value.sum(), 1):
        raise ValueError(f'{name} distribution must sum to one')
    return value


def _genre_matrix(genre_fraction):
    matrix = np.asarray(genre_fraction, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError('Invalid genre matrix')
    totals = matrix.sum(axis=1)
    if not np.all(np.isclose(totals, 1) | np.isclose(totals, 0)):
        raise ValueError('Genre rows must be fractional distributions or padding')
    return matrix


def _terciles(values, names):
    """Ties stay together; a tied population need not populate three groups."""
    if not values:
        return {}, []
    limits = np.quantile(list(values.values()), [1 / 3, 2 / 3])
    labels = {u: names[int(np.searchsorted(limits, x, side='right'))] for u, x in values.items()}
    return labels, limits.tolist()


def training_taste_groups(user_ids, train_ratings, genre_fraction, *, like_threshold=4.,
                          dislike_threshold=2., min_likes=3, min_dislikes=3,
                          min_negative_genre_mass=1.):
    """Return activity, positive-genre breadth, and contradiction-proxy groups.

    ``train_ratings`` contains ``(user_id, integer_item_index, rating)`` triples.
    No validation or test ratings may be supplied. A liked item is contradictory
    when over half of its genre mass belongs to genres with more disliked than
    *other* liked training mass. Fractional multi-genre weights prevent counting
    one movie multiple times. Remove the candidate like before the comparison.
    This is a metadata-dependent proxy, not a measure of emotion or psychology.
    Sparse profiles get an explicit insufficient-evidence group.
    """
    users = list(user_ids)
    if len(set(users)) != len(users) or not users:
        raise ValueError('Unique, nonempty users required')
    if like_threshold <= dislike_threshold or min_likes < 2 or min_dislikes < 1 or min_negative_genre_mass <= 0:
        raise ValueError('Invalid taste-group thresholds')
    genres = _genre_matrix(genre_fraction)
    records = defaultdict(list)
    observed = set()
    known = set(users)
    for user, item, rating in train_ratings:
        if user not in known or not isinstance(item, (int, np.integer)) or not 0 <= item < len(genres):
            raise ValueError('Unknown training user or item')
        if not np.isfinite(rating) or not np.isclose(genres[item].sum(), 1):
            raise ValueError('Invalid training rating or item metadata')
        if (user, item) in observed:
            raise ValueError('Duplicate training rating')
        observed.add((user, item))
        records[user].append((item, rating))
    features, profiles, liked_profiles = {}, {}, {}
    for user in users:
        rows = records[user]
        liked = [i for i, r in rows if r >= like_threshold]
        disliked = [i for i, r in rows if r <= dislike_threshold]
        profile = genres[[i for i, _ in rows]].mean(axis=0) if rows else None
        liked_profile = genres[liked].mean(axis=0) if liked else None
        profiles[user] = None if profile is None else profile.tolist()
        liked_profiles[user] = None if liked_profile is None else liked_profile.tolist()
        entropy = None if liked_profile is None else float(-np.sum(liked_profile * np.log2(np.maximum(liked_profile, 1e-15))))
        contradiction = None
        if len(liked) >= min_likes and len(disliked) >= min_dislikes:
            positive_mass = genres[liked].sum(axis=0)
            negative_mass = genres[disliked].sum(axis=0)
            fractions = []
            for item in liked:
                rejected = ((negative_mass >= min_negative_genre_mass)
                            & (negative_mass > positive_mass - genres[item]))
                fractions.append(float(genres[item] @ rejected) > .5)
            contradiction = float(np.mean(fractions))
        features[user] = {'activity': len(rows), 'likes': len(liked), 'dislikes': len(disliked),
                          'positive_genre_entropy': entropy, 'contradictory_like_rate': contradiction}
    activity, activity_limits = _terciles({u: f['activity'] for u, f in features.items()},
                                         ('sparse', 'medium', 'active'))
    breadth, entropy_limits = _terciles({u: f['positive_genre_entropy'] for u, f in features.items()
                                       if f['likes'] >= min_likes}, ('narrow', 'medium', 'broad'))
    # A fixed zero/nonzero cut avoids labelling identical zero rates as rich.
    contradiction = {u: ('insufficient-evidence' if f['contradictory_like_rate'] is None else
                          'has-contradictory-likes' if f['contradictory_like_rate'] > 0 else
                          'no-contradictory-likes') for u, f in features.items()}
    breadth = {u: breadth.get(u, 'insufficient-evidence') for u in users}
    return {'groups': {'activity': activity, 'taste_breadth': breadth, 'contradiction': contradiction},
            'features': features, 'profiles': profiles, 'liked_profiles': liked_profiles,
            'thresholds': {'activity_terciles': activity_limits, 'entropy_terciles': entropy_limits,
                           'like_rating_min': like_threshold, 'dislike_rating_max': dislike_threshold,
                           'min_likes': min_likes, 'min_dislikes': min_dislikes,
                           'min_negative_genre_mass': min_negative_genre_mass},
            'definition': 'Training-only genre contradiction proxy; no held-out labels define membership.'}


def calibration_diagnostics(recommendations, genre_fraction, profiles, liked_profiles=None):
    """Uniform and rank-discounted genre JSD against all/liked training histories.

    Empty profiles are explicitly excluded with counts, never replaced by a
    convenient zero. Lower JSD means distribution matching, not higher utility.
    Recommendations use integer item indices; profiles are keyed by user ID.
    """
    genres = _genre_matrix(genre_fraction)
    per_user = {}
    for user, recommendations_for_user in recommendations.items():
        selected = np.asarray(recommendations_for_user)
        if selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu':
            raise ValueError('Recommendations must be nonempty integer item lists')
        if len(set(selected.tolist())) != len(selected) or np.any(selected < 0) or np.any(selected >= len(genres)):
            raise ValueError('Duplicate or unknown recommended item')
        if not np.allclose(genres[selected].sum(axis=1), 1):
            raise ValueError('Missing genres for recommendation')
        uniform = genres[selected].mean(axis=0)
        weights = 1 / np.log2(np.arange(len(selected)) + 2)
        discounted = np.average(genres[selected], axis=0, weights=weights)
        row = {}
        for name, mapping in [('all_history', profiles), ('liked_history', liked_profiles)]:
            if mapping is None:
                continue
            if user not in mapping:
                raise ValueError('Missing user profile key; use None for empty history')
            profile = mapping[user]
            if profile is not None:
                profile = _distribution(profile, name)
                if profile.shape != uniform.shape:
                    raise ValueError('Profile and item genre dimensions differ')
                row[f'jsd_{name}'] = float(jsd(profile, uniform))
                row[f'discounted_jsd_{name}'] = float(jsd(profile, discounted))
        if 'jsd_all_history' in row and 'jsd_liked_history' in row:
            row['liked_minus_all_jsd'] = row['jsd_liked_history'] - row['jsd_all_history']
        per_user[user] = row
    keys = sorted(set().union(*(row.keys() for row in per_user.values())))
    return {'per_user': per_user, 'aggregate': {
        key: {'mean': float(np.mean([row[key] for row in per_user.values() if key in row])),
              'users': sum(key in row for row in per_user.values())} for key in keys}}


def group_diagnostics(per_user, groups, metric='ndcg@10', reference=None):
    """Report sample sizes, worst-group utility, and paired harm/retention.

    ``groups`` is user -> training-defined group. This function never constructs
    groups from the evaluation metric. Empty groups are absent, not zero-filled.
    """
    if not per_user or set(per_user) - set(groups):
        raise ValueError('Nonempty observations and complete groups required')
    if reference is not None and set(per_user) - set(reference):
        raise ValueError('Missing reference user')
    by_group = defaultdict(list)
    for user, row in per_user.items():
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError('Nonfinite utility')
        by_group[str(groups[user])].append(user)
    results = {}
    for group, members in sorted(by_group.items()):
        values = np.asarray([per_user[u][metric] for u in members], dtype=float)
        result = {'users': len(members), 'mean': float(values.mean()),
                  'standard_error': float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None}
        if reference is not None:
            baseline = np.asarray([reference[u][metric] for u in members], dtype=float)
            if not np.isfinite(baseline).all():
                raise ValueError('Nonfinite reference utility')
            difference = values - baseline
            result.update({'reference_mean': float(baseline.mean()),
                           'mean_difference': float(difference.mean()),
                           'retention': float(values.mean() / baseline.mean()) if baseline.mean() > 0 else None,
                           'fraction_improved': float(np.mean(difference > 0)),
                           'fraction_harmed': float(np.mean(difference < 0))})
        results[group] = result
    means = [r['mean'] for r in results.values()]
    return {'metric': metric, 'groups': results, 'worst_group_utility': min(means),
            'group_utility_gap': max(means) - min(means),
            'note': 'Differences are descriptive; group membership uses training information.'}


def candidate_quota_bounds(candidate_items, head_items, k, target_head_count=None):
    """Exact feasible head-count interval, checking both head and tail support."""
    candidates, head = set(candidate_items), set(head_items)
    if k < 1 or len(candidates) != len(candidate_items) or len(candidates) < k:
        raise ValueError('Unique candidate pool of size at least k required')
    heads = len(candidates & head)
    tails = len(candidates) - heads
    low, high = max(0, k - tails), min(k, heads)
    if target_head_count is not None and (not isinstance(target_head_count, (int, np.integer)) or not 0 <= target_head_count <= k):
        raise ValueError('Target head count must be an integer between zero and k')
    return {'head_candidates': heads, 'tail_candidates': tails, 'min_head_count': low,
            'max_head_count': high, 'target_feasible': None if target_head_count is None else low <= target_head_count <= high}


def item_exposure_diagnostics(recommendations, catalog, head_items):
    """Position-discounted exposure, catalogue coverage, and all-item Gini.

    Catalogue parity is one explicit target, not a universal fairness standard.
    Gini includes zero-exposure items and therefore cannot hide excluded items.
    """
    catalog = list(catalog)
    if not catalog or len(set(catalog)) != len(catalog) or not set(head_items) <= set(catalog):
        raise ValueError('Invalid catalogue or head items')
    index = {item: i for i, item in enumerate(catalog)}
    exposure, count = np.zeros(len(catalog)), np.zeros(len(catalog))
    total_lists = 0
    for recs in recommendations.values():
        if len(recs) == 0 or len(set(recs)) != len(recs) or set(recs) - set(catalog):
            raise ValueError('Invalid recommendations')
        total_lists += 1
        for rank, item in enumerate(recs):
            exposure[index[item]] += 1 / np.log2(rank + 2)
            count[index[item]] += 1
    if not total_lists:
        raise ValueError('No recommendation lists')
    sorted_exposure = np.sort(exposure)
    n = len(catalog)
    gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_exposure) / (n * exposure.sum()))
    head_indices = [index[item] for item in set(head_items)]
    head_fraction = float(count[head_indices].sum() / count.sum())
    discounted_head = float(exposure[head_indices].sum() / exposure.sum())
    target = len(set(head_items)) / n
    return {'coverage': float(np.mean(count > 0)), 'discounted_exposure_gini': gini,
            'head_exposure': head_fraction, 'discounted_head_exposure': discounted_head,
            'catalogue_head_fraction': target, 'head_parity_gap': abs(head_fraction - target),
            'discounted_head_parity_gap': abs(discounted_head - target),
            'note': 'Catalogue-proportional exposure is an explicit diagnostic target.'}


def discounted_exposure_metrics(recommendations, catalog, item_groups):
    """Lecture-style item exposure with discounts 1/log2(position + 1).

    Positions start at one. Group exposure is the mean item exposure within the
    group, averaged over users, so unequal group sizes do not imply inequality.
    Gini and entropy include every catalogue item, including unexposed ones.
    A different log base scales raw exposure but leaves normalized diagnostics
    unchanged. These quantities describe allocation, not item/provider merit.
    """
    catalog = list(catalog)
    if not catalog or len(set(catalog)) != len(catalog) or set(catalog) - set(item_groups):
        raise ValueError('Unique catalogue and complete item groups required')
    if not recommendations:
        raise ValueError('No recommendation lists')
    catalog_set = set(catalog)
    exposure = {item: 0. for item in catalog}
    lengths = set()
    for recs in recommendations.values():
        if len(recs) == 0 or len(set(recs)) != len(recs) or set(recs) - catalog_set:
            raise ValueError('Invalid recommendations')
        lengths.add(len(recs))
        for rank, item in enumerate(recs):
            exposure[item] += 1 / np.log2(rank + 2)
    if len(lengths) != 1:
        raise ValueError('Exposure comparison requires a common cutoff')
    values = np.asarray(list(exposure.values())) / len(recommendations)
    n = len(values)
    sorted_values = np.sort(values)
    gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_values) / (n * values.sum()))
    proportions = values / values.sum()
    entropy = float(-np.sum(proportions * np.log2(np.maximum(proportions, 1e-15))))
    members = defaultdict(list)
    for item in catalog:
        members[str(item_groups[item])].append(item)
    group_exposure = {group: {'items': len(items), 'mean_item_exposure': float(
        np.mean([exposure[i] for i in items]) / len(recommendations)),
        'fraction_total_exposure': float(sum(exposure[i] for i in items) / sum(exposure.values()))}
        for group, items in sorted(members.items())}
    means = [x['mean_item_exposure'] for x in group_exposure.values()]
    return {'gini': gini, 'entropy_bits': entropy,
            'normalized_entropy': entropy / np.log2(n) if n > 1 else 0.,
            'groups': group_exposure, 'group_mean_exposure_gap': max(means) - min(means),
            'users': len(recommendations), 'catalogue_items': n}


def popularity_calibration_rerank(scores, eligible, item_groups, target, k,
                                 strength, pool_size=100):
    """Greedily trade score relevance for matching a user's popularity profile.

    ``item_groups`` has a zero-based popularity category for every item index;
    ``target`` is this user's training-history distribution over those groups.
    This is the user-popularity-deviation/JSD objective, distinct from imposing
    the same catalogue head/tail ratio on every user. No labels are read here.
    """
    target = _distribution(target, 'popularity target')
    scores = np.asarray(scores, dtype=float)
    categories = np.asarray(item_groups)
    eligible = np.asarray(eligible)
    if scores.ndim != 1 or categories.shape != scores.shape or categories.dtype.kind not in 'iu':
        raise ValueError('Scores and integer popularity categories must align')
    if eligible.ndim != 1 or eligible.dtype.kind not in 'iu' or len(set(eligible.tolist())) != len(eligible):
        raise ValueError('Invalid eligible item indices')
    if np.any(eligible < 0) or np.any(eligible >= len(scores)) or np.any(categories[eligible] < 0) or np.any(categories[eligible] >= len(target)):
        raise ValueError('Invalid eligible item or popularity category')
    if not 0 <= strength <= 1 or k < 1 or pool_size < k or not np.isfinite(scores[eligible]).all():
        raise ValueError('Invalid reranking parameters')
    pool = ranking(scores, eligible, min(pool_size, len(eligible)))
    if len(pool) < k:
        raise ValueError('Candidate pool smaller than k')
    # Zero must preserve ranking even if near-constant scores normalize poorly.
    if strength == 0:
        return pool[:k]
    relevance = scores[pool].copy()
    relevance = (relevance - relevance.min()) / max(np.ptp(relevance), 1e-8)
    one_hot = np.eye(len(target))[categories[pool]]
    cumulative = np.zeros(len(target))
    selected_positions = []
    for step in range(k):
        distribution = (cumulative[None, :] + one_hot) / (step + 1)
        objective = (1 - strength) * relevance + strength * (1 - jsd(target[None, :], distribution))
        objective[selected_positions] = -np.inf
        position = int(np.argmax(objective))
        selected_positions.append(position)
        cumulative += one_hot[position]
    return pool[selected_positions]


def user_popularity_deviation(recommendations, item_groups, profiles):
    """User-wise JSD of recommended and training popularity distributions."""
    categories = np.asarray(item_groups)
    if categories.ndim != 1 or categories.dtype.kind not in 'iu':
        raise ValueError('Integer popularity categories required')
    results = {}
    for user, recs in recommendations.items():
        target = _distribution(profiles[user], 'popularity target')
        selected = np.asarray(recs)
        if selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu' or len(set(selected.tolist())) != len(selected):
            raise ValueError('Unique, nonempty integer recommendations required')
        if np.any(selected < 0) or np.any(selected >= len(categories)) or np.any(categories[selected] < 0) or np.any(categories[selected] >= len(target)):
            raise ValueError('Invalid item or category')
        distribution = np.bincount(categories[selected], minlength=len(target)) / len(selected)
        results[user] = float(jsd(target, distribution))
    if not results:
        raise ValueError('No recommendations')
    return {'mean_jsd': float(np.mean(list(results.values()))), 'per_user': results}


def fit_group_policy(utilities, groups, calibration_users, *, fit_users, selection_users,
                     baseline, secondary=None, objective='utility', retention=.95,
                     confidence=.95, bootstrap=5000, seed=2026, min_group_size=20,
                     bound='bootstrap'):
    """Fit a user-group reranker selector on a distinct calibration cohort.

    ``utilities[option][user]`` must be bounded in [0, 1] (e.g. NDCG).
    ``secondary[option][user]`` is an optional cost to minimize, such as exposure
    parity gap. Options and the base scorer must be frozen without calibration
    labels. Explicit fit/selection IDs reject cohort leakage at this boundary.

    ``utility`` chooses the option with best lower paired-improvement bound in
    each group (thereby maximizing the minimum of these separable conservative
    group utilities). ``exposure`` minimizes the secondary cost subject to the
    lower bound on E[utility - retention * baseline] being nonnegative.

    Bonferroni adjusts the one-sided level for both margins and all nonbaseline
    option/group comparisons. Percentile bootstrap bounds are approximate, not
    distribution-free guarantees; small samples can be overconfident. Optional
    Hoeffding bounds assume independent bounded user observations and frozen
    candidates. Neither mode establishes real-world fairness or causal effects.
    The untouched audit cohort must be reported even when retention fails there.
    """
    calibration = list(calibration_users)
    if not calibration or len(set(calibration)) != len(calibration):
        raise ValueError('Unique, nonempty calibration users required')
    if set(calibration) & (set(fit_users) | set(selection_users)):
        raise ValueError('Policy calibration overlaps model fitting or selection users')
    if baseline not in utilities or objective not in ('utility', 'exposure') or bound not in ('bootstrap', 'hoeffding'):
        raise ValueError('Invalid baseline, objective, or bound method')
    if not 0 < retention <= 1 or not 0 < confidence < 1 or bootstrap < 100 or min_group_size < 2:
        raise ValueError('Invalid policy calibration parameters')
    if objective == 'exposure' and secondary is None:
        raise ValueError('Exposure objective requires a secondary cost')
    if set(calibration) - set(groups):
        raise ValueError('Missing calibration groups')
    options = sorted(utilities)
    for option in options:
        if set(calibration) - set(utilities[option]):
            raise ValueError('Missing calibration utility')
        values = np.asarray([utilities[option][u] for u in calibration], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError('Utilities must be finite and bounded in [0, 1]')
        if secondary is not None:
            if option not in secondary or set(calibration) - set(secondary[option]):
                raise ValueError('Missing secondary cost')
            if not np.isfinite([secondary[option][u] for u in calibration]).all():
                raise ValueError('Nonfinite secondary cost')
    cohort_groups = defaultdict(list)
    for user in calibration:
        cohort_groups[str(groups[user])].append(user)
    comparisons = max(1, 2 * (len(options) - 1) * len(cohort_groups))
    alpha = (1 - confidence) / comparisons
    rng = np.random.default_rng(seed)
    policy, diagnostics = {}, {}
    for group in sorted(set(map(str, groups.values()))):
        members = cohort_groups[group]
        if len(members) < min_group_size:
            policy[group] = baseline
            diagnostics[group] = {'users': len(members), 'selected': baseline,
                                  'reason': 'insufficient-independent-calibration-users'}
            continue
        n = len(members)
        reference = np.asarray([utilities[baseline][u] for u in members])
        # Shared bootstrap indices preserve pairing for every proposed option.
        draws = rng.integers(0, n, size=(bootstrap, n))
        candidates = {}
        for option in options:
            values = np.asarray([utilities[option][u] for u in members])
            improvement = values - reference
            margin = values - retention * reference
            improvement_draws = improvement[draws].mean(axis=1)
            margin_draws = margin[draws].mean(axis=1)
            if option == baseline:
                lower_improvement, lower_margin = 0., 0.
            elif bound == 'bootstrap':
                lower_improvement = float(np.quantile(improvement_draws, alpha))
                lower_margin = float(np.quantile(margin_draws, alpha))
            else:
                factor = np.sqrt(np.log(1 / alpha) / (2 * n))
                lower_improvement = float(improvement.mean() - 2 * factor)
                lower_margin = float(margin.mean() - (1 + retention) * factor)
            candidates[option] = {'mean_utility': float(values.mean()),
                                  'mean_difference': float(improvement.mean()),
                                  'paired_difference_95_percent_interval': np.quantile(improvement_draws, [.025, .975]).tolist(),
                                  'lower_difference_bound': lower_improvement,
                                  'mean_retention_margin': float(margin.mean()),
                                  'lower_retention_margin_bound': lower_margin,
                                  'admissible': option == baseline or lower_margin >= 0,
                                  'secondary_mean': None if secondary is None else float(np.mean([secondary[option][u] for u in members]))}
        admitted = [o for o in options if candidates[o]['admissible']]
        if objective == 'utility':
            # Equal support chooses baseline, then lexical order reproducibly.
            chosen = min(admitted, key=lambda o: (-candidates[o]['lower_difference_bound'], o != baseline, o))
        else:
            chosen = min(admitted, key=lambda o: (candidates[o]['secondary_mean'], o != baseline, o))
        policy[group] = chosen
        diagnostics[group] = {'users': n, 'selected': chosen, 'candidates': candidates}
    return {'policy': policy, 'baseline': baseline, 'groups': diagnostics,
            'objective': objective, 'retention_target': retention, 'confidence': confidence,
            'bound_method': bound, 'per_comparison_alpha': alpha,
            'bonferroni_comparisons': comparisons, 'bootstrap_resamples': bootstrap,
            'bootstrap_expected_lower_tail_draws': bootstrap * alpha,
            'calibration_users': sorted(calibration),
            'note': 'Frozen options, independent policy calibration, and separate audit required. Bootstrap bounds are approximate; no holdout or real-world guarantee.'}


def apply_group_policy(policy, groups, recommendations_by_option, users=None):
    """Apply a frozen policy without reading relevance labels."""
    selected_users = list(groups) if users is None else list(users)
    if len(set(selected_users)) != len(selected_users):
        raise ValueError('Duplicate policy user')
    result = {}
    for user in selected_users:
        if user not in groups:
            raise ValueError('Unknown policy user')
        option = policy['policy'].get(str(groups[user]), policy['baseline'])
        if option not in recommendations_by_option or user not in recommendations_by_option[option]:
            raise ValueError('Missing frozen-option recommendations')
        result[user] = list(recommendations_by_option[option][user])
    return result


def rerank_options(scores, eligible, genre_binary, genre_fraction, profiles, head, k,
                   strengths=(0., .2, .5, .8), mode='exposure', pool_size=100):
    """Generate frozen candidate options using existing study.rerank mechanics.

    All arrays are row-aligned; result is option -> integer row -> item indices.
    Callers can translate row IDs to user IDs before policy fitting.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2 or len(eligible) != len(scores) or len(profiles) != len(scores):
        raise ValueError('Misaligned reranking users')
    strengths = tuple(float(s) for s in strengths)
    if len(set(strengths)) != len(strengths) or not strengths:
        raise ValueError('Distinct, nonempty reranking strengths required')
    return {f'{mode}-{strength:g}': {
        user: rerank(scores[user], eligible[user], genre_binary, genre_fraction,
                     profiles[user], head, k, strength, mode, pool_size).tolist()
        for user in range(len(scores))} for strength in strengths}
