"""Deterministic mixed, meta-level, switching and rank-fusion recommenders."""
from __future__ import annotations

import numpy as np


def _scores(scores, observed):
    scores, observed = np.asarray(scores, dtype=float), np.asarray(observed, dtype=bool)
    if scores.ndim != 2 or observed.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Require finite user/item scores and a matching observed mask")
    mask = observed.copy()
    mask[:, 0] = True
    return scores, mask


def rank_scores(scores, observed, k=10):
    scores, mask = _scores(scores, observed)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("Require positive integer k")
    if np.any((~mask).sum(axis=1) < k):
        raise ValueError("Insufficient eligible catalog items")
    masked = scores.copy()
    masked[mask] = -np.inf
    return np.argsort(-masked, axis=1, kind="stable")[:, :k]


def mix_lists(ranked_lists, quotas, seen=(), k=10, padding=0):
    """Interleave unique picks; skipped duplicates do not consume quota.

    At output position t (zero-based), choose the available expert maximizing
    (t+1)*quota - k*already_taken. Exact ties use input expert order. If every
    still-needed expert exhausts, fill from expert zero's remaining list.
    """
    quotas = np.asarray(quotas)
    if (quotas.ndim != 1 or len(quotas) != len(ranked_lists) or not len(quotas) or
            not np.issubdtype(quotas.dtype, np.integer) or (quotas < 0).any() or
            not isinstance(k, int) or isinstance(k, bool) or k < 1 or quotas.sum() != k):
        raise ValueError("Nonnegative integer quotas must sum to k")
    lists = [list(values) for values in ranked_lists]
    excluded = set(seen) | {padding}
    chosen, sources = [], []
    positions = np.zeros(len(lists), dtype=int)
    taken = np.zeros(len(lists), dtype=int)
    exhausted = np.zeros(len(lists), dtype=bool)

    def next_valid(expert):
        while positions[expert] < len(lists[expert]):
            item = lists[expert][positions[expert]]
            if item in excluded:
                positions[expert] += 1
            else:
                return item
        exhausted[expert] = True
        return None

    while len(chosen) < k:
        available = []
        for expert in range(len(lists)):
            if taken[expert] < quotas[expert] and not exhausted[expert] and next_valid(expert) is not None:
                available.append(expert)
        if not available:
            break
        t = len(chosen)
        expert = max(available, key=lambda e: (t + 1) * int(quotas[e]) - k * int(taken[e]))
        item = next_valid(expert)
        chosen.append(item)
        sources.append(expert)
        excluded.add(item)
        positions[expert] += 1
        taken[expert] += 1
    fallback = 0
    for item in lists[0]:
        if len(chosen) == k:
            break
        if item not in excluded:
            chosen.append(item)
            sources.append(-1)
            excluded.add(item)
            fallback += 1
    if len(chosen) != k:
        raise ValueError("EASE fallback cannot supply k unique eligible items")
    return np.asarray(chosen, dtype=int), {"quota_counts": taken.tolist(), "fallback_picks": fallback,
                                           "sources": sources}


def mixed_rankings(expert_scores, observed, quotas, k=10):
    values = np.asarray(expert_scores, dtype=float)
    if values.ndim != 3:
        raise ValueError("Require expert/user/item scores")
    for value in values:
        _scores(value, observed)
    ordered = np.argsort(-values, axis=2, kind="stable")
    result, allocated, fallback = [], np.zeros(len(values), dtype=int), 0
    for row in range(values.shape[1]):
        ranked, diagnostics = mix_lists(ordered[:, row], quotas, np.flatnonzero(observed[row]), k)
        result.append(ranked)
        allocated += diagnostics["quota_counts"]
        fallback += diagnostics["fallback_picks"]
    return np.asarray(result), {"users": values.shape[1], "quota_counts_total": allocated.tolist(),
                               "fallback_picks_total": fallback}


def fit_meta_level(binary, genres, penalty):
    """Two sum-loss ridge stages; the collaborative decoder consumes only P."""
    x, g = np.asarray(binary, dtype=float), np.asarray(genres, dtype=float)
    if (x.ndim != 2 or g.ndim != 2 or x.shape[1] != g.shape[0] or
            not np.isfinite(x).all() or not np.isfinite(g).all() or
            not np.isin(x, [0., 1.]).all() or (g < 0).any() or
            not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("Require binary TRAIN, finite nonnegative genre features and positive penalty")
    if not np.allclose(g.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Each real item must have fractional genre mass one")
    identity = np.eye(g.shape[1])
    profiles = np.linalg.solve(g.T @ g + penalty * identity, (x @ g).T).T
    item_factors = np.linalg.solve(profiles.T @ profiles + penalty * identity, (x.T @ profiles).T).T
    scores = profiles @ item_factors.T
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite meta-level prediction")
    return {"profiles": profiles, "item_factors": item_factors, "scores": scores}


def fit_switch(activity, values, fit_indices, group_count, expert_order):
    activity = np.asarray(activity, dtype=float)
    values = np.asarray(values, dtype=float)
    fit_indices = np.asarray(fit_indices)
    if (activity.ndim != 1 or not np.isfinite(activity).all() or (activity < 0).any() or
            not isinstance(group_count, int) or isinstance(group_count, bool) or group_count < 1 or
            fit_indices.ndim != 1 or not np.issubdtype(fit_indices.dtype, np.integer) or
            len(set(fit_indices.tolist())) != len(fit_indices) or not len(fit_indices) or
            (fit_indices < 0).any() or (fit_indices >= len(activity)).any() or
            len(set(expert_order)) != len(expert_order) or
            values.shape != (len(expert_order), len(fit_indices)) or not np.isfinite(values).all()):
        raise ValueError("Require finite TRAIN activity and meta-fit-only expert values")
    thresholds = np.quantile(activity, np.arange(1, group_count) / group_count)
    groups = np.searchsorted(thresholds, activity, side="right")
    global_means = values.mean(axis=1)
    global_index = int(np.argmax(global_means))
    winners, sizes, means = {}, {}, {}
    for group in range(group_count):
        members = groups[fit_indices] == group
        group_means = values[:, members].mean(axis=1) if members.any() else global_means
        winners[str(group)] = expert_order[int(np.argmax(group_means))]
        sizes[str(group)] = int(members.sum())
        means[str(group)] = dict(zip(expert_order, group_means.tolist()))
    return {"group_count": group_count, "thresholds": thresholds.tolist(), "winners": winners,
            "global_winner": expert_order[global_index], "meta_fit_group_users": sizes,
            "meta_fit_group_means": means, "expert_order": list(expert_order),
            "group_rule": "Full TRAIN-population linear quantiles; ties assigned right"}


def switch_scores(experts, activity, policy):
    values = list(experts.values())
    if not values or any(np.asarray(value).shape != np.asarray(values[0]).shape for value in values):
        raise ValueError("Expert score shapes differ")
    groups = np.searchsorted(policy["thresholds"], np.asarray(activity), side="right")
    return np.stack([experts[policy["winners"][str(int(group))]][row] for row, group in enumerate(groups)])


def rrf_scores(expert_scores, observed, constant):
    values = np.asarray(expert_scores, dtype=float)
    if values.ndim != 3 or not len(values) or not np.isfinite(constant) or constant <= 0:
        raise ValueError("Require expert/user/item scores and positive RRF offset")
    result = np.zeros(values.shape[1:], dtype=float)
    for expert in values:
        scores, mask = _scores(expert, observed)
        for row in range(len(scores)):
            eligible = np.flatnonzero(~mask[row])
            ordered = eligible[np.argsort(-scores[row, eligible], kind="stable")]
            result[row, ordered] += 1 / (constant + np.arange(1, len(ordered) + 1))
    return result
