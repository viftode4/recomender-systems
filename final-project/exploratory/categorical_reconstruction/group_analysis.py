"""Pure aggregate evaluation; groups and reference distributions use TRAIN only.

No file access, fitting, selection, or user/item identifiers in returned data.
Call only after model selection has been sealed for every study seed.
"""
from collections import Counter
from itertools import combinations
import math

import numpy as np

from metrics import evaluate
from societal import discounted_exposure_metrics


ACTIVITY_NAMES = ("sparse", "medium", "dense")


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _jsd(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    middle = (first + second) / 2
    return float(sum(np.sum(p * np.log2(np.maximum(p, 1e-15) /
                                       np.maximum(middle, 1e-15))) for p in (first, second)) / 2)


def analyze_groups(recommendations, truth, history, catalog, counts, item_genres,
                   *, k=10, group_users=None):
    """Evaluate one model on an already selected cohort, returning only aggregates.

    All IDs are original string tokens. ``truth`` maps evaluation users to all
    observed held-out item tokens, independent of rating value. ``history``
    contains the complete TRAIN population and ``counts`` its item frequencies.
    ``catalog`` contains real items only. ``group_users`` defaults to every
    history key; pass the full source user order, never the evaluation subset.
    Tercile thresholds use NumPy's default linear quantile at 1/3 and 2/3;
    ties go to the upper group. Head items are the top ceil(20% of catalog),
    ordered by descending TRAIN frequency then lexicographic item token.

    This helper cannot attest to the caller's data provenance or selection
    barrier. The runner must seal both before passing development truth here.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError("Require integer k >= 2 for pairwise diversity")
    catalog = list(catalog)
    if (not catalog or any(not isinstance(item, str) or item == "[PAD]" for item in catalog)
            or len(set(catalog)) != len(catalog)):
        raise ValueError("Require a unique non-padding catalog of original string tokens")
    catalog_set = set(catalog)
    population = list(history) if group_users is None else list(group_users)
    if (not population or len(set(population)) != len(population)
            or set(population) != set(history)):
        raise ValueError("Grouping population must equal the complete supplied TRAIN population")
    train_history = {user: set(items) for user, items in history.items()}
    if any(not items or items - catalog_set for items in train_history.values()):
        raise ValueError("Every TRAIN user requires nonempty history in the real catalog")
    if set(recommendations) - set(population) or set(truth) - set(population):
        raise ValueError("Evaluation users must belong to the TRAIN grouping population")
    expected_counts = Counter(item for items in train_history.values() for item in items)
    if (set(counts) - catalog_set or any(not np.isfinite(value) or value < 0 for value in counts.values())
            or any(counts.get(item, 0) != expected_counts.get(item, 0) for item in catalog)):
        raise ValueError("Item counts must equal complete binary TRAIN histories")
    if any(isinstance(item_genres.get(item), str) for item in catalog):
        raise ValueError("Genres must be token collections, not unparsed strings")
    genres = {item: set(item_genres.get(item, ())) for item in catalog}
    if any(not values or any(not isinstance(genre, str) for genre in values) for values in genres.values()):
        raise ValueError("Every real item requires a nonempty set of string genres")
    truth = {user: set(items) for user, items in truth.items()}
    if any(items & train_history[user] for user, items in truth.items()):
        raise ValueError("Held-out truth overlaps TRAIN history")
    result = evaluate(recommendations, truth, train_history, catalog, counts, k)
    thresholds = np.quantile([len(train_history[user]) for user in population], [1/3, 2/3])
    activity = {user: ACTIVITY_NAMES[int(np.searchsorted(thresholds, len(train_history[user]), side="right"))]
                for user in population}
    head_count = int(math.ceil(.2 * len(catalog)))
    head = set(sorted(catalog, key=lambda item: (-counts.get(item, 0), item))[:head_count])
    item_membership = {item: "head" if item in head else "tail" for item in catalog}
    vocabulary = sorted(set().union(*genres.values()))
    fractions = {item: np.array([float(genre in genres[item]) / len(genres[item])
                                for genre in vocabulary]) for item in catalog}

    for user, recs in recommendations.items():
        diversity = [1 - len(genres[a] & genres[b]) / len(genres[a] | genres[b])
                     for a, b in combinations(recs, 2)]
        profile = np.mean([fractions[item] for item in sorted(train_history[user])], axis=0)
        recommended_profile = np.mean([fractions[item] for item in recs], axis=0)
        history_head = sum(item in head for item in train_history[user]) / len(train_history[user])
        exposure_head = sum(item in head for item in recs) / k
        result["per_user"][user].update(
            diversity=float(np.mean(diversity)), calibration_jsd=_jsd(profile, recommended_profile),
            head_exposure=exposure_head,
            popularity_jsd=_jsd([1-history_head, history_head], [1-exposure_head, exposure_head]))
    for field in ("diversity", "calibration_jsd", "head_exposure", "popularity_jsd"):
        result["aggregate"][field] = _mean(row[field] for row in result["per_user"].values())

    groups = {}
    for name in ACTIVITY_NAMES:
        members = [user for user in recommendations if activity[user] == name]
        groups[name] = {"users": len(members), "status": "evaluated" if members else "empty",
                        **{field: _mean(result["per_user"][user][field] for user in members)
                           for field in next(iter(result["per_user"].values()))}}
    ndcgs = [group[f"ndcg@{k}"] for group in groups.values() if group["users"]]
    result["aggregate"].update(activity_ndcg_gap=max(ndcgs)-min(ndcgs),
                               worst_activity_ndcg=min(ndcgs),
                               head_exposure_gap=abs(result["aggregate"]["head_exposure"]-head_count/len(catalog)))
    exposure = discounted_exposure_metrics(recommendations, catalog, item_membership)
    result["aggregate"].update(exposure_gini=exposure["gini"],
                               exposure_entropy=exposure["normalized_entropy"],
                               item_group_exposure_gap=exposure["group_mean_exposure_gap"])
    item_groups = {}
    for name in ("head", "tail"):
        members = {item for item in catalog if item_membership[item] == name}
        conditional = []
        relevant_pairs = hits = slots = 0
        for user, recs in recommendations.items():
            relevant = truth[user] & members
            count = len(set(recs) & relevant)
            relevant_pairs += len(relevant)
            hits += count
            slots += len(set(recs) & members)
            if relevant:
                conditional.append(count / len(relevant))
        group_exposure = exposure["groups"].get(name, {})
        item_groups[name] = {"items": len(members), "users_with_positives": len(conditional),
                             "heldout_positive_pairs": relevant_pairs, "hits": hits,
                             f"recall@{k}": _mean(conditional),
                             "exposure": slots / (len(recommendations)*k),
                             "discounted_exposure": group_exposure.get("fraction_total_exposure", 0.),
                             "mean_item_exposure": group_exposure.get("mean_item_exposure", None)}
        result["aggregate"][f"{name}_recall"] = _mean(conditional)
    return {"schema_version": 1, "users": result["users"], "aggregate": result["aggregate"],
            "user_groups": groups, "item_groups": item_groups, "item_exposure": exposure,
            "definitions": {"grouping_population_users": len(population),
                "activity_thresholds": thresholds.tolist(),
                "activity_groups": "TRAIN history size; linear quantiles 1/3,2/3; searchsorted side=right",
                "head_items": head_count, "catalog_items": len(catalog),
                "head_definition": "Top ceil(20% of real catalog), descending TRAIN count then lexicographic token",
                "user_averaging": "Macro averages over evaluation users in each TRAIN-defined group",
                "item_recall": "Macro conditional recall over users with held-out positives in that item group",
                "diversity": "Mean pairwise Jaccard distance among recommended genre sets",
                "genre_calibration": "Jensen-Shannon divergence in bits from mean fractional TRAIN genre profile",
                "relevance": "All observed held-out records, irrespective of their rating category",
                "exposure_scope": "Allocation diagnostics; catalog parity is not a claim about item merit"}}
