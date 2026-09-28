"""Meta-only calibrated affine hybrids for categorical reconstruction research.

This module performs no I/O. ``validation`` must contain exactly the meta-fit
users: passing the whole validation map is rejected before its values are read.
Expert hyperparameters may already have used all meta users. The inner split is
therefore a practical hybrid-penalty selection split, not an unbiased nested
assessment of the previously selected experts. Development remains separate.
"""

from collections.abc import Mapping
import hashlib
import json

import numpy as np

from hybrid_constraints import fit_score_calibration, fit_sum_to_one_ridge
from metrics import ranking_metrics


EXPERTS = ("binary", "slim", "real", "shuffled")
ARMS = {
    "baseline2": ("binary", "slim"),
    "real3": ("binary", "slim", "real"),
    "shuffled3": ("binary", "slim", "shuffled"),
}
PENALTIES = (0.001, 0.01, 0.1, 1.0)
NEGATIVE_RATIO = 5
K = 10
PAD_IDS = {"0", "[PAD]", "<PAD>"}


def _ordered_ids(values, name):
    ids = [str(value) for value in values]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{name} must be nonempty and unique after string conversion")
    return ids


def _mapping_keys(values, name):
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must map user IDs to item IDs")
    keys = {str(key): key for key in values.keys()}
    if len(keys) != len(values):
        raise ValueError(f"{name} contains ambiguous user IDs")
    return keys


def _indices_for_items(values, item_index, name):
    ids = [str(value) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate interactions")
    if set(ids) - set(item_index):
        raise ValueError(f"{name} contains unknown items")
    if set(ids) & PAD_IDS:
        raise ValueError(f"{name} contains padding interactions")
    return np.asarray(sorted(item_index[item] for item in ids), dtype=np.int64)


def rank_scores(scores, eligible, k=K):
    """Rank eligible columns only; exact score ties retain catalogue order."""
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=np.int64)
    if scores.ndim != 1 or eligible.ndim != 1 or len(eligible) < k or k < 1:
        raise ValueError("Need a score vector and at least k eligible columns")
    if len(set(eligible.tolist())) != len(eligible) or np.any(eligible < 0) or np.any(eligible >= len(scores)):
        raise ValueError("Invalid eligible columns")
    eligible = np.sort(eligible)
    if not np.isfinite(scores[eligible]).all():
        raise ValueError("Eligible scores must be finite")
    return eligible[np.argsort(-scores[eligible], kind="stable")[:k]]


def _normalize(scores, eligible):
    normalized = np.zeros_like(scores, dtype=float)
    for row, candidates in enumerate(eligible):
        values = scores[row, candidates]
        if not np.isfinite(values).all():
            raise ValueError("Expert scores must be finite on eligible items")
        centered = values - values.mean()
        normalized[row, candidates] = centered / max(float(values.std()), 1e-8)
    if not np.isfinite(normalized).all():
        raise ValueError("Score normalization produced nonfinite values")
    return normalized


def _sample_cells(meta_users, user_index, truth, eligible, seed):
    """Shared unique cells, excluding all observed meta validation from zeros."""
    rng = np.random.default_rng(seed + 24011)
    rows, columns, labels = [], [], []
    shortage_users, requested = 0, 0
    for user in sorted(meta_users):
        row = user_index[user]
        positive = truth[user]
        available = np.setdiff1d(eligible[row], positive, assume_unique=True)
        want = NEGATIVE_RATIO * len(positive)
        requested += want
        shortage_users += int(len(available) < want)
        negative = np.sort(rng.choice(available, min(want, len(available)), replace=False))
        cells = np.concatenate((positive, negative))
        rows.extend([row] * len(cells))
        columns.extend(cells.tolist())
        labels.extend([1.0] * len(positive) + [0.0] * len(negative))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(labels, dtype=float),
        {"requested_negatives": requested, "users_with_negative_shortage": shortage_users},
    )


def _fit_pipeline(features, labels, penalty):
    slopes, offsets = fit_score_calibration(features, labels)
    calibrated = features * slopes + offsets
    weights, intercept = fit_sum_to_one_ridge(calibrated, labels, penalty)
    effective_weights = weights * slopes
    effective_intercept = float(weights @ offsets + intercept)
    return {
        "penalty": float(penalty),
        "weights": weights.tolist(),
        "intercept": float(intercept),
        "calibration_slopes": slopes.tolist(),
        "calibration_offsets": offsets.tolist(),
        "effective_weights": effective_weights.tolist(),
        "effective_intercept": effective_intercept,
        "training_examples": int(len(labels)),
        "sum_to_one_on": "weights_after_monotone_affine_score_calibration",
        "nonnegative_weights": False,
        "bounded_probability_output": False,
    }


def _predict(features, fitted):
    return features @ np.asarray(fitted["effective_weights"]) + fitted["effective_intercept"]


def _mean_ndcg(scores, selection_users, user_index, eligible, truth):
    values = []
    for user in selection_users:
        row = user_index[user]
        recommended = rank_scores(scores[row], eligible[row])
        values.append(ranking_metrics(recommended.tolist(), truth[user].tolist(), K)["ndcg@10"])
    return float(np.mean(values))


def prepare_hybrids(scoresdict, users, items, observed, validation, fit_users, seed):
    """Return ``(arm_scores, metadata)`` using only provided meta labels.

    ``scoresdict`` has exactly binary/slim/real/shuffled keys. Each score matrix
    is aligned to ordered ``users`` and ``items``. ``observed`` maps any/all score
    users to TRAIN item IDs. ``validation`` maps exactly ``fit_users`` to every
    observed validation item for those users: all such records are relevant.
    IDs are normalized to strings without changing array order. Item IDs 0,
    [PAD] and <PAD> are excluded; padding need not appear in the catalogue.

    The inner coefficient cohort contains floor(n_meta/2) users. Both affine
    calibration and ridge coefficients fit on that cohort; the remaining users
    select a penalty by full-catalogue nDCG@10, ties preferring larger penalties.
    Both transforms are then refitted on all meta examples with the chosen
    penalty. All arms and penalties share the same sampled cells and cohorts.
    Returned arrays are finite unbounded ranking scores, with TRAIN/PAD set to
    zero placeholders. The caller MUST exclude those columns while ranking;
    zero is not a safe masking score when valid predictions can be negative.
    No development labels, data files or final evaluation inputs are accepted.
    """
    users = _ordered_ids(users, "users")
    items = _ordered_ids(items, "items")
    meta_users = _ordered_ids(sorted(fit_users, key=str), "fit_users")
    if len(meta_users) < 2 or set(meta_users) - set(users):
        raise ValueError("Need at least two known meta-fit users")
    validation_keys = _mapping_keys(validation, "validation")
    # Do this before reading ANY validation values, including potentially private
    # development entries supplied accidentally by the caller.
    if set(validation_keys) != set(meta_users):
        raise ValueError("validation must contain exactly the meta-fit users; development labels are forbidden")
    observed_keys = _mapping_keys(observed, "observed")
    if set(observed_keys) - set(users):
        raise ValueError("TRAIN contains unknown users")
    if set(scoresdict) != set(EXPERTS):
        raise ValueError(f"scoresdict must contain exactly {EXPERTS}")
    user_index = {user: row for row, user in enumerate(users)}
    item_index = {item: column for column, item in enumerate(items)}
    real_columns = np.asarray([column for column, item in enumerate(items) if item not in PAD_IDS])
    eligible = []
    for user in users:
        history = _indices_for_items(observed[observed_keys[user]], item_index, "TRAIN") if user in observed_keys else np.array([], dtype=np.int64)
        candidates = np.setdiff1d(real_columns, history, assume_unique=True)
        if len(candidates) < K:
            raise ValueError("Every score user needs at least ten non-TRAIN, non-PAD candidates")
        eligible.append(candidates)
    truth = {}
    for user in meta_users:
        positive = _indices_for_items(validation[validation_keys[user]], item_index, "meta validation")
        if not len(positive) or np.setdiff1d(positive, eligible[user_index[user]]).size:
            raise ValueError("Meta validation needs nonempty positives disjoint from TRAIN")
        truth[user] = positive

    standardized = {}
    raw = {}
    for expert in EXPERTS:
        scores = np.asarray(scoresdict[expert], dtype=float)
        if scores.shape != (len(users), len(items)):
            raise ValueError("Every expert score matrix must match ordered users and items")
        raw[expert] = scores
        standardized[expert] = _normalize(scores, eligible)
    rows, columns, labels, sampling_extra = _sample_cells(meta_users, user_index, truth, eligible, int(seed))
    shuffled_users = np.asarray(sorted(meta_users))
    np.random.default_rng(int(seed) + 24012).shuffle(shuffled_users)
    split = len(shuffled_users) // 2
    coefficient_users = sorted(shuffled_users[:split].tolist())
    selection_users = sorted(shuffled_users[split:].tolist())
    coefficient_rows = np.isin(rows, [user_index[user] for user in coefficient_users])
    if not np.any(coefficient_rows):
        raise ValueError("Coefficient cohort has no fitting examples")
    cells_digest = hashlib.sha256(json.dumps(
        [[users[row], items[column], int(label)] for row, column, label in zip(rows, columns, labels)],
        separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    metadata = {
        "version": 1,
        "seed": int(seed),
        "endpoint": "all_observed_validation_records_ndcg@10",
        "expert_keys": list(EXPERTS),
        "penalty_grid": list(PENALTIES),
        "penalty_tie_rule": "larger_penalty_on_exact_equal_inner_ndcg",
        "ranking_tie_rule": "ordered_catalogue_column_ascending",
        "normalization": "per_user_expert_zscore_ddof0_over_nonTRAIN_nonPAD; std_floor=1e-8; no labels",
        "prediction_mask": "TRAIN_and_PAD_are_finite_zero_placeholders; caller_must_exclude_them_from_ranking",
        "cohorts": {
            "meta_fit": sorted(meta_users),
            "coefficient_fit": coefficient_users,
            "penalty_selection": selection_users,
            "split_seed": int(seed) + 24012,
        },
        "sampling": {
            "seed": int(seed) + 24011,
            "negatives_per_positive": NEGATIVE_RATIO,
            "method": "unique_per_user_without_replacement; capped_by_available_unobserved_items",
            "excluded": "TRAIN_and_ALL_observed_META_validation_records_and_PAD",
            "positive_examples": int(labels.sum()),
            "negative_examples": int((labels == 0).sum()),
            "sampled_cells_sha256": cells_digest,
            **sampling_extra,
        },
        "selection_limit": "Expert hyperparameters were selected using all meta users; the inner hybrid split is not an unbiased nested expert evaluation. Development labels are not used.",
        "refit": "Both affine score calibration and constrained ridge refit on all meta sampled examples after inner penalty selection.",
        "calibration_note": "Nonnegative-slope response alignment of sampled labels, not probability calibration. Effective weights need not sum to one; calibrated weights do.",
        "duplicates": {},
        "arms": {},
    }
    for expert in ("real", "shuffled"):
        metadata["duplicates"][expert] = {
            "identical_to_binary_on_eligible_scores": all(np.array_equal(raw[expert][row, candidates], raw["binary"][row, candidates]) for row, candidates in enumerate(eligible)),
            "identical_to_binary_after_normalization": np.array_equal(standardized[expert], standardized["binary"]),
            "duplicate_retained": True,
            "note": "A duplicate expert is retained as declared; it can change the ridge penalty geometry and does not establish added category information.",
        }
    arm_scores = {}
    for arm, experts in ARMS.items():
        features = np.stack([standardized[expert] for expert in experts], axis=-1)
        sampled = features[rows, columns]
        candidates = []
        for penalty in PENALTIES:
            fitted = _fit_pipeline(sampled[coefficient_rows], labels[coefficient_rows], penalty)
            metric = _mean_ndcg(_predict(features, fitted), selection_users, user_index, eligible, truth)
            candidates.append({"penalty": penalty, "selection_ndcg@10": metric, "coefficient_fit": fitted})
        selected = max(candidates, key=lambda row: (row["selection_ndcg@10"], row["penalty"]))
        final = _fit_pipeline(sampled, labels, selected["penalty"])
        scores = _predict(features, final)
        if not np.isfinite(scores).all():
            raise ValueError("Hybrid prediction produced nonfinite scores")
        masked_scores = np.zeros(scores.shape)
        for row, candidate_columns in enumerate(eligible):
            masked_scores[row, candidate_columns] = scores[row, candidate_columns]
        arm_scores[arm] = masked_scores
        metadata["arms"][arm] = {
            "experts": list(experts),
            "candidates": candidates,
            "selected_penalty": selected["penalty"],
            "selection_ndcg@10": selected["selection_ndcg@10"],
            "final": final,
        }
    # Fail before returning if any audit value cannot be written as strict JSON.
    json.dumps(metadata, allow_nan=False)
    return arm_scores, metadata
