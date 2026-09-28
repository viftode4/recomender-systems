"""Shared, identity-free interpretation of query-relative donor evidence.

Feature values have a target-hidden contract: only candidates absent from the
supplied context are eligible. The caller owns padding and seen-item masks.
This module never reads data files, constructs holdouts, or chooses checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
import torch
from torch import nn


FEATURE_NAMES = (
    "log_context_count",
    "log_donor_support",
    "donor_support_fraction",
    "log_weighted_support",
    "weighted_support_fraction",
    "mean_supporter_weight",
    "log_kish_count",
    "log_pattern_count",
    "pattern_coverage",
)
PATTERN_CHANNELS = (7, 8)
VARIANTS = ("full_pattern", "no_pattern", "marginal_only")


def _binary_csr(value: Any, name: str) -> sparse.csr_matrix:
    """Copy and validate binary incidence; duplicate sparse entries are errors."""
    if sparse.issparse(value):
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        original = value.tocoo(copy=True)
        if (original.dtype.kind not in "biuf" or not np.isfinite(original.data).all()
                or not np.isin(original.data, (0, 1)).all()):
            raise ValueError(f"{name} must contain only finite binary values")
        result = sparse.csr_matrix(value, dtype=np.float64, copy=True)
        result.sum_duplicates()
        if result.nnz != original.nnz:
            raise ValueError(f"{name} must not contain duplicate sparse entries")
        result.eliminate_zeros()
        result.sort_indices()
        return result
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite binary values")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only finite binary values")
    return sparse.csr_matrix(array, dtype=np.float64)


@dataclass(frozen=True)
class DonorEvidence:
    """TRAIN incidence and additive counts; there are no learned ID features."""

    history: sparse.csr_matrix
    row_counts: np.ndarray
    support_counts: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.history.shape


def prepare_donors(binary_history: Any) -> DonorEvidence:
    history = _binary_csr(binary_history, "binary_history")
    return DonorEvidence(
        history=history,
        row_counts=np.asarray(history.sum(axis=1)).ravel(),
        support_counts=np.asarray(history.sum(axis=0)).ravel(),
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    """Exact zero convention, without an epsilon that distorts small weights."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def extract_features(
    prepared: DonorEvidence, contexts: Any, exclude_rows: Any,
) -> np.ndarray:
    """Return float32[B, catalog, 9] query-relative evidence features.

    Each excluded donor row is absent from *all* counts, denominators, weights
    and patterns. ``-1`` denotes a query external to the donor bank. Only
    candidates with context[candidate] == 0 may be ranked or supervised; seen
    candidates are not separately recomputed with leave-one-item-out contexts.
    """
    query = _binary_csr(contexts, "contexts")
    if query.shape[1] != prepared.shape[1]:
        raise ValueError("contexts and donor bank must have the same catalog")
    exclusions = np.asarray(exclude_rows)
    if exclusions.shape != (query.shape[0],) or exclusions.dtype.kind not in "iu":
        raise ValueError("exclude_rows must be an integer vector, one per query")
    n_donors, n_items = prepared.shape
    if ((exclusions < -1) | (exclusions >= n_donors)).any():
        raise ValueError("exclude_rows entries must be -1 or valid donor rows")
    output = np.zeros((query.shape[0], n_items, len(FEATURE_NAMES)), dtype=np.float32)
    donor_transpose = prepared.history.T.tocsr()
    for row, excluded in enumerate(exclusions):
        indices = query.indices[query.indptr[row]:query.indptr[row + 1]]
        h = len(indices)
        support = prepared.support_counts.copy()
        n_available = n_donors - int(excluded >= 0)
        if excluded >= 0:
            observed = prepared.history.indices[
                prepared.history.indptr[excluded]:prepared.history.indptr[excluded + 1]
            ]
            support[observed] -= 1
        output[row, :, 0] = np.log1p(h)
        output[row, :, 1] = np.log1p(support)
        output[row, :, 2] = _ratio(support, n_available)
        if h == 0 or n_available == 0:
            continue

        intersections = prepared.history[:, indices]
        overlap = np.asarray(intersections.sum(axis=1)).ravel()
        weights = _ratio(overlap * overlap, h * prepared.row_counts)
        if excluded >= 0:
            weights[excluded] = 0
        if not np.any(weights):
            continue
        mass = np.asarray(donor_transpose @ weights).ravel()
        square_mass = np.asarray(donor_transpose @ (weights * weights)).ravel()

        # a_v is the unit-L2 query/donor intersection. Sparse multiplication
        # forms all candidates' summed supporter patterns without donor pairs.
        factors = _ratio(weights, np.sqrt(overlap))
        weighted_patterns = intersections.multiply(factors[:, None]).tocsr()
        weighted_patterns.eliminate_zeros()
        summed_patterns = (donor_transpose @ weighted_patterns).tocsr()
        summed_patterns.eliminate_zeros()
        squared_norm = np.asarray(summed_patterns.multiply(summed_patterns).sum(axis=1)).ravel()
        pattern_count = _ratio(mass * mass, squared_norm)
        kish_count = _ratio(mass * mass, square_mass)
        coverage = summed_patterns.getnnz(axis=1) / h

        output[row, :, 3] = np.log1p(mass)
        output[row, :, 4] = _ratio(mass, weights.sum())
        output[row, :, 5] = _ratio(mass, support)
        output[row, :, 6] = np.log1p(kish_count)
        output[row, :, 7] = np.log1p(pattern_count)
        output[row, :, 8] = coverage
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite donor evidence")
    return output


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.size != len(FEATURE_NAMES):
            raise ValueError("normalizer must have one mean/scale per feature")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("normalizer mean/scale must be finite and scales positive")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, features: Any) -> np.ndarray:
        array = np.asarray(features)
        if array.ndim < 2 or array.shape[-1] != self.mean.size:
            raise ValueError("features have the wrong final dimension")
        if not np.isfinite(array).all():
            raise ValueError("features must be finite")
        return ((array - self.mean) / self.scale).astype(np.float32)


def fit_normalizer(features: Any, eligible: Any) -> FeatureNormalizer:
    """Fit population moments using only eligible TRAIN-episode positions."""
    array = np.asarray(features)
    mask = np.asarray(eligible)
    if array.ndim < 2 or array.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("features have the wrong final dimension")
    if mask.shape != array.shape[:-1] or mask.dtype.kind != "b":
        raise ValueError("eligible must be a Boolean mask over feature positions")
    selected = array[mask]
    if selected.shape[0] == 0 or not np.isfinite(selected).all():
        raise ValueError("normalizer needs nonempty finite eligible TRAIN features")
    mean = selected.mean(axis=0, dtype=np.float64)
    scale = selected.std(axis=0, dtype=np.float64)
    # A constant feature carries no contrast; retain its centered value zero.
    scale[scale == 0] = 1
    return FeatureNormalizer(mean, scale)


class SharedEvidenceScorer(nn.Module):
    """One shared 9 -> 16 -> 1 SiLU scorer, with no item/user parameters."""

    def __init__(
        self, n_features: int = len(FEATURE_NAMES), width: int = 16,
        variant: str = "full_pattern", seed: int = 0,
    ) -> None:
        super().__init__()
        if n_features != len(FEATURE_NAMES) or width < 1 or int(width) != width:
            raise ValueError("expected nine features and a positive integer width")
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.config = {
            "n_features": int(n_features), "width": int(width),
            "variant": variant, "seed": int(seed),
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(
                nn.Linear(n_features, int(width)), nn.SiLU(), nn.Linear(int(width), 1),
            )
        keep = torch.ones(n_features, dtype=torch.float32)
        if variant == "no_pattern":
            keep[list(PATTERN_CHANNELS)] = 0
        elif variant == "marginal_only":
            keep[3:] = 0
        # Deliberately not checkpointed: loading full weights into no_pattern
        # preserves the requested inference intervention instead of undoing it.
        self.register_buffer("feature_mask", keep, persistent=False)

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        if standardized_features.ndim < 2 or standardized_features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("standardized_features must end in nine channels")
        return self.network(standardized_features * self.feature_mask).squeeze(-1)


def masked_listwise_loss(
    scores: torch.Tensor, positive_mask: torch.Tensor, eligible_mask: torch.Tensor,
) -> torch.Tensor:
    """Equal-episode multinomial NLL over eligible recorded-item outcomes.

    All hidden TRAIN positives are jointly eligible; their target mass sums to
    one per episode. Missing candidates compete in the normalizer, which is not
    a claim that they are dislikes. Context and padding logits have zero gradient.
    """
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("scores must be a nonempty batch-by-catalog matrix")
    if positive_mask.shape != scores.shape or eligible_mask.shape != scores.shape:
        raise ValueError("loss masks must match scores")
    if positive_mask.dtype != torch.bool or eligible_mask.dtype != torch.bool:
        raise ValueError("loss masks must be Boolean")
    if bool((positive_mask & ~eligible_mask).any()):
        raise ValueError("every positive must be eligible")
    counts = positive_mask.sum(dim=1)
    if bool((counts == 0).any()) or bool((eligible_mask.sum(dim=1) == 0).any()):
        raise ValueError("every episode needs positives and eligible candidates")
    if not bool(torch.isfinite(scores[eligible_mask]).all()):
        raise ValueError("eligible scores must be finite")
    denominator = torch.logsumexp(scores.masked_fill(~eligible_mask, -torch.inf), dim=1)
    positive_mean = scores.masked_fill(~positive_mask, 0).sum(dim=1) / counts
    return (denominator - positive_mean).mean()
