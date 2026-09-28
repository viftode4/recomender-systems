"""Regression weights satisfying the lecture's sum-to-one constraint.

The constraint does not imply nonnegative weights. This affine combination may
subtract an expert signal; it must not be described as a convex mixture.
"""

import numpy as np


def fit_score_calibration(features, target):
    """Fit each expert's monotone affine least-squares response calibration.

    Pass only meta-fit observation rows, never selection/calibration/test labels.
    Each independent fit minimizes squared error for ``a * score + b`` subject
    to ``a >= 0``. Constant or anticorrelated columns collapse to the target mean.
    Returns ``(slopes, offsets)`` for ``features * slopes + offsets``.

    These are response-scale transforms, not bounded or calibrated probabilities:
    sampled-negative prevalence affects the fitted intercepts and slopes. A
    subsequent sum-to-one weight constraint applies to these transformed scores,
    not to the effective coefficients after composing both affine maps.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError('Need a nonempty observations-by-experts feature matrix')
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError('Target must have one value per observation')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Features and targets must be finite')
    mean_x, mean_y = x.mean(axis=0), y.mean()
    centered = x - mean_x
    variance = np.mean(centered * centered, axis=0)
    covariance = centered.T @ (y - mean_y) / len(y)
    slopes = np.maximum(np.divide(covariance, variance, out=np.zeros_like(covariance),
                                  where=variance > 0), 0.)
    return slopes, mean_y - slopes * mean_x


def fit_sum_to_one_ridge(features, target, penalty):
    """Fit mean squared error + penalty * ||weights||² with sum(weights)=1.

    The intercept is fitted without a penalty. Every column must be an expert's
    score on the same observations, with any score normalization already applied.
    Returns ``(weights, intercept)`` like ``study.fit_ridge``. A positive penalty
    keeps the system nonsingular even when experts are perfectly correlated.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError('Need a nonempty observations-by-experts feature matrix')
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError('Target must have one value per observation')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Features and targets must be finite')
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Penalty must be finite and positive')

    mean_x, mean_y = x.mean(axis=0), y.mean()
    centered = x - mean_x
    system = centered.T @ centered / len(y) + penalty * np.eye(x.shape[1])
    right = centered.T @ (y - mean_y) / len(y)
    # The KKT multiplier shifts the unconstrained optimum along A^-1 1.
    solutions = np.linalg.solve(system, np.column_stack((right, np.ones(x.shape[1]))))
    unconstrained, direction = solutions[:, 0], solutions[:, 1]
    weights = unconstrained + direction * ((1.0 - unconstrained.sum()) / direction.sum())
    intercept = float(mean_y - mean_x @ weights)
    return weights, intercept
