"""Context/probe explicit-negative information audit; no validation or test I/O."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata


def split_context_probe(training, seed, fraction=.8):
    """Split pair identities without consulting their rating values."""
    if not 0 < fraction < 1:
        raise ValueError('Context fraction must be between zero and one')
    by_user = defaultdict(list)
    seen = set()
    for user, item, value in training:
        if (user, item) in seen:
            raise ValueError('Duplicate training pair')
        seen.add((user, item))
        by_user[user].append((user, item, value))
    context, probe = [], []
    rng = np.random.default_rng(seed)
    for user in sorted(by_user):
        rows = sorted(by_user[user], key=lambda row: row[1])
        if len(rows) < 2:
            raise ValueError('Every training user needs at least two observations')
        order = rng.permutation(len(rows))
        cut = min(len(rows)-1, max(1, int(np.floor(fraction*len(rows)))))
        context.extend(rows[j] for j in order[:cut])
        probe.extend(rows[j] for j in order[cut:])
    return context, probe


@dataclass
class RidgeDecoder:
    coefficients: np.ndarray
    penalty: float

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.coefficients.shape[0] or not np.isfinite(x).all():
            raise ValueError('Invalid query features')
        scores = x @ self.coefficients
        scores[:, 0] = 0
        return scores


def fit_decoder(features, targets, penalty, channels):
    """Exact constrained ridge via a dual solve and per-target Schur correction.

    This independently returns coefficients so context and full-history queries
    use the identical fit. Target-item features are constrained in both channels.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or len(x) == 0
            or channels not in (1, 2) or x.shape[1] != channels*y.shape[1]
            or penalty <= 0 or not np.isfinite(penalty)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Invalid constrained decoder settings')
    m = y.shape[1]
    excluded = np.stack([np.arange(m)+a*m for a in range(channels)], axis=1)
    factor = cho_factor(x @ x.T + penalty*np.eye(len(x)))
    inverse_x, inverse_y = cho_solve(factor, x), cho_solve(factor, y)
    coefficients = x.T @ inverse_y
    block = np.empty((m, channels, channels))
    bias = np.empty((m, channels))
    for a in range(channels):
        xa = x[:, excluded[:, a]]
        bias[:, a] = coefficients[excluded[:, a], np.arange(m)]
        for b in range(channels):
            block[:, a, b] = ((a == b)-np.sum(xa*inverse_x[:, excluded[:, b]], axis=0))/penalty
    correction = np.linalg.solve(block, bias[:, :, None])[:, :, 0]
    for a in range(channels):
        coefficients += (x.T @ inverse_x[:, excluded[:, a]]) * (correction[None, :, a]/penalty)
        coefficients[excluded[:, a], np.arange(m)] -= correction[:, a]/penalty
    # These coefficients are structurally constrained; eliminate roundoff.
    for a in range(channels):
        coefficients[excluded[:, a], np.arange(m)] = 0
    coefficients[:, 0] = 0
    return RidgeDecoder(coefficients, float(penalty))


def surprise_dislikes(positive, negative, teacher):
    """Mean-one weights on confirmed dislikes; teacher is context-fitted."""
    p, d = np.asarray(positive, dtype=bool), np.asarray(negative, dtype=bool)
    if p.shape != d.shape or np.any(p & d) or np.any(p[:, 0]) or np.any(d[:, 0]):
        raise ValueError('Invalid positive/dislike supports')
    predictions = teacher.predict(p.astype(float))
    result = np.zeros(p.shape, dtype=float)
    for row in range(len(p)):
        candidates = np.flatnonzero(~p[row])
        candidates = candidates[candidates != 0]
        if not d[row].any():
            continue
        percentile = (rankdata(predictions[row, candidates], method='average')-1)/max(1, len(candidates)-1)
        weights = .1+.9*percentile[np.isin(candidates, np.flatnonzero(d[row]))]
        result[row, np.flatnonzero(d[row])] = weights/weights.mean()
    return result


def permute_dislike_weights(weights, seed):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or not np.isfinite(w).all() or np.any(w < 0) or np.any(w[:, 0]):
        raise ValueError('Invalid dislike weights')
    out = w.copy()
    rng = np.random.default_rng(seed)
    for row in range(len(w)):
        indices = np.flatnonzero(w[row] > 0)
        out[row, indices] = rng.permutation(w[row, indices])
    return out


def channel_features(positive, negative=None):
    p = np.asarray(positive, dtype=float)
    if negative is None:
        return p
    d = np.asarray(negative, dtype=float)
    if p.shape != d.shape or np.any((p != 0) & (d != 0)):
        raise ValueError('Positive and negative channels overlap or differ in shape')
    return np.concatenate([p, d], axis=1)
