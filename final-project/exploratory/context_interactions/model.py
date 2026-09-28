"""Exact regularized reconstruction with every distinct binary source pair.

The kernel evaluates an explicit linear-plus-pair feature map without allocating
the pair dictionary. Each target excludes itself from *every* source feature.
This is a standard kernel ridge model and makes no novelty claim. All matrices
are TRAIN-derived; ranking masks and model selection belong to the caller.
"""
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse


def _binary_csr(value, n_items=None):
    if sparse.issparse(value):
        coo = value.tocoo(copy=True)
        nonzero = coo.data != 0
        rows, columns, data = coo.row[nonzero], coo.col[nonzero], coo.data[nonzero]
        shape = coo.shape
        keys = rows.astype(np.int64) * shape[1] + columns
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Duplicate observed user/item entries")
    else:
        matrix = np.asarray(value)
        if matrix.ndim != 2:
            raise ValueError("History must be a two-dimensional binary matrix")
        rows, columns = np.nonzero(matrix)
        data, shape = matrix[rows, columns], matrix.shape
    if (len(shape) != 2 or shape[1] < 1 or not np.isfinite(data).all()
            or np.any(data != 1) or (n_items is not None and shape[1] != n_items)):
        raise ValueError("History must have a matching catalog and only zeros/ones")
    result = sparse.csr_matrix((np.ones(len(data)), (rows, columns)), shape=shape)
    result.sort_indices()
    return result


def distinct_pair_gram(gram):
    """Inner products of all unordered *distinct-item* binary conjunctions."""
    return gram * (gram - 1.0) / 2.0


@dataclass(frozen=True)
class PreparedFeatures:
    binary: sparse.csr_matrix
    G: np.ndarray
    pair_gram: np.ndarray
    normalizer: float
    history_counts: np.ndarray


def prepare_features(binary_history):
    matrix = _binary_csr(binary_history)
    if matrix.shape[0] < 1:
        raise ValueError("TRAIN must contain at least one user")
    gram = (matrix @ matrix.T).toarray()
    histories = np.asarray(matrix.sum(axis=1)).ravel()
    linear_mass = float(histories.sum())
    pair_mass = float(distinct_pair_gram(histories).sum())
    normalizer = pair_mass / linear_mass if pair_mass > 0 and linear_mass > 0 else 1.0
    pairs = distinct_pair_gram(gram)
    for array in (gram, pairs, histories):
        array.setflags(write=False)
    return PreparedFeatures(matrix, gram, pairs, normalizer, histories)


def _removed_contribution(query, query_gram, own_dual, gamma, penalty):
    """Contributions from the target singleton and every pair containing it."""
    values = (gamma * (query_gram @ own_dual)
              + (1.0 - gamma) * own_dual.sum(axis=0)[None, :]) / penalty
    return query.multiply(values).toarray()


@dataclass
class ContextInteractionModel:
    prepared: PreparedFeatures
    lambda_binary: float
    pair_weight: float
    dual_coefficients: np.ndarray
    diagnostics: dict

    @property
    def gamma(self):
        return self.pair_weight / self.prepared.normalizer

    def _query(self, value):
        if value is None:
            return self.prepared.binary, self.prepared.G
        query = _binary_csr(value, self.prepared.binary.shape[1])
        return query, (query @ self.prepared.binary.T).toarray()

    def predict_unadjusted(self, query_binary=None):
        """Raw kernel prediction, valid only where the query target is absent."""
        _, gram = self._query(query_binary)
        kernel = (gram + self.gamma * distinct_pair_gram(gram)) / self.lambda_binary
        scores = kernel @ self.dual_coefficients
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite kernel scores")
        return scores

    def predict(self, query_binary=None):
        """Unmasked, complete catalog scores for arbitrary binary histories."""
        query, gram = self._query(query_binary)
        kernel = (gram + self.gamma * distinct_pair_gram(gram)) / self.lambda_binary
        own_dual = self.prepared.binary.multiply(self.dual_coefficients).toarray()
        result = kernel @ self.dual_coefficients
        result -= _removed_contribution(query, gram, own_dual, self.gamma, self.lambda_binary)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite constrained scores")
        return result


def _direct_target(prepared, penalty, gamma, item):
    """Stable fallback constructing the target-removed kernel directly."""
    target = prepared.binary[:, item].toarray().ravel()
    removed_gram = prepared.G - np.outer(target, target)
    kernel = (removed_gram + gamma * distinct_pair_gram(removed_gram)) / penalty
    system = np.eye(len(target)) + kernel
    factor = linalg.cho_factor(system, lower=True, check_finite=False)
    solution = linalg.cho_solve(factor, target, check_finite=False)
    residual = float(np.linalg.norm(system @ solution - target) / max(1.0, np.linalg.norm(target)))
    if not np.isfinite(residual) or residual > 2e-7:
        raise FloatingPointError("Direct target kernel failed residual check")
    return solution, residual


def fit(prepared, lambda_binary, pair_weight=0.0):
    """Fit one joint singleton/pair reconstruction model in user space.

    Pair weights must be nonnegative to preserve a positive-semidefinite kernel.
    Pair weight zero is exactly binary EASE with the same lambda. Positive pair
    weight is equivalent to singleton penalty lambda and independent pair
    penalty lambda*normalizer/pair_weight. No pretrained predictions enter fit.
    """
    if not isinstance(prepared, PreparedFeatures):
        raise TypeError("prepared must come from prepare_features")
    penalty, weight = float(lambda_binary), float(pair_weight)
    if not np.isfinite(penalty) or penalty <= 0 or not np.isfinite(weight) or weight < 0:
        raise ValueError("lambda_binary must be positive and pair_weight nonnegative, both finite")
    gamma = weight / prepared.normalizer
    kernel = (prepared.G + gamma * prepared.pair_gram) / penalty
    system = np.eye(prepared.G.shape[0]) + kernel
    if not np.isfinite(system).all():
        raise FloatingPointError("Kernel exceeds finite numeric range")
    factor = linalg.cho_factor(system, lower=True, check_finite=False)
    inverse = linalg.cho_solve(factor, np.eye(system.shape[0]), check_finite=False)
    inverse = (inverse + inverse.T) * .5
    n_users, n_items = prepared.binary.shape
    targets = prepared.binary.toarray()
    coefficients = np.zeros((n_users, n_items))
    fallback = set()
    supports = []
    minimum_whitened_diagonal = 1.0
    for item in range(n_items):
        support = np.flatnonzero(targets[:, item])
        supports.append(len(support))
        if not len(support):
            continue
        if weight == 0:
            # Singleton exclusion is rank one and has an exact scalar update.
            h_target = inverse[:, support].sum(axis=1)
            denominator = 1.0 - h_target[support].sum() / penalty
            if denominator <= 1e-10 or not np.isfinite(denominator):
                fallback.add(item)
            else:
                coefficients[:, item] = h_target / denominator
            continue
        support_inverse = inverse[np.ix_(support, support)]
        delta = (1.0 + gamma * (prepared.G[np.ix_(support, support)] - 1.0)) / penalty
        try:
            lower = linalg.cholesky(support_inverse, lower=True, check_finite=False)
            whitened = np.eye(len(support)) - lower.T @ delta @ lower
            whitened = (whitened + whitened.T) * .5
            minimum_whitened_diagonal = min(minimum_whitened_diagonal, float(np.diag(whitened).min()))
            block_factor = linalg.cho_factor(whitened, lower=True, check_finite=False)
            middle = linalg.cho_solve(block_factor, lower.T @ np.ones(len(support)), check_finite=False)
            multiplier = linalg.solve_triangular(lower.T, middle, lower=False, check_finite=False)
            coefficients[:, item] = inverse[:, support] @ multiplier
        except linalg.LinAlgError:
            fallback.add(item)
    own_dual = prepared.binary.multiply(coefficients).toarray()
    residuals = system @ coefficients - targets
    residuals -= _removed_contribution(prepared.binary, prepared.G, own_dual, gamma, penalty)
    relative = np.linalg.norm(residuals, axis=0) / np.maximum(1.0, np.linalg.norm(targets, axis=0))
    fallback.update(np.flatnonzero(~np.isfinite(relative) | (relative > 2e-8)).tolist())
    maximum_fallback_residual = 0.0
    for item in sorted(fallback):
        solution, residual = _direct_target(prepared, penalty, gamma, item)
        coefficients[:, item] = solution
        maximum_fallback_residual = max(maximum_fallback_residual, residual)
    regular = np.ones(n_items, dtype=bool)
    regular[list(fallback)] = False
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Nonfinite dual solution")
    diagnostics = dict(solver="shared user-space Cholesky and target-observer block downdates",
        n_users=n_users, n_items=n_items, lambda_binary=penalty, pair_weight=weight,
        pair_normalizer=prepared.normalizer, implicit_singleton_features=n_items,
        implicit_pair_features=n_items*(n_items-1)//2 if weight else 0,
        largest_target_support=max(supports, default=0),
        sum_target_support_cubes=int(sum(int(count)**3 for count in supports)),
        fallback_count=len(fallback), minimum_whitened_diagonal=minimum_whitened_diagonal,
        maximum_regular_relative_residual=float(np.max(relative[regular], initial=0.0)),
        maximum_fallback_relative_residual=maximum_fallback_residual)
    return ContextInteractionModel(prepared, penalty, weight, coefficients, diagnostics)
