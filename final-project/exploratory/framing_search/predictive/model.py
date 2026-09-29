"""Exact singleton/grouped-pair reconstruction with full target exclusion.

The conceptual dictionary contains every unordered pair of distinct catalog
items. A grouped pair is present only when both visible records have the same
row-local group label. ``group_labels=None`` gives the ordinary all-pairs bag
model. Sparse storage omits dictionary columns that are zero throughout FIT;
their optimal ridge coefficients are exactly zero, rather than unmodelled.

No ranking masks, holdout records, selection, or timestamp inference occur here.
This is a partition-conditioned instance of constrained pair-feature ridge,
not a claim of a new inference method. Group labels need not be chronological.
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
    return gram * (gram - 1.0) / 2.0


def _grouped_pairs(binary, group_labels, dictionary=None):
    labels = np.asarray(group_labels)
    if labels.shape != binary.shape or labels.dtype.kind not in "iuf":
        raise ValueError("group_labels must be a numeric matrix matching the history")
    coo = binary.tocoo()
    observed = labels[coo.row, coo.col]
    if (not np.isfinite(observed).all() or np.any(observed != np.floor(observed))
            or np.any(np.abs(observed.astype(np.float64)) > 2**53)):
        raise ValueError("Observed group labels must be finite exact integers")
    rows, keys = [], []
    n_items = binary.shape[1]
    for user in range(binary.shape[0]):
        items = binary.indices[binary.indptr[user]:binary.indptr[user + 1]]
        user_labels = labels[user, items]
        for label in np.unique(user_labels):
            group = items[user_labels == label]
            if len(group) < 2:
                continue
            left, right = np.triu_indices(len(group), k=1)
            group_keys = group[left].astype(np.int64) * n_items + group[right]
            rows.extend([user] * len(group_keys))
            keys.extend(group_keys.tolist())
    rows, keys = np.asarray(rows, dtype=np.int64), np.asarray(keys, dtype=np.int64)
    if dictionary is None:
        dictionary, columns = np.unique(keys, return_inverse=True)
    else:
        columns = np.searchsorted(dictionary, keys)
        valid = columns < len(dictionary)
        # Avoid indexing a nonexistent entry in an empty dictionary.
        if len(dictionary):
            valid &= dictionary[np.minimum(columns, len(dictionary) - 1)] == keys
        rows, columns = rows[valid], columns[valid]
    result = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                               shape=(binary.shape[0], len(dictionary)))
    result.sort_indices()
    return result, dictionary


@dataclass(frozen=True)
class PreparedFeatures:
    binary: sparse.csr_matrix
    G: np.ndarray
    pair_gram: np.ndarray
    normalizer: float
    history_counts: np.ndarray
    mode: str
    pairs: sparse.csr_matrix | None
    pair_keys: np.ndarray | None
    pair_items: np.ndarray | None
    own_pair_columns: tuple | None
    pair_mass: float


def prepare_features(binary_history, group_labels=None, *, normalizer=None):
    """Build FIT-only kernels, using equality of visible group labels.

    The default normalizer is total present pairs / total present singletons,
    or 1 if either mass is zero. True and size-preserving shuffled partitions
    have the same value; callers may pass that shared value explicitly. A bag
    arm has its own deterministic trace normalizer. No hidden group size enters.
    """
    binary = _binary_csr(binary_history)
    if binary.shape[0] < 1:
        raise ValueError("FIT must contain at least one user")
    gram = (binary @ binary.T).toarray()
    histories = np.asarray(binary.sum(axis=1)).ravel()
    if group_labels is None:
        mode, pairs, keys, pair_items, own_columns = "bag", None, None, None, None
        pair_gram = distinct_pair_gram(gram)
        pair_mass = float(distinct_pair_gram(histories).sum())
    else:
        mode = "grouped"
        pairs, keys = _grouped_pairs(binary, group_labels)
        pair_gram = (pairs @ pairs.T).toarray()
        pair_mass = float(pairs.nnz)
        pair_items = np.column_stack((keys // binary.shape[1], keys % binary.shape[1]))
        lists = [[] for _ in range(binary.shape[1])]
        for column, (left, right) in enumerate(pair_items):
            lists[int(left)].append(column)
            lists[int(right)].append(column)
        own_columns = tuple(np.asarray(values, dtype=np.int64) for values in lists)
    linear_mass = float(histories.sum())
    if normalizer is None:
        normalizer = pair_mass / linear_mass if pair_mass > 0 and linear_mass > 0 else 1.0
    normalizer = float(normalizer)
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("normalizer must be positive and finite")
    for array in (gram, pair_gram, histories, keys, pair_items):
        if array is not None:
            array.setflags(write=False)
    return PreparedFeatures(binary, gram, pair_gram, normalizer, histories,
                            mode, pairs, keys, pair_items, own_columns, pair_mass)


def _own_pair_weights(prepared, coefficients):
    """Sparse feature-to-output matrix for only prohibited endpoint weights."""
    coo = prepared.pairs.tocoo()
    count = prepared.pairs.shape[1]
    endpoints = prepared.pair_items
    left = np.bincount(coo.col, weights=coo.data * coefficients[coo.row, endpoints[coo.col, 0]],
                       minlength=count)
    right = np.bincount(coo.col, weights=coo.data * coefficients[coo.row, endpoints[coo.col, 1]],
                        minlength=count)
    return sparse.csr_matrix((np.concatenate((left, right)),
        (np.tile(np.arange(count), 2), np.concatenate((endpoints[:, 0], endpoints[:, 1])))),
        shape=(count, prepared.binary.shape[1]))


def _removed_contribution(prepared, query, gram, query_pairs, coefficients, gamma, penalty):
    own_dual = prepared.binary.multiply(coefficients).toarray()
    if prepared.mode == "bag":
        values = (gamma * (gram @ own_dual)
                  + (1.0 - gamma) * own_dual.sum(axis=0)[None, :]) / penalty
        return query.multiply(values).toarray()
    result = query.multiply(own_dual.sum(axis=0)[None, :] / penalty).toarray()
    if gamma:
        result += (gamma / penalty) * (query_pairs @ _own_pair_weights(prepared, coefficients)).toarray()
    return result


@dataclass
class GroupedPairModel:
    prepared: PreparedFeatures
    lambda_binary: float
    pair_weight: float
    dual_coefficients: np.ndarray
    diagnostics: dict

    @property
    def gamma(self):
        return self.pair_weight / self.prepared.normalizer

    def _query(self, value, groups):
        if value is None:
            query, gram = self.prepared.binary, self.prepared.G
            if groups is None:
                return query, gram, self.prepared.pairs, self.prepared.pair_gram
        else:
            query = _binary_csr(value, self.prepared.binary.shape[1])
            gram = (query @ self.prepared.binary.T).toarray()
        if self.prepared.mode == "bag":
            if groups is not None:
                raise ValueError("A bag model does not consume query group labels")
            return query, gram, None, distinct_pair_gram(gram)
        if groups is None:
            if self.pair_weight:
                raise ValueError("New grouped queries require query_groups")
            pairs = sparse.csr_matrix((query.shape[0], self.prepared.pairs.shape[1]))
        else:
            pairs, _ = _grouped_pairs(query, groups, self.prepared.pair_keys)
        return query, gram, pairs, (pairs @ self.prepared.pairs.T).toarray()

    def predict_unadjusted(self, query_binary=None, query_groups=None):
        """Kernel scores valid only for candidates absent from query history."""
        _, gram, _, pairs_gram = self._query(query_binary, query_groups)
        scores = ((gram + self.gamma * pairs_gram) / self.lambda_binary) @ self.dual_coefficients
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite kernel scores")
        return scores

    def predict(self, query_binary=None, query_groups=None):
        """Finite, unmasked scores; removes every target-containing feature."""
        query, gram, pairs, pairs_gram = self._query(query_binary, query_groups)
        result = ((gram + self.gamma * pairs_gram) / self.lambda_binary) @ self.dual_coefficients
        result -= _removed_contribution(self.prepared, query, gram, pairs,
                                        self.dual_coefficients, self.gamma, self.lambda_binary)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite constrained scores")
        return result


def _direct_target(prepared, penalty, gamma, item):
    target = prepared.binary[:, item].toarray().ravel()
    removed_gram = prepared.G - np.outer(target, target)
    if prepared.mode == "bag":
        pairs = distinct_pair_gram(removed_gram)
    else:
        allowed = np.ones(prepared.pairs.shape[1], dtype=bool)
        allowed[prepared.own_pair_columns[item]] = False
        retained = prepared.pairs[:, allowed]
        pairs = (retained @ retained.T).toarray()
    system = np.eye(len(target)) + (removed_gram + gamma * pairs) / penalty
    factor = linalg.cho_factor(system, lower=True, check_finite=False)
    solution = linalg.cho_solve(factor, target, check_finite=False)
    residual = float(np.linalg.norm(system @ solution - target) / max(1.0, np.linalg.norm(target)))
    if not np.isfinite(residual) or residual > 2e-7:
        raise FloatingPointError("Direct target kernel failed residual check")
    return solution, residual


def fit(prepared, lambda_binary, pair_weight=0.0):
    """Exact ridge on [X, sqrt(beta/nu)*Phi], deleting the whole target block.

    Uses a shared user-space Cholesky and small target-observer downdates. Zero
    beta gives binary EASE exactly. All source values and normalizers are fixed
    by prepare_features; lambda>0 and beta>=0 must both be finite.
    """
    if not isinstance(prepared, PreparedFeatures):
        raise TypeError("prepared must come from prepare_features")
    penalty, weight = float(lambda_binary), float(pair_weight)
    if not np.isfinite(penalty) or penalty <= 0 or not np.isfinite(weight) or weight < 0:
        raise ValueError("lambda_binary must be positive and pair_weight nonnegative, both finite")
    gamma = weight / prepared.normalizer
    system = np.eye(prepared.G.shape[0]) + (prepared.G + gamma * prepared.pair_gram) / penalty
    if not np.isfinite(system).all():
        raise FloatingPointError("Kernel exceeds finite numeric range")
    factor = linalg.cho_factor(system, lower=True, check_finite=False)
    inverse = linalg.cho_solve(factor, np.eye(system.shape[0]), check_finite=False)
    inverse = (inverse + inverse.T) * .5
    n_users, n_items = prepared.binary.shape
    targets = prepared.binary.toarray()
    coefficients = np.zeros((n_users, n_items))
    fallback, supports = set(), []
    minimum_whitened_diagonal = 1.0
    for item in range(n_items):
        support = np.flatnonzero(targets[:, item])
        supports.append(len(support))
        if not len(support):
            continue
        if weight == 0:
            h_target = inverse[:, support].sum(axis=1)
            denominator = 1.0 - h_target[support].sum() / penalty
            if denominator <= 1e-10 or not np.isfinite(denominator):
                fallback.add(item)
            else:
                coefficients[:, item] = h_target / denominator
            continue
        if prepared.mode == "bag":
            own_pair_gram = prepared.G[np.ix_(support, support)] - 1.0
        else:
            own = prepared.pairs[support][:, prepared.own_pair_columns[item]]
            own_pair_gram = (own @ own.T).toarray()
        delta = (1.0 + gamma * own_pair_gram) / penalty
        try:
            lower = linalg.cholesky(inverse[np.ix_(support, support)], lower=True, check_finite=False)
            whitened = np.eye(len(support)) - lower.T @ delta @ lower
            whitened = (whitened + whitened.T) * .5
            minimum_whitened_diagonal = min(minimum_whitened_diagonal, float(np.diag(whitened).min()))
            block_factor = linalg.cho_factor(whitened, lower=True, check_finite=False)
            middle = linalg.cho_solve(block_factor, lower.T @ np.ones(len(support)), check_finite=False)
            multiplier = linalg.solve_triangular(lower.T, middle, lower=False, check_finite=False)
            coefficients[:, item] = inverse[:, support] @ multiplier
        except linalg.LinAlgError:
            fallback.add(item)
    residuals = system @ coefficients - targets
    residuals -= _removed_contribution(prepared, prepared.binary, prepared.G, prepared.pairs,
                                       coefficients, gamma, penalty)
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
        mode=prepared.mode, n_users=n_users, n_items=n_items, lambda_binary=penalty,
        pair_weight=weight, pair_normalizer=prepared.normalizer, pair_mass=prepared.pair_mass,
        implicit_singleton_features=n_items, implicit_pair_features=n_items*(n_items-1)//2,
        stored_pair_features=(prepared.pairs.shape[1] if prepared.pairs is not None else None),
        largest_target_support=max(supports, default=0),
        sum_target_support_cubes=int(sum(int(count)**3 for count in supports)),
        fallback_count=len(fallback), minimum_whitened_diagonal=minimum_whitened_diagonal,
        maximum_regular_relative_residual=float(np.max(relative[regular], initial=0.0)),
        maximum_fallback_relative_residual=maximum_fallback_residual)
    return GroupedPairModel(prepared, penalty, weight, coefficients, diagnostics)
