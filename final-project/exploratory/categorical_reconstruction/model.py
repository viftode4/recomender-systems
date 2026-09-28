"""Exact ridge reconstruction with categorical source features and item-block exclusion.

All feature statistics are fitted on the supplied TRAIN matrix. Scores have no
masking, normalization, padding convention, or validation-dependent operations.
The binary-only submodel is exactly EASE's constrained ridge objective.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import linalg, sparse


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _ratings_csr(ratings, n_items=None):
    """Validate ratings without treating absent entries as observed category zero."""
    if sparse.issparse(ratings):
        coo = ratings.tocoo(copy=True)
        if coo.ndim != 2:
            raise ValueError("ratings must be a two-dimensional matrix")
        nonzero = coo.data != 0
        data, rows, columns = coo.data[nonzero], coo.row[nonzero], coo.col[nonzero]
        keys = rows.astype(np.int64) * coo.shape[1] + columns
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Duplicate observed user/item entries are not allowed")
        shape = coo.shape
    else:
        array = np.asarray(ratings)
        if array.ndim != 2:
            raise ValueError("ratings must be a two-dimensional matrix")
        rows, columns = np.nonzero(array)
        data, shape = array[rows, columns], array.shape
    if shape[1] < 1 or (n_items is not None and shape[1] != n_items):
        raise ValueError("ratings have an invalid number of item columns")
    if (not np.isfinite(data).all() or np.any(data < 1) or np.any(data > 5)
            or not np.equal(data, np.floor(data)).all()):
        raise ValueError("Observed ratings must be finite integers in 1..5; zero means absent")
    result = sparse.csr_matrix((data.astype(np.float64), (rows, columns)), shape=shape)
    result.sort_indices()
    return result


def _features(ratings, item_probabilities, include_categories):
    binary = ratings.copy()
    binary.data = np.ones_like(binary.data)
    if not include_categories:
        return binary
    residuals = []
    for category in range(5):
        residual = binary.copy()
        residual.data = ((ratings.data == category + 1).astype(np.float64)
                         - item_probabilities[ratings.indices, category])
        residual.eliminate_zeros()
        residuals.append(residual)
    return sparse.hstack([binary] + residuals, format="csr")


@dataclass(frozen=True)
class PreparedFeatures:
    """Reusable TRAIN statistics and n-by-n kernels, independent of ridge values.

    Feature columns are channel-major: X, Z_1, ..., Z_5. In categorical
    mode item i owns columns i + n_items * arange(6). Arrays are float64.
    """
    ratings: sparse.csr_matrix
    binary: sparse.csr_matrix
    categorical_features: sparse.csr_matrix
    global_probabilities: np.ndarray
    item_probabilities: np.ndarray
    K_binary: np.ndarray
    K_category: np.ndarray
    smoothing: float

    def feature_matrix(self, include_categories=True):
        return self.categorical_features if include_categories else self.binary

    def transform(self, query_ratings, include_categories=True):
        query = _ratings_csr(query_ratings, self.binary.shape[1])
        return _features(query, self.item_probabilities, include_categories)


def prepare_features(ratings, smoothing=20.0):
    """Fit observed-only categorical residual features using TRAIN exclusively.

    p_ir=(count_ir+smoothing*global_p_r)/(count_i+smoothing). With no
    observations anywhere the global prior is uniform. At smoothing zero,
    an empty item's unused categorical center also falls back to that prior.
    """
    smoothing = float(smoothing)
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative")
    matrix = _ratings_csr(ratings)
    if matrix.shape[0] < 1:
        raise ValueError("TRAIN must contain at least one user row")
    n_items = matrix.shape[1]
    counts = np.zeros((n_items, 5), dtype=np.float64)
    np.add.at(counts, (matrix.indices, matrix.data.astype(np.int64) - 1), 1.0)
    global_counts = counts.sum(axis=0)
    global_p = global_counts / global_counts.sum() if global_counts.sum() else np.full(5, .2)
    denominator = counts.sum(axis=1) + smoothing
    item_p = np.broadcast_to(global_p, counts.shape).copy()
    np.divide(counts + smoothing * global_p, denominator[:, None], out=item_p,
              where=denominator[:, None] > 0)
    all_features = _features(matrix, item_p, True)
    binary = all_features[:, :n_items].tocsr()
    categories = all_features[:, n_items:].tocsr()
    kb = (binary @ binary.T).toarray()
    kr = (categories @ categories.T).toarray()
    for array in (global_p, item_p, kb, kr):
        array.setflags(write=False)
    return PreparedFeatures(matrix, binary, all_features, global_p, item_p, kb, kr, smoothing)


def _own_values(features, coefficient_values, n_items, channels):
    """Per-row contribution of each target's own feature block, without p-by-m B."""
    result = np.zeros((features.shape[0], n_items), dtype=np.float64)
    for channel in range(channels):
        block = features[:, channel * n_items:(channel + 1) * n_items]
        result += block.multiply(coefficient_values[channel]).toarray()
    return result


@dataclass
class ReconstructionModel:
    prepared: PreparedFeatures
    lambda_binary: float
    category_ratio: object
    dual_coefficients: np.ndarray
    own_coefficients: np.ndarray
    fallback_items: np.ndarray
    fallback_coefficients: np.ndarray
    diagnostics: dict

    @property
    def include_categories(self):
        return self.category_ratio is not None

    @property
    def ridge(self):
        n_items = self.prepared.binary.shape[1]
        if not self.include_categories:
            return np.full(n_items, self.lambda_binary)
        return np.concatenate([np.full(n_items, self.lambda_binary),
                               np.full(5 * n_items, self.lambda_binary * self.category_ratio)])

    def _query_features(self, query_ratings):
        if query_ratings is None:
            return self.prepared.feature_matrix(self.include_categories)
        return self.prepared.transform(query_ratings, self.include_categories)

    def _unadjusted(self, query_features):
        reference = self.prepared.feature_matrix(self.include_categories)
        kernel = (query_features.multiply(1.0 / self.ridge) @ reference.T).toarray()
        scores = kernel @ self.dual_coefficients
        if len(self.fallback_items):
            # Extremely ill-conditioned targets use directly solved coefficients;
            # their own blocks are already exactly zero.
            scores[:, self.fallback_items] = query_features @ self.fallback_coefficients
        return scores

    def predict_unadjusted(self, query_ratings=None):
        """Kernel scores, valid for target-absent queries only.

        The normal kernel path still contains excluded own-feature coefficients.
        Use predict for arbitrary queries. Exceptional SVD columns are already
        constrained, and therefore need no subsequent own-block correction.
        """
        return self._unadjusted(self._query_features(query_ratings))

    def predict(self, query_ratings=None):
        """Full arbitrary-history reconstruction; caller owns all candidate masks."""
        query = self._query_features(query_ratings)
        scores = self._unadjusted(query)
        scores -= _own_values(query, self.own_coefficients, self.prepared.binary.shape[1],
                              6 if self.include_categories else 1)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite reconstruction scores")
        return scores

    def coefficient_matrix(self):
        """Materialize p-by-m coefficients for small audits, not required for fitting."""
        features = self.prepared.feature_matrix(self.include_categories)
        coefficients = np.asarray(features.T @ self.dual_coefficients) / self.ridge[:, None]
        if len(self.fallback_items):
            coefficients[:, self.fallback_items] = self.fallback_coefficients
        n_items = self.prepared.binary.shape[1]
        for channel in range(6 if self.include_categories else 1):
            coefficients[channel * n_items + np.arange(n_items), np.arange(n_items)] = 0.0
        return coefficients

    def save(self, path):
        """Portable numeric NPZ, no pickle and no validation/test inputs."""
        path = Path(path)
        matrix = self.prepared.ratings
        metadata = dict(format_version=1, lambda_binary=self.lambda_binary,
                        category_ratio=self.category_ratio, smoothing=self.prepared.smoothing,
                        diagnostics=self.diagnostics)
        with path.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                                ratings_data=matrix.data, ratings_indices=matrix.indices,
                                ratings_indptr=matrix.indptr, ratings_shape=np.array(matrix.shape),
                                global_probabilities=self.prepared.global_probabilities,
                                item_probabilities=self.prepared.item_probabilities,
                                dual_coefficients=self.dual_coefficients,
                                own_coefficients=self.own_coefficients,
                                fallback_items=self.fallback_items,
                                fallback_coefficients=self.fallback_coefficients)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported reconstruction archive version")
            ratings = sparse.csr_matrix((archive["ratings_data"], archive["ratings_indices"],
                                         archive["ratings_indptr"]),
                                        shape=tuple(archive["ratings_shape"]))
            prepared = prepare_features(ratings, metadata["smoothing"])
            for name in ("global_probabilities", "item_probabilities"):
                if not np.array_equal(getattr(prepared, name), archive[name]):
                    raise ValueError("Saved TRAIN statistics do not match saved TRAIN ratings")
            result = cls(prepared, metadata["lambda_binary"], metadata["category_ratio"],
                         archive["dual_coefficients"].copy(), archive["own_coefficients"].copy(),
                         archive["fallback_items"].copy(), archive["fallback_coefficients"].copy(),
                         metadata["diagnostics"])
        return result


def _svd_target(features, ridge, target, own_indices):
    """Numerically robust exceptional solve; no feature-by-feature Gram/inverse."""
    keep = np.ones(features.shape[1], dtype=bool)
    keep[own_indices] = False
    selected = features[:, keep]
    inverse_scale = 1.0 / np.sqrt(ridge[keep])
    z = selected.multiply(inverse_scale).toarray()
    u, singular, vt = linalg.svd(z, full_matrices=False, check_finite=False)
    projection = u.T @ target
    scaled_beta = vt.T @ ((singular / (1.0 + singular ** 2)) * projection)
    beta = np.zeros(features.shape[1])
    beta[keep] = inverse_scale * scaled_beta
    residual = target - selected @ beta[keep]
    gradient = selected.T @ (-residual) + ridge[keep] * beta[keep]
    normalizer = max(1.0, np.linalg.norm(selected.T @ target))
    relative_stationarity = float(np.linalg.norm(gradient) / normalizer)
    if not np.isfinite(relative_stationarity) or relative_stationarity > 2e-7:
        raise FloatingPointError("SVD fallback failed the ridge stationarity check")
    return residual, beta, relative_stationarity


def fit(prepared, lambda_binary, category_ratio=None):
    """Solve exactly, excluding every target's entire own-item source block.

    category_ratio=None is binary-only. Otherwise lambda_category equals
    lambda_binary*category_ratio; both penalties must be finite and positive.
    The ordinary path factors an n-users square matrix plus m blocks of size6.
    Tiny-penalty/failed-residual targets use a reduced-design SVD fallback, which
    can be slower but never constructs/inverts a p-features square matrix.
    """
    if not isinstance(prepared, PreparedFeatures):
        raise TypeError("prepared must be returned by prepare_features")
    lambda_binary = _positive(lambda_binary, "lambda_binary")
    if category_ratio is not None:
        category_ratio = _positive(category_ratio, "category_ratio")
        lambda_category = _positive(lambda_binary * category_ratio, "lambda_category")
    else:
        lambda_category = None
    categorical = category_ratio is not None
    features = prepared.feature_matrix(categorical)
    n_users, n_items = prepared.binary.shape
    channels = 6 if categorical else 1
    penalties = np.array([lambda_binary] + ([lambda_category] * 5 if categorical else []))
    ridge = np.repeat(penalties, n_items)
    system = np.eye(n_users) + prepared.K_binary / lambda_binary
    if categorical:
        system += prepared.K_category / lambda_category
    targets = prepared.binary.toarray()
    coefficients = np.zeros((n_users, n_items))
    own = np.zeros((channels, n_items))
    fallback = set()
    min_schur_relative = 1.0
    # With huge weighted kernels the dual representation loses useful digits
    # even if a Cholesky factor technically exists. A direct weighted SVD has
    # stable predictions/coefficient reconstruction in this exceptional regime.
    force_svd = bool(np.max(np.diag(system)) > 1e8)
    if force_svd:
        fallback.update(np.flatnonzero(np.any(targets != 0, axis=0)).tolist())
    else:
        try:
            factor = linalg.cho_factor(system, lower=True, check_finite=False)
            inverse = linalg.cho_solve(factor, np.eye(n_users), check_finite=False)
        except linalg.LinAlgError:
            fallback.update(np.flatnonzero(np.any(targets != 0, axis=0)).tolist())
            inverse = None
        if inverse is not None:
            h_targets = np.asarray(prepared.binary.T @ inverse.T).T
            for start in range(0, n_items, 128):
                items = np.arange(start, min(start + 128, n_items))
                indices = (items[:, None] + n_items * np.arange(channels)).reshape(-1)
                block_sparse = features[:, indices]
                block = block_sparse.toarray().reshape(n_users, len(items), channels)
                h_block = np.asarray(block_sparse.T @ inverse.T).T.reshape(block.shape)
                schur = np.diag(penalties)[None, :, :] - np.einsum("nbk,nbl->bkl", block, h_block)
                schur = (schur + schur.transpose(0, 2, 1)) * .5
                rhs = np.einsum("nbk,nb->bk", block, h_targets[:, items])
                for position, item in enumerate(items):
                    if not np.any(targets[:, item]):
                        continue
                    current = schur[position]
                    relative_eigenvalue = float(np.linalg.eigvalsh(
                        current / np.sqrt(penalties[:, None] * penalties[None, :]))[0])
                    min_schur_relative = min(min_schur_relative, relative_eigenvalue)
                    try:
                        if relative_eigenvalue < 1e-9:
                            raise linalg.LinAlgError("Ill-conditioned item Schur block")
                        sf = linalg.cho_factor(current, lower=True, check_finite=False)
                        adjustment = linalg.cho_solve(sf, rhs[position], check_finite=False)
                        coefficients[:, item] = h_targets[:, item] + h_block[:, position] @ adjustment
                        own[:, item] = (block[:, position].T @ coefficients[:, item]) / penalties
                    except linalg.LinAlgError:
                        fallback.add(int(item))
    # This checks the actual reduced normal system for every regular target,
    # rather than trusting a successful factorization or an identity derivation.
    residual = system @ coefficients - _own_values(features, own, n_items, channels) - targets
    relative = np.linalg.norm(residual, axis=0) / np.maximum(1.0, np.linalg.norm(targets, axis=0))
    fallback.update(np.flatnonzero(~np.isfinite(relative) | (relative > 2e-8)).tolist())
    fallback_items = np.array(sorted(fallback), dtype=np.int64)
    fallback_coefficients = np.zeros((features.shape[1], len(fallback_items)))
    max_stationarity = 0.0
    for offset, item in enumerate(fallback_items):
        own_indices = item + n_items * np.arange(channels)
        c, beta, stationarity = _svd_target(features, ridge, targets[:, item], own_indices)
        coefficients[:, item] = c
        own[:, item] = 0.0  # These columns predict via their directly constrained B.
        fallback_coefficients[:, offset] = beta
        max_stationarity = max(max_stationarity, stationarity)
    regular = np.ones(n_items, dtype=bool)
    regular[fallback_items] = False
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Non-finite dual coefficients")
    diagnostics = dict(solver="user-space Cholesky with item-block Schur correction",
                       n_users=n_users, n_items=n_items, n_features=features.shape[1],
                       feature_nnz=int(features.nnz), lambda_binary=lambda_binary,
                       lambda_category=lambda_category, category_ratio=category_ratio,
                       smoothing=prepared.smoothing, fallback_count=len(fallback_items),
                       fallback_items=fallback_items.tolist(), forced_svd=force_svd,
                       minimum_relative_schur_eigenvalue=min_schur_relative,
                       maximum_regular_dual_relative_residual=float(np.max(relative[regular], initial=0.0)),
                       maximum_fallback_relative_stationarity=max_stationarity)
    return ReconstructionModel(prepared, lambda_binary, category_ratio, coefficients, own,
                               fallback_items, fallback_coefficients, diagnostics)


def load(path):
    return ReconstructionModel.load(path)
