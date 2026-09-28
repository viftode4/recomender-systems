"""Independent algebra and information-access tests; synthetic matrices only.

The reference physically removes each candidate's source block and solves an
augmented least-squares problem. It does not reuse the implementation's kernel,
inverse, block correction, centering transform, or prediction helpers.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse

from exploratory.categorical_reconstruction.model import fit, load, prepare_features


def fixture():
    # Item 0 is PAD; item 6 is an entirely unobserved catalog item.
    return np.array([
        [0, 5, 2, 0, 4, 0, 0],
        [0, 1, 0, 4, 3, 5, 0],
        [0, 0, 5, 2, 0, 3, 0],
        [0, 4, 3, 1, 5, 0, 0],
        [0, 2, 0, 5, 0, 1, 0],
        [0, 0, 1, 3, 2, 4, 0],
        [0, 3, 4, 0, 1, 2, 0],
        [0, 5, 1, 4, 0, 0, 0],
        [0, 0, 2, 1, 4, 5, 0],
        [0, 1, 5, 0, 2, 3, 0],
        [0, 4, 0, 2, 3, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int64)


def dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def reference_features(train, query, smoothing, categories=True):
    """Compute the declared TRAIN probabilities without production helpers."""
    train = np.asarray(train)
    query = np.asarray(query)
    counts = np.stack([(train == rating).sum(axis=0) for rating in range(1, 6)], axis=1)
    global_probabilities = counts.sum(axis=0) / counts.sum()
    probabilities = (counts + smoothing * global_probabilities) / (
        counts.sum(axis=1, keepdims=True) + smoothing)
    binary = (query > 0).astype(np.float64)
    if not categories:
        return binary, probabilities, global_probabilities
    channels = [binary] + [((query == rating).astype(np.float64)
                           - binary * probabilities[:, rating-1])
                          for rating in range(1, 6)]
    return np.concatenate(channels, axis=1), probabilities, global_probabilities


def reduced_primal(features, targets, lambda_binary, category_ratio):
    """Constrained reference using SVD least squares, not normal equations."""
    n_features, n_items = features.shape[1], targets.shape[1]
    penalties = np.full(n_features, lambda_binary, dtype=np.float64)
    if category_ratio is not None:
        penalties[n_items:] *= category_ratio
    owners = np.arange(n_features) % n_items
    coefficients = np.zeros((n_features, n_items), dtype=np.float64)
    for item in range(n_items):
        allowed = owners != item
        reduced = features[:, allowed]
        augmented = np.vstack([reduced, np.diag(np.sqrt(penalties[allowed]))])
        rhs = np.concatenate([targets[:, item], np.zeros(allowed.sum())])
        coefficients[allowed, item] = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    return coefficients


class CategoricalReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.ratings = fixture()
        self.smoothing = 2.0

    def test_dual_coefficients_and_all_scores_match_reduced_primal(self):
        query = np.array([[0, 0, 2, 5, 0, 1, 0], [0, 1, 0, 0, 3, 5, 4]])
        for category_ratio in (None, 0.3, 2.0):
            with self.subTest(category_ratio=category_ratio):
                prepared = prepare_features(self.ratings, smoothing=self.smoothing)
                model = fit(prepared, lambda_binary=0.7, category_ratio=category_ratio)
                features, _, _ = reference_features(
                    self.ratings, self.ratings, self.smoothing, category_ratio is not None)
                query_features, _, _ = reference_features(
                    self.ratings, query, self.smoothing, category_ratio is not None)
                expected = reduced_primal(features, self.ratings > 0, 0.7, category_ratio)
                np.testing.assert_allclose(model.coefficient_matrix(), expected, atol=2e-10, rtol=2e-10)
                np.testing.assert_allclose(model.predict(), features @ expected, atol=2e-10, rtol=2e-10)
                np.testing.assert_allclose(model.predict(query), query_features @ expected, atol=2e-10, rtol=2e-10)

    def test_every_candidate_own_item_feature_is_exactly_zero(self):
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        model = fit(prepared, lambda_binary=1.3, category_ratio=0.4)
        coefficients = model.coefficient_matrix()
        n_items = self.ratings.shape[1]
        for item in range(n_items):
            np.testing.assert_array_equal(coefficients[item+n_items*np.arange(6), item], 0)
        # The prediction for j cannot change when only j's query rating changes.
        for item in range(1, n_items):
            queries = np.repeat(self.ratings[[0]], 6, axis=0)
            queries[:, item] = np.arange(6)
            scores = model.predict(queries)[:, item]
            np.testing.assert_allclose(scores, np.repeat(scores[0], 6), atol=2e-11, rtol=2e-11)

    def test_binary_model_is_exact_ease(self):
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        model = fit(prepared, lambda_binary=0.8, category_ratio=None)
        binary = (self.ratings > 0).astype(float)
        inverse = np.linalg.solve(binary.T @ binary + 0.8*np.eye(binary.shape[1]),
                                  np.eye(binary.shape[1]))
        expected = -inverse/np.diag(inverse)[None, :]
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.coefficient_matrix(), expected, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(model.predict(), binary @ expected, atol=1e-11, rtol=1e-11)

    def test_unseen_fast_path_matches_full_and_candidate_zeroing(self):
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        query = self.ratings[:4].copy()
        model = fit(prepared, lambda_binary=0.9, category_ratio=0.6)
        exact = model.predict(query)
        fast = model.predict_unadjusted(query)
        np.testing.assert_allclose(fast[query == 0], exact[query == 0], atol=1e-11, rtol=1e-11)
        self.assertGreater(np.max(np.abs(fast[query > 0]-exact[query > 0])), 1e-6)
        for item in range(query.shape[1]):
            without_target = query.copy()
            without_target[:, item] = 0
            score = model.predict_unadjusted(without_target)[:, item]
            np.testing.assert_allclose(score, exact[:, item], atol=1e-11, rtol=1e-11)

    def test_centering_probabilities_use_training_only_and_keep_absence_zero(self):
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        query = np.array([[0, 1, 0, 0, 5, 0, 4], [0, 5, 5, 5, 5, 5, 5]])
        expected, probabilities, global_probabilities = reference_features(
            self.ratings, query, self.smoothing)
        np.testing.assert_allclose(prepared.item_probabilities, probabilities, atol=1e-15)
        np.testing.assert_allclose(prepared.global_probabilities, global_probabilities, atol=1e-15)
        stored = prepared.item_probabilities.copy()
        np.testing.assert_allclose(dense(prepared.transform(query)), expected, atol=1e-15)
        np.testing.assert_allclose(dense(prepared.transform(query[:1])), expected[:1], atol=1e-15)
        prepared.transform(np.repeat(query[[1]], 50, axis=0))
        np.testing.assert_array_equal(prepared.item_probabilities, stored)
        blocks = expected.reshape(len(query), 6, query.shape[1])
        for row, item in zip(*np.where(query == 0)):
            np.testing.assert_array_equal(blocks[row, :, item], 0)
        np.testing.assert_allclose(blocks[:, 1:, :].sum(axis=1), 0, atol=3e-16)

    def test_dense_and_sparse_inputs_are_identical(self):
        query = self.ratings[:3]
        first = prepare_features(self.ratings, smoothing=self.smoothing)
        second = prepare_features(sparse.csr_matrix(self.ratings), smoothing=self.smoothing)
        np.testing.assert_array_equal(dense(first.feature_matrix()), dense(second.feature_matrix()))
        model_a = fit(first, lambda_binary=0.9, category_ratio=1.4)
        model_b = fit(second, lambda_binary=0.9, category_ratio=1.4)
        np.testing.assert_allclose(model_a.predict(query), model_b.predict(sparse.csr_matrix(query)), atol=1e-12)

    def test_permuting_rating_names_preserves_scores(self):
        # Categories have no privileged numerical ordering in this model.
        relabel = np.array([0, 5, 3, 4, 1, 2])
        query = self.ratings[:3].copy()
        original = fit(prepare_features(self.ratings, smoothing=self.smoothing),
                       lambda_binary=0.9, category_ratio=1.3)
        renamed = fit(prepare_features(relabel[self.ratings], smoothing=self.smoothing),
                      lambda_binary=0.9, category_ratio=1.3)
        np.testing.assert_allclose(original.predict(query), renamed.predict(relabel[query]), atol=2e-11, rtol=2e-11)

    def test_very_small_regularization_matches_stable_reference(self):
        # Redundant category columns and zero rows make inverse subtraction hard.
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        features, _, _ = reference_features(self.ratings, self.ratings, self.smoothing)
        ridge, ratio = 1e-10, 0.4
        expected = reduced_primal(features, self.ratings > 0, ridge, ratio)
        model = fit(prepared, lambda_binary=ridge, category_ratio=ratio)
        actual = model.coefficient_matrix()
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.isfinite(model.predict()).all())
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(model.predict(), features @ expected, atol=2e-5, rtol=2e-5)

    def test_invalid_regularization_is_rejected(self):
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(lambda_binary=value), self.assertRaises((ValueError, TypeError)):
                fit(prepared, lambda_binary=value)
            with self.subTest(category_ratio=value), self.assertRaises((ValueError, TypeError)):
                fit(prepared, lambda_binary=1.0, category_ratio=value)

    def test_empty_training_and_zero_smoothing_have_defined_cold_prior(self):
        empty = np.zeros((4, 6), dtype=np.int64)
        for smoothing in (0.0, 20.0):
            with self.subTest(smoothing=smoothing):
                prepared = prepare_features(empty, smoothing=smoothing)
                np.testing.assert_allclose(prepared.global_probabilities, np.repeat(0.2, 5))
                np.testing.assert_allclose(prepared.item_probabilities, np.full((6, 5), 0.2))
                np.testing.assert_array_equal(dense(prepared.feature_matrix()), 0)
                for ratio in (None, 0.2):
                    model = fit(prepared, lambda_binary=1.0, category_ratio=ratio)
                    np.testing.assert_array_equal(model.predict(), 0)
                    np.testing.assert_array_equal(model.predict(np.ones((2, 6), dtype=int)), 0)
        prepared = prepare_features(self.ratings, smoothing=0.0)
        global_counts = np.bincount(self.ratings.ravel(), minlength=6)[1:]
        expected_global = global_counts/global_counts.sum()
        np.testing.assert_allclose(prepared.item_probabilities[6], expected_global)
        first_counts = np.bincount(self.ratings[:, 1], minlength=6)[1:]
        np.testing.assert_allclose(prepared.item_probabilities[1], first_counts/first_counts.sum())

    def test_portable_roundtrip_preserves_regular_and_fallback_predictions(self):
        query = np.array([[0, 0, 3, 2, 5, 1, 4], [0, 5, 0, 0, 1, 0, 0]])
        prepared = prepare_features(self.ratings, smoothing=self.smoothing)
        with tempfile.TemporaryDirectory() as directory:
            for ridge in (1.3, 1e-10):
                with self.subTest(lambda_binary=ridge):
                    model = fit(prepared, lambda_binary=ridge, category_ratio=0.4)
                    path = Path(directory)/f"model-{ridge}.npz"
                    model.save(path)
                    restored = load(path)
                    np.testing.assert_array_equal(restored.predict(query), model.predict(query))
                    np.testing.assert_array_equal(restored.predict(), model.predict())
                    np.testing.assert_array_equal(restored.coefficient_matrix(), model.coefficient_matrix())
                    self.assertEqual(restored.diagnostics, model.diagnostics)

    def test_invalid_ratings_shapes_and_duplicate_observations_are_rejected(self):
        for invalid in (-1.0, 1.5, 6.0, float("inf"), float("nan")):
            ratings = self.ratings.astype(float)
            ratings[0, 1] = invalid
            with self.subTest(rating=invalid), self.assertRaises(ValueError):
                prepare_features(ratings)
        for malformed in (np.zeros(6), np.zeros((0, 6)), np.zeros((2, 0))):
            with self.subTest(shape=malformed.shape), self.assertRaises(ValueError):
                prepare_features(malformed)
        # Construct duplicates directly; an ordinary CSR constructor can sum
        # them before the model sees whether two observations existed.
        duplicate = sparse.csr_matrix((np.array([1.0, 2.0]), np.array([1, 1]),
                                       np.array([0, 2, 2])), shape=(2, 3))
        self.assertFalse(duplicate.has_canonical_format)
        with self.assertRaises(ValueError):
            prepare_features(duplicate)
        prepared = prepare_features(self.ratings)
        with self.assertRaises(ValueError):
            prepared.transform(np.zeros((2, self.ratings.shape[1]+1)))


if __name__ == "__main__":
    unittest.main()
