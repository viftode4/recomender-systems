"""Independent explicit-feature checks, no dataset access or model selection."""
from itertools import combinations
import unittest

import numpy as np
from scipy import sparse

from exploratory.context_interactions.model import distinct_pair_gram, fit, prepare_features


def explicit_features(binary, weight, normalizer):
    n_items = binary.shape[1]
    owners = [(item,) for item in range(n_items)]
    columns = [binary[:, item] for item in range(n_items)]
    if weight:
        for first, second in combinations(range(n_items), 2):
            owners.append((first, second))
            columns.append(np.sqrt(weight / normalizer) * binary[:, first] * binary[:, second])
    return np.column_stack(columns), owners


def primal_reference(train, query, penalty, weight, normalizer):
    features, owners = explicit_features(train, weight, normalizer)
    queries, _ = explicit_features(query, weight, normalizer)
    scores = np.zeros((len(query), train.shape[1]))
    for item in range(train.shape[1]):
        keep = np.array([item not in group for group in owners])
        design = features[:, keep]
        # The augmented least-squares solve is independent of kernel algebra.
        augmented = np.vstack([design, np.sqrt(penalty)*np.eye(keep.sum())])
        target = np.r_[train[:, item], np.zeros(keep.sum())]
        beta = np.linalg.lstsq(augmented, target, rcond=None)[0]
        scores[:, item] = queries[:, keep] @ beta
    return scores


class ContextInteractionTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(917)
        self.train = (generator.random((13, 7)) < .4).astype(float)
        self.train[:, [0, 6]] = 0
        self.query = (generator.random((5, 7)) < .5).astype(float)
        self.query[:, 0] = 0

    def test_pair_kernel_equals_complete_explicit_distinct_pair_dictionary(self):
        train_features, owners = explicit_features(self.train, 1., 1.)
        query_features, _ = explicit_features(self.query, 1., 1.)
        m = self.train.shape[1]
        self.assertEqual(len(owners)-m, m*(m-1)//2)
        np.testing.assert_array_equal(distinct_pair_gram(self.train @ self.train.T),
                                      train_features[:, m:] @ train_features[:, m:].T)
        np.testing.assert_array_equal(distinct_pair_gram(self.query @ self.train.T),
                                      query_features[:, m:] @ train_features[:, m:].T)

    def test_full_solver_scores_match_independent_reduced_primal(self):
        prepared = prepare_features(self.train)
        for penalty, weight in ((.2, .1), (3., 10.), (1.7, 0.)):
            with self.subTest(penalty=penalty, weight=weight):
                model = fit(prepared, penalty, weight)
                reference = primal_reference(self.train, self.query, penalty, weight, prepared.normalizer)
                np.testing.assert_allclose(model.predict(self.query), reference, atol=2e-11, rtol=2e-11)
                np.testing.assert_allclose(model.predict(), primal_reference(
                    self.train, self.train, penalty, weight, prepared.normalizer), atol=2e-11, rtol=2e-11)
                self.assertLess(model.diagnostics['maximum_regular_relative_residual'], 1e-10)

    def test_binary_limit_equals_ease(self):
        penalty = 2.5
        model = fit(prepare_features(self.train), penalty, 0)
        inverse = np.linalg.inv(self.train.T @ self.train + penalty*np.eye(self.train.shape[1]))
        coefficients = -inverse / np.diag(inverse)
        np.fill_diagonal(coefficients, 0)
        np.testing.assert_allclose(model.predict(self.query), self.query @ coefficients, atol=1e-12, rtol=1e-12)

    def test_implicit_coefficients_match_unique_reduced_primal_solution(self):
        prepared = prepare_features(self.train)
        penalty, weight = .6, 2.1
        model = fit(prepared, penalty, weight)
        features, owners = explicit_features(self.train, weight, prepared.normalizer)
        for item in range(self.train.shape[1]):
            keep = np.array([item not in group for group in owners])
            design = features[:, keep]
            augmented = np.vstack([design, np.sqrt(penalty)*np.eye(keep.sum())])
            rhs = np.r_[self.train[:, item], np.zeros(keep.sum())]
            expected = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
            implied = design.T @ model.dual_coefficients[:, item] / penalty
            np.testing.assert_allclose(implied, expected, atol=2e-11, rtol=2e-11)

    def test_complete_target_exclusion_and_unseen_fast_path(self):
        model = fit(prepare_features(self.train), 2., 2.)
        full, fast = model.predict(self.query), model.predict_unadjusted(self.query)
        np.testing.assert_allclose(full[self.query == 0], fast[self.query == 0], atol=1e-12)
        self.assertGreater(np.max(abs(full[self.query != 0]-fast[self.query != 0])), 1e-4)
        for item in range(self.train.shape[1]):
            altered = self.query.copy()
            altered[:, item] = 1-altered[:, item]
            np.testing.assert_allclose(model.predict(altered)[:, item], full[:, item], atol=2e-12, rtol=2e-12)

    def test_normalizer_is_train_only_and_balances_kernel_trace(self):
        prepared = prepare_features(self.train)
        np.testing.assert_allclose(np.trace(prepared.pair_gram)/prepared.normalizer,
                                   np.trace(prepared.G))
        value = prepared.normalizer
        fit(prepared, 1., 1.).predict(np.ones((9, self.train.shape[1])))
        self.assertEqual(prepared.normalizer, value)

    def test_empty_cold_sparse_and_deterministic_replay(self):
        prepared = prepare_features(self.train)
        first = fit(prepared, 2., .5)
        repeated = fit(prepare_features(sparse.csr_matrix(self.train)), 2., .5)
        np.testing.assert_array_equal(first.dual_coefficients, repeated.dual_coefficients)
        np.testing.assert_array_equal(first.predict(self.query), repeated.predict(sparse.csr_matrix(self.query)))
        np.testing.assert_array_equal(first.predict(self.query)[:, [0, 6]], 0)
        np.testing.assert_array_equal(first.predict(np.zeros_like(self.query)), 0)
        for matrix in (np.zeros((5, 7)), np.eye(5)):
            item = prepare_features(matrix)
            self.assertEqual(item.normalizer, 1.)
            model = fit(item, 1., 4.)
            np.testing.assert_allclose(model.predict(), 0, atol=1e-12)

    def test_pair_representation_can_fit_xor_and_conjunction_without_self_copying(self):
        # Algebraic capacity check only, not a performance benchmark. Target
        # column is physically excluded from its features during fitting.
        bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        for outcome in (np.logical_xor(bits[:, 0], bits[:, 1]), bits[:, 0]*bits[:, 1]):
            train = np.column_stack([bits, outcome])
            model = fit(prepare_features(train), 1e-5, 1.)
            query = train.copy()
            query[:, 2] = 0
            error = np.max(abs(model.predict(query)[:, 2]-outcome))
            self.assertLess(error, 1e-4)
            linear = fit(prepare_features(train), 1e-5, 0.)
            self.assertGreater(np.max(abs(linear.predict(query)[:, 2]-outcome)), .3)

    def test_invalid_input_and_kernel_parameters(self):
        prepared = prepare_features(self.train)
        for penalty, weight in ((0, 1), (-1, 1), (float('inf'), 1), (1, -1), (1, float('nan'))):
            with self.assertRaises(ValueError):
                fit(prepared, penalty, weight)
        for matrix in (np.array([1, 0]), np.array([[0., .5]]), np.array([[1., np.nan]])):
            with self.assertRaises(ValueError):
                prepare_features(matrix)
        with self.assertRaises(ValueError):
            fit(prepared, 1.).predict(np.zeros((2, 8)))


if __name__ == '__main__':
    unittest.main()
