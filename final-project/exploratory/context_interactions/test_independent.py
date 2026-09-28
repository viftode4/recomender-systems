"""Independent exhaustive-query and failure-path audit; synthetic data only."""
from itertools import combinations, product
import unittest
from unittest import mock

import numpy as np
from scipy import linalg

from exploratory.context_interactions import model as implementation


def reference(train, query, penalty, pair_weight):
    """Physically delete each target before enumerating source features."""
    counts = train.sum(axis=1)
    pair_mass = sum(count * (count - 1) / 2 for count in counts)
    normalizer = pair_mass / counts.sum() if pair_mass and counts.sum() else 1.
    scores = np.zeros((len(query), train.shape[1]))
    dual = np.zeros_like(train, dtype=float)
    for target in range(train.shape[1]):
        sources = [column for column in range(train.shape[1]) if column != target]
        x, q = train[:, sources], query[:, sources]
        columns_x = [x[:, column] for column in range(len(sources))]
        columns_q = [q[:, column] for column in range(len(sources))]
        if pair_weight:
            scale = np.sqrt(pair_weight / normalizer)
            for first, second in combinations(range(len(sources)), 2):
                columns_x.append(scale * x[:, first] * x[:, second])
                columns_q.append(scale * q[:, first] * q[:, second])
        design = np.column_stack(columns_x)
        queries = np.column_stack(columns_q)
        dual[:, target] = np.linalg.solve(
            np.eye(len(train)) + design @ design.T / penalty, train[:, target])
        scores[:, target] = queries @ design.T @ dual[:, target] / penalty
    return scores, dual


class IndependentKernelAudit(unittest.TestCase):
    def test_every_binary_query_against_physically_deleted_target_reference(self):
        queries = np.array(list(product((0., 1.), repeat=6)))
        for seed in (11, 19):
            train = (np.random.default_rng(seed).random((9, 6)) < .45).astype(float)
            train[:, 0] = 0  # Cold item is also tested when present in a query.
            for penalty in (.001, 3., 3000.):
                for weight in (0., .2, 50.):
                    with self.subTest(seed=seed, penalty=penalty, weight=weight):
                        fitted = implementation.fit(implementation.prepare_features(train), penalty, weight)
                        expected, dual = reference(train, queries, penalty, weight)
                        np.testing.assert_allclose(fitted.predict(queries), expected, atol=2e-7, rtol=2e-7)
                        np.testing.assert_allclose(fitted.dual_coefficients, dual, atol=2e-7, rtol=2e-7)

    def test_user_and_item_permutations_preserve_predictor(self):
        rng = np.random.default_rng(31)
        train = (rng.random((12, 7)) < .5).astype(float)
        query = np.array(list(product((0., 1.), repeat=7)))
        row_order, column_order = rng.permutation(len(train)), rng.permutation(train.shape[1])
        original = implementation.fit(implementation.prepare_features(train), .3, 2.)
        permuted = implementation.fit(
            implementation.prepare_features(train[row_order][:, column_order]), .3, 2.)
        inverse_columns = np.argsort(column_order)
        np.testing.assert_allclose(permuted.predict(query[:, column_order])[:, inverse_columns],
                                   original.predict(query), atol=2e-11, rtol=2e-11)
        restored_dual = permuted.dual_coefficients[np.argsort(row_order)][:, inverse_columns]
        np.testing.assert_allclose(restored_dual, original.dual_coefficients, atol=2e-11, rtol=2e-11)

    def test_support_factor_failure_uses_exact_target_specific_fallback(self):
        train = (np.random.default_rng(41).random((10, 6)) < .5).astype(float)
        train[:, 0] = 0
        prepared = implementation.prepare_features(train)
        original = implementation.fit(prepared, .6, 3.)
        query = np.array(list(product((0., 1.), repeat=6)))
        # Global and direct fallback factors use cho_factor, so only the
        # optimized support-factor route fails in this controlled injection.
        with mock.patch.object(implementation.linalg, "cholesky", side_effect=linalg.LinAlgError("audit injection")):
            fallback = implementation.fit(prepared, .6, 3.)
        self.assertEqual(fallback.diagnostics["fallback_count"], int(np.any(train, axis=0).sum()))
        self.assertLess(fallback.diagnostics["maximum_fallback_relative_residual"], 1e-12)
        np.testing.assert_allclose(fallback.predict(query), original.predict(query), atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(fallback.dual_coefficients, original.dual_coefficients, atol=2e-11, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
