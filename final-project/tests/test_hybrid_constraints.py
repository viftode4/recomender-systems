import unittest

import numpy as np

from hybrid_constraints import fit_score_calibration, fit_sum_to_one_ridge


class ScoreCalibrationTests(unittest.TestCase):
    def test_monotone_fit_recovers_positive_and_clamps_negative_or_constant(self):
        x = np.array([[0., 2., 7.], [1., 1., 7.], [2., 0., 7.]])
        slopes, offsets = fit_score_calibration(x, [0., 1., 2.])
        np.testing.assert_allclose(slopes, [1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(offsets, [0., 1., 1.], atol=1e-12)

    def test_positive_affine_rescaling_does_not_change_fitted_responses(self):
        x = np.array([[0., 2.], [1., 1.], [2., 3.], [3., 0.]])
        y = np.array([0., 0., 1., 1.])
        a, b = fit_score_calibration(x, y)
        rescaled = x * np.array([12., .3]) + np.array([7., -4.])
        c, d = fit_score_calibration(rescaled, y)
        np.testing.assert_allclose(x * a + b, rescaled * c + d, atol=1e-12)

    def test_affine_composition_preserves_predictions_and_constraint_scope(self):
        x = np.array([[0., 2.], [1., 1.], [2., 3.], [3., 0.]])
        y = np.array([0., 0., 1., 1.])
        slopes, offsets = fit_score_calibration(x, y)
        transformed = x * slopes + offsets
        weights, intercept = fit_sum_to_one_ridge(transformed, y, .01)
        effective_weights = slopes * weights
        effective_intercept = float(offsets @ weights + intercept)
        np.testing.assert_allclose(transformed @ weights + intercept,
                                   x @ effective_weights + effective_intercept, atol=1e-12)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertFalse(np.isclose(effective_weights.sum(), 1.))

    def test_invalid_calibration_inputs_are_rejected(self):
        for x, y in [([], []), ([[1.]], []), ([[1.]], [[1.]]),
                     ([[float('nan')]], [1.]), ([[1.]], [float('inf')])]:
            with self.subTest(features=x, target=y), self.assertRaises(ValueError):
                fit_score_calibration(x, y)


class SumToOneRidgeTests(unittest.TestCase):
    def test_analytic_two_expert_solution_allows_negative_weights(self):
        # Orthogonal centered columns have variance 1/4. With lambda=1/20,
        # the exact constrained solution is [-1/3, 4/3], intercept 3.
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        y = x @ np.array([-.5, 1.5]) + 3.
        weights, bias = fit_sum_to_one_ridge(x, y, .05)
        np.testing.assert_allclose(weights, [-1/3, 4/3], atol=1e-12)
        self.assertAlmostEqual(weights.sum(), 1., places=12)
        self.assertAlmostEqual(bias, 3., places=12)

    def test_single_expert_has_unit_weight_and_unpenalized_intercept(self):
        weights, bias = fit_sum_to_one_ridge([[1.], [2.], [3.]], [5., 7., 9.], 100.)
        np.testing.assert_allclose(weights, [1.], atol=1e-12)
        self.assertAlmostEqual(bias, 5.)

    def test_identical_experts_split_weights_equally(self):
        x = np.tile(np.arange(10., dtype=float)[:, None], (1, 4))
        weights, bias = fit_sum_to_one_ridge(x, np.arange(10.) + 2., .1)
        np.testing.assert_allclose(weights, [.25] * 4, atol=1e-12)
        self.assertAlmostEqual(bias, 2.)

    def test_target_shift_changes_only_intercept(self):
        x = np.array([[0., 1.], [2., 0.], [1., 3.], [4., 2.]])
        y = np.array([1., 3., 2., 4.])
        weights, bias = fit_sum_to_one_ridge(x, y, .2)
        shifted, shifted_bias = fit_sum_to_one_ridge(x, y + 7., .2)
        np.testing.assert_allclose(weights, shifted, atol=1e-12)
        self.assertAlmostEqual(shifted_bias - bias, 7.)

    def test_invalid_inputs_are_rejected(self):
        cases = [
            ([], [], .1),
            ([[1., 2.]], [], .1),
            ([[1., 2.]], [[1.]], .1),
            ([[float('nan')]], [1.], .1),
            ([[1.]], [float('inf')], .1),
            ([[1.]], [1.], 0.),
            ([[1.]], [1.], float('inf')),
        ]
        for x, y, penalty in cases:
            with self.subTest(features=x, target=y, penalty=penalty):
                with self.assertRaises(ValueError):
                    fit_sum_to_one_ridge(x, y, penalty)


if __name__ == '__main__':
    unittest.main()
