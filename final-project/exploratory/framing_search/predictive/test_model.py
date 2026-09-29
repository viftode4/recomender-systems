"""Independent literal-feature references for grouped-pair reconstruction.

The reference physically drops every feature owned by a target and solves the
augmented primal least-squares problem, without using the production kernel.
"""
from itertools import combinations
import unittest

import numpy as np


def explicit_features(binary, groups, pair_weight, normalizer):
    binary = np.asarray(binary, dtype=float)
    groups = np.asarray(groups)
    columns = [binary[:, item] for item in range(binary.shape[1])]
    owners = [(item,) for item in range(binary.shape[1])]
    for first, second in combinations(range(binary.shape[1]), 2):
        value = (binary[:, first] * binary[:, second]
                 * (groups[:, first] == groups[:, second]))
        columns.append(np.sqrt(pair_weight / normalizer) * value)
        owners.append((first, second))
    return np.column_stack(columns), owners


def primal_reference(train, train_groups, query, query_groups, penalty,
                     pair_weight, normalizer):
    design, owners = explicit_features(train, train_groups, pair_weight, normalizer)
    query_design, _ = explicit_features(query, query_groups, pair_weight, normalizer)
    scores = np.zeros((len(query), train.shape[1]))
    for target in range(train.shape[1]):
        keep = np.array([target not in owner for owner in owners])
        reduced = design[:, keep]
        augmented = np.vstack((reduced, np.sqrt(penalty) * np.eye(keep.sum())))
        labels = np.r_[train[:, target], np.zeros(keep.sum())]
        coefficients = np.linalg.lstsq(augmented, labels, rcond=None)[0]
        scores[:, target] = query_design[:, keep] @ coefficients
    return scores

from exploratory.framing_search.predictive.model import fit, prepare_features


class GroupedPairModelTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(2909)
        self.train = (random.random((13, 7)) < .5).astype(float)
        self.train[:, [0, 6]] = 0
        self.groups = random.integers(0, 3, self.train.shape)
        self.query = (random.random((5, 7)) < .5).astype(float)
        self.query[:, 0] = 0
        self.query[0] = 0
        self.query_groups = random.integers(0, 3, self.query.shape)

    def test_group_kernel_matches_literal_complete_pair_dictionary(self):
        prepared = prepare_features(self.train, self.groups)
        features, _ = explicit_features(self.train, self.groups, 1., 1.)
        count = self.train.shape[1]
        np.testing.assert_array_equal(prepared.G, self.train @ self.train.T)
        np.testing.assert_array_equal(prepared.pair_gram,
                                      features[:, count:] @ features[:, count:].T)

    def test_constrained_predictions_match_independent_augmented_primal(self):
        prepared = prepare_features(self.train, self.groups)
        for penalty, weight in ((.3, .1), (2., 1.), (4., 0.)):
            with self.subTest(penalty=penalty, weight=weight):
                model = fit(prepared, penalty, weight)
                expected = primal_reference(self.train, self.groups, self.query,
                    self.query_groups, penalty, weight, prepared.normalizer)
                np.testing.assert_allclose(model.predict(self.query, self.query_groups),
                                           expected, atol=3e-10, rtol=3e-10)
                training_expected = primal_reference(self.train, self.groups,
                    self.train, self.groups, penalty, weight, prepared.normalizer)
                np.testing.assert_allclose(model.predict(), training_expected,
                                           atol=3e-10, rtol=3e-10)

    def test_bag_mode_matches_independent_primal_with_one_group(self):
        prepared = prepare_features(self.train)
        model = fit(prepared, .7, 1.)
        expected = primal_reference(self.train, np.zeros_like(self.train), self.query,
            np.zeros_like(self.query), .7, 1., prepared.normalizer)
        np.testing.assert_allclose(model.predict(self.query), expected,
                                   atol=3e-10, rtol=3e-10)

    def test_zero_pair_weight_matches_item_space_ease(self):
        penalty = 1.2
        inverse = np.linalg.inv(self.train.T @ self.train
                                + penalty * np.eye(self.train.shape[1]))
        coefficients = -inverse / np.diag(inverse)
        np.fill_diagonal(coefficients, 0.)
        expected = self.query @ coefficients
        for groups in (self.groups, None):
            model = fit(prepare_features(self.train, groups), penalty, 0.)
            scores = model.predict(self.query, self.query_groups if groups is not None else None)
            np.testing.assert_allclose(scores, expected, atol=2e-11, rtol=2e-11)

    def test_query_target_and_its_pair_membership_cannot_self_vote(self):
        model = fit(prepare_features(self.train, self.groups), .4, 1.)
        for target in range(1, self.train.shape[1]):
            absent = self.query.copy()
            present = self.query.copy()
            absent[:, target] = 0
            present[:, target] = 1
            changed_groups = self.query_groups.copy()
            changed_groups[:, target] = changed_groups[:, 1]
            before = model.predict(absent, self.query_groups)[:, target]
            after = model.predict(present, changed_groups)[:, target]
            np.testing.assert_allclose(before, after, atol=3e-10, rtol=3e-10)

    def test_item_permutation_preserves_corresponding_predictions(self):
        permutation = np.array([3, 0, 5, 2, 6, 1, 4])
        expected = fit(prepare_features(self.train, self.groups), 2., .1).predict(
            self.query, self.query_groups)
        actual = fit(prepare_features(self.train[:, permutation],
                                     self.groups[:, permutation]), 2., .1).predict(
            self.query[:, permutation], self.query_groups[:, permutation])
        np.testing.assert_allclose(actual, expected[:, permutation],
                                   atol=3e-10, rtol=3e-10)

    def test_row_permutation_and_row_local_label_renaming_are_invariant(self):
        permutation = np.arange(len(self.train))[::-1]
        renamed = self.groups * 11 + np.arange(len(self.train))[:, None] * 103 + 17
        query_renamed = self.query_groups * 7 + np.arange(len(self.query))[:, None] * 97
        expected = fit(prepare_features(self.train, self.groups), 2., 1.).predict(
            self.query, self.query_groups)
        actual = fit(prepare_features(self.train[permutation], renamed[permutation]),
                     2., 1.).predict(self.query, query_renamed)
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)

    def test_different_partition_can_change_same_bag_candidate_score(self):
        train = np.tile([0., 1., 1., 1., 1., 0.], (12, 1))
        train[:6, 5] = 1.
        groups = np.tile([9, 1, 2, 1, 2, 3], (12, 1))
        groups[:6] = [9, 1, 1, 2, 2, 3]
        query = np.tile([0., 1., 1., 1., 1., 0.], (2, 1))
        query_groups = np.array([[9, 1, 1, 2, 2, 3], [9, 1, 2, 1, 2, 3]])
        grouped = fit(prepare_features(train, groups), 1., 1.).predict(query, query_groups)
        self.assertGreater(grouped[0, 5], grouped[1, 5] + .1)
        bag = fit(prepare_features(train), 1., 1.).predict(query)
        np.testing.assert_array_equal(bag[0], bag[1])

    def test_cold_targets_empty_context_and_unseen_pairs_are_finite(self):
        prepared = prepare_features(self.train, self.groups)
        model = fit(prepared, 2., 1.)
        scores = model.predict(self.query, self.query_groups)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores[:, [0, 6]], 0., atol=1e-12)
        np.testing.assert_allclose(scores[0], 0., atol=1e-12)
        # All singleton groups leave no surviving same-group pair evidence.
        singleton_groups = np.tile(np.arange(7), (len(self.query), 1))
        expected = primal_reference(self.train, self.groups, self.query,
            singleton_groups, 2., 1., prepared.normalizer)
        np.testing.assert_allclose(model.predict(self.query, singleton_groups), expected,
                                   atol=3e-10, rtol=3e-10)

    def test_absent_target_fast_path_agrees_only_where_target_absent(self):
        model = fit(prepare_features(self.train, self.groups), 2., 1.)
        complete = model.predict(self.query, self.query_groups)
        fast = model.predict_unadjusted(self.query, self.query_groups)
        np.testing.assert_allclose(complete[self.query == 0], fast[self.query == 0],
                                   atol=3e-10, rtol=3e-10)

    def test_new_grouped_query_requires_its_known_partition(self):
        model = fit(prepare_features(self.train, self.groups), 2., 1.)
        with self.assertRaises(ValueError):
            model.predict(self.query)


    def test_empty_fitted_pair_dictionary_keeps_unseen_query_pairs_zero(self):
        singleton_groups = np.tile(np.arange(self.train.shape[1]), (len(self.train), 1))
        prepared = prepare_features(self.train, singleton_groups)
        self.assertEqual(prepared.pairs.shape[1], 0)
        self.assertEqual(prepared.pair_mass, 0.)
        same_group_query = np.zeros(self.query.shape, dtype=int)
        model = fit(prepared, 2., 1.)
        expected = primal_reference(self.train, singleton_groups, self.query,
            same_group_query, 2., 1., prepared.normalizer)
        np.testing.assert_allclose(model.predict(self.query, same_group_query), expected,
                                   atol=3e-10, rtol=3e-10)


if __name__ == '__main__':
    unittest.main()
