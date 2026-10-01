"""Independent small examples for the coursework methods, before any study fit."""
from __future__ import annotations

import unittest

import numpy as np

from coursework_completion import models


class RankingTests(unittest.TestCase):
    def test_ranking_masks_seen_and_padding_and_breaks_ties_by_catalog(self):
        scores = np.array([[100., 8., 8., 9., 8., 7.]])
        observed = np.array([[False, False, True, False, False, False]])
        np.testing.assert_array_equal(models.rank_scores(scores, observed, k=4),
                                      [[3, 1, 4, 5]])

    def test_mixed_disjoint_lists_respect_exact_quotas(self):
        lists = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        ranks, diagnostic = models.mix_lists(lists, [2, 2, 2], k=6)
        np.testing.assert_array_equal(ranks, [1, 5, 9, 2, 6, 10])
        self.assertEqual(list(diagnostic['quota_counts']), [2, 2, 2])
        self.assertEqual(diagnostic['fallback_picks'], 0)

    def test_mixed_overlap_seen_padding_do_not_consume_quota(self):
        lists = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 7, 8, 9, 10]]
        ranks, diagnostic = models.mix_lists(lists, [3, 3], seen={2}, k=6)
        np.testing.assert_array_equal(ranks, [1, 7, 3, 8, 4, 9])
        self.assertEqual(list(diagnostic['quota_counts']), [3, 3])
        again, _ = models.mix_lists(lists, [3, 3], seen={2}, k=6)
        np.testing.assert_array_equal(ranks, again)

    def test_mixed_exhausted_expert_backfills_from_first_expert(self):
        ranks, diagnostic = models.mix_lists(
            [[1, 2, 3, 4, 5, 6, 7], [1]], [3, 3], k=6)
        np.testing.assert_array_equal(ranks, [1, 2, 3, 4, 5, 6])
        self.assertEqual(diagnostic['fallback_picks'], 3)
        self.assertEqual(list(diagnostic['quota_counts']), [3, 0])

    def test_mixed_zero_quota_and_batch_exclusion(self):
        scores = np.array([[[100, 10, 9, 8, 7, 6, 5]],
                           [[100, 1, 2, 3, 4, 5, 6]]], dtype=float)
        observed = np.array([[False, True, False, False, False, False, False]])
        ranks, _ = models.mixed_rankings(scores, observed, [0, 4], k=4)
        np.testing.assert_array_equal(ranks, [[6, 5, 4, 3]])

    def test_rrf_matches_literal_eligible_rank_sum(self):
        scores = np.array([[[99, 5, 5, 9, 1, 2]],
                           [[99, 8, 1, 8, 9, 0]]], dtype=float)
        observed = np.array([[False, False, True, False, False, False]])
        constant = 3.
        actual = models.rrf_scores(scores, observed, constant)
        expected = np.zeros(6)
        for expert in scores[:, 0]:
            ordered = sorted([1, 3, 4, 5], key=lambda j: (-expert[j], j))
            for position, item in enumerate(ordered, 1):
                expected[item] += 1. / (constant + position)
        np.testing.assert_allclose(actual[0, [1, 3, 4, 5]], expected[[1, 3, 4, 5]],
                                   rtol=1e-14, atol=1e-14)
        ranks = models.rank_scores(actual, observed, 4)
        np.testing.assert_array_equal(ranks, [sorted([1, 3, 4, 5],
                                                   key=lambda j: (-expected[j], j))])


class MetaLevelTests(unittest.TestCase):
    def test_both_stages_match_independent_augmented_least_squares(self):
        binary = np.array([[1, 0, 1, 0], [0, 1, 1, 1],
                           [1, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
        genres = np.array([[1, 0, 0], [0, 1, 0],
                           [.5, .5, 0], [0, .5, .5]])
        penalty = .7
        n, m = binary.shape
        g = genres.shape[1]
        # Stage 1: each user learns a content representation, directly from X.
        p = np.linalg.lstsq(np.vstack([genres, np.sqrt(penalty) * np.eye(g)]),
                            np.vstack([binary.T, np.zeros((g, n))]),
                            rcond=None)[0].T
        # Stage 2: each item is learned from those representations, directly from X.
        q = np.linalg.lstsq(np.vstack([p, np.sqrt(penalty) * np.eye(g)]),
                            np.vstack([binary, np.zeros((g, m))]),
                            rcond=None)[0].T
        fit = models.fit_meta_level(binary, genres, penalty)
        np.testing.assert_allclose(fit['profiles'], p, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit['item_factors'], q, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fit['scores'], p @ q.T, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(fit['scores'][3], np.zeros(m))

    def test_genre_basis_permutation_does_not_change_scores(self):
        binary = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        genres = np.array([[1, 0], [0, 1], [.5, .5]])
        a = models.fit_meta_level(binary, genres, 2.)
        b = models.fit_meta_level(binary, genres[:, ::-1], 2.)
        np.testing.assert_allclose(a['scores'], b['scores'], atol=1e-12)

    def test_zero_observations_cannot_manufacture_signal(self):
        fit = models.fit_meta_level(np.zeros((3, 3)), np.array([[1, 0], [0, 1], [.5, .5]]), 1.)
        np.testing.assert_array_equal(fit['profiles'], np.zeros((3, 2)))
        np.testing.assert_array_equal(fit['item_factors'], np.zeros((3, 2)))
        np.testing.assert_array_equal(fit['scores'], np.zeros((3, 3)))


class SwitchingTests(unittest.TestCase):
    def test_quantile_equality_uses_upper_bin_and_meta_only_winners(self):
        activity = np.array([0, 1, 2, 3, 4, 5, 6.])
        # Three groups have boundaries 2 and 4. Only users 0,2,4 select experts.
        policy = models.fit_switch(activity, np.array([[1., 0., 1.], [0., 1., 0.]]),
                                   np.array([0, 2, 4]), 3, ['first', 'second'])
        experts = {'first': np.full((7, 2), 10.), 'second': np.full((7, 2), 20.)}
        actual = models.switch_scores(experts, activity, policy)
        np.testing.assert_array_equal(actual[:, 0], [10, 10, 20, 20, 10, 10, 10])

    def test_empty_groups_use_global_meta_winner_and_exact_ties_first(self):
        activity = np.array([0, 1, 2, 3, 4, 5, 6.])
        experts = {'first': np.full((7, 2), 10.), 'second': np.full((7, 2), 20.)}
        policy = models.fit_switch(activity, np.array([[0., 0.], [1., 1.]]),
                                   np.array([0, 1]), 3, ['first', 'second'])
        np.testing.assert_array_equal(models.switch_scores(experts, activity, policy),
                                      np.full((7, 2), 20.))
        tie = models.fit_switch(activity, np.ones((2, 2)), np.array([0, 1]), 3,
                                ['first', 'second'])
        np.testing.assert_array_equal(models.switch_scores(experts, activity, tie),
                                      np.full((7, 2), 10.))

    def test_tied_activity_collapses_groups_without_label_dependent_thresholds(self):
        activity = np.ones(5)
        policy = models.fit_switch(activity, np.array([[0., 0.], [1., 1.]]),
                                   np.array([1, 4]), 3, ['first', 'second'])
        experts = {'first': np.zeros((5, 1)), 'second': np.ones((5, 1))}
        np.testing.assert_array_equal(models.switch_scores(experts, activity, policy),
                                      np.ones((5, 1)))


if __name__ == '__main__':
    unittest.main()
