import unittest

import numpy as np

from negative_controls import MAX_SWITCH_ATTEMPTS, switch_negative_labels


class NegativeControlTests(unittest.TestCase):
    def test_only_checkerboard_swaps_and_exact_margins_on_irregular_support(self):
        support = np.array([[0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 0, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=bool)
        negative = np.array([[0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1]], dtype=bool)
        before_negative, before_support = negative.copy(), support.copy()
        for seed in range(5):
            result, info = switch_negative_labels(negative, support, seed, 31)
            np.testing.assert_array_equal(result.sum(axis=0), negative.sum(axis=0))
            np.testing.assert_array_equal(result.sum(axis=1), negative.sum(axis=1))
            self.assertFalse(np.any(result & ~support))
            np.testing.assert_array_equal(result[3], negative[3])
            self.assertGreater(info['accepted'], 0)
            self.assertEqual(info['attempts'], 31 * int(negative.sum()))
            self.assertTrue(all(info['invariants'].values()))
            self.assertEqual(info['initially_flippable_rows'], 3)
            self.assertEqual(info['rows_ever_switched'], 3)
            self.assertEqual(info['final_overlap'], int(np.sum(result & negative)))
        np.testing.assert_array_equal(negative, before_negative)
        np.testing.assert_array_equal(support, before_support)

    def test_seed_reproducible_and_padding_column_only(self):
        support = np.ones((4, 6), dtype=bool)
        support[:, 0] = False
        negative = np.array([[0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0],
                             [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]], dtype=bool)
        first, info = switch_negative_labels(negative, support, 99)
        second, other = switch_negative_labels(negative, support, 99)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info, other)
        self.assertFalse(first[:, 0].any())
        self.assertEqual(first[0].sum(), 2)  # User zero is real, not padding.

    def test_rejected_proposals_are_counted(self):
        support = np.array([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=bool)
        negative = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=bool)
        _, info = switch_negative_labels(negative, support, 8, 100)
        self.assertEqual(info['attempts'], 200)
        self.assertGreater(info['accepted'], 0)
        self.assertLess(info['accepted'], info['attempts'])
        self.assertEqual(info['overlap_trace'][-1]['attempts'], 200)

    def test_structural_zeros_can_prevent_all_two_by_two_moves(self):
        # A six-cycle has two perfect matchings, but no allowed 2x2 switch.
        support = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=bool)
        negative = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
        result, info = switch_negative_labels(negative, support, 42)
        np.testing.assert_array_equal(result, negative)
        self.assertEqual(info['fixed_support_row_pairs'], 0)
        self.assertEqual(info['accepted'], 0)
        self.assertEqual(info['initially_flippable_rows'], 0)
        self.assertIn('no global connectivity', info['sampling_claim'])

    def test_fixed_groups_block_cross_group_moves(self):
        support = np.array([[0, 1, 1]] * 4, dtype=bool)
        negative = np.array([[0, 1, 0], [0, 1, 0],
                             [0, 0, 1], [0, 0, 1]], dtype=bool)
        result, info = switch_negative_labels(negative, support, 17,
                                              row_groups=[0, 0, 1, 1])
        np.testing.assert_array_equal(result, negative)
        self.assertEqual(info['accepted'], 0)
        self.assertEqual(info['fixed_support_row_pairs'], 2)
        self.assertEqual(info['attempts'], 50 * int(negative.sum()))
        self.assertTrue(info['invariants']['group_item_negative_counts_preserved'])
        _, unrestricted = switch_negative_labels(negative, support, 17)
        self.assertGreater(unrestricted['accepted'], 0)

    def test_switches_preserve_each_groups_item_counts_with_one_global_budget(self):
        support = np.array([[0, 1, 1, 1]] * 6, dtype=bool)
        negative = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                             [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
        groups = np.array(['fit']*3 + ['excluded']*3)
        result, info = switch_negative_labels(negative, support, 18, 31, row_groups=groups)
        for group in np.unique(groups):
            np.testing.assert_array_equal(result[groups == group].sum(axis=0),
                                          negative[groups == group].sum(axis=0))
        self.assertGreater(info['accepted'], 0)
        self.assertEqual(info['attempts'], 31 * int(negative.sum()))
        self.assertTrue(all(g['item_negative_counts_preserved'] and
                            g['row_negative_counts_preserved'] for g in info['group_margins']))
        self.assertTrue(info['row_groups_supplied'])

    def test_zero_negative_zero_budget_and_unflippable_labels(self):
        support = np.array([[0, 1, 1], [0, 1, 1]], dtype=bool)
        for negative in (np.zeros_like(support), support):
            result, info = switch_negative_labels(negative, support, 42)
            np.testing.assert_array_equal(result, negative)
            self.assertEqual(info['accepted'], 0)
            self.assertEqual(info['changed_label_fraction'], 0)
        result, info = switch_negative_labels(support, support, 42, 0)
        np.testing.assert_array_equal(result, support)
        self.assertEqual(info['attempts'], 0)

    def test_cap_declared_even_when_support_has_no_moves(self):
        mask = np.array([[0, 1]], dtype=bool)
        _, info = switch_negative_labels(mask, mask, 1, MAX_SWITCH_ATTEMPTS + 1)
        self.assertEqual(info['attempt_budget'], MAX_SWITCH_ATTEMPTS)
        self.assertEqual(info['attempts'], 0)

    def test_invalid_inputs_fail_before_perturbing(self):
        valid = np.array([[0, 1, 0], [0, 0, 1]])
        for invalid in (np.array([0, 1]), np.array([[0, 2]]),
                        np.array([[0, np.nan]])):
            with self.assertRaises(ValueError):
                switch_negative_labels(invalid, invalid, 1)
        with self.assertRaisesRegex(ValueError, 'same shape'):
            switch_negative_labels(valid, valid[:, :2], 1)
        with self.assertRaisesRegex(ValueError, 'padding'):
            switch_negative_labels([[1, 0]], [[1, 1]], 1)
        with self.assertRaisesRegex(ValueError, 'inside'):
            switch_negative_labels(valid, np.zeros_like(valid), 1)
        for invalid in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                switch_negative_labels(valid, valid, 1, invalid)
        for groups in ([0], [[0], [1]], [0, np.nan]):
            with self.assertRaisesRegex(ValueError, 'row_groups'):
                switch_negative_labels(valid, valid, 1, row_groups=groups)


if __name__ == '__main__':
    unittest.main()
