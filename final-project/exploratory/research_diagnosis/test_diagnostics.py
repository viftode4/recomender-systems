import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np

from exploratory.research_diagnosis import diagnostics as d


class DiagnosisTests(unittest.TestCase):
    def test_masks_and_catalog_ties(self):
        row = d.ranked_user([100, 99, 5, 5, 3], [0, 1, 0, 0, 0], [2, 4], k=2)
        np.testing.assert_array_equal(row['top'], [2, 3])
        np.testing.assert_array_equal(row['positive_ranks'], [1, 3])
        self.assertAlmostEqual(row['ndcg'], 1 / (1 + 1 / np.log2(3)))
        with self.assertRaisesRegex(ValueError, 'truth'):
            d.ranked_user([10, 9, 8, 7], [0, 1, 0, 0], [1], k=2)

    def test_positive_denominators(self):
        rows = [d.ranked_user(np.arange(15.), np.zeros(15), truth)
                for truth in ([14, 1], [12])]
        result = d.positive_summary(rows)
        self.assertEqual(result['positive_pairs'], 3)
        self.assertEqual(result['hits@10'], 2)
        self.assertEqual(result['misses@10'], 1)
        self.assertAlmostEqual(result['macro_conditional_recall@10'], .75)
        self.assertAlmostEqual(result['pooled_recall@10'], 2/3)
        conditional = d.positive_summary(rows, [np.array([True, False]), np.array([False])])
        self.assertEqual(conditional['users_with_positive_pairs'], 1)
        self.assertEqual(conditional['positive_pairs'], 1)

    def test_common_shift_scale_and_identical_oracle(self):
        a = np.arange(20.)
        b = 4 * a + 90
        first = d.ranked_user(a, np.zeros(20), [1, 13, 19])
        second = d.ranked_user(b, np.zeros(20), [1, 13, 19])
        result = d.compare_models([a], [b], [first], [second])
        self.assertAlmostEqual(result['score_pearson_mean'], 1)
        self.assertAlmostEqual(result['eligible_spearman_mean'], 1)
        self.assertEqual(result['top10_overlap_fraction'], 1)
        self.assertEqual(result['oracle_whole_list_gain_over_better_fixed_model'], 0)
        self.assertEqual(result['positive_hits_both_at_10'], 2)
        self.assertEqual(result['positive_missed_both_at_10'], 1)
        self.assertEqual(result['positive_hits_second_only_at_10'], 0)

    def test_complementarity_and_slot_budget(self):
        a = np.arange(21.)
        b = -a
        first = d.ranked_user(a, np.zeros(21), [1, 20])
        second = d.ranked_user(b, np.zeros(21), [1, 20])
        result = d.compare_models([a], [b], [first], [second])
        self.assertEqual(result['positive_hits_both_at_10'], 0)
        self.assertEqual(result['positive_hits_first_only_at_10'], 1)
        self.assertEqual(result['positive_hits_second_only_at_10'], 1)
        self.assertEqual(result['union_of_two_top10_macro_recall_up_to_20_slots'], 1)
        self.assertEqual(result['oracle_whole_list_gain_over_better_fixed_model'], 0)

    def test_barrier_failure_precedes_score_access_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'diagnosis'
            with patch.object(d, 'check_barrier', side_effect=ValueError('bad seal')):
                with patch.object(d, 'load_seed') as loader:
                    with self.assertRaisesRegex(ValueError, 'bad seal'):
                        d.run(d.ROOT / 'runs/synthetic-diagnosis', out)
                    loader.assert_not_called()
                    self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
