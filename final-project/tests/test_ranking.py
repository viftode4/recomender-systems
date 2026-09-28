import unittest
import numpy as np

from run import top_k


class RankingTests(unittest.TestCase):
    def test_padding_history_and_stable_ties(self):
        self.assertEqual(top_k([99, 4, 4, 5], ["padding", "a", "b", "c"], {"c"}, 2), ["a", "b"])

    def test_not_enough_unseen(self):
        with self.assertRaises(ValueError):
            top_k([0, 1, 2], ["padding", "a", "b"], {"a"}, 2)

    def test_nonfinite_candidate_rejected(self):
        with self.assertRaises(ValueError):
            top_k([0, np.nan], ["padding", "a"], set(), 1)
