import math
import unittest

from metrics import evaluate, ranking_metrics


class MetricTests(unittest.TestCase):
    def test_hand_computed_ranking(self):
        row = ranking_metrics(["a", "b", "c"], {"b", "c", "d"}, 3)
        self.assertAlmostEqual(row["precision@3"], 2 / 3)
        self.assertAlmostEqual(row["recall@3"], 2 / 3)
        self.assertEqual(row["mrr@3"], 0.5)
        self.assertAlmostEqual(row["ndcg@3"], (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3) + 0.5))

    def test_perfect_and_missed(self):
        self.assertEqual(ranking_metrics(["a", "b"], {"a"}, 2)["ndcg@2"], 1)
        self.assertEqual(ranking_metrics(["a", "b"], {"c"}, 2)["mrr@2"], 0)

    def test_bad_rankings(self):
        for recs, truth, k in [(["a", "a"], {"a"}, 2), (["a"], {"a"}, 2), (["a"], set(), 1)]:
            with self.assertRaises(ValueError):
                ranking_metrics(recs, truth, k)

    def test_macro_average_and_coverage(self):
        result = evaluate({"u": ["a"], "v": ["b"]}, {"u": {"a"}, "v": {"a"}}, {},
                          {"a", "b", "c"}, {"a": 3}, 1)
        self.assertEqual(result["aggregate"]["recall@1"], 0.5)
        self.assertAlmostEqual(result["aggregate"]["coverage@1"], 2 / 3)
        self.assertAlmostEqual(result["aggregate"]["novelty@1"], (-math.log2(4 / 6) - math.log2(1 / 6)) / 2)

    def test_seen_unknown_or_missing_user_rejected(self):
        for recs, history in [({"u": ["a"]}, {"u": {"a"}}), ({"u": ["z"]}, {}), ({}, {})]:
            with self.assertRaises(ValueError):
                evaluate(recs, {"u": {"b"}}, history, {"a", "b"}, {"a": 1}, 1)


if __name__ == "__main__":
    unittest.main()
