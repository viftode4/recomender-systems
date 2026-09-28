"""Hand-calculated and metamorphic checks for aggregate-only group evaluation."""
from collections import Counter
from copy import deepcopy
import json
import math
import unittest

from exploratory.categorical_reconstruction.group_analysis import analyze_groups


def fixture():
    catalog = list("abcdefghij")
    history = {f"private-user-{n}": set(catalog[:n+1]) for n in range(6)}
    counts = Counter(item for items in history.values() for item in items)
    genres = {item: {"A" if item in "abef" else "B"} for item in catalog}
    genres["i"], genres["j"] = {"C"}, {"A", "B"}
    recs = {"private-user-0": ["b", "c"], "private-user-4": ["f", "g"],
            "private-user-5": ["g", "h"]}
    truth = {"private-user-0": {"b", "d"}, "private-user-4": {"f", "h"},
             "private-user-5": {"g", "h"}}
    return recs, truth, history, catalog, counts, genres


class AggregateGroupAnalysisTests(unittest.TestCase):
    def test_hand_calculated_user_and_conditional_item_metrics(self):
        result = analyze_groups(*fixture(), k=2)
        ideal = 1 + 1/math.log2(3)
        self.assertAlmostEqual(result["aggregate"]["ndcg@2"], (2/ideal+1)/3)
        self.assertAlmostEqual(result["aggregate"]["recall@2"], 2/3)
        self.assertAlmostEqual(result["aggregate"]["diversity"], 2/3)
        self.assertEqual(result["user_groups"]["sparse"]["users"], 1)
        self.assertEqual(result["user_groups"]["dense"]["users"], 2)
        self.assertEqual(result["user_groups"]["medium"]["status"], "empty")
        self.assertIsNone(result["user_groups"]["medium"]["ndcg@2"])
        self.assertAlmostEqual(result["user_groups"]["dense"]["diversity"], .5)
        self.assertEqual(result["item_groups"]["head"]["users_with_positives"], 1)
        self.assertEqual(result["item_groups"]["head"]["recall@2"], 1)
        self.assertEqual(result["item_groups"]["tail"]["users_with_positives"], 3)
        self.assertAlmostEqual(result["item_groups"]["tail"]["recall@2"], .5)
        self.assertAlmostEqual(result["item_groups"]["head"]["exposure"], 1/6)
        self.assertAlmostEqual(result["item_groups"]["head"]["discounted_exposure"], 1/(3*ideal))
        self.assertAlmostEqual(result["item_groups"]["head"]["mean_item_exposure"], 1/6)
        expected_novelty = (-math.log2(6/31)-math.log2(5/31))/2
        self.assertAlmostEqual(result["user_groups"]["sparse"]["novelty@2"], expected_novelty)

    def test_train_population_ties_stay_together_and_empty_item_targets_are_null(self):
        catalog = list("abcde")
        history = {"one": {"a"}, "two": {"a"}, "three": {"a"}}
        result = analyze_groups({"one": ["b", "c"]}, {"one": {"b"}}, history,
                                catalog, {"a": 3}, {i: {"same"} for i in catalog}, k=2)
        self.assertEqual(result["definitions"]["activity_thresholds"], [1., 1.])
        self.assertEqual(result["user_groups"]["dense"]["users"], 1)
        self.assertEqual(result["user_groups"]["sparse"]["users"], 0)
        self.assertEqual(result["user_groups"]["medium"]["users"], 0)
        self.assertIsNone(result["item_groups"]["head"]["recall@2"])
        self.assertEqual(result["item_groups"]["head"]["users_with_positives"], 0)
        self.assertEqual(result["aggregate"]["diversity"], 0.)
        self.assertEqual(result["aggregate"]["calibration_jsd"], 0.)

    def test_truth_changes_cannot_change_train_groups_or_exposure(self):
        args = fixture()
        first = analyze_groups(*args, k=2)
        changed = deepcopy(args[1])
        changed["private-user-0"] = {"i"}
        second = analyze_groups(args[0], changed, *args[2:], k=2)
        self.assertEqual(first["definitions"], second["definitions"])
        self.assertEqual(first["item_exposure"], second["item_exposure"])
        for group in ("sparse", "medium", "dense"):
            for metric in ("users", "diversity", "calibration_jsd", "head_exposure", "novelty@2"):
                self.assertEqual(first["user_groups"][group][metric], second["user_groups"][group][metric])

    def test_catalog_permutation_cannot_change_frequency_tie_membership(self):
        catalog = ["d", "c", "b", "a", "e"]
        history = {"one": {"a", "b"}, "two": {"b", "c"}, "three": {"c", "a"}}
        recs, truth = {"two": ["a", "d"]}, {"two": {"a"}}
        counts = {"a": 2, "b": 2, "c": 2}
        genres = {item: {"genre"} for item in catalog}
        first = analyze_groups(recs, truth, history, catalog, counts, genres, k=2)
        second = analyze_groups(recs, truth, history, list(reversed(catalog)), counts, genres, k=2)
        self.assertEqual(first, second)
        self.assertEqual(first["item_groups"]["head"]["recall@2"], 1.)
        self.assertEqual(first["item_groups"]["head"]["exposure"], .5)

    def test_rejects_seen_unknown_truth_and_mismatched_train_counts(self):
        recs, truth, history, catalog, counts, genres = fixture()
        with self.assertRaisesRegex(ValueError, "counts"):
            analyze_groups(recs, truth, history, catalog, {**counts, "a": 5}, genres, k=2)
        with self.assertRaisesRegex(ValueError, "overlaps TRAIN"):
            analyze_groups(recs, {**truth, "private-user-0": {"a"}}, history, catalog, counts, genres, k=2)
        with self.assertRaisesRegex(ValueError, "Seen item"):
            analyze_groups({**recs, "private-user-0": ["a", "b"]}, truth, history, catalog, counts, genres, k=2)
        with self.assertRaisesRegex(ValueError, "Unknown item"):
            analyze_groups(recs, {**truth, "private-user-0": {"unknown"}}, history, catalog, counts, genres, k=2)
        with self.assertRaisesRegex(ValueError, "population"):
            analyze_groups(recs, truth, history, catalog, counts, genres, k=2, group_users=list(recs))

    def test_aggregate_output_has_no_identifiers_and_does_not_mutate_inputs(self):
        args = fixture()
        before = deepcopy(args)
        result = analyze_groups(*args, k=2)
        self.assertEqual(args, before)
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("per_user", serialized)
        self.assertNotIn('"a"', serialized)
        self.assertNotIn('"b"', serialized)


if __name__ == "__main__":
    unittest.main()
