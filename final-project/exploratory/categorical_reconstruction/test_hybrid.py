"""Synthetic-only tests: no MovieLens partitions or model outputs are opened."""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from exploratory.categorical_reconstruction.hybrid import (
    ARMS,
    EXPERTS,
    PENALTIES,
    _fit_pipeline,
    _normalize,
    _sample_cells,
    prepare_hybrids,
    rank_scores,
)


def fixture():
    users = [str(i) for i in range(1, 9)]
    items = ["[PAD]"] + [str(i) for i in range(1, 21)]
    observed = {user: {"1", "2"} for user in users}
    meta = set(users[:6])
    validation = {user: {str(3 + (int(user) % 3))} for user in sorted(meta)}
    rng = np.random.default_rng(914)
    scores = {expert: rng.normal(size=(len(users), len(items))) for expert in EXPERTS}
    return scores, users, items, observed, validation, meta, 2026


class GuardedValidation(dict):
    """A development value must not be read, even to reject an invalid map."""

    def __getitem__(self, key):
        if key == "7":
            raise AssertionError("Development validation value was read")
        return super().__getitem__(key)


class HybridTests(unittest.TestCase):
    def test_constraints_hold_on_calibrated_weights_and_scores_are_not_clipped(self):
        # Both experts correlate positively with the binary response, but their
        # correlated error makes subtraction useful. Nonnegative calibration
        # must not accidentally turn the later affine mixture into a simplex.
        y = np.array([0., 0., 1., 1.])
        sign, noise = 2 * y - 1, np.array([-1., 1., -1., 1.])
        x = np.column_stack((sign + noise, .5 * sign + noise))
        fitted = _fit_pipeline(x, y, .001)
        weights = np.asarray(fitted["weights"])
        self.assertAlmostEqual(float(weights.sum()), 1., places=12)
        self.assertGreater(weights[0], 1.)
        self.assertLess(weights[1], 0.)
        self.assertTrue(np.all(np.asarray(fitted["calibration_slopes"]) >= 0))
        predicted = x @ np.asarray(fitted["effective_weights"]) + fitted["effective_intercept"]
        self.assertLess(predicted.min(), 0.)
        self.assertGreater(predicted.max(), 1.)
        self.assertFalse(np.isclose(sum(fitted["effective_weights"]), 1.))

    def test_normalization_and_ranking_exclude_train_and_padding(self):
        scores, users, items, observed, validation, meta, seed = fixture()
        first, metadata = prepare_hybrids(scores, users, items, observed, validation, meta, seed)
        changed = {name: value.copy() for name, value in scores.items()}
        for value in changed.values():
            value[:, :3] = [1e200, -1e200, np.nan]
        second, other_metadata = prepare_hybrids(changed, users, items, observed, validation, meta, seed)
        self.assertEqual(metadata, other_metadata)
        for arm in ARMS:
            np.testing.assert_array_equal(first[arm], second[arm])
            self.assertTrue(np.isfinite(first[arm]).all())
            np.testing.assert_array_equal(first[arm][:, :3], 0.)
        eligible = np.arange(3, len(items))
        misleading = np.full(len(items), -5.)
        misleading[:3] = 1e100
        np.testing.assert_array_equal(rank_scores(misleading, eligible), np.arange(3, 13))
        standardized = _normalize(scores["binary"], [eligible] * len(users))
        np.testing.assert_allclose(standardized[:, eligible].mean(axis=1), 0., atol=1e-14)
        np.testing.assert_allclose(standardized[:, eligible].std(axis=1), 1., atol=1e-14)

    def test_sampler_uses_same_unique_cells_and_excludes_every_meta_positive(self):
        users = ["u1", "u2"]
        user_index = {user: row for row, user in enumerate(users)}
        eligible = [np.arange(2, 18), np.arange(3, 13)]
        truth = {"u1": np.array([3, 5, 6]), "u2": np.array([4, 7])}
        rows, columns, labels, audit = _sample_cells(users, user_index, truth, eligible, 99)
        repeated = _sample_cells(list(reversed(users)), user_index, truth, eligible, 99)
        for actual, expected in zip((rows, columns, labels), repeated[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(set(zip(rows.tolist(), columns.tolist()))), len(rows))
        for user, row in user_index.items():
            negatives = set(columns[(rows == row) & (labels == 0)])
            positives = set(columns[(rows == row) & (labels == 1)])
            self.assertEqual(positives, set(truth[user]))
            self.assertFalse(negatives & set(truth[user]))
            self.assertTrue(negatives <= set(eligible[row]))
        self.assertEqual(audit["requested_negatives"], 25)
        self.assertEqual(audit["users_with_negative_shortage"], 2)
        self.assertEqual(int((labels == 0).sum()), 21)

    def test_development_labels_rejected_before_values_are_read(self):
        args = list(fixture())
        guarded = GuardedValidation(args[4])
        guarded["7"] = object()
        args[4] = guarded
        with self.assertRaisesRegex(ValueError, "development labels are forbidden"):
            prepare_hybrids(*args)

    def test_development_score_changes_cannot_select_or_fit_hybrid(self):
        args = fixture()
        original_scores, original = prepare_hybrids(*args)
        modified = {name: scores.copy() for name, scores in args[0].items()}
        for index, values in enumerate(modified.values()):
            values[6:] = np.random.default_rng(index).normal(size=values[6:].shape) * 30
        changed_scores, changed = prepare_hybrids(modified, *args[1:])
        self.assertEqual(original, changed)
        for arm in ARMS:
            np.testing.assert_array_equal(original_scores[arm][:6], changed_scores[arm][:6])

    def test_inner_split_and_final_refit_are_shared_and_deterministic(self):
        args = fixture()
        fit_sizes = []
        def capture(features, labels, penalty):
            fit_sizes.append((len(labels), penalty))
            return _fit_pipeline(features, labels, penalty)
        with patch("exploratory.categorical_reconstruction.hybrid._fit_pipeline", side_effect=capture):
            scores, metadata = prepare_hybrids(*args)
        other_scores, other = prepare_hybrids(*args)
        self.assertEqual(metadata, other)
        expected_users = np.asarray(sorted(args[5]))
        np.random.default_rng(args[6] + 24012).shuffle(expected_users)
        self.assertEqual(metadata["cohorts"]["coefficient_fit"], sorted(expected_users[:3]))
        self.assertEqual(metadata["cohorts"]["penalty_selection"], sorted(expected_users[3:]))
        self.assertEqual(metadata["sampling"]["positive_examples"], 6)
        self.assertEqual(metadata["sampling"]["negative_examples"], 30)
        for index, arm in enumerate(ARMS):
            self.assertEqual(fit_sizes[index * 5:index * 5 + 4], [(18, penalty) for penalty in PENALTIES])
            self.assertEqual(fit_sizes[index * 5 + 4][0], 36)
            row = metadata["arms"][arm]
            self.assertEqual(row["final"]["training_examples"], 36)
            self.assertAlmostEqual(sum(row["final"]["weights"]), 1., places=12)
            self.assertEqual([entry["penalty"] for entry in row["candidates"]], list(PENALTIES))
            self.assertEqual(row["selected_penalty"], max(row["candidates"], key=lambda x: (x["selection_ndcg@10"], x["penalty"]))["penalty"])
            np.testing.assert_array_equal(scores[arm], other_scores[arm])
        json.dumps(metadata, allow_nan=False)

    def test_constant_scores_choose_larger_penalty_and_duplicate_is_disclosed(self):
        args = list(fixture())
        args[0] = {key: np.ones_like(value) for key, value in args[0].items()}
        scores, metadata = prepare_hybrids(*args)
        for arm, names in ARMS.items():
            self.assertEqual(metadata["arms"][arm]["selected_penalty"], 1.)
            np.testing.assert_allclose(metadata["arms"][arm]["final"]["weights"], 1 / len(names))
            np.testing.assert_array_equal(rank_scores(scores[arm][0], np.arange(3, len(args[2]))), np.arange(3, 13))
        for expert in ("real", "shuffled"):
            self.assertTrue(metadata["duplicates"][expert]["identical_to_binary_on_eligible_scores"])
            self.assertTrue(metadata["duplicates"][expert]["duplicate_retained"])

    def test_meta_split_471_users_is_235_and_236(self):
        users = [str(i) for i in range(1, 474)]
        items = ["0"] + [str(i) for i in range(1, 13)]
        scores = {expert: np.zeros((len(users), len(items))) for expert in EXPERTS}
        observed = {user: {"1"} for user in users}
        meta = users[:471]
        validation = {user: {"2"} for user in meta}
        _, metadata = prepare_hybrids(scores, users, items, observed, validation, meta, 2026)
        self.assertEqual(len(metadata["cohorts"]["coefficient_fit"]), 235)
        self.assertEqual(len(metadata["cohorts"]["penalty_selection"]), 236)
        self.assertEqual(len(metadata["cohorts"]["meta_fit"]), 471)

    def test_invalid_inputs_fail_and_inputs_are_unchanged(self):
        args = fixture()
        snapshot = copy.deepcopy(args)
        prepare_hybrids(*args)
        for expert in EXPERTS:
            np.testing.assert_array_equal(args[0][expert], snapshot[0][expert])
        self.assertEqual(args[1:], snapshot[1:])
        scenarios = []
        invalid = list(copy.deepcopy(args)); invalid[0]["real"][0, 4] = np.nan; scenarios.append(invalid)
        invalid = list(copy.deepcopy(args)); invalid[4]["1"] = {"1"}; scenarios.append(invalid)
        invalid = list(copy.deepcopy(args)); invalid[4]["1"] = set(); scenarios.append(invalid)
        invalid = list(copy.deepcopy(args)); invalid[0].pop("slim"); scenarios.append(invalid)
        invalid = list(copy.deepcopy(args)); invalid[3]["1"] = {"100"}; scenarios.append(invalid)
        for index, invalid in enumerate(scenarios):
            with self.subTest(index=index), self.assertRaises(ValueError):
                prepare_hybrids(*invalid)


if __name__ == "__main__":
    unittest.main()
