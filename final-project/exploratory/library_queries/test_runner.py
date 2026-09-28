"""Independent finite runner checks; no experiment launch or official seeds."""
from dataclasses import replace
import io
from itertools import combinations
import json
import unittest
from unittest.mock import patch

import numpy as np

from . import core
from . import run_experiment as runner


class FiniteQueryRunnerTests(unittest.TestCase):
    def test_vectorized_transfer_matches_literal_support_only_search(self):
        evaluator = runner.TransferEvaluator(core)
        truths = ([0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0])
        for model in (3, 6, "all_tables"):
            pairs = ([(model, p) for p in range(18)] if isinstance(model, int)
                     else [(t, p) for t in range(16) for p in range(18)])
            for truth in truths:
                for size in (2, 4):
                    risks, support_errors, calls = [], [], []
                    for support in combinations(range(8), size):
                        labels = [truth[x] for x in support]
                        def key(tp):
                            table, program = tp
                            prediction = core.PREDICTIONS[table, program]
                            return (sum(int(prediction[x]) != y for x, y in zip(support, labels)),
                                    core.PROGRAMS[program].calls, tuple(prediction), table, program)
                        table, program = min(pairs, key=key)
                        prediction = core.PREDICTIONS[table, program]
                        chosen, errors = evaluator.fit_supports(model, [support], [labels])
                        np.testing.assert_array_equal(
                            evaluator.models[model][0][chosen[0]], prediction)
                        self.assertEqual(errors[0], key((table, program))[0])
                        remaining = [x for x in range(8) if x not in support]
                        risks.append(sum(int(prediction[x]) != truth[x] for x in remaining) / len(remaining))
                        support_errors.append(int(errors[0]))
                        calls.append(core.PROGRAMS[program].calls)
                    actual = evaluator.evaluate(model, truth, size)
                    self.assertAlmostEqual(actual["bounded_loss"], np.mean(risks), places=14)
                    self.assertEqual(actual["support_error_count_total"], sum(support_errors))
                    self.assertEqual(actual["fits_with_nonzero_support_error"], sum(x > 0 for x in support_errors))
                    self.assertAlmostEqual(actual["mean_program_calls"], np.mean(calls), places=14)
                    self.assertAlmostEqual(actual["exact_complement_accuracy"], np.mean(np.asarray(risks) == 0), places=14)

    def test_changing_heldout_complement_cannot_change_first_support_choice(self):
        evaluator = runner.TransferEvaluator(core)
        first = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        second = first.copy()
        second[2:] = 1
        for model in (6, "all_tables"):
            a = evaluator.evaluate(model, first, 2)["first_support_example"]
            b = evaluator.evaluate(model, second, 2)["first_support_example"]
            self.assertEqual(a["support_indices"], [0, 1])
            for field in ("support_labels", "selected_table", "selected_program", "selected_predictions"):
                self.assertEqual(a[field], b[field])

    def test_semantic_deduplication_empty_universe_and_failure_are_distinct(self):
        evaluator = runner.TransferEvaluator(core)
        for table in range(16):
            train = core.PREDICTIONS[table, [0, 0, 1]]
            truths, total, excluded = evaluator.universe(table, train)
            expected = {tuple(x) for x in core.PREDICTIONS[table]} - {tuple(x) for x in train}
            self.assertEqual(truths, sorted(expected))
            self.assertEqual(total - excluded, len(truths))
        empty = evaluator.aggregate(None, 0, [], 2)
        self.assertEqual(empty["status"], "empty_transfer_universe")
        self.assertIsNone(empty["learned"])
        failed = evaluator.aggregate(None, 6, [tuple(core.PREDICTIONS[6, 0])], 2)
        self.assertEqual(failed["learned"]["bounded_loss"], 1.)
        self.assertIsNone(failed["learned"]["conditional_error"])
        self.assertEqual(failed["learned"]["prediction_coverage"], 0.)
        self.assertFalse(failed["all_tables"]["abstained"])
        self.assertFalse(failed["oracle"]["abstained"])
        json.dumps([empty, failed], allow_nan=False)

    def test_runner_query_tie_is_against_global_maximum(self):
        posterior = core.exact_posterior(core.Evidence.empty(3))
        base = core.query_stats(posterior, 0, 0)
        values = np.zeros((3, 8))
        values[0, :3] = [0., .75e-12, 1.5e-12]
        with patch.object(core, "query_stats", side_effect=lambda p, t, x:
                          replace(base, output_entropy=values[t, x])):
            pair, _, requests, _ = runner.choose_from_observations(
                core, posterior, np.zeros((3, 8), bool), "entropy", range(24))
        self.assertEqual(pair, (0, 1))
        self.assertEqual(requests, 24)

    def test_acquisition_preserves_prefix_budget_and_explicit_inconsistency(self):
        # Constants from different tables violate sharing once enough inputs are seen.
        world = {"seed": -100, "regime": "no_sharing", "table_key": 0,
                 "initial_inputs": [0, 0, 0], "random_order": list(range(24)),
                 "oracle_labels": np.array([[0]*8, [1]*8, [0]*8], dtype=np.uint8)}
        failures = io.StringIO()
        checkpoints = runner.acquire(core, world, "random", failures)
        self.assertEqual(set(checkpoints), set(runner.BUDGETS))
        final_queries = checkpoints[12]["queries"]
        self.assertEqual(len({(q["task"], q["input"]) for q in final_queries}), 15)
        for budget, snapshot in checkpoints.items():
            self.assertEqual(snapshot["oracle_queries"], 3 + budget)
            self.assertEqual(snapshot["queries"], final_queries[:3 + budget])
            self.assertEqual(snapshot["cost"]["posterior_requests"], 1 + budget)
        self.assertTrue(checkpoints[12]["abstained"])
        self.assertIsNone(checkpoints[12]["selected_table"])
        first_failure = checkpoints[12]["first_failure_extra_queries"]
        self.assertLessEqual(first_failure, 7)
        for budget, snapshot in checkpoints.items():
            if budget >= first_failure:
                self.assertTrue(snapshot["abstained"])
                self.assertEqual(sum(snapshot["table_probabilities"]), 0.)
        self.assertTrue(failures.getvalue())
        with self.assertRaisesRegex(RuntimeError, "Shared generating hypothesis"):
            runner.acquire(core, {**world, "regime": "shared"}, "random", io.StringIO())

    def test_summary_weights_available_seeds_within_tables_before_tables(self):
        records = []
        for regime in ("shared", "no_sharing"):
            for arm in runner.ARMS:
                for budget in runner.BUDGETS:
                    for table in (0, 1):
                        for seed in (-20, -19):
                            available = not (table == 0 and seed == -19)
                            value = (.2 if table == 0 else .8) + (.1 if arm == "library_information" else 0.)
                            controls = {field: value for field in (
                                "bounded_loss", "conditional_error", "support_error", "mean_support_error_count",
                                "exact_complement_accuracy", "mean_program_calls")}
                            controls["prediction_coverage"] = 1.
                            transfer = {"truth_functions": int(available),
                                        **{name: dict(controls) if available else None
                                           for name in ("learned", "all_tables", "oracle")}}
                            records.append({"regime": regime, "seed": seed, "arm": arm,
                                            "extra_queries": budget, "true_table": table,
                                            "training_world_key": table if regime == "shared" else 0,
                                            "abstained": False,
                                            "transfer": {str(size): transfer for size in runner.SUPPORT_SIZES}})
        summary = runner.summarize(records, [], runner.TransferEvaluator(core), 0.)
        for row in summary["metrics"]:
            expected = .6 if row["arm"] == "library_information" else .5
            self.assertAlmostEqual(row["controls"]["learned"]["bounded_loss"]["mean"], expected)
            self.assertEqual(row["available_transfer_cells"], 3)
            self.assertEqual(row["empty_transfer_cells"], 1)
            self.assertEqual(row["acquisition_training_worlds"], 2 if row["regime"] == "no_sharing" else 4)
        for comparison in summary["focused_minus_entropy"]:
            self.assertAlmostEqual(comparison["mean"], .1)
            self.assertEqual(comparison["paired_cells"], 3)


if __name__ == "__main__":
    unittest.main()
