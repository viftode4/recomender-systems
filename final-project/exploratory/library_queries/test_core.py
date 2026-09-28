"""Exact small-state checks; no generated experiment worlds or trial runner."""
from dataclasses import replace
from itertools import combinations, product as cartesian_product
import math
import unittest
from unittest.mock import patch

import numpy as np

from . import core


class ExactLibraryCoreTests(unittest.TestCase):
    def test_interpreter_all_tables_programs_inputs(self):
        self.assertEqual(core.PREDICTIONS.shape, (16, 18, 8))
        self.assertEqual([p.calls for p in core.PROGRAMS], [1] * 6 + [2] * 12)
        for table in range(16):
            def literal(a, b):
                return (table // (2 ** (2 * a + b))) % 2
            for p_index, program in enumerate(core.PROGRAMS):
                truth = []
                for x in core.INPUTS.tolist():
                    i, j = program.indices[:2]
                    if program.op == "direct":
                        value = literal(x[i], x[j])
                    elif program.op == "left":
                        value = literal(literal(x[i], x[j]), x[program.indices[2]])
                    else:
                        value = literal(x[i], literal(x[j], x[program.indices[2]]))
                    truth.append(value)
                np.testing.assert_array_equal(core.PREDICTIONS[table, p_index], truth)

    def test_exact_factorization_against_brute_joint_two_tasks(self):
        evidence = (core.Evidence.empty(2).observe(0, 0, 1).observe(0, 7, 0)
                    .observe(1, 2, 1).observe(1, 5, 0))
        posterior = core.exact_posterior(evidence)
        joint = np.zeros((16, 18, 18))
        for table, first, second in cartesian_product(range(16), range(18), range(18)):
            programs = (first, second)
            good = all(core.PREDICTIONS[table, programs[t], x] == evidence.outcomes[t, x]
                       for t in range(2) for x in range(8) if evidence.observed[t, x])
            joint[table, first, second] = int(good) / (16 * 18 ** 2)
        probability_of_evidence = joint.sum()
        joint /= probability_of_evidence
        np.testing.assert_allclose(posterior.p_library, joint.sum(axis=(1, 2)), atol=1e-15, rtol=0)
        self.assertAlmostEqual(posterior.log_evidence, math.log(probability_of_evidence), places=13)
        for table in range(16):
            mass = joint[table].sum()
            if mass:
                np.testing.assert_allclose(posterior.conditional_program[table, 0], joint[table].sum(axis=1) / mass)
                np.testing.assert_allclose(posterior.conditional_program[table, 1], joint[table].sum(axis=0) / mass)
        expected = sum(joint[l, a, b] * core.PREDICTIONS[l, a, 1]
                       for l, a, b in cartesian_product(range(16), range(18), range(18)))
        self.assertAlmostEqual(core.query_stats(posterior, 0, 1).p_y1, expected, places=13)

    def test_alias_mixture_preserves_prior_and_handles_unequal_conditionals(self):
        uniform = core.exact_posterior(core.Evidence.empty(1))
        stats = core.query_stats(uniform, 0, 0)
        classes = core.CANONICAL_CLASSES.tolist()
        self.assertEqual(len(classes), 12)
        self.assertEqual(stats.canonical_probabilities[classes.index(0)], 1 / 16)
        self.assertEqual(stats.canonical_probabilities[classes.index(2)], 2 / 16)
        weights, outcomes = np.zeros(16), np.zeros(16)
        weights[2], weights[4] = .75, .25  # same canonical class; unequal outputs
        outcomes[4] = 1.
        mixed = core.information_from_conditional(weights, outcomes)
        self.assertAlmostEqual(mixed.p_y1, .25)
        self.assertAlmostEqual(mixed.library_information, 0.)
        self.assertAlmostEqual(mixed.concrete_library_information, core.binary_entropy(.25))
        self.assertAlmostEqual(mixed.p_y1_by_canonical[classes.index(2)], .25)

    def test_information_equals_expected_reduction_of_class_entropy(self):
        posterior = core.exact_posterior(core.Evidence.empty(2).observe(0, 2, 1))
        def class_entropy(p):
            mass = np.array([p.p_library[core.CANONICAL_TABLE == group].sum() for group in core.CANONICAL_CLASSES])
            positive = mass > 0
            return float(-np.sum(mass[positive] * np.log2(mass[positive])))
        prior_entropy = class_entropy(posterior)
        for task, input_index in cartesian_product(range(2), range(8)):
            stats = core.query_stats(posterior, task, input_index)
            self.assertGreaterEqual(stats.library_information, 0.)
            self.assertLessEqual(stats.library_information, min(stats.output_entropy, prior_entropy) + 1e-12)
            expected_remaining = 0.
            for outcome, probability in ((0, 1 - stats.p_y1), (1, stats.p_y1)):
                if probability > 1e-14:
                    updated = core.exact_posterior(posterior.evidence.observe(task, input_index, outcome))
                    expected_remaining += probability * class_entropy(updated)
            self.assertAlmostEqual(stats.library_information, prior_entropy - expected_remaining, places=12)

    def test_fixed_empty_prior_witness_entropy_and_focus_rank_differently(self):
        # Explicit fixed witness: two tasks, no observations. No adaptive witness search.
        posterior = core.exact_posterior(core.Evidence.empty(2))
        entropy = core.choose_query(posterior, "entropy")
        focused = core.choose_query(posterior, "library_information")
        self.assertEqual((entropy.task, entropy.input_index), (0, 1))
        self.assertEqual((focused.task, focused.input_index), (0, 0))
        self.assertGreater(entropy.stats.output_entropy, focused.stats.output_entropy)
        self.assertGreater(focused.stats.library_information, entropy.stats.library_information)
        self.assertAlmostEqual(entropy.stats.output_entropy, .9910760598382222)
        self.assertAlmostEqual(focused.stats.library_information, .573934896284056)

    def test_query_avoids_observed_and_uses_global_tolerance_tie(self):
        posterior = core.exact_posterior(core.Evidence.empty(1))
        base = core.query_stats(posterior, 0, 0)
        values = [0., .75e-12, 1.5e-12, 0., 0., 0., 0., 0.]
        with patch.object(core, "query_stats", side_effect=lambda p, t, x: replace(base, output_entropy=values[x])):
            self.assertEqual(core.choose_query(posterior, "entropy").input_index, 1)
        known = core.exact_posterior(core.Evidence.empty(2).observe(0, 0, 0).observe(0, 1, 0))
        for criterion in ("entropy", "library_information"):
            query = core.choose_query(known, criterion)
            self.assertFalse(known.evidence.observed[query.task, query.input_index])
        exhausted = core.exact_posterior(core.Evidence(np.zeros((2, 8)), np.ones((2, 8), bool)))
        self.assertIsNone(core.choose_query(exhausted))

    def test_hidden_outcomes_are_scrubbed_and_cannot_change_queries(self):
        mask = np.zeros((2, 8), bool)
        mask[0, 2] = True
        clean = np.zeros((2, 8))
        clean[0, 2] = 1
        poisoned = np.full((2, 8), np.nan)
        poisoned[0, 2] = 1
        first, second = core.Evidence(clean, mask), core.Evidence(poisoned, mask)
        np.testing.assert_array_equal(first.outcomes, second.outcomes)
        for criterion in ("entropy", "library_information"):
            a = core.choose_query(core.exact_posterior(first), criterion)
            b = core.choose_query(core.exact_posterior(second), criterion)
            self.assertEqual((a.task, a.input_index), (b.task, b.input_index))
        clean[:] = 0
        self.assertEqual(first.outcomes[0, 2], 1)
        with self.assertRaises(ValueError):
            first.outcomes[0, 2] = 0

    def test_inconsistent_state_is_explicit_not_uniform_reset(self):
        majority = (core.INPUTS.sum(axis=1) >= 2).astype(np.int8)
        self.assertFalse(np.any(np.all(core.PREDICTIONS == majority, axis=2)))
        posterior = core.exact_posterior(core.Evidence(majority[None, :], np.ones((1, 8), bool)))
        self.assertFalse(posterior.valid)
        self.assertEqual(posterior.p_library.sum(), 0.)
        self.assertTrue(np.isfinite(posterior.p_library).all())
        self.assertTrue(np.isfinite(posterior.conditional_program).all())
        with self.assertRaises(core.InconsistentEvidence):
            core.choose_query(posterior)
        with self.assertRaises(core.InconsistentEvidence):
            core.select_map_library(posterior)
        with self.assertRaises(core.InconsistentEvidence):
            core.Evidence.empty(1).observe(0, 0, 0).observe(0, 0, 1)

    def test_empty_task_collection_and_concrete_map_ties(self):
        posterior = core.exact_posterior(core.Evidence.empty(0))
        np.testing.assert_array_equal(posterior.p_library, np.full(16, 1 / 16))
        self.assertEqual(posterior.log_evidence, 0.)
        self.assertIsNone(core.choose_query(posterior))
        self.assertEqual(core.select_map_library(posterior), 0)  # class-MAP would favor a two-table class
        altered = np.full(16, 1 / 16)
        altered[1] += 2e-16
        altered[0] -= 2e-16
        self.assertEqual(core.select_map_library(replace(posterior, p_library=altered)), 1)

    def test_transfer_freezes_library_and_minimizes_declared_tie(self):
        indices = [1, 4, 7]
        outcomes = [1, 0, 1]
        chosen = core.select_transfer_program(6, indices, outcomes)
        expected = min(range(18), key=lambda p: (
            sum(int(core.PREDICTIONS[6, p, x] != y) for x, y in zip(indices, outcomes)),
            core.PROGRAMS[p].calls, tuple(core.PREDICTIONS[6, p].tolist()), p))
        self.assertEqual(chosen.table, 6)
        self.assertEqual(chosen.program_index, expected)
        self.assertEqual(chosen.error_count, sum(chosen.predictions[indices] != outcomes))
        self.assertEqual(core.select_transfer_program(6, [], []).program.calls, 1)
        with self.assertRaises(ValueError):
            core.select_transfer_program(6, [1, 1], [0, 1])
        with self.assertRaises(ValueError):
            core.select_transfer_program(6, [1, 4], [0] * 8)

    def test_transfer_predictions_are_invariant_to_transposed_alias(self):
        for table in range(16):
            self.assertEqual(core.transpose_table(core.transpose_table(table)), table)
            for count in range(4):
                for indices in combinations(range(8), count):
                    for outcomes in cartesian_product((0, 1), repeat=count):
                        first = core.select_transfer_program(table, indices, outcomes)
                        second = core.select_transfer_program(core.transpose_table(table), indices, outcomes)
                        self.assertEqual(first.error_count, second.error_count)
                        self.assertEqual(first.program.calls, second.program.calls)
                        np.testing.assert_array_equal(first.predictions, second.predictions)


if __name__ == "__main__":
    unittest.main()
