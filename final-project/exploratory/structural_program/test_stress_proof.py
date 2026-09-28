import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .core import atom, fit_pool, propose_edits
from . import stress_proof as stress


class StressProofTests(unittest.TestCase):
    def test_stress_pool_preserves_successful_original_calibration_exactly(self):
        fit, labels, _ = stress.sample("clean_conjunction", 16, 3101, 16, "fit")
        events = []
        old = fit_pool(fit, labels, atom(5), stress.CONFIG)
        new = stress.stress_fit_pool(fit, labels, atom(5), events)
        self.assertEqual(old.sha256, new.sha256)
        self.assertEqual(events, [])

    def test_fallback_optimizes_same_objective_and_checks_stationarity(self):
        fit, labels, _ = stress.sample("no_signal", 16, 3103, 16, "fit")
        events = []
        with patch.object(stress, "calibrate", side_effect=RuntimeError("forced solver failure")):
            fitted = stress.robust_calibrate(atom(0), fit, labels, .0001, events)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["objective_unchanged"])
        self.assertLess(events[0]["projected_gradient_residual"], 1e-5)
        self.assertTrue(np.isfinite(fitted.logits(fit)).all())

    def test_enumeration_covers_all_allowed_one_edit_neighbors(self):
        config = stress.SearchConfig(rounds=1, beam_width=2, max_nodes=5, max_depth=3, max_candidates=30)
        space = stress.grammar_space(3, config)
        self.assertEqual(len(space), len(set(space)))
        for tree in space:
            self.assertTrue(set(propose_edits(tree, 3, config)).issubset(set(space)))

    def test_random_pool_is_unique_matched_and_reproducible(self):
        space = stress.grammar_space(7)
        chosen = stress.random_trees(space, atom(5), 137, 42)
        self.assertEqual(len(chosen), 137)
        self.assertEqual(len(set(chosen)), 137)
        self.assertEqual(chosen[0], atom(5))
        self.assertEqual(chosen, stress.random_trees(space, atom(5), 137, 42))

    def test_generator_separates_partitions_and_exposes_declared_noise(self):
        fit, y, probability = stress.sample("no_signal", 100, 3101, 100, "fit")
        probe, _, _ = stress.sample("no_signal", 100, 3101, 100, "probe")
        self.assertFalse(set(fit.candidate_ids) & set(probe.candidate_ids))
        self.assertNotEqual(fit.sha256, probe.sha256)
        np.testing.assert_array_equal(probability, np.full(100, .5))
        again, again_y, _ = stress.sample("no_signal", 100, 3101, 100, "fit")
        self.assertEqual(fit.sha256, again.sha256)
        np.testing.assert_array_equal(y, again_y)

    def test_choices_persist_before_evaluation_generation_and_replay(self):
        original_sample = stress.sample
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            def guarded_sample(task, n, seed, size, partition):
                if partition == "evaluation":
                    self.assertTrue((case / "selection.json").is_file())
                    chosen = json.loads((case / "selection.json").read_text())
                    self.assertEqual(set(chosen["models"]), set(stress.METHODS))
                    self.assertFalse(chosen["evaluation_generated"])
                return original_sample(task, n, seed, size, partition)
            with patch.object(stress, "sample", side_effect=guarded_sample):
                result = stress.run_case(case, "clean_conjunction", 16, 3101, stress.grammar_space(7), evaluation_n=100)
            self.assertEqual(result["models"]["beam"]["candidate_count"], result["models"]["random_grammar"]["candidate_count"])
            chosen = json.loads((case / "selection.json").read_text())
            evaluation, _, _ = original_sample("clean_conjunction", 100, 3101, 16, "evaluation")
            for model in chosen["models"].values():
                self.assertTrue(np.isfinite(stress.replay_logits(model, evaluation)).all())


if __name__ == "__main__":
    unittest.main()
