import unittest
import numpy as np

from .core import (QueryAtoms, SearchConfig, Tree, atom, build_atoms, calibrate,
                   distinct_product, evaluate, fit_pool, merge, product, propose_edits, select)
from .synthetic_proof import sample


class StructuralProgramTests(unittest.TestCase):
    def test_distinct_product_matches_bruteforce(self):
        rng = np.random.default_rng(3)
        x = rng.random((31, 7, 4))
        for a in range(4):
            for b in range(4):
                reference = np.max(np.stack([x[:, j, a] * x[:, k, b] for j in range(7) for k in range(7) if j != k]), axis=0)
                np.testing.assert_allclose(distinct_product(x, a, b), reference, atol=0, rtol=0)

    def test_clone_does_not_fabricate_second_witness(self):
        batch = build_atoms(["same", "same"], ["query"], 2, lambda *_: 1.)
        self.assertEqual(batch.evidence_ids, ("same",))
        self.assertEqual(evaluate(atom(0), batch)[0], 1.)
        self.assertEqual(evaluate(product(0, 1), batch)[0], 0.)
        with self.assertRaisesRegex(ValueError, "duplicate evidence"):
            QueryAtoms(np.ones((1, 2, 2)), ("query",), ("same", "same"))

    def test_no_self_evidence_and_empty_context(self):
        with self.assertRaisesRegex(ValueError, "own evidence"):
            QueryAtoms(np.ones((1, 1, 2)), ("same",), ("same",))
        empty = QueryAtoms(np.empty((2, 0, 2)), ("x", "y"), ())
        np.testing.assert_array_equal(evaluate(merge(atom(0), product(0, 1)), empty), [0., 0.])

    def test_real_product_merge_and_deletion_edits(self):
        config = SearchConfig()
        self.assertIn(product(0, 1), propose_edits(atom(0), 3, config))
        self.assertIn(merge(atom(0), atom(2)), propose_edits(atom(0), 3, config))
        self.assertIn(atom(1), propose_edits(merge(atom(0), atom(1)), 3, config))
        self.assertTrue(all(t.nodes <= config.max_nodes and t.depth <= config.max_depth for t in propose_edits(merge(atom(0), atom(1)), 3, config)))

    def test_pool_and_calibration_cannot_consume_probe_labels(self):
        fit, y, _ = sample("conjunction", 100, 5, "fit")
        probe, probe_y, _ = sample("conjunction", 100, 6, "probe")
        pool = fit_pool(fit, y, atom(3), SearchConfig(rounds=2, max_candidates=70))
        before = pool.sha256
        select(pool, probe, probe_y)
        select(pool, probe, 1 - probe_y)
        self.assertEqual(pool.sha256, before)
        again = fit_pool(fit, y, atom(3), SearchConfig(rounds=2, max_candidates=70))
        self.assertEqual(again.sha256, before)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            select(pool, fit, y)

    def test_conjunction_recovers_out_of_sample_amid_noise(self):
        fit, y, _ = sample("conjunction", 600, 41, "fit")
        probe, probe_y, _ = sample("conjunction", 600, 141, "probe")
        pool = fit_pool(fit, y, atom(3), SearchConfig())
        selected, _ = select(pool, probe, probe_y)
        heldout, _, probability = sample("conjunction", 1000, 905, "heldout")
        recovered_truth = evaluate(selected.tree, heldout)
        np.testing.assert_array_equal(recovered_truth, (probability > .5).astype(float))
        self.assertEqual(selected.tree, product(0, 1))
        self.assertLessEqual(len(pool.candidates), pool.config.max_candidates)

    def test_redundant_branch_deleted_without_changing_function(self):
        fit, y, _ = sample("redundancy", 300, 42, "fit")
        probe, probe_y, _ = sample("redundancy", 300, 142, "probe")
        initial = merge(atom(0), atom(1))
        selected, _ = select(fit_pool(fit, y, initial, SearchConfig(rounds=2)), probe, probe_y)
        self.assertEqual(selected.tree.nodes, 1)
        np.testing.assert_array_equal(evaluate(initial, probe), evaluate(selected.tree, probe))

    def test_program_serialization_and_candidate_permutation(self):
        batch, y, _ = sample("conjunction", 40, 11, "fit")
        fitted = calibrate(product(0, 1), batch, y)
        replay = Tree.from_dict(fitted.tree.to_dict())
        np.testing.assert_array_equal(evaluate(replay, batch), evaluate(fitted.tree, batch))
        perm = np.random.default_rng(7).permutation(40)
        other = QueryAtoms(batch.values[perm], tuple(batch.candidate_ids[i] for i in perm), batch.evidence_ids)
        np.testing.assert_array_equal(fitted.logits(other), fitted.logits(batch)[perm])


if __name__ == "__main__":
    unittest.main()
