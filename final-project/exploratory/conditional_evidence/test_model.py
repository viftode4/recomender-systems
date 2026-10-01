"""Synthetic invariance, exclusion, and differentiability checks."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from .model import EvidenceBank, GraphReader, SummaryReader, combine_batches, scramble_history


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(902)
    values = rng.integers(0, 6, size=(24, 15), dtype=np.uint8)
    values[rng.random(values.shape) < .35] = 0
    values[:, 0] = 0
    context = np.zeros(15, dtype=np.uint8)
    context[1:8] = [1, 2, 3, 4, 5, 1, 2]
    return values, context


class ModelTests(unittest.TestCase):
    def test_excluded_row_cannot_change_features_or_retrieval(self) -> None:
        values, context = fixture()
        original = EvidenceBank(values).build_query(context, 0, [8, 9, 10])
        values[0, 1:] = 5 - values[0, 1:]
        altered = EvidenceBank(values).build_query(context, 0, [8, 9, 10])
        self.assertEqual(original.donor_rows, altered.donor_rows)
        for name in ("x", "edge_index", "rating", "features9"):
            torch.testing.assert_close(getattr(original, name), getattr(altered, name), rtol=0, atol=0)
        self.assertTrue(all(0 not in donors for donors in original.donor_rows))

    def test_fold_rows_disappear_from_population_and_selection(self) -> None:
        values, context = fixture()
        allowed = np.arange(values.shape[0]) % 3 != 0
        bank = EvidenceBank(values, allowed_rows=allowed)
        original = bank.build_query(context, 0, [8, 9])
        values[~allowed, 1:] = 5 - values[~allowed, 1:]
        altered = EvidenceBank(values, allowed_rows=allowed).build_query(context, 0, [8, 9])
        self.assertEqual(original.donor_rows, altered.donor_rows)
        self.assertTrue(all(allowed[list(rows)].all() for rows in original.donor_rows))
        torch.testing.assert_close(original.x, altered.x, rtol=0, atol=0)
        torch.testing.assert_close(original.features9, altered.features9, rtol=0, atol=0)

    def test_scramble_preserves_every_rating_margin_and_is_bounded(self) -> None:
        rng = np.random.default_rng(191)
        panel = rng.integers(0, 6, (16, 30), dtype=np.uint8)
        panel[rng.random(panel.shape) < .7] = 0
        result, stats = scramble_history(panel, 72)
        repeated, repeated_stats = scramble_history(panel, 72)
        np.testing.assert_array_equal(result, repeated)
        self.assertEqual(stats, repeated_stats)
        self.assertGreater(stats["successes"], 0)
        self.assertLessEqual(stats["attempts"], 20 * np.count_nonzero(panel))
        for rating in range(1, 6):
            np.testing.assert_array_equal((panel == rating).sum(0), (result == rating).sum(0))
            np.testing.assert_array_equal((panel == rating).sum(1), (result == rating).sum(1))
        immutable = np.full((3, 5), 2, dtype=np.uint8)
        unchanged, stats = scramble_history(immutable, 7)
        np.testing.assert_array_equal(immutable, unchanged)
        self.assertEqual(stats["successes"], 0)

    def test_candidate_chunking_and_order_do_not_change_graph(self) -> None:
        values, context = fixture()
        bank = EvidenceBank(values)
        for variant in ("raw", "scrambled", "summary"):
            whole = bank.build_query(context, 0, [8, 9, 10], variant, 91)
            split = combine_batches([bank.build_query(context, 0, [item], variant, 91)
                                     for item in (8, 9, 10)])
            for name in ("x", "edge_index", "rating", "features9", "query_idx", "donor_idx"):
                torch.testing.assert_close(getattr(whole, name), getattr(split, name), rtol=0, atol=0)
            reader = GraphReader(seed=4)
            with torch.no_grad():
                reverse = bank.build_query(context, 0, [10, 9, 8], variant, 91)
                torch.testing.assert_close(reader(whole), reader(reverse).flip(0), rtol=1e-6, atol=1e-7)

    def test_scramble_keeps_query_candidate_edges_and_initial_features(self) -> None:
        values, context = fixture()
        bank = EvidenceBank(values)
        original = bank.build_query(context, 0, [8, 9, 10])
        scrambled = bank.build_query(context, 0, [8, 9, 10], "scrambled", 18)
        self.assertEqual(original.donor_rows, scrambled.donor_rows)
        torch.testing.assert_close(original.x, scrambled.x, rtol=0, atol=0)
        def boundary(batch):
            nodes = torch.cat((batch.query_idx, batch.candidate_idx))
            selected = torch.isin(batch.edge_index[0], nodes) | torch.isin(batch.edge_index[1], nodes)
            return sorted(zip(batch.edge_index[0, selected].tolist(),
                              batch.edge_index[1, selected].tolist(), batch.rating[selected].tolist()))
        self.assertEqual(boundary(original), boundary(scrambled))

    def test_graph_node_permutation_invariance(self) -> None:
        values, context = fixture()
        batch = EvidenceBank(values).build_query(context, 0, [8, 9])
        permutation = torch.randperm(batch.x.shape[0], generator=torch.Generator().manual_seed(81))
        inverse = torch.argsort(permutation)
        changed = replace(batch, x=batch.x[permutation], edge_index=inverse[batch.edge_index],
                          query_idx=inverse[batch.query_idx], candidate_idx=inverse[batch.candidate_idx],
                          donor_idx=inverse[batch.donor_idx])
        reader = GraphReader(seed=9)
        torch.testing.assert_close(reader(batch), reader(changed), rtol=2e-6, atol=2e-7)

    def test_context_only_graph_has_no_candidate_edges_and_keeps_donor_order(self) -> None:
        values, context = fixture()
        bank = EvidenceBank(values)
        batch = bank.build_context_query(context, 0, [9, 2, 17])
        self.assertEqual(batch.donor_rows, ((9, 2, 17),))
        self.assertFalse(torch.isin(batch.edge_index, batch.candidate_idx).any())
        self.assertEqual(float(batch.x[batch.candidate_idx[0], 4]), 0)
        self.assertEqual(GraphReader(seed=3).node_embeddings(batch).shape, (12, 16))
        with self.assertRaises(ValueError):
            bank.build_context_query(context, 0, [0])

    def test_extra_donors_are_deduplicated_and_validate_boundary(self) -> None:
        values, context = fixture()
        bank = EvidenceBank(values)
        base = bank.build_query(context, 0, [8])
        extra = next(row for row in range(1, 24) if row not in base.donor_rows[0])
        changed = bank.build_query(context, 0, [8], extra_donor={8: [extra, extra, base.donor_rows[0][0]]})
        self.assertEqual(changed.donor_rows[0], base.donor_rows[0] + (extra,))
        with self.assertRaises(ValueError):
            bank.build_query(context, 0, [8], extra_donor={8: [0]})

    def test_empty_context_bank_and_candidate_batch_are_finite(self) -> None:
        bank = EvidenceBank(np.zeros((1, 5), dtype=np.uint8))
        batch = bank.build_query(np.zeros(5, dtype=np.uint8), 0, [1, 2])
        self.assertEqual(batch.donor_rows, ((), ()))
        reader = GraphReader(seed=2)
        self.assertTrue(torch.isfinite(reader(batch)).all())
        empty = bank.build_query(np.zeros(5, dtype=np.uint8), 0, [])
        self.assertEqual(tuple(reader(empty).shape), (0,))
        self.assertEqual(tuple(SummaryReader()(empty).shape), (0,))

    def test_forward_backward_seed_and_parameter_count(self) -> None:
        values, context = fixture()
        batch = EvidenceBank(values).build_query(context, 0, [8, 9, 10])
        torch.manual_seed(654)
        state = torch.get_rng_state().clone()
        first, second = GraphReader(seed=44), GraphReader(seed=44)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        torch.testing.assert_close(first(batch), second(batch), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in first.parameters()), 5553)
        self.assertEqual(sum(p.numel() for p in SummaryReader().parameters()), 177)
        loss = -torch.log_softmax(first(batch), dim=0)[0]
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in first.parameters()))

    def test_summary_can_omit_unused_graph_with_identical_features_and_gradients(self) -> None:
        values, context = fixture()
        bank = EvidenceBank(values)
        full = bank.build_query(context, 0, [8, 9, 10], "summary")
        small = bank.build_query(context, 0, [8, 9, 10], "summary", materialize_graph=False)
        self.assertEqual(small.x.shape[0], 0)
        self.assertEqual(small.metadata["materialized_node_count"], 0)
        self.assertEqual(full.metadata["graphs"], small.metadata["graphs"])
        first, second = SummaryReader(seed=71), SummaryReader(seed=71)
        torch.testing.assert_close(first(full), second(small), rtol=0, atol=0)
        first(full).square().sum().backward()
        second(small).square().sum().backward()
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            GraphReader()(small)
        with self.assertRaises(ValueError):
            bank.build_query(context, 0, [8], "raw", materialize_graph=False)

    def test_padding_seen_and_malformed_ratings_rejected(self) -> None:
        values, context = fixture()
        bank = EvidenceBank(values)
        for ids in ([0], [1], [8, 8], [15]):
            with self.assertRaises(ValueError):
                bank.build_query(context, 0, ids)
        bad = values.astype(float)
        bad[1, 2] = 2.5
        with self.assertRaises(ValueError):
            EvidenceBank(bad)


if __name__ == "__main__":
    unittest.main()
