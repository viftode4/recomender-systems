"""Independent synthetic checks of retrieval information and loss boundaries.

These tests open no datasets and do not train the recommendation model.
"""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from exploratory.conditional_evidence import retrieval as r
from exploratory.conditional_evidence.model import EvidenceBank, GraphReader


def fixture():
    values = np.zeros((27, 10), dtype=np.uint8)
    for row in range(len(values)):
        values[row, 1] = 1 + row % 5
        values[row, 2] = 1 + (row // 5) % 5
        for item in range(3, 10):
            if (row * 7 + item * 3) % 11 < 6:
                values[row, item] = 1 + (row + item) % 5
    episodes = []
    for row in range(len(values)):
        context = values[row].copy()
        context[3:] = 0
        episodes.append(r.Episode(row, context, np.flatnonzero(values[row, 3:]) + 3))
    return values, episodes


def open_gate():
    return r.meta_gate({2026: .4}, {'scrambled': {2026: .2}, 'summary': {2026: .3}},
                       {'reference': {2026: .35}}, stage='meta_fit')


class IndependentRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_loss_accounts_for_competing_candidates_and_not_rating_dislike(self):
        mask = np.array([False, False, True, True, True])
        before = np.array([np.nan, np.nan, math.log(2), math.log(3), math.log(5)])
        after = before.copy()
        after[4] = math.log(15)  # Target logits do not change; another item does.
        expected = math.log(10) - math.log(2)
        self.assertAlmostEqual(r.full_catalog_loss(before, [2], mask), expected, places=14)
        self.assertAlmostEqual(r.loss_reduction(before, after, [2], mask), -math.log(2), places=14)
        self.assertAlmostEqual(r.full_catalog_loss(before + 1000, [2, 3], mask),
                               math.log(10) - math.log(6) / 2, places=12)
        # Targets are record membership, with no positive-rating threshold input.
        with self.assertRaises(ValueError):
            r.full_catalog_loss(before, [1], mask)
        with self.assertRaises(ValueError):
            r.full_catalog_loss(before, [2, 2], mask)

    def test_eight_actions_match_literal_cosine_with_exclusions(self):
        matrix, episodes = fixture()
        context = episodes[0].context
        allowed = np.ones(len(matrix), dtype=bool)
        allowed[[2, 7, 11]] = False
        query = set(np.flatnonzero(context))
        literal = []
        for row in range(len(matrix)):
            if row == 0 or not allowed[row]:
                continue
            history = set(np.flatnonzero(matrix[row]))
            similarity = len(query & history) / math.sqrt(len(query) * len(history))
            literal.append((-similarity, row))
        ordered = [row for _, row in sorted(literal)]
        costs = r.Costs()
        actual = r.donor_actions(matrix, context, exclude_row=0, allowed_rows=allowed, costs=costs)
        self.assertEqual(actual, tuple(ordered[8:16]))
        self.assertEqual(len(set(actual)), 8)
        self.assertFalse(set(actual) & set(ordered[:8]))
        self.assertEqual(costs.donor_cosine_rows, len(ordered))

    def test_selector_graph_ignores_noncontext_rating_categories_and_labels(self):
        matrix, episodes = fixture()
        context = episodes[0].context
        actions = r.donor_actions(matrix, context, exclude_row=0)
        changed = matrix.copy()
        changed[:, 3:] = np.where(matrix[:, 3:] > 0, 6 - matrix[:, 3:], 0)
        reader = GraphReader(seed=31)
        first_bank, second_bank = EvidenceBank(matrix), EvidenceBank(changed)
        first = r.context_action_features(reader, first_bank, context, 0, actions, r.Costs())
        second = r.context_action_features(reader, second_bank, context, 0, actions, r.Costs())
        np.testing.assert_array_equal(first, second)
        graph = first_bank.build_context_query(context, 0, actions)
        candidate_node = int(graph.candidate_idx[0])
        self.assertFalse(torch.any(graph.edge_index == candidate_node).item())
        self.assertEqual(graph.candidate_ids.tolist(), [-1])
        expected_edges = 2 * (np.count_nonzero(context) + sum(
            np.count_nonzero(matrix[row][context > 0]) for row in actions))
        self.assertEqual(graph.edge_index.shape[1], expected_edges)
        self.assertEqual(graph.x.shape[0], np.count_nonzero(context) + 2 + len(actions))
        # The same context with different hidden labels must have identical
        # action features, even though its supervised utility can differ.
        one = r.collect_query(reader, first_bank, r.Episode(0, context, np.array([3])), r.Costs())
        two = r.collect_query(reader, first_bank, r.Episode(0, context, np.array([4])), r.Costs())
        np.testing.assert_array_equal(one.features, two.features)
        self.assertEqual(one.actions, two.actions)
        for observation, target in ((one, 3), (two, 4)):
            base = r.score_catalog(reader, first_bank, context, 0, None, r.Costs())
            candidates = np.flatnonzero(r.eligible_items(context))
            literal_base = math.log(sum(math.exp(base[i]) for i in candidates)) - base[target]
            self.assertAlmostEqual(observation.base_loss, literal_base, places=13)
            literal_reductions = []
            for action in observation.actions:
                scores = r.score_catalog(reader, first_bank, context, 0, action, r.Costs())
                loss = math.log(sum(math.exp(scores[i]) for i in candidates)) - scores[target]
                literal_reductions.append(literal_base - loss)
            np.testing.assert_allclose(observation.reductions, literal_reductions, atol=1e-13, rtol=0)

    def test_addition_is_deduplicated_per_candidate_and_counts_actual_work(self):
        matrix, episodes = fixture()
        context, bank = episodes[0].context, EvidenceBank(matrix)
        candidates = np.flatnonzero(r.eligible_items(context))
        action = r.donor_actions(matrix, context, exclude_row=0)[0]
        initial = bank.build_query(context, 0, candidates)
        additions = r.deduplicated_additions(candidates, initial.donor_rows, action)
        augmented = bank.build_query(context, 0, candidates, extra_donor=additions)
        present = 0
        for candidate, before, after in zip(candidates, initial.donor_rows, augmented.donor_rows):
            expected = before if action in before else before + (action,)
            self.assertEqual(after, expected)
            self.assertEqual(len(after), len(set(after)))
            self.assertEqual(additions[int(candidate)], () if action in before else (action,))
            present += action in before
        # Ensure this example covers both a no-op and a real insertion.
        self.assertGreater(present, 0)
        self.assertLess(present, len(candidates))
        reader, costs = GraphReader(seed=5), r.Costs()
        scores = r.score_catalog(reader, bank, context, 0, action, costs, chunk_size=2)
        self.assertEqual(costs.donor_already_present, present)
        self.assertEqual(costs.donor_insertions, len(candidates) - present)
        self.assertEqual(costs.scored_candidates, len(candidates))
        self.assertEqual(costs.candidate_graphs, 2 * len(candidates))
        packed = r.score_catalog(reader, bank, context, 0, action, r.Costs(), chunk_size=100)
        np.testing.assert_allclose(scores[candidates], packed[candidates], atol=2e-7, rtol=0)

    def test_entire_fold_is_absent_from_learner_and_bank_with_remapped_episodes(self):
        matrix, episodes = fixture()
        seed = 419
        assignments = np.empty(len(matrix), dtype=int)
        assignments[np.random.default_rng(seed).permutation(len(matrix))] = np.arange(len(matrix)) % 3
        seen, current = [], {}

        def fit_reader(request):
            expected = np.flatnonzero(assignments != request.fold)
            np.testing.assert_array_equal(request.original_user_rows, expected)
            np.testing.assert_array_equal(request.categories, matrix[expected])
            self.assertFalse(request.categories.flags.writeable)
            self.assertFalse(request.original_user_rows.flags.writeable)
            self.assertEqual(len(request.episodes), len(expected))
            for episode in request.episodes:
                original = int(expected[episode.user_row])
                self.assertNotEqual(assignments[original], request.fold)
                np.testing.assert_array_equal(episode.context, episodes[original].context)
                np.testing.assert_array_equal(episode.targets, episodes[original].targets)
            current['fold'] = request.fold
            current['expected'] = expected
            current['held'] = iter(e for e in episodes if assignments[e.user_row] == request.fold)
            seen.append(request.fold)
            return r.ReaderFit(GraphReader(seed=request.selected_config['seed']), request.epochs)

        def collect(reader, bank, episode, costs):
            original = next(current['held'])
            self.assertEqual(episode.user_row, -1)
            np.testing.assert_array_equal(bank.categories, matrix[current['expected']])
            np.testing.assert_array_equal(episode.context, original.context)
            np.testing.assert_array_equal(episode.targets, original.targets)
            actions = r.donor_actions(bank.categories, episode.context, exclude_row=-1)
            features = r.context_action_features(reader, bank, episode.context, -1, actions, costs)
            return r.QueryObservations(actions, features, np.zeros(len(actions)), 0., np.zeros(len(actions)))

        result = r.crossfit_labels(matrix, episodes, gate=open_gate(), selected_config={'seed': 71},
            selected_epochs=2, seed=seed, fit_reader=fit_reader, query_collector=collect)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(result.features.shape, (len(matrix) * 8, 35))
        self.assertEqual(result.costs.fold_reader_fits, 3)
        self.assertEqual(result.costs.declared_fold_reader_epochs, 6)

    def test_crossfit_rejects_full_population_bank_and_epoch_mismatch(self):
        matrix, episodes = fixture()
        kwargs = dict(gate=open_gate(), selected_config={}, selected_epochs=2, seed=11)
        with self.assertRaisesRegex(ValueError, 'other-fold TRAIN'):
            r.crossfit_labels(matrix, episodes, **kwargs,
                fit_reader=lambda request: r.ReaderFit(GraphReader(seed=0), 2),
                bank_factory=lambda _: EvidenceBank(matrix))
        with self.assertRaisesRegex(ValueError, 'epoch budget'):
            r.crossfit_labels(matrix, episodes, **kwargs,
                fit_reader=lambda request: r.ReaderFit(GraphReader(seed=0), 1))
        closed = r.meta_gate({2026: .35}, {'scrambled': {2026: .2}, 'summary': {2026: .3}},
                            {'reference': {2026: .35}}, stage='meta_fit')
        never = mock.Mock(side_effect=AssertionError('closed gate must not fit'))
        result = r.fit_extension(matrix, episodes, gate=closed, selected_config={},
                                 selected_epochs=2, seed=11, fit_reader=never)
        self.assertIsNone(result.utility)
        never.assert_not_called()

    def test_deployment_has_no_fitting_or_hidden_target_loss(self):
        matrix, episodes = fixture()
        reader, bank = GraphReader(seed=44), EvidenceBank(matrix)
        original = {name: value.clone() for name, value in reader.state_dict().items()}
        utility = mock.Mock()
        utility.predict.side_effect = lambda x, costs: -np.ones(len(x))
        with mock.patch.object(torch.optim, 'Adam', side_effect=AssertionError('no optimizer')), \
             mock.patch.object(r, 'full_catalog_loss', side_effect=AssertionError('no labels')), \
             mock.patch.object(r, 'fit_utility', side_effect=AssertionError('no utility fitting')):
            result = r.deploy(reader, bank, episodes[0].context, 0, utility)
        self.assertIsNone(result['action'])
        self.assertEqual(result['costs'].context_graph_passes, 1)
        self.assertEqual(result['costs'].full_catalog_losses, 0)
        self.assertEqual(result['costs'].utility_training_steps, 0)
        self.assertEqual(result['costs'].scored_candidates, 7)
        for name, value in reader.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'oracle'):
            r.deploy(reader, bank, episodes[0].context, 0, utility, policy='oracle')

    def test_actual_fold_training_resumes_exact_model_optimizer_and_losses(self):
        # Run the real callback against a tiny supplied bank, forbid any global
        # source loader, and interrupt after the first durable epoch checkpoint.
        from exploratory.conditional_evidence import workflow
        matrix, _ = fixture()
        request = r.FoldTrainingRequest(0, matrix[:5], np.arange(5), (),
            {'seed': 29, 'learning_rate': .001}, 2, 611)
        original_save = workflow.run.save_checkpoint

        def interrupt_after_save(path, model, optimizer, epoch, guard, history):
            original_save(path, model, optimizer, epoch, guard, history)
            if epoch == 1:
                raise RuntimeError('synthetic interruption after durable epoch one')

        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(workflow.run, 'load_inputs', side_effect=AssertionError('no global input')), \
             mock.patch.object(workflow.data, 'load_inputs', side_effect=AssertionError('no global data')):
            direct_dir, resumed_dir = Path(temporary) / 'direct', Path(temporary) / 'resumed'
            direct = workflow.fit_fold(request, direct_dir, {})
            with mock.patch.object(workflow.run, 'save_checkpoint', side_effect=interrupt_after_save):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    workflow.fit_fold(request, resumed_dir, {})
            resumed = workflow.fit_fold(request, resumed_dir, {})
            first = torch.load(direct_dir / 'latest.pt', weights_only=True)
            second = torch.load(resumed_dir / 'latest.pt', weights_only=True)
            self.assertEqual(direct.completed_epochs, resumed.completed_epochs)
            self.assertEqual(direct.reader.config['seed'], request.selected_config['seed'])
            self.assertEqual(first['guard'], second['guard'])
            for key in ('model', 'optimizer'):
                def assert_same(a, b):
                    if isinstance(a, torch.Tensor):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                    elif isinstance(a, dict):
                        self.assertEqual(a.keys(), b.keys())
                        for subkey in a:
                            assert_same(a[subkey], b[subkey])
                    elif isinstance(a, (list, tuple)):
                        self.assertEqual(len(a), len(b))
                        for aa, bb in zip(a, b):
                            assert_same(aa, bb)
                    else:
                        self.assertEqual(a, b)
                assert_same(first[key], second[key])
            for a, b in zip(first['history']['epochs'], second['history']['epochs']):
                self.assertEqual(a['epoch'], b['epoch'])
                self.assertEqual(a['queries'], b['queries'])
                self.assertEqual(a['mean_query_loss'], b['mean_query_loss'])


if __name__ == '__main__':
    unittest.main()
