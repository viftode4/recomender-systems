"""Independent finite examples; no real dataset, development or TEST access."""
from __future__ import annotations

import math
import copy
import random
import contextlib
import io
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from exploratory.conditional_evidence import data
from exploratory.conditional_evidence import run_experiment as runner
from exploratory.conditional_evidence import model as model_api, evaluation as evaluation_api
from exploratory.conditional_evidence.model import EvidenceBank, GraphReader, combine_batches, scramble_history


class IndependentLossTests(unittest.TestCase):
    def test_sampled_partition_and_analytic_gradient(self):
        # Two recorded targets and one of five uniformly sampled alternatives.
        values = [0.2, -0.4, 0.8]
        scores = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        actual = data.sampled_multinomial_loss(scores, [True, True, False], 5)
        masses = [math.exp(values[0]), math.exp(values[1]), 5 * math.exp(values[2])]
        partition = sum(masses)
        expected = math.log(partition) - (values[0] + values[1]) / 2
        self.assertAlmostEqual(actual.item(), expected, places=14)
        actual.backward()
        expected_gradient = [masses[0] / partition - .5,
                             masses[1] / partition - .5, masses[2] / partition]
        np.testing.assert_allclose(scores.grad.numpy(), expected_gradient, atol=2e-15, rtol=0)

    def test_exhaustive_and_no_alternative_limits(self):
        values = [-1., 2., .5, -.5]
        actual = data.sampled_multinomial_loss(
            torch.tensor(values, dtype=torch.float64), [True, False, True, False], 2)
        expected = math.log(sum(math.exp(x) for x in values)) - (values[0] + values[2]) / 2
        self.assertAlmostEqual(actual.item(), expected, places=14)
        only_targets = data.sampled_multinomial_loss(
            torch.zeros(3, dtype=torch.float64), [True] * 3, 0)
        self.assertAlmostEqual(only_targets.item(), math.log(3), places=14)

    def test_equal_episode_weight_not_equal_record_weight(self):
        # Row 1 has one positive; row 2 has two. Each contributes one half.
        scores = torch.tensor([[0., 1., 2.], [1., -1., 0.]], dtype=torch.float64)
        actual = data.sampled_multinomial_loss(
            scores, [[True, False, False], [True, True, False]], [2, 1])
        first = math.log(1 + math.e + math.exp(2))
        second = math.log(math.e + math.exp(-1) + 1)
        self.assertAlmostEqual(actual.item(), (first + second) / 2, places=14)
        self.assertNotAlmostEqual(actual.item(), (first + 2 * second) / 3, places=6)

    def test_partition_is_unbiased_but_log_loss_is_not(self):
        # Enumerate the entire uniform one-of-two sampling distribution.
        # Positive mass=2, absent masses=1 and 9. Full partition=12.
        losses = [data.sampled_multinomial_loss(
            torch.tensor([math.log(2), math.log(weight)], dtype=torch.float64),
            [True, False], 2).item() for weight in (1, 9)]
        self.assertEqual(((2 + 2 * 1) + (2 + 2 * 9)) / 2, 12)
        self.assertAlmostEqual(sum(losses) / 2, math.log(20) / 2, places=14)
        self.assertLess(sum(losses) / 2, math.log(12) - math.log(2))


class IndependentDataBoundaryTests(unittest.TestCase):
    def test_other_cohort_item_text_is_not_interpreted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pairs.tsv'
            path.write_text('user_id\titem_id\n'
                            'dev\tnot-an-item\textra-invalid-field\n'
                            'meta\titem2\n'
                            'dev-with-no-item\n')
            result = data.read_user_pairs(path, {'meta'}, real_items={'item1', 'item2'},
                                          train_pairs={('meta', 'item1')})
            self.assertEqual(result, [('meta', 'item2')])
            path.write_text('user_id\titem_id\nmeta\tnot-an-item\textra-invalid-field\n')
            with self.assertRaises(ValueError):
                data.read_user_pairs(path, {'meta'}, real_items={'item1', 'item2'}, train_pairs=set())

    def test_episode_partition_has_no_hidden_record_in_alternatives(self):
        matrix = np.array([[0, 1, 2, 3, 4, 5, 0, 0],
                           [0, 0, 0, 0, 0, 0, 5, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        for fraction in (.8, .9):
            episodes = list(data.episodes(matrix, epoch=3, retain_fraction=fraction, seed=21))
            self.assertEqual(len(episodes), 1)  # Singleton and empty histories omitted.
            row = episodes[0]
            context = set(np.flatnonzero(row['context']))
            targets = set(row['candidate_ids'][row['target_mask']])
            alternatives = set(row['candidate_ids'][~row['target_mask']])
            self.assertEqual(len(context), 4)
            self.assertEqual(context | targets, {1, 2, 3, 4, 5})
            self.assertFalse(context & targets)
            self.assertEqual(alternatives, {6, 7})
            self.assertEqual(row['unobserved_population_count'], 2)
            self.assertEqual(row['target_count'], 1)
            np.testing.assert_array_equal(row['context'][list(context)], matrix[0, list(context)])


class IndependentEvidenceTests(unittest.TestCase):
    @staticmethod
    def example():
        # Query row contains two deliberately hidden ratings; neither may enter
        # the graph through edges, popularity counts, donor retrieval or norms.
        bank = np.array([[0, 1, 2, 4, 5, 0],
                         [0, 1, 0, 5, 0, 0],
                         [0, 0, 2, 4, 0, 0],
                         [0, 1, 2, 0, 0, 0]], dtype=np.uint8)
        context = np.array([0, 1, 2, 0, 0, 0], dtype=np.uint8)
        return bank, context

    def test_literal_panel_nodes_edges_and_same_donor_summary(self):
        matrix, context = self.example()
        batch = EvidenceBank(matrix).build_query(context, 0, np.array([3]), variant='raw')
        self.assertEqual(batch.donor_rows, ((3, 1, 2),))
        # Node order: history1, history2, query, candidate, donor3, donor1, donor2.
        expected = np.zeros((7, 5))
        expected[[0, 1], 2] = 1
        expected[2, 0] = expected[3, 3] = 1
        expected[[4, 5, 6], 1] = 1
        expected[:, 4] = math.log(3)
        np.testing.assert_allclose(batch.x.numpy(), expected, atol=3e-8, rtol=0)
        undirected = [(2, 0, 1), (2, 1, 2), (4, 0, 1), (4, 1, 2),
                      (5, 0, 1), (5, 3, 5), (6, 1, 2), (6, 3, 4)]
        actual_edges = sorted((int(s), int(t), int(r)) for (s, t), r in
                              zip(batch.edge_index.T, batch.rating))
        expected_edges = sorted(undirected + [(t, s, r) for s, t, r in undirected])
        self.assertEqual(actual_edges, expected_edges)
        # Cosine-squared weights are 1, 1/4, 1/4; only latter two support item3.
        expected9 = [math.log(3), math.log(3), 2 / 3, math.log(1.5), 1 / 3,
                     .25, math.log(3), math.log(3), 1]
        np.testing.assert_allclose(batch.features9.numpy()[0], expected9, atol=3e-8, rtol=0)
        summary = EvidenceBank(matrix).build_query(context, 0, np.array([3]), variant='summary')
        self.assertEqual(summary.donor_rows, batch.donor_rows)
        np.testing.assert_array_equal(summary.features9.numpy(), batch.features9.numpy())

    def test_hidden_query_row_cannot_change_any_evidence_but_donor_rating_can(self):
        matrix, context = self.example()
        first = EvidenceBank(matrix).build_query(context, 0, np.array([3, 4]))
        poisoned = matrix.copy()
        poisoned[0] = [0, 5, 5, 0, 0, 4]
        second = EvidenceBank(poisoned).build_query(context, 0, np.array([3, 4]))
        for field in ('x', 'edge_index', 'rating', 'query_idx', 'candidate_idx',
                      'donor_idx', 'donor_graph_idx', 'features9', 'candidate_ids'):
            torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)
        self.assertEqual(first.donor_rows, second.donor_rows)
        # Cold item4 has no permitted population evidence despite query's hidden5.
        self.assertEqual(first.x[first.candidate_idx[1], 4].item(), 0)
        changed_donor = matrix.copy()
        changed_donor[1, 3] = 1
        third = EvidenceBank(changed_donor).build_query(context, 0, np.array([3, 4]))
        self.assertFalse(torch.equal(first.rating, third.rating))

    def test_literal_cosine_retrieval_and_sixteen_donor_limit(self):
        matrix = np.zeros((22, 9), dtype=np.uint8)
        for row in range(22):
            for item in range(1, 9):
                if (row * 7 + item * 3) % 11 < 5:
                    matrix[row, item] = 1 + (row + item) % 5
        context = np.array([0, 5, 4, 3, 0, 0, 0, 0, 0], dtype=np.uint8)
        batch = EvidenceBank(matrix).build_query(context, 0, np.array([4, 5, 6, 7, 8]))
        scored = []
        for row in range(1, len(matrix)):
            overlap = sum(matrix[row, item] > 0 for item in (1, 2, 3))
            degree = sum(matrix[row, item] > 0 for item in range(1, 9))
            score = overlap / math.sqrt(3 * degree) if degree else 0
            scored.append((-score, row))
        order = [row for _, row in sorted(scored)]
        for graph, candidate in enumerate((4, 5, 6, 7, 8)):
            expected = order[:8] + [row for row in order[8:] if matrix[row, candidate]][:8]
            self.assertEqual(batch.donor_rows[graph], tuple(expected))
            self.assertLessEqual(len(expected), 16)
            self.assertEqual(len(expected), len(set(expected)))

    def test_scramble_preserves_every_typed_degree_and_fixed_edges(self):
        panel = np.array([[1, 0, 0, 2], [0, 1, 2, 0], [0, 2, 1, 0],
                          [2, 0, 0, 1], [1, 2, 0, 0]], dtype=np.uint8)
        scrambled, counts = scramble_history(panel, 41)
        for rating in range(1, 6):
            for axis in (0, 1):
                np.testing.assert_array_equal((panel == rating).sum(axis),
                                              (scrambled == rating).sum(axis))
        self.assertGreater(counts['successes'], 0)
        self.assertLessEqual(counts['attempts'], 20 * np.count_nonzero(panel))
        matrix = np.column_stack((np.zeros(len(panel), dtype=np.uint8), panel,
                                  np.array([5, 4, 0, 3, 0], dtype=np.uint8)))
        context = np.array([0, 1, 2, 3, 4, 0], dtype=np.uint8)
        bank = EvidenceBank(matrix)
        raw = bank.build_query(context, -1, np.array([5]), 'raw', 41)
        altered = bank.build_query(context, -1, np.array([5]), 'scrambled', 41)
        fixed = {int(raw.query_idx[0]), int(raw.candidate_idx[0])}
        def fixed_edges(batch):
            return sorted((int(s), int(t), int(r)) for (s, t), r in
                          zip(batch.edge_index.T, batch.rating) if int(s) in fixed or int(t) in fixed)
        self.assertEqual(fixed_edges(raw), fixed_edges(altered))
        torch.testing.assert_close(raw.x, altered.x, rtol=0, atol=0)
        self.assertEqual(raw.donor_rows, altered.donor_rows)


class IndependentGraphNumericsTests(unittest.TestCase):
    def test_three_typed_mean_layers_and_head_against_literal_loops(self):
        matrix, context = IndependentEvidenceTests.example()
        batch = EvidenceBank(matrix).build_query(context, 0, np.array([3, 4]))
        batch = replace(batch, x=batch.x.double())
        model = GraphReader(width=2, seed=9).double()
        # Fixed non-symmetric values exercise self, relation and bias paths.
        with torch.no_grad():
            for k, parameter in enumerate(model.parameters()):
                parameter.copy_(torch.arange(parameter.numel(), dtype=torch.float64)
                                .reshape(parameter.shape) * .007 + .013 * (k + 1))
        affine = lambda layer, h: h @ layer.weight.detach().numpy().T + (
            layer.bias.detach().numpy() if layer.bias is not None else 0)
        silu = lambda h: h / (1 + np.exp(-h))
        hidden = affine(model.input_projection, batch.x.numpy())
        edges = [(int(s), int(t), int(r)) for (s, t), r in zip(batch.edge_index.T, batch.rating)]
        for layer in model.layers:
            updated = []
            for node in range(len(hidden)):
                value = affine(layer.self_transform, hidden[node])
                for rating in range(1, 6):
                    neighbors = [source for source, target, category in edges
                                 if target == node and category == rating]
                    if neighbors:
                        messages = [affine(layer.relations[rating - 1], hidden[source]) for source in neighbors]
                        value = value + np.mean(messages, axis=0)
                updated.append(silu(value))
            hidden = np.asarray(updated)
        expected_encoding = []
        for graph in range(batch.num_graphs):
            donor_nodes = [int(node) for node, membership in zip(batch.donor_idx, batch.donor_graph_idx)
                           if int(membership) == graph]
            expected_encoding.append(np.concatenate((hidden[int(batch.query_idx[graph])],
                hidden[int(batch.candidate_idx[graph])], np.mean(hidden[donor_nodes], axis=0))))
        expected_encoding = np.asarray(expected_encoding)
        expected_scores = affine(model.head[2], silu(affine(model.head[0], expected_encoding))).ravel()
        np.testing.assert_allclose(model.node_embeddings(batch).detach().numpy(), hidden, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(model.encode(batch).detach().numpy(), expected_encoding, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(model(batch).detach().numpy(), expected_scores, rtol=2e-13, atol=2e-13)

    def test_candidate_order_and_independent_graph_batching(self):
        matrix, context = IndependentEvidenceTests.example()
        bank = EvidenceBank(matrix)
        model = GraphReader(width=3, seed=12)
        for variant in ('raw', 'scrambled'):
            together = bank.build_query(context, 0, np.array([3, 4, 5]), variant, 912)
            pieces = [bank.build_query(context, 0, np.array([candidate]), variant, 912)
                      for candidate in (3, 4, 5)]
            packed = combine_batches(pieces)
            torch.testing.assert_close(model(together), model(packed), rtol=0, atol=0)
            reverse = bank.build_query(context, 0, np.array([5, 4, 3]), variant, 912)
            torch.testing.assert_close(model(together), model(reverse).flip(0), rtol=1e-6, atol=1e-7)


class IndependentRunnerTests(unittest.TestCase):
    def test_meta_score_digest_uses_catalog_order_not_cohort_container_order(self):
        value = {'users': ['c', 'a', 'b'], 'meta_users': ['b', 'c']}
        self.assertEqual(runner.user_indices(value, 'meta_users'), [0, 2])
        value['meta_users'] = {'b', 'c'}
        self.assertEqual(runner.user_indices(value, 'meta_users'), [0, 2])

    def test_two_pass_gradient_equals_literal_full_denominator(self):
        x = torch.tensor([[1., 2.], [-2., .5], [.7, -1.], [0., 4.]], dtype=torch.float64)
        full = torch.nn.Linear(2, 1, dtype=torch.float64)
        with torch.no_grad():
            full.weight.copy_(torch.tensor([[.2, -.3]], dtype=torch.float64))
            full.bias.fill_(.1)
        chunked = copy.deepcopy(full)
        y = full(x).ravel()
        # Explicit partition with two targets and two of five absent alternatives.
        literal_loss = torch.log(y[[0, 2]].exp().sum() + 2.5 * y[[1, 3]].exp().sum()) - y[[0, 2]].mean()
        (literal_loss / 3).backward()
        measured = runner.two_pass_backward(chunked, lambda: iter((x[:1], x[1:3], x[3:])),
            lambda values: data.sampled_multinomial_loss(values, [True, False, True, False], 5), divisor=3)
        self.assertAlmostEqual(measured, literal_loss.item(), places=14)
        for expected, actual in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-15, rtol=0)

    def test_nondeterministic_second_pass_is_rejected(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        calls = [0]
        def chunks():
            calls[0] += 1
            yield torch.tensor([[float(calls[0])], [2.]])
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            runner.two_pass_backward(model, chunks, lambda scores: scores.square().mean())

    def test_checkpoint_restores_adam_and_all_three_rng_streams(self):
        runner.numerical_setup(17)
        model = torch.nn.Linear(2, 1, dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        guard = {'source_sha256': {'synthetic.py': 'a' * 64},
                 'input_sha256': {'synthetic-input': 'b' * 64}, 'config': {'seed': 17, 'epochs': 2}}
        def step(current, adam):
            x = torch.randn(3, 2, dtype=torch.float64)
            target = torch.tensor(np.random.normal(size=3) + random.random(), dtype=torch.float64)
            adam.zero_grad()
            ((current(x).ravel() - target) ** 2).mean().backward()
            adam.step()
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'checkpoint.pt'
            runner.save_checkpoint(path, model, optimizer, 1, guard, {'checkpoints': []})
            step(model, optimizer)
            expected_model = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_random = (torch.rand(2), np.random.random(2), random.random())
            restored = torch.nn.Linear(2, 1, dtype=torch.float64)
            resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=.5)
            epoch, history = runner.restore_checkpoint(path, restored, resumed_optimizer, guard)
            self.assertEqual(epoch, 1)
            self.assertEqual(history, {'checkpoints': []})
            step(restored, resumed_optimizer)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected_model[key], atol=0, rtol=0)
            for key, state in resumed_optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, expected_optimizer['state'][key][name], atol=0, rtol=0)
            self.assertEqual(resumed_optimizer.state_dict()['param_groups'], expected_optimizer['param_groups'])
            torch.testing.assert_close(torch.rand(2), expected_random[0], atol=0, rtol=0)
            np.testing.assert_array_equal(np.random.random(2), expected_random[1])
            self.assertEqual(random.random(), expected_random[2])
            for changed_key in ('source_sha256', 'input_sha256', 'config'):
                altered = copy.deepcopy(guard)
                altered[changed_key] = {'changed': True}
                with self.assertRaises(ValueError):
                    runner.restore_checkpoint(path, restored, resumed_optimizer, altered)

    def test_tiny_full_training_interrupt_resume_matches_uninterrupted(self):
        # This exercises the real episode generator, sampled objective, Adam,
        # checkpoint selection and resume loop. All labels below are synthetic.
        matrix = np.zeros((4, 16), dtype=np.uint8)
        matrix[0, [1, 2, 3]] = [5, 1, 2]
        matrix[1, [2, 4, 5]] = [4, 3, 1]
        matrix[2, [1, 4, 6]] = [2, 5, 3]
        matrix[3, [3, 5, 7]] = [1, 2, 4]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'synthetic-source.txt'
            source.write_text('synthetic fixed inputs')
            inputs = {'categories': matrix, 'users': ['u0', 'u1', 'u2', 'u3'],
                'items': ['[PAD]'] + [str(item) for item in range(1, 16)],
                'meta_users': {'u0', 'u1'}, 'meta_truth': {'u0': {'14'}, 'u1': {'15'}},
                'input_hashes': {'fixture': runner.digest(source)}, 'source_paths': {'fixture': source}}
            with mock.patch.object(runner, 'CHECKPOINT_INTERVAL', 1), \
                    mock.patch.object(runner, 'BASE_EPOCHS', 2), \
                    mock.patch.object(runner, 'EXTENDED_EPOCHS', 4), \
                    contextlib.redirect_stdout(io.StringIO()):
                spec = runner.config_spec(99, 'summary', .001)
                full = runner.train_configuration(root / 'full', inputs, model_api, data,
                    evaluation_api, spec, 2, {}, resume=False)
                runner.train_configuration(root / 'resumed', inputs, model_api, data,
                    evaluation_api, spec, 1, {}, resume=False)
                resumed = runner.train_configuration(root / 'resumed', inputs, model_api, data,
                    evaluation_api, spec, 2, {}, resume=True)
            self.assertEqual(full['selected']['epoch'], resumed['selected']['epoch'])
            self.assertEqual([r['meta_ndcg@10'] for r in full['history']['checkpoints']],
                             [r['meta_ndcg@10'] for r in resumed['history']['checkpoints']])
            self.assertEqual([r['mean_query_loss'] for r in full['history']['epochs']],
                             [r['mean_query_loss'] for r in resumed['history']['epochs']])
            self.assertEqual([r['meta_score_array_sha256'] for r in full['history']['checkpoints']],
                             [r['meta_score_array_sha256'] for r in resumed['history']['checkpoints']])
            left = torch.load(root / 'full/latest.pt', weights_only=True)
            right = torch.load(root / 'resumed/latest.pt', weights_only=True)
            for key in left['model']:
                torch.testing.assert_close(left['model'][key], right['model'][key], atol=0, rtol=0)
            for key, state in left['optimizer']['state'].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(tensor, right['optimizer']['state'][key][name], atol=0, rtol=0)

    def test_complete_seal_binds_global_artifacts_before_development_access(self):
        # Synthetic hash-chain fixture. Prediction payloads are opaque bytes here:
        # this test targets the release verifier, not model/evaluation arithmetic.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'synthetic-source.txt'
            source.write_text('synthetic source')
            source_hashes = {str(source): runner.digest(source)}
            plan = {'source_sha256': source_hashes, 'runtime': runner.runtime(), 'inputs': {}}
            runner.atomic_json(root / 'plan.json', plan)
            runner.atomic_json(root / 'training-summary.json', {'status': 'complete'})
            runner.atomic_json(root / 'meta-screen.json', {'synthetic': True})
            manifests = {}
            names = ['selection.json', 'ids.json', 'reference-metadata.json'] + [
                f'selected-scores/{role}.npy' for role in (*runner.ARMS, 'EASEexpanded', 'categorical', 'SLIM', 'neighbor')]
            for seed in runner.SEEDS:
                directory = root / str(seed)
                artifacts = {}
                for name in names:
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'synthetic payload only')
                    artifacts[name] = runner.digest(path)
                runner.atomic_json(directory / 'selection-manifest.json',
                    {'seed': seed, 'arms': list(runner.ARMS), 'artifacts': artifacts})
                manifests[str(seed)] = runner.digest(directory / 'selection-manifest.json')
            seal = {'status': 'frozen', 'seeds': list(runner.SEEDS),
                'seed_manifest_sha256': manifests, 'source_sha256': source_hashes,
                'plan_sha256': runner.digest(root / 'plan.json'),
                'training_summary_sha256': runner.digest(root / 'training-summary.json'),
                'meta_screen_sha256': runner.digest(root / 'meta-screen.json'),
                'development_read': False, 'original_test_read': False}
            runner.atomic_json(root / 'SELECTIONS-FROZEN.json', seal)
            (root / 'SELECTIONS-FROZEN.sha256').write_text(runner.digest(root / 'SELECTIONS-FROZEN.json') + '\n')
            self.assertEqual(runner.verify_global_seal(root), seal)
            for name in ('training-summary.json', 'meta-screen.json',
                         f'{runner.SEEDS[0]}/selected-scores/raw.npy', 'synthetic-source.txt'):
                path = root / name
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaises(ValueError):
                    # Empty inputs would fail with KeyError if accessed before
                    # the rejecting seal callback. No outcome file is present.
                    data.load_development({}, root, verify_global_seal=runner.verify_global_seal)
                path.write_bytes(original)
            with mock.patch.object(runner, 'runtime', return_value={'changed': True}):
                with self.assertRaisesRegex(ValueError, 'runtime'):
                    runner.verify_global_seal(root)
            changed = copy.deepcopy(seal)
            changed['seeds'] = list(runner.SEEDS[:-1])
            runner.atomic_json(root / 'SELECTIONS-FROZEN.json', changed)
            (root / 'SELECTIONS-FROZEN.sha256').write_text(runner.digest(root / 'SELECTIONS-FROZEN.json') + '\n')
            with self.assertRaisesRegex(ValueError, 'all-seed'):
                runner.verify_global_seal(root)


if __name__ == '__main__':
    unittest.main()
