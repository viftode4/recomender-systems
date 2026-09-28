import copy
import unittest

import numpy as np
import torch

from exploratory.addressed_evidence.model import AddressedEvidenceModel, build_neighbors


class AddressedEvidenceTests(unittest.TestCase):
    def graph(self):
        return np.asarray([[0, 0, 0], [2, 3, 4], [1, 3, 0], [1, 4, 0], [2, 3, 0]])

    def model(self, variant='pair', **kwargs):
        return AddressedEvidenceModel(n_items=5, k=3, variant=variant, seed=7,
                                      neighbors=self.graph(), **kwargs)

    def test_neighbor_cosine_tie_order_and_empty_sources(self):
        ratings = np.asarray([[0, 5, 2, 3, 0, 0], [0, 2, 4, 1, 0, 0],
                              [0, 1, 0, 0, 4, 0], [0, 0, 0, 0, 5, 0]])
        graph = build_neighbors(ratings, k=4)
        self.assertEqual(graph[1].tolist(), [2, 3, 4, 0])
        self.assertEqual(graph[2].tolist(), [3, 1, 0, 0])
        self.assertFalse(graph[0].any())
        self.assertFalse(graph[5].any())
        for i, row in enumerate(graph):
            self.assertNotIn(i, row[row != 0])
        np.testing.assert_array_equal(graph, build_neighbors(ratings[::-1], k=4))
        np.testing.assert_array_equal(graph, build_neighbors(np.where(ratings, 1, 0), k=4))

    def test_same_initial_parameters_and_exact_additive_pair_outputs(self):
        first, second = self.model('additive'), self.model('pair')
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))
        context = torch.tensor([[0, 5, 3, 0, 1], [0, 0, 0, 2, 0]])
        self.assertTrue(torch.equal(first(context)['rating_logits'], second(context)['rating_logits']))
        self.assertEqual(sum(p.numel() for p in first.parameters()), sum(p.numel() for p in second.parameters()))
        first(context)['rating_logits'].square().sum().backward()
        second(context)['rating_logits'].square().sum().backward()
        self.assertTrue(torch.equal(first.potentials.grad, second.potentials.grad))
        self.assertTrue(torch.equal(first.bias.grad, second.bias.grad))
        self.assertIsNone(first.pair_raw.grad)
        self.assertTrue(second.pair_raw.grad.abs().sum() > 0)

    def test_empty_context_is_bias_and_padding_is_zero(self):
        model = self.model()
        with torch.no_grad():
            model.bias.fill_(.7)
            model.pair_raw.fill_(.8)
        result = model(torch.zeros((2, 5), dtype=torch.long), return_diagnostics=True)
        self.assertTrue(torch.equal(result['rating_logits'][:, 1:], torch.full((2, 4, 5), .7)))
        self.assertFalse(result['rating_logits'][:, 0].any())
        self.assertFalse(result['diagnostics']['distinct_pair_mean'].any())

    def test_source_item_and_rating_identity_and_explicit_distinct_pair_formula(self):
        model = self.model(init_scale=0.)
        with torch.no_grad():
            model.potentials[1, 0, 1] = torch.tensor([1., 2., 3., 4., 5.])  # item2, rating2
            model.potentials[1, 1, 4] = torch.tensor([2., -1., 4., -3., 6.])  # item3, rating5
            model.potentials[1, 2, 0] = torch.tensor([-2., 3., 1., 2., 4.])  # item4, rating1
            model.pair_raw.fill_(.3)
        context = torch.tensor([[0, 0, 2, 5, 1]])
        result = model(context, return_diagnostics=True)
        votes = torch.stack([model.potentials[1, 0, 1], model.potentials[1, 1, 4], model.potentials[1, 2, 0]])
        pair_sum = votes[0]*votes[1] + votes[0]*votes[2] + votes[1]*votes[2]
        expected_pair = pair_sum/(.01*np.sqrt(3))
        torch.testing.assert_close(result['diagnostics']['distinct_pair_mean'][0, 1], pair_sum/3)
        torch.testing.assert_close(result['diagnostics']['distinct_pair_feature'][0, 1], expected_pair)
        torch.testing.assert_close(result['rating_logits'][0, 1], votes.sum(0)/np.sqrt(3)+np.tanh(.3)*expected_pair)
        swapped = context.clone()
        swapped[:, [2, 3]] = swapped[:, [3, 2]]
        self.assertFalse(torch.equal(result['rating_logits'][0, 1], model(swapped)['rating_logits'][0, 1]))

    def test_self_rating_does_not_reach_own_candidate_and_single_source_has_no_pair(self):
        model = self.model()
        with torch.no_grad():
            model.pair_raw.fill_(.8)
        a, b = torch.tensor([[0, 0, 3, 0, 0]]), torch.tensor([[0, 5, 3, 0, 0]])
        self.assertTrue(torch.equal(model(a)['rating_logits'][:, 1], model(b)['rating_logits'][:, 1]))
        self.assertFalse(model(a, return_diagnostics=True)['diagnostics']['distinct_pair_mean'][0, 1].any())

    def test_source_slot_permutation_with_corresponding_tables_preserves_predictions(self):
        model = self.model()
        with torch.no_grad():
            model.pair_raw.fill_(.5)
        permuted = copy.deepcopy(model)
        order = torch.tensor([2, 0, 1])
        with torch.no_grad():
            permuted.neighbors.copy_(model.neighbors[:, order])
            permuted.potentials.copy_(model.potentials[:, order])
        context = torch.tensor([[0, 1, 2, 3, 4], [0, 4, 0, 2, 5]])
        torch.testing.assert_close(model(context)['rating_logits'], permuted(context)['rating_logits'])

    def test_absent_evidence_has_no_gradient_and_pair_coefficient_can_learn(self):
        model = self.model()
        context = torch.tensor([[0, 0, 2, 5, 0]])
        model(context)['rating_logits'][0, 1].sum().backward()
        self.assertFalse(model.potentials.grad[1, 2].any())  # absent item4
        self.assertFalse(model.potentials.grad[1, 0, 0].any())  # wrong source category
        self.assertTrue(model.potentials.grad[1, 0, 1].abs().sum() > 0)
        self.assertTrue(model.pair_raw.grad[1].abs().sum() > 0)
        self.assertFalse(model.potentials.grad[0].any())

    def test_strict_restore_replays_graph_and_scores(self):
        model = self.model()
        restored = AddressedEvidenceModel(**model.config)
        restored.load_state_dict(model.state_dict(), strict=True)
        context = torch.tensor([[0, 2, 5, 4, 1]])
        self.assertTrue(torch.equal(model(context)['rating_logits'], restored(context)['rating_logits']))
        self.assertTrue(torch.equal(model.neighbors, restored.neighbors))

    def test_pair_scale_is_fixed_positive_and_saved_in_config(self):
        first, second = self.model(pair_vote_scale=.01), self.model(pair_vote_scale=.1)
        with torch.no_grad():
            first.pair_raw.fill_(.4)
            second.pair_raw.fill_(.4)
        context = torch.tensor([[0, 1, 2, 3, 4]])
        a, b = first(context, return_diagnostics=True), second(context, return_diagnostics=True)
        torch.testing.assert_close(a['diagnostics']['distinct_pair_feature'], b['diagnostics']['distinct_pair_feature']*10)
        self.assertEqual(second.config['pair_vote_scale'], .1)
        self.assertTrue(torch.equal(a['diagnostics']['direct'], b['diagnostics']['direct']))
        for invalid in (0., -.01, float('nan'), float('inf')):
            with self.subTest(scale=invalid), self.assertRaises(ValueError):
                self.model(pair_vote_scale=invalid)

    def test_single_slot_pair_feature_and_pair_gradient_are_zero(self):
        graph = np.asarray([[0], [2], [1]])
        model = AddressedEvidenceModel(3, k=1, variant='pair', neighbors=graph)
        with torch.no_grad():
            model.pair_raw.fill_(.8)
        result = model(torch.tensor([[0, 3, 5]]), return_diagnostics=True)
        self.assertFalse(result['diagnostics']['distinct_pair_feature'].any())
        self.assertTrue(torch.isfinite(result['rating_logits']).all())
        result['rating_logits'].sum().backward()
        self.assertFalse(model.pair_raw.grad.any())

    def test_invalid_graph_and_context_rejected(self):
        graph = self.graph()
        graph[1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'self edges'):
            AddressedEvidenceModel(5, k=3, neighbors=graph)
        graph = self.graph()
        graph[1, 1] = 2
        with self.assertRaisesRegex(ValueError, 'repeated'):
            AddressedEvidenceModel(5, k=3, neighbors=graph)
        for bad in (torch.ones((1, 5), dtype=torch.long), torch.zeros((1, 5)),
                    torch.tensor([[0, 0, 0, 6, 0]])):
            with self.assertRaises(ValueError):
                self.model()(bad)


if __name__ == '__main__':
    unittest.main()
