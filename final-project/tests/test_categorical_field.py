import io
import unittest

import torch
from torch.nn import functional as F

from categorical_field import CategoricalEvidenceField, VARIANTS, summarize_diagnostics


class CategoricalFieldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.context = torch.tensor([[0, 5, 1, 0, 3, 0], [0, 2, 0, 4, 1, 0]], dtype=torch.long)
        self.model = CategoricalEvidenceField(6, dim=4, ports=3, steps=4)

    def variant(self, name):
        model = CategoricalEvidenceField(**{**self.model.config, 'variant': name})
        model.load_state_dict(self.model.state_dict())
        return model

    def test_context_contract_and_categorical_output(self):
        output = self.model(self.context)['rating_logits']
        self.assertEqual(tuple(output.shape), (2, 6, 5))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.softmax(-1).sum(-1), torch.ones(2, 6))
        invalid = [self.context.float(), self.context[:, 1:], self.context[:0]]
        for context in invalid:
            with self.assertRaises(ValueError):
                self.model(context)
        for item, value in [(0, 1), (1, 6), (1, -1)]:
            bad = self.context.clone()
            bad[0, item] = value
            with self.assertRaises(ValueError):
                self.model(bad)

    def test_absent_values_never_supply_source_or_upward_routing(self):
        result = self.model(self.context, return_diagnostics=True)
        d = result['diagnostics']
        absent = self.context == 0
        self.assertTrue((d['source'][absent] == 0).all())
        self.assertTrue((d['fidelity'].masked_select(absent[:, None, :]) == 0).all())
        self.assertTrue((d['up'].masked_select(absent[:, None, None, :]) == 0).all())
        torch.testing.assert_close(d['up'].sum(-1), torch.ones(2, 4, 3))
        # Even a corrupt absent-category embedding cannot turn absence into data.
        with torch.no_grad():
            self.model.rating_embedding.weight[0].fill_(100)
        torch.testing.assert_close(self.model(self.context)['rating_logits'], result['rating_logits'])

    def test_empty_context_stays_zero_and_finite_in_all_variants(self):
        empty = torch.zeros_like(self.context)
        for name in VARIANTS:
            result = self.variant(name)(empty, return_diagnostics=True)
            for key in ('states', 'source', 'up', 'fidelity'):
                self.assertTrue((result['diagnostics'][key] == 0).all(), (name, key))
            self.assertTrue(torch.isfinite(result['rating_logits']).all())
            self.assertEqual(summarize_diagnostics(result['diagnostics'])['empty_context_users'], 2)

    def test_masked_probe_categories_cannot_change_prediction(self):
        first, second = self.context.clone(), self.context.clone()
        first[:, 5], second[:, 5] = torch.tensor([1, 5]), torch.tensor([5, 1])
        probe = torch.zeros_like(first, dtype=torch.bool)
        probe[:, [2, 5]] = True
        first[probe], second[probe] = 0, 0
        torch.testing.assert_close(self.model(first)['rating_logits'], self.model(second)['rating_logits'], rtol=0, atol=0)

    def test_unobserved_candidate_cannot_be_reintroduced_as_source(self):
        before = self.model(self.context)['rating_logits'].detach()
        with torch.no_grad():
            self.model.item_embedding.weight[5].add_(10.)
        after = self.model(self.context)['rating_logits']
        torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)
        self.assertFalse(torch.allclose(before[:, 5], after[:, 5]))

    def test_soft_and_hard_source_behavior_and_convex_bounds(self):
        soft = self.model(self.context, True)['diagnostics']
        hard = self.variant('hard_clamp')(self.context, True)['diagnostics']
        observed = self.context != 0
        for step in range(1, 5):
            torch.testing.assert_close(hard['states'][:, step][observed], hard['source'][observed])
        self.assertGreater(float((soft['states'][:, -1][observed] - soft['source'][observed]).detach().abs().max()), 1e-4)
        for name in VARIANTS:
            states = self.variant(name)(self.context, True)['diagnostics']['states']
            self.assertLessEqual(float(states.detach().abs().max()), 1.)
            self.assertTrue((states[:, :, 0] == 0).all())

    def test_dynamic_mechanisms_and_isolated_controls(self):
        adaptive = self.model(self.context, True)['diagnostics']
        fixed = self.variant('fixed_flow')(self.context, True)['diagnostics']
        route_fixed = self.variant('fixed_routing')(self.context, True)['diagnostics']
        gate_fixed = self.variant('fixed_fidelity')(self.context, True)['diagnostics']
        for key in ('up', 'down', 'fidelity'):
            torch.testing.assert_close(fixed[key][:, -1], fixed[key][:, 0], rtol=0, atol=0)
            self.assertGreater(float((adaptive[key][:, -1] - adaptive[key][:, 0]).detach().abs().max()), 1e-7)
        for key in ('up', 'down'):
            torch.testing.assert_close(route_fixed[key][:, -1], route_fixed[key][:, 0], rtol=0, atol=0)
        torch.testing.assert_close(gate_fixed['fidelity'][:, -1], gate_fixed['fidelity'][:, 0], rtol=0, atol=0)
        summary = summarize_diagnostics(adaptive)
        self.assertGreater(summary['steps'][-1]['up_routing_tv_from_initial'], 0)
        self.assertGreater(summary['steps'][-1]['source_gate_mean_absolute_change_from_initial'], 0)
        self.assertEqual(summarize_diagnostics(fixed)['steps'][-1]['up_routing_tv_from_initial'], 0)

    def test_one_step_fixed_flow_and_adaptive_are_identical(self):
        self.model.steps = 1
        fixed = self.variant('fixed_flow')
        fixed.steps = 1
        torch.testing.assert_close(self.model(self.context)['rating_logits'], fixed(self.context)['rating_logits'], rtol=0, atol=0)

    def test_probe_loss_reaches_routing_categories_and_source_gates(self):
        logits = self.model(self.context)['rating_logits'][:, 5]
        F.cross_entropy(logits, torch.tensor([0, 4])).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        self.assertTrue((self.model.rating_embedding.weight.grad[0] == 0).all())

    def test_context_changes_probe_and_users_are_independent(self):
        logits = self.model(self.context)['rating_logits']
        self.assertFalse(torch.allclose(logits[0, 5], logits[1, 5]))
        torch.testing.assert_close(self.model(self.context.flip(0))['rating_logits'], logits.flip(0))
        torch.testing.assert_close(self.model(self.context[:1])['rating_logits'], logits[:1])

    def test_all_variants_allocate_same_parameters_and_replay_exactly(self):
        count = sum(parameter.numel() for parameter in self.model.parameters())
        for name in VARIANTS:
            self.assertEqual(sum(parameter.numel() for parameter in self.variant(name).parameters()), count)
        buffer = io.BytesIO()
        torch.save({'config': self.model.config, 'state_dict': self.model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = CategoricalEvidenceField(**saved['config'])
        restored.load_state_dict(saved['state_dict'])
        torch.testing.assert_close(self.model(self.context)['rating_logits'], restored(self.context)['rating_logits'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
