import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp, softmax
import torch
from torch import nn

from categorical_experiment import categorical_metrics, make_model, masked_episode
from joint_field_experiment import (baseline_logits, joint_metrics, joint_probe_loss,
    predict_logits, ranking_scores, train_variant)


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float64))

    def forward(self, context):
        return {'rating_logits': self.logits}


class JointFieldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.users = ['u', 'v']
        self.items = ['[PAD]', 'a', 'b', 'c']
        self.context = np.array([[0, 2, 0, 0], [0, 0, 3, 0]], dtype=np.int64)
        self.labels = {('u', 'b'): 5, ('u', 'c'): 1, ('v', 'a'): 2}
        self.logits = np.arange(40, dtype=float).reshape(2, 4, 5) / 13

    def test_known_factorized_mass_and_joint_decomposition(self):
        masses = np.array([999, 1, 2, 3.])
        categories = np.array([.05, .1, .15, .3, .4])
        logits = np.broadcast_to(np.log(masses[:, None] * categories), (2, 4, 5))
        metrics = joint_metrics(logits, self.users, self.items, self.context, self.labels, self.users)
        expected_u = np.mean([-np.log(2 / 5) - np.log(.4), -np.log(3 / 5) - np.log(.05)])
        expected_v = -np.log(1 / 4) - np.log(.1)
        self.assertAlmostEqual(metrics['macro_joint_nll'], (expected_u + expected_v) / 2)
        self.assertAlmostEqual(metrics['pooled_joint_nll'], (2 * expected_u + expected_v) / 3)
        self.assertAlmostEqual(metrics['macro_joint_nll'], metrics['macro_item_event_nll'] + metrics['macro_conditional_category_nll'])

    def test_masked_context_and_padding_never_enter_normalizer(self):
        before = joint_metrics(self.logits, self.users, self.items, self.context, self.labels, self.users)
        shifted = self.logits.copy()
        shifted[:, 0] += 1000
        shifted[self.context != 0] += 1000
        after = joint_metrics(shifted, self.users, self.items, self.context, self.labels, self.users)
        self.assertEqual(before, after)

    def test_conditional_shift_ambiguity_removed_but_global_shift_invariant(self):
        before = joint_metrics(self.logits, self.users, self.items, self.context, self.labels, self.users)
        shifted = self.logits.copy()
        shifted[:, 3] += 3
        after = joint_metrics(shifted, self.users, self.items, self.context, self.labels, self.users)
        self.assertAlmostEqual(before['macro_conditional_category_nll'], after['macro_conditional_category_nll'])
        self.assertNotAlmostEqual(before['macro_item_event_nll'], after['macro_item_event_nll'])
        global_shift = joint_metrics(self.logits + 42, self.users, self.items, self.context, self.labels, self.users)
        self.assertAlmostEqual(before['macro_joint_nll'], global_shift['macro_joint_nll'])

    def test_training_loss_matches_independent_evaluator_and_masks_gradients(self):
        targets = self.context.copy()
        probes = np.zeros_like(targets, dtype=bool)
        for (u, i), rating in self.labels.items():
            row, col = self.users.index(u), self.items.index(i)
            targets[row, col], probes[row, col] = rating, True
        model = FixedLogits(self.logits)
        loss = joint_probe_loss(model, torch.from_numpy(self.context), torch.from_numpy(targets), torch.from_numpy(probes))
        expected = joint_metrics(self.logits, self.users, self.items, self.context, self.labels, self.users)
        self.assertAlmostEqual(float(loss.detach()), expected['macro_joint_nll'])
        loss.backward()
        self.assertTrue((model.logits.grad[:, 0] == 0).all())
        self.assertTrue((model.logits.grad[torch.from_numpy(self.context != 0)] == 0).all())
        # Eligible but unrecorded categories compete, without a dislike target.
        self.assertGreater(float(model.logits.grad[0, 2, 0]), 0.)
        self.assertLess(float(model.logits.grad[0, 2, 4]), 0.)

    def test_training_rejects_unmasked_probes_and_padding_targets(self):
        targets = torch.tensor([[0, 5, 2, 0]])
        context = torch.tensor([[0, 5, 0, 0]])
        model = FixedLogits(np.zeros((1, 4, 5)))
        for probes in (torch.tensor([[False, True, False, False]]),
                       torch.tensor([[True, False, False, False]]),
                       torch.zeros((1, 4), dtype=torch.bool)):
            with self.assertRaises(ValueError):
                joint_probe_loss(model, context, targets, probes)

    def test_other_cohort_labels_cannot_change_selection_metric(self):
        baseline = joint_metrics(self.logits, self.users, self.items, self.context, self.labels, {'u'})
        poisoned = {**self.labels, ('v', 'a'): np.nan, ('v', '[PAD]'): 99}
        self.assertEqual(baseline, joint_metrics(self.logits, self.users, self.items, self.context, poisoned, {'u'}))
        with self.assertRaises(ValueError):
            joint_metrics(self.logits, self.users, self.items, self.context, {('u', 'a'): 3}, {'u'})

    def test_ranking_adapters_have_declared_distinct_meanings(self):
        logits = np.log(np.array([[[1, 1, 1, 1, 1], [100, 100, 100, 1, 1], [1, 1, 1, 20, 20]]], dtype=float))
        scores = ranking_scores(logits)
        self.assertGreater(scores['all_observed_adapter'][0, 1], scores['all_observed_adapter'][0, 2])
        self.assertLess(scores['liked_record_adapter'][0, 1], scores['liked_record_adapter'][0, 2])
        np.testing.assert_allclose(scores['all_observed_adapter'], np.log(np.exp(logits).sum(-1)))
        np.testing.assert_allclose(scores['liked_record_adapter'], np.log(np.exp(logits[:, :, 3:]).sum(-1)))

    def test_fixed_count_baselines_match_declared_formulas(self):
        matrix = np.array([[0, 5, 2, 0], [0, 4, 0, 1]], dtype=np.int64)
        baselines = baseline_logits(matrix)
        prior = np.array([2, 2, 1, 2, 2]) / 9
        event = np.array([1, 3, 2, 2])
        np.testing.assert_allclose(np.exp(baselines['event_global_category']), np.broadcast_to(event[None, :, None] * prior, (2, 4, 5)), rtol=1e-6)
        category_counts = np.array([0, 0, 0, 1, 1])
        item = (category_counts + 10 * prior) / 12
        np.testing.assert_allclose(softmax(baselines['event_item_category'][0, 1]), item, rtol=1e-6)
        self.assertFalse(np.allclose(baselines['event_item_user_tilt'][0], baselines['event_item_user_tilt'][1]))

    def test_tiny_training_selection_replay_ignores_development_labels(self):
        matrix = np.array([[0, 1, 5, 0, 2, 0], [0, 4, 0, 1, 5, 0]], dtype=np.int64)
        users, items = ['u', 'v'], ['[PAD]', 'a', 'b', 'c', 'd', 'e']
        labels = {('u', 'c'): 4, ('v', 'b'): 2}
        with tempfile.TemporaryDirectory() as tmp, patch('joint_field_experiment.CHECKPOINTS', (1, 2)):
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            first.mkdir(); second.mkdir()
            a = train_variant(matrix, users, items, labels, {'u'}, 9, 'adaptive', first)
            b = train_variant(matrix, users, items, {**labels, ('v', 'b'): np.nan}, {'u'}, 9, 'adaptive', second)
            self.assertEqual(a['epoch'], b['epoch'])
            self.assertEqual(a['logits_array_sha256'], b['logits_array_sha256'])
            self.assertEqual(a['meta_fit_macro_joint_nll'], b['meta_fit_macro_joint_nll'])
            self.assertTrue(a['checkpoint_replay_exact'])


if __name__ == '__main__':
    unittest.main()
