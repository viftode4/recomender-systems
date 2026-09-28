import tempfile
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from recbole.data.interaction import Interaction

from experiment import variant_settings
from run import (FISMCorrected, NGCFCompatible, full_catalog_scores,
                 genre_content_features, recbole_model_name)


def toy_fism():
    model = FISMCorrected.__new__(FISMCorrected)
    torch.nn.Module.__init__(model)
    model.USER_ID, model.ITEM_ID, model.LABEL = 'user', 'item', 'label'
    model.n_items, model.alpha, model.reg_weights = 4, .5, [0., 0.]
    model.history_item_matrix = torch.tensor([[0, 0], [1, 2], [2, 0], [0, 0]])
    model.mask_mat = torch.tensor([[0., 0.], [1., 1.], [1., 0.], [0., 0.]])
    model.history_lens = torch.tensor([0, 2, 1, 0])
    model.item_src_embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor([[100., 200.], [1., 2.], [3., 4.], [5., 6.]]), freeze=False)
    model.item_dst_embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor([[200., 100.], [2., 1.], [4., 3.], [6., 5.]]), freeze=False)
    model.user_bias = torch.nn.Parameter(torch.tensor([0., .1, .2, .3]))
    model.item_bias = torch.nn.Parameter(torch.tensor([0., .3, .2, .1]))
    model.bceloss = torch.nn.BCEWithLogitsLoss()
    return model


class RunnerModelTests(unittest.TestCase):
    def test_fism_point_and_full_match_independent_leave_target_out_equation(self):
        model = toy_fism()
        with torch.no_grad():
            full = model.full_sort_predict(Interaction({'user': torch.tensor([1, 2, 3])})).reshape(3, 4)
            expected = np.empty((3, 4))
            for row, user in enumerate([1, 2, 3]):
                for item in range(4):
                    history = [int(i) for i in model.history_item_matrix[user] if i and i != item]
                    similarity = sum(float(model.item_src_embedding.weight[h] @ model.item_dst_embedding.weight[item])
                                     for h in history)
                    expected[row, item] = similarity / max(len(history), 1) ** .5 + float(model.user_bias[user] + model.item_bias[item])
            np.testing.assert_allclose(full.numpy(), expected, rtol=1e-6)
            interactions = Interaction({'user': torch.tensor([1, 2, 3]).repeat_interleave(4),
                                        'item': torch.arange(4).repeat(3)})
            torch.testing.assert_close(model.predict(interactions), torch.sigmoid(full.reshape(-1)))

    def test_fism_loss_consumes_logits_and_training_target_cannot_predict_itself(self):
        model = toy_fism()
        interaction = Interaction({'user': torch.tensor([1]), 'item': torch.tensor([1]), 'label': torch.tensor([0.])})
        # Only item 2 may support target 1; its dot product is 10, plus biases .4.
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([10.4]), torch.tensor([0.]))
        loss = model.calculate_loss(interaction)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(model.item_src_embedding.weight.grad[1].abs().sum()), 0.)
        self.assertGreater(float(model.item_src_embedding.weight.grad[2].abs().sum()), 0.)

    def test_ngcf_public_sparse_normalization_matches_dense_equation(self):
        model = NGCFCompatible.__new__(NGCFCompatible)
        torch.nn.Module.__init__(model)
        model.n_users, model.n_items = 2, 3
        model.interaction_matrix = sp.coo_matrix(np.array([[0., 1., 1.], [0., 1., 0.]], dtype=np.float32))
        dense = np.block([[np.zeros((2, 2)), model.interaction_matrix.toarray()],
                          [model.interaction_matrix.toarray().T, np.zeros((3, 3))]])
        scale = (dense.sum(axis=1) + 1e-7) ** -.5
        expected = dense * scale[:, None] * scale[None, :]
        np.testing.assert_allclose(model.get_norm_adj_mat().to_dense().numpy(), expected, rtol=1e-6)

    def test_neumf_full_catalog_adapter_preserves_point_scores_across_chunks(self):
        class PointModel:
            def predict(self, interaction):
                return interaction['u'].float() * 10 + interaction['i'].float()
        scores = full_catalog_scores(PointModel(), 'NeuMF', 2, 'u', 'i', 2050, 'cpu')
        np.testing.assert_array_equal(scores, 20 + np.arange(2050))

    def test_genre_profile_uses_only_given_training_items_and_original_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'items.tsv'
            path.write_text('item_id:token\tclass:token_seq\n42\tDrama\n9\tAction\n7\tDrama Action\n')
            features, profiles, genres = genre_content_features(path, ['PAD', '9', '42', '7'], ['PAD', 'alice', 'bob'], [('alice', '42')])
        scores = profiles @ features.T
        self.assertEqual(genres, ['Action', 'Drama'])
        self.assertAlmostEqual(scores[1, 2], 1.)
        self.assertAlmostEqual(scores[1, 1], 0.)
        self.assertGreater(scores[1, 3], 0.)
        np.testing.assert_array_equal(scores[2], np.zeros(4))

    def test_model_specific_parameters_do_not_contaminate_graph_regularization(self):
        self.assertEqual(recbole_model_name('UserKNN'), 'ItemKNN')
        self.assertEqual(recbole_model_name('FISMCorrected'), 'FISM')
        for model in ('LightGCN', 'NGCF'):
            variants = variant_settings(model, True)
            self.assertTrue(all(v['reg_weight'] == 1e-5 for v in variants))
            self.assertEqual([v['epochs'] for v in variants], [20, 60])
            self.assertEqual(len(variant_settings(model, True, epochs=1)), 1)

    def test_thorough_graph_budget_keeps_other_models_and_default_grid_unchanged(self):
        self.assertEqual([v['epochs'] for v in variant_settings('LightGCN', True)], [20, 60])
        for tune in (False, True):
            variants = variant_settings('LightGCN', tune, thorough_lightgcn_budget=True)
            self.assertEqual([v['epochs'] for v in variants], [20, 60, 100, 200])
            self.assertTrue(all(v['embedding_size'] == 64 and v['reg_weight'] == 1e-5 for v in variants))
        self.assertEqual(variant_settings('NGCF', True, thorough_lightgcn_budget=True), variant_settings('NGCF', True))

    def test_invalid_study_and_conflicting_budgets_fail_before_creating_runs(self):
        root = Path(__file__).resolve().parents[1]
        cases = [(['--models', 'Random', 'EASE'], 'two non-Random experts'),
                 (['--models', 'LightGCN', '--skip-study', '--thorough-lightgcn-budget', '--epochs', '1'], 'cannot be combined')]
        with tempfile.TemporaryDirectory() as tmp:
            for index, (arguments, error) in enumerate(cases):
                output = Path(tmp) / str(index)
                result = subprocess.run([sys.executable, str(root/'experiment.py'), '--data-path', tmp,
                                         '--out', str(output), *arguments], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
