import math
import unittest

import numpy as np

from categorical_experiment import (categorical_baselines, categorical_matrix,
                                    categorical_metrics, evaluate_predictions,
                                    ranking_scores, validate_probabilities)


class CategoricalExperimentTests(unittest.TestCase):
    def test_five_category_training_keeps_neutral_and_unseen_distinct(self):
        matrix = categorical_matrix(['a', 'b'], ['[PAD]', 'i', 'j', 'k'],
                                    [('a', 'i', 1), ('a', 'j', 3), ('b', 'j', 5)])
        np.testing.assert_array_equal(matrix, [[0, 1, 3, 0], [0, 0, 5, 0]])
        for invalid in ([('a', 'i', 2.5)], [('a', 'i', 6)], [('a', '[PAD]', 2)],
                        [('a', 'i', 2), ('a', 'i', 4)]):
            with self.assertRaises(ValueError):
                categorical_matrix(['a'], ['[PAD]', 'i'], invalid)

    def test_macro_and_pooled_ce_use_only_observed_labels_and_chosen_users(self):
        users, items = ['a', 'b', 'c'], ['[PAD]', 'i', 'j']
        p = np.full((3, 3, 5), .2)
        p[0, 1] = [.5, .125, .125, .125, .125]
        p[1, 1] = [.25, .1875, .1875, .1875, .1875]
        p[1, 2] = [.25, .1875, .1875, .1875, .1875]
        labels = {('a', 'i'): 1, ('b', 'i'): 1, ('b', 'j'): 1, ('c', 'j'): 5}
        result = categorical_metrics(p, users, items, labels, {'a', 'b'})
        self.assertAlmostEqual(result['macro_cross_entropy'], (math.log(2)+math.log(4))/2)
        self.assertAlmostEqual(result['pooled_cross_entropy'], (math.log(2)+2*math.log(4))/3)
        self.assertEqual(result['observations'], 3)
        self.assertEqual(set(result['per_user']), {'a', 'b'})
        labels['c', 'j'] = float('nan')
        self.assertEqual(result, categorical_metrics(p, users, items, labels, {'a', 'b'}))

    def test_uniform_and_perfect_probabilities_have_known_ce_and_brier(self):
        users, items, labels = ['u'], ['[PAD]', 'i'], {('u', 'i'): 3}
        p = np.full((1, 2, 5), .2)
        result = categorical_metrics(p, users, items, labels, users)
        self.assertAlmostEqual(result['macro_cross_entropy'], math.log(5))
        self.assertAlmostEqual(result['macro_brier'], .8)
        self.assertAlmostEqual(result['macro_expected_rating_rmse'], 0.)
        p[0, 1] = [0, 0, 1, 0, 0]
        perfect = categorical_metrics(p, users, items, labels, users)
        self.assertAlmostEqual(perfect['macro_cross_entropy'], 0)
        self.assertAlmostEqual(perfect['macro_brier'], 0)
        self.assertEqual(perfect['macro_category_accuracy'], 1.)

    def test_unseen_and_invalid_probabilities_rejected(self):
        for p in (np.ones((1, 2, 5)), np.full((1, 2, 5), np.nan), np.full((1, 2, 4), .25)):
            with self.assertRaises(ValueError):
                validate_probabilities(p, ['u'], ['[PAD]', 'i'])
        with self.assertRaisesRegex(ValueError, 'no rated'):
            categorical_metrics(np.full((2, 2, 5), .2), ['u', 'v'], ['[PAD]', 'i'],
                                {('u', 'i'): 4}, {'u', 'v'})

    def test_ranking_readout_masks_every_observed_rating_including_neutral(self):
        users, items = ['u'], ['[PAD]', 'a', 'b', 'c', 'd']
        training = [('u', 'a', 1), ('u', 'b', 3)]
        matrix = categorical_matrix(users, items, training)
        p = np.full((1, 5, 5), .2)
        p[0, 1] = [0, 0, 0, 0, 1]
        p[0, 2] = [0, 0, 0, 0, 1]
        p[0, 3] = [0, 0, 0, .5, .5]
        result, recs = evaluate_predictions(p, users, items, matrix,
            [(u, i) for u, i, _ in training], {('u', 'c'): 4}, users, k=2)
        self.assertEqual(recs['u'], ['c', 'd'])
        self.assertEqual(result['ranking']['denominators']['all_observed_users'], 1)
        np.testing.assert_array_equal(ranking_scores(p, users, items), [[.4, 1., 1., 1., .4]])

    def test_smoothed_baselines_match_count_formulas_and_cold_item_prior(self):
        ratings = np.array([[0, 1, 3, 0], [0, 1, 5, 0], [0, 5, 0, 0]])
        baselines = categorical_baselines(ratings)
        prior = np.array([3, 1, 2, 1, 3])/10.
        np.testing.assert_allclose(baselines['global_histogram'][0, 1], prior)
        np.testing.assert_allclose(baselines['item_histogram'][0, 1], (np.array([2, 0, 0, 0, 1])+10*prior)/13)
        np.testing.assert_allclose(baselines['item_histogram'][0, 3], prior)
        user_prior = (np.array([1, 0, 1, 0, 0])+10*prior)/12
        np.testing.assert_allclose(baselines['item_user_product'][0, 3], user_prior)
        for probabilities in baselines.values():
            validate_probabilities(probabilities, ['a', 'b', 'c'], ['[PAD]', 'a', 'b', 'cold'])


class CategoricalTrainingBoundaryTests(unittest.TestCase):
    def test_hidden_rating_values_never_enter_context_or_choose_probe(self):
        from categorical_experiment import masked_episode
        matrix = np.array([[0, 1, 2, 3, 4, 5, 1], [0, 5, 4, 3, 2, 1, 5]])
        context, mask = masked_episode(matrix, 11, 3)
        changed = matrix.copy()
        changed[mask] = 6-changed[mask]
        other_context, other_mask = masked_episode(changed, 11, 3)
        np.testing.assert_array_equal(mask, other_mask)
        np.testing.assert_array_equal(context, other_context)
        self.assertTrue(np.all(context[mask] == 0))
        self.assertTrue(np.all(mask.sum(axis=1) >= 1))
        self.assertTrue(np.all((context > 0).sum(axis=1) >= 1))

    def test_probe_loss_rejects_unmasked_targets_and_ignores_nonprobe_values(self):
        import torch
        from categorical_experiment import make_model, masked_episode, training_loss
        torch.set_num_threads(1)
        matrix = np.array([[0, 1, 2, 3, 4, 5, 1], [0, 5, 4, 3, 2, 1, 5]])
        context, mask = masked_episode(matrix, 11, 3)
        model = make_model(7, 17, 'adaptive')
        args = [torch.tensor(context), torch.tensor(matrix), torch.tensor(mask)]
        first = training_loss(model, *args)
        changed = args[1].clone()
        changed[~args[2]] = 0
        torch.testing.assert_close(first, training_loss(model, args[0], changed, args[2]))
        with self.assertRaisesRegex(ValueError, 'absent'):
            training_loss(model, args[1], args[1], args[2])

    def test_checkpoint_probability_replay_and_identical_initialization(self):
        import tempfile
        from pathlib import Path
        import torch
        from categorical_experiment import (VARIANTS, make_model, predict_probabilities,
                                            restore, snapshot, state_digest)
        torch.set_num_threads(1)
        models = [make_model(7, 31, variant) for variant in VARIANTS]
        self.assertEqual(len({state_digest(model) for model in models}), 1)
        matrix = np.array([[0, 1, 2, 3, 4, 5, 1], [0, 5, 4, 3, 2, 1, 5]])
        before = predict_probabilities(models[0], matrix)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.pt'
            torch.save(snapshot(models[0], 10, 31), path)
            restored, saved = restore(path)
            self.assertEqual(saved['epoch'], 10)
            np.testing.assert_array_equal(before, predict_probabilities(restored, matrix))


if __name__ == '__main__':
    unittest.main()
