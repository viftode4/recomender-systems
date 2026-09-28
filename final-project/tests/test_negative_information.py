import unittest

import numpy as np

from exception_model import rating_matrices
from negative_information import (channel_features, fit_decoder, permute_dislike_weights,
                                  split_context_probe, surprise_dislikes)
from negative_information_experiment import array_digest, prepare


class NegativeInformationTests(unittest.TestCase):
    def test_context_split_uses_identities_not_rating_values(self):
        training = [('u'+str(u), 'i'+str(i), 1+(u+i)%5) for u in range(5) for i in range(9)]
        context, probe = split_context_probe(training, 17)
        changed = [(u, i, 6-r) for u, i, r in reversed(training)]
        other_context, other_probe = split_context_probe(changed, 17)
        pairs = lambda rows: {(u, i) for u, i, _ in rows}
        self.assertEqual(pairs(context), pairs(other_context))
        self.assertEqual(pairs(probe), pairs(other_probe))
        self.assertFalse(pairs(context) & pairs(probe))
        self.assertEqual(pairs(context) | pairs(probe), pairs(training))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            split_context_probe(training+training[:1], 17)

    def test_decoder_matches_independent_reduced_feature_fit_and_query(self):
        rng = np.random.default_rng(111)
        x, y, query = rng.normal(size=(7, 10)), rng.normal(size=(7, 5)), rng.normal(size=(3, 10))
        for channels in (1, 2):
            xx, qq = x[:, :channels*5], query[:, :channels*5]
            model = fit_decoder(xx, y, .3, channels)
            expected = np.zeros((len(query), 5))
            for item in range(1, 5):
                excluded = [item+a*5 for a in range(channels)]
                allowed = np.setdiff1d(np.arange(channels*5), excluded)
                design = xx[:, allowed]
                coefficients = np.linalg.solve(design.T@design+.3*np.eye(len(allowed)), design.T@y[:, item])
                expected[:, item] = qq[:, allowed]@coefficients
                np.testing.assert_array_equal(model.coefficients[excluded, item], 0)
            np.testing.assert_allclose(model.predict(qq), expected, atol=1e-10)

    def test_probe_changes_do_not_change_teacher_or_context_features(self):
        users = ['a', 'b', 'c']
        items = ['[PAD]']+list('abcdefgh')
        training = [(u, i, 1+(j+k)%5) for k, u in enumerate(users) for j, i in enumerate(items[1:])]
        context, probe = split_context_probe(training, 91)
        altered_probe = [(u, i, 5 if r <= 2 else 1) for u, i, r in probe]
        original_context, _ = split_context_probe(context+probe, 91)
        altered_context, _ = split_context_probe(context+altered_probe, 91)
        a, _ = rating_matrices(users, items, original_context)
        b, _ = rating_matrices(users, items, altered_context)
        np.testing.assert_array_equal(a, b)
        teachers = [fit_decoder((z > 0).astype(float), (z > 0).astype(float), 250., 1) for z in (a, b)]
        np.testing.assert_array_equal(teachers[0].coefficients, teachers[1].coefficients)
        np.testing.assert_array_equal(surprise_dislikes(a > 0, a < 0, teachers[0]),
                                      surprise_dislikes(b > 0, b < 0, teachers[1]))

    def test_weights_preserve_support_mass_and_permutation_multiset(self):
        p = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0]], bool)
        d = np.array([[0, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], bool)
        teacher = fit_decoder(p, p, 250., 1)
        weights = surprise_dislikes(p, d, teacher)
        np.testing.assert_array_equal(weights > 0, d)
        np.testing.assert_allclose(weights.sum(axis=1), d.sum(axis=1))
        permuted = permute_dislike_weights(weights, 177)
        np.testing.assert_array_equal(permuted > 0, d)
        np.testing.assert_array_equal(np.sort(permuted, axis=1), np.sort(weights, axis=1))
        np.testing.assert_array_equal(permuted, permute_dislike_weights(weights, 177))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            channel_features(p, p)

    def test_fit_rejects_empty_or_invalid_inputs(self):
        for x, y, penalty, channels in [(np.zeros((0, 2)), np.zeros((0, 2)), 1., 1),
                                      (np.zeros((2, 2)), np.zeros((2, 2)), 0., 1),
                                      (np.zeros((2, 3)), np.zeros((2, 2)), 1., 1)]:
            with self.assertRaises(ValueError):
                fit_decoder(x, y, penalty, channels)

    def test_runner_probe_changes_never_reach_context_encoder_or_controls(self):
        users = ['u'+str(j) for j in range(5)]
        items = ['[PAD]']+['i'+str(j) for j in range(15)]
        training = [(u, i, 1+(j+k)%5) for k, u in enumerate(users) for j, i in enumerate(items[1:])]
        context, probe = split_context_probe(training, 9+70001)
        changed_probe = [(u, i, 5 if r < 4 else 1) for u, i, r in probe]
        first = prepare(context+probe, users, items, 9)
        second = prepare(context+changed_probe, users, items, 9)
        self.assertEqual(first['diagnostics']['teacher_coefficient_sha256'], second['diagnostics']['teacher_coefficient_sha256'])
        self.assertFalse(np.array_equal(first['target'], second['target']))
        np.testing.assert_array_equal(first['eligible'], second['eligible'])
        np.testing.assert_array_equal(first['observed'], second['observed'])
        for branch in first['branches']:
            np.testing.assert_array_equal(first['branches'][branch]['context'], second['branches'][branch]['context'])
        self.assertNotEqual(array_digest(first['target']), array_digest(second['target']))


if __name__ == '__main__':
    unittest.main()
