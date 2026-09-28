import unittest
import numpy as np
from study import partition_users, fit_ridge, jsd, normalize_scores, rerank, ranking, build_features, rrf


class StudyTests(unittest.TestCase):
    def test_user_partition_is_disjoint_stable_and_complete(self):
        users = [str(i) for i in range(10)]
        a, b = partition_users(users, 5)
        self.assertFalse(a & b)
        self.assertEqual(a | b, set(users))
        self.assertEqual((a, b), partition_users(users[::-1], 5))

    def test_regression_recovers_known_mapping_and_shrinks(self):
        x = np.arange(20.).reshape(-1, 1)
        y = 3*x[:, 0] + 7
        w, bias = fit_ridge(x, y, 1e-8)
        np.testing.assert_allclose(w, [3], atol=1e-6)
        self.assertAlmostEqual(bias, 7, places=5)
        strong, _ = fit_ridge(x, y, 100)
        self.assertLess(abs(strong[0]), abs(w[0]))

    def test_jsd_identical_disjoint_and_symmetric(self):
        self.assertAlmostEqual(float(jsd([1,0], [1,0])), 0)
        self.assertAlmostEqual(float(jsd([1,0], [0,1])), 1)
        self.assertAlmostEqual(float(jsd([.2,.8], [.7,.3])), float(jsd([.7,.3], [.2,.8])))

    def test_normalization_excludes_seen_padding(self):
        a = normalize_scores(np.array([[999., -999., 1., 3.]]), [np.array([2,3])])
        np.testing.assert_allclose(a, [[0,0,-1,1]])

    def test_context_features_have_expected_interactions(self):
        base = np.ones((2,3,2))
        f = build_features(base, np.array([2,3]), np.array([4,5]), np.array([6,7,8]), 'context')
        self.assertEqual(f.shape, (2,3,9))
        np.testing.assert_allclose(f[0,0], [1,1,2,2,4,4,6,6,0])

    def test_reranking_zero_weight_and_diversity_extreme(self):
        scores = np.array([999., 3., 2., 1.])
        eligible = np.array([1,2,3])
        genres = np.array([[0,0], [1,0], [1,0], [0,1]], dtype=float)
        head = np.array([False,True,False,False])
        for mode in ('diversity','calibration','exposure'):
            recs = rerank(scores, eligible, genres, genres, np.array([.5,.5]), head, 2, 0, mode)
            np.testing.assert_array_equal(recs, [1,2])
        recs = rerank(scores, eligible, genres, genres, np.array([.5,.5]), head, 2, 1, 'diversity')
        np.testing.assert_array_equal(recs, [1,3])
        self.assertEqual(len(set(recs)),2)

    def test_rrf_agrees_with_identical_experts(self):
        scores = np.array([[999.,3.,1.,2.]])
        eligible = [np.array([1,2,3])]
        fused = rrf({'a':scores,'b':scores}, eligible)
        np.testing.assert_array_equal(ranking(fused[0],eligible[0],3), [1,3,2])


if __name__ == '__main__':
    unittest.main()
