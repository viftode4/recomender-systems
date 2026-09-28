import unittest

import numpy as np

from exception_model import (constrained_linear_scores, fit_geometry, matched_pairs,
                             neighborhood_scores, pair_candidate_scores, pair_features,
                             permuted_partners, rating_matrices, shuffled_partners, unit_rows)
from exception_experiment import evaluate_ratings, recommendations


class ExceptionModelTests(unittest.TestCase):
    def test_training_rating_semantics_and_neutral_mask(self):
        signed, observed = rating_matrices(['u'], ['[PAD]','a','b','c'],
                                          [('u','a',5),('u','b',1),('u','c',3)])
        np.testing.assert_array_equal(signed, [[0,1,-1,0]])
        np.testing.assert_array_equal(observed, [[False,True,True,True]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            rating_matrices(['u'], ['[PAD]','a'], [('u','a',5),('u','a',1)])

    def test_geometry_is_train_only_finite_and_reproducible(self):
        signed = np.array([[0,1,-1,0,1],[0,-1,1,1,0],[0,1,0,-1,1],[0,0,1,-1,0]],float)
        genres = np.array([[0,0],[1,0],[1,0],[0,1],[1,1]],float)
        a = fit_geometry(signed,genres,2,42)
        b = fit_geometry(signed,genres,2,42)
        np.testing.assert_allclose(a,b)
        np.testing.assert_array_equal(a[0],0)
        self.assertTrue(np.isfinite(a).all())

    def test_constrained_ridge_matches_independent_reduced_feature_solution(self):
        rng = np.random.default_rng(42)
        x,y = rng.normal(size=(9,8)),rng.normal(size=(9,4))
        x[:,[0,4]],y[:,0] = 0,0
        excluded = np.stack([np.arange(4),4+np.arange(4)],axis=1)
        actual = constrained_linear_scores(x,y,excluded,.3)
        expected = np.zeros_like(y)
        for item in range(1,4):
            allowed = np.setdiff1d(np.arange(8),excluded[item])
            xx = x[:,allowed]
            coefficients = np.linalg.solve(xx.T@xx+.3*np.eye(len(allowed)),xx.T@y[:,item])
            expected[:,item] = xx@coefficients
        np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_candidate_gate_keeps_anchor_and_penalizes_disliked_side(self):
        geometry = unit_rows(np.array([[0.,0.],[1.,0.],[0.,1.],[.9,.1],[.1,.9]]))
        signed = np.array([[0,1,-1,0,0]])
        scores = pair_candidate_scores(geometry,[[(1,2)]],signed,'gate',.1)
        self.assertGreater(scores[0,3],scores[0,4])
        reversed_scores = pair_candidate_scores(geometry,[[(2,1)]],-signed,'gate',.1)
        self.assertGreater(reversed_scores[0,4],reversed_scores[0,3])
        anchors = pair_candidate_scores(geometry,[[(1,2)]],signed,'anchor')
        changed_negative = pair_candidate_scores(geometry,[[(1,4)]],signed,'anchor')
        np.testing.assert_array_equal(anchors,changed_negative)

    def test_pairing_requires_shared_genre_and_chooses_closest_dislike(self):
        geometry = unit_rows([[0,0],[1,0],[.8,.2],[0,1],[1,0]])
        genres = np.asarray([[0,0],[1,0],[1,0],[1,0],[0,1]])
        signed = np.asarray([[0,1,-1,-1,-1],[0,-1,1,0,0],[0,1,0,0,0]])
        self.assertEqual(matched_pairs(signed, geometry, genres), [[(1,2)],[(2,1)],[]])

    def test_relation_orientation_and_unordered_ablation(self):
        geometry = np.asarray([[0.,0.],[1.,0.],[0.,1.]])
        projection = np.random.default_rng(42).normal(size=(2,100))
        phase = np.random.default_rng(7).uniform(0,2*np.pi,size=100)
        a = pair_features([(1,2)], geometry, projection, phase, 'oriented')
        b = pair_features([(2,1)], geometry, projection, phase, 'oriented')
        self.assertGreater(np.linalg.norm(a-b), .2)
        a = pair_features([(1,2)], geometry, projection, phase, 'unordered')
        b = pair_features([(2,1)], geometry, projection, phase, 'unordered')
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(pair_features([],geometry,projection,phase,'oriented'),np.zeros(100))

    def test_partner_shuffle_never_uses_unobserved_or_other_user_dislikes(self):
        signed = np.array([[0,1,-1,-1,0],[0,-1,1,0,-1]])
        original = [[(1,2),(1,3)],[(2,1)]]
        shuffled = shuffled_partners(original,signed,99)
        self.assertEqual(shuffled,shuffled_partners(original,signed,99))
        for row,pairs in enumerate(shuffled):
            self.assertEqual([p for p,_ in pairs],[p for p,_ in original[row]])
            self.assertTrue(all(signed[row,d] == -1 for _,d in pairs))

    def test_exact_pairing_control_preserves_counterpart_reuse_counts(self):
        pairs=[[(1,4),(2,4),(3,5),(6,7)],[]]
        permuted=permuted_partners(pairs,99)
        self.assertEqual([p for p,_ in permuted[0]],[p for p,_ in pairs[0]])
        self.assertEqual(sorted(d for _,d in permuted[0]),sorted(d for _,d in pairs[0]))
        self.assertEqual(permuted[1],[])
        self.assertEqual(permuted,permuted_partners(pairs,99))

    def test_decoder_excludes_self_and_forbidden_pair_neighbors(self):
        signed = np.asarray([[0,1,0,0],[0,0,1,-1],[0,0,-1,1]],dtype=float)
        similarity = np.asarray([[999.,.9,.3],[.9,999.,.4],[.3,.4,999.]])
        scores = neighborhood_scores(similarity,signed,neighbors=1,dislike_weight=1)
        np.testing.assert_array_equal(scores[0],[0,0,1,-1])
        forbidden = np.zeros((3,3),dtype=bool)
        forbidden[0,1] = True
        excluded = neighborhood_scores(similarity,signed,1,1,forbidden)
        np.testing.assert_array_equal(excluded[0],[0,0,-1,1])

    def test_recommendations_mask_all_ratings_including_neutrals_and_padding(self):
        users,items = ['u'],['[PAD]','neutral','dislike','a','b']
        observed = np.array([[False,True,True,False,False]])
        scores = np.array([[100,99,98,1,2]])
        self.assertEqual(recommendations(scores,users,items,observed,2), {'u':['b','a']})

    def test_liked_metrics_use_explicit_user_and_dislike_denominators(self):
        recs = {'u':['a','b'],'v':['b','c']}
        valid = {('u','a'):5,('u','b'):1,('v','b'):2,('v','c'):3}
        result = evaluate_ratings(recs,{'u','v'},valid,{},['a','b','c'],{},2)
        self.assertEqual(result['liked_ratings']['users'],1)
        self.assertEqual(result['denominators']['users_without_liked_validation_excluded_from_liked_metrics'],1)
        self.assertEqual(result['known_dislike_rate_per_slot'],.5)
        self.assertEqual(result['denominators']['recommended_known_dislikes'],2)
        self.assertEqual(result['all_observed']['aggregate']['ndcg@2'],1)
        self.assertEqual(result['liked_ratings']['aggregate']['ndcg@2'],1)


if __name__ == '__main__':
    unittest.main()
