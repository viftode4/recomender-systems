import unittest
from pathlib import Path
import tempfile
import contextlib
import io
import json
from unittest.mock import patch

import numpy as np

from societal import (apply_group_policy, calibration_diagnostics,
                      candidate_quota_bounds, discounted_exposure_metrics,
                      fit_group_policy, group_diagnostics,
                      item_exposure_diagnostics, popularity_calibration_rerank,
                      training_taste_groups, user_popularity_deviation)
from audit_study import audit_models, load_training_ratings, load_final_audit_inputs, main as audit_main
from study import digest


class IndependentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.users = ['a', 'b', 'c', 'd']
        self.groups = {'a': 'sparse', 'b': 'sparse', 'c': 'active', 'd': 'active'}
        self.utility = {'base': dict.fromkeys(self.users, .5),
                        'first': {'a': .7, 'b': .7, 'c': .2, 'd': .2},
                        'second': {'a': .1, 'b': .1, 'c': .8, 'd': .8}}
        self.kwargs = {'fit_users': ['fit'], 'selection_users': ['select'], 'baseline': 'base',
                       'min_group_size': 2, 'bootstrap': 1000, 'seed': 9}

    def test_calibration_must_be_disjoint_from_fit_and_selection(self):
        for key in ('fit_users', 'selection_users'):
            arguments = dict(self.kwargs, **{key: ['a']})
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                fit_group_policy(self.utility, self.groups, self.users, **arguments)

    def test_user_policy_selects_different_options_and_applies_without_labels(self):
        policy = fit_group_policy(self.utility, self.groups, self.users, **self.kwargs)
        self.assertEqual(policy['policy'], {'sparse': 'first', 'active': 'second'})
        options = {option: {u: [option] for u in self.users} for option in self.utility}
        recs = apply_group_policy(policy, self.groups, options)
        self.assertEqual(recs['a'], ['first'])
        self.assertEqual(recs['d'], ['second'])

    def test_no_audit_label_dependency(self):
        self.groups['audit'] = 'sparse'
        for option in self.utility:
            self.utility[option]['audit'] = 0.
        a = fit_group_policy(self.utility, self.groups, self.users, **self.kwargs)
        self.utility['first']['audit'] = 1.
        b = fit_group_policy(self.utility, self.groups, self.users, **self.kwargs)
        self.assertEqual(a, b)

    def test_pairing_retains_constant_difference_despite_variable_utilities(self):
        utility = {'base': {'a': .1, 'b': .8}, 'candidate': {'a': .15, 'b': .85}}
        policy = fit_group_policy(utility, {'a': 'g', 'b': 'g'}, ['a', 'b'], **self.kwargs)
        interval = policy['groups']['g']['candidates']['candidate']['paired_difference_95_percent_interval']
        np.testing.assert_allclose(interval, [.05, .05])

    def test_harmful_low_cost_option_is_rejected(self):
        cost = {'base': dict.fromkeys(self.users, 1.),
                'first': dict.fromkeys(self.users, .5),
                'second': dict.fromkeys(self.users, 0.)}
        result = fit_group_policy(self.utility, self.groups, self.users,
                                  objective='exposure', secondary=cost, **self.kwargs)
        self.assertEqual(result['policy']['sparse'], 'first')
        self.assertFalse(result['groups']['sparse']['candidates']['second']['admissible'])

    def test_insufficient_or_absent_groups_fall_back_to_baseline(self):
        groups = dict(self.groups, unseen='new')
        result = fit_group_policy(self.utility, groups, self.users,
                                  **dict(self.kwargs, min_group_size=3))
        self.assertEqual(set(result['policy'].values()), {'base'})
        self.assertEqual(result['groups']['new']['users'], 0)

    def test_hoeffding_does_not_certify_tiny_sample_improvement(self):
        result = fit_group_policy(self.utility, self.groups, self.users,
                                  bound='hoeffding', **self.kwargs)
        self.assertEqual(set(result['policy'].values()), {'base'})

    def test_bounded_utility_and_complete_label_checks(self):
        self.utility['base']['a'] = 1.01
        with self.assertRaisesRegex(ValueError, 'bounded'):
            fit_group_policy(self.utility, self.groups, self.users, **self.kwargs)
        del self.utility['base']['a']
        with self.assertRaisesRegex(ValueError, 'Missing'):
            fit_group_policy(self.utility, self.groups, self.users, **self.kwargs)


class SocietalDiagnosticsTests(unittest.TestCase):
    def test_final_sidecar_uses_only_train_context_and_exports_aggregate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen, evaluated, data = root / 'frozen', root / 'evaluated', root / 'data'
            for path in (frozen, evaluated, data):
                path.mkdir()
            users, items = ['private-u', 'private-v'], ['[PAD]', 'a', 'b', 'c']
            (frozen / 'train.tsv').write_text('user_id\titem_id\nprivate-u\ta\nprivate-v\tb\n')
            (data / 'sample.inter').write_text('user_id:token\titem_id:token\trating:float\n'
                                             'private-u\ta\t5\nprivate-v\tb\t4\n'
                                             'private-u\tc\tDO_NOT_PARSE\n')
            (data / 'sample.item').write_text('item_id:token\tclass:token_seq\na\tA\nb\tB\nc\tA B\n')
            np.savez_compressed(frozen / 'frozen.npz', users=users, items=items)
            signature = {p.name: digest(p) for p in data.iterdir()}
            bundle = {'status': 'frozen', 'test_read': False, 'k': 1,
                      'expected_test_sha256': 'unopened-test-hash',
                      'selection': {'best_expert': 'base'},
                      'sources': {'base': {'manifest': {'data_sha256': signature}}},
                      'artifacts_sha256': {p.name: digest(p) for p in frozen.iterdir()}}
            (frozen / 'freeze.json').write_text(json.dumps(bundle))
            (frozen / 'freeze.sha256').write_text(digest(frozen / 'freeze.json'))
            results = {'base': {'per_user': {'private-u': {'ndcg@1': 1.}, 'private-v': {'ndcg@1': 0.}},
                                'aggregate': {'ndcg@1': .5}}}
            (evaluated / 'results.json').write_text(json.dumps(results))
            (evaluated / 'recommendations.json').write_text(json.dumps({'base': {'private-u': ['b'], 'private-v': ['a']}}))
            (evaluated / 'manifest.json').write_text(json.dumps({
                'status': 'complete', 'test_model_selection': False, 'k': 1,
                'freeze_sha256': digest(frozen / 'freeze.json'), 'test_sha256': 'unopened-test-hash',
                'output_sha256': {p.name: digest(p) for p in evaluated.iterdir()}}))
            original_open = Path.open
            def forbid_labels(path, *args, **kwargs):
                if path.name in ('test.tsv', 'valid.tsv'):
                    raise AssertionError('Sidecar opened held-out interaction labels')
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', forbid_labels):
                inputs = load_final_audit_inputs(frozen, evaluated, data)
                self.assertEqual(inputs['ratings'], [('private-u', 1, 5.), ('private-v', 2, 4.)])
                out = root / 'audit'
                argv = ['audit_study.py', '--final', str(evaluated), '--frozen', str(frozen),
                        '--data', str(data), '--out', str(out), '--aggregate-only']
                with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    audit_main()
            for path in out.iterdir():
                self.assertNotIn('private-', path.read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertTrue(manifest['test_derived_metrics_read'])
            self.assertFalse(manifest['test_interactions_opened'])
            self.assertFalse(manifest['individual_user_data_exported'])
            self.assertEqual(len(manifest['output_sha256']), 3)
            (evaluated / 'recommendations.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'recommendations differ'):
                load_final_audit_inputs(frozen, evaluated, data)

    def test_rating_join_ignores_nontraining_values_and_requires_complete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.inter'
            path.write_text('user_id:token\titem_id:token\trating:float\n'
                            'u\ti\t5\n'
                            'u\theld-out\tDO_NOT_PARSE\n')
            joined = load_training_ratings(path, [('u', 'i')], {'i': 1})
            self.assertEqual(joined, [('u', 1, 5.)])
            with self.assertRaisesRegex(ValueError, 'missing a training'):
                load_training_ratings(path, [('u', 'absent')], {'absent': 1})

    def test_sidecar_audit_preserves_user_ids_and_separates_cost_from_utility(self):
        items = ['PAD', 'train', 'candidate']
        genres = np.array([[0., 0.], [1., 0.], [0., 1.]])
        recs = {'base': {'u': ['candidate']}}
        results = {'base': {'per_user': {'u': {'ndcg@1': 1.}}}}
        audit = audit_models(recs, results, ['u'], items, [('u', 1, 5.)], genres,
                             baseline='base', k=1)
        self.assertEqual(audit['models']['base']['groups']['activity']['worst_group_utility'], 1.)
        self.assertEqual(audit['models']['base']['calibration']['aggregate']['jsd_liked_history']['mean'], 1.)
        self.assertEqual(audit['training_groups']['profiles']['u'], [1., 0.])

    def test_training_contradictions_use_rejections_and_remove_own_like(self):
        genres = np.asarray([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]], dtype=float)
        ratings = [('u', 1, 5.), ('u', 2, 5.), ('u', 3, 4.),
                   ('u', 4, 1.), ('u', 5, 2.), ('u', 6, 1.)]
        result = training_taste_groups(['u', 'cold'], ratings, genres)
        self.assertAlmostEqual(result['features']['u']['contradictory_like_rate'], 1 / 3)
        self.assertEqual(result['groups']['contradiction']['u'], 'has-contradictory-likes')
        self.assertEqual(result['groups']['contradiction']['cold'], 'insufficient-evidence')
        self.assertIsNone(result['profiles']['cold'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            training_taste_groups(['u'], ratings + [ratings[0]], genres)

    def test_genre_calibration_distinguishes_liked_and_all_history(self):
        genres = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=float)
        result = calibration_diagnostics({'u': [1, 2]}, genres, {'u': [.5, .5]}, {'u': [1., 0.]})
        row = result['per_user']['u']
        self.assertEqual(row['jsd_all_history'], 0.)
        self.assertGreater(row['jsd_liked_history'], 0.)
        self.assertGreater(row['discounted_jsd_all_history'], 0.)
        self.assertLess(row['discounted_jsd_liked_history'], row['jsd_liked_history'])

    def test_missing_like_profile_is_counted_as_missing_not_perfect(self):
        genres = np.eye(2)
        result = calibration_diagnostics({'u': [0], 'v': [1]}, genres,
                                         {'u': [1., 0.], 'v': [0., 1.]},
                                         {'u': None, 'v': [0., 1.]})
        self.assertEqual(result['aggregate']['jsd_liked_history']['users'], 1)
        self.assertNotIn('jsd_liked_history', result['per_user']['u'])

    def test_group_utility_reports_harm_and_zero_reference_without_infinity(self):
        current = {'u': {'ndcg@10': .2}, 'v': {'ndcg@10': .4}, 'w': {'ndcg@10': .1}}
        reference = {'u': {'ndcg@10': .3}, 'v': {'ndcg@10': .3}, 'w': {'ndcg@10': 0.}}
        result = group_diagnostics(current, {'u': 'a', 'v': 'a', 'w': 'b'}, reference=reference)
        self.assertEqual(result['worst_group_utility'], .1)
        self.assertEqual(result['groups']['a']['fraction_harmed'], .5)
        self.assertIsNone(result['groups']['b']['retention'])
        self.assertIsNone(result['groups']['b']['standard_error'])

    def test_quota_checks_both_head_and_tail_capacity(self):
        all_tail = candidate_quota_bounds([1, 2, 3], {4}, 2, target_head_count=1)
        self.assertFalse(all_tail['target_feasible'])
        self.assertEqual(all_tail['max_head_count'], 0)
        all_head = candidate_quota_bounds([1, 2, 3], {1, 2, 3}, 2, target_head_count=1)
        self.assertFalse(all_head['target_feasible'])
        self.assertEqual(all_head['min_head_count'], 2)

    def test_unexposed_items_contribute_to_gini_and_entropy(self):
        result = discounted_exposure_metrics({'u': ['a']}, ['a', 'b', 'c', 'd'],
                                             {'a': 'head', 'b': 'tail', 'c': 'tail', 'd': 'tail'})
        self.assertEqual(result['gini'], .75)
        self.assertEqual(result['entropy_bits'], 0.)
        self.assertEqual(result['groups']['head']['mean_item_exposure'], 1.)
        self.assertEqual(result['groups']['tail']['mean_item_exposure'], 0.)
        simple = item_exposure_diagnostics({'u': np.array([1])}, [1, 2, 3, 4], {1})
        self.assertEqual(simple['discounted_exposure_gini'], .75)

    def test_group_size_normalization_and_position_discount(self):
        recs = {str(i): list(np.roll(np.arange(4), i)) for i in range(4)}
        result = discounted_exposure_metrics(recs, range(4), {0: 'one', 1: 'three', 2: 'three', 3: 'three'})
        self.assertAlmostEqual(result['gini'], 0.)
        self.assertAlmostEqual(result['normalized_entropy'], 1.)
        self.assertAlmostEqual(result['group_mean_exposure_gap'], 0.)
        self.assertAlmostEqual(result['groups']['one']['fraction_total_exposure'], .25)
        expected = sum(1 / np.log2(np.arange(4) + 2)) / 4
        self.assertAlmostEqual(result['groups']['one']['mean_item_exposure'], expected)

    def test_user_popularity_reranker_respects_personal_target(self):
        scores = np.array([999., 4., 3., 2., 1.])
        eligible = np.array([1, 2, 3, 4])
        groups = np.array([0, 0, 0, 1, 1])
        baseline = popularity_calibration_rerank(scores, eligible, groups, [0., 1.], 2, 0.)
        calibrated = popularity_calibration_rerank(scores, eligible, groups, [0., 1.], 2, 1.)
        np.testing.assert_array_equal(baseline, [1, 2])
        np.testing.assert_array_equal(calibrated, [3, 4])
        base_metric = user_popularity_deviation({'u': baseline}, groups, {'u': [0., 1.]})
        calibrated_metric = user_popularity_deviation({'u': calibrated}, groups, {'u': [0., 1.]})
        self.assertEqual(base_metric['mean_jsd'], 1.)
        self.assertEqual(calibrated_metric['mean_jsd'], 0.)
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            popularity_calibration_rerank(scores, eligible, groups, [.2, .3], 2, 1.)


if __name__ == '__main__':
    unittest.main()
