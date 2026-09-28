"""Reporting guards against silently misdescribing evaluation cohorts."""
import unittest

from summarize import aggregate_policy, cohort_summary, mean_available


class SummaryContentTests(unittest.TestCase):
    def test_legacy_two_cohort_evidence_remains_explicit(self):
        result = cohort_summary({'meta_fit_users': 2, 'development_users': 3},
                                {'meta_fit': ['a', 'b'], 'development': ['c', 'd', 'e']})
        self.assertEqual(result, {'meta_fit': 2, 'development': 3, 'calibration': 0})

    def test_three_cohort_counts_come_from_verified_artifacts(self):
        manifest = {'meta_fit_users': 2, 'development_users': 1, 'calibration_users': 2}
        cohorts = {'meta_fit': ['a', 'b'], 'development': ['c'], 'calibration': ['d', 'e']}
        self.assertEqual(cohort_summary(manifest, cohorts)['calibration'], 2)
        cohorts['calibration'] = ['b', 'e']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            cohort_summary(manifest, cohorts)

    def test_stale_manifest_cannot_supply_false_cohort_description(self):
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            cohort_summary({'meta_fit_users': 2, 'development_users': 2},
                           {'meta_fit': ['a', 'b'], 'development': ['c']})

    def test_missing_metric_is_not_zero_filled(self):
        self.assertIsNone(mean_available([{'ndcg@10': .2}], 'popularity_jsd'))
        self.assertAlmostEqual(mean_available([{'ndcg@10': .2}, {'ndcg@10': .4}], 'ndcg@10'), .3)

    def test_curated_policy_excludes_individual_ids_without_mutating_source(self):
        raw = {'strengths': {'0': .2}, 'development_retention': {'0': .98},
               'independent_calibration': {'calibration_users': ['person-a', 'person-b'],
                   'per_user': {'person-a': .3}, 'groups': {'0': {'users': 2}},
                   'bound_method': 'bootstrap'}}
        clean = aggregate_policy(raw)
        details = clean['independent_calibration']
        self.assertNotIn('calibration_users', details)
        self.assertNotIn('per_user', details)
        self.assertEqual(details['calibration_user_count'], 2)
        self.assertEqual(details['groups'], {'0': {'users': 2}})
        self.assertEqual(clean['strengths'], raw['strengths'])
        self.assertEqual(len(raw['independent_calibration']['calibration_users']), 2)


if __name__ == '__main__':
    unittest.main()
