"""Synthetic aggregate-analysis tests; no actual held-out artifacts are read."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_frozen import (aggregate_comparisons, compare_one, digest,
                            load_completed, paired_bootstrap,
                            predefined_comparisons, write_report)


def fixture(seed=7):
    names = ['validation-best', 'test-best-unselected', 'context-ridge',
             'constrained-ridge', 'rrf', 'group-switch', 'group-utility-budget-exposure']
    bundle = {'status': 'frozen', 'test_read': False, 'seed': seed, 'k': 10,
              'expected_test_sha256': 'predeclared-test-hash',
              'selection': {'metric': 'ndcg@10', 'best_expert': 'validation-best',
                            'families': {'context': 'context-ridge', 'constrained': 'constrained-ridge'}},
              'models': {name: {} for name in names}}
    baseline = np.array([.1, .2, .3, .4])
    offsets = {'validation-best': 0., 'test-best-unselected': .5,
               'context-ridge': .03, 'constrained-ridge': .01, 'rrf': -.02,
               'group-switch': 0., 'group-utility-budget-exposure': -.01}
    users = ['private-ada', 'private-bob', 'private-cal', 'private-dee']
    results = {name: {'aggregate': {'ndcg@10': float((baseline + offsets[name]).mean()),
                                   'calibration_jsd': .1, 'tail_recall': None},
                      'per_user': {u: {'ndcg@10': float(value + offsets[name])}
                                   for u, value in zip(users, baseline)}} for name in names}
    return bundle, results


def write_fixture(root, seed=7):
    root = Path(root)
    frozen, evaluated = root / f'frozen-{seed}', root / f'results-{seed}'
    frozen.mkdir()
    evaluated.mkdir()
    bundle, results = fixture(seed)
    (frozen / 'freeze.json').write_text(json.dumps(bundle))
    freeze_hash = digest(frozen / 'freeze.json')
    (frozen / 'freeze.sha256').write_text(freeze_hash + '\n')
    (evaluated / 'results.json').write_text(json.dumps(results))
    (evaluated / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'test_model_selection': False, 'k': 10,
        'freeze_sha256': freeze_hash, 'test_sha256': bundle['expected_test_sha256'],
        'output_sha256': {'results.json': digest(evaluated / 'results.json')}}))
    return frozen, evaluated


class FrozenComparisonTests(unittest.TestCase):
    def test_pairing_recovers_constant_gain_and_bonferroni_is_wider(self):
        differences = np.array([[.03, -.3], [.03, -.1], [.03, .1], [.03, .3]])
        result = paired_bootstrap(differences, seed=3)
        np.testing.assert_allclose(result['bonferroni_intervals'][0], [.03, .03])
        ordinary = result['pointwise_intervals'][1]
        adjusted = result['bonferroni_intervals'][1]
        self.assertLessEqual(adjusted[0], ordinary[0])
        self.assertGreaterEqual(adjusted[1], ordinary[1])
        self.assertGreaterEqual(result['resamples'], 20000)
        self.assertGreaterEqual(result['expected_draws_per_adjusted_tail'], 20)
        self.assertEqual(result, paired_bootstrap(differences, seed=3))

    def test_rejects_inadequate_resampling_and_nonpaired_data(self):
        with self.assertRaisesRegex(ValueError, '20000'):
            paired_bootstrap([.1, .2], resamples=100)
        with self.assertRaisesRegex(ValueError, 'two paired'):
            paired_bootstrap([.1])

    def test_winner_is_frozen_even_when_other_expert_dominates_test(self):
        bundle, results = fixture()
        comparisons = predefined_comparisons(bundle)
        self.assertEqual(comparisons['hybrid:context']['reference'], 'validation-best')
        report = compare_one(bundle, results)
        self.assertAlmostEqual(report['comparisons']['hybrid:context']['mean_ndcg_difference'], .03)
        self.assertEqual(report['comparisons']['policy:group-utility-budget-exposure']['reference'], 'context-ridge')
        np.testing.assert_allclose(report['comparisons']['policy:group-utility-budget-exposure']['bonferroni_interval'], [-.04, -.04])

    def test_missing_user_or_wrong_aggregate_rejected(self):
        bundle, results = fixture()
        del results['context-ridge']['per_user']['private-ada']
        with self.assertRaisesRegex(ValueError, 'same paired users'):
            compare_one(bundle, results)
        bundle, results = fixture()
        results['context-ridge']['aggregate']['ndcg@10'] = .99
        with self.assertRaisesRegex(ValueError, 'Aggregate NDCG'):
            compare_one(bundle, results)

    def test_cross_seed_report_has_no_independence_assuming_interval(self):
        reports = [compare_one(*fixture(seed)) for seed in (7, 8, 9)]
        aggregate = aggregate_comparisons(reports)
        self.assertEqual(aggregate['seeds'], [7, 8, 9])
        self.assertAlmostEqual(aggregate['comparisons']['hybrid:context']['mean_ndcg_difference'], .03)
        self.assertIsNone(aggregate['cross_seed_confidence_interval'])
        self.assertEqual(aggregate['comparisons']['hybrid:context']['metric_seed_counts_candidate_metrics']['tail_recall'], 0)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            aggregate_comparisons([reports[0], reports[0]])

    def test_provenance_validation_and_export_excludes_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = [write_fixture(root, seed) for seed in (7, 8, 9)]
            out = root / 'shareable'
            report = write_report([f for f, _ in pairs], [r for _, r in pairs], out)
            self.assertEqual(report['seeds'], [7, 8, 9])
            for path in out.iterdir():
                self.assertNotIn('private-', path.read_text())
                self.assertNotIn('per_user', path.read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertFalse(manifest['individual_user_data_exported'])
            self.assertFalse(manifest['test_interactions_opened'])
            self.assertEqual(set(manifest['output_sha256']),
                             {'aggregate.json', 'per-seed-comparisons.json', 'SUMMARY.md'})
            with self.assertRaises(FileExistsError):
                write_report([pairs[0][0]], [pairs[0][1]], out)
            result_file = pairs[0][1] / 'results.json'
            result_file.write_text(result_file.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'Results hash'):
                load_completed(*pairs[0])

    def test_freeze_and_evaluation_must_correspond(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = write_fixture(directory, 7), write_fixture(directory, 8)
            with self.assertRaisesRegex(ValueError, 'different frozen bundle'):
                load_completed(first[0], second[1])


if __name__ == '__main__':
    unittest.main()
