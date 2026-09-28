import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from compare_fields_final import (ENDPOINTS, REFERENCES, TRACKS, aggregate_reports,
    compare_seed, digest, load_completed, write_json, write_report)


USERS = ['private-user-a', 'private-user-b', 'private-user-c', 'private-user-d']


def ranked(offset=0):
    result = {'stage': 'test', 'denominators': {
        'all_observed_users': 4, 'liked_ratings_users': 3, 'test_observations': 12,
        'test_likes': 5, 'test_dislikes': 3, 'recommendation_slots': 40}}
    for endpoint in ENDPOINTS:
        users = USERS if endpoint == 'all_observed' else USERS[:3]
        values = [.1 + .15*i + offset for i in range(len(users))]
        result[endpoint] = {'per_user': {u: {'ndcg@10': v, 'recall@10': v/2}
                                             for u, v in zip(users, values)},
                            'aggregate': {'ndcg@10': float(np.mean(values)),
                                          'recall@10': float(np.mean(values)/2)}}
    return result


def seed_record(kind, seed=2026):
    models = list(REFERENCES) if kind == 'references' else ['adaptive', 'fixed_flow']
    bundle = {'status': 'frozen', 'seed': seed, 'k': 10, 'models': models,
        'test_read': False, 'test_evaluated': False,
        'data_sha256': {'ml.inter': 'data'},
        'split_sha256': {'train': f'train-{seed}', 'valid': f'valid-{seed}', 'test': f'test-{seed}'},
        'expected_test_sha256': f'test-{seed}',
        'selections': {name: {'epoch': 60, 'selection_cohort': 'meta_fit'} for name in models}}
    if kind == 'references':
        bundle['selections']['PositiveEASE']['selection_objective'] = 'liked_ratings'
    metrics = {}
    for name in models:
        offset = .08 if name == 'adaptive' else .02
        if kind == 'references':
            metrics[name] = ranked(offset)
        elif kind == 'conditional':
            metrics[name] = {'ranking': ranked(offset)}
        else:
            metrics[name] = {'ranking': {'all_observed_adapter': ranked(offset),
                                        'liked_record_adapter': ranked(offset + .01)}}
    return {'bundle': bundle, 'metrics': metrics,
            'provenance': {'bundle_sha256': 'bundle', 'metrics_sha256': 'metrics'}}


def experiments(seed=2026):
    return {kind: {'kind': kind, 'seeds': {seed: seed_record(kind, seed)}}
            for kind in (*TRACKS, 'references')}


def write_fixture(root, kind, seed=2026):
    record = seed_record(kind, seed)
    frozen, results = root / kind / 'frozen', root / kind / 'results'
    (frozen / str(seed)).mkdir(parents=True)
    (results / str(seed)).mkdir(parents=True)
    bundle = record['bundle']
    if kind != 'references':
        protocol = {'training_objective': 'categorical_cross_entropy' if kind == 'conditional'
                    else 'joint_recorded_item_rating_multinomial',
                    'epochs': 400 if kind == 'joint400' else 100, 'seeds': [seed]}
        if kind == 'joint400':
            protocol['study_kind'] = 'post_v1_convergence_sensitivity'
        write_json(frozen / str(seed) / 'protocol.json', protocol)
        bundle['source_protocol_sha256'] = digest(frozen / str(seed) / 'protocol.json')
        bundle['payload_sha256'] = {'protocol.json': bundle['source_protocol_sha256']}
    write_json(frozen / str(seed) / 'bundle.json', bundle)
    manifest = {'status': 'frozen', 'test_read': False, 'test_evaluated': False,
        'seeds': [seed], 'code_sha256': {'evaluator.py': 'sealed-code'},
        'bundle_sha256': {str(seed): digest(frozen / str(seed) / 'bundle.json')}}
    write_json(frozen / 'manifest.json', manifest)
    (frozen / 'manifest.sha256').write_text(digest(frozen / 'manifest.json') + '\n')
    write_json(results / str(seed) / 'metrics.json', record['metrics'])
    evaluated = {'status': 'complete', 'test_read': True, 'test_evaluated': True,
        'selection_after_test': False, 'code_sha256': manifest['code_sha256'],
        'frozen_manifest_sha256': digest(frozen / 'manifest.json'),
        'output_sha256': {f'{seed}/metrics.json': digest(results / str(seed) / 'metrics.json')}}
    write_json(results / 'manifest.json', evaluated)
    write_json(frozen / 'TEST-OPENED.json', {'status': 'test_evaluated',
               'evaluation_manifest_sha256': digest(results / 'manifest.json')})
    return frozen, results


class FinalFieldComparisonTests(unittest.TestCase):
    def test_fixed_24_comparisons_use_correct_adapters_and_denominators(self):
        report = compare_seed(experiments(), 2026)
        self.assertEqual(len(report['comparisons']), 24)
        self.assertEqual(report['bootstrap']['family_comparisons'], 24)
        self.assertGreaterEqual(report['bootstrap']['resamples_per_endpoint'], 20_000)
        self.assertGreaterEqual(report['bootstrap']['expected_draws_per_adjusted_tail'], 20)
        for name, row in report['comparisons'].items():
            self.assertEqual(row['users'], 3 if '/liked_ratings/' in name else 4)
            expected = .07 if name.startswith(('joint100/liked', 'joint400/liked')) and row['reference'] != 'fixed_flow' else .06
            self.assertAlmostEqual(row['mean_ndcg_difference'], expected)
            low, high = row['bonferroni_family_interval_95']
            ordinary_low, ordinary_high = row['pointwise_interval_95']
            self.assertLessEqual(low, ordinary_low)
            self.assertGreaterEqual(high, ordinary_high)

    def test_real_pairing_preserved_when_values_vary(self):
        data = experiments()
        row = data['conditional']['seeds'][2026]['metrics']['adaptive']['ranking']['all_observed']
        values = [.7, .1, .2, .4]
        for user, value in zip(USERS, values):
            row['per_user'][user]['ndcg@10'] = value
        row['aggregate']['ndcg@10'] = np.mean(values)
        report = compare_seed(data, 2026)
        compared = report['comparisons']['conditional/all_observed/adaptive-minus-EASE']
        expected = np.array(values) - np.array([.12, .27, .42, .57])
        self.assertAlmostEqual(compared['mean_ndcg_difference'], expected.mean())
        self.assertEqual(compared['fraction_users_improved'], .25)
        self.assertLess(compared['bonferroni_family_interval_95'][0], 0)
        self.assertGreater(compared['bonferroni_family_interval_95'][1], 0)

    def test_changed_cohort_denominator_or_split_is_rejected(self):
        data = experiments()
        del data['references']['seeds'][2026]['metrics']['EASE']['liked_ratings']['per_user'][USERS[0]]
        with self.assertRaisesRegex(ValueError, 'denominator'):
            compare_seed(data, 2026)
        data = experiments()
        data['joint100']['seeds'][2026]['bundle']['split_sha256']['test'] = 'other'
        with self.assertRaisesRegex(ValueError, 'split_sha256'):
            compare_seed(data, 2026)
        data = experiments()
        data['references']['seeds'][2026]['metrics']['EASE']['denominators']['test_likes'] += 1
        with self.assertRaisesRegex(ValueError, 'denominators'):
            compare_seed(data, 2026)

    def test_bad_aggregate_does_not_pass_as_metric_evidence(self):
        data = experiments()
        data['references']['seeds'][2026]['metrics']['EASE']['all_observed']['aggregate']['ndcg@10'] = .99
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            compare_seed(data, 2026)
        with self.assertRaisesRegex(ValueError, 'three'):
            compare_seed({'references': data['references']}, 2026)

    def test_report_never_reads_interactions_or_exports_individual_users(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {kind: write_fixture(root, kind) for kind in (*TRACKS, 'references')}
            original_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.suffix in ('.tsv', '.npz', '.pt', '.inter'):
                    raise AssertionError('Reporter attempted forbidden input access')
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded):
                aggregate = write_report(paths, root / 'shareable')
            self.assertIsNone(aggregate['cross_seed_confidence_interval'])
            for path in (root / 'shareable').iterdir():
                text = path.read_text()
                self.assertFalse(any(user in text for user in USERS))
            manifest = json.loads((root / 'shareable' / 'manifest.json').read_text())
            self.assertFalse(manifest['test_interactions_opened'])
            self.assertFalse(manifest['selection_performed'])

    def test_tampered_metrics_and_mislabeled_experiment_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, results = write_fixture(root, 'joint100')
            with self.assertRaisesRegex(ValueError, 'epoch budget'):
                load_completed(frozen, results, 'joint400')
            path = results / '2026' / 'metrics.json'
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'metrics hash'):
                load_completed(frozen, results, 'joint100')

    def test_overlapping_seed_summary_has_no_pooled_interval(self):
        first = compare_seed(experiments(), 2026)
        second = copy.deepcopy(first)
        second['seed'] = 2027
        result = aggregate_reports([first, second])
        self.assertIsNone(result['cross_seed_confidence_interval'])
        self.assertEqual(result['seeds'], [2026, 2027])
        with self.assertRaisesRegex(ValueError, 'unique'):
            aggregate_reports([first, first])


if __name__ == '__main__':
    unittest.main()
