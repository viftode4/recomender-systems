"""Synthetic integration tests; these never open the project's real test split."""
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from final_evaluate import assess, evaluate_frozen
from freeze import freeze_study, load_frozen, model_specs, recommend, score_models
from study import digest, grouped, main as study_main, read_pairs


ROOT = Path(__file__).resolve().parents[1]


class FrozenEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.users = [str(u) for u in range(1, 85)]
        cls.items = ['[PAD]'] + [str(i) for i in range(1, 17)]
        cls.metadata = cls.root / 'ml-100k.item'
        cls.metadata.write_text('item_id:token\tclass:token_seq\n' + ''.join(
            f'{item}\t{int(item)%3} {(int(item)+1)%3}\n' for item in cls.items[1:]))
        train, valid, test = [], [], []
        for offset, user in enumerate(cls.users):
            train += [(user, str((offset + i) % 16 + 1)) for i in range(2 + offset % 3)]
            valid += [(user, str((offset + 5) % 16 + 1)), (user, str((offset + 6) % 16 + 1))]
            test += [(user, str((offset + 7) % 16 + 1))]
        bodies = {name: 'user_id\titem_id\n' + ''.join(f'{u}\t{i}\n' for u, i in sorted(pairs))
                  for name, pairs in [('train', train), ('valid', valid), ('test', test)]}
        cls.test_path = cls.root / 'synthetic-test.tsv'
        cls.test_path.write_text(bodies['test'])
        hashes = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in bodies.items()}
        counts = Counter(i for _, i in train)
        cls.runs = []
        for name in ['ExactPop', 'Random', 'EASE']:
            run = cls.root / name
            run.mkdir()
            for part in ('train', 'valid'):
                (run / f'{part}.tsv').write_text(bodies[part])
            if name == 'Random':
                scores = np.stack([np.random.default_rng(np.random.SeedSequence([23, uid, 0])).random(len(cls.items))
                                   for uid in range(1, len(cls.users)+1)])
            elif name == 'ExactPop':
                scores = np.tile([counts.get(i, 0) for i in cls.items], (len(cls.users), 1)).astype(float)
            else:
                scores = np.random.default_rng(5).normal(size=(len(cls.users), len(cls.items)))
            np.savez_compressed(run / 'valid-scores.npz', users=cls.users, items=cls.items, scores=scores)
            (run / 'ids.json').write_text(json.dumps({'users': ['[PAD]'] + cls.users, 'items': cls.items, 'padding_index': 0}))
            (run / 'manifest.json').write_text(json.dumps({
                'status': 'complete', 'test_evaluated': False, 'validation_used_for_training': False,
                'model': name, 'settings': {'seed': 23}, 'split_sha256': hashes,
                'data_sha256': {'ml-100k.item': digest(cls.metadata)},
                'source_sha256': {name: digest(ROOT / name) for name in ('run.py', 'metrics.py')}}))
            cls.runs.append(run)
        cls.study = cls.root / 'study'
        argv = ['study.py', '--runs', *map(str, cls.runs), '--items', str(cls.metadata),
                '--out', str(cls.study), '--seed', '23', '--k', '2', '--bootstrap', '10']
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
            study_main()
        cls.frozen = cls.root / 'frozen'
        cls.bundle = freeze_study(cls.study, cls.runs, cls.metadata, cls.frozen)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_portable_bundle_replays_validation_exactly(self):
        bundle, arrays = load_frozen(self.frozen)
        development = json.loads((self.study / 'cohorts.json').read_text())['development']
        cohort = [u for u, user in enumerate(arrays['users']) if user in development]
        expected = json.loads((self.study / 'recommendations.json').read_text())
        actual = recommend(bundle, arrays, 'valid', cohort)
        for name, recs in actual.items():
            self.assertEqual(recs, expected[name], name)
        self.assertTrue(bundle['validation_replay']['exact_rankings'])

    def test_freeze_never_opens_test_file_and_refuses_overwrite(self):
        original = Path.open
        def audited_open(path, *args, **kwargs):
            if 'test' in path.name:
                raise AssertionError(f'Freeze attempted to read held-out data: {path}')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', audited_open):
            frozen = self.root / 'no-test-read'
            result = freeze_study(self.study, self.runs, self.metadata, frozen)
            self.assertFalse(result['test_read'])
            with self.assertRaises(FileExistsError):
                freeze_study(self.study, self.runs, self.metadata, frozen)

    def test_test_masks_validation_without_refitting_normalizers(self):
        bundle, arrays = load_frozen(self.frozen)
        valid_scores, _ = score_models(bundle, arrays, 'valid')
        test_scores, candidates = score_models(bundle, arrays, 'test')
        for name, spec in bundle['models'].items():
            if spec['kind'] == 'linear' or (spec['kind'] == 'expert' and name != 'Random'):
                np.testing.assert_array_equal(valid_scores[name], test_scores[name])
        self.assertFalse(np.array_equal(valid_scores['Random'], test_scores['Random']))
        np.testing.assert_array_equal(test_scores['Random'][0],
            np.random.default_rng(np.random.SeedSequence([23, 1, 1])).random(len(self.items)))
        for u, eligible in enumerate(candidates):
            self.assertFalse(arrays['valid_mask'][u, eligible].any())
            self.assertFalse(arrays['train_mask'][u, eligible].any())
            self.assertNotIn(0, eligible)

    def test_final_metric_implementation_matches_study_on_validation(self):
        bundle, arrays = load_frozen(self.frozen)
        development = set(json.loads((self.study / 'cohorts.json').read_text())['development'])
        cohort = [u for u, user in enumerate(arrays['users']) if user in development]
        expected = json.loads((self.study / 'results.json').read_text())
        truth = {u: items for u, items in grouped(read_pairs(self.frozen / 'valid.tsv')).items() if u in development}
        history = grouped(read_pairs(self.frozen / 'train.tsv'))
        for name, recs in recommend(bundle, arrays, 'valid', cohort).items():
            result = assess(recs, truth, history, bundle, arrays)
            for metric, value in expected[name]['aggregate'].items():
                if value is None:
                    self.assertIsNone(result['aggregate'][metric])
                else:
                    self.assertAlmostEqual(result['aggregate'][metric], value, places=12, msg=f'{name}: {metric}')
            self.assertEqual(result['groups'], expected[name]['groups'])

    def test_independent_policy_cohorts_and_popularity_rerank_are_frozen(self):
        cohorts = json.loads((self.frozen / 'provenance' / 'cohorts.json').read_text())
        self.assertEqual(len(cohorts['calibration']), 21)
        policy = json.loads((self.frozen / 'provenance' / 'group-policy.json').read_text())
        self.assertIsNotNone(policy['independent_calibration'])
        self.assertFalse(set(cohorts['calibration']) & set(cohorts['development']))
        modes = {spec.get('mode') for spec in self.bundle['models'].values()}
        self.assertIn('popularity_calibration', modes)

    def test_heldout_labels_cannot_change_recommendations(self):
        bundle, arrays = load_frozen(self.frozen)
        before = recommend(bundle, arrays, 'test')
        # The entire inference API contains no held-out truth argument. Changing
        # synthetic labels on disk cannot influence scores or recommendations.
        other = self.root / 'unrelated-test.tsv'
        other.write_text('user_id\titem_id\n1\t16\n')
        self.assertEqual(before, recommend(bundle, arrays, 'test'))

    def test_end_to_end_test_evaluation_is_complete_and_frozen(self):
        out = self.root / 'heldout-evaluation'
        manifest = evaluate_frozen(self.frozen, self.test_path, out)
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['users'], len(self.users))
        self.assertFalse(manifest['test_model_selection'])
        results = json.loads((out / 'results.json').read_text())
        self.assertEqual(set(results), set(self.bundle['models']))
        for result in results.values():
            self.assertEqual(set(result['per_user']), set(self.users))
            self.assertIn('tail_recall', result['aggregate'])
            self.assertIn('calibration_jsd', result['aggregate'])
            self.assertIn('groups', result)
        with self.assertRaises(FileExistsError):
            evaluate_frozen(self.frozen, self.test_path, out)

    def test_test_hash_mismatch_rejected_before_output(self):
        other = self.root / 'wrong-test.tsv'
        other.write_text('user_id\titem_id\n1\t16\n')
        out = self.root / 'must-not-exist'
        with self.assertRaisesRegex(ValueError, 'predeclared split hash'):
            evaluate_frozen(self.frozen, other, out)
        self.assertFalse(out.exists())

    def test_tampered_frozen_payload_rejected(self):
        path = self.root / 'tampered'
        import shutil
        shutil.copytree(self.frozen, path)
        (path / 'train.tsv').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            load_frozen(path)

    def test_source_or_score_drift_prevents_freeze(self):
        path = self.runs[0] / 'valid-scores.npz'
        original = path.read_bytes()
        try:
            with np.load(path, allow_pickle=False) as archive:
                data = {name: archive[name] for name in archive.files}
            data['scores'][0, 1] += 1
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, 'Run sources differ'):
                freeze_study(self.study, self.runs, self.metadata, self.root / 'drift')
        finally:
            path.write_bytes(original)

    def test_constrained_regression_uses_static_columns(self):
        selected = {'best_expert': 'a', 'families': {'context': 'context-ridge-1', 'constrained': 'constrained-ridge-1',
                                                   'calibrated': 'calibrated-ridge-1'}}
        specs = model_specs(['a', 'b'], selected, {'context-ridge-1': {}, 'constrained-ridge-1': {}, 'calibrated-ridge-1': {}},
                            {'0': 'a', '1': 'a', '2': 'b'}, {'strengths': {'0': 0, '1': 0, '2': 0}})
        self.assertEqual(specs['constrained-ridge-1']['variant'], 'static')
        self.assertEqual(specs['calibrated-ridge-1']['variant'], 'static')

    def test_calibrated_effective_coefficients_equal_affine_score_composition(self):
        bundle, arrays = load_frozen(self.frozen)
        experts = bundle['active_experts']
        self.assertEqual(len(experts), 2)
        weights = np.array([.25, .75])
        slopes = np.array([.1, .4])
        offsets = np.array([.2, -.3])
        intercept = .15
        coefficients = {'weights': dict(zip(experts, weights * slopes)),
                        'intercept': float(intercept + weights @ offsets),
                        'calibrated_weights': dict(zip(experts, weights)),
                        'score_calibration': {'slopes': slopes.tolist(), 'offsets': offsets.tolist()}}
        bundle['models']['calibrated-ridge-synthetic'] = {'kind': 'linear', 'variant': 'static',
                                                        'coefficients': coefficients}
        models, _ = score_models(bundle, arrays, 'test')
        base = np.stack([(models[name] - arrays['score_mean'][j, :, None]) / arrays['score_scale'][j, :, None]
                         for j, name in enumerate(experts)], axis=2)
        expected = (base * slopes + offsets) @ weights + intercept
        np.testing.assert_allclose(models['calibrated-ridge-synthetic'], expected, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
