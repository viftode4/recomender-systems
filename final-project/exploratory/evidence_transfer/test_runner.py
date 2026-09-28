"""Small synthetic runner checks; no real dataset or TEST artifacts."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from exploratory.context_interactions import test_runner as context_tests
from exploratory.evidence_transfer import run_experiment as run


class EvidenceRunnerTests(unittest.TestCase):
    def setUp(self):
        # Reuse the already reviewed purely synthetic source/reference builder.
        self.fixture = context_tests.RunnerTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.patches = [patch.object(run, 'source_hashes', return_value={'synthetic.py': 'fixed'}),
            patch.object(run, 'CHECKPOINTS', (0, 1)), patch.object(run, 'BATCH_SIZE', 4)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.fixture.tearDown()

    def research(self, name='evidence-research'):
        path = self.root/name; path.mkdir()
        run.write_json(path/'protocol.json', run.protocol([7, 8]))
        run.write_json(path/'reference-input-signature.json', {
            'manifest_sha256': run.digest(self.fixture.reference/'manifest.json'),
            'selection_freeze_sha256': run.digest(self.fixture.reference/'SELECTIONS-FROZEN.json')})
        return path

    def select(self, path, seed):
        data = self.fixture
        with redirect_stdout(io.StringIO()):
            return run.select_seed(data.sources/f'{seed}-EASE-1', data.ratings, data.metadata,
                data.cohorts/str(seed), data.reference, path/str(seed), seed, run.protocol([7, 8]))

    def test_fixed_train_masks_retain_all_hidden_targets_and_exclude_sparse_users(self):
        history = np.array([[0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]], dtype=bool)
        first, second = run.make_episodes(history, 7), run.make_episodes(history, 7)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first['exclude_rows'], [0, 2, 0, 2])
        np.testing.assert_array_equal(first['excluded_training_rows'], [1])
        self.assertFalse(first['eligible'][:, 0].any())
        self.assertFalse((first['contexts'] & first['targets']).any())
        self.assertTrue(first['contexts'].any(1).all()); self.assertTrue(first['targets'].any(1).all())
        np.testing.assert_array_equal(first['contexts'] | first['targets'], history[first['exclude_rows']])
        np.testing.assert_array_equal(first['targets'] & first['eligible'], first['targets'])

    def test_nontrain_rating_poison_cannot_change_training_and_only_meta_pairs_select(self):
        data = self.fixture
        first = self.research('first'); self.select(first, 7)
        train = set(data.parts['train']); lines = data.ratings.read_text().splitlines()
        data.ratings.write_text('\n'.join([lines[0]]+[line if tuple(line.split('\t')[:2]) in train
            else line.rsplit('\t', 1)[0]+'\tNEVER_PARSE_VALID_TEST' for line in lines[1:]])+'\n')
        data.update_raw_hash(); data.make_references()
        second = self.research('second')
        original = run.meta_ndcg
        def guarded(scores, users, items, training, validation, fit, *args):
            self.assertEqual(set(user for user, _ in validation), set(fit))
            return original(scores, users, items, training, validation, fit, *args)
        with patch.object(run, 'meta_ndcg', side_effect=guarded), patch.object(run, 'load_source', side_effect=AssertionError('DEV parsed during training')):
            self.select(second, 7)
        before = json.loads((first/'7/selection.json').read_text())
        after = json.loads((second/'7/selection.json').read_text())
        for name in run.MODEL_NAMES:
            self.assertEqual(before[name]['scores_array_sha256'], after[name]['scores_array_sha256'])
        self.assertEqual(json.loads((first/'7/training-audit.json').read_text()),
                         json.loads((second/'7/training-audit.json').read_text()))

    def test_exact_ties_choose_epoch_zero_with_identical_control_budgets(self):
        path = self.research()
        with patch.object(run, 'meta_ndcg', return_value=.5):
            self.select(path, 7)
        choices = json.loads((path/'7/selection.json').read_text())
        for arm in run.ARMS:
            self.assertEqual(choices[arm]['epoch'], 0)
            self.assertTrue(choices[arm]['refit_replay_exact'])
        self.assertEqual(choices['full_zero_pattern']['epoch'], choices['full_pattern']['epoch'])
        audits = json.loads((path/'7/training-audit.json').read_text())
        self.assertEqual(audits['full_pattern'], audits['no_pattern'])
        self.assertEqual(audits['full_pattern'], audits['marginal_only'])
        run.validate_choices(path/'7', run.protocol([7, 8]))
        choices['full_pattern']['epoch'] = 1
        run.write_json(path/'7/selection.json', choices)
        with self.assertRaisesRegex(ValueError, 'first exact maximum'):
            run.validate_choices(path/'7', run.protocol([7, 8]))

    def test_all_seed_barrier_precedes_development_and_curation_has_no_individual_rows(self):
        path = self.research(); self.select(path, 7)
        with self.assertRaises(FileNotFoundError):
            run.freeze_selections(path)
        with patch.object(run, 'load_source', side_effect=AssertionError('DEV opened before all selections')):
            with self.assertRaises(FileNotFoundError):
                run.evaluate_selected(path, self.fixture.sources, self.fixture.ratings, self.fixture.metadata)
        self.assertFalse((path/'DEVELOPMENT-OPENED.json').exists())
        self.select(path, 8); run.freeze_selections(path)
        original, calls = run.load_source, []
        def guarded(*args):
            self.assertEqual(set(run.verify_barrier(path)['selection_manifest_sha256']), {'7', '8'})
            calls.append(1)
            return original(*args)
        with patch.object(run, 'load_source', side_effect=guarded):
            result = run.evaluate_selected(path, self.fixture.sources, self.fixture.ratings, self.fixture.metadata)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result['fresh_confirmation'])
        model = result['seeds']['7']['models']['full_pattern']
        self.assertEqual(model['groups']['users'], model['denominators']['all_observed_users'])
        self.assertEqual(model['liked_groups']['users'], model['denominators']['liked_ratings_users'])
        output = self.root/'evidence'; run.curate(path, output)
        payload = (output/'aggregates.json').read_text()
        self.assertNotIn('per_user', payload)
        self.assertNotIn('exclude_rows', payload)
        for name, signature in json.loads((output/'SHA256.json').read_text()).items():
            self.assertEqual(run.digest(output/name), signature)
        with self.assertRaises(FileExistsError):
            run.evaluate_selected(path, self.fixture.sources, self.fixture.ratings, self.fixture.metadata)

    def test_score_or_checkpoint_change_blocks_development(self):
        path = self.research(); self.select(path, 7); self.select(path, 8); run.freeze_selections(path)
        with patch.object(run, 'source_hashes', return_value={'synthetic.py': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'barrier'):
                run.verify_barrier(path)
        selected = json.loads((path/'7/selection.json').read_text())['full_pattern']
        checkpoint = path/'7'/selected['checkpoint_file']
        checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
        with patch.object(run, 'load_source', side_effect=AssertionError('DEV opened after mutation')):
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run.evaluate_selected(path, self.fixture.sources, self.fixture.ratings, self.fixture.metadata)
        self.assertFalse((path/'DEVELOPMENT-OPENED.json').exists())


if __name__ == '__main__':
    unittest.main()
