import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from categorical_experiment import run_seed
from categorical_evaluation import (evaluate_frozen, freeze_research, frozen_predictions,
                                    preflight, verify_choices)
from study import digest, write_json


class CategoricalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root/'source'
        cls.source.mkdir()
        cls.seed = 7
        cls.users = [f'u{j}' for j in range(8)]
        cls.items = ['[PAD]']+[f'i{j}' for j in range(24)]
        cls.training, cls.validation, cls.testing = [], [], []
        rows = []
        for u, user in enumerate(cls.users):
            items = [cls.items[1+(u*2+j)%24] for j in range(7)]
            for j, item in enumerate(items[:5]):
                cls.training.append((user, item))
                rows.append((user, item, 1+(u+j)%5))
            cls.validation.append((user, items[5]))
            cls.testing.append((user, items[6]))
            rows += [(user, items[5], 5), (user, items[6], 4 if u%2 else 2)]
        for part, pairs in (('train', cls.training), ('valid', cls.validation), ('test', cls.testing)):
            (cls.source/f'{part}.tsv').write_text('user_id\titem_id\n'+''.join(f'{u}\t{i}\n' for u, i in pairs))
        cls.ratings = cls.root/'tiny.inter'
        cls.ratings.write_text('user_id:token\titem_id:token\trating:float\n'+''.join(f'{u}\t{i}\t{r}\n' for u, i, r in rows)+
                               'unselected\tunknown\tUNPARSED_HELDOUT\n')
        cls.metadata = cls.root/'tiny.item'
        cls.metadata.write_text('item_id:token\tclass:token_seq\n'+''.join(f'{i}\tDrama\n' for i in cls.items[1:]))
        write_json(cls.source/'ids.json', {'padding_index': 0, 'users': ['[PAD]']+cls.users, 'items': cls.items})
        np.savez(cls.source/'valid-scores.npz', users=cls.users, items=cls.items, scores=np.zeros((8, 25)))
        write_json(cls.source/'manifest.json', {
            'status': 'complete', 'test_evaluated': False, 'validation_used_for_training': False,
            'data_sha256': {p.name: digest(p) for p in (cls.ratings, cls.metadata)},
            'split_sha256': {part: digest(cls.source/f'{part}.tsv') for part in ('train', 'valid', 'test')}})
        cls.research = cls.root/'research'
        cls.research.mkdir()
        with patch('categorical_experiment.CHECKPOINTS', (1, 2)), patch('categorical_experiment.benchmark', return_value={'synthetic_test': True}):
            run_seed(cls.source, cls.ratings, cls.metadata, cls.research/str(cls.seed), cls.seed)
        cls.frozen = cls.root/'frozen'
        freeze_research(cls.research, cls.frozen, [cls.seed])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_bundle(self, root):
        target = root/'frozen'
        shutil.copytree(self.frozen, target)
        return target

    def test_freeze_never_opens_test_and_replays_all_six_models(self):
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'test.tsv':
                raise AssertionError('Freeze opened TEST')
            return original_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, 'open', guarded):
            frozen = Path(temp)/'frozen'
            freeze_research(self.research, frozen, [self.seed])
            manifest, bundles = preflight(frozen)
            users, items, train, valid, predictions = frozen_predictions(frozen/str(self.seed), bundles[0])
            self.assertEqual(set(predictions), {'adaptive', 'fixed_flow', 'hard_clamp', 'global_histogram', 'item_histogram', 'item_user_product'})
            self.assertEqual(users, self.users)
            self.assertEqual(items, self.items)
            self.assertEqual(train, self.training)
            self.assertFalse(manifest['test_read'])
            self.assertFalse((frozen/'TEST-OPENED.json').exists())

    def test_synthetic_test_masks_train_and_validation_and_refuses_second_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self.copy_bundle(root)
            output = root/'evaluation'
            evaluate_frozen(frozen, {self.seed: self.source/'test.tsv'}, self.ratings, output)
            recommendations = json.loads((output/str(self.seed)/'recommendations.json').read_text())
            history = {}
            for user, item in self.training+self.validation:
                history.setdefault(user, set()).add(item)
            for recs in recommendations.values():
                for user, items in recs.items():
                    self.assertFalse(set(items) & history[user])
                    self.assertEqual(len(items), 10)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertTrue(manifest['test_read'])
            self.assertFalse(manifest['selection_after_test'])
            result = json.loads((output/'aggregates.json').read_text())
            self.assertEqual(result['stage'], 'test')
            for model in result['seeds'][str(self.seed)]['models'].values():
                self.assertEqual(model['categorical']['observations'], 8)
                self.assertEqual(model['ranking']['denominators']['liked_ratings_users'], 4)
            with self.assertRaisesRegex(ValueError, 'repeated'):
                evaluate_frozen(frozen, {self.seed: self.source/'test.tsv'}, self.ratings, root/'second')

    def test_payload_tamper_and_wrong_test_hash_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self.copy_bundle(root)
            (frozen/str(self.seed)/'train.tsv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'payload'):
                preflight(frozen)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self.copy_bundle(root)
            bad_test = root/'bad.tsv'
            bad_test.write_text('not the frozen split')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                evaluate_frozen(frozen, {self.seed: bad_test}, self.ratings, root/'evaluation')
            self.assertTrue((frozen/'TEST-OPENED.json').exists())

    def test_portable_bundle_needs_no_research_source_path(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen = self.copy_bundle(Path(temp))
            with patch('categorical_evaluation.verify_choices', side_effect=AssertionError('Source access')):
                _, bundles = preflight(frozen)
                _, _, _, _, predictions = frozen_predictions(frozen/str(self.seed), bundles[0])
            self.assertEqual(len(predictions), 6)

    def test_selection_grid_tamper_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            research = root/'research'
            shutil.copytree(self.research, research)
            folder = research/str(self.seed)
            grid_path = folder/'adaptive'/'selection-grid.json'
            rows = json.loads(grid_path.read_text())
            rows[0]['meta_fit_macro_cross_entropy'] = -100.
            write_json(grid_path, rows)
            with self.assertRaisesRegex(ValueError, 'artifact hash'):
                verify_choices(folder)
        with self.assertRaises(FileExistsError):
            freeze_research(self.research, self.frozen, [self.seed])

    def test_nonminimum_choice_and_different_episode_budget_are_rejected(self):
        for changed in ('minimum', 'episodes'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                research = Path(temp)/'research'
                shutil.copytree(self.research, research)
                folder = research/str(self.seed)
                filename = 'adaptive/selection-grid.json' if changed == 'minimum' else 'fixed_flow/training-trace.json'
                path = folder/filename
                rows = json.loads(path.read_text())
                if changed == 'minimum':
                    selection = json.loads((folder/'selection.json').read_text())['adaptive']['epoch']
                    next(row for row in rows if row['epoch'] != selection)['meta_fit_macro_cross_entropy'] = -100.
                else:
                    rows[0]['probe_mask_sha256'] = 'different-mask'
                write_json(path, rows)
                manifest_path = folder/'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['output_sha256'][filename] = digest(path)
                write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, 'minimum|episodes'):
                    verify_choices(folder)

    def test_runtime_and_code_mismatch_rejected_before_test_open(self):
        for name in ('runtime_signature', 'source_hashes'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                frozen = self.copy_bundle(root)
                with patch(f'categorical_evaluation.{name}', return_value={'changed': 'signature'}):
                    with self.assertRaisesRegex(ValueError, 'runtime|code'):
                        evaluate_frozen(frozen, {self.seed: self.source/'test.tsv'}, self.ratings, root/'evaluation')
                self.assertFalse((frozen/'TEST-OPENED.json').exists())
                self.assertFalse((root/'evaluation').exists())


if __name__ == '__main__':
    unittest.main()
