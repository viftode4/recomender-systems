import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from field_reference_evaluation import (MODELS, evaluate_frozen, freeze_references,
                                      locked_positive, locked_reference, preflight)
from study import digest, write_json


class FieldReferenceEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        users = ['fictional-a', 'fictional-b']
        items = ['[PAD]']+[str(i) for i in range(1, 17)]
        train = 'user_id\titem_id\n'+''.join(f'{u}\t1\n' for u in users)
        valid = 'user_id\titem_id\n'+''.join(f'{u}\t2\n' for u in users)
        test = 'user_id\titem_id\n'+''.join(f'{u}\t3\n{u}\t4\n' for u in users)
        hashes = {part: hashlib.sha256(text.encode()).hexdigest()
                  for part, text in (('train', train), ('valid', valid), ('test', test))}
        ratings = root/'synthetic.inter'
        ratings.write_text('user_id:token\titem_id:token\trating:float\n'
                           'fictional-a\t1\t3\nfictional-a\t2\t1\n'
                           'fictional-a\t3\t5\nfictional-a\t4\t1\n'
                           'fictional-b\t1\t5\nfictional-b\t2\t3\n'
                           'fictional-b\t3\t2\nfictional-b\t4\t3\n'
                           'unselected\tunknown\tUNPARSED\n')
        references, positives = root/'references', root/'positives'
        references.mkdir()
        paths, selected = {}, {}
        for model in MODELS:
            run = (positives/'2026'/'positive_ease-liked_ratings'
                   if model == 'PositiveEASE' else references/f'2026-{model}-0')
            run.mkdir(parents=True)
            paths[model] = run
            (run/'train.tsv').write_text(train)
            (run/'valid.tsv').write_text(valid)
            write_json(run/'ids.json', {'users': ['[PAD]']+users, 'items': items, 'padding_index': 0})
            np.savez(run/'valid-scores.npz', users=users, items=items,
                     scores=np.tile(np.arange(len(items), 0, -1, dtype=float), (len(users), 1)))
            manifest = {'status': 'complete', 'model': model, 'test_evaluated': False,
                        'validation_used_for_training': False,
                        'settings': {'seed': 2026, 'penalty': 250.}, 'split_sha256': hashes,
                        'data_sha256': {ratings.name: digest(ratings)},
                        'export_sha256': {'valid-scores.npz': digest(run/'valid-scores.npz')}}
            if model == 'PositiveEASE':
                choice = {'model': 'positive_ease', 'seed': 2026, 'penalty': 250.,
                          'selection_cohort': 'meta_fit', 'selection_objective': 'liked_ratings',
                          'selection_ndcg': .5}
                manifest['selection'] = choice
                write_json(positives/'2026'/'selection.json', {'positive_ease:liked_ratings': choice})
                write_json(positives/'2026'/'selection-grid.json', [
                    {'model': 'positive_ease', 'penalty': 250., 'liked_ratings': .5},
                    {'model': 'positive_ease', 'penalty': 1000., 'liked_ratings': .4}])
            else:
                choice = {'path': str(run), 'settings': {'penalty': 250.}, 'meta_fit_ndcg': .5}
                selected[model] = {'selected': choice, 'candidates': [choice, {
                    'path': str(references/f'2026-{model}-1'), 'settings': {'penalty': 1000.},
                    'meta_fit_ndcg': .4}]}
            write_json(run/'manifest.json', manifest)
        write_json(references/'expert-selection-2026.json', selected)
        return references, positives, paths, ratings, test

    def freeze(self, root, references, positives):
        target = root/'frozen'
        freeze_references(references, references, references, positives, target, [2026], k=2)
        return target

    def test_freeze_does_not_open_test_or_ratings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, positives, _, ratings, _ = self.fixture(root)
            original_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.name == 'test.tsv' or path == ratings:
                    raise AssertionError('Freezer opened a label source')
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded):
                frozen = self.freeze(root, references, positives)
                manifest, bundles = preflight(frozen)
            self.assertFalse(manifest['test_read'])
            self.assertEqual(bundles[0]['models'], list(MODELS))
            self.assertTrue(all(row['grid_winner_verified'] for row in bundles[0]['selections'].values()))
            self.assertFalse((frozen/'TEST-OPENED.json').exists())

    def test_synthetic_end_to_end_is_self_contained_and_masks_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, positives, _, ratings, testing = self.fixture(root)
            frozen = self.freeze(root, references, positives)
            shutil.rmtree(references)
            shutil.rmtree(positives)
            test = root/'test.tsv'
            test.write_text(testing)
            out = root/'evaluation'
            evaluate_frozen(frozen, {2026: test}, ratings, out)
            metrics = json.loads((out/'2026'/'metrics.json').read_text())
            recs = json.loads((out/'2026'/'recommendations.json').read_text())
            self.assertEqual(set(metrics), set(MODELS))
            for model, row in metrics.items():
                self.assertEqual(row['stage'], 'test')
                self.assertEqual(row['denominators']['all_observed_users'], 2)
                self.assertEqual(row['denominators']['liked_ratings_users'], 1)
                self.assertEqual(row['denominators']['test_likes'], 1)
                self.assertEqual(row['denominators']['recommendation_slots'], 4)
                self.assertEqual(row['all_observed']['aggregate']['ndcg@2'], 1.)
                self.assertEqual(row['liked_ratings']['aggregate']['ndcg@2'], 1.)
                self.assertEqual(set(row['liked_ratings']['per_user']), {'fictional-a'})
                self.assertEqual(row['known_dislike_rate_per_slot'], .5)
                self.assertEqual(list(recs[model].values()), [['3', '4'], ['3', '4']])
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertTrue(manifest['test_read'])
            self.assertTrue(manifest['test_evaluated'])
            self.assertFalse(manifest['selection_after_test'])
            self.assertEqual(manifest['frozen_manifest_sha256'], digest(frozen/'manifest.json'))
            for name, expected in manifest['output_sha256'].items():
                self.assertEqual(digest(out/name), expected)
            marker = json.loads((frozen/'TEST-OPENED.json').read_text())
            self.assertEqual(marker['status'], 'test_evaluated')
            self.assertEqual(marker['evaluation_manifest_sha256'], digest(out/'manifest.json'))
            with self.assertRaisesRegex(ValueError, 'repeated'):
                evaluate_frozen(frozen, {2026: test}, ratings, root/'second')

    def test_reference_nonmaximum_and_later_tied_choice_rejected(self):
        for value in (.6, .5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                references, _, _, _, _ = self.fixture(Path(temporary))
                path = references/'expert-selection-2026.json'
                records = json.loads(path.read_text())
                wrong = copy.deepcopy(records['EASE']['candidates'][1])
                wrong['meta_fit_ndcg'] = value
                records['EASE']['candidates'].insert(0, wrong)
                write_json(path, records)
                with self.assertRaisesRegex(ValueError, 'first recorded'):
                    locked_reference(references, 2026, 'EASE')

    def test_positive_nonmaximum_tie_and_objective_rejected(self):
        for change in ('maximum', 'tie', 'objective'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                _, positives, paths, _, _ = self.fixture(Path(temporary))
                if change == 'objective':
                    path = paths['PositiveEASE']/'manifest.json'
                    row = json.loads(path.read_text())
                    row['selection']['selection_objective'] = 'all_observed'
                    write_json(path, row)
                else:
                    path = positives/'2026'/'selection-grid.json'
                    rows = json.loads(path.read_text())
                    rows.insert(0, {'model': 'positive_ease', 'penalty': 1000.,
                                    'liked_ratings': .6 if change == 'maximum' else .5})
                    write_json(path, rows)
                with self.assertRaisesRegex(ValueError, 'liked-selected|first recorded'):
                    locked_positive(positives, 2026)

    def test_shared_expected_test_hash_family_seed_and_cohort_required(self):
        for change in ('test', 'family', 'seed', 'cohort'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                references, positives, paths, _, _ = self.fixture(root)
                if change == 'cohort':
                    path = references/'expert-selection-2026.json'
                    data = json.loads(path.read_text())
                    data['SLIMElastic']['selected']['selection_cohort'] = 'development'
                else:
                    path = paths['SLIMElastic']/'manifest.json'
                    data = json.loads(path.read_text())
                    if change == 'test':
                        data['split_sha256']['test'] = '0'*64
                    elif change == 'family':
                        data['model'] = 'EASE'
                    else:
                        data['settings']['seed'] = 2027
                write_json(path, data)
                with self.assertRaisesRegex(ValueError, 'TEST split|family or seed|meta-fit'):
                    self.freeze(root, references, positives)

    def test_payload_code_and_runtime_tampering_precede_marker(self):
        for change in ('payload', 'source_hashes', 'runtime_signature'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                references, positives, _, ratings, _ = self.fixture(root)
                frozen = self.freeze(root, references, positives)
                if change == 'payload':
                    (frozen/'2026'/'EASE'/'scores.npz').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'payload'):
                        evaluate_frozen(frozen, {2026: root/'absent'}, ratings, root/'evaluation')
                else:
                    with patch(f'field_reference_evaluation.{change}', return_value={'changed': True}):
                        with self.assertRaisesRegex(ValueError, 'source or runtime'):
                            evaluate_frozen(frozen, {2026: root/'absent'}, ratings, root/'evaluation')
                self.assertFalse((frozen/'TEST-OPENED.json').exists())
                self.assertFalse((root/'evaluation').exists())

    def test_failed_label_read_keeps_marker_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references, positives, _, ratings, testing = self.fixture(root)
            frozen = self.freeze(root, references, positives)
            test = root/'test.tsv'
            test.write_text('wrong synthetic split')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                evaluate_frozen(frozen, {2026: test}, ratings, root/'evaluation')
            self.assertTrue((frozen/'TEST-OPENED.json').exists())
            test.write_text(testing)
            with self.assertRaisesRegex(ValueError, 'repeated'):
                evaluate_frozen(frozen, {2026: test}, ratings, root/'second')


if __name__ == '__main__':
    unittest.main()
