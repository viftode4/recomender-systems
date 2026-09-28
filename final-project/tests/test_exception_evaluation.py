import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exception_evaluation import (CODE_FILES, ROOT, evaluate_frozen, freeze_sources,
                                  preflight, read_test_labels)
from study import digest, write_json


class ExceptionEvaluationTests(unittest.TestCase):
    def fixture(self,root):
        users = ['u0','u1','u2','u3']
        items = ['[PAD]']+[str(i) for i in range(1,17)]
        train = 'user_id\titem_id\n'+''.join(f'{u}\t1\n' for u in users)
        valid = 'user_id\titem_id\n'+''.join(f'{u}\t2\n' for u in users)
        test = 'user_id\titem_id\n'+''.join(f'{u}\t3\n{u}\t4\n' for u in users)
        hashes = {name:hashlib.sha256(text.encode()).hexdigest() for name,text in [('train',train),('valid',valid),('test',test)]}
        ratings = root/'fixture.inter'
        ratings.write_text('user_id:token\titem_id:token\trating:float\n'+
                           ''.join(f'{u}\t1\t3\n{u}\t2\t1\n{u}\t3\t5\n{u}\t4\t1\n' for u in users)+
                           'ignored\t99\tUNPARSED\n')
        runs = []
        for model in ('PositiveEASE','SignedEASE'):
            run = root/model
            run.mkdir()
            (run/'train.tsv').write_text(train)
            (run/'valid.tsv').write_text(valid)
            write_json(run/'ids.json',{'users':['[PAD]']+users,'items':items,'padding_index':0})
            np.savez(run/'valid-scores.npz',users=users,items=items,
                     scores=np.tile(np.arange(len(items),0,-1,dtype=float),(len(users),1)))
            write_json(run/'manifest.json',{
                'status':'complete','model':model,'test_evaluated':False,
                'predictor_kind':'static_train_fitted_full_catalog','validation_used_for_training':False,
                'selection':{'selection_cohort':'meta_fit','seed':2026,'selection_objective':'liked_ratings'},
                'split_sha256':hashes,'data_sha256':{ratings.name:digest(ratings)},
                'export_sha256':{'valid-scores.npz':digest(run/'valid-scores.npz')}})
            runs.append(run)
        frozen = root/'frozen'
        frozen.mkdir()
        bundle_hash = freeze_sources(runs,frozen/'2026',2026)
        write_json(frozen/'manifest.json',{'status':'frozen','seeds':[2026],'test_read':False,
                                         'bundle_sha256':{'2026':bundle_hash},
                                         'code_sha256':{p:digest(ROOT/p) for p in CODE_FILES}})
        return runs,frozen,ratings,test

    def test_freeze_succeeds_without_a_test_file(self):
        with tempfile.TemporaryDirectory() as temp:
            runs,frozen,ratings,test = self.fixture(Path(temp))
            _,bundles = preflight(frozen)
            self.assertEqual(len(bundles[0]['models']),2)
            self.assertFalse(bundles[0]['test_read'])
            self.assertEqual(list(Path(temp).rglob('test.tsv')),[])

    def test_changed_predictions_rejected_before_test_open_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            runs,frozen,ratings,test = self.fixture(root)
            (runs[0]/'valid-scores.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                evaluate_frozen(frozen,root/'absent_test_source',ratings,root/'results')
            self.assertFalse((frozen/'TEST-OPENED.json').exists())

    def test_changed_choice_rejected_before_test_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            runs,frozen,ratings,test = self.fixture(root)
            path=frozen/'2026'/'bundle.json'
            value=json.loads(path.read_text())
            value['models'][0]['selection']['selection_objective']='all_observed'
            write_json(path,value)
            with self.assertRaisesRegex(ValueError,'choices or bundle changed'):
                evaluate_frozen(frozen,root/'absent_test_source',ratings,root/'results')
            self.assertFalse((frozen/'TEST-OPENED.json').exists())

    def test_transitive_helper_change_rejected_before_test_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            runs,frozen,ratings,test = self.fixture(root)
            manifest=json.loads((frozen/'manifest.json').read_text())
            self.assertIn('hybrid_constraints.py',manifest['code_sha256'])
            manifest['code_sha256']['hybrid_constraints.py']='0'*64
            write_json(frozen/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'code changed: hybrid_constraints.py'):
                evaluate_frozen(frozen,root/'absent_test_source',ratings,root/'results')
            self.assertFalse((frozen/'TEST-OPENED.json').exists())

    def test_synthetic_end_to_end_masks_validation_and_has_test_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            runs,frozen,ratings,test=self.fixture(root)
            source=root/'source'/'2026-EASE-1'
            source.mkdir(parents=True)
            (source/'test.tsv').write_text(test)
            evaluate_frozen(frozen,root/'source',ratings,root/'results')
            rows=json.loads((root/'results'/'2026'/'metrics.json').read_text())
            for result in rows.values():
                self.assertEqual(result['stage'],'test')
                self.assertEqual(result['denominators']['test_likes'],4)
                self.assertNotIn('validation_likes',result['denominators'])
                self.assertEqual(result['known_dislike_rate_per_slot'],.1)
                self.assertEqual(result['liked_ratings']['aggregate']['ndcg@10'],1.)
            manifest=json.loads((root/'results'/'manifest.json').read_text())
            for path,expected in manifest['output_sha256'].items():
                self.assertEqual(digest(root/'results'/path),expected)
            with self.assertRaisesRegex(ValueError,'already opened test'):
                evaluate_frozen(frozen,root/'source',ratings,root/'again')

    def test_test_history_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            runs,frozen,ratings,test=self.fixture(root)
            path=root/'test.tsv'
            path.write_text(test)
            with self.assertRaisesRegex(ValueError,'history leakage'):
                read_test_labels(path,ratings,digest(path),digest(ratings),[('u0','3')],
                                 ['u0','u1','u2','u3'],['[PAD]']+[str(i) for i in range(1,17)])


if __name__=='__main__':
    unittest.main()
