import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exception_experiment import export_run, load_source
from exception_model import rating_matrices
from study import digest


class ExceptionArtifactTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'
        source.mkdir()
        (source/'train.tsv').write_text('user_id\titem_id\nu\ta\nv\tb\n')
        (source/'valid.tsv').write_text('user_id\titem_id\nu\tb\nv\ta\n')
        # A test path is intentionally absent. Unselected data has a nonnumeric
        # rating to prove it is never parsed into a training/evaluation label.
        ratings = root/'tiny.inter'
        ratings.write_text('user_id:token\titem_id:token\trating:float\n'
                           'u\ta\t5\nv\tb\t1\nu\tb\t4\nv\ta\t3\n'
                           'never_read\tz\tPRIVATE_HELD_OUT_LABEL\n')
        metadata = root/'tiny.item'
        metadata.write_text('item_id:token\tclass:token_seq\na\tDrama\nb\tDrama Comedy\n')
        np.savez(source/'valid-scores.npz',users=['u','v'],items=['[PAD]','a','b'],scores=np.zeros((2,3)))
        (source/'ids.json').write_text(json.dumps({'padding_index':0,'users':['[PAD]','v','u'],'items':['[PAD]','a','b']}))
        manifest = {'status':'complete','test_evaluated':False,
                    'split_sha256':{p:digest(source/f'{p}.tsv') for p in ('train','valid')},
                    'data_sha256':{p.name:digest(p) for p in (ratings,metadata)}}
        (source/'manifest.json').write_text(json.dumps(manifest))
        return source,ratings,metadata

    def test_export_keeps_original_ids_without_inheriting_upstream_model_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source,ratings,metadata = self.fixture(root)
            *_,manifest = load_source(source,ratings,metadata)
            manifest.update({'checkpoint':'missing-upstream.pth','checkpoint_sha256':'bad',
                             'recbole_model':'EASE','content_model_sha256':'stale'})
            export_run(root/'export',source,['u','v'],['[PAD]','a','b'],np.zeros((2,3)),
                       manifest,'ContrastTransfer',{'selection_cohort':'meta_fit'})
            exported=json.loads((root/'export'/'manifest.json').read_text())
            self.assertEqual(exported['model'],'ContrastTransfer')
            self.assertEqual(exported['predictor_kind'],'static_train_fitted_full_catalog')
            self.assertEqual((root/'export'/'ids.json').read_bytes(),(source/'ids.json').read_bytes())
            for key in ('checkpoint','checkpoint_sha256','recbole_model','content_model_sha256'):
                self.assertNotIn(key,exported)
            self.assertFalse((root/'export'/'test.tsv').exists())

    def test_only_explicit_train_valid_pairs_supply_ratings(self):
        with tempfile.TemporaryDirectory() as temp:
            source,ratings,metadata = self.fixture(Path(temp))
            users,items,_,_,training,validation,*_ = load_source(source,ratings,metadata)
            self.assertEqual(training,[('u','a',5.),('v','b',1.)])
            self.assertEqual(validation,{('u','b'):4.,('v','a'):3.})
            signed,_ = rating_matrices(users,items,training)
            self.assertEqual(signed[0,2],0)
            self.assertFalse((source/'test.tsv').exists())

    def test_validation_rating_changes_cannot_change_training_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            source,ratings,metadata = self.fixture(Path(temp))
            before = load_source(source,ratings,metadata)
            ratings.write_text(ratings.read_text().replace('u\tb\t4','u\tb\t1'))
            manifest = json.loads((source/'manifest.json').read_text())
            manifest['data_sha256'][ratings.name] = digest(ratings)
            (source/'manifest.json').write_text(json.dumps(manifest))
            after = load_source(source,ratings,metadata)
            np.testing.assert_array_equal(rating_matrices(before[0],before[1],before[4])[0],
                                          rating_matrices(after[0],after[1],after[4])[0])
            self.assertNotEqual(before[5],after[5])

    def test_corrupted_split_and_dataset_fail_before_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            source,ratings,metadata = self.fixture(Path(temp))
            (source/'train.tsv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'split hash mismatch'):
                load_source(source,ratings,metadata)
        with tempfile.TemporaryDirectory() as temp:
            source,ratings,metadata = self.fixture(Path(temp))
            ratings.write_text('changed')
            with self.assertRaisesRegex(ValueError,'dataset hash mismatch'):
                load_source(source,ratings,metadata)


if __name__ == '__main__':
    unittest.main()
