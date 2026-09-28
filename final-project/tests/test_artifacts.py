import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from study import load_runs


class ArtifactTests(unittest.TestCase):
    def make_run(self, root, name):
        p = root/name
        p.mkdir()
        hashes = {}
        for split, item in [('train','a'),('valid','b')]:
            body = f'user_id\titem_id\nu\t{item}\n'
            (p/f'{split}.tsv').write_text(body)
            hashes[split] = hashlib.sha256(body.encode()).hexdigest()
        np.savez_compressed(p/'valid-scores.npz', users=np.array(['u']),
                            items=np.array(['padding','a','b']), scores=np.array([[0.,1.,2.]]))
        manifest = {'status':'complete','test_evaluated':False, 'split_sha256':hashes,
                    'data_sha256':{'ml-100k.item':'example'}, 'model':'EASE'}
        (p/'manifest.json').write_text(json.dumps(manifest))
        return p

    def test_matching_runs_and_tampered_split(self):
        with tempfile.TemporaryDirectory() as d:
            a,b = [self.make_run(Path(d),name) for name in ('a','b')]
            self.assertEqual(load_runs([a,b])[0], ['u'])
            (b/'train.tsv').write_text('user_id\titem_id\nu\tb\n')
            with self.assertRaisesRegex(ValueError,'Split contents'):
                load_runs([a,b])

    def test_misaligned_score_columns_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b = [self.make_run(Path(d),name) for name in ('a','b')]
            np.savez_compressed(b/'valid-scores.npz', users=np.array(['u']),
                                items=np.array(['padding','b','a']), scores=np.array([[0.,1.,2.]]))
            with self.assertRaisesRegex(ValueError,'score ID order'):
                load_runs([a,b])

    def test_test_evaluated_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b = [self.make_run(Path(d),name) for name in ('a','b')]
            m=json.loads((a/'manifest.json').read_text());m['test_evaluated']=True
            (a/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'validation-only'):
                load_runs([a,b])
