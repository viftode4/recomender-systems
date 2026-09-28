import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import joint_convergence_evaluation as evaluation
from joint_field_experiment import run_seed as original_run
from joint_field_convergence import run_seed as convergence_run
from study import digest, write_json


class ConvergenceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.seed = 7
        cls.source = cls.root/'source'
        cls.source.mkdir()
        users, items = [f'u{j}' for j in range(8)], ['[PAD]']+[f'i{j}' for j in range(24)]
        pairs, rows = {'train': [], 'valid': [], 'test': []}, []
        for u, user in enumerate(users):
            for j in range(7):
                item = items[1+(u*2+j)%24]
                pairs['train' if j < 5 else 'valid' if j == 5 else 'test'].append((user, item))
                rows.append((user, item, 1+(u+j)%5))
        for part, entries in pairs.items():
            (cls.source/f'{part}.tsv').write_text('user_id\titem_id\n'+''.join(f'{u}\t{i}\n' for u, i in entries))
        cls.ratings = cls.root/'tiny.inter'
        cls.ratings.write_text('user_id:token\titem_id:token\trating:float\n'+''.join(f'{u}\t{i}\t{r}\n' for u, i, r in rows))
        metadata = cls.root/'tiny.item'
        metadata.write_text('item_id:token\tclass:token_seq\n'+''.join(f'{i}\tDrama\n' for i in items[1:]))
        write_json(cls.source/'ids.json', {'padding_index': 0, 'users': ['[PAD]']+users, 'items': items})
        np.savez(cls.source/'valid-scores.npz', users=users, items=items, scores=np.zeros((8, 25)))
        write_json(cls.source/'manifest.json', {'status': 'complete', 'test_evaluated': False,
            'validation_used_for_training': False,
            'data_sha256': {p.name: digest(p) for p in (cls.ratings, metadata)},
            'split_sha256': {part: digest(cls.source/f'{part}.tsv') for part in pairs}})
        cls.prior, cls.research = cls.root/'prior', cls.root/'research'
        with redirect_stdout(io.StringIO()):
            original_run(cls.source, cls.ratings, metadata, cls.prior/str(cls.seed), cls.seed)
            cls.prior_digest = digest(cls.prior/str(cls.seed)/'manifest.json')
            convergence_run(cls.source, cls.ratings, metadata, cls.research/str(cls.seed), cls.seed,
                            cls.prior/str(cls.seed))
        cls.frozen = cls.root/'frozen'
        evaluation.freeze_research(cls.research, cls.frozen, [cls.seed])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_tiny_100_to_400_prefix_replayed_and_proof_copied_without_test(self):
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'test.tsv':
                raise AssertionError('Opened TEST during freeze')
            return original_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, 'open', guarded):
            frozen = Path(temp)/'frozen'
            evaluation.freeze_research(self.research, frozen, [self.seed])
            manifest, bundles = evaluation.preflight(frozen)
            self.assertFalse(manifest['test_read'])
            self.assertTrue(bundles[0]['prefix100_independently_replayed'])
            self.assertEqual(bundles[0]['prior_v1_manifest_sha256'], self.prior_digest)
            for variant in evaluation.VARIANTS:
                proof = frozen/str(self.seed)/variant/'verification.json'
                self.assertEqual(digest(proof), bundles[0]['payload_sha256'][f'{variant}/verification.json'])
                self.assertTrue(json.loads(proof.read_text())['prefix_training_trace_exact'])
        self.assertEqual(digest(self.prior/str(self.seed)/'manifest.json'), self.prior_digest)

    def test_portable_frozen_predictions_no_longer_need_prior_research(self):
        with patch.object(evaluation, 'verify_choices', side_effect=AssertionError('Source access')):
            _, bundles = evaluation.preflight(self.frozen)
            *_, predictions = evaluation.frozen_predictions(self.frozen/str(self.seed), bundles[0])
        self.assertEqual(set(predictions), set(evaluation.VARIANTS+evaluation.BASELINES))

    def test_resealed_invalid_prefix_claims_are_rejected(self):
        for changed in ('flag', 'epoch', 'checkpoints', 'old_manifest', 'trace', 'state_digest', 'runtime'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)/'research'
                shutil.copytree(self.research/str(self.seed), directory)
                manifest = evaluation.read_json(directory/'manifest.json')
                choices = evaluation.read_json(directory/'selection.json')
                proof_path = directory/'adaptive'/'verification.json'
                proof = evaluation.read_json(proof_path)
                if changed == 'flag':
                    proof['prefix_training_trace_exact'] = False
                elif changed == 'epoch':
                    proof['replay_epoch'] = 99
                elif changed == 'checkpoints':
                    del proof['checkpoints']['30']
                elif changed == 'old_manifest':
                    proof['prior_manifest_sha256'] = 'wrong'
                elif changed == 'state_digest':
                    proof['checkpoints']['10']['state_sha256'] = 'wrong'
                elif changed == 'runtime':
                    manifest['runtime']['torch'] = 'changed'
                else:
                    trace_path = directory/'adaptive'/'training-trace.json'
                    trace = evaluation.read_json(trace_path)
                    trace[0]['macro_train_joint_probe_nll'] += .01
                    write_json(trace_path, trace)
                    manifest['output_sha256']['adaptive/training-trace.json'] = digest(trace_path)
                write_json(proof_path, proof)
                choices['adaptive']['prefix100_verification_sha256'] = digest(proof_path)
                write_json(directory/'selection.json', choices)
                manifest['prefix100_verification']['adaptive'] = proof
                for name in ('adaptive/verification.json', 'selection.json'):
                    manifest['output_sha256'][name] = digest(directory/name)
                write_json(directory/'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'replay evidence|signature|trajectory|state|runtime'):
                    evaluation.verify_choices(directory)

    def test_synthetic_final_evaluation_once_and_test_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root/'frozen'
            shutil.copytree(self.frozen, frozen)
            evaluation.evaluate_frozen(frozen, {self.seed: self.source/'test.tsv'}, self.ratings, root/'output')
            result = evaluation.read_json(root/'output'/'aggregates.json')
            self.assertEqual(result['stage'], 'test')
            self.assertEqual(len(result['seeds'][str(self.seed)]['models']), 5)
            history = {}
            for part in ('train', 'valid'):
                for user, item in evaluation.read_pairs(self.source/f'{part}.tsv'):
                    history.setdefault(user, set()).add(item)
            recs = evaluation.read_json(root/'output'/str(self.seed)/'recommendations.json')
            for model in recs.values():
                for adapter in model.values():
                    for user, items in adapter.items():
                        self.assertFalse(set(items) & history[user])
            with self.assertRaisesRegex(ValueError, 'repeated'):
                evaluation.evaluate_frozen(frozen, {self.seed: self.source/'test.tsv'}, self.ratings, root/'again')


if __name__ == '__main__':
    unittest.main()
