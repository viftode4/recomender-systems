import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import joint_field_convergence as convergence
import joint_field_experiment as original
from categorical_experiment import restore


class ConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prior = cls.root / 'prior' / 'adaptive'
        cls.prior.mkdir(parents=True)
        (cls.prior.parent / 'manifest.json').write_text('{"synthetic": true}')
        cls.matrix = np.array([[0, 1, 5, 0, 2, 0], [0, 4, 0, 1, 5, 0]], dtype=np.int64)
        cls.users = ['u', 'v']
        cls.items = ['[PAD]', 'a', 'b', 'c', 'd', 'e']
        cls.labels = {('u', 'c'): 4, ('v', 'b'): 2}
        # Real original 100-epoch loop on a tiny synthetic catalog, without
        # replacing any original global or touching any project data.
        original.train_variant(cls.matrix, cls.users, cls.items, cls.labels,
                               {'u'}, 9, 'adaptive', cls.prior)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_explicit_protocol_does_not_mutate_original_budget(self):
        old = original.CHECKPOINTS
        protocol = convergence.protocol([2026, 2027, 2028])
        self.assertEqual(protocol['checkpoints'], [10, 30, 60, 100, 200, 300, 400])
        self.assertEqual(protocol['epochs'], 400)
        self.assertEqual(protocol['study_kind'], 'post_v1_convergence_sensitivity')
        self.assertEqual(original.CHECKPOINTS, old)
        self.assertNotIn('joint_field_convergence.py', original.SOURCE_FILES)

    def test_fresh_restart_exactly_replays_prefix_and_ignores_development_labels(self):
        first, second = self.root / 'new-a', self.root / 'new-b'
        first.mkdir(); second.mkdir()
        with patch.object(convergence, 'CHECKPOINTS', (10, 30, 60, 100, 102)):
            a = convergence.train_variant(self.matrix, self.users, self.items, self.labels,
                {'u'}, 9, 'adaptive', first, self.prior)
            b = convergence.train_variant(self.matrix, self.users, self.items,
                {**self.labels, ('v', 'b'): np.nan}, {'u'}, 9, 'adaptive', second, self.prior)
        verification = json.loads((first / 'verification.json').read_text())
        self.assertTrue(verification['restarted_from_same_seed'])
        self.assertTrue(verification['prefix_training_trace_exact'])
        self.assertEqual(verification['prefix_epoch_count'], 100)
        self.assertEqual(set(verification['checkpoints']), {'10', '30', '60', '100'})
        self.assertTrue(all(row['state_exact'] and row['raw_logits_exact']
                            for row in verification['checkpoints'].values()))
        self.assertEqual(a['epoch'], b['epoch'])
        self.assertEqual(a['logits_array_sha256'], b['logits_array_sha256'])
        self.assertTrue(a['checkpoint_replay_exact'])

    def test_changed_prefix_training_loss_aborts_before_extension(self):
        folder = self.root / 'tampered-prior'
        folder.mkdir()
        trace = json.loads((self.prior / 'training-trace.json').read_text())
        trace[0]['macro_train_joint_probe_nll'] += .1
        (folder / 'training-trace.json').write_text(json.dumps(trace))
        (folder / 'selection.json').write_text((self.prior / 'selection.json').read_text())
        output = self.root / 'rejected'
        output.mkdir()
        with self.assertRaisesRegex(AssertionError, 'training trace mismatch'):
            convergence.train_variant(self.matrix, self.users, self.items, self.labels,
                {'u'}, 9, 'adaptive', output, folder)
        self.assertFalse((output / 'verification.json').exists())

    def test_different_initial_seed_rejected_before_first_update(self):
        output = self.root / 'bad-initialization'
        output.mkdir()
        with self.assertRaisesRegex(AssertionError, 'initial model state'):
            convergence.train_variant(self.matrix, self.users, self.items, self.labels,
                {'u'}, 10, 'adaptive', output, self.prior)

    def test_changed_prior_model_state_rejected(self):
        model, _ = restore(self.prior / 'epoch-100.pt')
        with torch.no_grad():
            model.decoder[-1].bias.add_(.1)
        logits = convergence.predict_logits(model, self.matrix)
        with self.assertRaisesRegex(AssertionError, 'state replay mismatch'):
            convergence.verify_prefix_checkpoint(model, logits, self.prior, 100,
                                                  self.matrix, self.prior / 'epoch-100.pt')


if __name__ == '__main__':
    unittest.main()
