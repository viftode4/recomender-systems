"""Exact execution parity on finite synthetic data; no held-out labels."""
from __future__ import annotations

import copy
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exploratory.conditional_evidence import run_experiment as runner
from exploratory.conditional_evidence import data, model as model_api
from operations import conditional_evidence_acceleration as acceleration

import numpy as np
import torch


def matrix():
    rng = np.random.default_rng(481)
    value = rng.integers(1, 6, size=(19, 29), dtype=np.uint8)
    value[rng.random(value.shape) > .58] = 0
    value[:, 0] = 0
    value[:, 1:4] = [1, 2, 3]
    return value


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        runner.numerical_setup(312)
        self.bank = model_api.EvidenceBank(matrix())

    def tearDown(self):
        acceleration.uninstall()

    def assertStateEqual(self, first, second):
        self.assertTrue(acceleration._equal_state(first, second))

    def test_install_is_explicit_idempotent_and_guarded(self):
        original = runner.two_pass_backward
        self.assertIs(original, acceleration._ORIGINAL)
        with self.assertRaises(ValueError):
            acceleration.install(expected_accelerator_sha256='incorrect')
        self.assertIs(runner.two_pass_backward, original)
        receipt = acceleration.receipt()
        self.assertEqual(acceleration.install(expected_accelerator_sha256=receipt['accelerator_sha256'],
                                             expected_baseline_sha256=receipt['baseline_sha256']), receipt)
        self.assertIs(runner.two_pass_backward, acceleration.one_forward_backward)
        self.assertEqual(acceleration.install(), receipt)
        acceleration.uninstall()
        self.assertIs(runner.two_pass_backward, original)
        with patch.object(runner, 'two_pass_backward', lambda *args: None):
            with self.assertRaises(ValueError):
                acceleration.install()

    def test_no_grad_and_grad_forward_are_bitwise_equal_all_arms(self):
        episode = next(data.episodes(matrix(), epoch=1, retain_fraction=.8, seed=22))
        for arm in runner.ARMS:
            model = runner.make_reader(model_api, arm, 11)
            model.train()
            batch = runner.build_batch(self.bank, episode, episode['candidate_ids'][:7], arm, 22)
            state, rng = copy.deepcopy(model.state_dict()), runner.rng_state()
            with torch.no_grad():
                first = model(batch)
            second = model(batch)
            self.assertTrue(torch.equal(first, second.detach()), arm)
            self.assertStateEqual(state, model.state_dict())
            self.assertStateEqual(rng, runner.rng_state())

    def test_gradients_losses_adam_and_rng_exact_multiple_steps_all_arms(self):
        episodes = list(data.episodes(matrix(), epoch=2, retain_fraction=(.8, .9), seed=22))[:6]
        for arm in runner.ARMS:
            models = [runner.make_reader(model_api, arm, 11) for _ in range(2)]
            optimizers = [torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
                          for model in models]
            for step in range(3):
                results, rng_states, build_counts = [], [], []
                initial_rng = runner.rng_state()
                for model, optimizer, method in zip(models, optimizers,
                        (acceleration._ORIGINAL, acceleration.one_forward_backward)):
                    runner.restore_rng(initial_rng)
                    optimizer.zero_grad(set_to_none=True)
                    calls = [0]
                    losses = []
                    for episode in episodes[step * 2:step * 2 + 2]:
                        episode = {**episode, 'scramble_seed': runner.episode_seed(22, episode, epoch=2)}
                        def chunks():
                            for start in range(0, len(episode['candidate_ids']), 7):
                                calls[0] += 1
                                yield runner.build_batch(self.bank, episode,
                                    episode['candidate_ids'][start:start + 7], arm, 22)
                        def loss(logits):
                            return data.sampled_multinomial_loss(logits, episode['target_mask'],
                                                               episode['unobserved_population_count'])
                        losses.append(method(model, chunks, loss, divisor=2.0))
                    gradients = [parameter.grad.clone() for parameter in model.parameters()]
                    optimizer.step()
                    results.append((losses, gradients))
                    rng_states.append(runner.rng_state())
                    build_counts.append(calls[0])
                self.assertStateEqual(results[0], results[1])
                self.assertStateEqual(models[0].state_dict(), models[1].state_dict())
                self.assertStateEqual(optimizers[0].state_dict(), optimizers[1].state_dict())
                self.assertStateEqual(rng_states[0], rng_states[1])
                self.assertEqual(build_counts[0], 2 * build_counts[1])

    def test_train_epoch_semantics_and_physical_work_counters(self):
        for arm in runner.ARMS:
            results, states = [], []
            for method in (acceleration._ORIGINAL, acceleration.one_forward_backward):
                runner.numerical_setup(31)
                model = runner.make_reader(model_api, arm, 31)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
                with patch.object(runner, 'two_pass_backward', method):
                    result = runner.train_epoch(model, self.bank, {'categories': matrix()}, data,
                        optimizer, 1, 31, arm, episode_limit=9, chunk=7)
                results.append(result)
                states.append((model.state_dict(), optimizer.state_dict(), runner.rng_state()))
            self.assertStateEqual(states[0], states[1])
            for key in ('queries', 'epoch', 'mean_query_loss'):
                self.assertEqual(results[0][key], results[1][key])
            for key in results[0]['resources']:
                if key != 'graph_build_seconds':
                    self.assertEqual(results[0]['resources'][key], 2 * results[1]['resources'][key], key)

    def test_existing_checkpoint_resume_to_accelerated_matches_baseline(self):
        for arm in runner.ARMS:
            with tempfile.TemporaryDirectory() as folder:
                runner.numerical_setup(73)
                model = runner.make_reader(model_api, arm, 73)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
                runner.train_epoch(model, self.bank, {'categories': matrix()}, data, optimizer,
                                   1, 73, arm, episode_limit=4, chunk=7)
                checkpoint = Path(folder) / 'latest.pt'
                guard = {'fixed': 'synthetic source and config binding'}
                history = {'completed_epoch': 1}
                runner.save_checkpoint(checkpoint, model, optimizer, 1, guard, history)
                states, outputs = [], []
                for method in (acceleration._ORIGINAL, acceleration.one_forward_backward):
                    current = runner.make_reader(model_api, arm, 73)
                    adam = torch.optim.Adam(current.parameters(), lr=.001, weight_decay=1e-4)
                    self.assertEqual(runner.restore_checkpoint(checkpoint, current, adam, guard), (1, history))
                    with patch.object(runner, 'two_pass_backward', method):
                        result = runner.train_epoch(current, self.bank, {'categories': matrix()}, data,
                            adam, 2, 73, arm, episode_limit=4, chunk=7)
                    outputs.append(result['mean_query_loss'])
                    states.append((current.state_dict(), adam.state_dict(), runner.rng_state()))
                self.assertEqual(outputs[0], outputs[1])
                self.assertStateEqual(states[0], states[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError(args.receipt)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AccelerationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.receipt:
        runner.atomic_json(args.receipt, {
            'status': 'pass' if result.wasSuccessful() else 'fail',
            'stage': 'synthetic_execution_parity',
            'tests_run': result.testsRun,
            'checks': {name: result.wasSuccessful() for name in (
                'model_state_exact', 'optimizer_state_exact', 'rng_state_exact',
                'loss_exact', 'gradient_exact', 'logits_exact', 'checkpoint_resume_exact')},
            'accelerator_sha256': acceleration.digest(acceleration.__file__),
            'baseline_sha256': acceleration.digest(runner.__file__),
            'tests_sha256': acceleration.digest(__file__),
            'arms': list(runner.ARMS),
            'runtime': runner.runtime(),
            'study_result': False,
            'real_data_or_heldout_labels_read': False,
        })
    raise SystemExit(0 if result.wasSuccessful() else 1)
