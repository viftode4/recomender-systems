"""Focused orchestration tests; no study data, labels or real fits."""
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from exploratory.conditional_evidence import run_experiment as run


def pool_failure_probe(payload):
    """Top-level callable so the spawn test exercises real child processes."""
    folder, fail = payload
    folder = Path(folder)
    if fail:
        deadline = time.monotonic() + 10
        while not (folder / 'slow-worker.pid').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        if not (folder / 'slow-worker.pid').exists():
            raise RuntimeError('second worker did not start')
        raise ValueError('synthetic trajectory failure')
    (folder / 'slow-worker.pid').write_text(str(os.getpid()))
    time.sleep(30)
    return {'unexpected': 'slow task should have been terminated'}


class RunnerTests(unittest.TestCase):
    def test_checkpoint_tie_prefers_earliest_epoch(self):
        rows = [{'epoch': 25, 'meta_ndcg@10': .3}, {'epoch': 0, 'meta_ndcg@10': .3}]
        self.assertEqual(run.best_checkpoint({'checkpoints': rows})['epoch'], 0)
        with self.assertRaises(ValueError):
            run.best_checkpoint({'checkpoints': [{'epoch': 0, 'meta_ndcg@10': float('nan')}]})

    def test_learning_rate_tie_order_after_epoch(self):
        rows = [{'name': 'first', 'selected': {'epoch': 25, 'meta_ndcg@10': .3}},
                {'name': 'second', 'selected': {'epoch': 0, 'meta_ndcg@10': .3}}]
        self.assertEqual(run.choose_configuration(rows)['name'], 'second')
        rows[0]['selected']['epoch'] = 0
        self.assertEqual(run.choose_configuration(rows)['name'], 'first')

    def test_epoch_and_checkpoint_completeness(self):
        history = {'epochs': [{'epoch': n, 'mean_query_loss': 1.} for n in range(1, 26)],
                   'checkpoints': [{'epoch': n, 'meta_ndcg@10': .2} for n in (0, 25)]}
        run.validate_history(history, 25)
        history['epochs'].pop(3)
        with self.assertRaises(ValueError):
            run.validate_history(history, 25)

    def test_scramble_seed_depends_only_on_episode_identity(self):
        episode = {'user_index': 7, 'retain_fraction': .8, 'context': np.array([0, 1, 0])}
        first = run.episode_seed(2026, episode, epoch=2)
        changed_values = {**episode, 'context': np.array([0, 5, 0])}
        self.assertEqual(first, run.episode_seed(2026, changed_values, epoch=2))
        variants = [run.episode_seed(2026, {**episode, 'user_index': 8}, epoch=2),
                    run.episode_seed(2026, {**episode, 'retain_fraction': .9}, epoch=2),
                    run.episode_seed(2026, episode, epoch=3), run.episode_seed(2026, episode)]
        self.assertTrue(all(first != value for value in variants))

    def test_two_pass_gradient_equals_full_query(self):
        torch.manual_seed(11)
        model = torch.nn.Linear(3, 1, dtype=torch.float64)
        reference = torch.nn.Linear(3, 1, dtype=torch.float64)
        reference.load_state_dict(model.state_dict())
        x = torch.randn(7, 3, dtype=torch.float64)
        def loss(logits):
            return torch.logsumexp(logits, 0) - logits[:2].mean()
        expected = loss(reference(x).reshape(-1))
        expected.backward()
        actual = run.two_pass_backward(model, lambda: iter((x[:3], x[3:5], x[5:])), loss)
        self.assertAlmostEqual(actual, float(expected.detach()), places=13)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-12)

    def test_changed_chunk_inputs_are_rejected(self):
        model = torch.nn.Linear(2, 1)
        calls = 0
        def chunks():
            nonlocal calls
            calls += 1
            yield torch.ones(2, 2) * calls
        with self.assertRaises(ValueError):
            run.two_pass_backward(model, chunks, lambda logits: logits.square().mean())

    def test_atomic_json_and_escape_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'value.json'
            run.atomic_json(path, {'x': 1})
            self.assertEqual(json.loads(path.read_text()), {'x': 1})
            self.assertEqual(list(Path(folder).glob('*.tmp-*')), [])
            with self.assertRaises(ValueError):
                run.verified_relative(Path(folder) / 'child', '../value.json', run.digest(path))

    def test_release_must_agree_with_meta_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            run.atomic_json(directory / 'SELECTIONS-FROZEN.json', {'status': 'frozen'})
            run.atomic_json(directory / 'meta-screen.json', {'passed': True})
            seal = {'meta_screen_sha256': run.digest(directory / 'meta-screen.json')}
            release = {'status': 'ready', 'base_seal_sha256': run.digest(directory / 'SELECTIONS-FROZEN.json'),
                       'meta_screen_sha256': seal['meta_screen_sha256'], 'extension_status': 'not_triggered'}
            run.atomic_json(directory / 'release.json', release)
            with patch.object(run, 'verify_global_seal', return_value=seal):
                with self.assertRaises(ValueError):
                    run.verify_assessment_release(directory, directory / 'release.json')

    def test_partial_evidence_export_resumes_without_replacing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            aggregates, plan, provenance = {'status': 'complete'}, {'source': 'fixed'}, {'hash': 'fixed'}
            run.immutable_json(directory / 'aggregates.json', aggregates)
            before = (directory / 'aggregates.json').stat().st_mtime_ns
            run.export_evidence(directory, aggregates, plan, provenance)
            run.export_evidence(directory, aggregates, plan, provenance)
            self.assertEqual((directory / 'aggregates.json').stat().st_mtime_ns, before)
            self.assertEqual(set(json.loads((directory / 'SHA256.json').read_text())),
                             {'aggregates.json', 'protocol.json', 'provenance.json'})
            with self.assertRaises(ValueError):
                run.export_evidence(directory, {'status': 'different'}, plan, provenance)

    def test_resume_flag_only_controls_recovery_not_training_budget(self):
        args = run.parser().parse_args(['assess', '--out', 'x', '--ratings', 'r',
                                       '--assessment-release', 'a', '--evidence', 'e', '--resume'])
        self.assertTrue(args.resume)
        args = run.parser().parse_args(['train', '--out', 'x', '--ratings', 'r', '--workers', '6'])
        self.assertEqual(args.workers, 6)
        self.assertFalse(hasattr(args, 'max_epochs'))

    def test_parallel_failure_terminates_owned_workers_promptly(self):
        before = {child.pid for child in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as folder:
            started = time.monotonic()
            with self.assertRaisesRegex(ValueError, 'synthetic trajectory failure'):
                run.run_parallel_tasks([(folder, True), (folder, False)], 2, worker=pool_failure_probe)
            self.assertLess(time.monotonic() - started, 15)
            self.assertTrue((Path(folder) / 'slow-worker.pid').exists())
            remaining = {child.pid for child in multiprocessing.active_children()}
            self.assertEqual(remaining - before, set())


if __name__ == '__main__':
    unittest.main()
