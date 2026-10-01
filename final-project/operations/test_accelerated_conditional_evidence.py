"""Synthetic execution-overlay and process-safety checks; no live signals."""
from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone
import fcntl
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from operations import accelerated_conditional_evidence as launcher


class ProcessSafetyTests(unittest.TestCase):
    def fixture(self, study):
        now = datetime.now().astimezone().replace(microsecond=0)
        command = [sys.executable, '-u', '-m', 'exploratory.conditional_evidence.workflow',
                   '--out', str(study), '--workers', '6']
        record = {'pid': 81234, 'command': command, 'cwd': str(launcher.ROOT),
                  'started_utc': now.astimezone(timezone.utc).isoformat()}
        identity = {'pid': 81234, 'ppid': 1, 'command': command,
                    'started': now.strftime('%a %b %d %H:%M:%S %Y')}
        return record, {'pid': 81234, 'stage': 'training'}, identity

    def test_command_pid_and_start_time_are_required_before_signal(self):
        study = launcher.ROOT / 'runs/synthetic-unused'
        record, progress, identity = self.fixture(study)
        launcher.validate_parent(record, progress, identity, study)
        variants = [None, {**identity, 'pid': 81235},
                    {**identity, 'command': ['python', 'unrelated.py']},
                    {**identity, 'started': (datetime.now() - timedelta(days=1)).strftime('%a %b %d %H:%M:%S %Y')}]
        with mock.patch.object(launcher.os, 'kill') as kill:
            for value in variants:
                with self.assertRaises(ValueError):
                    launcher.validate_parent(record, progress, value, study)
            kill.assert_not_called()

    def test_takeover_signals_only_verified_parent_then_waits(self):
        study = launcher.ROOT / 'runs/synthetic-unused'
        record, progress, identity = self.fixture(study)
        child = {**identity, 'pid': 81235, 'ppid': 81234}
        with mock.patch.object(launcher, 'lock_held', return_value=True), \
             mock.patch.object(launcher, 'read_json', side_effect=[record, progress]), \
             mock.patch.object(launcher, 'process_identity', return_value=identity), \
             mock.patch.object(launcher, 'descendants', return_value=[child]), \
             mock.patch.object(launcher.os, 'kill') as kill, \
             mock.patch.object(launcher, 'wait_stopped') as wait:
            result = launcher.takeover(study)
            kill.assert_called_once_with(81234, launcher.signal.SIGINT)
            wait.assert_called_once_with(identity, [child], study.with_suffix('.lock'), 60)
            self.assertTrue(result['signalled'])

    def test_changed_process_or_denied_identity_never_signalled(self):
        study = launcher.ROOT / 'runs/synthetic-unused'
        record, progress, identity = self.fixture(study)
        with mock.patch.object(launcher, 'lock_held', return_value=True), \
             mock.patch.object(launcher, 'read_json', side_effect=[record, progress]), \
             mock.patch.object(launcher, 'process_identity', side_effect=[identity, None]), \
             mock.patch.object(launcher, 'descendants', return_value=[]), \
             mock.patch.object(launcher.os, 'kill') as kill:
            with self.assertRaises(ValueError):
                launcher.takeover(study)
            kill.assert_not_called()
        denied = subprocess.CompletedProcess([], 1, '', 'Operation not permitted')
        with mock.patch.object(launcher.subprocess, 'run', return_value=denied), \
             mock.patch.object(launcher.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'no signal'):
                launcher.process_identity(81234)
            kill.assert_not_called()

    def test_timeout_never_forces_processes_and_retains_workers_requirement(self):
        identity = {'pid': 81234, 'started': 'unchanged'}
        with mock.patch.object(launcher, 'process_identity', return_value=identity), \
             mock.patch.object(launcher, 'lock_held', return_value=False), \
             mock.patch.object(launcher.os, 'kill') as kill:
            with self.assertRaises(TimeoutError):
                launcher.wait_stopped(identity, [], Path('unused'),
                    monotonic=iter([0., 61.]).__next__, sleep=lambda _: None)
            kill.assert_not_called()

    def test_default_resume_refuses_held_lock_without_signalling(self):
        with tempfile.TemporaryDirectory(dir=launcher.ROOT / 'runs') as temporary:
            study = Path(temporary) / 'study'
            lock = study.parent / 'study.lock'
            with lock.open('w') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                args = Namespace(out=study, resume=True, takeover=False, workers=1,
                                 parity_report=Path(temporary) / 'unused.json')
                with mock.patch.object(launcher, 'preflight'), \
                     mock.patch.object(launcher.os, 'kill') as kill:
                    with self.assertRaisesRegex(RuntimeError, 'Study still owned'):
                        launcher.run_foreground(args)
                    kill.assert_not_called()


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=launcher.ROOT / 'runs')
        self.folder = Path(self.temporary.name)
        self.study = self.folder / 'study'
        self.study.mkdir()
        self.run = importlib.import_module('exploratory.conditional_evidence.run_experiment')
        self.helper = importlib.import_module('operations.conditional_evidence_acceleration')
        self.run.numerical_setup(7)
        self.originals = {name: getattr(self.run, name) for name in (
            'two_pass_backward', 'train_epoch', 'full_scores', 'check_hashes', 'atomic_torch', 'restore_checkpoint')}
        self.old_installed = launcher._INSTALLED
        self.old_environment = {key: os.environ.get(key) for key in (launcher.ENV_PATH, launcher.ENV_HASH)}
        plan = {'source_sha256': {launcher.BASELINE: launcher.digest(launcher.ROOT / launcher.BASELINE)},
                'runtime': self.run.runtime()}
        launcher.atomic_json(self.study / 'plan.json', plan)
        self.parity = self.folder / 'parity.json'
        launcher.atomic_json(self.parity, {'status': 'pass', 'checks': {key: True for key in launcher.PARITY_CHECKS},
            'runtime': self.run.runtime(),
            'accelerator_sha256': launcher.digest(launcher.ROOT / launcher.HELPER),
            'baseline_sha256': launcher.digest(launcher.ROOT / launcher.BASELINE)})

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(self.run, name, value)
        launcher._INSTALLED = self.old_installed
        for key, value in self.old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temporary.cleanup()

    def model_optimizer(self):
        model = self.run.torch.nn.Linear(2, 1)
        optimizer = self.run.torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad()
        model(self.run.torch.ones(1, 2)).sum().backward()
        optimizer.step()
        return model, optimizer

    def test_atomic_checkpoint_carries_overlay_and_restores_adam_rng(self):
        model, optimizer = self.model_optimizer()
        guard = {'synthetic': True}
        original = self.study / '2026/raw-lr0/latest.pt'
        self.run.save_checkpoint(original, model, optimizer, 1, guard, {'epochs': []})
        original_hash = launcher.digest(original)
        manifest, expected = launcher.prepare_manifest(self.study, self.parity)
        document = launcher.read_json(manifest)
        backup = document['original_latest_backups'][str(original.relative_to(launcher.ROOT))]
        self.assertEqual(launcher.digest(launcher.ROOT / backup['path']), original_hash)
        launcher.install_overlay(manifest, expected)
        epoch, _ = self.run.restore_checkpoint(original, model, optimizer, guard)
        self.assertEqual(epoch, 1)
        next_path = original.parent / 'new.pt'
        self.run.save_checkpoint(next_path, model, optimizer, 2, guard, {'epochs': []})
        payload = self.run.torch.load(next_path, weights_only=True)
        self.assertEqual(payload['execution_overlay']['manifest_sha256'], expected)
        self.assertEqual(payload['guard'], guard)
        states = {key: value.clone() for key, value in model.state_dict().items()}
        fresh, fresh_optimizer = self.model_optimizer()
        self.run.restore_checkpoint(next_path, fresh, fresh_optimizer, guard)
        for key in states:
            self.run.torch.testing.assert_close(states[key], fresh.state_dict()[key], atol=0, rtol=0)
        for key, values in optimizer.state_dict()['state'].items():
            for name, value in values.items():
                self.run.torch.testing.assert_close(value, fresh_optimizer.state_dict()['state'][key][name], atol=0, rtol=0)
        self.run.torch.testing.assert_close(self.run.torch.get_rng_state(), payload['rng']['torch'], atol=0, rtol=0)
        self.assertEqual(launcher.digest(original), original_hash)

    def test_unrecorded_checkpoint_and_changed_manifest_are_rejected(self):
        model, optimizer = self.model_optimizer()
        manifest, expected = launcher.prepare_manifest(self.study, self.parity)
        rogue = self.study / 'unrecorded.pt'
        self.run.save_checkpoint(rogue, model, optimizer, 0, {}, {})
        launcher.install_overlay(manifest, expected)
        with self.assertRaisesRegex(ValueError, 'Unrecorded original'):
            self.run.restore_checkpoint(rogue, model, optimizer, {})
        manifest.write_text(manifest.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            self.run.train_epoch(None, None, None, None, None, 1, 1, 'raw')

    def test_failed_or_stale_parity_refuses_preparation(self):
        value = launcher.read_json(self.parity)
        value['checks']['gradient_exact'] = False
        launcher.atomic_json(self.parity, value)
        with self.assertRaisesRegex(ValueError, 'strict numerical parity'):
            launcher.prepare_manifest(self.study, self.parity)
        self.assertFalse((self.folder / 'study-execution-overlay-v1').exists())

    def test_spawn_style_import_installs_before_canonical_worker_execution(self):
        manifest, expected = launcher.prepare_manifest(self.study, self.parity)
        script = (
            "import runpy; runpy.run_module('operations.accelerated_conditional_evidence', run_name='__mp_main__'); "
            "from exploratory.conditional_evidence import run_experiment as r; "
            "assert r.two_pass_backward.__module__ == 'operations.conditional_evidence_acceleration'; "
            "assert hasattr(r.train_epoch, '__wrapped__'); print('spawn installer verified')"
        )
        result = subprocess.run([sys.executable, '-c', script], cwd=launcher.ROOT,
            env={**os.environ, launcher.ENV_PATH: str(manifest), launcher.ENV_HASH: expected},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('spawn installer verified', result.stdout)

    def test_runtime_mismatch_is_rejected_before_takeover(self):
        plan = launcher.read_json(self.study / 'plan.json')
        plan['runtime']['torch'] = 'changed-runtime'
        launcher.atomic_json(self.study / 'plan.json', plan)
        args = Namespace(out=self.study, resume=True, takeover=True, workers=6,
                         parity_report=self.parity)
        with mock.patch.object(launcher, 'takeover') as takeover:
            with self.assertRaisesRegex(ValueError, 'runtimes must agree'):
                launcher.run_foreground(args)
            takeover.assert_not_called()

    def test_failure_resume_command_and_completion_export_use_overlay(self):
        workflow = Namespace(progress=mock.Mock(), curate=mock.Mock(return_value={'done': True}))
        progress, curate = workflow.progress, workflow.curate
        args = Namespace(workers=3, evidence=self.folder / 'evidence')
        launcher.install_workflow_hooks(workflow, self.study, self.parity, Path('manifest'), 'hash')
        workflow.progress(args, 'interrupted', resume_command='obsolete original command')
        command = progress.call_args.kwargs['resume_command']
        self.assertIn(launcher.MODULE, command)
        self.assertIn('--workers 3', command)
        with mock.patch.object(launcher, 'export_overlay') as export:
            self.assertEqual(workflow.curate(args), {'done': True})
            curate.assert_called_once_with(args)
            export.assert_called_once_with(args.evidence, Path('manifest'), 'hash')

    def test_public_overlay_export_is_separately_hashed_and_omits_private_snapshots(self):
        manifest, expected = launcher.prepare_manifest(self.study, self.parity)
        evidence = self.folder / 'evidence'
        launcher.export_overlay(evidence, manifest, expected)
        folder = evidence / 'execution-overlay'
        for name, digest in launcher.read_json(folder / 'SHA256.json').items():
            self.assertEqual(launcher.digest(folder / name), digest)
        description = launcher.read_json(folder / 'execution.json')
        self.assertNotIn('original_checkpoint_sha256', description)
        self.assertNotIn('original_latest_backups', description)
        self.assertFalse(description['scientific_config_changed'])


if __name__ == '__main__':
    unittest.main()
