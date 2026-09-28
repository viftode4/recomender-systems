import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import run_final_batch as batch


class FinalBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = self.root/'plan.json'
        self.ratings = self.root/'tiny.inter'
        self.ratings.write_text('ratings are never opened in preflight')
        for seed in batch.SEEDS:
            path = self.root/f'runs/research-v2/{seed}-EASE-1/test.tsv'
            path.parent.mkdir(parents=True)
            path.write_text('synthetic unopened TEST')
        for name in batch.ANALYSIS_FILES:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic analysis source')
        self.state = {'per_seed_data': {str(seed): {'data_sha256': {'tiny.inter': batch.digest(self.ratings)}} for seed in batch.SEEDS},
                      'frozen': {'synthetic': 'sealed'}, 'runtime': {'python': 'fixed'}}
        self.patch_root = patch.object(batch, 'ROOT', self.root)
        self.patch_collect = patch.object(batch, 'collect_preflight', side_effect=lambda _: copy.deepcopy(self.state))
        self.patch_root.start()
        self.patch_collect.start()

    def tearDown(self):
        self.patch_collect.stop()
        self.patch_root.stop()
        self.temp.cleanup()

    def prepare(self):
        return batch.prepare(self.plan, Path(sys.executable), self.ratings, self.root)

    def successful_subprocess(self, argv, **kwargs):
        marker = batch.read_json(self.root/'runs/FINAL-TEST-BATCH.json')
        self.assertEqual(marker['plan_sha256'], batch.digest(self.plan))
        self.assertEqual(marker['status'], 'reserved_before_test')
        self.assertFalse(kwargs['check'])
        self.assertEqual({key: kwargs['env'][key] for key in batch.THREADS}, batch.THREADS)
        output = Path(argv[argv.index('--out')+1])
        output.mkdir(parents=True)
        batch.write_json(output/'manifest.json', {'status': 'complete'})
        return subprocess.CompletedProcess(argv, 0)

    def test_prepare_reads_no_labels_and_declares_exact_eight_jobs(self):
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'test.tsv':
                raise AssertionError('Read TEST during prepare')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            plan = self.prepare()
        self.assertFalse(plan['test_read'])
        self.assertEqual(len(plan['jobs']), 8)
        self.assertEqual(len(plan['analysis_jobs']), 5)
        self.assertEqual({job['name'] for job in plan['jobs']},
            {'primary-2026', 'primary-2027', 'primary-2028', 'exception', 'conditional', 'joint100', 'joint400', 'references'})
        self.assertFalse((self.root/'runs/FINAL-TEST-BATCH.json').exists())
        self.assertEqual(set(plan['analysis_sha256']), set(batch.ANALYSIS_FILES))

    def test_marker_precedes_all_jobs_and_comparisons_follow_success(self):
        self.prepare()
        finished = set()
        lock = threading.Lock()
        active, maximum = 0, 0
        def execute(argv, **kwargs):
            nonlocal active, maximum
            with lock:
                if Path(argv[1]).name.startswith('compare_'):
                    self.assertEqual(len(finished), 8 if 'compare_frozen.py' in argv[1] else 9)
                active += 1
                maximum = max(maximum, active)
            time.sleep(.01)
            result = self.successful_subprocess(argv, **kwargs)
            with lock:
                active -= 1
                finished.add(tuple(argv))
            return result
        with patch.object(batch.subprocess, 'run', side_effect=execute):
            result = batch.run_plan(self.plan)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['results']), 13)
        self.assertGreater(maximum, 1)
        self.assertLessEqual(maximum, 3)
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            batch.run_plan(self.plan)

    def test_resealed_command_or_changed_source_or_runtime_fails_before_marker(self):
        for changed in ('command', 'source', 'runtime'):
            with self.subTest(changed=changed):
                plan = self.prepare()
                if changed == 'command':
                    plan['jobs'][0]['argv'].append('--unreviewed-choice')
                    batch.write_json(self.plan, plan)
                    self.plan.with_suffix('.json.sha256').write_text(batch.digest(self.plan)+'\n')
                elif changed == 'source':
                    (self.root/'compare_fields_final.py').write_text('changed choice')
                else:
                    self.state['runtime']['python'] = 'changed'
                with patch.object(batch.subprocess, 'run', side_effect=AssertionError('Launched before verification')):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        batch.run_plan(self.plan)
                self.assertFalse((self.root/'runs/FINAL-TEST-BATCH.json').exists())
                self.plan.unlink()
                self.plan.with_suffix('.json.sha256').unlink()

    def test_failure_keeps_reservation_and_never_runs_comparisons_or_retries(self):
        self.prepare()
        def fail(argv, **kwargs):
            self.assertFalse(Path(argv[1]).name.startswith('compare_'))
            self.assertTrue((self.root/'runs/FINAL-TEST-BATCH.json').exists())
            return subprocess.CompletedProcess(argv, 3)
        with patch.object(batch.subprocess, 'run', side_effect=fail) as calls:
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                batch.run_plan(self.plan)
        self.assertEqual(calls.call_count, 8)
        record = batch.read_json(self.root/'runs/FINAL-TEST-BATCH.json')
        self.assertEqual(record['status'], 'failed_reserved')
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            self.prepare()

    def test_preflight_failure_or_existing_output_prevents_reservation(self):
        with patch.object(batch, 'collect_preflight', side_effect=ValueError('Frozen mismatch')):
            with self.assertRaisesRegex(ValueError, 'Frozen mismatch'):
                self.prepare()
        output = self.root/'runs/final-v3/2026'
        output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'existing output'):
            self.prepare()
        self.assertFalse(self.plan.exists())
        self.assertFalse((self.root/'runs/FINAL-TEST-BATCH.json').exists())

    def test_changed_ratings_bytes_rejected_before_marker_or_job(self):
        self.prepare()
        self.ratings.write_text('changed raw dataset')
        with patch.object(batch.subprocess, 'run', side_effect=AssertionError('Unexpected launch')):
            with self.assertRaisesRegex(ValueError, 'Ratings dataset bytes'):
                batch.run_plan(self.plan)
        self.assertFalse((self.root/'runs/FINAL-TEST-BATCH.json').exists())

    def test_different_interpreter_path_rejected_before_preflight(self):
        with patch.object(batch, 'collect_preflight', side_effect=AssertionError('Premature preflight')):
            with self.assertRaisesRegex(ValueError, 'verified interpreter'):
                batch.prepare(self.plan, self.root/'another-venv/bin/python', self.ratings, self.root)


class SharedSignatureTests(unittest.TestCase):
    def test_real_collector_rejects_cross_family_id_split_data_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            users, items = ['u1', 'u2'], ['[PAD]', 'a', 'b']
            primary = root/'runs/frozen-v3'
            index = {'seeds': list(batch.SEEDS), 'test_read': False, 'bundles': {}}
            primary_bundles, families, identities = {}, {}, {}
            arrays = {'users': SimpleNamespace(tolist=lambda: users), 'items': SimpleNamespace(tolist=lambda: items)}
            def setup_folder(folder, seed):
                folder.mkdir(parents=True)
                for part in ('train', 'valid'):
                    (folder/f'{part}.tsv').write_text(f'{part} synthetic {seed}')
                splits = {part: batch.digest(folder/f'{part}.tsv') for part in ('train', 'valid')}
                splits['test'] = f'never-opened-{seed}'
                return {'data_sha256': {'tiny.inter': 'shared'}, 'split_sha256': splits}
            for seed in batch.SEEDS:
                folder = primary/str(seed)
                signature = setup_folder(folder, seed)
                bundle = {'seed': seed, 'sources': {'one': {'manifest': signature}},
                          'expected_test_sha256': signature['split_sha256']['test'], 'code_sha256': {'shared.py': 'same'}}
                batch.write_json(folder/'freeze.json', bundle)
                primary_bundles[seed] = bundle
                index['bundles'][str(seed)] = {'freeze_sha256': batch.digest(folder/'freeze.json')}
            batch.write_json(primary/'manifest.json', index)
            for name, (module, frozen, _) in batch.TRACKS.items():
                parent = root/'runs'/frozen
                bundles, hashes = [], {}
                identities[frozen] = [users[:], items[:]]
                for seed in batch.SEEDS:
                    folder = parent/str(seed)
                    signature = setup_folder(folder, seed)
                    bundle = {'seed': seed, **signature, 'users': users[:], 'items': items[:],
                              'expected_test_sha256': signature['split_sha256']['test']}
                    batch.write_json(folder/'bundle.json', bundle)
                    batch.write_json(folder/'score-ids.json', {'users': users, 'items': items})
                    bundles.append(bundle)
                    hashes[str(seed)] = batch.digest(folder/'bundle.json')
                manifest = {'seeds': list(batch.SEEDS), 'bundle_sha256': hashes, 'code_sha256': {'shared.py': 'same'}}
                batch.write_json(parent/'manifest.json', manifest)
                families[name] = (manifest, bundles)
            modules = {'freeze': SimpleNamespace(load_frozen=lambda path: (primary_bundles[int(path.name)], arrays))}
            for name, (module, _, _) in batch.TRACKS.items():
                modules[module] = SimpleNamespace(preflight=lambda _, name=name: families[name])
            modules['categorical_evaluation'].load_training = lambda folder: (*identities[folder.parent.name], None, None, None)
            modules['joint_evaluation'].runtime_signature = lambda: {'synthetic': 'fixed'}
            with patch.dict(sys.modules, modules):
                self.assertEqual(len(batch.collect_preflight(root)['frozen']), 6)
                for changed in ('ids', 'split', 'data', 'source'):
                    with self.subTest(changed=changed):
                        saved = copy.deepcopy(families['conditional'])
                        old_ids = copy.deepcopy(identities['frozen-adaptive-v1'])
                        if changed == 'ids':
                            identities['frozen-adaptive-v1'][0].reverse()
                        elif changed == 'split':
                            families['conditional'][1][0]['split_sha256']['test'] = 'different-test'
                        elif changed == 'data':
                            families['conditional'][1][0]['data_sha256']['tiny.inter'] = 'different-dataset'
                        else:
                            families['conditional'][0]['code_sha256']['shared.py'] = 'different-source'
                        with self.assertRaisesRegex(ValueError, 'disagree'):
                            batch.collect_preflight(root)
                        families['conditional'] = saved
                        identities['frozen-adaptive-v1'] = old_ids


if __name__ == '__main__':
    unittest.main()
