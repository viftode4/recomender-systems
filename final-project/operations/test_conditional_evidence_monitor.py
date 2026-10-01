from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from . import conditional_evidence_monitor as monitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / 'runs/study'
        self.study.mkdir(parents=True)
        self.source = self.root / 'reader.py'
        self.source.write_text('fixed source')
        self.write(self.study/'plan.json', {'source_sha256': {'reader.py': monitor.sha256(self.source)}})
        self.write(self.root/'runs/study-progress.json', {'stage': 'training'})
        self.log = self.root/'runs/study.log'
        self.log.write_text('')
        self.clock = 10000
        self.lock = patch.object(monitor, 'lock_held', return_value=True)
        self.lock.start()
        self.addCleanup(self.lock.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def checkpoint(self, name='2026/raw-lr0', modified=9900):
        path = self.study/name/'latest.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'Not a Torch checkpoint: the monitor must never deserialize it')
        os.utime(path, (modified, modified))
        return path

    def snapshot(self):
        return monitor.snapshot(self.study, now=self.clock)

    def test_healthy_active_and_queued_do_not_read_checkpoint_contents(self):
        self.checkpoint()
        result = self.snapshot()
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['trajectories']['2026/raw-lr0']['state'], 'active')
        self.assertEqual(result['trajectories']['2027/raw-lr0']['state'], 'queued')
        self.assertFalse(result['private_outcome_files_read'])

    def test_stale_one_worker_is_flagged_even_with_other_work(self):
        self.checkpoint(modified=100)
        self.checkpoint('2026/raw-lr1', modified=9999)
        result = self.snapshot()
        self.assertEqual(result['status'], 'attention_required')
        self.assertEqual(result['trajectories']['2026/raw-lr0']['state'], 'stale')
        self.assertEqual(result['trajectories']['2026/raw-lr1']['state'], 'active')

    def test_completed_and_queued_extension_are_not_stale(self):
        self.checkpoint(modified=100)
        self.write(self.study/'2026/raw-lr0/result.json', {'completed_epoch': 300})
        self.assertEqual(self.snapshot()['trajectories']['2026/raw-lr0']['state'], 'complete')
        self.write(self.study/'extension.json', {'target_epochs': 600})
        os.utime(self.study/'extension.json', (9000, 9000))
        self.assertEqual(self.snapshot()['trajectories']['2026/raw-lr0']['state'], 'queued')

    def test_source_change_requires_attention(self):
        self.source.write_text('changed')
        self.assertIn('Frozen source changed: reader.py', self.snapshot()['alerts'])

    def test_nonfinite_loss_is_reported_and_status_remains_serializable(self):
        self.log.write_text(json.dumps({'seed':2026,'run':'raw-lr0','training':{
            'epoch':1,'queries':1886,'seconds':100,'mean_query_loss':float('nan')}})+'\n')
        result = self.snapshot()
        self.assertEqual(result['status'], 'attention_required')
        json.dumps(result, allow_nan=False)

    def test_partial_line_ignored_and_resumed_errors_are_historical(self):
        self.log.write_text('Traceback (most recent call last):\n'+json.dumps({
            'stage':'training','updated_utc':'new'})+'\n{"incomplete":')
        self.assertEqual(self.snapshot()['alerts'], [])

    def test_completion_requires_valid_public_artifacts(self):
        self.write(self.root/'runs/study-progress.json', {'stage':'assessment_complete'})
        self.assertEqual(self.snapshot()['status'], 'attention_required')
        self.write(self.study/'aggregates.json', {'status':'complete','original_test_read':False})
        public = self.root/'exploratory/conditional_evidence/results-v1'
        self.write(public/'aggregates.json', {'status':'complete'})
        for name in ('SHA256.json','RESULTS-SHA256.json'):
            self.write(public/name, {'aggregates.json': monitor.sha256(public/'aggregates.json')})
        self.assertEqual(self.snapshot()['status'], 'completed')

    def test_missing_workflow_lock_is_not_called_healthy(self):
        with patch.object(monitor,'lock_held',return_value=False):
            self.assertEqual(self.snapshot()['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
