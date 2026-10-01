"""Workflow recovery and stage boundaries on synthetic inputs only."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import data, evaluation, model, retrieval, run_experiment as run, workflow


class WorkflowTests(unittest.TestCase):
    def test_compute_report_sums_costs_and_retains_both_curves(self):
        history = {'parameter_count': 5553, 'epochs': [
            {'epoch': 1, 'mean_query_loss': 4., 'seconds': 2., 'resources': {'nodes': 10}},
            {'epoch': 2, 'mean_query_loss': 3., 'seconds': 3., 'resources': {'nodes': 20}}],
            'checkpoints': [{'epoch': 0, 'meta_ndcg@10': .1, 'resources': {'seconds': 7.}},
                            {'epoch': 2, 'meta_ndcg@10': .2, 'resources': {'seconds': 11.}}]}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run.atomic_json(directory / '2026/raw-lr0/result.json', {'history': history,
                'config': {'arm': 'raw'}, 'completed_epoch': 2, 'peak_process_memory_bytes': 1234})
            run.atomic_json(directory / '2026/reference-metadata.json', {'candidate_count': 12})
            with patch.object(run, 'SEEDS', (2026,)), patch.object(run, 'ARMS', ('raw',)), \
                 patch.object(run, 'LEARNING_RATES', (.001,)):
                result = workflow.compute_report(directory)
            row = result['trajectories']['2026/raw-lr0']
            self.assertEqual(row['training_seconds'], 5.)
            self.assertEqual(row['meta_evaluation_seconds'], 18.)
            self.assertEqual(row['training_resources']['nodes'], 30)
            self.assertEqual(row['meta_curve'], history['checkpoints'])
            self.assertEqual(row['peak_process_memory_bytes'], 1234)

    def test_positive_gate_fits_seals_and_reuses_complete_retrieval(self):
        run.numerical_setup(2026)
        categories = np.random.default_rng(44).integers(0, 6, (18, 9), dtype=np.uint8)
        categories[:, 0] = 0
        categories[:, 1:3] = 4
        values = {'2026': {'raw': .4, 'summary': .2, 'scrambled': .1, 'EASE': .3}}
        screen = evaluation.meta_screen(values, 'raw', ('summary', 'scrambled'), ('EASE',))
        with tempfile.TemporaryDirectory() as tmp:
            args = workflow.parser().parse_args(['--out', str(Path(tmp) / 'study')])
            selected = {'config': {'seed': 2026, 'learning_rate': .001},
                        'selected': {'epoch': 0, 'checkpoint': 'best.pt'}}
            run.atomic_json(args.out / '2026/selection.json', {'raw': selected})
            run.atomic_json(args.out / 'SELECTIONS-FROZEN.json', {'synthetic': True})
            run.atomic_json(args.out / 'meta-screen.json', screen)
            directory = args.out / '2026' / run.config_id('raw', 0)
            reader = run.make_reader(model, 'raw', 2026)
            optimizer = torch.optim.Adam(reader.parameters(), lr=.001, weight_decay=1e-4)
            run.save_checkpoint(directory / 'best.pt', reader, optimizer, 0, {}, {})
            run.atomic_json(directory / 'result.json', {'guard': {}})
            seal = {'source_sha256': {}, 'meta_screen_sha256': 'synthetic'}
            with patch.object(run, 'SEEDS', (2026,)), \
                 patch.object(run, 'verify_global_seal', return_value=seal), \
                 patch.object(run, 'load_inputs', return_value={'categories': categories}):
                result = workflow.fit_retrieval(args)
                self.assertEqual(result['extension_status'], 'complete')
                self.assertEqual(set(result['extension_scores']['2026']), set(workflow.POLICIES))
                target = args.out / 'retrieval/2026'
                shift = json.loads((target / 'feature-shift.json').read_text())
                self.assertEqual(shift['deployment_action_count'], 18*8)
                self.assertEqual(shift['crossfit_bank_rows'], [12, 12, 12])
                self.assertEqual(len(shift['mean_shift_in_crossfit_scale']), 35)
                self.assertFalse(shift['outcome_labels_used'])
                with patch.object(retrieval, 'fit_extension', side_effect=AssertionError('unexpected refit')):
                    self.assertEqual(result, workflow.fit_retrieval(args))
                manifest_path = target / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                del manifest['artifacts']['learned.npy']
                run.atomic_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, 'Incomplete or mismatched'):
                    workflow.fit_retrieval(args)

    def test_retrieval_assessment_completed_resume_checks_bytes_without_outcome_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), release=Path(tmp) / 'release.json')
            run.atomic_json(args.release, {'synthetic': True})
            result_path = args.out / 'retrieval-assessment.json'
            run.atomic_json(result_path, {'public': {'synthetic': 1}})
            run.atomic_json(args.out / 'retrieval-assessment-complete.json', {
                'release_sha256': run.digest(args.release), 'result_sha256': run.digest(result_path)})
            with patch.object(run, 'verify_assessment_release'), \
                 patch.object(data, 'load_development', side_effect=AssertionError('unexpected DEV read')):
                self.assertEqual(workflow.assess_retrieval(args, {'extension_status': 'complete'}),
                                 {'synthetic': 1})
                run.atomic_json(result_path, {'public': {'changed': 1}})
                with self.assertRaisesRegex(ValueError, 'Completed retrieval assessment changed'):
                    workflow.assess_retrieval(args, {'extension_status': 'complete'})

    def test_curation_keeps_base_export_immutable_and_reports_only_final_boundary(self):
        roles = [*run.ARMS, 'EASEexpanded', 'categorical', 'SLIM', 'neighbor']
        metric = {'all_observed': {'ndcg@10': .1, 'recall@10': .2, 'mrr@10': .3}}
        aggregate = {'status': 'complete', 'original_test_read': False, 'seeds': {
            '2026': {'models': {role: metric for role in roles}, 'selections': {
                'raw': {'selected': {'epoch': 300}}, 'scrambled': {'selected': {'epoch': 600}}}}},
            'development_gate': {'strongest_meta_reference': 'EASEexpanded', 'substantial_target_met': False}}
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), evidence=Path(tmp) / 'evidence', release=Path(tmp) / 'release')
            run.atomic_json(args.out / 'aggregates.json', aggregate)
            run.atomic_json(args.out / 'training-summary.json', {'extension': {'target_epochs': 600}})
            run.export_evidence(args.evidence, aggregate, {}, {})
            initial = run.digest(args.evidence / 'SHA256.json')
            with patch.object(run, 'verify_assessment_release'), \
                 patch.object(workflow, 'compute_report', return_value={'synthetic': True}):
                workflow.curate(args)
                workflow.curate(args)
            run.export_evidence(args.evidence, aggregate, {}, {})
            self.assertEqual(initial, run.digest(args.evidence / 'SHA256.json'))
            report = (args.evidence / 'RESULTS.md').read_text()
            self.assertNotIn('2026/raw: 300', report)
            self.assertIn('2026/scrambled: 600', report)

    def test_closed_gate_is_recomputed_and_cannot_skip_positive_gate(self):
        values = {str(seed): {'raw': .1, 'summary': .11, 'scrambled': .09, 'EASE': .2}
                  for seed in run.SEEDS}
        screen = evaluation.meta_screen(values, 'raw', ('summary', 'scrambled'), ('EASE',))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            run.atomic_json(directory / 'meta-screen.json', screen)
            run.atomic_json(directory / 'SELECTIONS-FROZEN.json', {'synthetic': True})
            with patch.object(run, 'verify_global_seal', return_value={'meta_screen_sha256': 'screen'}):
                result = workflow.release_for_closed_gate(directory)
                self.assertEqual(result['extension_status'], 'not_triggered')
                screen['passed'] = True
                run.atomic_json(directory / 'meta-screen.json', screen)
                with self.assertRaises(ValueError):
                    workflow.release_for_closed_gate(directory)

    def test_fold_callback_resumes_exactly_and_keeps_initialization_separate(self):
        run.numerical_setup(1)
        categories = np.array([[0,5,4,0,2,0], [0,0,3,4,0,1],
                               [0,2,0,4,1,0], [0,0,4,0,3,2]], dtype=np.uint8)
        request = retrieval.FoldTrainingRequest(0, categories, np.arange(4), (),
                         {'seed': 7, 'learning_rate': .001}, 2, 920)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            straight = workflow.fit_fold(request, root / 'straight', {})
            original = run.train_epoch
            def interrupted(*args, **kwargs):
                if args[5] == 2:
                    raise KeyboardInterrupt()
                return original(*args, **kwargs)
            with patch.object(run, 'train_epoch', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    workflow.fit_fold(request, root / 'resumed', {})
            resumed = workflow.fit_fold(request, root / 'resumed', {})
            self.assertEqual(resumed.diagnostics['initialization_seed'], 7)
            self.assertEqual(resumed.diagnostics['training_seed'], 920)
            for name, value in straight.reader.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.reader.state_dict()[name]), name)
            # Completed folds replay without doing extra optimizer steps.
            with patch.object(run, 'train_epoch', side_effect=AssertionError('unexpected refit')):
                replay = workflow.fit_fold(request, root / 'resumed', {})
            self.assertEqual(replay.completed_epochs, 2)

    def test_retrieval_failure_prevents_development_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = workflow.parser().parse_args(['--out', str(Path(tmp) / 'study')])
            def train(_):
                args.out.mkdir()
                run.atomic_json(args.out / 'training-summary.json', {})
            def seal(_):
                run.atomic_json(args.out / 'SELECTIONS-FROZEN.json', {})
            with patch.object(run, 'execute_train', side_effect=train), \
                 patch.object(run, 'execute_seal', side_effect=seal), \
                 patch.object(run, 'verify_global_seal', return_value={}), \
                 patch.object(workflow, 'fit_retrieval', side_effect=ValueError('unsealed retrieval')), \
                 patch.object(run, 'execute_assess') as assess:
                with self.assertRaisesRegex(ValueError, 'unsealed retrieval'):
                    workflow.execute(args)
                assess.assert_not_called()
            progress = json.loads((Path(tmp) / 'study-progress.json').read_text())
            self.assertEqual(progress['stage'], 'failed')
            self.assertFalse(progress['original_test_read'])


if __name__ == '__main__':
    unittest.main()
