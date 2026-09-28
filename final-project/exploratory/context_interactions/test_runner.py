"""Synthetic data only: selection independence, seal order, ties and exports."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exploratory.context_interactions import run_experiment as run


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources, self.cohorts = self.root/'sources', self.root/'cohorts'
        self.users = [f'u{i}' for i in range(8)]
        self.items = ['[PAD]']+[f'i{i}' for i in range(24)]
        self.parts = {'train': [], 'valid': []}
        records = []
        for index, user in enumerate(self.users):
            for j in range(7):
                item = self.items[1+(index*2+j)%24]
                if j < 6:
                    self.parts['train' if j < 5 else 'valid'].append((user, item))
                records.append((user, item, str(1+(index+j)%5) if j < 6 else 'FORBIDDEN_TEST_CATEGORY'))
        self.ratings, self.metadata = self.root/'tiny.inter', self.root/'tiny.item'
        self.ratings.write_text('user_id:token\titem_id:token\trating:float\n'+''.join(f'{u}\t{i}\t{r}\n' for u, i, r in records))
        self.metadata.write_text('item_id:token\tclass:token_seq\n'+''.join(f'{item}\tDrama\n' for item in self.items[1:]))
        for seed in (7, 8):
            source, cohort = self.sources/f'{seed}-EASE-1', self.cohorts/str(seed)
            source.mkdir(parents=True); cohort.mkdir(parents=True)
            for part, pairs in self.parts.items():
                data = 'user_id\titem_id\n'+''.join(f'{u}\t{i}\n' for u, i in pairs)
                (source/f'{part}.tsv').write_text(data); (cohort/f'{part}.tsv').write_text(data)
            run.write_json(source/'ids.json', {'padding_index': 0, 'users': ['[PAD]']+self.users, 'items': self.items})
            np.savez(source/'valid-scores.npz', users=self.users, items=self.items,
                scores=np.zeros((len(self.users), len(self.items))))
            run.write_json(source/'manifest.json', {'status': 'complete', 'test_evaluated': False,
                'validation_used_for_training': False, 'data_sha256': {p.name: run.digest(p) for p in (self.ratings, self.metadata)},
                'split_sha256': {part: run.digest(source/f'{part}.tsv') for part in self.parts}})
            fit, dev = run.partition_users(self.users, seed)
            run.write_json(cohort/'cohorts.json', {'meta_fit': sorted(fit), 'development': sorted(dev)})
        self.patches = [patch.object(run, 'LAMBDAS', (.5, 2.)), patch.object(run, 'PAIR_WEIGHTS', (1.,)),
            patch.object(run, 'source_hashes', return_value={'synthetic.py': 'fixed'}),
            patch.object(run.categorical, 'verify_barrier', return_value={})]
        for item in self.patches:
            item.start()
        self.reference = self.root/'reference'
        self.reference.mkdir()
        run.write_json(self.reference/'SELECTIONS-FROZEN.json', {'synthetic': True})
        self.make_references()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def make_references(self):
        signatures, files = {}, {}
        for seed in (7, 8):
            source = self.sources/f'{seed}-EASE-1'
            users, items, train, valid, matrix, fit, _, signature = run.load_selection_inputs(
                source, self.ratings, self.metadata, self.cohorts/str(seed), seed)
            signatures[str(seed)] = signature
            prepared = run.prepared_model(matrix)
            fitted = [(penalty, run.solve(prepared, penalty, 0)[0]) for penalty in run.LAMBDAS]
            penalty, scores = max(fitted, key=lambda row: run.meta_ndcg(row[1], users, items, train, valid, fit))
            directory = self.reference/str(seed)
            directory.mkdir(exist_ok=True)
            run.write_json(directory/'selection-manifest.json', {'synthetic': True})
            rows = {}
            for name in ('binary_expanded', 'categorical'):
                path = directory/f'{name}.npy'
                np.save(path, scores, allow_pickle=False)
                rows[name] = {'candidate_id': 'synthetic-binary', 'penalty': penalty, 'category_ratio': None,
                    'feature_kind': 'binary', 'scores_file': path.name, 'scores_file_sha256': run.digest(path),
                    'scores_array_sha256': run.array_digest(scores)}
                files[path.relative_to(self.reference).as_posix()] = run.digest(path)
            run.write_json(directory/'selection.json', rows)
        run.write_json(self.reference/'manifest.json', {'status': 'complete', 'test_read': False, 'test_evaluated': False,
            'selection_freeze_sha256': run.digest(self.reference/'SELECTIONS-FROZEN.json'),
            'input_signatures': signatures, 'output_sha256': files})

    def research(self, name='research'):
        path = self.root/name
        path.mkdir()
        run.write_json(path/'protocol.json', run.protocol([7, 8]))
        run.write_json(path/'reference-input-signature.json', {
            'manifest_sha256': run.digest(self.reference/'manifest.json'),
            'selection_freeze_sha256': run.digest(self.reference/'SELECTIONS-FROZEN.json')})
        return path

    def select(self, path, seed):
        with redirect_stdout(io.StringIO()):
            return run.select_seed(self.sources/f'{seed}-EASE-1', self.ratings, self.metadata,
                self.cohorts/str(seed), self.reference, path/str(seed), seed, run.protocol([7, 8]))

    def update_raw_hash(self):
        for seed in (7, 8):
            path = self.sources/f'{seed}-EASE-1'/'manifest.json'
            row = json.loads(path.read_text()); row['data_sha256'][self.ratings.name] = run.digest(self.ratings)
            run.write_json(path, row)

    def test_selection_sees_only_meta_pairs_and_never_parses_nontrain_ratings(self):
        baseline = self.research('baseline'); self.select(baseline, 7)
        train = set(self.parts['train'])
        lines = self.ratings.read_text().splitlines()
        self.ratings.write_text('\n'.join([lines[0]]+[line if tuple(line.split('\t')[:2]) in train
            else line.rsplit('\t', 1)[0]+'\tDO_NOT_PARSE' for line in lines[1:]])+'\n')
        self.update_raw_hash(); self.make_references()
        changed = self.research('changed')
        original = run.meta_ndcg
        def guarded(scores, users, items, training, validation, fit, *args):
            self.assertEqual(set(u for u, _ in validation), set(fit))
            return original(scores, users, items, training, validation, fit, *args)
        with patch.object(run, 'meta_ndcg', side_effect=guarded), patch.object(run, 'load_source', side_effect=AssertionError('Selection opened VALID ratings')):
            self.select(changed, 7)
        first = json.loads((baseline/'7/selection.json').read_text())
        second = json.loads((changed/'7/selection.json').read_text())
        for role in run.MODEL_NAMES:
            self.assertEqual(first[role]['candidate_id'], second[role]['candidate_id'])
            self.assertEqual(first[role]['scores_array_sha256'], second[role]['scores_array_sha256'])

    def test_global_barrier_blocks_dev_until_all_seeds_and_exports_only_aggregates(self):
        path = self.research(); self.select(path, 7)
        with self.assertRaises(FileNotFoundError):
            run.freeze_selections(path)
        with patch.object(run, 'load_source', side_effect=AssertionError('DEV opened early')):
            with self.assertRaises(FileNotFoundError):
                run.evaluate_selected(path, self.sources, self.ratings, self.metadata)
        self.assertFalse((path/'DEVELOPMENT-OPENED.json').exists())
        self.select(path, 8); run.freeze_selections(path)
        original, calls = run.load_source, []
        def guarded(*args):
            self.assertEqual(set(run.verify_barrier(path)['selection_manifest_sha256']), {'7', '8'})
            calls.append(1)
            return original(*args)
        with patch.object(run, 'load_source', side_effect=guarded):
            result = run.evaluate_selected(path, self.sources, self.ratings, self.metadata)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result['fresh_confirmation'])
        self.assertEqual(set(result['seeds']['7']['models']), set(run.MODEL_NAMES))
        row = result['seeds']['7']['models']['pair_only']
        self.assertEqual(row['groups']['users'], row['denominators']['all_observed_users'])
        self.assertEqual(row['liked_groups']['users'], row['denominators']['liked_ratings_users'])
        evidence = self.root/'evidence'; run.curate(path, evidence)
        self.assertNotIn('per_user', (evidence/'aggregates.json').read_text())
        self.assertNotIn('solver_diagnostics', (evidence/'aggregates.json').read_text())
        for name, signature in json.loads((evidence/'SHA256.json').read_text()).items():
            self.assertEqual(signature, run.digest(evidence/name))
        with self.assertRaises(FileExistsError):
            run.evaluate_selected(path, self.sources, self.ratings, self.metadata)

    def test_exact_ties_and_candidate_completeness(self):
        rows = [{'penalty': penalty, 'pair_weight': weight, 'meta_fit_all_observed_ndcg@10': .5,
                 'candidate_id': run.candidate_id(penalty, weight)} for penalty, weight in run.candidates()]
        self.assertIs(run.choose(rows, 'binary'), rows[0])
        self.assertIs(run.choose(rows, 'nested'), rows[0])
        self.assertIs(run.choose(rows, 'pair_only'), rows[2])
        path = self.research(); self.select(path, 7)
        declared = run.protocol([7, 8]); run.validate_choices(path/'7', declared)
        grid_path = path/'7/candidate-grid.json'; grid = json.loads(grid_path.read_text())
        run.write_json(grid_path, grid[:-1])
        with self.assertRaisesRegex(ValueError, 'candidate grid'):
            run.validate_choices(path/'7', declared)
        run.write_json(grid_path, grid)
        selected_path = path/'7/selection.json'; selected = json.loads(selected_path.read_text())
        selected['nested']['candidate_id'] = 'wrong'
        run.write_json(selected_path, selected)
        with self.assertRaisesRegex(ValueError, 'exact maximum'):
            run.validate_choices(path/'7', declared)

    def test_tamper_or_source_change_prevents_any_development_label_read(self):
        path = self.research(); self.select(path, 7); self.select(path, 8); run.freeze_selections(path)
        with patch.object(run, 'source_hashes', return_value={'synthetic.py': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'barrier'):
                run.verify_barrier(path)
        scores_path = path/'7/selected-scores/pair_only.npy'
        scores = np.load(scores_path, allow_pickle=False); scores[0, 1] += 1
        np.save(scores_path, scores, allow_pickle=False)
        with patch.object(run, 'load_source', side_effect=AssertionError('DEV opened after tamper')):
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run.evaluate_selected(path, self.sources, self.ratings, self.metadata)
        self.assertFalse((path/'DEVELOPMENT-OPENED.json').exists())

    def test_benchmark_reads_no_validation_artifact(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name in ('valid.tsv', 'test.tsv', 'valid-scores.npz'):
                raise AssertionError('TRAIN benchmark opened non-TRAIN artifact')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            result = run.benchmark(self.sources/'7-EASE-1', self.ratings, self.root/'timing.json')
        self.assertFalse(result['validation_read']); self.assertFalse(result['test_read'])


if __name__ == '__main__':
    unittest.main()
