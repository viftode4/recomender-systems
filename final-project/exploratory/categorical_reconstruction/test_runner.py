import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exploratory.categorical_reconstruction import run_experiment as run
from study import digest, write_json


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root, self.cohort_root = self.root/'sources', self.root/'cohorts'
        self.source_root.mkdir(); self.cohort_root.mkdir()
        self.users = [f'u{i}' for i in range(8)]
        self.items = ['[PAD]']+[f'i{i}' for i in range(24)]
        self.pairs = {'train': [], 'valid': [], 'test': []}
        rows = []
        for u, user in enumerate(self.users):
            for j in range(7):
                item = self.items[1+(2*u+j)%24]
                part = 'train' if j < 5 else 'valid' if j == 5 else 'test'
                self.pairs[part].append((user, item))
                rows.append((user, item, str(1+(u+j)%5) if part != 'test' else 'NEVER_PARSE_TEST'))
        self.ratings = self.root/'tiny.inter'
        self.ratings.write_text('user_id:token\titem_id:token\trating:float\n'+''.join(f'{u}\t{i}\t{r}\n' for u, i, r in rows))
        self.metadata = self.root/'tiny.item'
        self.metadata.write_text('item_id:token\tclass:token_seq\n'+''.join(f'{i}\tDrama\n' for i in self.items[1:]))
        for seed in (7, 8):
            source = self.source_root/f'{seed}-EASE-1'; source.mkdir()
            cohort = self.cohort_root/str(seed); cohort.mkdir()
            for part, pairs in self.pairs.items():
                payload = 'user_id\titem_id\n'+''.join(f'{u}\t{i}\n' for u, i in pairs)
                (source/f'{part}.tsv').write_text(payload)
                if part != 'test': (cohort/f'{part}.tsv').write_text(payload)
            write_json(source/'ids.json', {'padding_index': 0, 'users': ['[PAD]']+self.users, 'items': self.items})
            np.savez(source/'valid-scores.npz', users=self.users, items=self.items,
                scores=np.tile(np.arange(len(self.items), dtype=float), (len(self.users), 1)))
            write_json(source/'manifest.json', {'status': 'complete', 'test_evaluated': False,
                'validation_used_for_training': False, 'settings': {},
                'data_sha256': {p.name: digest(p) for p in (self.ratings, self.metadata)},
                'split_sha256': {part: digest(source/f'{part}.tsv') for part in self.pairs}})
            fit, dev = run.partition_users(self.users, seed)
            write_json(cohort/'cohorts.json', {'meta_fit': sorted(fit), 'development': sorted(dev)})
            write_json(self.source_root/f'expert-selection-{seed}.json', {'SLIMElastic': {'selected': {
                'path': str(source), 'settings': {}, 'meta_fit_ndcg': .1}}})
        self.patches = [patch.object(run, 'source_hashes', return_value={'synthetic.py': 'fixed'}),
            patch.object(run, 'LAMBDAS', (.5, 2.)), patch.object(run, 'ORIGINAL_LAMBDAS', (.5,)),
            patch.object(run, 'CATEGORY_RATIOS', (1.,))]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def research(self, name='research'):
        root = self.root/name; root.mkdir()
        write_json(root/'protocol.json', run.protocol([7, 8]))
        return root

    def select(self, research, seed):
        source = self.source_root/f'{seed}-EASE-1'
        with redirect_stdout(io.StringIO()):
            _, selected = run.reference_path(self.source_root, seed, 'SLIMElastic')
            return run.select_seed(source, self.ratings, self.metadata, self.cohort_root/str(seed),
                source, research/str(seed), seed, run.protocol([7, 8]),
                {**selected, 'selection_file': str(self.source_root/f'expert-selection-{seed}.json')})

    def references(self):
        evidence, raw = self.root/'reference-evidence', self.root/'reference-run'
        evidence.mkdir(); raw.mkdir()
        seeds = {}
        for seed in (7, 8):
            source = self.source_root/f'{seed}-EASE-1'
            loaded = run.load_source(source, self.ratings, self.metadata)
            users, items, train, _, training, labels = loaded[:6]
            matrix = run.categorical_matrix(users, items, training)
            _, dev = run.partition_users(users, seed)
            recs = run.recommendations(loaded[7], users, items, matrix != 0, 10)
            metric = run.evaluate_ratings(recs, dev, labels, run.grouped(train), items[1:], Counter(i for _, i in train), 10)
            (raw/str(seed)).mkdir(); write_json(raw/str(seed)/'details.json', {})
            rows = {name: {'endpoints': {endpoint: {'metrics': metric[endpoint]['aggregate'],
                'denominators': metric['denominators']} for endpoint in ('all_observed', 'liked_ratings')}}
                for name in ('EASE', 'SLIMElastic', 'PositiveEASE')}
            _, selected = run.reference_path(self.source_root, seed, 'SLIMElastic')
            rows['SLIMElastic']['selection'] = selected
            _, _, slim = run.load_scores(source, loaded[-1], users, items, digest(source/'ids.json'))
            seeds[str(seed)] = {'rows': rows, 'data_sha256': loaded[-1]['data_sha256'],
                'provenance': {'SLIMElastic': {endpoint: slim for endpoint in ('all_observed', 'liked_ratings')}},
                'split_sha256': {part: loaded[-1]['split_sha256'][part] for part in ('train', 'valid')},
                'ordered_identity_sha256': run.identity_digest(users, items),
                'cohort_file_sha256': digest(self.cohort_root/str(seed)/'cohorts.json'),
                'details_sha256': digest(raw/str(seed)/'details.json')}
        write_json(evidence/'aggregates.json', {'seeds': seeds})
        write_json(raw/'manifest.json', {'status': 'complete', 'test_read': False, 'test_evaluated': False,
            'source_sha256': {}, 'evidence_sha256': {'aggregates.json': digest(evidence/'aggregates.json')}})
        return evidence, raw

    def test_benchmark_never_opens_validation_arrays_or_splits_or_parses_nontrain_categories(self):
        source = self.source_root/'7-EASE-1'
        allowed = set(self.pairs['train'])
        text = self.ratings.read_text().splitlines()
        self.ratings.write_text('\n'.join([text[0]]+[line if tuple(line.split('\t')[:2]) in allowed
            else line.rsplit('\t', 1)[0]+'\tUNPARSED_NONTRAIN' for line in text[1:]])+'\n')
        manifest = json.loads((source/'manifest.json').read_text())
        manifest['data_sha256'][self.ratings.name] = digest(self.ratings); write_json(source/'manifest.json', manifest)
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name in ('valid.tsv', 'test.tsv', 'valid-scores.npz'):
                raise AssertionError('Benchmark opened non-TRAIN artifact')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            result = run.benchmark(source, self.ratings, self.root/'benchmark.json')
        self.assertFalse(result['validation_read']); self.assertFalse(result['test_read'])

    def test_selection_does_not_parse_any_validation_categories_and_replay_is_exact(self):
        first = self.research('first'); self.select(first, 7)
        original = json.loads((first/'7/selection.json').read_text())
        train = set(self.pairs['train'])
        text = self.ratings.read_text().splitlines()
        self.ratings.write_text('\n'.join([text[0]]+[line if tuple(line.split('\t')[:2]) in train
            else line.rsplit('\t', 1)[0]+'\tPOISONED_VALIDATION_CATEGORY' for line in text[1:]])+'\n')
        for seed in (7, 8):
            path = self.source_root/f'{seed}-EASE-1'/'manifest.json'
            manifest = json.loads(path.read_text()); manifest['data_sha256'][self.ratings.name] = digest(self.ratings)
            write_json(path, manifest)
        second = self.research('second'); self.select(second, 7)
        changed = json.loads((second/'7/selection.json').read_text())
        for name in run.MODEL_NAMES:
            self.assertEqual(original[name]['candidate_id'], changed[name]['candidate_id'])
            self.assertEqual(original[name]['scores_array_sha256'], changed[name]['scores_array_sha256'])
            self.assertTrue(changed[name]['refit_replay_exact'])
        self.assertEqual(json.loads((first/'7/hybrid-selection.json').read_text()),
                         json.loads((second/'7/hybrid-selection.json').read_text()))

    def test_global_barrier_precedes_every_development_call_and_curation_is_aggregate_only(self):
        references, raw = self.references()
        research = self.research()
        self.select(research, 7)
        with self.assertRaises(FileNotFoundError): run.freeze_selections(research)
        self.assertFalse((research/'SELECTIONS-FROZEN.json').exists())
        self.select(research, 8); run.freeze_selections(research)
        original_open, original_evaluate = Path.open, run.evaluate_ratings
        calls = []
        def guarded(path, *args, **kwargs):
            if path.name == 'test.tsv' or path.name.startswith('final-'):
                raise AssertionError('Opened forbidden TEST/final artifact')
            return original_open(path, *args, **kwargs)
        def checked(*args, **kwargs):
            barrier = json.loads((research/'SELECTIONS-FROZEN.json').read_text())
            self.assertEqual(set(barrier['selection_manifest_sha256']), {'7', '8'})
            calls.append(1)
            return original_evaluate(*args, **kwargs)
        with patch.object(Path, 'open', guarded), patch.object(run, 'evaluate_ratings', side_effect=checked):
            result = run.evaluate_selected(research, self.source_root, self.ratings, self.metadata, references, raw)
        self.assertEqual(len(calls), 2*len(run.MODEL_NAMES))
        self.assertFalse(result['fresh_confirmation'])
        self.assertEqual(set(result['seeds']['7']['models']['categorical']['groups']['user_groups']), {'sparse', 'medium', 'dense'})
        evidence = self.root/'curated'; run.curate(research, evidence)
        self.assertNotIn('per_user', (evidence/'aggregates.json').read_text())
        self.assertNotIn('fallback_items', (evidence/'aggregates.json').read_text())
        for name, signature in json.loads((evidence/'SHA256.json').read_text()).items():
            self.assertEqual(digest(evidence/name), signature)
        with self.assertRaises(FileExistsError):
            run.evaluate_selected(research, self.source_root, self.ratings, self.metadata, references, raw)

    def test_grid_completeness_and_exact_first_tie_are_independently_checked(self):
        research = self.research(); self.select(research, 7)
        folder = research/'7'; declared = run.protocol([7, 8])
        run.validate_choices(folder, declared)
        path = folder/'candidate-grid.json'; grid = json.loads(path.read_text())
        write_json(path, grid[:-1])
        with self.assertRaisesRegex(ValueError, 'candidate grid'): run.validate_choices(folder, declared)
        write_json(path, grid)
        selections_path = folder/'selection.json'; selected = json.loads(selections_path.read_text())
        selected['binary_expanded']['candidate_id'] = 'different'
        write_json(selections_path, selected)
        with self.assertRaisesRegex(ValueError, 'exact maximum'): run.validate_choices(folder, declared)

    def test_development_pair_identity_changes_cannot_change_any_selection(self):
        first = self.research('before-pair-poison'); self.select(first, 7)
        selected_before = json.loads((first/'7/selection.json').read_text())
        source = self.source_root/'7-EASE-1'
        _, dev = run.partition_users(self.users, 7)
        history = run.grouped(self.pairs['train'])
        changed = [(u, next(i for i in self.items[1:] if i not in history[u] and i != item))
                   if u in dev else (u, item) for u, item in self.pairs['valid']]
        payload = 'user_id\titem_id\n'+''.join(f'{u}\t{i}\n' for u, i in changed)
        (source/'valid.tsv').write_text(payload)
        (self.cohort_root/'7/valid.tsv').write_text(payload)
        manifest = json.loads((source/'manifest.json').read_text())
        manifest['split_sha256']['valid'] = digest(source/'valid.tsv'); write_json(source/'manifest.json', manifest)
        second = self.research('after-pair-poison'); self.select(second, 7)
        selected_after = json.loads((second/'7/selection.json').read_text())
        for name in run.MODEL_NAMES:
            self.assertEqual(selected_before[name]['candidate_id'], selected_after[name]['candidate_id'])
            self.assertEqual(selected_before[name]['scores_array_sha256'], selected_after[name]['scores_array_sha256'])

    def test_changed_score_or_source_is_rejected_before_development(self):
        research = self.research(); self.select(research, 7); self.select(research, 8)
        run.freeze_selections(research)
        with patch.object(run, 'source_hashes', return_value={'synthetic.py': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'barrier'): run.verify_barrier(research)
        selected = json.loads((research/'7/selection.json').read_text())['categorical']
        path = research/'7'/selected['scores_file']; scores = np.load(path)
        scores[0, 1] += 1; np.save(path, scores)
        with self.assertRaisesRegex(ValueError, 'artifact changed'): run.verify_barrier(research)
        self.assertFalse((research/'DEVELOPMENT-OPENED.json').exists())

    def test_shuffle_preserves_presence_and_item_category_histograms(self):
        _, _, _, matrix, _ = run.load_train(self.source_root/'7-EASE-1', self.ratings)
        shuffled = run.shuffle_categories(matrix, 7)
        np.testing.assert_array_equal(matrix != 0, shuffled != 0)
        for rating in range(1, 6):
            np.testing.assert_array_equal((matrix == rating).sum(0), (shuffled == rating).sum(0))
        np.testing.assert_array_equal(shuffled, run.shuffle_categories(matrix, 7))


if __name__ == '__main__':
    unittest.main()
