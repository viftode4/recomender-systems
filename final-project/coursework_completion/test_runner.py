"""Independent label-access and selection-barrier tests for the new runner."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from coursework_completion import run


def canonical_pairs(value):
    if isinstance(value, dict):
        return {(user, item) for user, items in value.items() for item in items}
    return set(map(tuple, value))


class LabelBoundaryTests(unittest.TestCase):
    def test_skipped_calibration_item_is_not_interpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'valid.tsv'
            path.write_text('user_id\titem_id\nmeta\t1\ncalibration\tPOISON\n'
                            'development\t2\ncalibration\n')
            actual = run.read_valid_subset(path, {'meta', 'development'}, {'1', '2'})
            self.assertEqual(canonical_pairs(actual), {('meta', '1'), ('development', '2')})

    def test_duplicate_or_unknown_allowed_items_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'valid.tsv'
            for rows in ('meta\t1\nmeta\t1\n', 'meta\tPOISON\n'):
                path.write_text('user_id\titem_id\n' + rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    run.read_valid_subset(path, {'meta'}, {'1', '2'})

    def test_assessment_guard_runs_before_any_label_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run, 'verify_barrier', side_effect=ValueError('not frozen')) as barrier:
                with mock.patch.object(run, 'read_valid_subset', side_effect=AssertionError('labels read')) as reader:
                    with self.assertRaisesRegex(ValueError, 'not frozen'):
                        run.read_assessment(Path(tmp), 2026)
                reader.assert_not_called()
            barrier.assert_called_once()


def source_fixture(root):
    seed = root / '2026'
    (seed / 'provenance').mkdir(parents=True)
    users = np.array(['meta', 'development', 'calibration'])
    items = np.array(['[PAD]'] + [str(i) for i in range(1, 13)])
    observed = np.zeros((3, 13), dtype=bool)
    observed[np.arange(3), np.arange(1, 4)] = True
    genre = np.ones((13, 1)); genre[0] = 0
    arrays = {name: np.zeros(1) for name in run.SAFE_ARRAYS}
    arrays.update(users=users, items=items, train_mask=observed,
                  raw_scores=np.zeros((1, 3, 13)), raw_activity=observed.sum(axis=1),
                  training_counts=observed.sum(axis=0), genre_binary=genre.astype(bool),
                  genre_fraction=genre, valid_mask=np.ones((3, 13), dtype=bool))
    np.savez(seed / 'frozen.npz', **arrays)
    (seed / 'train.tsv').write_text('user_id\titem_id\nmeta\t1\ndevelopment\t2\ncalibration\t3\n')
    (seed / 'valid.tsv').write_text('user_id\titem_id\nmeta\t4\ndevelopment\t5\ncalibration\tPOISON\n')
    metadata = root / 'items.atomic'
    metadata.write_text('synthetic metadata; real parser separately checked against source in production')
    run.write_json(seed / 'provenance/manifest.json', {'metadata_sha256': run.digest(metadata)})
    run.write_json(seed / 'provenance/cohorts.json',
                   {'meta_fit': ['meta'], 'development': ['development'], 'calibration': ['calibration']})
    bundle = {'status': 'frozen', 'seed': 2026, 'test_read': False, 'code_sha256': {},
              'expert_order': ['fixture-EASE'], 'artifacts_sha256': {}}
    def reseal():
        bundle['artifacts_sha256'] = {p.relative_to(seed).as_posix(): run.digest(p)
            for p in seed.rglob('*') if p.is_file() and p.name not in ('freeze.json', 'freeze.sha256')}
        run.write_json(seed / 'freeze.json', bundle)
        (seed / 'freeze.sha256').write_text(run.digest(seed / 'freeze.json'))
        run.write_json(root / 'manifest.json', {'status': 'complete', 'test_read': False,
            'bundles': {'2026': {'freeze_sha256': run.digest(seed / 'freeze.json')}}})
        (root / 'manifest.sha256').write_text(run.digest(root / 'manifest.json'))
    reseal()
    return seed, arrays, metadata, genre, reseal


class SourceBoundaryTests(unittest.TestCase):
    def test_source_loader_never_reads_valid_mask_or_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed, arrays, metadata, genre, reseal = source_fixture(root)
            with np.load(seed / 'frozen.npz') as archive:
                archive_type = type(archive)
            original_getitem = archive_type.__getitem__
            original_open = Path.open
            touched = []
            def getitem(archive, name):
                touched.append(name)
                if name == 'valid_mask':
                    raise AssertionError('calibration membership array read before seal')
                return original_getitem(archive, name)
            def guarded_open(path, *args, **kwargs):
                if Path(path).name == 'test.tsv':
                    raise AssertionError('TEST access')
                return original_open(path, *args, **kwargs)
            with mock.patch.object(archive_type, '__getitem__', getitem), \
                 mock.patch.object(Path, 'open', guarded_open), \
                 mock.patch.object(run, 'load_genres', return_value=(genre.astype(bool), genre)):
                loaded = run.load_seed(root, 2026, metadata)
            self.assertEqual(set(loaded['truth']), {'meta', 'development'})
            self.assertNotIn('valid_mask', touched)
            self.assertEqual(set(touched), set(run.SAFE_ARRAYS))

    def test_source_hash_and_shared_train_user_alignment_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed, arrays, metadata, genre, reseal = source_fixture(root)
            with (seed / 'train.tsv').open('a') as stream:
                stream.write('meta\t6\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run.load_seed(root, 2026, metadata)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed, arrays, metadata, genre, reseal = source_fixture(root)
            arrays['users'] = arrays['users'][::-1]
            np.savez(seed / 'frozen.npz', **arrays)
            reseal()
            with mock.patch.object(run, 'load_genres', return_value=(genre.astype(bool), genre)):
                with self.assertRaisesRegex(ValueError, 'TRAIN matrix differs'):
                    run.load_seed(root, 2026, metadata)


def selection_fixture(root):
    run.write_json(root / 'protocol.json', {'source_sha256': run.source_hashes(),
        'runtime': run.runtime(), 'input_artifacts_sha256': {}})
    for seed in run.SEEDS:
        directory = root / str(seed)
        directory.mkdir()
        users = [f'u{i}' for i in range(943)]
        run.write_json(directory / 'ids.json', {'users': users, 'items': ['[PAD]'] + list(map(str, range(1, 13)))})
        run.write_json(directory / 'cohorts.json', {'meta_fit': users[:471], 'development': users[471:707], 'calibration': users[707:]})
        run.write_json(directory / 'input-signature.json', {})
        candidates, choices = [], {}
        for family in run.FAMILIES:
            for index, setting in enumerate(run.GRIDS[family]):
                settings = ({'quotas': list(setting)} if family == 'mixed' else
                            {'lambda': setting} if family == 'meta' else
                            {'group_count': setting} if family == 'tuned_switch' else {'offset': setting})
                row = {'family': family, 'candidate_id': f'{family}-{index}', 'settings': settings,
                       'development': {'users': 236, 'aggregate': {'ndcg@10': .1}}}
                candidates.append(row)
                choices.setdefault(family, row)
        run.write_json(directory / 'candidate-grid.json', candidates)
        run.write_json(directory / 'selections.json', {'families': choices,
            'references': {role: 'synthetic frozen reference' for role in run.REFERENCES}})
        np.savez(directory / 'predictions.npz', **{role: np.tile(np.arange(1, 11), (943, 1)) for role in run.ROLES})
        np.savez(directory / 'candidate-predictions.npz',
                 **{row['candidate_id']: np.tile(np.arange(1, 11), (943, 1)) for row in candidates})
        seal_selection(directory, seed)


def seal_selection(directory, seed):
    run.write_json(directory / 'selection-manifest.json', {
        'seed': seed, 'roles': list(run.ROLES), 'candidate_count': 13,
        'selection_cohort': 'development', 'assessment_cohort': 'reused calibration',
        'assessment_labels_parsed': False,
        'artifact_sha256': {p.name: run.digest(p) for p in directory.iterdir()
                           if p.is_file() and p.name != 'selection-manifest.json'}})


class SelectionBarrierTests(unittest.TestCase):
    def test_complete_three_seed_barrier_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection_fixture(root)
            run.write_barrier(root)
            run.verify_barrier(root)
            with (root / '2027/cohorts.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run.verify_barrier(root)

    def test_omitted_hash_cannot_unseal_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection_fixture(root)
            target = root / '2026/selection-manifest.json'
            value = json.loads(target.read_text())
            del value['artifact_sha256']['cohorts.json']
            run.write_json(target, value)
            with self.assertRaises(ValueError):
                run.write_barrier(root)

    def test_later_exact_tie_and_selected_prediction_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection_fixture(root)
            target = root / '2026/selections.json'
            value = json.loads(target.read_text())
            grid = json.loads((root / '2026/candidate-grid.json').read_text())
            value['families']['mixed'] = grid[1]
            run.write_json(target, value)
            seal_selection(root / '2026', 2026)
            with self.assertRaisesRegex(ValueError, 'first exact'):
                run.write_barrier(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection_fixture(root)
            arrays = {role: np.tile(np.arange(1, 11), (943, 1)) for role in run.ROLES}
            arrays['mixed'] = arrays['mixed'][:, ::-1]
            np.savez(root / '2026/predictions.npz', **arrays)
            seal_selection(root / '2026', 2026)
            with self.assertRaises(ValueError):
                run.write_barrier(root)

    def test_missing_third_seed_prevents_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection_fixture(root)
            (root / '2028/selection-manifest.json').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                run.write_barrier(root)


if __name__ == '__main__':
    unittest.main()
