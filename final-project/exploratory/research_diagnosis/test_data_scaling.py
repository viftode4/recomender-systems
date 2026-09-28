import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exploratory.research_diagnosis import data_scaling as d


def fixture(root, poison_development=False):
    root.mkdir()
    users = [f'u{n:02}' for n in range(12)]
    items = ['[PAD]'] + [f'i{n:02}' for n in range(20)]
    rng = np.random.default_rng(90)
    matrix = np.zeros((len(users), len(items)), dtype=np.int16)
    for row in range(len(users)):
        columns = rng.choice(np.arange(1, len(items)), 3, replace=False)
        matrix[row, columns] = rng.integers(1, 6, size=3)
    np.savez_compressed(root / 'training-categories.npz', users=users, items=items, ratings=matrix)
    d.json_write(root / 'input-signature.json', dict(training_categories_sha256=d.array_digest(matrix), ordered_identity_sha256=d.ordered_digest(users, items)))
    d.json_write(root / 'cohorts.json', dict(meta_fit=users[:8], development=users[8:]))
    with (root / 'valid.tsv').open('w') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['user_id', 'item_id'])
        for row, user in enumerate(users):
            available = np.flatnonzero(matrix[row] == 0)
            available = available[available != 0]
            position = -1 if poison_development and row >= 8 else 0
            writer.writerow([user, items[available[position]]])


class DataScalingTests(unittest.TestCase):
    def test_real_design_disjoint_fixed_queries_nested_counts(self):
        meta, dev = [str(i) for i in range(471)], [str(i) for i in range(471,943)]
        queries, donors = d.split_donors(meta, dev, 2026)
        self.assertEqual((len(queries), len(donors)), (118,353))
        self.assertTrue(set(donors).isdisjoint(queries + dev))
        self.assertEqual([int(len(donors)*fraction) for fraction in d.FRACTIONS], [88,176,264,353])
        self.assertEqual(d.split_donors(meta[::-1], dev, 2026), (queries, donors))

    def test_complete_selection_replay_and_development_pair_poison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, poisoned = root/'source', root/'poisoned'
            fixture(source)
            fixture(poisoned, poison_development=True)
            signature = dict(source_sha256=d.source_hashes(), protocol_sha256='synthetic', source_study_manifest_sha256='synthetic')
            with patch('builtins.print'):
                first = d.select_seed(source, root/'first', 2026, signature)
                second = d.select_seed(poisoned, root/'second', 2026, signature)
            a = d.verify_selection(root/'first', first)
            b = d.verify_selection(root/'second', second)
            self.assertEqual(first['candidate_fits'], 36)
            self.assertEqual(first['inner_query_users'], 2)
            for key in a:
                self.assertEqual(a[key]['penalty'], b[key]['penalty'])
                self.assertEqual(a[key]['scores_array_sha256'], b[key]['scores_array_sha256'])
                self.assertTrue(a[key]['replay_exact'])
            # Tampering with any saved selected score invalidates the barrier.
            path = root/'first'/a[next(iter(a))]['scores_file']
            path.write_bytes(b'not a score array')
            with self.assertRaisesRegex(ValueError, 'payload'):
                d.verify_selection(root/'first', first)

    def test_development_requires_global_seal_before_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with patch.object(d, 'read_truth') as truth:
                with self.assertRaisesRegex(ValueError, 'barrier'):
                    d.evaluate_seed(path, path, 2026, {}, path)
                truth.assert_not_called()


if __name__ == '__main__':
    unittest.main()
