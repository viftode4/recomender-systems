from pathlib import Path
import tempfile
import unittest

import curate_final_evidence as curate


class FinalCurationTests(unittest.TestCase):
    def test_primary_allowlist_discards_individual_records_and_keeps_denominators(self):
        results = {'model': {'aggregate': {'ndcg@10': .2}, 'groups': {'0': {'users': 2}},
            'item_groups': {}, 'item_exposure': {'gini': .7},
            'per_user': {'private-user': {'ndcg@10': 1}}, 'recommendations': {'private-user': ['item']}}}
        result = curate.primary_aggregate(results, {'models': ['model'], 'users': 2, 'interactions': 5},
                                          {'best_expert': 'model'})
        self.assertNotIn('private-user', str(result))
        self.assertEqual(result['denominators'], {'users': 2, 'interactions': 5})
        self.assertEqual(result['models']['model']['aggregate']['ndcg@10'], .2)
        with self.assertRaisesRegex(ValueError, 'model set'):
            curate.primary_aggregate(results, {'models': ['different']}, {})

    def test_aggregate_copy_preserves_exact_bytes_and_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root/'source', root/'out'
            source.mkdir(); out.mkdir()
            (source/'aggregates.json').write_text('{"metric": 0.123}\n')
            manifest = {'output_sha256': {'aggregates.json': curate.digest(source/'aggregates.json')}}
            curate.copy_aggregate(source, out, manifest, 'aggregates.json')
            self.assertEqual((source/'aggregates.json').read_bytes(), (out/'aggregates.json').read_bytes())
            (source/'aggregates.json').write_text('{"metric": 0.456}')
            with self.assertRaisesRegex(ValueError, 'digest'):
                curate.copy_aggregate(source, out, manifest, 'aggregates.json')

    def test_even_hash_verified_aggregate_with_user_records_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root/'source', root/'out'
            source.mkdir(); out.mkdir()
            (source/'aggregates.json').write_text('{"per_user": {"private": 1}}')
            manifest = {'output_sha256': {'aggregates.json': curate.digest(source/'aggregates.json')}}
            with self.assertRaisesRegex(ValueError, 'Individual records'):
                curate.copy_aggregate(source, out, manifest, 'aggregates.json')
            self.assertFalse((out/'aggregates.json').exists())


if __name__ == '__main__':
    unittest.main()
