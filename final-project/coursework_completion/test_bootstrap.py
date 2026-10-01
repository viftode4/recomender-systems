from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from coursework_completion import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_public_recipe_routes_every_standard_model_to_exact_original_adapter(self):
        recipe = json.loads(Path(bootstrap.__file__).with_name('rebuild_recipe.json').read_text())
        seeds = bootstrap.validate_recipe(recipe)
        self.assertEqual(len(seeds), 3)
        for seed in seeds:
            for entry in seed['experts']:
                if entry['kind'] == 'standard':
                    self.assertEqual(bootstrap.digest(bootstrap.standard_runner(entry)),
                                     entry['source_sha256']['run.py'])
        broken = copy.deepcopy(recipe)
        broken['seeds'][0]['experts'][0]['name'] = '../escape'
        broken['seeds'][0]['expert_order'][0] = '../escape'
        with self.assertRaises(ValueError):
            bootstrap.validate_recipe(broken)

    def test_source_mismatch_does_not_silently_choose_newer_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run.py').write_text('new code')
            (root / 'metrics.py').write_text('metrics')
            entry = {'name': 'example', 'source_sha256': {
                'run.py': 'unknown-historical-hash', 'metrics.py': bootstrap.digest(root / 'metrics.py')}}
            with self.assertRaisesRegex(ValueError, 'No bundled adapter'):
                bootstrap.standard_runner(entry, root)

    def test_hash_tree_uses_relative_source_paths_and_ignores_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first', Path(directory) / 'second'
            for root in (first, second):
                (root / 'config').mkdir(parents=True)
                (root / 'module.py').write_text('value=1\n')
                (root / 'config' / 'base.yaml').write_text('seed: 2026\n')
            (first / 'cache.pyc').write_bytes(b'ignored')
            self.assertEqual(bootstrap.source_tree_digest(first), bootstrap.source_tree_digest(second))
            self.assertEqual(bootstrap.source_tree_digest(first)[1], 2)
            (second / 'module.py').write_text('value=2\n')
            self.assertNotEqual(bootstrap.source_tree_digest(first), bootstrap.source_tree_digest(second))

    def test_existing_output_is_refused_before_import_or_input_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(bootstrap, 'checked_inputs', side_effect=AssertionError('must not read')):
                with self.assertRaises(FileExistsError):
                    bootstrap.bootstrap(Path(directory) / 'missing', Path(directory))

    def test_wrong_raw_data_fails_before_source_or_recbole_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ml-100k').mkdir()
            (root / 'ml-100k' / 'ml-100k.inter').write_text('different data')
            with patch.object(bootstrap, 'source_tree_digest', side_effect=AssertionError('must not inspect')):
                with self.assertRaisesRegex(ValueError, 'dataset hash mismatch'):
                    bootstrap.checked_inputs(root, root / 'missing')


if __name__ == '__main__':
    unittest.main()
