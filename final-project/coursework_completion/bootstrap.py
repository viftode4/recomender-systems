"""Recreate the exact original TRAIN split without historical model artifacts.

No training, ranking, or original TEST evaluation occurs here. RecBole processes
the complete pinned raw dataset to construct its original partition; only TRAIN
identities and the public item/user catalog are exported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

PINNED_COMMIT = '081c3f6edf8e466d3ed5e163631a1afb6fe892bf'
PINNED_SOURCE_TREE = 'ddeb35cfb86fe80d7f74b8353598dd929d500dc2f4d6a03a3a71df1e9de8881a'
PINNED_SOURCE_FILES = 259
EXPECTED_DATA = {
    'ml-100k.inter': '4edb74e2a81178c2ba9ff381495f754f996c4aea351b1272ca36b43da0935eff',
    'ml-100k.item': '51d7cdf777ce5c0f5b32c1d947a4a81fe07d75e78abbe761e0cd4d0756064532',
    'ml-100k.user': '4f670007d9cfbeb9807e757209af1555b9bcc186bde25e767f67cb67c6dd5972',
}
EXPECTED_TRAIN_SHA = 'f4792fba583e3c01a26482f5fe0df62e02738cd8814672eaba32cb136c2fb7ec'
EXPECTED_IDS_SHA = '780252f4f0511dea952a45628c94dc1f6f8d357fa1064ed66ed1044d960a0cb6'
SETTINGS = {
    'dataset': 'ml-100k', 'embedding_size': 64, 'epochs': 20,
    'eval_args': {'group_by': 'user', 'mode': 'full', 'order': 'RO',
                  'split': {'RS': [.8, .1, .1]}},
    'eval_step': 1, 'k': 100, 'knn_method': 'item',
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
    'reg_weight': 250., 'reproducibility': True, 'seed': 2026,
    'show_progress': False, 'stopping_step': 5, 'topk': [10],
    'train_batch_size': 2048, 'use_gpu': False, 'valid_metric': 'MRR@10', 'worker': 0,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def save_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def source_tree_digest(root):
    root = Path(root)
    paths = sorted(path for path in root.rglob('*')
                   if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'))
    total = hashlib.sha256()
    for path in paths:
        total.update(path.relative_to(root).as_posix().encode() + b'\0'
                     + digest(path).encode() + b'\n')
    return total.hexdigest(), len(paths)


def checked_inputs(data_path, instructor_checkout):
    data_path, instructor_checkout = Path(data_path), Path(instructor_checkout)
    for name, expected in EXPECTED_DATA.items():
        path = data_path / 'ml-100k' / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Pinned dataset hash mismatch or missing file: {name}')
    fingerprint, count = source_tree_digest(instructor_checkout / 'recbole')
    if fingerprint != PINNED_SOURCE_TREE or count != PINNED_SOURCE_FILES:
        raise ValueError('Instructor RecBole source/config tree differs from the pinned checkout')
    return {'data_sha256': dict(EXPECTED_DATA), 'instructor_commit': PINNED_COMMIT,
            'recbole_source_tree_sha256': fingerprint, 'recbole_source_files': count,
            'source_tree_rule': 'Sorted relative .py/.yaml/.yml paths, NUL, file SHA hex, newline.'}


def train_and_catalog(config, dataset, train_loader):
    import numpy as np
    features = train_loader.dataset.inter_feat
    user_field, item_field = config['USER_ID_FIELD'], config['ITEM_ID_FIELD']
    users = dataset.id2token(user_field, features[user_field].cpu().numpy())
    items = dataset.id2token(item_field, features[item_field].cpu().numpy())
    pairs = sorted((str(user), str(item)) for user, item in zip(users, items))
    if len(pairs) != 80808 or len(set(pairs)) != len(pairs):
        raise ValueError('Regenerated original TRAIN has unexpected size or duplicates')
    body = ('user_id\titem_id\n' + ''.join(f'{u}\t{i}\n' for u, i in pairs)).encode()
    ids = {'users': [str(v) for v in dataset.id2token(user_field, np.arange(dataset.user_num))],
           'items': [str(v) for v in dataset.id2token(item_field, np.arange(dataset.item_num))],
           'padding_index': 0}
    if hashlib.sha256(body).hexdigest() != EXPECTED_TRAIN_SHA:
        raise ValueError('Regenerated TRAIN differs from the recorded original split; refusing substitute')
    if hashlib.sha256(json_bytes(ids)).hexdigest() != EXPECTED_IDS_SHA:
        raise ValueError('Regenerated catalog order differs from the recorded original IDs')
    return body, ids


def bootstrap(data_path, output_root, instructor_checkout=None):
    started = time.monotonic()
    data_path, output_root = Path(data_path).resolve(), Path(output_root).resolve()
    checkout = Path(instructor_checkout).resolve() if instructor_checkout else data_path.parent
    if output_root.exists():
        raise FileExistsError('Refusing to overwrite existing bootstrap output')
    verified = checked_inputs(data_path, checkout)
    sys.path.insert(0, str(checkout))
    recbole = importlib.import_module('recbole')
    if Path(recbole.__file__).resolve().parent != checkout / 'recbole':
        raise ValueError('Imported RecBole does not come from the verified instructor checkout')
    import numpy as np
    import scipy
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    output_root.mkdir(parents=True)
    source = output_root / 'source-seed2026'
    source.mkdir()
    initial_cwd = Path.cwd()
    try:
        os.chdir(output_root)
        config = Config(model='EASE', dataset='ml-100k',
            config_dict={**SETTINGS, 'data_path': str(data_path),
                         'checkpoint_dir': str(output_root / 'checkpoints')})
        init_seed(config['seed'], config['reproducibility'])
        dataset = create_dataset(config)
        train, valid, test = data_preparation(config, dataset)
        # The latter two loaders are never iterated or evaluated. Only TRAIN's
        # identities and the known catalog are extracted from RecBole.
        train_body, ids = train_and_catalog(config, dataset, train)
        del valid, test
    finally:
        os.chdir(initial_cwd)
    runtime = {'python': platform.python_version(), 'numpy': np.__version__,
               'scipy': scipy.__version__, 'torch': torch.__version__,
               'recbole': getattr(recbole, '__version__', 'unknown'), 'numerical_threads': 1}
    (source / 'train.tsv').write_bytes(train_body)
    save_json(source / 'ids.json', ids)
    manifest = {'status': 'complete', 'artifact_kind': 'split_only_reproduction_bootstrap',
        'model': 'none; EASE configuration is used only to reproduce the original split',
        'settings': SETTINGS, 'split_sha256': {'train': EXPECTED_TRAIN_SHA},
        'split_sizes': {'train': 80808}, 'data_sha256': dict(EXPECTED_DATA),
        'test_evaluated': False, 'test_read': False, 'validation_used_for_training': False,
        'model_fitted': False, 'raw_dataset_processed_by_recbole_for_split': True,
        'original_valid_test_loaders_created_but_not_iterated_or_exported': True,
        'versions': runtime, 'bootstrap_source_sha256': digest(Path(__file__)),
        'source_verification': verified}
    save_json(source / 'manifest.json', manifest)
    signature = {'artifact_kind': 'compatible_split_signature_not_a_categorical_model_run',
        'source_manifest_sha256': digest(source / 'manifest.json'),
        'split_sha256': {'train': EXPECTED_TRAIN_SHA}, 'ids_sha256': EXPECTED_IDS_SHA,
        'data_sha256': dict(EXPECTED_DATA)}
    save_json(output_root / 'input-signature.json', signature)
    verification = {'status': 'PASS', 'original_train_sha256_identical': True,
        'original_catalog_sha256_identical': True, 'model_fitted': False,
        'original_test_evaluated': False, 'runtime': runtime,
        'elapsed_seconds': time.monotonic()-started,
        'output_sha256': {str(path.relative_to(output_root)): digest(path)
            for path in sorted(output_root.rglob('*')) if path.is_file()}}
    save_json(output_root / 'BOOTSTRAP-VERIFIED.json', verification)
    return {'status': 'PASS', 'output_root': str(output_root),
            'source': 'source-seed2026', 'signature': 'input-signature.json',
            'train_sha256': EXPECTED_TRAIN_SHA, 'catalog_sha256': EXPECTED_IDS_SHA,
            'seconds': verification['elapsed_seconds']}


def standard_runner(entry, root=ROOT):
    """Use the recorded adapter version; never silently replace its source."""
    root = Path(root)
    expected = entry['source_sha256']['run.py']
    candidates = (root / 'run.py', root / 'coursework_completion' / 'legacy' / 'run.py')
    for candidate in candidates:
        if candidate.is_file() and digest(candidate) == expected:
            metric_file = candidate.with_name('metrics.py')
            if digest(metric_file) != entry['source_sha256']['metrics.py']:
                raise ValueError('Metric source differs from the recorded standard expert')
            return candidate
    raise ValueError(f"No bundled adapter matches source for {entry['name']}")


def validate_recipe(recipe):
    seeds = recipe.get('seeds', [])
    if [row.get('seed') for row in seeds] != [2026, 2027, 2028]:
        raise ValueError('Require fixed three-seed completion rebuild recipe')
    for seed in seeds:
        entries = seed['experts']
        if [entry['name'] for entry in entries] != seed['expert_order'] or len(entries) != 15:
            raise ValueError('Recipe expert order/count differs')
        if len({entry['name'] for entry in entries}) != 15:
            raise ValueError('Duplicate expert recipe name')
        if sum(entry['kind'] == 'standard' for entry in entries) != 12:
            raise ValueError('Require twelve standard experts per seed')
        if sum(entry['kind'] == 'research' for entry in entries) != 3:
            raise ValueError('Require three research experts per seed')
        for entry in entries:
            name = entry['name']
            if Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
                raise ValueError('Expert output must be one safe relative basename')
            if entry['kind'] == 'standard' and entry['settings']['seed'] != seed['seed']:
                raise ValueError('Expert seed differs from declared seed')
    return seeds


def run_logged(command, log_path, *, cwd, env):
    with Path(log_path).open('x') as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def rebuild_completion(data_path, output_root, instructor_checkout=None, recipe_path=None, workers=3):
    """Train all fixed historical expert configurations and rebuild frozen inputs."""
    started = time.monotonic()
    data_path, output_root = Path(data_path).resolve(), Path(output_root).resolve()
    checkout = Path(instructor_checkout).resolve() if instructor_checkout else data_path.parent
    recipe_path = Path(recipe_path) if recipe_path else Path(__file__).with_name('rebuild_recipe.json')
    if output_root.exists():
        raise FileExistsError('Refusing to overwrite existing completion rebuild')
    if workers not in (1, 2, 3):
        raise ValueError('Use one to three independent seed workers')
    verified = checked_inputs(data_path, checkout)
    recipe = json.loads(recipe_path.read_text())
    seeds = validate_recipe(recipe)
    for seed in seeds:
        for entry in seed['experts']:
            if entry['kind'] == 'standard':
                standard_runner(entry)
    from coursework_completion.rebuild_research import rebuild_research
    output_root.mkdir(parents=True)
    for name in ('configs', 'logs', 'experts', 'studies', 'frozen'):
        (output_root / name).mkdir()
    save_json(output_root / 'rebuild_recipe.json', recipe)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(checkout) + os.pathsep + str(ROOT)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    commands, rebuilt, bundles = [], [], {}
    progress_lock = threading.Lock()
    current_sources = {str(path.relative_to(ROOT)): digest(path) for path in (
        ROOT / 'run.py', ROOT / 'metrics.py', ROOT / 'exception_model.py',
        ROOT / 'exception_experiment.py', ROOT / 'study.py', ROOT / 'freeze.py',
        ROOT / 'hybrid_constraints.py', ROOT / 'societal.py', Path(__file__),
        ROOT / 'coursework_completion' / 'rebuild_research.py',
        ROOT / 'coursework_completion' / 'legacy' / 'run.py',
        ROOT / 'coursework_completion' / 'legacy' / 'metrics.py')}
    save_json(output_root / 'REBUILD-STARTED.json', {'source_sha256': current_sources,
        'recipe_sha256': digest(recipe_path), 'source_verification': verified,
        'scope': 'Fixed configurations; no expert search; no original TEST evaluation.'})
    def rebuild_seed(seed):
        number = seed['seed']
        expert_root = output_root / 'experts' / str(number)
        expert_root.mkdir()
        for entry in seed['experts']:
            if entry['kind'] != 'standard':
                continue
            config = output_root / 'configs' / (entry['name'] + '.json')
            save_json(config, entry['settings'])
            target = expert_root / entry['name']
            command = [sys.executable, str(standard_runner(entry)), '--model', entry['model'],
                       '--data-path', str(data_path), '--config', str(config),
                       '--out', str(target), '--fixed-epochs']
            tick = time.monotonic()
            run_logged(command, output_root / 'logs' / (entry['name'] + '.log'), cwd=ROOT, env=env)
            manifest = json.loads((target / 'manifest.json').read_text())
            if (manifest.get('status') != 'complete' or manifest.get('test_evaluated') is not False
                    or manifest.get('validation_used_for_training') is not False):
                raise ValueError('Rebuilt expert does not preserve fixed-budget/no-TEST boundary')
            if digest(target / 'train.tsv') != seed['expected_train_sha256']:
                raise ValueError('Rebuilt expert TRAIN differs from original split')
            if digest(target / 'ids.json') != seed['expected_ids_sha256']:
                raise ValueError('Rebuilt expert catalog differs from original order')
            expected = entry['expected_artifacts_sha256']['valid-scores.npz']
            row = {'seed': number, 'name': entry['name'], 'model': entry['model'],
                   'kind': 'standard', 'seconds': time.monotonic()-tick,
                   'adapter_sha256_matches_original': True,
                   'prediction_file_sha256_matches_original': digest(target / 'valid-scores.npz') == expected,
                   'prediction_file_sha256': digest(target / 'valid-scores.npz')}
            with progress_lock:
                rebuilt.append(row)
                commands.append(command)
                save_json(output_root / 'PROGRESS.json', rebuilt)
            print(json.dumps({'stage': 'standard_expert', **row}, sort_keys=True), flush=True)
        source_entry = next(entry for entry in seed['experts'] if entry['model'] == 'EASE')
        research_entries = [entry for entry in seed['experts'] if entry['kind'] == 'research']
        tick = time.monotonic()
        research_paths = rebuild_research(expert_root / source_entry['name'],
            data_path / 'ml-100k' / 'ml-100k.inter',
            data_path / 'ml-100k' / 'ml-100k.item', research_entries, expert_root)
        for entry, target in zip(research_entries, research_paths):
            if target.name != entry['name']:
                raise ValueError('Research helper changed expert order/name')
            expected = entry['expected_artifacts_sha256']['valid-scores.npz']
            with progress_lock:
                rebuilt.append({'seed': number, 'name': entry['name'], 'model': entry['model'],
                    'kind': 'research', 'adapter_sha256_matches_original': False,
                    'prediction_file_sha256_matches_original': digest(target / 'valid-scores.npz') == expected,
                    'prediction_file_sha256': digest(target / 'valid-scores.npz')})
        print(json.dumps({'stage': 'research_experts', 'seed': number,
                          'seconds': time.monotonic()-tick}, sort_keys=True), flush=True)
        ordered_paths = [expert_root / name for name in seed['expert_order']]
        study_path, frozen_path = output_root / 'studies' / str(number), output_root / 'frozen' / str(number)
        metadata = data_path / 'ml-100k' / 'ml-100k.item'
        command = [sys.executable, str(ROOT / 'study.py'), '--runs', *map(str, ordered_paths),
                   '--items', str(metadata), '--out', str(study_path), '--seed', str(number),
                   '--calibration-fraction', '.5']
        run_logged(command, output_root / 'logs' / f'{number}-study.log', cwd=ROOT, env=env)
        commands.append(command)
        command = [sys.executable, str(ROOT / 'freeze.py'), '--study', str(study_path),
                   '--runs', *map(str, ordered_paths), '--items', str(metadata), '--out', str(frozen_path)]
        run_logged(command, output_root / 'logs' / f'{number}-freeze.log', cwd=ROOT, env=env)
        commands.append(command)
        frozen = json.loads((frozen_path / 'freeze.json').read_text())
        bundles[str(number)] = {'path': str(frozen_path),
            'freeze_sha256': digest(frozen_path / 'freeze.json'), 'models': len(frozen['models']),
            'test_read': False, 'users': 943, 'catalog_items': 1682,
            'validation_replay': frozen['validation_replay'],
            'expected_test_sha256': frozen['expected_test_sha256']}
        print(json.dumps({'stage': 'seed_frozen', 'seed': number,
                          'validation_replay': frozen['validation_replay']}, sort_keys=True), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(rebuild_seed, seeds))
    for relative, expected in current_sources.items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Rebuild source changed during execution')
    manifest = {'status': 'complete', 'test_read': False, 'seeds': [2026, 2027, 2028],
                'numerical_library_threads': 1, 'bundles': bundles,
                'seconds': time.monotonic()-started, 'source_kind': 'fixed_config_fresh_data_rebuild'}
    frozen_root = output_root / 'frozen'
    save_json(frozen_root / 'manifest.json', manifest)
    (frozen_root / 'manifest.sha256').write_text(digest(frozen_root / 'manifest.json') + '\n')
    receipt = {'status': 'PASS', 'all_standard_adapters_match_original_source': True,
        'all_prediction_files_match_original': all(row['prediction_file_sha256_matches_original'] for row in rebuilt),
        'prediction_comparison_scope': 'Archive-file bytes, not separately normalized floating-point arrays.',
        'original_test_evaluated': False, 'validation_used_in_rebuilt_historical_study': True,
        'expert_count': len(rebuilt), 'experts': rebuilt, 'source_sha256': current_sources,
        'recipe_sha256': digest(recipe_path), 'frozen_manifest_sha256': digest(frozen_root / 'manifest.json'),
        'seconds': manifest['seconds'], 'commands': commands}
    save_json(output_root / 'REBUILD-VERIFIED.json', receipt)
    return {'status': 'PASS', 'output_root': str(output_root), 'frozen_source_root': str(frozen_root),
            'experts': len(rebuilt), 'all_prediction_files_match_original': receipt['all_prediction_files_match_original'],
            'seconds': manifest['seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True,
                        help='Directory containing ml-100k/*.inter,item,user atomic files')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory; existing paths are refused')
    parser.add_argument('--instructor-checkout', type=Path,
                        help='Existing pinned checkout; defaults to parent of --data-path')
    parser.add_argument('--stage', choices=('grouping', 'completion'), default='grouping',
                        help='Exact split only, or full three-seed fixed-expert/study rebuild')
    parser.add_argument('--recipe', type=Path,
                        help='Completion fixed recipe; defaults to bundled rebuild_recipe.json')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3,
                        help='Independent completion seed workers, each with one numerical thread')
    args = parser.parse_args()
    if args.stage == 'completion':
        result = rebuild_completion(args.data_path, args.output_root, args.instructor_checkout, args.recipe, args.workers)
    else:
        if args.recipe is not None:
            parser.error('--recipe is only used for completion')
        result = bootstrap(args.data_path, args.output_root, args.instructor_checkout)
    print(json.dumps(result, sort_keys=True))
