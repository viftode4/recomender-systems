"""Probe fixed NGCF training numerics; never score VALID or TEST.

Run in separate processes with the desired thread environment set before startup.
The standard RecBole split processes the raw dataset; only TRAIN is consumed after
partitioning. Checkpoints and embeddings belong in a private runs/ output tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instructor-checkout', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'coursework_completion/rebuild_recipe.json')
    parser.add_argument('--seed', type=int, choices=(2026, 2027, 2028), default=2026)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    checkout, data, output = (p.resolve() for p in (args.instructor_checkout, args.data_path, args.out))
    if output.exists():
        parser.error('output already exists')
    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text())
    seed = next(row for row in recipe['seeds'] if row['seed'] == args.seed)
    entry = next(row for row in seed['experts'] if row.get('model') == 'NGCF')
    for name, expected in seed['data_sha256'].items():
        if digest(data / 'ml-100k' / name) != expected:
            raise ValueError(f'dataset hash mismatch: {name}')
    for name, expected in entry['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'adapter source hash mismatch: {name}')
    if not (checkout / 'recbole/model/general_recommender/ngcf.py').is_file():
        parser.error('missing instructor checkout')
    sys.path[:0] = [str(ROOT), str(checkout)]
    import numpy as np
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_seed
    from run import construct_model, split_pairs

    output.mkdir(parents=True)
    os.chdir(output)
    config = Config(model='NGCF', dataset='ml-100k', config_dict={
        **entry['settings'], 'epochs': args.epochs, 'data_path': str(data),
        'checkpoint_dir': str(output / 'checkpoints')})
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train, _, _ = data_preparation(config, dataset)
    pairs = split_pairs(train, dataset, config['USER_ID_FIELD'], config['ITEM_ID_FIELD'])
    body = 'user_id\titem_id\n' + ''.join(f'{u}\t{i}\n' for u, i in pairs)
    train_digest = hashlib.sha256(body.encode()).hexdigest()
    if train_digest != seed['expected_train_sha256']:
        raise ValueError('TRAIN partition differs from the declared historical split')
    del pairs, body
    init_seed(config['seed'], config['reproducibility'])
    model = construct_model('NGCF', config, train.dataset)

    def state_hashes():
        return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                for k, v in model.state_dict().items()}

    initial = state_hashes()
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer.fit(train, None, saved=True, show_progress=False)
    trainer.tensorboard.close()
    result = {
        'seed': args.seed, 'epochs': args.epochs, 'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(),
        'environment': {key: os.environ.get(key) for key in THREAD_KEYS},
        'initial_state_sha256': initial, 'final_state_sha256': state_hashes(),
        'loss': trainer.train_loss_dict,
        'loss_float32': {key: float(np.float32(value)) for key, value in trainer.train_loss_dict.items()},
        'train_sha256': train_digest, 'original_validation_test_evaluated': False,
        'scope': 'Raw dataset processed for the standard split; only TRAIN consumed after partitioning.',
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                    'system': platform.system(), 'machine': platform.machine()},
        'source_sha256': {'probe.py': digest(Path(__file__).resolve()),
                          'run.py': digest(ROOT / 'run.py'),
                          'metrics.py': digest(ROOT / 'metrics.py'),
                          'recipe.json': digest(recipe_path)},
        'instructor_source_sha256': {
            str(path.relative_to(checkout)): digest(path)
            for path in sorted((checkout / 'recbole').rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.yaml')},
    }
    (output / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('seed', 'epochs', 'threads', 'loss')}))


if __name__ == '__main__':
    main()
