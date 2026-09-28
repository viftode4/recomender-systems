"""Freeze exactly three locked references, then separately open TEST once.

The freezer reads TRAIN/validation artifacts and producer metadata only. It
copies finite full-catalog predictions into a self-contained bundle. The final
stage neither refits models nor changes their input histories or selections.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy

from exception_evaluation import read_test_labels
from exception_experiment import evaluate_ratings, recommendations
from field_reference_comparison import identity_digest, load_scores, reference_path
from study import digest, grouped, read_pairs, write_json


ROOT = Path(__file__).resolve().parent
MODELS = ('EASE', 'SLIMElastic', 'PositiveEASE')
CODE_FILES = ('field_reference_evaluation.py', 'field_reference_comparison.py',
              'exception_evaluation.py', 'exception_experiment.py', 'exception_model.py',
              'study.py', 'metrics.py', 'hybrid_constraints.py')


def source_hashes():
    return {name: digest(ROOT/name) for name in CODE_FILES}


def runtime_signature():
    return {'python': platform.python_version(), 'system': platform.system(),
            'machine': platform.machine(), 'numpy': np.__version__, 'scipy': scipy.__version__}


def read_json(path):
    return json.loads(Path(path).read_text())


def locked_reference(root, seed, model):
    path, selection = reference_path(root, seed, model)
    record = read_json(root/f'expert-selection-{seed}.json')[model]
    candidates = record['candidates']
    if not candidates or any(not np.isfinite(row['meta_fit_ndcg']) for row in candidates):
        raise ValueError('Invalid locked reference candidate grid')
    winner = max(candidates, key=lambda row: row['meta_fit_ndcg'])
    if any(record['selected'][key] != winner[key] for key in ('path', 'settings', 'meta_fit_ndcg')):
        raise ValueError('Reference choice differs from first recorded meta-fit maximum')
    manifest = read_json(path/'manifest.json')
    if manifest['model'] != model or manifest['settings'].get('seed') != seed:
        raise ValueError('Reference family or seed differs from requested choice')
    return path, {**selection, 'seed': seed, 'model': model,
                  'grid_winner_verified': True, 'tie_rule': 'first recorded maximum'}


def locked_positive(root, seed):
    directory = root/str(seed)
    path = directory/'positive_ease-liked_ratings'
    manifest = read_json(path/'manifest.json')
    selected = manifest['selection']
    if (manifest['model'] != 'PositiveEASE' or selected.get('model') != 'positive_ease'
            or selected.get('selection_objective') != 'liked_ratings'
            or selected.get('selection_cohort') != 'meta_fit' or selected.get('seed') != seed):
        raise ValueError('Require the exact liked-selected PositiveEASE export')
    recorded = read_json(directory/'selection.json')['positive_ease:liked_ratings']
    if recorded != selected:
        raise ValueError('PositiveEASE export differs from recorded selection')
    candidates = [row for row in read_json(directory/'selection-grid.json') if row['model'] == 'positive_ease']
    if not candidates or any(not np.isfinite(row['liked_ratings']) for row in candidates):
        raise ValueError('Invalid PositiveEASE candidate grid')
    winner = max(candidates, key=lambda row: row['liked_ratings'])
    if winner['penalty'] != selected['penalty'] or winner['liked_ratings'] != selected['selection_ndcg']:
        raise ValueError('PositiveEASE choice differs from first recorded liked maximum')
    return path, {**selected, 'grid_winner_verified': True, 'tie_rule': 'first recorded maximum'}


def validate_pairs(users, items, train, valid):
    if (len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) & set(valid)
            or {u for u, _ in train} != set(users) or {u for u, _ in valid} != set(users)
            or {i for _, i in train+valid} - set(items[1:])):
        raise ValueError('Invalid frozen TRAIN/validation identities')


def freeze_seed(reference_root, slim_root, positive_root, out, seed, k=10):
    ease, ease_choice = locked_reference(reference_root, seed, 'EASE')
    slim, slim_choice = locked_reference(slim_root, seed, 'SLIMElastic')
    positive, positive_choice = locked_positive(positive_root, seed)
    reference = read_json(ease/'manifest.json')
    with np.load(ease/'valid-scores.npz', allow_pickle=False) as archive:
        users, items = archive['users'].tolist(), archive['items'].tolist()
    ids = read_json(ease/'ids.json')
    if (not users or len(set(users)) != len(users) or len(set(items)) != len(items)
            or not items or items[0] != '[PAD]' or ids.get('padding_index') != 0
            or ids['items'] != items or not set(users).issubset(ids['users'])):
        raise ValueError('Invalid original reference ID mapping')
    train, valid = read_pairs(ease/'train.tsv'), read_pairs(ease/'valid.tsv')
    validate_pairs(users, items, train, valid)
    history = grouped(train+valid)
    if k < 1 or any(len(items)-1-len(history[u]) < k for u in users):
        raise ValueError('Insufficient unseen candidates for frozen cutoff')
    ids_hash = digest(ease/'ids.json')
    paths = {'EASE': ease, 'SLIMElastic': slim, 'PositiveEASE': positive}
    selections = {'EASE': ease_choice, 'SLIMElastic': slim_choice, 'PositiveEASE': positive_choice}
    signatures = {}
    for name, path in paths.items():
        _, manifest, signatures[name] = load_scores(path, reference, users, items, ids_hash)
        if manifest['model'] != name or selections[name]['selection_cohort'] != 'meta_fit':
            raise ValueError('Frozen reference model or selection cohort differs')
        if manifest['split_sha256']['test'] != reference['split_sha256']['test']:
            raise ValueError('Reference expected TEST split hash differs')
    out.mkdir(parents=True, exist_ok=False)
    for filename in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(ease/filename, out/filename)
    write_json(out/'score-ids.json', {'users': users, 'items': items})
    for name, path in paths.items():
        target = out/name
        target.mkdir()
        shutil.copyfile(path/'valid-scores.npz', target/'scores.npz')
        shutil.copyfile(path/'manifest.json', target/'producer-manifest.json')
        write_json(target/'selection.json', selections[name])
        if (digest(target/'scores.npz') != signatures[name]['score_file_sha256']
                or digest(target/'producer-manifest.json') != signatures[name]['manifest_sha256']):
            raise ValueError('Score or producer manifest changed while freezing')
    snapshots = out/'selection-snapshots'
    snapshots.mkdir()
    for name, source in (
        ('ease-grid.json', reference_root/f'expert-selection-{seed}.json'),
        ('slim-grid.json', slim_root/f'expert-selection-{seed}.json'),
        ('positive-selection.json', positive_root/str(seed)/'selection.json'),
        ('positive-grid.json', positive_root/str(seed)/'selection-grid.json')):
        shutil.copyfile(source, snapshots/name)
    bundle = {'format_version': 1, 'status': 'frozen', 'seed': seed, 'k': k, 'models': list(MODELS),
              'selections': selections, 'test_read': False, 'test_evaluated': False,
              'data_sha256': reference['data_sha256'], 'split_sha256': reference['split_sha256'],
              'expected_test_sha256': reference['split_sha256']['test'],
              'ordered_identity_sha256': identity_digest(users, items),
              'original_prediction_provenance': signatures,
              'inference': 'Reuse exact saved TRAIN-fitted full-catalog predictions; no refit or new input labels. Exclude all TRAIN+validation observations from final candidates.',
              'relevance': 'All observed TEST ratings; separately ratings>=4. Ratings<=2 are known dislikes, rating3 neutral, missing ratings unknown.',
              'payload_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'bundle.json', bundle)
    return digest(out/'bundle.json')


def freeze_references(reference_root, slim_root, slim_more_root, positive_root, out, seeds, k=10):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require distinct nonempty seeds')
    codes = source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    bundles = {str(seed): freeze_seed(reference_root, slim_root if seed == 2026 else slim_more_root,
                 positive_root, out/str(seed), seed, k) for seed in seeds}
    if source_hashes() != codes:
        raise ValueError('Reference evaluation code changed while freezing')
    write_json(out/'manifest.json', {'format_version': 1, 'status': 'frozen', 'seeds': list(seeds),
        'models': list(MODELS), 'test_read': False, 'test_evaluated': False,
        'code_sha256': codes, 'runtime': runtime_signature(), 'bundle_sha256': bundles})
    (out/'manifest.sha256').write_text(digest(out/'manifest.json')+'\n')


def load_frozen_seed(directory, bundle):
    ids = read_json(directory/'score-ids.json')
    users, items = ids['users'], ids['items']
    if (not users or len(set(users)) != len(users) or not items or items[0] != '[PAD]'
            or len(set(items)) != len(items) or identity_digest(users, items) != bundle['ordered_identity_sha256']):
        raise ValueError('Frozen ordered IDs differ')
    train, valid = read_pairs(directory/'train.tsv'), read_pairs(directory/'valid.tsv')
    validate_pairs(users, items, train, valid)
    scores = {}
    for name in MODELS:
        with np.load(directory/name/'scores.npz', allow_pickle=False) as archive:
            if archive['users'].tolist() != users or archive['items'].tolist() != items:
                raise ValueError('Frozen score IDs differ')
            values = archive['scores']
        if values.shape != (len(users), len(items)) or not np.isfinite(values).all():
            raise ValueError('Invalid frozen scores')
        scores[name] = values
    return users, items, train, valid, scores


def preflight(frozen):
    manifest = read_json(frozen/'manifest.json')
    if digest(frozen/'manifest.json') != (frozen/'manifest.sha256').read_text().strip():
        raise ValueError('Frozen manifest digest differs')
    if (manifest.get('status') != 'frozen' or manifest.get('test_read') or manifest.get('test_evaluated')
            or manifest.get('models') != list(MODELS)):
        raise ValueError('Require an unused frozen reference bundle')
    if manifest['code_sha256'] != source_hashes() or manifest['runtime'] != runtime_signature():
        raise ValueError('Frozen reference source or runtime differs')
    bundles = []
    for seed in manifest['seeds']:
        directory = frozen/str(seed)
        if digest(directory/'bundle.json') != manifest['bundle_sha256'][str(seed)]:
            raise ValueError('Frozen reference choices differ')
        bundle = read_json(directory/'bundle.json')
        if bundle['seed'] != seed or bundle['models'] != list(MODELS) or bundle['test_read'] or bundle['test_evaluated']:
            raise ValueError('Invalid frozen reference choices')
        for filename, expected in bundle['payload_sha256'].items():
            path = (directory/filename).resolve()
            if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
                raise ValueError('Frozen reference payload differs')
        for part in ('train', 'valid'):
            if digest(directory/f'{part}.tsv') != bundle['split_sha256'][part]:
                raise ValueError('Frozen split differs')
        for model in MODELS:
            choice = bundle['selections'][model]
            if choice.get('selection_cohort') != 'meta_fit' or choice['seed'] != seed or not choice['grid_winner_verified']:
                raise ValueError('Frozen selection provenance differs')
        if bundle['selections']['PositiveEASE']['selection_objective'] != 'liked_ratings':
            raise ValueError('Frozen PositiveEASE selection objective differs')
        bundles.append(bundle)
    return manifest, bundles


def evaluate_frozen(frozen, test_paths, ratings_path, out):
    manifest, bundles = preflight(frozen)
    if set(test_paths) != set(manifest['seeds']):
        raise ValueError('Supply exactly one test path per frozen seed')
    marker = frozen/'TEST-OPENED.json'
    if marker.exists() or out.exists():
        raise ValueError('Refuse repeated test opening or output overwrite')
    prepared = {bundle['seed']: load_frozen_seed(frozen/str(bundle['seed']), bundle) for bundle in bundles}
    # Exclusive marker is created before any test file/dataset read, and remains
    # after a failure. A label-opening failure must not authorize a fresh try.
    with marker.open('x') as stream:
        json.dump({'status': 'test_opening', 'output': str(out.resolve()),
                   'frozen_manifest_sha256': digest(frozen/'manifest.json')}, stream)
    out.mkdir(parents=True, exist_ok=False)
    aggregates = {'schema_version': 1, 'stage': 'test', 'test_read': True, 'seeds': {}}
    for bundle in bundles:
        seed = bundle['seed']
        users, items, train, valid, scores = prepared[seed]
        labels = read_test_labels(test_paths[seed], ratings_path, bundle['expected_test_sha256'],
                                 bundle['data_sha256'][ratings_path.name], train+valid, users, items)
        chosen = {u for u, _ in labels}
        if chosen != set(users):
            raise ValueError('Final reference users differ from frozen cohort')
        observed = np.zeros((len(users), len(items)), dtype=bool)
        ui, ii = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
        for user, item in train+valid:
            observed[ui[user], ii[item]] = True
        counts, history = Counter(item for _, item in train), grouped(train+valid)
        metrics, all_recs = {}, {}
        for name in MODELS:
            recs = recommendations(scores[name], users, items, observed, bundle['k'])
            metrics[name] = evaluate_ratings(recs, chosen, labels, history, items[1:], counts, bundle['k'], stage='test')
            all_recs[name] = recs
        directory = out/str(seed)
        directory.mkdir()
        write_json(directory/'metrics.json', metrics)
        write_json(directory/'recommendations.json', all_recs)
        aggregates['seeds'][str(seed)] = {name: {
            'all_observed': row['all_observed']['aggregate'],
            'liked_ratings': row['liked_ratings']['aggregate'] if row['liked_ratings'] else None,
            'denominators': row['denominators'], 'known_dislike_rate_per_slot': row['known_dislike_rate_per_slot']}
            for name, row in metrics.items()}
    write_json(out/'aggregates.json', aggregates)
    write_json(out/'manifest.json', {'status': 'complete', 'test_read': True, 'test_evaluated': True,
        'selection_after_test': False, 'frozen_manifest_sha256': digest(frozen/'manifest.json'),
        'code_sha256': manifest['code_sha256'],
        'output_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    write_json(marker, {'status': 'test_evaluated', 'output': str(out.resolve()),
                        'evaluation_manifest_sha256': digest(out/'manifest.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freezing = commands.add_parser('freeze')
    freezing.add_argument('--reference-root', type=Path, default=Path('runs/research-v2'))
    freezing.add_argument('--slim-root', type=Path, default=Path('runs/coverage-v1'))
    freezing.add_argument('--slim-more-root', type=Path, default=Path('runs/coverage-v1-more'))
    freezing.add_argument('--positive-root', type=Path, default=Path('runs/exception-v2'))
    freezing.add_argument('--out', type=Path, required=True)
    freezing.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--frozen', type=Path, required=True)
    evaluation.add_argument('--ratings', type=Path, required=True)
    evaluation.add_argument('--test', action='append', required=True, help='SEED=PATH, exactly once per frozen seed')
    evaluation.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze_references(args.reference_root, args.slim_root, args.slim_more_root, args.positive_root, args.out, args.seeds)
    else:
        values = [value.split('=', 1) for value in args.test]
        paths = {int(seed): Path(path) for seed, path in values}
        if len(paths) != len(values):
            parser.error('Duplicate test seed')
        evaluate_frozen(args.frozen, paths, args.ratings, args.out)


if __name__ == '__main__':
    main()
