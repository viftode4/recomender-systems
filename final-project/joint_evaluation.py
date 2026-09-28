"""Portable joint-field freeze and a separately invoked held-out audit.

Freezing verifies validation choices and replays TRAIN-query predictions without
opening TEST. Evaluation never refits a model or changes a selected checkpoint;
it retains TRAIN-only context while excluding TRAIN+validation items from rank.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from categorical_experiment import array_digest, categorical_metrics, restore
from categorical_evaluation import load_training
from joint_field_experiment import (SOURCE_FILES, VARIANTS, baseline_logits,
    joint_metrics, predict_logits, ranking_scores, validate_logits)
from scipy.special import softmax
from exception_evaluation import read_test_labels
from exception_experiment import evaluate_ratings, recommendations
from study import digest, grouped, read_pairs, write_json


ROOT = Path(__file__).resolve().parent
CODE_FILES = tuple(dict.fromkeys(('joint_evaluation.py', 'categorical_evaluation.py', 'exception_evaluation.py',
                                 'exception_model.py') + SOURCE_FILES))
BASELINES = ('event_global_category', 'event_item_category', 'event_item_user_tilt')


def read_json(path):
    return json.loads(path.read_text())


def source_hashes():
    return {name: digest(ROOT/name) for name in CODE_FILES}


def runtime_signature():
    import scipy
    import torch
    return {'python': platform.python_version(), 'system': platform.system(),
            'machine': platform.machine(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'torch': str(torch.__version__)}


def verify_choices(directory):
    manifest, protocol = read_json(directory/'manifest.json'), read_json(directory/'protocol.json')
    if manifest.get('status') != 'complete' or manifest.get('test_read') or manifest.get('test_evaluated'):
        raise ValueError('Require completed validation-only research')
    if (protocol.get('training_objective') != 'joint_recorded_item_rating_multinomial'
            or protocol.get('selection_metric') != 'meta_fit_macro_joint_nll'
            or protocol.get('selection_direction') != 'minimize; exact ties retain earlier checkpoint'
            or protocol.get('variants') != list(VARIANTS)
            or protocol.get('training_categories') != [1, 2, 3, 4, 5]
            or protocol.get('ranking_adapters') != {'all_observed': 'logsumexp(all five rating logits)', 'liked_record': 'logsumexp(rating 4 and 5 logits)'}):
        raise ValueError('Unsupported joint protocol')
    if manifest['protocol_sha256'] != digest(directory/'protocol.json'):
        raise ValueError('Protocol hash mismatch')
    if set(manifest['source_sha256']) != set(SOURCE_FILES):
        raise ValueError('Incomplete source signature')
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Research code changed: {name}')
    for name, expected in manifest['output_sha256'].items():
        candidate = directory/name
        if not candidate.resolve().is_relative_to(directory.resolve()) or digest(candidate) != expected:
            raise ValueError('Research artifact hash mismatch')
    selections = read_json(directory/'selection.json')
    if set(selections) != set(VARIANTS):
        raise ValueError('Core variant choices incomplete')
    reference_trace = None
    for variant in VARIANTS:
        selection = selections[variant]
        trace = read_json(directory/variant/'training-trace.json')
        if ([row['epoch'] for row in trace] != list(range(1, protocol['epochs']+1))
                or any(not np.isfinite(row['macro_train_joint_probe_nll']) for row in trace)):
            raise ValueError('Training trace does not match complete declared budget')
        masks = [(row['context_sha256'], row['probe_mask_sha256']) for row in trace]
        if reference_trace is not None and masks != reference_trace:
            raise ValueError('Variants used different training episodes')
        reference_trace = masks
        grid = read_json(directory/variant/'selection-grid.json')
        if [row['epoch'] for row in grid] != protocol['checkpoints']:
            raise ValueError('Checkpoint grid differs from declared protocol')
        winner = min(grid, key=lambda row: row['meta_fit_macro_joint_nll'])
        if any(selection[key] != winner[key] for key in ('epoch', 'meta_fit_macro_joint_nll')):
            raise ValueError('Selected checkpoint is not the declared joint NLL minimum')
        if (selection.get('selection_cohort') != 'meta_fit'
                or not selection.get('checkpoint_replay_exact')
                or selection['checkpoint'] != f"epoch-{selection['epoch']}.pt"
                or digest(directory/variant/selection['checkpoint']) != selection['checkpoint_sha256']):
            raise ValueError('Invalid selected checkpoint provenance')
    if (len({selection['initial_state_sha256'] for selection in selections.values()}) != 1
            or len({selection['parameter_count'] for selection in selections.values()}) != 1):
        raise ValueError('Variants differ in allocated parameters or initialization')
    return manifest, protocol, selections


def freeze_seed(directory, out, seed):
    manifest, protocol, selections = verify_choices(directory)
    if manifest['seed'] != seed or protocol['seeds'] != [seed]:
        raise ValueError('Research seed differs from requested seed')
    users, items, matrix, _, _ = load_training(directory)
    if array_digest(matrix) != manifest['train_categories_sha256']:
        raise ValueError('Training category digest differs from research manifest')
    for part in ('train', 'valid'):
        if digest(directory/f'{part}.tsv') != manifest['split_sha256'][part]:
            raise ValueError('Original split hash mismatch')
    logit_hashes = {}
    for variant in VARIANTS:
        selected = selections[variant]
        model, checkpoint = restore(directory/variant/selected['checkpoint'])
        expected_config = {**protocol['model_config'], 'n_items': len(items), 'variant': variant}
        if checkpoint['config'] != expected_config or checkpoint['seed'] != seed or checkpoint['epoch'] != selected['epoch']:
            raise ValueError('Checkpoint config, seed or epoch differs from frozen selection')
        logits = predict_logits(model, matrix)
        validate_logits(logits, users, items, matrix)
        with np.load(directory/variant/'selected-logits.npz', allow_pickle=False) as archive:
            if (archive['users'].tolist() != users or archive['items'].tolist() != items
                    or not np.array_equal(archive['logits'], logits)):
                raise ValueError('Selected checkpoint predictions fail exact replay')
        logit_hashes[variant] = array_digest(logits)
        if logit_hashes[variant] != selected['logits_array_sha256']:
            raise ValueError('Selected prediction digest differs')
    for name, logits in baseline_logits(matrix).items():
        logit_hashes[name] = array_digest(logits)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('train.tsv', 'valid.tsv', 'ids.json', 'catalog.json', 'training-categories.npz',
                 'protocol.json', 'selection.json', 'cohorts.json'):
        shutil.copyfile(directory/name, out/name)
    for variant in VARIANTS:
        (out/variant).mkdir()
        shutil.copyfile(directory/variant/selections[variant]['checkpoint'], out/variant/'model.pt')
    bundle = {'format_version': 1, 'status': 'frozen', 'seed': seed, 'test_read': False,
              'test_evaluated': False, 'k': 10, 'models': list(VARIANTS)+list(BASELINES),
              'selections': selections, 'train_category_array_sha256': array_digest(matrix),
              'logits_array_sha256': logit_hashes, 'exact_checkpoint_replay': True,
              'source_research_manifest_sha256': digest(directory/'manifest.json'),
              'source_protocol_sha256': digest(directory/'protocol.json'),
              'data_sha256': manifest['data_sha256'], 'split_sha256': manifest['split_sha256'],
              'expected_test_sha256': manifest['split_sha256']['test'],
              'inference': 'Frozen model and TRAIN-only rating context. No refit or fitted score transformation. Mask TRAIN+validation for ranking and recompute only the declared eligible-set joint event normalizer; score only test-rated pairs.',
              'payload_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'bundle.json', bundle)
    return digest(out/'bundle.json')


def freeze_research(research_root, out, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require unique nonempty seed list')
    codes = source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    bundles = {str(seed): freeze_seed(research_root/str(seed), out/str(seed), seed) for seed in seeds}
    if codes != source_hashes():
        raise ValueError('Inference source changed while freezing')
    write_json(out/'manifest.json', {'format_version': 1, 'status': 'frozen', 'seeds': list(seeds),
                                    'test_read': False, 'test_evaluated': False,
                                    'runtime': runtime_signature(),
                                    'portability': 'Self-contained data and checkpoints; bitwise replay requires the recorded platform and dependency versions.',
                                    'bundle_sha256': bundles, 'code_sha256': codes})
    (out/'manifest.sha256').write_text(digest(out/'manifest.json')+'\n')


def preflight(frozen):
    manifest = read_json(frozen/'manifest.json')
    if digest(frozen/'manifest.json') != (frozen/'manifest.sha256').read_text().strip():
        raise ValueError('Frozen manifest digest differs')
    if manifest.get('status') != 'frozen' or manifest.get('test_read'):
        raise ValueError('Require an unused frozen joint bundle')
    if manifest['code_sha256'] != source_hashes():
        raise ValueError('Frozen evaluation code changed')
    if manifest.get('runtime') != runtime_signature():
        raise ValueError('Frozen exact-replay runtime differs')
    bundles = []
    for seed in manifest['seeds']:
        directory = frozen/str(seed)
        if digest(directory/'bundle.json') != manifest['bundle_sha256'][str(seed)]:
            raise ValueError('Frozen bundle digest differs')
        bundle = read_json(directory/'bundle.json')
        if bundle['seed'] != seed or bundle['test_read'] or bundle['models'] != list(VARIANTS)+list(BASELINES):
            raise ValueError('Invalid frozen choices')
        for filename, expected in bundle['payload_sha256'].items():
            path = directory/filename
            if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
                raise ValueError('Frozen payload changed')
        bundles.append(bundle)
    return manifest, bundles


def frozen_predictions(folder, bundle):
    users, items, matrix, train, valid = load_training(folder)
    predictions = baseline_logits(matrix)
    for variant in VARIANTS:
        model, _ = restore(folder/variant/'model.pt')
        predictions[variant] = predict_logits(model, matrix)
    for name, logits in predictions.items():
        validate_logits(logits, users, items, matrix)
        if array_digest(logits) != bundle['logits_array_sha256'][name]:
            raise ValueError('Frozen TRAIN-query logits failed exact replay')
    return users, items, train, valid, predictions


def evaluate_frozen(frozen, test_paths, ratings_path, out):
    """Explicit final stage only. No invocation is made by freeze or training."""
    frozen_manifest, bundles = preflight(frozen)
    if set(test_paths) != set(frozen_manifest['seeds']):
        raise ValueError('Supply exactly one test path for every frozen seed')
    marker = frozen/'TEST-OPENED.json'
    if marker.exists() or out.exists():
        raise ValueError('Refuse repeated test opening or output overwrite')
    # Complete deterministic inference and hash checks before opening any labels.
    prepared = {bundle['seed']: frozen_predictions(frozen/str(bundle['seed']), bundle) for bundle in bundles}
    with marker.open('x') as stream:
        json.dump({'status': 'test_opening', 'output': str(out.resolve()),
                   'frozen_manifest_sha256': digest(frozen/'manifest.json')}, stream)
    out.mkdir(parents=True, exist_ok=False)
    aggregates = {'schema_version': 1, 'stage': 'test', 'test_read': True, 'seeds': {}}
    for bundle in bundles:
        seed = bundle['seed']
        users, items, train, valid, logits = prepared[seed]
        labels = read_test_labels(test_paths[seed], ratings_path, bundle['expected_test_sha256'],
                                 bundle['data_sha256'][ratings_path.name], train+valid, users, items)
        chosen_users = {user for user, _ in labels}
        if chosen_users != set(users):
            raise ValueError('Test users differ from frozen cohort')
        observed = np.zeros((len(users), len(items)), dtype=bool)
        ui, ii = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
        for user, item in train+valid:
            observed[ui[user], ii[item]] = True
        evaluated, all_recs, aggregate_models = {}, {}, {}
        # This matrix is ONLY a candidate-exclusion mask. Model queries above
        # used the original TRAIN categories and never these validation slots.
        exclusions = np.where(observed, 1, 0).astype(np.int64)
        for name in bundle['models']:
            values = logits[name]
            ranked, model_recs = {}, {}
            for adapter, scores in ranking_scores(values).items():
                recs = recommendations(scores, users, items, observed, bundle['k'])
                ranked[adapter] = evaluate_ratings(recs, chosen_users, labels, grouped(train+valid),
                    items[1:], Counter(i for _, i in train), bundle['k'], stage='test')
                model_recs[adapter] = recs
            evaluated[name] = {'joint': joint_metrics(values, users, items, exclusions, labels, chosen_users),
                              'categorical': categorical_metrics(softmax(values, axis=-1), users, items, labels, chosen_users),
                              'ranking': ranked}
            all_recs[name] = model_recs
            aggregate_models[name] = {
                'joint': {key: value for key, value in evaluated[name]['joint'].items() if key != 'per_user'},
                'categorical': {key: value for key, value in evaluated[name]['categorical'].items() if key != 'per_user'},
                'ranking': {adapter: {
                    'all_observed': result['all_observed']['aggregate'],
                    'liked_ratings': result['liked_ratings']['aggregate'] if result['liked_ratings'] else None,
                    'denominators': result['denominators'],
                    'known_dislike_rate_per_slot': result['known_dislike_rate_per_slot'],
                    'dislike_rate_among_test_rated_recommendations': result['dislike_rate_among_test_rated_recommendations'],
                    'caution': result['caution']} for adapter, result in ranked.items()}}
        folder = out/str(seed)
        folder.mkdir()
        write_json(folder/'metrics.json', evaluated)
        write_json(folder/'recommendations.json', all_recs)
        aggregates['seeds'][str(seed)] = {'models': aggregate_models, 'selections': bundle['selections']}
    write_json(out/'aggregates.json', aggregates)
    write_json(out/'manifest.json', {'status': 'complete', 'test_read': True, 'test_evaluated': True,
        'frozen_manifest_sha256': digest(frozen/'manifest.json'), 'code_sha256': frozen_manifest['code_sha256'],
        'selection_after_test': False,
        'output_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    write_json(marker, {'status': 'test_evaluated', 'output': str(out.resolve()),
                        'evaluation_manifest_sha256': digest(out/'manifest.json')})


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freezing = commands.add_parser('freeze')
    freezing.add_argument('--research-root', type=Path, required=True)
    freezing.add_argument('--out', type=Path, required=True)
    freezing.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--frozen', type=Path, required=True)
    evaluate.add_argument('--ratings', type=Path, required=True)
    evaluate.add_argument('--test', action='append', required=True, help='SEED=PATH; supply each frozen seed once')
    evaluate.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == 'freeze':
        freeze_research(args.research_root, args.out, args.seeds)
    else:
        mappings = [value.split('=', 1) for value in args.test]
        paths = {int(seed): Path(path) for seed, path in mappings}
        if len(paths) != len(mappings):
            parser.error('Duplicate test seed')
        evaluate_frozen(args.frozen, paths, args.ratings, args.out)


if __name__ == '__main__':
    main()
