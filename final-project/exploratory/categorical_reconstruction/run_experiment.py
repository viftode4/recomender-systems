"""Post-test exploratory categorical reconstruction, with a global selection barrier.

Only TRAIN categories reach the solver. Meta-fit validation pair identities choose
the fixed grid. Every seed's choices are sealed before any development evaluation.
TEST files and earlier final-test artifacts are never opened.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from categorical_experiment import array_digest, categorical_matrix
from exception_experiment import evaluate_ratings, load_source, recommendations
from field_reference_comparison import identity_digest, load_scores, reference_path, training_frequency
from metrics import evaluate
from study import digest, grouped, partition_users, read_pairs, write_json


LAMBDAS = (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)
ORIGINAL_LAMBDAS = (50., 250., 1000.)
CATEGORY_RATIOS = (.1, 1., 10., 100.)
SMOOTHING = 20.
RECONSTRUCTION_NAMES = ('binary_original_grid', 'binary_expanded', 'categorical', 'shuffled_categories')
HYBRID_NAMES = ('hybrid_baseline2', 'hybrid_real3', 'hybrid_shuffled3')
MODEL_NAMES = RECONSTRUCTION_NAMES+HYBRID_NAMES
SOURCE_FILES = ('exploratory/categorical_reconstruction/run_experiment.py',
    'exploratory/categorical_reconstruction/model.py',
    'exploratory/categorical_reconstruction/hybrid.py',
    'exploratory/categorical_reconstruction/group_analysis.py',
    'exploratory/categorical_reconstruction/PROTOCOL.md',
    'categorical_experiment.py', 'exception_experiment.py', 'exception_model.py',
    'field_reference_comparison.py', 'study.py', 'metrics.py', 'hybrid_constraints.py', 'societal.py')


def runtime_signature():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'machine': platform.machine(), 'system': platform.system(), 'numerical_threads': 1}


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def protocol(seeds):
    from exploratory.categorical_reconstruction.hybrid import PENALTIES
    return {'schema_version': 1, 'study_kind': 'post_final_test_exploratory',
        'stage': 'reused_development', 'test_read': False, 'test_evaluated': False,
        'dataset_test_previously_evaluated': True, 'fresh_confirmation': False,
        'seeds': list(seeds), 'models': list(MODEL_NAMES), 'lambdas': list(LAMBDAS),
        'original_binary_lambdas': list(ORIGINAL_LAMBDAS), 'category_ratios': list(CATEGORY_RATIOS),
        'smoothing': SMOOTHING, 'training_target': 'Original TRAIN observation indicator for every item.',
        'training_objective': 'Constrained regularized least-squares reconstruction; every own-item input channel is excluded.',
        'inputs': 'TRAIN observation indicator plus five TRAIN-centered categorical one-hot channels; no metadata or pretrained scores.',
        'selection_metric': 'meta_fit_all_observed_ndcg@10',
        'selection_rule': 'Exact maximum; ties retain the earlier declared candidate. Cached binary candidates precede categorical candidates.',
        'cohorts': 'Original seeded half-user partition: 471 reused meta-fit and 472 disjoint reused development users in MovieLens100K.',
        'selection_boundary': 'Every declared seed selection manifest is hash-sealed in SELECTIONS-FROZEN.json before any development evaluation.',
        'candidate_budgets': {'binary_original_grid': len(ORIGINAL_LAMBDAS),
            'binary_expanded': len(LAMBDAS), 'categorical': len(LAMBDAS)*(len(CATEGORY_RATIOS)+1),
            'shuffled_categories': len(LAMBDAS)*(len(CATEGORY_RATIOS)+1),
            'hybrid_baseline2': 4, 'hybrid_real3': 4, 'hybrid_shuffled3': 4,
            'unique_fits_per_seed': len(LAMBDAS)*(1+2*len(CATEGORY_RATIOS))},
        'shuffle': 'Permute observed TRAIN category values independently within each item, RNG seed+17011; preserve X and exact per-item rating histogram. Use shuffled features for fit and query.',
        'ranking': 'One selected raw reconstruction score for both relevance endpoints; exclude PAD and original TRAIN observations; k=10.',
        'hybrid_penalties': list(PENALTIES),
        'hybrids': 'Secondary baseline2=binary+SLIM, real3=baseline2+categorical, shuffled3=baseline2+shuffled; calibrated sum-to-one ridge, same inner meta235/236 split, declared hybrid_penalties, then refit all471 meta users. Inner split is not an unbiased nested expert evaluation because experts were selected on all meta users.',
        'groups': 'TRAIN activity terciles from all original users; all-observed nDCG/recall and genre-Jaccard diversity. TRAIN popularity head top20% versus tail: conditional recall and exposure. Groups never select models.',
        'inference': 'Original TRAIN only; no validation context assimilation. Scores are ranking values, not probabilities.',
        'missing_records': 'Zeros are the explicit implicit-reconstruction target convention, not confirmed dislikes.',
        'claims': 'Exploratory reused-validation evidence with unequal tuning opportunities; no fresh confirmation or proven novelty.',
        'runtime': runtime_signature(), 'source_sha256': source_hashes()}


def shuffle_categories(matrix, seed):
    result = np.asarray(matrix).copy()
    rng = np.random.default_rng(seed+17011)
    for item in range(1, result.shape[1]):
        rows = np.flatnonzero(result[:, item])
        result[rows, item] = rng.permutation(result[rows, item])
    if not np.array_equal(result != 0, matrix != 0):
        raise AssertionError('Shuffle changed observation identities')
    for category in range(1, 6):
        if not np.array_equal((result == category).sum(0), (matrix == category).sum(0)):
            raise AssertionError('Shuffle changed item category counts')
    return result


def load_train(source, ratings):
    """Read only TRAIN values; validation/test rating text is not parsed."""
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('test_evaluated', True)
            or manifest.get('test_read', False) or manifest.get('validation_used_for_training', True)):
        raise ValueError('Require a complete original TRAIN-only source')
    if digest(source/'train.tsv') != manifest['split_sha256']['train']:
        raise ValueError('TRAIN split signature differs')
    if digest(ratings) != manifest['data_sha256'].get(ratings.name):
        raise ValueError('Raw ratings signature differs')
    ids = json.loads((source/'ids.json').read_text())
    train = read_pairs(source/'train.tsv')
    present = set(grouped(train))
    users = [user for user in ids['users'] if user in present]
    items = ids['items']
    if (ids.get('padding_index') != 0 or not items or items[0] != '[PAD]'
            or len(users) != len(present) or len(set(users)) != len(users)
            or len(set(items)) != len(items) or len(set(train)) != len(train)):
        raise ValueError('Invalid original TRAIN or catalog identities')
    allowed, retained = set(train), {}
    with ratings.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = row['user_id:token'], row['item_id:token']
            if pair in allowed:
                if pair in retained:
                    raise ValueError('Duplicate TRAIN category')
                retained[pair] = float(row['rating:float'])
    if set(retained) != allowed:
        raise ValueError('Missing TRAIN categories')
    matrix = categorical_matrix(users, items, [(u, i, retained[u, i]) for u, i in train])
    return users, items, train, matrix, manifest


def load_selection_inputs(source, ratings, metadata, cohort_directory, seed):
    users, items, train, matrix, manifest = load_train(source, ratings)
    if digest(metadata) != manifest['data_sha256'].get(metadata.name):
        raise ValueError('Metadata provenance differs')
    if digest(source/'valid.tsv') != manifest['split_sha256']['valid']:
        raise ValueError('Validation pair signature differs')
    valid = read_pairs(source/'valid.tsv')
    if (len(set(valid)) != len(valid) or set(train) & set(valid)
            or set(grouped(valid)) != set(users) or {i for _, i in valid} - set(items[1:])):
        raise ValueError('Invalid validation pair identities')
    fit, dev = partition_users(users, seed)
    cohorts = json.loads((cohort_directory/'cohorts.json').read_text())
    if cohorts != {'meta_fit': sorted(fit), 'development': sorted(dev)}:
        raise ValueError('Original cohort partition differs')
    for part in ('train', 'valid'):
        if digest(cohort_directory/f'{part}.tsv') != manifest['split_sha256'][part]:
            raise ValueError('Original cohort split differs')
    signature = {'source_manifest_sha256': digest(source/'manifest.json'),
        'data_sha256': manifest['data_sha256'],
        'split_sha256': {part: manifest['split_sha256'][part] for part in ('train', 'valid')},
        'ids_sha256': digest(source/'ids.json'), 'ordered_identity_sha256': identity_digest(users, items),
        'cohort_file_sha256': digest(cohort_directory/'cohorts.json'),
        'training_categories_sha256': array_digest(matrix)}
    return users, items, train, valid, matrix, fit, dev, signature


def prepared_model(matrix):
    # The solver module is independent of all validation/cohort data.
    from exploratory.categorical_reconstruction.model import prepare_features
    return prepare_features(matrix, smoothing=SMOOTHING)


def solve(prepared, penalty, ratio):
    from exploratory.categorical_reconstruction.model import fit
    model = fit(prepared, lambda_binary=penalty, category_ratio=ratio)
    scores, diagnostics = model.predict(), model.diagnostics
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any(scores[:, 0]):
        raise ValueError('Invalid unmasked full-catalog reconstruction scores')
    return scores, diagnostics


def meta_ndcg(scores, users, items, train, valid, fit, k=10):
    indices = [index for index, user in enumerate(users) if user in fit]
    chosen = [users[index] for index in indices]
    history = grouped(train)
    observed = np.asarray([[item in history[user] for item in items] for user in chosen])
    recs = recommendations(scores[indices], chosen, items, observed, k)
    valid_truth = grouped(valid)
    truth = {user: valid_truth[user] for user in chosen}
    return evaluate(recs, truth, history, items[1:], Counter(i for _, i in train), k)['aggregate'][f'ndcg@{k}']


def candidate_id(kind, penalty, ratio):
    return f'{kind}-lambda-{penalty:g}'+('' if ratio is None else f'-ratio-{ratio:g}')


def select_seed(source, ratings, metadata, cohort_directory, slim_source, output, seed, expected_protocol, slim_selection):
    if expected_protocol != protocol(expected_protocol['seeds']):
        raise ValueError('Protocol/source/runtime changed before selection')
    started = time.monotonic()
    users, items, train, valid, matrix, fit, dev, inputs = load_selection_inputs(source, ratings, metadata, cohort_directory, seed)
    selection_file = Path(slim_selection['selection_file'])
    checked_path, checked_selection = reference_path(selection_file.parent, seed, 'SLIMElastic')
    if (checked_path.resolve() != slim_source.resolve()
            or checked_selection != {key: value for key, value in slim_selection.items() if key != 'selection_file'}):
        raise ValueError('Locked SLIM selection provenance differs')
    slim_scores, _, slim_provenance = load_scores(slim_source, inputs, users, items, inputs['ids_sha256'])
    slim_provenance['selection'] = checked_selection
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'input-signature.json', inputs)
    write_json(output/'locked-slim.json', {'source_path': str(slim_source), **slim_provenance})
    np.save(output/'locked-slim-scores.npy', slim_scores, allow_pickle=False)
    write_json(output/'cohorts.json', {'meta_fit': sorted(fit), 'development': sorted(dev)})
    np.savez_compressed(output/'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    for name in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(source/name, output/name)
    matrices = {'binary': matrix, 'categorical': matrix, 'shuffled_categories': shuffle_categories(matrix, seed)}
    prepared = {name: prepared_model(value) for name, value in matrices.items() if name != 'binary'}
    prepared['binary'] = prepared['categorical']
    grid, choices, best_arrays = [], {}, {}
    specs = [('binary', penalty, None) for penalty in LAMBDAS]
    specs += [(name, penalty, ratio) for name in ('categorical', 'shuffled_categories')
              for penalty in LAMBDAS for ratio in CATEGORY_RATIOS]
    for kind, penalty, ratio in specs:
        tick = time.monotonic()
        scores, diagnostic = solve(prepared[kind], penalty, ratio)
        if scores.shape != matrix.shape:
            raise ValueError('Candidate prediction shape differs from original catalog')
        value = meta_ndcg(scores, users, items, train, valid, fit)
        if not np.isfinite(value):
            raise ValueError('Nonfinite selection metric')
        identifier = candidate_id(kind, penalty, ratio)
        row = {'candidate_id': identifier, 'feature_kind': kind, 'penalty': penalty,
            'category_ratio': ratio, 'meta_fit_all_observed_ndcg@10': value,
            'scores_array_sha256': array_digest(scores), 'fit_and_selection_seconds': time.monotonic()-tick,
            'solver_diagnostics': diagnostic}
        grid.append(row)
        families = ['binary_expanded', 'categorical', 'shuffled_categories'] if kind == 'binary' else [kind]
        if kind == 'binary' and penalty in ORIGINAL_LAMBDAS:
            families.append('binary_original_grid')
        for family in families:
            if family not in choices or value > choices[family]['meta_fit_all_observed_ndcg@10']:
                choices[family] = dict(row)
                best_arrays[family] = scores.copy()
        write_json(output/'candidate-grid.json', grid)
        write_json(output/'progress.json', {'status': 'selecting', 'seed': seed,
            'completed_fits': len(grid), 'total_fits': len(specs), 'test_read': False})
        print(f'{seed} reconstruction {len(grid)}/{len(specs)} {identifier}: meta nDCG {value:.6f}', flush=True)
    # Only selected arrays are retained. Repeat the deterministic TRAIN solve to
    # check fit replay, then independently round-trip each selected array file.
    score_directory = output/'selected-scores'
    score_directory.mkdir()
    replayed = {}
    for family in RECONSTRUCTION_NAMES:
        row = choices[family]
        identifier = row['candidate_id']
        if identifier not in replayed:
            scores, _ = solve(prepared[row['feature_kind']], row['penalty'], row['category_ratio'])
            if not np.array_equal(scores, best_arrays[family]):
                raise AssertionError('Selected TRAIN-only refit differs')
            path = score_directory/(identifier+'.npy')
            np.save(path, scores, allow_pickle=False)
            if not np.array_equal(np.load(path, allow_pickle=False), scores):
                raise AssertionError('Selected score file replay differs')
            replayed[identifier] = path
        path = replayed[identifier]
        row.update(selection_cohort='reused_meta_fit', selection_metric='all_observed_ndcg@10',
            candidate_count=expected_protocol['candidate_budgets'][family],
            scores_file=path.relative_to(output).as_posix(), scores_file_sha256=digest(path),
            refit_replay_exact=True, score_file_replay_exact=True)
    # The secondary hybrid learner receives meta pair identities only, never
    # development users' validation memberships or rating values.
    from exploratory.categorical_reconstruction.hybrid import prepare_hybrids
    valid_truth = grouped(valid)
    meta_truth = {user: valid_truth[user] for user in sorted(fit)}
    expert_scores = {'binary': best_arrays['binary_expanded'], 'slim': slim_scores,
                     'real': best_arrays['categorical'], 'shuffled': best_arrays['shuffled_categories']}
    hybrid_scores, hybrid_metadata = prepare_hybrids(expert_scores, users, items, grouped(train), meta_truth, fit, seed)
    replay_scores, replay_metadata = prepare_hybrids(expert_scores, users, items, grouped(train), meta_truth, fit, seed)
    if hybrid_metadata != replay_metadata:
        raise AssertionError('Hybrid fitting metadata replay differs')
    write_json(output/'hybrid-selection.json', hybrid_metadata)
    for arm, scores in hybrid_scores.items():
        name = 'hybrid_'+arm
        if name not in HYBRID_NAMES or scores.shape != matrix.shape or not np.isfinite(scores).all():
            raise ValueError('Invalid finite hybrid prediction array')
        if not np.array_equal(scores, replay_scores[arm]):
            raise AssertionError('Hybrid fit replay differs')
        path = score_directory/(name+'.npy')
        np.save(path, scores, allow_pickle=False)
        if not np.array_equal(np.load(path, allow_pickle=False), scores):
            raise AssertionError('Hybrid score file replay differs')
        arm_metadata = hybrid_metadata['arms'][arm]
        choices[name] = {'candidate_id': name, 'feature_kind': 'calibrated_weighted_hybrid',
            'selection_cohort': 'inner_reused_meta_fit', 'selection_metric': 'all_observed_ndcg@10',
            'candidate_count': expected_protocol['candidate_budgets'][name],
            'penalty': arm_metadata['selected_penalty'], 'category_ratio': None,
            'scores_file': path.relative_to(output).as_posix(), 'scores_file_sha256': digest(path),
            'scores_array_sha256': array_digest(scores), 'refit_replay_exact': True, 'score_file_replay_exact': True,
            'hybrid_arm': arm_metadata, 'duplicates': hybrid_metadata['duplicates'],
            'selection_limit': hybrid_metadata['selection_limit']}
    if set(choices) != set(MODEL_NAMES):
        raise ValueError('Missing reconstruction/hybrid selection')
    write_json(output/'selection.json', choices)
    final_inputs = load_selection_inputs(source, ratings, metadata, cohort_directory, seed)[-1]
    if (final_inputs != inputs or expected_protocol != protocol(expected_protocol['seeds'])
            or digest(slim_source/'manifest.json') != slim_provenance['manifest_sha256']
            or digest(slim_source/'valid-scores.npz') != slim_provenance['score_file_sha256']
            or digest(selection_file) != checked_selection['selection_file_sha256']):
        raise ValueError('Input/source/runtime changed during selection')
    write_json(output/'progress.json', {'status': 'selected', 'seed': seed, 'completed_fits': len(grid), 'test_read': False})
    manifest = {'status': 'selected', 'seed': seed, 'test_read': False, 'development_evaluated': False,
        'source_sha256': source_hashes(), 'runtime': runtime_signature(), 'input_signature': inputs,
        'shuffled_training_categories_sha256': array_digest(matrices['shuffled_categories']),
        'locked_slim_source': slim_provenance,
        'unique_fits': len(grid), 'elapsed_seconds': time.monotonic()-started,
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output/'selection-manifest.json', manifest)
    return manifest


def verify_files(directory, expected):
    for name, signature in expected.items():
        path = directory/name
        if not path.resolve().is_relative_to(directory.resolve()) or path.name.lower().startswith(('test.', 'test-', 'final-')):
            raise ValueError('Forbidden sealed artifact path')
        if digest(path) != signature:
            raise ValueError(f'Sealed artifact changed: {path}')


def validate_choices(directory, declared):
    """Independently reconstruct declared candidate membership and tie rules."""
    grid = json.loads((directory/'candidate-grid.json').read_text())
    choices = json.loads((directory/'selection.json').read_text())
    expected = [('binary', penalty, None) for penalty in LAMBDAS]
    expected += [(kind, penalty, ratio) for kind in ('categorical', 'shuffled_categories')
                 for penalty in LAMBDAS for ratio in CATEGORY_RATIOS]
    actual = [(row['feature_kind'], row['penalty'], row['category_ratio']) for row in grid]
    if actual != expected or set(choices) != set(MODEL_NAMES):
        raise ValueError('Incomplete, reordered or changed candidate grid')
    with np.load(directory/'training-categories.npz', allow_pickle=False) as archive:
        shape = archive['ratings'].shape
    for row in grid:
        if (row['candidate_id'] != candidate_id(row['feature_kind'], row['penalty'], row['category_ratio'])
                or not np.isfinite(row['meta_fit_all_observed_ndcg@10'])):
            raise ValueError('Invalid candidate identity or metric')
    for name in RECONSTRUCTION_NAMES:
        candidates = [row for row in grid if row['feature_kind'] == 'binary' or row['feature_kind'] == name]
        if name.startswith('binary'):
            candidates = [row for row in candidates if row['feature_kind'] == 'binary']
        if name == 'binary_original_grid':
            candidates = [row for row in candidates if row['penalty'] in ORIGINAL_LAMBDAS]
        winner = max(candidates, key=lambda row: row['meta_fit_all_observed_ndcg@10'])
        if (len(candidates) != declared['candidate_budgets'][name]
                or any(choices[name].get(key) != value for key, value in winner.items())):
            raise ValueError('Frozen choice differs from exact maximum/first tie')
    hybrid = json.loads((directory/'hybrid-selection.json').read_text())
    cohorts = json.loads((directory/'cohorts.json').read_text())
    if hybrid['cohorts']['meta_fit'] != cohorts['meta_fit'] or hybrid['penalty_grid'] != declared['hybrid_penalties']:
        raise ValueError('Hybrid meta cohort or penalty grid differs')
    for name in HYBRID_NAMES:
        arm = hybrid['arms'][name.removeprefix('hybrid_')]
        if [row['penalty'] for row in arm['candidates']] != declared['hybrid_penalties']:
            raise ValueError('Incomplete hybrid candidate grid')
        winner = max(arm['candidates'], key=lambda row: (row['selection_ndcg@10'], row['penalty']))
        if choices[name]['hybrid_arm'] != arm or arm['selected_penalty'] != winner['penalty']:
            raise ValueError('Hybrid choice differs from inner maximum/tie rule')
    for name, selected in choices.items():
        path = directory/selected['scores_file']
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != selected['scores_file_sha256']:
            raise ValueError('Selected score file signature differs')
        scores = np.load(path, allow_pickle=False)
        if (scores.shape != shape or not np.isfinite(scores).all()
                or array_digest(scores) != selected['scores_array_sha256']
                or not selected['refit_replay_exact'] or not selected['score_file_replay_exact']):
            raise ValueError('Selected score replay or identity differs')
    return choices


def freeze_selections(research):
    declared = json.loads((research/'protocol.json').read_text())
    if declared != protocol(declared['seeds']):
        raise ValueError('Declared protocol differs')
    frozen = {'status': 'all_selections_frozen', 'seeds': declared['seeds'], 'test_read': False,
              'development_evaluated': False, 'protocol_sha256': digest(research/'protocol.json'),
              'source_sha256': source_hashes(), 'runtime': runtime_signature(), 'selection_manifest_sha256': {}}
    for seed in declared['seeds']:
        directory = research/str(seed)
        manifest = json.loads((directory/'selection-manifest.json').read_text())
        if (manifest['status'] != 'selected' or manifest['seed'] != seed or manifest['test_read']
                or manifest['development_evaluated'] or manifest['source_sha256'] != source_hashes()
                or manifest['runtime'] != runtime_signature()):
            raise ValueError('Invalid selection manifest')
        verify_files(directory, manifest['output_sha256'])
        validate_choices(directory, declared)
        frozen['selection_manifest_sha256'][str(seed)] = digest(directory/'selection-manifest.json')
    with (research/'SELECTIONS-FROZEN.json').open('x') as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return frozen


def verify_barrier(research):
    frozen = json.loads((research/'SELECTIONS-FROZEN.json').read_text())
    declared = json.loads((research/'protocol.json').read_text())
    if (frozen['status'] != 'all_selections_frozen' or frozen['test_read']
            or frozen['seeds'] != declared['seeds'] or frozen['protocol_sha256'] != digest(research/'protocol.json')
            or frozen['source_sha256'] != source_hashes() or frozen['runtime'] != runtime_signature()
            or declared != protocol(declared['seeds'])):
        raise ValueError('Global selection barrier differs')
    for seed in frozen['seeds']:
        directory = research/str(seed)
        if digest(directory/'selection-manifest.json') != frozen['selection_manifest_sha256'][str(seed)]:
            raise ValueError('Selection manifest changed after global freeze')
        verify_files(directory, json.loads((directory/'selection-manifest.json').read_text())['output_sha256'])
        validate_choices(directory, declared)
    return frozen


def aggregate_metrics(metrics, recs, users, counts, catalog):
    return {**{name: metrics[name]['aggregate'] if metrics[name] else None for name in ('all_observed', 'liked_ratings')},
        'denominators': metrics['denominators'], 'known_dislike_rate_per_slot': metrics['known_dislike_rate_per_slot'],
        'dislike_rate_among_validation_rated_recommendations': metrics['dislike_rate_among_validation_rated_recommendations'],
        'training_frequency': training_frequency(recs, users, counts, catalog)}


def matched_references(reference_evidence, reference_run, inputs, metrics, locked_slim):
    manifest = json.loads((reference_run/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['test_read'] or manifest['test_evaluated']:
        raise ValueError('Require development reference comparison')
    verify_files(reference_evidence, manifest['evidence_sha256'])
    verify_files(ROOT, manifest['source_sha256'])
    all_rows = json.loads((reference_evidence/'aggregates.json').read_text())['seeds']
    rows = {}
    fixed = ('all_observed_users', 'liked_ratings_users', 'validation_observations', 'validation_likes', 'validation_dislikes', 'recommendation_slots')
    for seed, input_signature in inputs.items():
        ref = all_rows[str(seed)]
        for key in ('data_sha256', 'split_sha256', 'ordered_identity_sha256', 'cohort_file_sha256'):
            if ref[key] != input_signature[key]:
                raise ValueError('Locked development reference identity/cohort/split differs')
        if digest(reference_run/str(seed)/'details.json') != ref['details_sha256']:
            raise ValueError('Locked reference details signature differs')
        actual_slim = locked_slim[str(seed)]
        for endpoint in ('all_observed', 'liked_ratings'):
            expected_slim = ref['provenance']['SLIMElastic'][endpoint]
            for field in ('manifest_sha256', 'score_file_sha256', 'ids_file_sha256', 'ordered_identity_sha256'):
                if actual_slim[field] != expected_slim[field]:
                    raise ValueError('Hybrid SLIM expert differs from locked development reference')
        if actual_slim['selection'] != ref['rows']['SLIMElastic']['selection']:
            raise ValueError('Hybrid SLIM selection differs from locked development reference')
        for name in ('EASE', 'SLIMElastic', 'PositiveEASE'):
            for endpoint in ('all_observed', 'liked_ratings'):
                denominator = ref['rows'][name]['endpoints'][endpoint]['denominators']
                if any(denominator[key] != metrics[str(seed)]['binary_expanded']['denominators'][key] for key in fixed):
                    raise ValueError('Locked reference evaluation denominators differ')
        rows[str(seed)] = {name: ref['rows'][name] for name in ('EASE', 'SLIMElastic', 'PositiveEASE')}
    return rows


def evaluate_selected(research, source_root, ratings, metadata, reference_evidence, reference_run):
    frozen = verify_barrier(research)
    detailed, aggregate, inputs, locked_slim = {}, {}, {}, {}
    # Reserve an explicit one-use development stage only after the global seal.
    with (research/'DEVELOPMENT-OPENED.json').open('x') as stream:
        json.dump({'status': 'started', 'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'),
                   'test_read': False, 'fresh_confirmation': False}, stream, indent=2)
    for seed in frozen['seeds']:
        directory = research/str(seed)
        loaded = load_source(source_root/f'{seed}-EASE-1', ratings, metadata)
        users, items, train, valid, training, validation = loaded[:6]
        matrix = categorical_matrix(users, items, training)
        inputs[str(seed)] = json.loads((directory/'input-signature.json').read_text())
        locked_slim[str(seed)] = json.loads((directory/'selection-manifest.json').read_text())['locked_slim_source']
        if (identity_digest(users, items) != inputs[str(seed)]['ordered_identity_sha256']
                or array_digest(matrix) != inputs[str(seed)]['training_categories_sha256']
                or any(loaded[-1]['split_sha256'][part] != inputs[str(seed)]['split_sha256'][part] for part in ('train', 'valid'))
                or loaded[-1]['data_sha256'] != inputs[str(seed)]['data_sha256']):
            raise ValueError('Evaluation source differs from selected TRAIN inputs')
        _, dev = partition_users(users, seed)
        history, counts = grouped(train), Counter(item for _, item in train)
        selections = json.loads((directory/'selection.json').read_text())
        from exploratory.categorical_reconstruction.group_analysis import analyze_groups
        genre_binary = loaded[6]
        item_genres = {item: [str(column) for column in np.flatnonzero(genre_binary[index])]
                       for index, item in enumerate(items) if index}
        validation_truth = grouped(valid)
        dev_truth = {user: validation_truth[user] for user in dev}
        detailed[str(seed)], aggregate[str(seed)] = {}, {}
        for name in MODEL_NAMES:
            selected = selections[name]
            scores = np.load(directory/selected['scores_file'], allow_pickle=False)
            if array_digest(scores) != selected['scores_array_sha256'] or scores.shape != matrix.shape:
                raise ValueError('Selected prediction payload differs')
            recs = recommendations(scores, users, items, matrix != 0, 10)
            metrics = evaluate_ratings(recs, dev, validation, history, items[1:], counts, 10)
            detailed[str(seed)][name] = metrics
            aggregate[str(seed)][name] = aggregate_metrics(metrics, recs, dev, counts, items[1:])
            aggregate[str(seed)][name]['groups'] = analyze_groups(
                {user: recs[user] for user in dev}, dev_truth, history, items[1:], counts,
                item_genres, k=10, group_users=users)
        write_json(directory/'development-metrics.json', detailed[str(seed)])
    references = matched_references(reference_evidence, reference_run, inputs, aggregate, locked_slim)
    result = {'schema_version': 1, 'stage': 'reused_development', 'study_kind': 'post_final_test_exploratory',
        'test_read': False, 'dataset_test_previously_evaluated': True, 'fresh_confirmation': False, 'seeds': {}}
    for seed in frozen['seeds']:
        key = str(seed)
        selections = json.loads((research/key/'selection.json').read_text())
        for selected in selections.values():
            selected.get('solver_diagnostics', {}).pop('fallback_items', None)
        comparisons = {}
        contrasts = [('categorical', 'binary_expanded'), ('shuffled_categories', 'binary_expanded'),
            ('categorical', 'shuffled_categories'), ('hybrid_real3', 'hybrid_baseline2'),
            ('hybrid_shuffled3', 'hybrid_baseline2'), ('hybrid_real3', 'hybrid_shuffled3')]
        for name, baseline in contrasts:
            for endpoint in ('all_observed', 'liked_ratings'):
                a = detailed[key][name][endpoint]
                b = detailed[key][baseline][endpoint]
                comparison_name = f'{name}_minus_'+('binary' if baseline == 'binary_expanded' else baseline)+f':{endpoint}'
                if a is None or b is None:
                    comparisons[comparison_name] = None
                    continue
                if set(a['per_user']) != set(b['per_user']):
                    raise ValueError('Paired development users differ')
                delta = np.array([a['per_user'][u]['ndcg@10']-b['per_user'][u]['ndcg@10'] for u in sorted(a['per_user'])])
                comparisons[comparison_name] = {'users': len(delta),
                    'mean_ndcg_delta': float(delta.mean()), 'fraction_users_improved': float(np.mean(delta > 0)),
                    'interpretation': 'Descriptive reused-development comparison; no new confidence claim.'}
        result['seeds'][key] = {'models': aggregate[key], 'selections': selections,
            'references': references[key], 'comparison': comparisons,
            'cohorts': {'meta_fit': len(json.loads((research/key/'cohorts.json').read_text())['meta_fit']),
                        'development': aggregate[key]['binary_expanded']['denominators']['all_observed_users']}}
    verify_barrier(research)
    write_json(research/'DEVELOPMENT-OPENED.json', {'status': 'complete',
        'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'), 'test_read': False, 'fresh_confirmation': False})
    write_json(research/'aggregates.json', result)
    write_json(research/'manifest.json', {'status': 'complete', 'test_read': False, 'test_evaluated': False,
        'fresh_confirmation': False, 'source_sha256': source_hashes(), 'runtime': runtime_signature(),
        'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'),
        'reference_manifest_sha256': digest(reference_run/'manifest.json'),
        'reference_aggregate_sha256': digest(reference_evidence/'aggregates.json'),
        'input_signatures': inputs,
        'locked_slim_sources': locked_slim,
        'output_sha256': {p.relative_to(research).as_posix(): digest(p) for p in sorted(research.rglob('*')) if p.is_file()}})
    return result


def curate(research, output):
    from package_project import check_aggregate_only
    manifest = json.loads((research/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['test_read'] or manifest['source_sha256'] != source_hashes() or manifest['runtime'] != runtime_signature():
        raise ValueError('Incomplete or changed research run')
    verify_files(research, manifest['output_sha256'])
    data = json.loads((research/'aggregates.json').read_text())
    check_aggregate_only(data)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(research/'aggregates.json', output/'aggregates.json')
    shutil.copyfile(research/'protocol.json', output/'protocol.json')
    write_json(output/'provenance.json', {'manifest_sha256': digest(research/'manifest.json'),
        'selection_freeze_sha256': manifest['selection_freeze_sha256'], 'source_sha256': manifest['source_sha256'],
        'runtime': manifest['runtime'], 'input_signatures': manifest['input_signatures'],
        'locked_slim_sources': manifest['locked_slim_sources'],
        'reference_manifest_sha256': manifest['reference_manifest_sha256'],
        'reference_aggregate_sha256': manifest['reference_aggregate_sha256']})
    lines = ['# Categorical reconstruction: reused-development exploratory results', '',
        'The original dataset TEST was examined before this study. These results are not fresh held-out confirmation. '
        'Every declared seed selection was sealed before any development metric was computed. All model inputs and categorical centering use TRAIN only.', '',
        'The expanded binary model has 9 selection candidates. Each real/shuffled categorical family has 36 categorical candidates '
        'plus the same 9 cached binary candidates. The original-grid binary row selects from 3 of those binary fits. '
        'Thus tuning opportunities differ; nested fallback can select the binary model. The shuffle preserves per-item TRAIN rating histograms and observation identities.', '',
        '| Seed | Model | All nDCG@10 | Liked nDCG@10 | Selected features | Lambda | Category ratio |',
        '|---|---|---:|---:|---|---:|---:|']
    means = {name: [] for name in MODEL_NAMES}
    for seed, row in data['seeds'].items():
        for name in MODEL_NAMES:
            metric, selected = row['models'][name], row['selections'][name]
            liked = metric['liked_ratings']['ndcg@10'] if metric['liked_ratings'] else None
            liked_text = f'{liked:.6f}' if liked is not None else 'n/a'
            lines.append(f"| {seed} | {name} | {metric['all_observed']['ndcg@10']:.6f} | {liked_text} | {selected['feature_kind']} | {selected['penalty']:g} | {selected['category_ratio'] if selected['category_ratio'] is not None else '—'} |")
            means[name].append((metric['all_observed']['ndcg@10'], liked))
    lines += ['', 'Equal-seed means are descriptive; overlapping splits are not independent datasets.', '',
              '| Model | Mean all nDCG@10 | Mean liked nDCG@10 |', '|---|---:|---:|']
    for name, rows in means.items():
        liked = [row[1] for row in rows if row[1] is not None]
        lines.append(f'| {name} | {np.mean([row[0] for row in rows]):.6f} | '+(f'{np.mean(liked):.6f}' if liked else 'n/a')+' |')
    lines += ['', 'Existing locked EASE/SLIM/PositiveEASE development references, full ranking metrics, explicit denominators, '
        'frequency diagnostics and paired descriptive differences are retained in aggregates.json. Reference models have different '
        'information/selection budgets. No probabilities, likelihood equivalence, population confidence claim or general architectural novelty is inferred.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    write_json(output/'SHA256.json', {p.name: digest(p) for p in sorted(output.iterdir())})


def benchmark(source, ratings, output):
    if output.exists():
        raise ValueError('Refuse benchmark overwrite')
    before = source_hashes()
    _, _, _, matrix, upstream = load_train(source, ratings)
    started = time.monotonic()
    prepared = prepared_model(matrix)
    preparation_seconds = time.monotonic()-started
    timing = {}
    for name, ratio in (('binary', None), ('categorical', 1.)):
        started = time.monotonic()
        scores, diagnostic = solve(prepared, 250., ratio)
        timing[name] = {'seconds': time.monotonic()-started, 'scores_array_sha256': array_digest(scores), 'solver_diagnostics': diagnostic}
    if source_hashes() != before:
        raise ValueError('Benchmark source changed')
    result = {'status': 'complete', 'validation_read': False, 'test_read': False, 'development_evaluated': False,
        'source_sha256': before, 'runtime': runtime_signature(), 'preparation_seconds': preparation_seconds,
        'timing': timing, 'training_categories_sha256': array_digest(matrix),
        'source_manifest_sha256': digest(source/'manifest.json'), 'data_sha256': upstream['data_sha256']}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def run(args):
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Duplicate seeds')
    declared = protocol(args.seeds)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', declared)
    # Seal the immutable reference evidence before any new candidate fit, but
    # inspect its development metric values only after all new choices freeze.
    write_json(args.out/'reference-input-signature.json', {
        'manifest_sha256': digest(args.reference_run/'manifest.json'),
        'aggregate_sha256': digest(args.reference_evidence/'aggregates.json')})
    slim = {}
    for seed in args.seeds:
        root = args.slim_root if seed == 2026 else args.slim_more_root
        path, selected = reference_path(root, seed, 'SLIMElastic')
        slim[seed] = (path, {**selected, 'selection_file': str(root/f'expert-selection-{seed}.json')})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(select_seed, args.source_root/f'{seed}-EASE-1', args.ratings,
            args.item_metadata, args.cohort_root/str(seed), slim[seed][0],
            args.out/str(seed), seed, declared, slim[seed][1]): seed for seed in args.seeds}
        for future in as_completed(futures):
            future.result()
    freeze_selections(args.out)
    pinned = json.loads((args.out/'reference-input-signature.json').read_text())
    if pinned != {'manifest_sha256': digest(args.reference_run/'manifest.json'), 'aggregate_sha256': digest(args.reference_evidence/'aggregates.json')}:
        raise ValueError('Reference evidence changed during selection')
    evaluate_selected(args.out, args.source_root, args.ratings, args.item_metadata, args.reference_evidence, args.reference_run)
    curate(args.out, args.evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    bench = sub.add_parser('benchmark')
    bench.add_argument('--source', type=Path, required=True)
    bench.add_argument('--ratings', type=Path, required=True)
    bench.add_argument('--out', type=Path, required=True)
    running = sub.add_parser('run')
    running.add_argument('--source-root', type=Path, default=Path('runs/research-v2'))
    running.add_argument('--cohort-root', type=Path, default=Path('runs/adaptive-v1'))
    running.add_argument('--reference-evidence', type=Path, default=Path('evidence/field-reference-v1'))
    running.add_argument('--reference-run', type=Path, default=Path('runs/field-reference-v1'))
    running.add_argument('--slim-root', type=Path, default=Path('runs/coverage-v1'))
    running.add_argument('--slim-more-root', type=Path, default=Path('runs/coverage-v1-more'))
    running.add_argument('--ratings', type=Path, required=True)
    running.add_argument('--item-metadata', type=Path, required=True)
    running.add_argument('--out', type=Path, required=True)
    running.add_argument('--evidence', type=Path, required=True)
    running.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    running.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    curation = sub.add_parser('curate')
    curation.add_argument('--research', type=Path, required=True)
    curation.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'benchmark':
        print(json.dumps(benchmark(args.source, args.ratings, args.out), indent=2))
    elif args.command == 'curate':
        curate(args.research, args.out)
    else:
        if 'runs' not in args.out.parts:
            parser.error('Detailed results must stay under runs/')
        run(args)


if __name__ == '__main__':
    main()
