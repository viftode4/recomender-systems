"""Standalone pair reconstruction: TRAIN fitting, meta selection, global DEV seal.

This is exploratory reuse of MovieLens100K after its original TEST was examined.
No TEST split or prior final-test output is opened by this runner.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
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

from categorical_experiment import array_digest, categorical_matrix
from exception_experiment import evaluate_ratings, load_source, recommendations
from exploratory.categorical_reconstruction import run_experiment as categorical
from exploratory.categorical_reconstruction.group_analysis import analyze_groups
from exploratory.categorical_reconstruction.run_experiment import (
    aggregate_metrics, load_selection_inputs, load_train, meta_ndcg, runtime_signature,
    verify_files)
from field_reference_comparison import identity_digest
from study import digest, grouped, partition_users, write_json


LAMBDAS = (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)
PAIR_WEIGHTS = (.01, .1, 1., 10.)
MODEL_NAMES = ('binary', 'pair_only', 'nested')
REFERENCE_NAMES = ('locked_binary', 'locked_categorical')
REFERENCE_KEYS = {'locked_binary': 'binary_expanded', 'locked_categorical': 'categorical'}
SOURCE_FILES = tuple(dict.fromkeys((
    'exploratory/context_interactions/run_experiment.py',
    'exploratory/context_interactions/model.py',
    'exploratory/context_interactions/PROTOCOL.md',
    *categorical.SOURCE_FILES, 'package_project.py')))


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def protocol(seeds):
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('Require distinct declared seeds')
    return {'schema_version': 1, 'study_kind': 'post_final_test_exploratory',
        'stage': 'reused_development', 'test_read': False, 'test_evaluated': False,
        'dataset_test_previously_evaluated': True, 'fresh_confirmation': False,
        'seeds': list(seeds), 'models': list(MODEL_NAMES), 'references': list(REFERENCE_NAMES),
        'lambdas': list(LAMBDAS), 'pair_weights': list(PAIR_WEIGHTS),
        'ordered_candidates': [{'candidate_id': candidate_id(penalty, weight), 'penalty': penalty, 'pair_weight': weight}
                               for penalty, weight in candidates()],
        'candidate_budgets': {'binary': len(LAMBDAS),
            'pair_only': len(LAMBDAS)*len(PAIR_WEIGHTS),
            'nested': len(LAMBDAS)*(1+len(PAIR_WEIGHTS)),
            'unique_fits_per_seed': len(LAMBDAS)*(1+len(PAIR_WEIGHTS))},
        'training_target': 'Original TRAIN observation indicator; missing records are reconstruction zeros, not confirmed dislikes.',
        'training_inputs': 'Binary TRAIN histories and all distinct unordered item conjunctions; no ratings, metadata or pretrained scores enter model fitting.',
        'target_mask': 'Exclude every target own-item binary feature and every pair containing that target.',
        'normalization': 'TRAIN-only nu=sum_u choose(history_size_u,2)/sum_u history_size_u; beta=0 is the exact binary submodel.',
        'selection_metric': 'meta_fit_all_observed_ndcg@10',
        'selection_rule': 'Binary candidates in declared lambda order, then pair candidates in lambda-major pair-weight order; exact ties retain the first maximum.',
        'cohorts': 'Reuse original seeded half-user meta-fit/development partition; only meta-fit validation pairs reach candidate scoring.',
        'selection_boundary': 'All declared seed manifests and locked reference scores are hash-sealed before any development rating is parsed or metric computed.',
        'ranking': 'One unbounded score array for all-observed and liked-record endpoints; exclude PAD and original TRAIN records; k=10.',
        'groups': 'TRAIN activity terciles over every source user and TRAIN top20% popularity head versus tail; frozen definitions, no group metric selects a model. Each endpoint uses its own eligible user cohort.',
        'inference': 'Original TRAIN only; no validation context assimilation.',
        'references_policy': 'Previously selected expanded binary and categorical reconstruction scores are frozen context, never inputs or tuning targets.',
        'claims': 'Descriptive reused-development evidence, unequal family search budgets, overlapping seeds, no fresh confirmation or architectural priority claim.',
        'runtime': runtime_signature(), 'source_sha256': source_hashes()}


def candidate_id(penalty, pair_weight):
    return f'lambda-{penalty:g}-beta-{pair_weight:g}'


def candidates():
    return [(penalty, 0.) for penalty in LAMBDAS]+[
        (penalty, weight) for penalty in LAMBDAS for weight in PAIR_WEIGHTS]


def choose(grid, role):
    if role not in MODEL_NAMES:
        raise ValueError('Unknown model role')
    allowed = [row for row in grid if role == 'nested'
               or (row['pair_weight'] == 0) == (role == 'binary')]
    if not allowed:
        raise ValueError('Empty candidate family')
    return max(allowed, key=lambda row: row['meta_fit_all_observed_ndcg@10'])


def prepared_model(matrix):
    from exploratory.context_interactions.model import prepare_features
    return prepare_features(np.asarray(matrix) != 0)


def solve(prepared, penalty, pair_weight):
    from exploratory.context_interactions.model import fit
    model = fit(prepared, lambda_binary=penalty, pair_weight=pair_weight)
    scores = np.asarray(model.predict(), dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any(scores[:, 0]):
        raise ValueError('Invalid unmasked prediction array')
    return scores, model.diagnostics


def lock_references(reference_root, seed, inputs, output):
    """Copy old selected predictions, never parse old development metrics."""
    categorical.verify_barrier(reference_root)
    manifest = json.loads((reference_root/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('test_read', True)
            or manifest.get('test_evaluated', True)
            or manifest['selection_freeze_sha256'] != digest(reference_root/'SELECTIONS-FROZEN.json')
            or manifest['input_signatures'][str(seed)] != inputs):
        raise ValueError('Locked categorical reference provenance differs')
    directory = reference_root/str(seed)
    selected = json.loads((directory/'selection.json').read_text())
    (output/'locked-scores').mkdir()
    result = {}
    for name, old_name in REFERENCE_KEYS.items():
        row = selected[old_name]
        path = directory/row['scores_file']
        if digest(path) != row['scores_file_sha256']:
            raise ValueError('Locked reference score signature differs')
        relative = path.relative_to(reference_root).as_posix()
        if manifest['output_sha256'].get(relative) != digest(path):
            raise ValueError('Reference completion does not seal selected score')
        scores = np.load(path, allow_pickle=False)
        if array_digest(scores) != row['scores_array_sha256'] or not np.isfinite(scores).all():
            raise ValueError('Locked reference array differs')
        local = f'locked-scores/{name}.npy'
        shutil.copyfile(path, output/local)
        result[name] = {'source_model': old_name, 'candidate_id': row['candidate_id'],
            'penalty': row['penalty'], 'category_ratio': row['category_ratio'],
            'feature_kind': row['feature_kind'], 'scores_file': local,
            'scores_file_sha256': digest(output/local), 'scores_array_sha256': array_digest(scores),
            'source_selection_manifest_sha256': digest(directory/'selection-manifest.json')}
    write_json(output/'locked-references.json', result)
    return result


def select_seed(source, ratings, metadata, cohort_directory, reference_root, output, seed, declared):
    if declared != protocol(declared['seeds']):
        raise ValueError('Protocol/source/runtime changed before selection')
    started = time.monotonic()
    users, items, train, valid, matrix, fit, dev, inputs = load_selection_inputs(
        source, ratings, metadata, cohort_directory, seed)
    meta_pairs = [(user, item) for user, item in valid if user in fit]
    if set(grouped(meta_pairs)) != set(fit):
        raise ValueError('Missing meta-fit validation pairs')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'input-signature.json', inputs)
    write_json(output/'cohorts.json', {'meta_fit': sorted(fit), 'development': sorted(dev)})
    np.savez_compressed(output/'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    for name in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(source/name, output/name)
    references = lock_references(reference_root, seed, inputs, output)
    prepared = prepared_model(matrix)
    grid, best_arrays, best_rows = [], {}, {}
    for penalty, pair_weight in candidates():
        begin = time.monotonic()
        try:
            scores, diagnostics = solve(prepared, penalty, pair_weight)
        except Exception as error:
            write_json(output/'numerical-failure.json', {'candidate_id': candidate_id(penalty, pair_weight),
                'error_type': type(error).__name__, 'error': str(error), 'test_read': False, 'development_evaluated': False})
            raise
        if scores.shape != matrix.shape:
            raise ValueError('Prediction dimensions differ from TRAIN')
        value = meta_ndcg(scores, users, items, train, meta_pairs, fit)
        row = {'candidate_id': candidate_id(penalty, pair_weight),
            'feature_kind': 'binary' if pair_weight == 0 else 'pair',
            'penalty': penalty, 'pair_weight': pair_weight,
            'meta_fit_all_observed_ndcg@10': value, 'scores_array_sha256': array_digest(scores),
            'fit_and_selection_seconds': time.monotonic()-begin, 'solver_diagnostics': diagnostics}
        grid.append(row)
        for role in MODEL_NAMES:
            if role != 'nested' and (pair_weight == 0) != (role == 'binary'):
                continue
            if role not in best_rows or value > best_rows[role]['meta_fit_all_observed_ndcg@10']:
                best_rows[role], best_arrays[role] = row, scores.copy()
        write_json(output/'candidate-grid.json', grid)
        print(json.dumps({'seed': seed, 'candidate': row['candidate_id'], 'meta_ndcg': value,
                          'seconds': row['fit_and_selection_seconds']}), flush=True)
    (output/'selected-scores').mkdir()
    selections, replayed = {}, {}
    for role in MODEL_NAMES:
        row = choose(grid, role)
        identifier = row['candidate_id']
        if identifier not in replayed:
            replayed[identifier] = solve(prepared, row['penalty'], row['pair_weight'])[0]
        scores = replayed[identifier]
        if not np.array_equal(scores, best_arrays[role]):
            raise ValueError('Selected deterministic refit differs')
        name = f'selected-scores/{role}.npy'
        np.save(output/name, scores, allow_pickle=False)
        if not np.array_equal(np.load(output/name, allow_pickle=False), scores):
            raise ValueError('Selected score export differs')
        selections[role] = {**row, 'selection_cohort': 'reused_meta_fit',
            'selection_metric': 'all_observed_ndcg@10',
            'candidate_count': declared['candidate_budgets'][role],
            'scores_file': name, 'scores_file_sha256': digest(output/name),
            'refit_replay_exact': True, 'score_file_replay_exact': True}
    # A checked reference, not a tolerance-based selection rule.
    old = np.load(output/references['locked_binary']['scores_file'], allow_pickle=False)
    new = best_arrays['binary']
    if (old.shape != new.shape or references['locked_binary']['penalty'] != selections['binary']['penalty']
            or not np.allclose(old, new, atol=2e-9, rtol=2e-9)
            or recommendations(old, users, items, matrix != 0, 10)
               != recommendations(new, users, items, matrix != 0, 10)):
        raise ValueError('New binary control differs from locked expanded binary reference')
    write_json(output/'selection.json', selections)
    manifest = {'status': 'selected', 'seed': seed, 'test_read': False, 'development_evaluated': False,
        'source_sha256': source_hashes(), 'runtime': runtime_signature(), 'input_signature': inputs,
        'unique_fits': len(grid), 'elapsed_seconds': time.monotonic()-started,
        'binary_reference_replay': {'maximum_absolute_score_error': float(np.max(np.abs(old-new))),
            'score_tolerance_atol_rtol': [2e-9, 2e-9], 'exact_top10_order': True, 'same_selected_penalty': True},
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output/'selection-manifest.json', manifest)
    return manifest


def validate_choices(directory, declared):
    grid = json.loads((directory/'candidate-grid.json').read_text())
    expected = [(penalty, 0.) for penalty in declared['lambdas']]+[
        (penalty, weight) for penalty in declared['lambdas'] for weight in declared['pair_weights']]
    if ([(row['penalty'], row['pair_weight']) for row in grid] != expected
            or any(row['candidate_id'] != candidate_id(row['penalty'], row['pair_weight'])
                   or row['feature_kind'] != ('binary' if row['pair_weight'] == 0 else 'pair')
                   or not np.isfinite(row['meta_fit_all_observed_ndcg@10'])
                   or not 0 <= row['meta_fit_all_observed_ndcg@10'] <= 1 for row in grid)):
        raise ValueError('Incomplete or inconsistent candidate grid')
    selected = json.loads((directory/'selection.json').read_text())
    if set(selected) != set(MODEL_NAMES):
        raise ValueError('Selected family set differs')
    with np.load(directory/'training-categories.npz', allow_pickle=False) as archive:
        shape = archive['ratings'].shape
    for role, row in selected.items():
        maximum = choose(grid, role)
        if (any(row.get(key) != value for key, value in maximum.items())
                or row['candidate_count'] != declared['candidate_budgets'][role]
                or row['selection_cohort'] != 'reused_meta_fit' or row['selection_metric'] != 'all_observed_ndcg@10'
                or row['refit_replay_exact'] is not True or row['score_file_replay_exact'] is not True
                or row['scores_file'] != f'selected-scores/{role}.npy'):
            raise ValueError('Selected candidate is not its exact maximum or replay is missing')
        check_scores(directory, row, shape)
    references = json.loads((directory/'locked-references.json').read_text())
    if set(references) != set(REFERENCE_NAMES):
        raise ValueError('Locked reference set differs')
    for name, row in references.items():
        if row['scores_file'] != f'locked-scores/{name}.npy':
            raise ValueError('Locked reference score path differs')
        check_scores(directory, row, shape)


def check_scores(directory, row, shape):
    path = directory/row['scores_file']
    if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != row['scores_file_sha256']:
        raise ValueError('Score file signature differs')
    scores = np.load(path, allow_pickle=False)
    if (scores.shape != shape or not np.isfinite(scores).all() or np.any(scores[:, 0])
            or array_digest(scores) != row['scores_array_sha256']):
        raise ValueError('Score array differs')
    return scores


def checked_manifests(research, declared):
    hashes = {}
    for seed in declared['seeds']:
        directory = research/str(seed)
        path = directory/'selection-manifest.json'
        manifest = json.loads(path.read_text())
        if (manifest['status'] != 'selected' or manifest['seed'] != seed or manifest['test_read']
                or manifest['development_evaluated'] or manifest['source_sha256'] != source_hashes()
                or manifest['runtime'] != runtime_signature()
                or manifest['unique_fits'] != declared['candidate_budgets']['unique_fits_per_seed']):
            raise ValueError('Invalid selection manifest')
        verify_files(directory, manifest['output_sha256'])
        validate_choices(directory, declared)
        hashes[str(seed)] = digest(path)
    return hashes


def freeze_selections(research):
    declared = json.loads((research/'protocol.json').read_text())
    if declared != protocol(declared['seeds']):
        raise ValueError('Declared protocol differs')
    manifests = checked_manifests(research, declared)
    frozen = {'status': 'all_selections_frozen', 'seeds': declared['seeds'], 'test_read': False,
        'development_evaluated': False, 'protocol_sha256': digest(research/'protocol.json'),
        'source_sha256': source_hashes(), 'runtime': runtime_signature(),
        'reference_input_signature_sha256': digest(research/'reference-input-signature.json'),
        'selection_manifest_sha256': manifests}
    with (research/'SELECTIONS-FROZEN.json').open('x') as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return frozen


def verify_barrier(research):
    frozen = json.loads((research/'SELECTIONS-FROZEN.json').read_text())
    declared = json.loads((research/'protocol.json').read_text())
    if (frozen['status'] != 'all_selections_frozen' or frozen['test_read'] or frozen['development_evaluated']
            or frozen['seeds'] != declared['seeds'] or declared != protocol(declared['seeds'])
            or frozen['protocol_sha256'] != digest(research/'protocol.json')
            or frozen['reference_input_signature_sha256'] != digest(research/'reference-input-signature.json')
            or frozen['source_sha256'] != source_hashes() or frozen['runtime'] != runtime_signature()
            or frozen['selection_manifest_sha256'] != checked_manifests(research, declared)):
        raise ValueError('Global selection barrier differs')
    return frozen


def paired_difference(first, second, endpoint):
    a, b = first[endpoint], second[endpoint]
    if a is None or b is None:
        return None
    if set(a['per_user']) != set(b['per_user']):
        raise ValueError('Paired development cohorts differ')
    delta = np.asarray([a['per_user'][user]['ndcg@10']-b['per_user'][user]['ndcg@10']
                        for user in sorted(a['per_user'])])
    return {'users': len(delta), 'mean_ndcg_delta': float(delta.mean()),
        'fraction_users_improved': float(np.mean(delta > 0)),
        'interpretation': 'Descriptive reused-development comparison; no new confidence claim.'}


def public_selection(row):
    # Explicit whitelist: solver diagnostics can contain individual target IDs.
    return {key: row[key] for key in ('candidate_id', 'feature_kind', 'penalty', 'pair_weight',
        'meta_fit_all_observed_ndcg@10', 'candidate_count', 'scores_array_sha256')}


def evaluate_selected(research, source_root, ratings, metadata):
    frozen = verify_barrier(research)  # Must precede load_source, which parses VALID ratings.
    with (research/'DEVELOPMENT-OPENED.json').open('x') as stream:
        json.dump({'status': 'started', 'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'),
                   'test_read': False, 'fresh_confirmation': False}, stream, indent=2)
    result = {'schema_version': 1, 'stage': 'reused_development', 'study_kind': 'post_final_test_exploratory',
        'test_read': False, 'dataset_test_previously_evaluated': True, 'fresh_confirmation': False, 'seeds': {}}
    input_signatures = {}
    for seed in frozen['seeds']:
        directory = research/str(seed)
        loaded = load_source(source_root/f'{seed}-EASE-1', ratings, metadata)
        users, items, train, valid, training, validation = loaded[:6]
        matrix = categorical_matrix(users, items, training)
        signature = json.loads((directory/'input-signature.json').read_text())
        input_signatures[str(seed)] = signature
        if (identity_digest(users, items) != signature['ordered_identity_sha256']
                or array_digest(matrix) != signature['training_categories_sha256']
                or loaded[-1]['data_sha256'] != signature['data_sha256']
                or any(loaded[-1]['split_sha256'][part] != signature['split_sha256'][part] for part in ('train', 'valid'))):
            raise ValueError('Development source differs from selected TRAIN inputs')
        fit, dev = partition_users(users, seed)
        if json.loads((directory/'cohorts.json').read_text()) != {'meta_fit': sorted(fit), 'development': sorted(dev)}:
            raise ValueError('Development cohort differs')
        history, counts = grouped(train), Counter(item for _, item in train)
        choices = json.loads((directory/'selection.json').read_text())
        references = json.loads((directory/'locked-references.json').read_text())
        genres = {item: [str(column) for column in np.flatnonzero(loaded[6][index])]
                  for index, item in enumerate(items) if index}
        all_truth = {user: values for user, values in grouped(valid).items() if user in dev}
        liked_truth = grouped([(user, item) for (user, item), value in validation.items() if user in dev and value >= 4])
        detailed, aggregate = {}, {}
        for name, selected in {**choices, **references}.items():
            scores = check_scores(directory, selected, matrix.shape)
            recs = recommendations(scores, users, items, matrix != 0, 10)
            metrics = evaluate_ratings(recs, dev, validation, history, items[1:], counts, 10)
            detailed[name] = metrics
            aggregate[name] = aggregate_metrics(metrics, recs, dev, counts, items[1:])
            for field, truth in (('groups', all_truth), ('liked_groups', liked_truth)):
                aggregate[name][field] = analyze_groups({user: recs[user] for user in truth},
                    truth, history, items[1:], counts, genres, k=10, group_users=users) if truth else None
        write_json(directory/'development-metrics.json', detailed)
        contrasts = [('pair_only', 'binary'), ('nested', 'binary'),
            ('pair_only', 'locked_categorical'), ('nested', 'locked_categorical'),
            ('binary', 'locked_binary')]
        result['seeds'][str(seed)] = {'models': {name: aggregate[name] for name in MODEL_NAMES},
            'references': {name: aggregate[name] for name in REFERENCE_NAMES},
            'selections': {name: public_selection(row) for name, row in choices.items()},
            'reference_selections': {name: {key: row[key] for key in ('source_model', 'candidate_id', 'penalty',
                'category_ratio', 'feature_kind', 'scores_array_sha256')} for name, row in references.items()},
            'comparison': {f'{first}_minus_{second}:{endpoint}': paired_difference(detailed[first], detailed[second], endpoint)
                for first, second in contrasts for endpoint in ('all_observed', 'liked_ratings')},
            'cohorts': {'meta_fit': len(fit), 'development': len(dev)}}
    verify_barrier(research)
    write_json(research/'DEVELOPMENT-OPENED.json', {'status': 'complete',
        'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'), 'test_read': False, 'fresh_confirmation': False})
    write_json(research/'aggregates.json', result)
    write_json(research/'manifest.json', {'status': 'complete', 'test_read': False, 'test_evaluated': False,
        'fresh_confirmation': False, 'source_sha256': source_hashes(), 'runtime': runtime_signature(),
        'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'),
        'reference_input_signature': json.loads((research/'reference-input-signature.json').read_text()),
        'input_signatures': input_signatures,
        'output_sha256': {p.relative_to(research).as_posix(): digest(p) for p in sorted(research.rglob('*')) if p.is_file()}})
    return result


def curate(research, output):
    from package_project import check_aggregate_only
    manifest = json.loads((research/'manifest.json').read_text())
    if (manifest['status'] != 'complete' or manifest['test_read'] or manifest['test_evaluated']
            or manifest['source_sha256'] != source_hashes() or manifest['runtime'] != runtime_signature()):
        raise ValueError('Incomplete or changed research run')
    verify_barrier(research)
    verify_files(research, manifest['output_sha256'])
    data = json.loads((research/'aggregates.json').read_text())
    check_aggregate_only(data)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('aggregates.json', 'protocol.json'):
        shutil.copyfile(research/name, output/name)
    write_json(output/'provenance.json', {'manifest_sha256': digest(research/'manifest.json'),
        'selection_freeze_sha256': manifest['selection_freeze_sha256'], 'source_sha256': manifest['source_sha256'],
        'runtime': manifest['runtime'], 'input_signatures': manifest['input_signatures'],
        'reference_input_signature': manifest['reference_input_signature']})
    lines = ['# Context interactions: exploratory reused-development results', '',
        'MovieLens100K TEST was already examined before this study. No TEST is opened here. '
        'All seed selections were sealed before any new development metric. This is not fresh confirmation.', '',
        'Binary selects from 9 penalties. Pair-only selects from 36 pair models. Nested selects from their '
        '45-candidate union and may choose binary. Search budgets differ. Old binary and categorical predictions '
        'are frozen references. All inputs to new models are binary TRAIN histories.', '',
        '| Seed | Model | All nDCG@10 | Liked nDCG@10 | Lambda | Pair weight |',
        '|---|---|---:|---:|---:|---:|']
    means = {name: [] for name in (*MODEL_NAMES, *REFERENCE_NAMES)}
    for seed, row in data['seeds'].items():
        for name, metrics in {**row['models'], **row['references']}.items():
            liked = metrics['liked_ratings']['ndcg@10'] if metrics['liked_ratings'] else None
            selected = row['selections'].get(name, row['reference_selections'].get(name))
            means[name].append((metrics['all_observed']['ndcg@10'], liked))
            liked_text = f'{liked:.6f}' if liked is not None else 'n/a'
            lines.append(f"| {seed} | {name} | {metrics['all_observed']['ndcg@10']:.6f} | {liked_text} | {selected['penalty']:g} | {selected.get('pair_weight', 'reference')} |")
    lines += ['', 'Equal-seed means are descriptive; overlapping splits are not independent datasets.', '',
        '| Model | Mean all nDCG@10 | Mean liked nDCG@10 |', '|---|---:|---:|']
    for name, values in means.items():
        liked = [value[1] for value in values if value[1] is not None]
        lines.append(f'| {name} | {np.mean([value[0] for value in values]):.6f} | '+(f'{np.mean(liked):.6f}' if liked else 'n/a')+' |')
    lines += ['', 'aggregates.json includes both relevance endpoints, their explicit denominators, '
        'TRAIN-defined activity/popularity groups, exposure, diversity, and paired descriptive nDCG differences. '
        'Group diagnostics do not select models. Scores are unbounded ranking values, not probabilities. '
        'Higher-order linear recommendation has existing prior art; this experiment makes no architectural priority claim.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    write_json(output/'SHA256.json', {p.name: digest(p) for p in sorted(output.iterdir())})


def benchmark(source, ratings, output):
    before = source_hashes()
    _, _, _, matrix, upstream = load_train(source, ratings)
    begin = time.monotonic()
    prepared = prepared_model(matrix)
    preparation = time.monotonic()-begin
    timing = {}
    for name, weight in (('binary', 0.), ('pair', 1.)):
        begin = time.monotonic()
        scores, diagnostics = solve(prepared, 250., weight)
        timing[name] = {'seconds': time.monotonic()-begin,
            'scores_array_sha256': array_digest(scores), 'solver_diagnostics': diagnostics}
    if source_hashes() != before:
        raise ValueError('Benchmark source changed')
    result = {'status': 'complete', 'validation_read': False, 'test_read': False, 'development_evaluated': False,
        'source_sha256': before, 'runtime': runtime_signature(), 'preparation_seconds': preparation,
        'timing': timing, 'training_categories_sha256': array_digest(matrix),
        'source_manifest_sha256': digest(source/'manifest.json'), 'data_sha256': upstream['data_sha256']}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def run(args):
    declared = protocol(args.seeds)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', declared)
    write_json(args.out/'reference-input-signature.json', {
        'manifest_sha256': digest(args.reference_run/'manifest.json'),
        'selection_freeze_sha256': digest(args.reference_run/'SELECTIONS-FROZEN.json')})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(select_seed, args.source_root/f'{seed}-EASE-1', args.ratings,
            args.item_metadata, args.cohort_root/str(seed), args.reference_run,
            args.out/str(seed), seed, declared) for seed in args.seeds]
        for future in as_completed(futures):
            future.result()
    pinned = json.loads((args.out/'reference-input-signature.json').read_text())
    if pinned != {'manifest_sha256': digest(args.reference_run/'manifest.json'),
                  'selection_freeze_sha256': digest(args.reference_run/'SELECTIONS-FROZEN.json')}:
        raise ValueError('Locked reference changed during selection')
    freeze_selections(args.out)
    evaluate_selected(args.out, args.source_root, args.ratings, args.item_metadata)
    curate(args.out, args.evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    bench = commands.add_parser('benchmark')
    bench.add_argument('--source', type=Path, required=True)
    bench.add_argument('--ratings', type=Path, required=True)
    bench.add_argument('--out', type=Path, required=True)
    running = commands.add_parser('run')
    running.add_argument('--source-root', type=Path, default=Path('runs/research-v2'))
    running.add_argument('--cohort-root', type=Path, default=Path('runs/adaptive-v1'))
    running.add_argument('--reference-run', type=Path, default=Path('runs/categorical-reconstruction-v1'))
    running.add_argument('--ratings', type=Path, required=True)
    running.add_argument('--item-metadata', type=Path, required=True)
    running.add_argument('--out', type=Path, default=Path('runs/context-interactions-v1'))
    running.add_argument('--evidence', type=Path, default=Path('exploratory/context_interactions/evidence-v1'))
    running.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    running.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    curation = commands.add_parser('curate')
    curation.add_argument('--research', type=Path, required=True)
    curation.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'benchmark':
        print(json.dumps(benchmark(args.source, args.ratings, args.out), indent=2))
    elif args.command == 'curate':
        curate(args.research, args.out)
    else:
        if 'runs' not in args.out.parts:
            parser.error('Individual data must stay under runs/')
        run(args)


if __name__ == '__main__':
    main()
