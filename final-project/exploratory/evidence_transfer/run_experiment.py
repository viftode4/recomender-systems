"""Masked TRAIN evidence learning with meta checkpoint choice and a global DEV seal.

Only synthetic tests and an explicitly invoked TRAIN benchmark may precede the
fixed study. This reused MovieLens100K experiment never opens TEST partitions.
"""
import argparse
from collections import Counter
import json
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
import torch

from categorical_experiment import array_digest, categorical_matrix
from exception_experiment import evaluate_ratings, load_source, recommendations
from exploratory.categorical_reconstruction import run_experiment as categorical
from exploratory.categorical_reconstruction.group_analysis import analyze_groups
from exploratory.categorical_reconstruction.run_experiment import (
    aggregate_metrics, load_selection_inputs, load_train, meta_ndcg, verify_files)
from exploratory.context_interactions.run_experiment import paired_difference
from field_reference_comparison import identity_digest
from study import digest, grouped, partition_users, write_json


ARMS = ('full_pattern', 'no_pattern', 'marginal_only')
MODEL_NAMES = (*ARMS, 'analytic_donor', 'full_zero_pattern')
CHECKPOINTS = (0, 10, 30, 60, 100)
RETENTIONS = (.8, .9)
WIDTH, BATCH_SIZE = 16, 64
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4
SOURCE_FILES = tuple(dict.fromkeys((
    'exploratory/evidence_transfer/run_experiment.py', 'exploratory/evidence_transfer/model.py',
    'exploratory/evidence_transfer/PROTOCOL.md', 'exploratory/context_interactions/run_experiment.py',
    *categorical.SOURCE_FILES, 'package_project.py')))


def configure_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def runtime_signature():
    configure_torch()
    return {**categorical.runtime_signature(), 'torch': str(torch.__version__),
            'device': 'cpu', 'model_dtype': 'float32', 'deterministic_algorithms': True}


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def protocol(seeds):
    from exploratory.evidence_transfer.model import FEATURE_NAMES
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Require distinct seeds')
    return {'schema_version': 1, 'study_kind': 'post_final_test_exploratory', 'stage': 'reused_development',
        'test_read': False, 'test_evaluated': False, 'fresh_confirmation': False,
        'dataset_test_previously_evaluated': True, 'seeds': list(seeds), 'arms': list(ARMS),
        'models': list(MODEL_NAMES), 'references': ['locked_binary'], 'checkpoints': list(CHECKPOINTS),
        'retentions': list(RETENTIONS), 'feature_names': list(FEATURE_NAMES),
        'width': WIDTH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'optimizer': 'Adam', 'mask_seed_offset': 41001, 'initialization_seed_offset': 42001,
        'batch_seed_offset': 43001,
        'episodes': 'For each retention then original user order, randomly retain floor(retention*history_size), clipped to [1,n-1]; all remaining original TRAIN observations are targets. RNG seed+41001.',
        'training': 'All original TRAIN users; fixed episodes; full-catalog eligible candidates exclude PAD and episode context. Uniform target mass over hidden TRAIN items; equal episode/user weight. Missing candidates are softmax alternatives, not labeled dislikes.',
        'donors': 'Original binary TRAIN bank. Exclude the query user full row from every donor statistic, including marginal counts and denominators.',
        'normalization': 'One shared feature mean/scale fitted only on eligible TRAIN-episode candidate cells; reused for every arm and inference.',
        'controls': 'Identical allocated width, initialization, batch order, optimizer, steps and checkpoint budget. no_pattern zeros standardized pattern channels; marginal_only retains context count and marginal donor support. analytic_donor is unfitted weighted support fraction.',
        'intervention': 'full_zero_pattern uses selected full_pattern weights with standardized pattern channels zeroed at inference; no refit or checkpoint choice. This shifts the learned feature distribution and is not the trained no_pattern control.',
        'selection_metric': 'meta_fit_all_observed_ndcg@10',
        'selection_rule': 'Maximum over checkpoints in declared order; exact ties retain the earlier checkpoint. Each trainable arm has the same checkpoint budget.',
        'selection_boundary': 'All seed selections, feature/episode hashes, normalizer, model states and predictions are sealed before development rating parsing or new development metrics.',
        'ranking': 'Original TRAIN query histories, never assimilate VALID; exclude PAD and original TRAIN observations; one score ranking for all-observed and liked endpoints, k=10.',
        'groups': 'Original full-population TRAIN activity terciles and top20% TRAIN-frequency item head; diagnostics only, never selection.',
        'claims': 'Descriptive reused-development evidence; overlapping seeds are not independent datasets. No fresh confirmation or architectural priority claim.',
        'source_sha256': source_hashes(), 'runtime': runtime_signature()}


def make_episodes(history, seed):
    history = np.asarray(history)
    if (history.ndim != 2 or history.shape[1] < 2 or np.any(history[:, 0])
            or not np.isin(history, [0, 1]).all()):
        raise ValueError('Require binary TRAIN histories and empty PAD')
    history = history.astype(bool)
    excluded_training_rows = np.flatnonzero(history.sum(1) < 2)
    if len(excluded_training_rows) == len(history):
        raise ValueError('No TRAIN user can provide both context and hidden target')
    contexts, targets, excluded = [], [], []
    rng = np.random.default_rng(seed+41001)
    for retention in RETENTIONS:
        for row, values in enumerate(history):
            present = np.flatnonzero(values)
            if len(present) < 2:
                continue
            count = min(len(present)-1, max(1, int(np.floor(retention*len(present)))))
            context = np.zeros(history.shape[1], dtype=bool)
            context[rng.choice(present, count, replace=False)] = True
            contexts.append(context); targets.append(values & ~context); excluded.append(row)
    contexts, targets = np.asarray(contexts), np.asarray(targets)
    eligible = ~contexts
    eligible[:, 0] = False
    return {'contexts': contexts, 'targets': targets, 'eligible': eligible,
            'exclude_rows': np.asarray(excluded, dtype=np.int64),
            'excluded_training_rows': excluded_training_rows.astype(np.int64)}


def prepare_inputs(matrix, seed):
    from exploratory.evidence_transfer.model import prepare_donors, extract_features, fit_normalizer
    history = np.asarray(matrix) != 0
    episodes = make_episodes(history, seed)
    donors = prepare_donors(history)
    features = extract_features(donors, episodes['contexts'], episodes['exclude_rows'])
    query_features = extract_features(donors, history, np.arange(len(history), dtype=np.int64))
    normalizer = fit_normalizer(features, episodes['eligible'])
    hashes = {'raw_train_features_sha256': array_digest(features),
        'raw_query_features_sha256': array_digest(query_features),
        'episode_arrays_sha256': {name: array_digest(value) for name, value in episodes.items()},
        'normalizer_mean_sha256': array_digest(normalizer.mean), 'normalizer_scale_sha256': array_digest(normalizer.scale)}
    normalized = normalizer.transform(features)
    query_normalized = normalizer.transform(query_features)
    hashes.update(normalized_train_features_sha256=array_digest(normalized),
                  normalized_query_features_sha256=array_digest(query_normalized))
    return episodes, normalized, query_normalized, query_features, normalizer, hashes


def score_model(model, features):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            values = model(torch.from_numpy(np.ascontiguousarray(features[start:start+BATCH_SIZE])))
            scores.append(values.detach().cpu().numpy())
    result = np.concatenate(scores).astype(np.float32, copy=False)
    result[:, 0] = 0
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite model predictions')
    return result


def restore_model(path, variant=None):
    from exploratory.evidence_transfer.model import SharedEvidenceScorer
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = dict(state['config'])
    if variant is not None:
        config['variant'] = variant
    model = SharedEvidenceScorer(**config)
    model.load_state_dict(state['state_dict'], strict=True)
    return model


def train_arm(features, episodes, query_features, seed, variant, output, checkpoint_callback):
    """Fixed updates; callback selects exports but cannot change optimization."""
    from exploratory.evidence_transfer.model import SharedEvidenceScorer, masked_listwise_loss
    configure_torch()
    model = SharedEvidenceScorer(n_features=features.shape[-1], width=WIDTH, variant=variant, seed=seed+42001)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    initial = {key: array_digest(value.detach().cpu().numpy()) for key, value in model.state_dict().items()}
    checkpoints, epochs = [], []
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(CHECKPOINTS[-1]+1):
        begin = time.monotonic()
        if epoch:
            model.train()
            order = np.random.default_rng(seed+43001+epoch).permutation(len(features))
            total = 0.
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start:start+BATCH_SIZE]
                scores = model(torch.from_numpy(np.ascontiguousarray(features[indices])))
                loss = masked_listwise_loss(scores, torch.from_numpy(episodes['targets'][indices]),
                                           torch.from_numpy(episodes['eligible'][indices]))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(indices)
            epochs.append({'epoch': epoch, 'mean_online_episode_loss': total/len(features),
                           'seconds': time.monotonic()-begin})
            write_json(output/'epoch-metrics.json', epochs)
        if epoch in CHECKPOINTS:
            path = output/f'epoch-{epoch}.pt'
            torch.save({'config': model.config, 'state_dict': model.state_dict()}, path)
            scores = score_model(model, query_features)
            metric = checkpoint_callback(epoch, scores)
            row = {'epoch': epoch, 'meta_fit_all_observed_ndcg@10': float(metric),
                'scores_array_sha256': array_digest(scores), 'checkpoint_file': path.name,
                'checkpoint_file_sha256': digest(path)}
            checkpoints.append(row)
            write_json(output/'checkpoint-metrics.json', checkpoints)
    selected = max(checkpoints, key=lambda row: row['meta_fit_all_observed_ndcg@10'])
    replay = score_model(restore_model(output/selected['checkpoint_file']), query_features)
    if array_digest(replay) != selected['scores_array_sha256']:
        raise ValueError('Selected checkpoint replay differs')
    return selected, replay, {'initial_state_sha256': initial, 'updates': CHECKPOINTS[-1]*int(np.ceil(len(features)/BATCH_SIZE)),
        'epochs': CHECKPOINTS[-1], 'episodes': len(features), 'parameter_count': sum(p.numel() for p in model.parameters())}


def lock_binary(reference_root, seed, signature, output):
    categorical.verify_barrier(reference_root)
    completed = json.loads((reference_root/'manifest.json').read_text())
    if (completed['status'] != 'complete' or completed['test_read'] or completed['test_evaluated']
            or completed['input_signatures'][str(seed)] != signature
            or completed['selection_freeze_sha256'] != digest(reference_root/'SELECTIONS-FROZEN.json')):
        raise ValueError('Locked binary reference provenance differs')
    directory = reference_root/str(seed)
    selected = json.loads((directory/'selection.json').read_text())['binary_expanded']
    source = directory/selected['scores_file']
    if (digest(source) != selected['scores_file_sha256']
            or completed['output_sha256'].get(source.relative_to(reference_root).as_posix()) != digest(source)):
        raise ValueError('Locked binary prediction signature differs')
    path = output/'locked-binary.npy'
    shutil.copyfile(source, path)
    row = {'source_model': 'binary_expanded', 'penalty': selected['penalty'],
        'candidate_id': selected['candidate_id'], 'scores_file': path.name,
        'scores_file_sha256': digest(path), 'scores_array_sha256': selected['scores_array_sha256'],
        'source_selection_manifest_sha256': digest(directory/'selection-manifest.json')}
    write_json(output/'locked-reference.json', row)
    return row


def select_seed(source, ratings, metadata, cohort_directory, reference_root, output, seed, declared):
    from exploratory.evidence_transfer.model import FEATURE_NAMES
    if declared != protocol(declared['seeds']):
        raise ValueError('Protocol/source/runtime changed before fitting')
    users, items, train, valid, matrix, fit, dev, signature = load_selection_inputs(
        source, ratings, metadata, cohort_directory, seed)
    meta_pairs = [(user, item) for user, item in valid if user in fit]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'input-signature.json', signature)
    write_json(output/'cohorts.json', {'meta_fit': sorted(fit), 'development': sorted(dev)})
    np.savez_compressed(output/'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    for name in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(source/name, output/name)
    lock_binary(reference_root, seed, signature, output)
    begin = time.monotonic()
    episodes, features, queries, raw_queries, normalizer, feature_hashes = prepare_inputs(matrix, seed)
    np.savez_compressed(output/'episodes.npz', **episodes)
    np.savez(output/'normalizer.npz', mean=normalizer.mean, scale=normalizer.scale)
    write_json(output/'feature-hashes.json', {**feature_hashes, 'feature_names': list(FEATURE_NAMES),
        'training_shape': list(features.shape), 'query_shape': list(queries.shape),
        'preparation_seconds': time.monotonic()-begin,
        'excluded_training_user_count': len(episodes['excluded_training_rows'])})
    (output/'selected-scores').mkdir()
    choices, audits = {}, {}
    for variant in ARMS:
        def on_checkpoint(epoch, scores):
            value = meta_ndcg(scores, users, items, train, meta_pairs, fit)
            print(json.dumps({'seed': seed, 'arm': variant, 'epoch': epoch, 'meta_ndcg': value}), flush=True)
            return value
        try:
            selected, scores, audit = train_arm(features, episodes, queries, seed, variant,
                output/'checkpoints'/variant, on_checkpoint)
        except Exception as error:
            write_json(output/'training-failure.json', {'arm': variant, 'error_type': type(error).__name__,
                'error': str(error), 'test_read': False, 'development_evaluated': False})
            raise
        audits[variant] = audit
        choices[variant] = {**selected, 'checkpoint_file': f'checkpoints/{variant}/'+selected['checkpoint_file'],
            'selection_cohort': 'reused_meta_fit', 'candidate_count': len(CHECKPOINTS), 'refit_replay_exact': True}
        save_scores(output, variant, scores, choices[variant])
    if (any(audits[arm] != audits[ARMS[0]] for arm in ARMS[1:])):
        raise ValueError('Controlled arms differ in initialization, allocated parameters or training budget')
    analytic = raw_queries[:, :, list(FEATURE_NAMES).index('weighted_support_fraction')].copy()
    analytic[:, 0] = 0
    choices['analytic_donor'] = {'selection_cohort': 'none_untrained', 'candidate_count': 1,
                                'feature': 'weighted_support_fraction'}
    save_scores(output, 'analytic_donor', analytic, choices['analytic_donor'])
    checkpoint = output/choices['full_pattern']['checkpoint_file']
    intervention = score_model(restore_model(checkpoint, variant='no_pattern'), queries)
    choices['full_zero_pattern'] = {'selection_cohort': 'none_fixed_intervention', 'candidate_count': 1,
        'source_arm': 'full_pattern', 'epoch': choices['full_pattern']['epoch'],
        'checkpoint_file': choices['full_pattern']['checkpoint_file'], 'checkpoint_file_sha256': digest(checkpoint),
        'interpretation': 'Selected full_pattern weights; pattern inputs zeroed after normalization; distribution-shift diagnostic.'}
    save_scores(output, 'full_zero_pattern', intervention, choices['full_zero_pattern'])
    write_json(output/'training-audit.json', audits)
    write_json(output/'selection.json', choices)
    manifest = {'status': 'selected', 'seed': seed, 'test_read': False, 'development_evaluated': False,
        'source_sha256': source_hashes(), 'runtime': runtime_signature(), 'input_signature': signature,
        'arms_trained': len(ARMS), 'checkpoints_per_arm': len(CHECKPOINTS),
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output/'selection-manifest.json', manifest)
    return manifest


def save_scores(directory, name, scores, row):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any(scores[:, 0]):
        raise ValueError('Invalid full-catalog predictions')
    path = directory/f'selected-scores/{name}.npy'
    np.save(path, scores, allow_pickle=False)
    row.update(scores_file=path.relative_to(directory).as_posix(), scores_file_sha256=digest(path),
               scores_array_sha256=array_digest(scores), score_file_replay_exact=True)


def check_scores(directory, selected, shape):
    path = directory/selected['scores_file']
    if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != selected['scores_file_sha256']:
        raise ValueError('Score file signature differs')
    scores = np.load(path, allow_pickle=False)
    if scores.shape != shape or not np.isfinite(scores).all() or np.any(scores[:, 0]) or array_digest(scores) != selected['scores_array_sha256']:
        raise ValueError('Score array differs')
    return scores


def validate_choices(directory, declared):
    selected = json.loads((directory/'selection.json').read_text())
    if set(selected) != set(MODEL_NAMES):
        raise ValueError('Selected model set differs')
    with np.load(directory/'training-categories.npz', allow_pickle=False) as archive:
        shape = archive['ratings'].shape
    for arm in ARMS:
        grid = json.loads((directory/'checkpoints'/arm/'checkpoint-metrics.json').read_text())
        if ([row['epoch'] for row in grid] != declared['checkpoints']
                or any(not np.isfinite(row['meta_fit_all_observed_ndcg@10'])
                       or not 0 <= row['meta_fit_all_observed_ndcg@10'] <= 1 for row in grid)):
            raise ValueError('Incomplete checkpoint grid')
        best = max(grid, key=lambda row: row['meta_fit_all_observed_ndcg@10'])
        row = selected[arm]
        expected = {**best, 'checkpoint_file': f'checkpoints/{arm}/'+best['checkpoint_file']}
        if (any(row.get(key) != value for key, value in expected.items())
                or row['candidate_count'] != len(declared['checkpoints']) or row['refit_replay_exact'] is not True
                or row['selection_cohort'] != 'reused_meta_fit'):
            raise ValueError('Selected checkpoint is not the first exact maximum')
        for candidate in grid:
            if digest(directory/'checkpoints'/arm/candidate['checkpoint_file']) != candidate['checkpoint_file_sha256']:
                raise ValueError('Checkpoint changed')
    zero = selected['full_zero_pattern']
    if (zero['source_arm'] != 'full_pattern' or zero['epoch'] != selected['full_pattern']['epoch']
            or zero['checkpoint_file'] != selected['full_pattern']['checkpoint_file']
            or zero['checkpoint_file_sha256'] != selected['full_pattern']['checkpoint_file_sha256']
            or zero['selection_cohort'] != 'none_fixed_intervention'
            or selected['analytic_donor']['selection_cohort'] != 'none_untrained'):
        raise ValueError('Fixed controls changed')
    for name, row in selected.items():
        if row['scores_file'] != f'selected-scores/{name}.npy' or row['score_file_replay_exact'] is not True:
            raise ValueError('Selected score path/replay differs')
        check_scores(directory, row, shape)
    check_scores(directory, json.loads((directory/'locked-reference.json').read_text()), shape)


def checked_manifests(research, declared):
    hashes = {}
    for seed in declared['seeds']:
        directory = research/str(seed)
        manifest = json.loads((directory/'selection-manifest.json').read_text())
        if (manifest['status'] != 'selected' or manifest['seed'] != seed or manifest['test_read']
                or manifest['development_evaluated'] or manifest['source_sha256'] != source_hashes()
                or manifest['runtime'] != runtime_signature() or manifest['arms_trained'] != len(ARMS)
                or manifest['checkpoints_per_arm'] != len(CHECKPOINTS)):
            raise ValueError('Invalid selection manifest')
        verify_files(directory, manifest['output_sha256'])
        validate_choices(directory, declared)
        hashes[str(seed)] = digest(directory/'selection-manifest.json')
    return hashes


def freeze_selections(research):
    declared = json.loads((research/'protocol.json').read_text())
    if declared != protocol(declared['seeds']):
        raise ValueError('Protocol changed')
    hashes = checked_manifests(research, declared)
    frozen = {'status': 'all_selections_frozen', 'seeds': declared['seeds'], 'test_read': False,
        'development_evaluated': False, 'protocol_sha256': digest(research/'protocol.json'),
        'source_sha256': source_hashes(), 'runtime': runtime_signature(),
        'reference_input_signature_sha256': digest(research/'reference-input-signature.json'),
        'selection_manifest_sha256': hashes}
    with (research/'SELECTIONS-FROZEN.json').open('x') as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return frozen


def verify_barrier(research):
    frozen = json.loads((research/'SELECTIONS-FROZEN.json').read_text())
    declared = json.loads((research/'protocol.json').read_text())
    if (declared != protocol(declared['seeds']) or frozen['status'] != 'all_selections_frozen'
            or frozen['test_read'] or frozen['development_evaluated'] or frozen['seeds'] != declared['seeds']
            or frozen['protocol_sha256'] != digest(research/'protocol.json')
            or frozen['reference_input_signature_sha256'] != digest(research/'reference-input-signature.json')
            or frozen['source_sha256'] != source_hashes() or frozen['runtime'] != runtime_signature()
            or frozen['selection_manifest_sha256'] != checked_manifests(research, declared)):
        raise ValueError('Global selection barrier differs')
    return frozen


def evaluate_selected(research, source_root, ratings, metadata):
    frozen = verify_barrier(research)
    with (research/'DEVELOPMENT-OPENED.json').open('x') as stream:
        json.dump({'status': 'started', 'test_read': False, 'fresh_confirmation': False,
                   'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json')}, stream)
    result = {'schema_version': 1, 'stage': 'reused_development', 'study_kind': 'post_final_test_exploratory',
        'test_read': False, 'dataset_test_previously_evaluated': True, 'fresh_confirmation': False, 'seeds': {}}
    signatures = {}
    for seed in frozen['seeds']:
        directory = research/str(seed)
        loaded = load_source(source_root/f'{seed}-EASE-1', ratings, metadata)
        users, items, train, valid, training, labels = loaded[:6]
        matrix = categorical_matrix(users, items, training)
        signature = json.loads((directory/'input-signature.json').read_text()); signatures[str(seed)] = signature
        if (identity_digest(users, items) != signature['ordered_identity_sha256']
                or array_digest(matrix) != signature['training_categories_sha256']
                or loaded[-1]['data_sha256'] != signature['data_sha256']
                or any(loaded[-1]['split_sha256'][p] != signature['split_sha256'][p] for p in ('train', 'valid'))):
            raise ValueError('Development source differs from TRAIN selection inputs')
        fit, dev = partition_users(users, seed)
        if json.loads((directory/'cohorts.json').read_text()) != {'meta_fit': sorted(fit), 'development': sorted(dev)}:
            raise ValueError('Development cohort differs')
        history, counts = grouped(train), Counter(item for _, item in train)
        selections = json.loads((directory/'selection.json').read_text())
        reference = json.loads((directory/'locked-reference.json').read_text())
        genres = {item: [str(column) for column in np.flatnonzero(loaded[6][i])] for i, item in enumerate(items) if i}
        all_truth = {u: values for u, values in grouped(valid).items() if u in dev}
        liked = grouped([(u, i) for (u, i), rating in labels.items() if u in dev and rating >= 4])
        details, aggregates = {}, {}
        for name, selected in {**selections, 'locked_binary': reference}.items():
            recs = recommendations(check_scores(directory, selected, matrix.shape), users, items, matrix != 0, 10)
            metric = evaluate_ratings(recs, dev, labels, history, items[1:], counts, 10)
            details[name] = metric
            aggregates[name] = aggregate_metrics(metric, recs, dev, counts, items[1:])
            for field, truth in (('groups', all_truth), ('liked_groups', liked)):
                aggregates[name][field] = analyze_groups({u: recs[u] for u in truth}, truth,
                    history, items[1:], counts, genres, group_users=users) if truth else None
        write_json(directory/'development-metrics.json', details)
        contrasts = [('full_pattern', 'no_pattern'), ('full_pattern', 'marginal_only'),
            ('full_pattern', 'analytic_donor'), ('full_pattern', 'locked_binary'),
            ('no_pattern', 'locked_binary'), ('full_zero_pattern', 'full_pattern')]
        result['seeds'][str(seed)] = {'models': {name: aggregates[name] for name in MODEL_NAMES},
            'references': {'locked_binary': aggregates['locked_binary']},
            'selections': {name: {key: value for key, value in row.items() if key not in ('checkpoint_file', 'scores_file')}
                for name, row in selections.items()},
            'reference_selections': {'locked_binary': {key: value for key, value in reference.items() if key != 'scores_file'}},
            'comparison': {f'{a}_minus_{b}:{endpoint}': paired_difference(details[a], details[b], endpoint)
                for a, b in contrasts for endpoint in ('all_observed', 'liked_ratings')},
            'cohorts': {'meta_fit': len(fit), 'development': len(dev)}}
    verify_barrier(research)
    write_json(research/'DEVELOPMENT-OPENED.json', {'status': 'complete', 'test_read': False, 'fresh_confirmation': False,
        'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json')})
    write_json(research/'aggregates.json', result)
    write_json(research/'manifest.json', {'status': 'complete', 'test_read': False, 'test_evaluated': False,
        'fresh_confirmation': False, 'source_sha256': source_hashes(), 'runtime': runtime_signature(),
        'selection_freeze_sha256': digest(research/'SELECTIONS-FROZEN.json'), 'input_signatures': signatures,
        'reference_input_signature': json.loads((research/'reference-input-signature.json').read_text()),
        'output_sha256': {p.relative_to(research).as_posix(): digest(p) for p in sorted(research.rglob('*')) if p.is_file()}})
    return result


def curate(research, output):
    from package_project import check_aggregate_only
    manifest = json.loads((research/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['test_read'] or manifest['test_evaluated']:
        raise ValueError('Require completed development study')
    verify_barrier(research)
    verify_files(research, manifest['output_sha256'])
    data = json.loads((research/'aggregates.json').read_text()); check_aggregate_only(data)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('aggregates.json', 'protocol.json'):
        shutil.copyfile(research/name, output/name)
    write_json(output/'provenance.json', {'manifest_sha256': digest(research/'manifest.json'),
        **{key: manifest[key] for key in ('selection_freeze_sha256', 'source_sha256', 'runtime', 'input_signatures', 'reference_input_signature')}})
    lines = ['# Shared evidence interpreter: reused-development exploration', '',
        'The original MovieLens100K TEST was previously examined. No TEST is opened here. '
        'Every seed was sealed before new development metrics. This is not fresh confirmation.', '',
        '| Seed | Model | All nDCG@10 | Liked nDCG@10 | Selected epoch |', '|---|---|---:|---:|---:|']
    means = {name: [] for name in (*MODEL_NAMES, 'locked_binary')}
    for seed, row in data['seeds'].items():
        for name, metric in {**row['models'], **row['references']}.items():
            liked = metric['liked_ratings']['ndcg@10'] if metric['liked_ratings'] else None
            means[name].append((metric['all_observed']['ndcg@10'], liked))
            liked_text = f'{liked:.6f}' if liked is not None else 'n/a'
            lines.append(f"| {seed} | {name} | {metric['all_observed']['ndcg@10']:.6f} | {liked_text} | {row['selections'].get(name, {}).get('epoch', 'fixed')} |")
    lines += ['', 'Equal-seed means are descriptive; overlapping reused splits are not independent datasets.', '',
              '| Model | Mean all nDCG@10 | Mean liked nDCG@10 |', '|---|---:|---:|']
    for name, rows in means.items():
        liked = [row[1] for row in rows if row[1] is not None]
        lines.append(f'| {name} | {np.mean([row[0] for row in rows]):.6f} | '+(f'{np.mean(liked):.6f}' if liked else 'n/a')+' |')
    lines += ['', 'All three trained controls have equal allocated capacity, initialization, episodes, updates and '
        'checkpoint opportunities. The full_zero_pattern row is an inference feature intervention using the '
        'selected full_pattern weights, with no retraining; it changes the input distribution and is distinct '
        'from no_pattern. The analytic donor control is unfitted. No model receives query-user donor evidence. '
        'Aggregate JSON includes explicit cohorts, TRAIN-defined user/item groups and paired descriptive differences. '
        'No population significance or novelty-priority claim follows from this experiment.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    write_json(output/'SHA256.json', {p.name: digest(p) for p in sorted(output.iterdir())})


def benchmark(source, ratings, output, steps=8):
    from exploratory.evidence_transfer.model import SharedEvidenceScorer, masked_listwise_loss
    configure_torch()
    before = source_hashes()
    _, _, _, matrix, manifest = load_train(source, ratings)
    begin = time.monotonic()
    episodes, features, queries, _, _, hashes = prepare_inputs(matrix, 2026)
    preparation_seconds = time.monotonic()-begin
    model = SharedEvidenceScorer(n_features=features.shape[-1], width=WIDTH, variant='full_pattern', seed=44027)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    indices = np.arange(min(BATCH_SIZE, len(features)))
    begin = time.monotonic()
    for _ in range(steps):
        score = model(torch.from_numpy(np.ascontiguousarray(features[indices])))
        loss = masked_listwise_loss(score, torch.from_numpy(episodes['targets'][indices]), torch.from_numpy(episodes['eligible'][indices]))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    step_seconds = (time.monotonic()-begin)/steps
    begin = time.monotonic(); score_model(model, queries); inference_seconds = time.monotonic()-begin
    if source_hashes() != before:
        raise ValueError('Benchmark source changed')
    result = {'status': 'complete', 'validation_read': False, 'test_read': False, 'development_evaluated': False,
        'source_sha256': before, 'runtime': runtime_signature(), 'source_manifest_sha256': digest(source/'manifest.json'),
        'data_sha256': manifest['data_sha256'], 'preparation_seconds': preparation_seconds,
        'train_feature_shape': list(features.shape), 'query_feature_shape': list(queries.shape),
        'cache_bytes': int(features.nbytes+queries.nbytes), 'benchmark_steps': steps,
        'mean_batch_update_seconds': step_seconds, 'full_query_inference_seconds': inference_seconds,
        'projected_training_seconds_per_seed': step_seconds*int(np.ceil(len(features)/BATCH_SIZE))*CHECKPOINTS[-1]*len(ARMS),
        'excluded_training_user_count': len(episodes['excluded_training_rows']),
        'projection_caution': 'Approximation from repeated updates of one full batch; excludes I/O, feature construction and checkpoint scoring.',
        'feature_hashes': hashes}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return result


def run(args):
    declared = protocol(args.seeds)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', declared)
    pinned = {'manifest_sha256': digest(args.reference_run/'manifest.json'),
              'selection_freeze_sha256': digest(args.reference_run/'SELECTIONS-FROZEN.json')}
    write_json(args.out/'reference-input-signature.json', pinned)
    for seed in args.seeds:
        select_seed(args.source_root/f'{seed}-EASE-1', args.ratings, args.item_metadata,
            args.cohort_root/str(seed), args.reference_run, args.out/str(seed), seed, declared)
    if pinned != {'manifest_sha256': digest(args.reference_run/'manifest.json'),
                  'selection_freeze_sha256': digest(args.reference_run/'SELECTIONS-FROZEN.json')}:
        raise ValueError('Locked reference changed during fitting')
    freeze_selections(args.out)
    evaluate_selected(args.out, args.source_root, args.ratings, args.item_metadata)
    curate(args.out, args.evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    timing = commands.add_parser('benchmark')
    timing.add_argument('--source', type=Path, required=True)
    timing.add_argument('--ratings', type=Path, required=True)
    timing.add_argument('--out', type=Path, required=True)
    running = commands.add_parser('run')
    running.add_argument('--source-root', type=Path, default=Path('runs/research-v2'))
    running.add_argument('--cohort-root', type=Path, default=Path('runs/adaptive-v1'))
    running.add_argument('--reference-run', type=Path, default=Path('runs/categorical-reconstruction-v1'))
    running.add_argument('--ratings', type=Path, required=True)
    running.add_argument('--item-metadata', type=Path, required=True)
    running.add_argument('--out', type=Path, default=Path('runs/evidence-transfer-v1'))
    running.add_argument('--evidence', type=Path, default=Path('exploratory/evidence_transfer/results-v1'))
    running.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
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
            parser.error('Detailed outputs must remain under runs/')
        run(args)


if __name__ == '__main__':
    main()
