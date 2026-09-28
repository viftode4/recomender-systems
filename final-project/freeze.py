"""Freeze validation-selected study models without opening held-out labels.

The bundle is portable: it contains train/validation interactions and all-user,
full-catalog raw predictions. EASE, ItemKNN, BPR and ExactPop do not condition
their saved predictions on evaluation phase. Random is regenerated with the
original internal user IDs and a distinct test-phase seed. No model is refit.

Expert z-score means/scales are frozen over validation-eligible items (training
history excluded). Test masking additionally excludes validation history, but
does not recompute these transforms. This is a deliberate fixed-transform
protocol; it is not equivalent to refitting normalizers over test candidates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from study import (build_features, digest, grouped, load_genres, load_runs,
                   ranking, rerank, rrf, write_json)

ROOT = Path(__file__).resolve().parent
SUPPORTED = {'Random', 'ExactPop', 'EASE', 'ItemKNN', 'UserKNN', 'BPR',
             'SLIMElastic', 'FISMCorrected', 'GenreContent', 'LightGCN', 'NeuMF', 'NGCF',
             'ContrastTransfer', 'PositiveEASE', 'SignedEASE', 'SignedChannelsLinear',
             'PairContrastGate', 'PairContrastRandomPartners', 'PairAnchorKernel', 'PairSignedKernel'}
CODE_FILES = ('freeze.py', 'final_evaluate.py', 'study.py', 'metrics.py', 'hybrid_constraints.py', 'societal.py')


def read_json(path):
    return json.loads(path.read_text())


def feature_names(experts, variant):
    names = list(experts)
    if variant in ('user', 'context'):
        names += [f'{n}*activity' for n in experts]
        names += [f'{n}*genre_entropy' for n in experts]
    if variant in ('item', 'context'):
        names += [f'{n}*item_popularity' for n in experts]
    if variant in ('disagreement', 'context'):
        names += ['expert_disagreement']
    return names


def standardize(values, reference=None):
    reference = values if reference is None else reference
    center, scale = float(reference.mean()), max(float(reference.std()), 1e-8)
    return (values - center) / scale, {'mean': center, 'scale': scale}


def model_specs(expert_names, selection, coefficients, switching, group_policy,
                modes=('diversity', 'calibration', 'exposure')):
    specs = {name: {'kind': 'expert', 'expert': name} for name in expert_names}
    for label in selection['families'].values():
        variant = label.split('-')[0]
        if variant in ('constrained', 'calibrated'):
            variant = 'static'
        if variant not in ('static', 'user', 'item', 'disagreement', 'context'):
            raise ValueError(f'Unsupported feature variant: {label}')
        specs[label] = {'kind': 'linear', 'variant': variant,
                        'coefficients': coefficients[label]}
    specs['rrf'] = {'kind': 'rrf', 'constant': 60}
    specs['group-switch'] = {'kind': 'switch', 'experts': switching}
    bases = list(dict.fromkeys([selection['best_expert'], selection['families']['context']]))
    for base in bases:
        for mode in modes:
            for strength in (.2, .5, .8):
                specs[f'{base}/{mode}-{strength}'] = {
                    'kind': 'rerank', 'base': base, 'mode': mode,
                    'strength': strength, 'pool_size': 100}
        for pool in ('adaptive', 'full'):
            specs[f'{base}/{pool}-pool-exposure'] = {
                'kind': 'rerank', 'base': base, 'mode': 'exposure',
                'strength': 1., 'pool_size': pool, 'minimum_pool_size': 100}
    for mode in modes:
        specs[f'rerank-experts-then-rrf/{mode}'] = {
            'kind': 'rerank_then_rrf', 'mode': mode, 'strength': .5,
            'pool_size': 100, 'constant': 60}
        specs[f'rrf-then-rerank/{mode}'] = {
            'kind': 'rerank', 'base': 'rrf', 'mode': mode,
            'strength': .5, 'pool_size': 100}
    specs['group-utility-budget-exposure'] = {
        'kind': 'rerank', 'base': selection['families']['context'],
        'mode': 'exposure', 'group_strengths': group_policy['strengths'],
        'pool_size': 100}
    return specs


def score_models(bundle, arrays, phase):
    """Apply the frozen score function; phase only changes masks and Random."""
    if phase not in ('valid', 'test'):
        raise ValueError('Unknown evaluation phase')
    users, items = arrays['users'].tolist(), arrays['items'].tolist()
    mask = arrays['train_mask'].copy()
    if phase == 'test':
        mask |= arrays['valid_mask']
    mask[:, 0] = True
    candidates = [np.flatnonzero(~row) for row in mask]
    if any(len(row) < bundle['k'] for row in candidates):
        raise ValueError('Insufficient unseen candidates')
    models = {}
    for index, name in enumerate(bundle['expert_order']):
        source = bundle['sources'][name]['manifest']
        if source['model'] == 'Random' and phase == 'test':
            seed = int(source['settings']['seed'])
            scores = np.stack([np.random.default_rng(np.random.SeedSequence(
                [seed, int(uid), 1])).random(len(items)) for uid in arrays['internal_user_ids']])
        else:
            scores = arrays['raw_scores'][index]
        models[name] = scores
    active = bundle['active_experts']
    base = np.stack([(models[name] - arrays['score_mean'][index, :, None]) /
                     arrays['score_scale'][index, :, None]
                     for index, name in enumerate(active)], axis=2)
    for name, spec in bundle['models'].items():
        if spec['kind'] != 'linear':
            continue
        columns = feature_names(active, spec['variant'])
        coefficients = spec['coefficients']
        if set(columns) != set(coefficients['weights']):
            raise ValueError(f'Coefficient feature mismatch: {name}')
        weights = np.asarray([coefficients['weights'][column] for column in columns])
        features = build_features(base, arrays['activity'], arrays['entropy'],
                                  arrays['popularity'], spec['variant'])
        models[name] = features @ weights + coefficients['intercept']
    models['rrf'] = rrf({name: models[name] for name in active}, candidates,
                        bundle['models']['rrf']['constant'])
    if any(not np.isfinite(scores).all() for scores in models.values()):
        raise ValueError('Non-finite frozen prediction')
    return models, candidates


def recommend(bundle, arrays, phase, user_indices=None):
    """Generate all frozen comparisons, with no relevance labels as input."""
    models, candidates = score_models(bundle, arrays, phase)
    cohort = range(len(arrays['users'])) if user_indices is None else user_indices
    k, binary, fractional = bundle['k'], arrays['genre_binary'], arrays['genre_fraction']
    profiles, head, groups = arrays['profiles'], arrays['head'], arrays['groups']
    result = {}
    for name, spec in bundle['models'].items():
        recommendations = {}
        for u in cohort:
            eligible = candidates[u]
            kind = spec['kind']
            if kind in ('expert', 'linear', 'rrf'):
                selected = ranking(models[name][u], eligible, k)
            elif kind == 'switch':
                selected = ranking(models[spec['experts'][str(groups[u])]][u], eligible, k)
            elif kind == 'rerank':
                scores = models[spec['base']][u]
                pool = spec['pool_size']
                if pool == 'full':
                    pool = len(eligible)
                elif pool == 'adaptive':
                    order = ranking(scores, eligible, len(eligible))
                    target_head = int(round(float(head[1:].mean()) * k))
                    requirements = [(head, target_head), (~head, k - target_head)]
                    minimum = 0
                    for membership, target in requirements:
                        positions = np.flatnonzero(membership[order])
                        needed = (0 if not target else int(positions[target-1] + 1)
                                  if len(positions) >= target else len(order))
                        minimum = max(minimum, needed)
                    pool = min(len(order), max(spec['minimum_pool_size'], minimum))
                strength = (spec['group_strengths'][str(groups[u])]
                            if 'group_strengths' in spec else spec['strength'])
                profile = arrays['popularity_profiles'][u] if spec['mode'] == 'popularity_calibration' else profiles[u]
                selected = rerank(scores, eligible, binary, fractional, profile,
                                  head, k, strength, spec['mode'], pool)
            elif kind == 'rerank_then_rrf':
                fused = np.zeros(len(head))
                for expert in bundle['active_experts']:
                    profile = arrays['popularity_profiles'][u] if spec['mode'] == 'popularity_calibration' else profiles[u]
                    promoted = rerank(models[expert][u], eligible, binary, fractional,
                                      profile, head, k, spec['strength'], spec['mode'], spec['pool_size'])
                    original = ranking(models[expert][u], eligible, len(eligible))
                    used = set(promoted)
                    order = np.concatenate([promoted, [i for i in original if i not in used]])
                    fused[order] += 1 / (spec['constant'] + np.arange(1, len(order) + 1))
                selected = ranking(fused, eligible, k)
            else:
                raise ValueError(f'Unknown frozen model kind: {kind}')
            recommendations[str(arrays['users'][u])] = [str(arrays['items'][i]) for i in selected]
        result[name] = recommendations
    return result


def freeze_study(study_path, runs, items_path, out):
    """Build a bundle after exact validation-ranking replay; never reads test.tsv."""
    study_path, items_path, out = Path(study_path), Path(items_path), Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    manifest = read_json(study_path / 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('test_read') is not False:
        raise ValueError('Require a complete validation-only study')
    for key, filename in [('study_source_sha256', 'study.py'), ('metrics_source_sha256', 'metrics.py')]:
        if manifest[key] != digest(ROOT / filename):
            raise ValueError(f'Study source changed: {filename}; reproduce the study before freezing')
    runs = [Path(path) for path in runs]
    users, items, experts, train, valid, sources = load_runs(runs)
    if sources != manifest['sources']:
        raise ValueError('Run sources differ from study sources')
    if digest(items_path) != manifest['metadata_sha256']:
        raise ValueError('Metadata differs from study metadata')
    for name, source in sources.items():
        if source['manifest']['model'] not in SUPPORTED:
            raise ValueError(f'Unsupported phase-independent model: {name}')
    selection = read_json(study_path / 'selection.json')
    coefficients = read_json(study_path / 'coefficients.json')
    cohorts = read_json(study_path / 'cohorts.json')
    switching = read_json(study_path / 'switching.json')
    group_policy = read_json(study_path / 'group-policy.json')
    recorded = read_json(study_path / 'recommendations.json')
    modes = ['diversity', 'calibration', 'exposure']
    if f"{selection['best_expert']}/popularity_calibration-0.2" in recorded:
        modes.append('popularity_calibration')
    if selection['metric'] != f"ndcg@{manifest['k']}":
        raise ValueError('Selection metric differs from study k')
    history = grouped(train)
    index = {item: i for i, item in enumerate(items)}
    user_index = {user: i for i, user in enumerate(users)}
    train_mask = np.zeros((len(users), len(items)), dtype=bool)
    valid_mask = np.zeros_like(train_mask)
    for pairs, mask in ((train, train_mask), (valid, valid_mask)):
        for user, item in pairs:
            if user not in user_index:
                raise ValueError('Frozen catalog requires every training user to have validation scores')
            mask[user_index[user], index[item]] = True
    eligible = [np.asarray([i for i, item in enumerate(items) if i and item not in history[user]]) for user in users]
    binary, fractional = load_genres(items_path, items)
    if any(not history[user] for user in users):
        raise ValueError('Training profile is empty')
    profiles = np.asarray([fractional[[index[item] for item in history[user]]].mean(axis=0) for user in users])
    counts = Counter(item for _, item in train)
    raw_activity = np.asarray([len(history[user]) for user in users])
    activity, activity_stats = standardize(np.log1p(raw_activity))
    raw_entropy = -np.sum(profiles * np.log2(np.maximum(profiles, 1e-15)), axis=1)
    entropy, entropy_stats = standardize(raw_entropy)
    raw_popularity = np.log1p([counts.get(item, 0) for item in items])
    popularity, popularity_stats = standardize(raw_popularity, raw_popularity[1:])
    head = np.zeros(len(items), dtype=bool)
    head[np.argsort(-np.asarray([counts.get(item, 0) for item in items[1:]]), kind='stable')[:int(np.ceil(.2 * (len(items) - 1)))]+1] = True
    thresholds = np.quantile(raw_activity, [1/3, 2/3])
    if not np.array_equal(thresholds, cohorts['activity_thresholds']):
        raise ValueError('Activity groups differ from study')
    groups = np.searchsorted(thresholds, raw_activity, side='right')
    active = [name for name in experts if sources[name]['manifest']['model'] != 'Random']
    score_mean = np.asarray([[experts[name][u, eligible[u]].mean() for u in range(len(users))] for name in active])
    score_scale = np.asarray([[max(experts[name][u, eligible[u]].std(), 1e-8) for u in range(len(users))] for name in active])
    ids = read_json(runs[0] / 'ids.json')
    if ids.get('padding_index') != 0 or ids['items'] != items or len(set(ids['users'])) != len(ids['users']):
        raise ValueError('Invalid source ID mapping')
    original_ids = {user: uid for uid, user in enumerate(ids['users'])}
    if any(user not in original_ids or original_ids[user] == 0 for user in users):
        raise ValueError('Missing original non-padding user ID')
    source_artifacts = {}
    for run in runs:
        if read_json(run / 'ids.json') != ids:
            raise ValueError('Experts have different internal ID mappings')
        source_manifest = sources[run.name]['manifest']
        if 'checkpoint' in source_manifest:
            checkpoint = (run / source_manifest['checkpoint']).resolve()
            if not checkpoint.is_relative_to(run.resolve()) or digest(checkpoint) != source_manifest.get('checkpoint_sha256'):
                raise ValueError(f'Checkpoint differs from source manifest: {run.name}')
        if 'content_model_sha256' in source_manifest:
            if digest(run / 'content-model.npz') != source_manifest['content_model_sha256']:
                raise ValueError(f'Content model differs from source manifest: {run.name}')
        files = [run / name for name in ('manifest.json', 'train.tsv', 'valid.tsv', 'valid-scores.npz', 'ids.json')]
        files += sorted((run / 'checkpoints').glob('*.pth'))
        if 'checkpoint' in source_manifest and checkpoint not in [path.resolve() for path in files]:
            files.append(checkpoint)
        if (run / 'content-model.npz').exists():
            files.append(run / 'content-model.npz')
        source_artifacts[run.name] = {str(path.relative_to(run)): digest(path) for path in files}
    bundle = {
        'format_version': 1, 'status': 'frozen', 'k': int(manifest['k']),
        'seed': int(manifest['seed']), 'test_read': False,
        'selection': selection, 'sources': sources, 'source_artifacts_sha256': source_artifacts,
        'expert_order': list(experts), 'active_experts': active,
        'models': model_specs(experts, selection, coefficients, switching, group_policy, modes),
        'standardizers': {'activity': activity_stats, 'entropy': entropy_stats, 'popularity': popularity_stats},
        'activity_thresholds': thresholds.tolist(),
        'policy_calibration': {
            'independent': group_policy.get('independent_calibration') is not None,
            'users': len(cohorts.get('calibration', [])),
            'retention_target': group_policy['fit_retention_floor'],
            'note': group_policy.get('note', '')},
        'expected_test_sha256': next(iter(sources.values()))['manifest']['split_sha256']['test'],
        'code_sha256': {name: digest(ROOT / name) for name in CODE_FILES},
        'protocol': {
            'relevance': 'All observed held-out interactions are positives, irrespective of rating.',
            'score_normalization': 'Per-user expert means/scales frozen on validation-eligible catalog; no test renormalization.',
            'features': 'Activity, genre profile/entropy, popularity and item groups use training interactions only.',
            'history_mask': 'Validation excludes train; test excludes train plus validation.',
            'raw_scores': 'Saved train-fitted full-catalog predictions reused for supported phase-independent models.',
            'random': 'Test uses SeedSequence([source seed, original internal user ID, 1]).',
            'test_scope': 'Evaluate all users with test labels; reject users absent from the frozen score archive.',
            'candidate_expansion': 'Adaptive prefix enforces both target head-count and tail-count feasibility; no global fairness guarantee.',
            'interpretation': 'Validation selected models and settings; repeated seed splits share users and are not independent populations.'},
    }
    arrays = dict(users=np.asarray(users), items=np.asarray(items), raw_scores=np.stack(list(experts.values())),
                  internal_user_ids=np.asarray([original_ids[user] for user in users]),
                  train_mask=train_mask, valid_mask=valid_mask, score_mean=score_mean, score_scale=score_scale,
                  activity=activity, entropy=entropy, popularity=popularity, raw_activity=raw_activity,
                  genre_binary=binary, genre_fraction=fractional, profiles=profiles, head=head, groups=groups,
                  popularity_profiles=np.asarray([[1 - head[[index[item] for item in history[user]]].mean(),
                                                    head[[index[item] for item in history[user]]].mean()] for user in users]),
                  training_counts=np.asarray([counts.get(item, 0) for item in items]))
    if set(cohorts['development']) & set(cohorts.get('calibration', [])):
        raise ValueError('Development and policy calibration users overlap')
    non_fit = set(cohorts['development']) | set(cohorts.get('calibration', []))
    if set(cohorts['meta_fit']) & non_fit or set(cohorts['meta_fit']) | non_fit != set(users):
        raise ValueError('Invalid source cohorts')
    development = [u for u, user in enumerate(users) if user in set(cohorts['development'])]
    replay = recommend(bundle, arrays, 'valid', development)
    for name, recs in replay.items():
        if recs != recorded.get(name):
            raise ValueError(f'Validation replay differs for {name}; do not evaluate test')
    bundle['validation_replay'] = {'exact_rankings': True, 'models': len(replay), 'users': len(development)}
    # Create output only after every validation passes. Exclusive creation also
    # prevents accidental replacement of a bundle after inspecting test results.
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'frozen.npz', **arrays)
    for filename in ('train.tsv', 'valid.tsv'):
        shutil.copyfile(runs[0] / filename, out / filename)
    provenance = out / 'provenance'
    provenance.mkdir()
    for filename in ('manifest.json', 'selection.json', 'coefficients.json', 'cohorts.json', 'switching.json', 'group-policy.json'):
        shutil.copyfile(study_path / filename, provenance / filename)
    bundle['artifacts_sha256'] = {str(path.relative_to(out)): digest(path)
                                  for path in sorted(out.rglob('*')) if path.is_file()}
    write_json(out / 'freeze.json', bundle)
    (out / 'freeze.sha256').write_text(digest(out / 'freeze.json') + '\n')
    return bundle


def load_frozen(path):
    """Verify bundle and inference source before exposing any held-out labels."""
    path = Path(path)
    if digest(path / 'freeze.json') != (path / 'freeze.sha256').read_text().strip():
        raise ValueError('Frozen manifest hash mismatch')
    bundle = read_json(path / 'freeze.json')
    if bundle.get('format_version') != 1 or bundle.get('status') != 'frozen' or bundle.get('test_read') is not False:
        raise ValueError('Invalid freeze manifest')
    if set(bundle['code_sha256']) != set(CODE_FILES):
        raise ValueError('Incomplete inference source hashes')
    for name, expected in bundle['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Inference source changed after freezing: {name}')
    for name, expected in bundle['artifacts_sha256'].items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path.resolve()) or digest(artifact) != expected:
            raise ValueError(f'Frozen artifact hash mismatch: {name}')
    with np.load(path / 'frozen.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return bundle, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True,
                        help='Selected expert paths in their original study invocation order')
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bundle = freeze_study(args.study, args.runs, args.items, args.out)
    print(json.dumps({'out': str(args.out), 'models': len(bundle['models']),
                      'validation_replay': bundle['validation_replay'], 'test_read': False}, indent=2))


if __name__ == '__main__':
    main()
