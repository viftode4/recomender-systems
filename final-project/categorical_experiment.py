"""Validation-only runner for a categorical recurrent rating field.

Only TRAIN ratings reach the model. Validation pair labels select declared
checkpoints on meta-fit users, then evaluate the disjoint development cohort.
The final TEST split is never opened. This standalone track changes no sealed
primary-study inference code.
"""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from exception_experiment import evaluate_ratings, load_source, recommendations
from study import digest, grouped, partition_users, write_json


def array_digest(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256(f'{array.dtype}:{array.shape}:'.encode())
    result.update(memoryview(array).cast('B'))
    return result.hexdigest()


def categorical_matrix(users, items, training):
    """Encode observed TRAIN ratings as 1..5, with zero meaning unobserved.

    This API deliberately accepts no validation argument. No hard positive or
    negative labels are created; rating category 3 is a supervised category.
    """
    if not items or items[0] != '[PAD]' or len(set(users)) != len(users) or len(set(items)) != len(items):
        raise ValueError('Require unique user/item IDs and padding item at zero')
    user_map, item_map = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    matrix = np.zeros((len(users), len(items)), dtype=np.int64)
    for user, item, rating in training:
        if user not in user_map or item not in item_map or item_map[item] == 0:
            raise ValueError('Training rating has unknown or padding ID')
        if not np.isfinite(rating) or rating != int(rating) or not 1 <= rating <= 5:
            raise ValueError('Require an integer rating category from 1 to 5')
        row, col = user_map[user], item_map[item]
        if matrix[row, col] != 0:
            raise ValueError('Duplicate training observation')
        matrix[row, col] = int(rating)
    return matrix


def validate_probabilities(probabilities, users, items):
    values = np.asarray(probabilities)
    if (values.shape != (len(users), len(items), 5)
            or not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values > 1)
            or not np.allclose(values.sum(axis=-1), 1., atol=2e-6, rtol=0)):
        raise ValueError('Require finite normalized five-category probabilities')
    return values


def ranking_scores(probabilities, users, items):
    """P(rating >=4) is an evaluation readout, never a training target."""
    values = validate_probabilities(probabilities, users, items)
    return values[:, :, 3:].sum(axis=-1)


def categorical_metrics(probabilities, users, items, labels, chosen_users):
    """Evaluate only the given cohort's observed validation ratings.

    CE uses natural logarithms. Brier sums five squared category errors per
    observation. Macro metrics give each user equal weight; pooled metrics give
    each rating equal weight. Unobserved user/item pairs have no target.
    """
    probabilities = validate_probabilities(probabilities, users, items)
    user_map, item_map = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    chosen = set(chosen_users)
    if not chosen or not chosen.issubset(user_map):
        raise ValueError('Invalid evaluation cohort')
    values = {user: [] for user in sorted(chosen)}
    for (user, item), rating in labels.items():
        if user not in chosen:
            continue
        if item not in item_map or item_map[item] == 0:
            raise ValueError('Unknown validation item')
        if not np.isfinite(rating) or rating != int(rating) or not 1 <= rating <= 5:
            raise ValueError('Invalid validation rating category')
        distribution = probabilities[user_map[user], item_map[item]].astype(float)
        target = np.eye(5)[int(rating)-1]
        values[user].append([
            -np.log(max(float(distribution[int(rating)-1]), 1e-15)),
            float(np.sum((distribution-target)**2)),
            float((distribution @ np.arange(1., 6.)-rating)**2),
            float(np.argmax(distribution) == int(rating)-1),
        ])
    if any(not rows for rows in values.values()):
        raise ValueError('Evaluation user has no rated validation pair')
    per_user = {user: {'cross_entropy': float(np.mean(rows, axis=0)[0]),
                       'brier': float(np.mean(rows, axis=0)[1]),
                       'expected_rating_mse': float(np.mean(rows, axis=0)[2]),
                       'category_accuracy': float(np.mean(rows, axis=0)[3]),
                       'observations': len(rows)} for user, rows in values.items()}
    macro = np.mean([[row[key] for key in ('cross_entropy', 'brier', 'expected_rating_mse', 'category_accuracy')]
                     for row in per_user.values()], axis=0)
    pooled = np.mean([row for rows in values.values() for row in rows], axis=0)
    return {'users': len(chosen), 'observations': sum(len(rows) for rows in values.values()),
            'macro_cross_entropy': float(macro[0]), 'pooled_cross_entropy': float(pooled[0]),
            'macro_brier': float(macro[1]), 'pooled_brier': float(pooled[1]),
            'macro_expected_rating_rmse': float(np.sqrt(macro[2])),
            'pooled_expected_rating_rmse': float(np.sqrt(pooled[2])),
            'macro_category_accuracy': float(macro[3]), 'pooled_category_accuracy': float(pooled[3]),
            'per_user': per_user,
            'scope': 'Conditional on observed rating entries; missing ratings are unknown.'}


def evaluate_predictions(probabilities, users, items, training_matrix, training_pairs, labels, chosen_users, k=10):
    scores = ranking_scores(probabilities, users, items)
    observed = np.asarray(training_matrix) != 0
    if observed.shape != scores.shape:
        raise ValueError('Training mask and scores differ')
    recs = recommendations(scores, users, items, observed, k)
    rating_metrics = categorical_metrics(probabilities, users, items, labels, chosen_users)
    ranked_metrics = evaluate_ratings(recs, chosen_users, labels, grouped(training_pairs),
                                     items[1:], Counter(i for _, i in training_pairs), k)
    return {'categorical': rating_metrics, 'ranking': ranked_metrics}, recs


def categorical_baselines(training_matrix, pseudocount=10.):
    """Fixed TRAIN-only global, item and item/user categorical references.

    All categories have one global pseudo-observation to keep prior ratios and
    log scores finite, including tiny synthetic datasets with absent categories.
    The per-item and per-user smoothing mass is fixed before validation access.
    """
    ratings = np.asarray(training_matrix)
    if (ratings.ndim != 2 or not len(ratings) or ratings.shape[1] < 2
            or np.any(ratings[:, 0]) or np.any(ratings < 0) or np.any(ratings > 5)
            or not np.equal(ratings, np.floor(ratings)).all()
            or not np.isfinite(pseudocount) or pseudocount <= 0
            or not np.count_nonzero(ratings)):
        raise ValueError('Invalid categorical baseline training data')
    category_counts = np.stack([(ratings == category).astype(float) for category in range(1, 6)], axis=-1)
    global_counts = category_counts.sum(axis=(0, 1)) + 1.
    prior = global_counts / global_counts.sum()
    item_counts = category_counts.sum(axis=0)
    user_counts = category_counts.sum(axis=1)
    item_probabilities = (item_counts+pseudocount*prior)/(item_counts.sum(axis=-1, keepdims=True)+pseudocount)
    user_probabilities = (user_counts+pseudocount*prior)/(user_counts.sum(axis=-1, keepdims=True)+pseudocount)
    combined = item_probabilities[None, :, :] * user_probabilities[:, None, :] / prior
    combined /= combined.sum(axis=-1, keepdims=True)
    return {
        'global_histogram': np.broadcast_to(prior, ratings.shape+(5,)).copy(),
        'item_histogram': np.broadcast_to(item_probabilities, ratings.shape+(5,)).copy(),
        'item_user_product': combined,
    }


VARIANTS = ('adaptive', 'fixed_flow', 'hard_clamp')
CHECKPOINTS = (10, 30, 60, 100)
DEFAULT_CONFIG = {'dim': 8, 'ports': 16, 'steps': 4, 'damping': .5}
SOURCE_FILES = ('categorical_field.py', 'categorical_experiment.py',
                'exception_experiment.py', 'study.py', 'metrics.py',
                'hybrid_constraints.py', 'ADAPTIVE_RESEARCH.md')


def masked_episode(training_matrix, seed, epoch, fraction=.8):
    """Identity-only masking; changing a hidden value cannot reveal it as input."""
    ratings = np.asarray(training_matrix)
    if ratings.ndim != 2 or np.any(ratings[:, 0]) or not 0 < fraction < 1:
        raise ValueError('Invalid episode inputs')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7331]))
    context, target_mask = ratings.copy(), np.zeros(ratings.shape, dtype=bool)
    for row in range(len(ratings)):
        observed = np.flatnonzero(ratings[row])
        if len(observed) < 2:
            raise ValueError('Each training user requires at least two observed ratings')
        keep = max(1, min(len(observed)-1, int(np.floor(fraction*len(observed)))))
        hidden = rng.permutation(observed)[keep:]
        context[row, hidden] = 0
        target_mask[row, hidden] = True
    return context, target_mask


def training_loss(model, context, targets, target_mask):
    """Macro-user five-category probe CE, with no missing-pair loss terms."""
    import torch
    import torch.nn.functional as F
    if (context.shape != targets.shape or context.shape != target_mask.shape
            or target_mask.dtype != torch.bool or torch.any(context[target_mask] != 0)
            or torch.any((targets[target_mask] < 1) | (targets[target_mask] > 5))):
        raise ValueError('Probe targets must be valid categories fully absent from context')
    logits = model(context)['rating_logits']
    rows, columns = torch.where(target_mask)
    if not len(rows):
        raise ValueError('No supervised probe ratings')
    losses = F.cross_entropy(logits[rows, columns], targets[rows, columns]-1, reduction='none')
    sums = torch.zeros(len(context), device=losses.device, dtype=losses.dtype).scatter_add_(0, rows, losses)
    counts = torch.bincount(rows, minlength=len(context))
    if torch.any(counts == 0):
        raise ValueError('Every training row requires a probe target')
    return (sums/counts).mean()


def predict_probabilities(model, training_matrix, batch_size=16):
    import torch
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(training_matrix), batch_size):
            context = torch.as_tensor(training_matrix[start:start+batch_size], dtype=torch.long)
            output.append(torch.softmax(model(context)['rating_logits'], dim=-1).cpu().numpy())
    return np.concatenate(output)


def state_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(array_digest(tensor.detach().cpu().numpy()).encode())
    return result.hexdigest()


def make_model(n_items, seed, variant, config=None):
    import torch
    from categorical_field import CategoricalEvidenceField
    torch.manual_seed(seed)
    return CategoricalEvidenceField(n_items=n_items, variant=variant, **(config or DEFAULT_CONFIG))


def snapshot(model, epoch, seed):
    return {'config': dict(model.config), 'epoch': epoch, 'seed': seed,
            'state_dict': {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}


def restore(path):
    import torch
    from categorical_field import CategoricalEvidenceField
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = CategoricalEvidenceField(**saved['config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model, saved


def aggregate_diagnostics(model, training_matrix, batch_size=16):
    """Aggregate TRAIN-query behavior, preserving each summary's denominator."""
    import torch
    from categorical_field import summarize_diagnostics
    summaries = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(training_matrix), batch_size):
            batch = torch.as_tensor(training_matrix[start:start+batch_size], dtype=torch.long)
            summaries.append(summarize_diagnostics(model(batch, return_diagnostics=True)['diagnostics']))
    rows = []
    for index in range(len(summaries[0]['steps'])):
        result = {'step': index+1}
        for key in summaries[0]['steps'][index]:
            if key == 'step':
                continue
            weights = [summary['context_observations'] if key.startswith('source_gate_') else
                       summary['users']-summary['empty_context_users'] if key.startswith('up_routing_') else
                       summary['users'] for summary in summaries]
            result[key] = float(np.average([summary['steps'][index][key] for summary in summaries], weights=weights))
        rows.append(result)
    return {'users': sum(summary['users'] for summary in summaries),
            'context_observations': sum(summary['context_observations'] for summary in summaries),
            'query': 'All original TRAIN ratings; aggregate behavior only, no held-out labels.',
            'interpretation': 'Computational gates and routes are not identified trust, feelings or explanations.',
            'steps': rows}


def protocol(seeds, mode):
    return {
        'schema_version': 1, 'stage': 'development', 'status': 'written_before_fitting',
        'seeds': list(seeds), 'mode': mode, 'test_read': False, 'test_evaluated': False,
        'training_objective': 'categorical_cross_entropy',
        'training_reduction': 'mean probe CE per user, then mean across batch users',
        'selection_metric': 'meta_fit_macro_cross_entropy',
        'selection_direction': 'minimize; exact ties retain earliest checkpoint',
        'rating_categories': [1, 2, 3, 4, 5], 'ranking_adapter': 'P(rating>=4)',
        'variants': list(VARIANTS), 'model_config': DEFAULT_CONFIG,
        'epochs': max(CHECKPOINTS), 'checkpoints': list(CHECKPOINTS),
        'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 16,
        'numerical_threads': 1, 'context_fraction': .8,
        'training_masks': 'Every epoch uses deterministic identity-only per-user masking with at least one context and one probe; identical across variants.',
        'initialization': 'Identical allocated parameter shapes and seed across variants; initialization hashes must match.',
        'effective_capacity': 'Equal allocated shapes do not mean equal active capacity: hard-clamp source-gate parameters are inactive; fixed-flow computes initial routes and gates only once.',
        'control_interpretation': 'Adaptive versus fixed-flow jointly tests recurrent routing and fidelity changes. Hard-clamp fixes observed source states, so observed-only pooling also fixes upward routing and port states; only candidate downward queries can change. It does not isolate gates alone.',
        'inference': 'All original TRAIN ratings supplied; TRAIN observations of every category excluded from ranking.',
        'cohorts': 'Original seed-based user half for meta-fit selection; other half for development. Same previously inspected validation splits, exploratory reuse.',
        'baselines': {'names': ['global_histogram', 'item_histogram', 'item_user_product'],
                      'global_category_pseudocount_each': 1., 'item_user_smoothing_mass': 10.,
                      'selection': 'Fixed formulas; no validation-fitted parameter or grid.'},
        'labels': 'Only original TRAIN and validation pair IDs authorize rating parsing. No final TEST split is opened or its rating values parsed.',
        'ranking_scope': 'Liked and all-observed ranking are diagnostics with a different target; CE is not ranking accuracy.',
        'claims': 'Exploratory reused validation evidence; no novelty, independent replication, causal, psychological or deployment claim.',
    }


def benchmark(training_matrix, seed, batch_size=16, measured_batches=20):
    """Train-only timing and shape benchmark, never evaluates validation."""
    import time
    import torch
    context, mask = masked_episode(training_matrix, seed, 1)
    count = min(batch_size, len(training_matrix))
    ctx = torch.as_tensor(context[:count], dtype=torch.long)
    targets = torch.as_tensor(training_matrix[:count], dtype=torch.long)
    probes = torch.as_tensor(mask[:count], dtype=torch.bool)
    results, reference_init = {}, None
    for variant in VARIANTS:
        model = make_model(training_matrix.shape[1], seed, variant)
        initial = state_digest(model)
        if reference_init is not None and initial != reference_init:
            raise ValueError('Variants do not share identical initialization')
        reference_init = initial
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        model.train()
        seconds = []
        for step in range(measured_batches+1):
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, ctx, targets, probes)
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite benchmark training loss')
            if step:
                seconds.append(time.perf_counter()-before)
        mean = float(np.mean(seconds))
        results[variant] = {'mean_training_batch_seconds': mean, 'measured_batches': measured_batches,
                            'batch_users': count, 'catalog_items_including_padding': training_matrix.shape[1],
                            'parameter_count': sum(p.numel() for p in model.parameters()),
                            'initial_state_sha256': initial,
                            'estimated_full_budget_fit_seconds': mean*int(np.ceil(len(training_matrix)/batch_size))*max(CHECKPOINTS),
                            'planned_epochs': max(CHECKPOINTS),
                            'training_loss_after_timing_only': float(loss.detach())}
    return {'training_only': True, 'validation_metrics_read': False, 'test_read': False,
            'variants': results, 'scope': 'CPU one-thread warm batch timing; excludes checkpoint prediction, I/O and contention.'}


def train_variant(matrix, users, items, validation, fit_users, seed, variant, output):
    import time
    import torch
    model = make_model(matrix.shape[1], seed, variant)
    initial_hash = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trace, selection_grid, best = [], [], None
    before = time.monotonic()
    for epoch in range(1, max(CHECKPOINTS)+1):
        model.train()
        context, mask = masked_episode(matrix, seed, epoch)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 977])).permutation(len(matrix))
        epoch_loss = 0.
        for start in range(0, len(matrix), 16):
            batch = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, torch.as_tensor(context[batch], dtype=torch.long),
                                 torch.as_tensor(matrix[batch], dtype=torch.long),
                                 torch.as_tensor(mask[batch], dtype=torch.bool))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())*len(batch)
        trace.append({'epoch': epoch, 'macro_train_probe_ce': epoch_loss/len(matrix),
                      'context_sha256': array_digest(context), 'probe_mask_sha256': array_digest(mask)})
        if epoch in CHECKPOINTS:
            probabilities = predict_probabilities(model, matrix)
            ce = categorical_metrics(probabilities, users, items, validation, fit_users)['macro_cross_entropy']
            row = {'epoch': epoch, 'meta_fit_macro_cross_entropy': ce}
            selection_grid.append(row)
            torch.save(snapshot(model, epoch, seed), output/f'epoch-{epoch}.pt')
            if best is None or ce < best['meta_fit_macro_cross_entropy']:
                best = dict(row)
                np.savez_compressed(output/'selected-probabilities.npz', users=np.asarray(users),
                                    items=np.asarray(items), probabilities=probabilities)
            print(f'{seed} {variant} epoch {epoch}: meta-fit macro CE {ce:.6f}', flush=True)
    selected_path = output/f"epoch-{best['epoch']}.pt"
    best.update({'checkpoint': selected_path.name, 'checkpoint_sha256': digest(selected_path),
                 'selection_metric': 'meta_fit_macro_cross_entropy', 'selection_cohort': 'meta_fit',
                 'initial_state_sha256': initial_hash,
                 'parameter_count': sum(p.numel() for p in model.parameters()),
                 'elapsed_seconds': time.monotonic()-before})
    write_json(output/'training-trace.json', trace)
    write_json(output/'selection-grid.json', selection_grid)
    write_json(output/'selection.json', best)
    # Replay the saved state instead of silently trusting a cached array.
    restored, _ = restore(selected_path)
    replayed = predict_probabilities(restored, matrix)
    with np.load(output/'selected-probabilities.npz', allow_pickle=False) as saved:
        if not np.array_equal(replayed, saved['probabilities']):
            raise AssertionError('Selected checkpoint did not replay exact probabilities')
    best['probability_array_sha256'] = array_digest(replayed)
    best['checkpoint_replay_exact'] = True
    write_json(output/'selection.json', best)
    return best


def ce_contrast(first, second, seed):
    """Negative differences favor the first model; descriptive reused users."""
    a, b = first['categorical']['per_user'], second['categorical']['per_user']
    if set(a) != set(b):
        raise ValueError('Contrast users differ')
    differences = np.asarray([a[user]['cross_entropy']-b[user]['cross_entropy'] for user in sorted(a)])
    means = np.random.default_rng(seed).choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    return {'users': len(differences), 'macro_ce_difference': float(differences.mean()),
            'descriptive_user_bootstrap_interval_95': np.quantile(means, [.025, .975]).tolist(),
            'scope': 'Negative favors first model. Descriptive, unadjusted, conditional on reused validation and checkpoint selection; not independent confirmation.'}


def aggregate_result(result):
    rating = {key: value for key, value in result['categorical'].items() if key != 'per_user'}
    ranked = result['ranking']
    return {'categorical': rating, 'ranking': {
        'all_observed': ranked['all_observed']['aggregate'],
        'liked_ratings': ranked['liked_ratings']['aggregate'] if ranked['liked_ratings'] else None,
        'denominators': ranked['denominators'],
        'known_dislike_rate_per_slot': ranked['known_dislike_rate_per_slot'],
        'dislike_rate_among_validation_rated_recommendations': ranked['dislike_rate_among_validation_rated_recommendations'],
        'caution': ranked['caution']}}


def run_seed(source, ratings, metadata, output, seed, benchmark_only=False):
    import json
    import shutil
    import time
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    sources = {name: digest(root/name) for name in SOURCE_FILES}
    write_json(output/'protocol.json', {**protocol([seed], 'benchmark' if benchmark_only else 'experiment'),
                                       'source_sha256': sources, 'source_run': str(source.resolve())})
    users, items, train, valid, training, validation, _, _, source_manifest = load_source(source, ratings, metadata)
    if source_manifest.get('validation_used_for_training', True):
        raise ValueError('Source must have trained without validation')
    matrix = categorical_matrix(users, items, training)
    fit_users, dev_users = partition_users(users, seed)
    write_json(output/'cohorts.json', {'meta_fit': sorted(fit_users), 'development': sorted(dev_users)})
    np.savez_compressed(output/'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    write_json(output/'catalog.json', {'padding_index': 0, 'items': items})
    for part in ('train', 'valid'):
        shutil.copyfile(source/f'{part}.tsv', output/f'{part}.tsv')
    shutil.copyfile(source/'ids.json', output/'ids.json')
    timing = benchmark(matrix, seed)
    write_json(output/'benchmark.json', timing)
    if benchmark_only:
        write_json(output/'manifest.json', {'status': 'benchmark_complete', 'test_read': False,
            'source_sha256': sources, 'source_manifest_sha256': digest(source/'manifest.json'),
            'split_sha256': source_manifest['split_sha256'], 'data_sha256': source_manifest['data_sha256'],
            'validation_metrics_read': False, 'seconds': time.monotonic()-started})
        return timing
    selections = {}
    for variant in VARIANTS:
        directory = output/variant
        directory.mkdir()
        selections[variant] = train_variant(matrix, users, items, validation, fit_users, seed, variant, directory)
    if len({value['initial_state_sha256'] for value in selections.values()}) != 1:
        raise AssertionError('Initial model state differed across core variants')
    # This persistence boundary occurs before any development metric is computed.
    write_json(output/'selection.json', selections)
    evaluated, behavior = {}, {}
    for variant in VARIANTS:
        with np.load(output/variant/'selected-probabilities.npz', allow_pickle=False) as saved:
            probabilities = saved['probabilities']
        evaluated[variant], recs = evaluate_predictions(probabilities, users, items, matrix, train, validation, dev_users)
        selected_model, _ = restore(output/variant/selections[variant]['checkpoint'])
        behavior[variant] = aggregate_diagnostics(selected_model, matrix)
        empty_probabilities = predict_probabilities(selected_model, np.zeros((1, len(items)), dtype=np.int64))[0]
        candidates = matrix == 0
        candidates[:, 0] = False
        tv = np.abs(probabilities-empty_probabilities[None]).sum(axis=-1)/2
        adapter_delta = np.abs(probabilities[:, :, 3:].sum(axis=-1)-empty_probabilities[:, 3:].sum(axis=-1))
        behavior[variant]['empty_context_comparison'] = {
            'macro_candidate_distribution_tv': float(np.mean([tv[row, mask].mean() for row, mask in enumerate(candidates)])),
            'macro_candidate_adapter_absolute_change': float(np.mean([adapter_delta[row, mask].mean() for row, mask in enumerate(candidates)])),
            'candidate_fraction_tv_above_0001': float(np.mean(tv[candidates] > .001)),
            'scope': 'All TRAIN-query users; candidates exclude every TRAIN observation and padding. No held-out labels; changes do not establish predictive usefulness.'}
        write_json(output/variant/'behavior.json', behavior[variant])
        scores = ranking_scores(probabilities, users, items)
        np.savez_compressed(output/variant/'valid-scores.npz', users=np.asarray(users), items=np.asarray(items), scores=scores)
        write_json(output/variant/'recommendations.json', {u: recs[u] for u in sorted(dev_users)})
        write_json(output/variant/'manifest.json', {
            'status': 'complete', 'model': 'CategoricalEvidenceField', 'variant': variant,
            'validation_used_for_training': False, 'selection': selections[variant],
            'source_sha256': sources, 'data_sha256': source_manifest['data_sha256'],
            'split_sha256': source_manifest['split_sha256'], 'test_read': False, 'test_evaluated': False,
            'training_objective': 'five-category macro-user masked TRAIN probe cross-entropy',
            'score_space': 'P(rating>=4), unmasked full catalog; categorical TRAIN-only inference',
            'predictor_kind': 'static_train_fitted_full_catalog',
            'export_sha256': {name: digest(output/variant/name) for name in ('valid-scores.npz', 'selected-probabilities.npz')},
            'model_config': {**DEFAULT_CONFIG, 'n_items': len(items), 'variant': variant}})
    for name, probabilities in categorical_baselines(matrix).items():
        evaluated[name], _ = evaluate_predictions(probabilities, users, items, matrix, train, validation, dev_users)
    comparisons = {f'adaptive_minus_{name}': ce_contrast(evaluated['adaptive'], evaluated[name], seed+871)
                   for name in evaluated if name != 'adaptive'}
    if {name: digest(root/name) for name in SOURCE_FILES} != sources:
        raise RuntimeError('Source changed during experiment')
    write_json(output/'metrics.json', evaluated)
    write_json(output/'comparisons.json', comparisons)
    result = {'models': {name: aggregate_result(value) for name, value in evaluated.items()},
              'selections': selections, 'comparisons': comparisons,
              'cohorts': {'meta_fit': len(fit_users), 'development': len(dev_users)},
              'behavior': behavior,
              'timing': {'benchmark': timing, 'total_seconds': time.monotonic()-started}}
    write_json(output/'aggregates.json', result)
    write_json(output/'manifest.json', {
        'status': 'complete', 'seed': seed, 'test_read': False, 'test_evaluated': False,
        'source_sha256': sources, 'source_manifest_sha256': digest(source/'manifest.json'),
        'source_run': str(source.resolve()), 'split_sha256': source_manifest['split_sha256'],
        'data_sha256': source_manifest['data_sha256'], 'train_categories_sha256': array_digest(matrix),
        'selection_protocol_sha256': digest(output/'protocol.json'),
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'seconds': time.monotonic()-started})
    return result


def curate(output, evidence, seeds):
    import json
    evidence.mkdir(parents=True, exist_ok=False)
    plan = json.loads((output/'protocol.json').read_text())
    aggregate = {'schema_version': 1, 'stage': 'development', 'test_read': False,
                 'seeds': {str(seed): json.loads((output/str(seed)/'aggregates.json').read_text()) for seed in seeds}}
    manifests = {str(seed): json.loads((output/str(seed)/'manifest.json').read_text()) for seed in seeds}
    provenance = {'schema_version': 1, 'test_read': False, 'test_evaluated': False,
                  'protocol_sha256': digest(output/'protocol.json'), 'seeds': {
                      seed: {key: value[key] for key in ('source_sha256', 'source_manifest_sha256', 'split_sha256',
                                                        'data_sha256', 'train_categories_sha256', 'selection_protocol_sha256')}
                      for seed, value in manifests.items()}}
    write_json(evidence/'protocol.json', plan)
    write_json(evidence/'aggregates.json', aggregate)
    write_json(evidence/'provenance.json', provenance)
    rows = ['# Categorical evidence-field pilot', '',
            'Exploratory development evidence on reused validation splits. Final TEST remains closed.', '',
            f"Each neural variant selects the smallest macro-user five-category cross-entropy on meta-fit users at epochs {', '.join(map(str, CHECKPOINTS))}. Training, initialization and masks share a fixed budget. Count baselines have no validation-selected settings.", '',
            '| Seed | Model | Macro rating CE | Macro Brier | Expected-rating RMSE | Liked nDCG@10 | All-observed nDCG@10 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for seed, result in aggregate['seeds'].items():
        for name, model in result['models'].items():
            cat, rank = model['categorical'], model['ranking']
            rows.append(f"| {seed} | {name} | {cat['macro_cross_entropy']:.5f} | {cat['macro_brier']:.5f} | {cat['macro_expected_rating_rmse']:.5f} | {rank['liked_ratings']['ndcg@10']:.5f} | {rank['all_observed']['ndcg@10']:.5f} |")
    rows += ['', 'Lower CE, Brier and RMSE are better. Ranking uses the declared P(rating>=4) readout and measures a different target; it does not determine checkpoints. Missing ratings are unknown. Categorical metrics condition on rated pairs and cannot establish satisfaction for unobserved items.', '',
             'The changing routes and source gates are model variables, not identified feelings, trust, or explanations. Shared validation reuse, one small dataset and descriptive unadjusted user intervals preclude claims of independent confirmation or state of the art.']
    (evidence/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    write_json(evidence/'SHA256.json', {p.name: digest(p) for p in sorted(evidence.iterdir()) if p.is_file()})


def main():
    import argparse
    import json
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--item-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    if args.benchmark_only and args.evidence:
        parser.error('A benchmark does not produce scientific evidence')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', protocol(args.seeds, 'benchmark' if args.benchmark_only else 'experiment'))
    results = {}
    for seed in args.seeds:
        results[str(seed)] = run_seed(args.source_root/f'{seed}-EASE-1', args.ratings,
                                     args.item_metadata, args.out/str(seed), seed, args.benchmark_only)
        write_json(args.out/'summary.json', results)
    if args.evidence:
        curate(args.out, args.evidence, args.seeds)
    print(json.dumps({'status': 'complete', 'benchmark_only': args.benchmark_only,
                      'out': str(args.out), 'test_read': False}), flush=True)


if __name__ == '__main__':
    main()
