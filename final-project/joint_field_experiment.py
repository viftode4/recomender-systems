"""TRAIN-only joint item/rating likelihood for the unchanged categorical field.

This separate experiment never reads TEST. Its protocol precedes fitting;
checkpoint selection uses only meta-fit joint NLL, then development is opened.
"""
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp, softmax

from categorical_experiment import (DEFAULT_CONFIG, aggregate_diagnostics,
    array_digest, categorical_matrix, categorical_metrics, make_model,
    masked_episode, restore, snapshot, state_digest)
from exception_experiment import evaluate_ratings, load_source, recommendations
from study import digest, grouped, partition_users, write_json


VARIANTS = ('adaptive', 'fixed_flow')
CHECKPOINTS = (10, 30, 60, 100)
SOURCE_FILES = ('joint_field_experiment.py', 'JOINT_FIELD_PROTOCOL.md',
    'categorical_field.py', 'categorical_experiment.py', 'exception_experiment.py',
    'exception_model.py', 'study.py', 'metrics.py', 'hybrid_constraints.py')


def protocol(seeds):
    return {'schema_version': 1, 'stage': 'development', 'test_read': False,
        'test_evaluated': False, 'status': 'written_before_fitting',
        'seeds': list(seeds), 'training_objective': 'joint_recorded_item_rating_multinomial',
        'selection_metric': 'meta_fit_macro_joint_nll',
        'selection_direction': 'minimize; exact ties retain earlier checkpoint',
        'variants': list(VARIANTS), 'model_config': DEFAULT_CONFIG,
        'epochs': max(CHECKPOINTS), 'checkpoints': list(CHECKPOINTS),
        'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 16,
        'numerical_threads': 1, 'context_fraction': .8,
        'training_reduction': 'Mean masked-probe joint NLL per user, then mean across users.',
        'training_categories': [1, 2, 3, 4, 5],
        'ranking_adapters': {'all_observed': 'logsumexp(all five rating logits)',
                             'liked_record': 'logsumexp(rating 4 and 5 logits)'},
        'conditional_rating_adapter': 'softmax over each item\'s five rating logits',
        'normalizer': 'All catalog items except padding and context observations, times five categories; all probes remain eligible.',
        'inference': 'Original TRAIN ratings only; validation labels never enter context.',
        'controls': 'Identical allocated parameters, initial weights, episodes and batch order; fixed_flow caches initial routes and gates jointly within each query.',
        'negative_assumption': 'Missing pairs compete as possible recorded outcomes in the denominator; they are not observed dislikes. Selection/exposure biases remain.',
        'likelihood_scope': 'One exchangeable recorded event conditional on eligible catalog, not binary ever-rated probability or chronological next event.',
        'decomposition': 'joint NLL = conditional category NLL + item-event NLL, unit coefficients; only query-global logit shift remains invariant.',
        'baselines': {'event_count_pseudocount': 1., 'global_category_pseudocount': 1.,
                      'item_user_category_smoothing_mass': 10.,
                      'names': ['event_global_category', 'event_item_category', 'event_item_user_tilt'],
                      'validation_tuning': False},
        'cohorts': 'Original user partition: meta-fit selection; disjoint development evaluation. Reused validation is exploratory.',
        'claims': 'No novelty, psychological interpretation, unbiased exposure, independent confirmation, or SOTA claim.'}


def joint_probe_loss(model, context, targets, target_mask):
    """Macro-user joint multinomial loss; targets never enter model evidence."""
    import torch
    if (context.shape != targets.shape or context.shape != target_mask.shape
            or context.dtype != torch.long or targets.dtype != torch.long
            or target_mask.dtype != torch.bool or torch.any(context[target_mask] != 0)
            or torch.any((targets[target_mask] < 1) | (targets[target_mask] > 5))
            or torch.any(target_mask[:, 0])):
        raise ValueError('Valid probe categories must be absent from context and exclude padding')
    rows, columns = torch.where(target_mask)
    counts = torch.bincount(rows, minlength=len(context))
    if not len(rows) or torch.any(counts == 0):
        raise ValueError('Every user requires a held-out training probe')
    logits = model(context)['rating_logits']
    eligible = context == 0
    eligible[:, 0] = False
    candidates = logits.masked_fill(~eligible.unsqueeze(-1), -torch.inf)
    denominator = torch.logsumexp(candidates.flatten(1), dim=1)
    losses = denominator[rows] - logits[rows, columns, targets[rows, columns] - 1]
    sums = torch.zeros(len(context), dtype=losses.dtype, device=losses.device).scatter_add_(0, rows, losses)
    return (sums / counts).mean()


def validate_logits(logits, users, items, context):
    values = np.asarray(logits)
    observed = np.asarray(context)
    if (values.shape != (len(users), len(items), 5) or not np.isfinite(values).all()
            or observed.shape != values.shape[:2] or len(set(users)) != len(users)
            or len(set(items)) != len(items) or not len(items) or items[0] != '[PAD]'
            or np.any(observed[:, 0]) or not np.isfinite(observed).all()
            or np.any((observed < 0) | (observed > 5))
            or not np.equal(observed, np.floor(observed)).all()):
        raise ValueError('Invalid finite joint logits, categorical context, or ID order')
    eligible = observed == 0
    eligible[:, 0] = False
    if not np.all(eligible.any(axis=1)):
        raise ValueError('Every user requires at least one eligible item')
    return values.astype(float, copy=False), eligible


def joint_metrics(logits, users, items, context, labels, chosen_users):
    values, eligible = validate_logits(logits, users, items, context)
    item_logits = logsumexp(values, axis=-1)
    denominators = logsumexp(np.where(eligible, item_logits, -np.inf), axis=1)
    um, im = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    chosen = set(chosen_users)
    if not chosen or not chosen.issubset(um):
        raise ValueError('Invalid evaluation cohort')
    rows = {user: [] for user in sorted(chosen)}
    for (user, item), rating in labels.items():
        if user not in chosen:
            continue
        if (item not in im or not np.isfinite(rating) or rating != int(rating)
                or not 1 <= rating <= 5 or not eligible[um[user], im[item]]):
            raise ValueError('Held-out rated event must be an eligible valid item/category')
        u, i, r = um[user], im[item], int(rating) - 1
        rows[user].append([denominators[u] - values[u, i, r],
                          denominators[u] - item_logits[u, i],
                          item_logits[u, i] - values[u, i, r]])
    if any(not observations for observations in rows.values()):
        raise ValueError('Each chosen user requires a held-out event')
    per_user = {u: {'joint_nll': float(np.mean(v, axis=0)[0]),
                    'item_event_nll': float(np.mean(v, axis=0)[1]),
                    'conditional_category_nll': float(np.mean(v, axis=0)[2]),
                    'observations': len(v)} for u, v in rows.items()}
    macro = np.mean([[v[key] for key in ('joint_nll', 'item_event_nll', 'conditional_category_nll')]
                     for v in per_user.values()], axis=0)
    pooled = np.mean([v for row in rows.values() for v in row], axis=0)
    return {'users': len(chosen), 'observations': sum(map(len, rows.values())),
            'macro_joint_nll': float(macro[0]), 'pooled_joint_nll': float(pooled[0]),
            'macro_item_event_nll': float(macro[1]),
            'macro_conditional_category_nll': float(macro[2]), 'per_user': per_user,
            'scope': 'One recorded item/category event among eligible outcomes; missing records are unknown.'}


def ranking_scores(logits):
    values = np.asarray(logits, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 5 or not np.isfinite(values).all():
        raise ValueError('Require finite five-category logits')
    return {'all_observed_adapter': logsumexp(values, axis=-1),
            'liked_record_adapter': logsumexp(values[:, :, 3:], axis=-1)}


def evaluate_logits(logits, users, items, context, training_pairs, labels, chosen_users, stage='validation'):
    joint = joint_metrics(logits, users, items, context, labels, chosen_users)
    categorical = categorical_metrics(softmax(logits, axis=-1), users, items, labels, chosen_users)
    ranked = {}
    for adapter, scores in ranking_scores(logits).items():
        recs = recommendations(scores, users, items, context != 0, 10)
        ranked[adapter] = evaluate_ratings(recs, chosen_users, labels, grouped(training_pairs),
            items[1:], Counter(i for _, i in training_pairs), 10, stage=stage)
    return {'joint': joint, 'categorical': categorical, 'ranking': ranked}


def baseline_logits(matrix):
    """Predeclared TRAIN count references; all masses are strictly positive."""
    ratings = np.asarray(matrix)
    if (ratings.ndim != 2 or not len(ratings) or ratings.shape[1] < 2
            or np.any(ratings[:, 0]) or np.any((ratings < 0) | (ratings > 5))
            or not np.isfinite(ratings).all() or not np.equal(ratings, np.floor(ratings)).all()
            or not np.count_nonzero(ratings)):
        raise ValueError('Invalid TRAIN categorical matrix')
    counts = np.stack([(ratings == c).astype(float) for c in range(1, 6)], axis=-1)
    global_counts = counts.sum((0, 1)) + 1.
    prior = global_counts / global_counts.sum()
    item_counts, user_counts = counts.sum(0), counts.sum(1)
    item = (item_counts + 10 * prior) / (item_counts.sum(-1, keepdims=True) + 10)
    user = (user_counts + 10 * prior) / (user_counts.sum(-1, keepdims=True) + 10)
    event_mass = np.count_nonzero(ratings, axis=0).astype(float) + 1.
    global_mass = np.broadcast_to(event_mass[None, :, None] * prior, ratings.shape + (5,))
    item_mass = np.broadcast_to(event_mass[None, :, None] * item[None, :, :], ratings.shape + (5,))
    tilted = item_mass * user[:, None, :] / prior
    return {name: np.log(value).astype(np.float32) for name, value in (
        ('event_global_category', global_mass), ('event_item_category', item_mass),
        ('event_item_user_tilt', tilted))}


def predict_logits(model, matrix):
    import torch
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(matrix), 16):
            result.append(model(torch.as_tensor(matrix[start:start+16], dtype=torch.long))['rating_logits'].numpy())
    return np.concatenate(result)


def train_variant(matrix, users, items, validation, fit_users, seed, variant, output):
    import torch
    before = time.monotonic()
    model = make_model(matrix.shape[1], seed, variant)
    initial = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trace, grid, best = [], [], None
    for epoch in range(1, max(CHECKPOINTS) + 1):
        model.train()
        context, mask = masked_episode(matrix, seed, epoch)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 977])).permutation(len(matrix))
        total_loss = 0.
        for start in range(0, len(matrix), 16):
            batch = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            loss = joint_probe_loss(model, torch.as_tensor(context[batch]),
                                    torch.as_tensor(matrix[batch]), torch.as_tensor(mask[batch]))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint training loss')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        trace.append({'epoch': epoch, 'macro_train_joint_probe_nll': total_loss / len(matrix),
                      'context_sha256': array_digest(context), 'probe_mask_sha256': array_digest(mask)})
        if epoch in CHECKPOINTS:
            logits = predict_logits(model, matrix)
            nll = joint_metrics(logits, users, items, matrix, validation, fit_users)['macro_joint_nll']
            row = {'epoch': epoch, 'meta_fit_macro_joint_nll': nll}
            grid.append(row)
            torch.save(snapshot(model, epoch, seed), output / f'epoch-{epoch}.pt')
            if best is None or nll < best['meta_fit_macro_joint_nll']:
                best = dict(row)
                np.savez_compressed(output / 'selected-logits.npz', users=np.asarray(users),
                                    items=np.asarray(items), logits=logits)
            print(f'{seed} joint {variant} epoch {epoch}: meta-fit joint NLL {nll:.6f}', flush=True)
    selected = output / f"epoch-{best['epoch']}.pt"
    best.update(checkpoint=selected.name, checkpoint_sha256=digest(selected),
                initial_state_sha256=initial, selection_metric='meta_fit_macro_joint_nll',
                selection_cohort='meta_fit', parameter_count=sum(p.numel() for p in model.parameters()),
                elapsed_seconds=time.monotonic() - before)
    write_json(output / 'training-trace.json', trace)
    write_json(output / 'selection-grid.json', grid)
    write_json(output / 'selection.json', best)
    restored, _ = restore(selected)
    replay = predict_logits(restored, matrix)
    with np.load(output / 'selected-logits.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['logits'], replay):
            raise AssertionError('Saved checkpoint failed exact raw-logit replay')
    best.update(logits_array_sha256=array_digest(replay), checkpoint_replay_exact=True)
    write_json(output / 'selection.json', best)
    return best


def aggregate_result(result):
    ranked = {}
    for adapter, metrics in result['ranking'].items():
        ranked[adapter] = {
            'all_observed': metrics['all_observed']['aggregate'],
            'liked_ratings': metrics['liked_ratings']['aggregate'] if metrics['liked_ratings'] else None,
            **{key: metrics[key] for key in ('denominators', 'known_dislike_rate_per_slot',
                'dislike_rate_among_validation_rated_recommendations', 'caution')}}
    return {'joint': {k: v for k, v in result['joint'].items() if k != 'per_user'},
            'categorical': {k: v for k, v in result['categorical'].items() if k != 'per_user'},
            'ranking': ranked}


def joint_contrast(first, second, seed):
    a, b = first['joint']['per_user'], second['joint']['per_user']
    if set(a) != set(b):
        raise ValueError('Contrast cohorts differ')
    differences = np.asarray([a[u]['joint_nll'] - b[u]['joint_nll'] for u in sorted(a)])
    means = np.random.default_rng(seed).choice(differences, (2000, len(differences)), replace=True).mean(1)
    return {'users': len(differences), 'macro_joint_nll_difference': float(differences.mean()),
            'descriptive_user_bootstrap_interval_95': np.quantile(means, [.025, .975]).tolist(),
            'scope': 'Negative favors first. Unadjusted descriptive interval on reused validation, not independent confirmation.'}


def run_seed(source, ratings, metadata, output, seed):
    before = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    sources = {name: digest(root / name) for name in SOURCE_FILES}
    write_json(output / 'protocol.json', {**protocol([seed]), 'source_sha256': sources})
    users, items, train, valid, training, validation, _, _, upstream = load_source(source, ratings, metadata)
    if upstream.get('validation_used_for_training', True):
        raise ValueError('Source must exclude validation from training')
    matrix = categorical_matrix(users, items, training)
    fit_users, dev_users = partition_users(users, seed)
    write_json(output / 'cohorts.json', {'meta_fit': sorted(fit_users), 'development': sorted(dev_users)})
    np.savez_compressed(output / 'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    write_json(output / 'catalog.json', {'padding_index': 0, 'items': items})
    for name in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(source / name, output / name)
    selections = {}
    for variant in VARIANTS:
        folder = output / variant
        folder.mkdir()
        selections[variant] = train_variant(matrix, users, items, validation, fit_users, seed, variant, folder)
    if len({selection['initial_state_sha256'] for selection in selections.values()}) != 1:
        raise AssertionError('Core variants had different initial weights')
    # Immutable selection boundary before ANY development metric below.
    write_json(output / 'selection.json', selections)
    evaluated, diagnostics = {}, {}
    for variant in VARIANTS:
        folder = output / variant
        with np.load(folder / 'selected-logits.npz', allow_pickle=False) as saved:
            logits = saved['logits']
        evaluated[variant] = evaluate_logits(logits, users, items, matrix, train, validation, dev_users)
        restored, _ = restore(folder / selections[variant]['checkpoint'])
        diagnostics[variant] = aggregate_diagnostics(restored, matrix)
        write_json(folder / 'diagnostics.json', diagnostics[variant])
        for adapter, scores in ranking_scores(logits).items():
            export = folder / adapter
            export.mkdir()
            np.savez_compressed(export / 'valid-scores.npz', users=np.asarray(users), items=np.asarray(items), scores=scores)
            for filename in ('train.tsv', 'valid.tsv', 'ids.json'):
                shutil.copyfile(source / filename, export / filename)
            write_json(export / 'manifest.json', {
                'status': 'complete', 'model': 'JointCategoricalField', 'variant': variant,
                'ranking_adapter': adapter, 'seed': seed, 'validation_used_for_training': False,
                'selection': selections[variant], 'source_sha256': sources,
                'data_sha256': upstream['data_sha256'], 'split_sha256': upstream['split_sha256'],
                'test_read': False, 'test_evaluated': False,
                'predictor_kind': 'static_train_fitted_full_catalog',
                'score_space': protocol([seed])['ranking_adapters'][adapter.removesuffix('_adapter')],
                'scores_sha256': digest(export / 'valid-scores.npz'),
                'export_sha256': {'valid-scores.npz': digest(export / 'valid-scores.npz')},
                'source_run': str(source.resolve()), 'source_manifest_sha256': digest(source / 'manifest.json')})
    for name, logits in baseline_logits(matrix).items():
        evaluated[name] = evaluate_logits(logits, users, items, matrix, train, validation, dev_users)
    comparisons = {f'adaptive_minus_{name}': joint_contrast(evaluated['adaptive'], result, seed + 419)
                   for name, result in evaluated.items() if name != 'adaptive'}
    if sources != {name: digest(root / name) for name in SOURCE_FILES}:
        raise RuntimeError('Source changed during joint experiment')
    write_json(output / 'metrics.json', evaluated)
    write_json(output / 'comparisons.json', comparisons)
    result = {'models': {name: aggregate_result(value) for name, value in evaluated.items()},
              'selections': selections, 'comparisons': comparisons,
              'cohorts': {'meta_fit': len(fit_users), 'development': len(dev_users)},
              'diagnostics': diagnostics, 'elapsed_seconds': time.monotonic() - before}
    write_json(output / 'aggregates.json', result)
    write_json(output / 'manifest.json', {'status': 'complete', 'seed': seed,
        'test_read': False, 'test_evaluated': False, 'source_sha256': sources,
        'source_run': str(source.resolve()), 'source_manifest_sha256': digest(source / 'manifest.json'),
        'data_sha256': upstream['data_sha256'], 'split_sha256': upstream['split_sha256'],
        'train_categories_sha256': array_digest(matrix), 'protocol_sha256': digest(output / 'protocol.json'),
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return result


def curate(output, evidence, seeds):
    evidence.mkdir(parents=True, exist_ok=False)
    manifests = {str(s): json.loads((output / str(s) / 'manifest.json').read_text()) for s in seeds}
    aggregate = {'schema_version': 1, 'stage': 'development', 'test_read': False,
                 'seeds': {str(s): json.loads((output / str(s) / 'aggregates.json').read_text()) for s in seeds}}
    write_json(evidence / 'protocol.json', json.loads((output / 'protocol.json').read_text()))
    write_json(evidence / 'aggregates.json', aggregate)
    write_json(evidence / 'provenance.json', {'schema_version': 1, 'test_read': False,
        'protocol_sha256': digest(output / 'protocol.json'), 'seeds': {seed: {
            key: data[key] for key in ('source_sha256', 'source_manifest_sha256', 'data_sha256',
                                       'split_sha256', 'train_categories_sha256', 'protocol_sha256')}
            for seed, data in manifests.items()}})
    lines = ['# Joint recorded-item and rating likelihood', '',
        'Exploratory development results on reused validation; final TEST remains closed.', '',
        'The unchanged categorical field uses joint recorded-event likelihood. Checkpoints minimize meta-fit macro-user joint NLL at epochs 10, 30, 60 or 100.', '',
        '| Seed | Model | Joint NLL | Conditional rating CE | All-observed nDCG@10 | Liked-record nDCG@10 |',
        '|---|---|---:|---:|---:|---:|']
    for seed, result in aggregate['seeds'].items():
        for name, model in result['models'].items():
            lines.append(f"| {seed} | {name} | {model['joint']['macro_joint_nll']:.5f} | {model['categorical']['macro_cross_entropy']:.5f} | {model['ranking']['all_observed_adapter']['all_observed']['ndcg@10']:.5f} | {model['ranking']['liked_record_adapter']['liked_ratings']['ndcg@10']:.5f} |")
    lines += ['', 'All-observed ranking uses logsumexp of all five logits. Liked-record ranking uses logsumexp of category 4 and 5 logits. Missing pairs compete as possible recorded outcomes, not observed dislikes. This is not an exposure-corrected or satisfaction probability model.', '',
        'Adaptive versus fixed flow jointly tests recurrent routing and source-gate changes. Shared validation reuse and unadjusted descriptive intervals limit interpretation. No claim of novel likelihood, SOTA, causal meaning, or identified emotions is made.']
    (evidence / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(evidence / 'SHA256.json', {p.name: digest(p) for p in sorted(evidence.iterdir()) if p.is_file()})


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--item-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'protocol.json', protocol(args.seeds))
    results = {}
    for seed in args.seeds:
        results[str(seed)] = run_seed(args.source_root / f'{seed}-EASE-1', args.ratings,
                                      args.item_metadata, args.out / str(seed), seed)
        write_json(args.out / 'summary.json', results)
    if args.evidence:
        curate(args.out, args.evidence, args.seeds)
    print(json.dumps({'status': 'complete', 'out': str(args.out), 'test_read': False}), flush=True)


if __name__ == '__main__':
    main()
