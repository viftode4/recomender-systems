"""Validation-only personalized-dislike audit with predeclared matched controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from exception_experiment import (evaluate_ratings, load_source, objective_value,
                                  recommendations)
from exception_model import rating_matrices
from negative_controls import switch_negative_labels
from negative_information import (channel_features, fit_decoder, permute_dislike_weights,
                                  split_context_probe, surprise_dislikes)
from study import digest, grouped, partition_users, write_json


PENALTIES = (50., 250., 1000.)
FIXED_PENALTY = 250.
REPLICATES = 5
MODES = ('context', 'full_train')
SOURCE_FILES = ('negative_information.py', 'negative_information_experiment.py',
                'negative_controls.py', 'exception_model.py', 'exception_experiment.py',
                'study.py', 'metrics.py', 'hybrid_constraints.py',
                'NEGATIVE_INFORMATION_PROTOCOL.md')


def array_digest(value):
    a = np.ascontiguousarray(value)
    signature = f'{a.dtype}:{a.shape}:'.encode()
    result = hashlib.sha256(signature)
    result.update(memoryview(a).cast('B'))
    return result.hexdigest()


def pairs_digest(rows):
    return hashlib.sha256('\n'.join(f'{u}\t{i}' for u, i, _ in sorted(rows)).encode()).hexdigest()


def prepare(training, users, items, seed):
    """No validation argument is accepted by the representation preparation API."""
    context, probe = split_context_probe(training, seed+70001)
    context_signed, context_observed = rating_matrices(users, items, context)
    probe_signed, probe_observed = rating_matrices(users, items, probe)
    if np.any(context_observed & probe_observed):
        raise AssertionError('Context and probe overlap')
    full_signed, observed = rating_matrices(users, items, training)
    if not np.array_equal(observed, context_observed | probe_observed):
        raise AssertionError('Context/probe do not partition original training')
    pc, dc = context_signed > 0, context_signed < 0
    pp, dp = probe_signed > 0, probe_signed < 0
    pf, df = full_signed > 0, full_signed < 0
    eligible = pc.sum(axis=1) >= 1
    if not eligible.any():
        raise ValueError('No decoder-fitting users meet predeclared support')
    teacher = fit_decoder(pc, pc, 250., 1)
    wc, wf = surprise_dislikes(pc, dc, teacher), surprise_dislikes(pf, df, teacher)
    branches = {
        'positive': {'group': 'positive', 'context': None, 'full_train': None},
        'plain': {'group': 'plain', 'context': dc, 'full_train': df},
        'surprise': {'group': 'surprise', 'context': wc, 'full_train': wf},
    }
    controls, weight_controls = {}, {}
    for replicate in range(REPLICATES):
        weight_seed = seed+100000+replicate
        permuted_context = permute_dislike_weights(wc, weight_seed)
        permuted_full = permute_dislike_weights(wf, weight_seed)
        for original, changed in ((wc, permuted_context), (wf, permuted_full)):
            if not (np.array_equal(original > 0, changed > 0)
                    and np.array_equal(np.sort(original, axis=1), np.sort(changed, axis=1))):
                raise AssertionError('Weight control changed support or multiset')
        branches[f'weight_permutation_{replicate}'] = {
            'group': 'weight_permutation', 'context': permuted_context,
            'full_train': permuted_full, 'replicate': replicate, 'seed': weight_seed}
        weight_controls[str(replicate)] = {'seed': weight_seed, **{
            mode: {'changed_rows': int(np.any(original != changed, axis=1).sum()),
                   'changed_nonzero_weights': int(np.count_nonzero(original != changed)),
                   'maximum_mass_error': float(np.abs(original.sum(axis=1)-changed.sum(axis=1)).max())}
            for mode, original, changed in (('context', wc, permuted_context), ('full_train', wf, permuted_full))}}
        nc, diagnostic_context = switch_negative_labels(dc, context_observed & ~pc,
                                                       seed+200000+replicate, row_groups=eligible.astype(int))
        np_, diagnostic_probe = switch_negative_labels(dp, probe_observed & ~pp,
                                                       seed+300000+replicate, row_groups=eligible.astype(int))
        nf = nc | np_
        invariants = {
            'context_positives_unchanged': bool(not np.any(nc & pc)),
            'full_positives_unchanged': bool(not np.any(nf & pf)),
            'context_observed_support_unchanged': bool(not np.any(nc & ~context_observed)),
            'full_observed_support_unchanged': bool(not np.any(nf & ~observed)),
            'full_row_negative_margins': bool(np.array_equal(nf.sum(axis=1), df.sum(axis=1))),
            'full_item_negative_margins': bool(np.array_equal(nf.sum(axis=0), df.sum(axis=0))),
            'context_null_retained_in_full_query': bool(np.array_equal(nf & context_observed, nc)),
            'fitted_context_item_negative_margins': bool(np.array_equal(nc[eligible].sum(axis=0), dc[eligible].sum(axis=0))),
            'fitted_full_item_negative_margins': bool(np.array_equal(nf[eligible].sum(axis=0), df[eligible].sum(axis=0))),
        }
        if not all(invariants.values()):
            raise AssertionError('Combined perturbation invariants failed')
        controls[str(replicate)] = {'context': diagnostic_context, 'probe': diagnostic_probe,
                                   'combined_invariants': invariants}
        branches[f'placement_null_{replicate}'] = {
            'group': 'placement_null', 'context': nc, 'full_train': nf,
            'replicate': replicate}
    diagnostics = {
        'context_pair_sha256': pairs_digest(context), 'probe_pair_sha256': pairs_digest(probe),
        'context_observations': len(context), 'probe_observations': len(probe),
        'users': len(users), 'decoder_fit_users': int(eligible.sum()),
        'decoder_fit_excluded_users': int((~eligible).sum()),
        'minimum_decoder_context_likes': 1, 'minimum_decoder_probe_likes': 0,
        'decoder_fit_users_without_probe_likes': int((eligible & (pp.sum(axis=1) == 0)).sum()),
        'users_without_context_likes': int((pc.sum(axis=1) == 0).sum()),
        'users_without_probe_likes': int((pp.sum(axis=1) == 0).sum()),
        'users_without_context_dislikes': int((dc.sum(axis=1) == 0).sum()),
        'users_with_at_least_two_context_dislikes': int((dc.sum(axis=1) >= 2).sum()),
        'context_likes': int(pc.sum()), 'context_dislikes': int(dc.sum()),
        'probe_likes': int(pp.sum()), 'probe_dislikes': int(dp.sum()),
        'teacher_coefficient_sha256': array_digest(teacher.coefficients),
        'context_positive_sha256': array_digest(pc), 'probe_target_sha256': array_digest(pp),
        'context_surprise_sha256': array_digest(wc), 'full_surprise_sha256': array_digest(wf),
        'maximum_context_dislike_mass_error': float(np.abs(wc.sum(axis=1)-dc.sum(axis=1)).max()),
        'maximum_full_dislike_mass_error': float(np.abs(wf.sum(axis=1)-df.sum(axis=1)).max()),
        'context_weight_variance_over_dislikes': {
            'mean_across_users_with_dislikes': float(np.mean([
                np.var(wc[row, dc[row]]) for row in range(len(users)) if dc[row].any()] or [0.])),
            'users_with_nonconstant_dislike_weights': int(sum(
                np.ptp(wc[row, dc[row]]) > 1e-12 for row in range(len(users)) if dc[row].any())),
        },
        'weight_controls': weight_controls,
        'placement_controls': controls,
    }
    return {'positive': {'context': pc, 'full_train': pf}, 'target': pp,
            'eligible': eligible, 'observed': observed, 'branches': branches,
            'diagnostics': diagnostics}


def build_decoder(prepared, branch, penalty):
    x = channel_features(prepared['positive']['context'], branch['context'])
    keep = prepared['eligible']
    return fit_decoder(x[keep], prepared['target'][keep], penalty,
                       1 if branch['context'] is None else 2)


def predict(prepared, branch, model, mode):
    if mode not in MODES:
        raise ValueError('Unknown inference mode')
    return model.predict(channel_features(prepared['positive'][mode], branch[mode]))


def contrast(first, second, seed, k=10):
    """Paired interval on users after averaging each control's metric, not scores."""
    key = f'ndcg@{k}'
    a = first['liked_ratings']['per_user']
    b = [result['liked_ratings']['per_user'] for result in second]
    if any(set(other) != set(a) for other in b):
        raise ValueError('Contrast users differ')
    delta = np.array([a[u][key]-np.mean([other[u][key] for other in b]) for u in sorted(a)])
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
    return {'users': len(delta), 'mean_delta': float(delta.mean()),
            'descriptive_user_bootstrap_interval_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'control_replicates': len(b),
            'control_development_ndcg': [objective_value(v, 'liked_ratings', k) for v in second],
            'scope': 'Descriptive and not multiplicity-adjusted; conditional on recorded control draws; not a randomization p-value or independent repeated-split inference.'}


def run_seed(source, ratings, metadata, out, seed, save_scores=False):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    sources = {name: digest(Path(__file__).parent/name) for name in SOURCE_FILES}
    write_json(out/'protocol.json', {
        'status': 'written_before_fitting', 'seed': seed, 'source_run': str(source.resolve()),
        'source_sha256': sources, 'penalties': PENALTIES, 'fixed_penalty': FIXED_PENALTY,
        'control_replicates': REPLICATES, 'selection_mode': 'context',
        'selection_objective': 'liked nDCG@10', 'test_read': False, 'test_evaluated': False,
        'primary_inference': 'context', 'secondary_inference': 'full_train',
    })
    users, items, train, valid, training, validation, _, _, source_manifest = load_source(source, ratings, metadata)
    prepared = prepare(training, users, items, seed)
    write_json(out/'diagnostics.json', prepared['diagnostics'])
    fit_users, dev_users = partition_users(users, seed)
    history, counts = grouped(train), Counter(i for _, i in train)
    write_json(out/'cohorts.json', {'meta_fit': sorted(fit_users), 'development': sorted(dev_users)})
    grid, timings = [], []
    for name, branch in prepared['branches'].items():
        for penalty in PENALTIES:
            before = time.monotonic()
            model = build_decoder(prepared, branch, penalty)
            scores = predict(prepared, branch, model, 'context')
            recs = recommendations(scores, users, items, prepared['observed'], 10)
            metrics = evaluate_ratings(recs, fit_users, validation, history, items[1:], counts, 10)
            grid.append({'branch': name, 'group': branch['group'], 'penalty': penalty,
                         'meta_fit_liked_ndcg': objective_value(metrics, 'liked_ratings', 10)})
            timings.append({'stage': 'selection', 'branch': name, 'penalty': penalty,
                            'seconds': time.monotonic()-before})
            del model, scores, recs, metrics
        print(f'{seed} meta-fit grid complete: {name}', flush=True)
    selected = {}
    for group in ('positive', 'plain', 'surprise', 'weight_permutation', 'placement_null'):
        values = {penalty: float(np.mean([v['meta_fit_liked_ndcg'] for v in grid
                                         if v['group'] == group and v['penalty'] == penalty]))
                  for penalty in PENALTIES}
        selected[group] = {'penalty': max(PENALTIES, key=lambda p: values[p]),
                           'meta_fit_means': values,
                           'replicates_averaged': REPLICATES if group in ('weight_permutation', 'placement_null') else 1}
    # Development labels affect no choice above this persistence boundary.
    write_json(out/'selection-grid.json', grid)
    write_json(out/'selection.json', selected)
    evaluated = {budget: {mode: {} for mode in MODES} for budget in ('fixed', 'selected')}
    score_hashes = {}
    for name, branch in prepared['branches'].items():
        chosen = selected[branch['group']]['penalty']
        for penalty in sorted({FIXED_PENALTY, chosen}):
            before = time.monotonic()
            model = build_decoder(prepared, branch, penalty)
            budgets = [b for b, p in (('fixed', FIXED_PENALTY), ('selected', chosen)) if p == penalty]
            for mode in MODES:
                scores = predict(prepared, branch, model, mode)
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite audit scores')
                recs = recommendations(scores, users, items, prepared['observed'], 10)
                metrics = evaluate_ratings(recs, dev_users, validation, history, items[1:], counts, 10)
                for budget in budgets:
                    evaluated[budget][mode][name] = metrics
                score_key = f'{name}-lambda{penalty:g}-{mode}'
                score_hashes[score_key] = array_digest(scores)
                if save_scores:
                    (out/'scores').mkdir(exist_ok=True)
                    np.savez_compressed(out/'scores'/f'{score_key}.npz',
                                        users=np.asarray(users), items=np.asarray(items), scores=scores)
                del scores, recs
            timings.append({'stage': 'development_both_modes', 'branch': name, 'penalty': penalty,
                            'seconds': time.monotonic()-before})
            del model
        print(f'{seed} development complete: {name}', flush=True)
    contrasts = {}
    for budget in evaluated:
        contrasts[budget] = {}
        for mode, metrics in evaluated[budget].items():
            contrasts[budget][mode] = {
                'plain_minus_positive': contrast(metrics['plain'], [metrics['positive']], seed+11),
                'surprise_minus_plain': contrast(metrics['surprise'], [metrics['plain']], seed+12),
                'surprise_minus_weight_permutation': contrast(metrics['surprise'],
                    [metrics[f'weight_permutation_{r}'] for r in range(REPLICATES)], seed+13),
                'plain_minus_placement_null': contrast(metrics['plain'],
                    [metrics[f'placement_null_{r}'] for r in range(REPLICATES)], seed+14),
            }
    current = {name: digest(Path(__file__).parent/name) for name in SOURCE_FILES}
    if current != sources:
        raise RuntimeError('Audit source files changed during run')
    write_json(out/'metrics.json', evaluated)
    write_json(out/'contrasts.json', contrasts)
    write_json(out/'timings.json', timings)
    manifest = {
        'status': 'complete', 'seed': seed, 'test_read': False, 'test_evaluated': False,
        'source_sha256': sources, 'source_run': str(source.resolve()),
        'source_manifest_sha256': digest(source/'manifest.json'),
        'data_sha256': source_manifest['data_sha256'],
        'split_sha256': {p: source_manifest['split_sha256'][p] for p in ('train', 'valid')},
        'score_array_sha256': score_hashes, 'scores_saved': save_scores,
        'validation_used_for_decoder_or_teacher_fit': False,
        'validation_used_for_ridge_selection': 'meta-fit users only; strict-context score objective',
        'label_access': 'Only source train and valid pair IDs authorize atomic rating parsing; no test split is opened.',
        'context_probe': {k: v for k, v in prepared['diagnostics'].items() if k != 'placement_controls'},
        'elapsed_seconds': time.monotonic()-started,
    }
    write_json(out/'manifest.json', manifest)
    return {budget: {mode: {name: {
                'liked_ndcg10': objective_value(result, 'liked_ratings', 10),
                'known_dislike_rate_per_slot': result['known_dislike_rate_per_slot'],
                'liked_users': result['denominators']['liked_ratings_users']}
                for name, result in metrics.items()} for mode, metrics in modes.items()}
            for budget, modes in evaluated.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--item-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--save-scores', action='store_true', help='Save large raw prediction archives for each evaluated model')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'plan.json', {
        'seeds': args.seeds, 'penalties': PENALTIES, 'fixed_penalty': FIXED_PENALTY,
        'replicates': REPLICATES, 'protocol_sha256': digest(Path(__file__).with_name('NEGATIVE_INFORMATION_PROTOCOL.md')),
        'test_read': False, 'test_evaluated': False, 'selection': 'context meta-fit liked nDCG@10',
    })
    summary = {}
    for seed in args.seeds:
        summary[str(seed)] = run_seed(args.source_root/f'{seed}-EASE-1', args.ratings,
            args.item_metadata, args.out/str(seed), seed, args.save_scores)
        write_json(args.out/'summary.json', summary)
    print(json.dumps({'status': 'complete', 'output': str(args.out), 'seeds': args.seeds}), flush=True)


if __name__ == '__main__':
    main()
