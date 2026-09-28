"""Validation-only, explicit-rating contrast study with matched-information controls.

All grids and representations are fixed before development metrics are evaluated.
Only train.tsv and valid.tsv identify usable labels; test.tsv is never opened.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from exception_model import (linear_rating_scores, neighborhood_scores, pair_candidate_scores,
                             permuted_partners, rating_matrices, representations, shuffled_partners)
from metrics import evaluate
from study import digest, grouped, load_genres, partition_users, ranking, read_pairs, write_json


def load_source(source, ratings, item_metadata):
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['test_evaluated']:
        raise ValueError('Require complete validation-only source')
    for part in ('train', 'valid'):
        if digest(source/f'{part}.tsv') != manifest['split_sha256'][part]:
            raise ValueError('Source split hash mismatch')
    for path in (ratings, item_metadata):
        if digest(path) != manifest['data_sha256'].get(path.name):
            raise ValueError('Source dataset hash mismatch')
    with np.load(source/'valid-scores.npz', allow_pickle=False) as archive:
        users, items = archive['users'].tolist(), archive['items'].tolist()
        source_scores = archive['scores'].astype(float)
    if len(set(users)) != len(users) or len(set(items)) != len(items):
        raise ValueError('Duplicate score IDs')
    if not items or items[0] != '[PAD]' or source_scores.shape != (len(users), len(items)):
        raise ValueError('Invalid score IDs or shape')
    if not np.isfinite(source_scores).all():
        raise ValueError('Nonfinite source scores')
    train, valid = read_pairs(source/'train.tsv'), read_pairs(source/'valid.tsv')
    if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) & set(valid):
        raise ValueError('Invalid train/validation partition')
    if set(grouped(valid)) != set(users) or set(grouped(train)) != set(users):
        raise ValueError('Interaction/score users mismatch')
    if {i for _, i in train + valid} - set(items[1:]):
        raise ValueError('Interaction/score items mismatch')
    allowed = set(train) | set(valid)
    selected_ratings = {}
    # The shared .inter file is scanned, but rating values outside the allowed
    # train/validation IDs are never parsed or retained. No test split is read.
    with ratings.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = (row['user_id:token'], row['item_id:token'])
            if pair in allowed:
                if pair in selected_ratings:
                    raise ValueError('Duplicate rated interaction')
                value = float(row['rating:float'])
                if not np.isfinite(value) or not 1 <= value <= 5:
                    raise ValueError('Invalid rating')
                selected_ratings[pair] = value
    if set(selected_ratings) != allowed:
        raise ValueError('Missing train/validation ratings')
    training = [(u, i, selected_ratings[u, i]) for u, i in train]
    validation = {(u, i): selected_ratings[u, i] for u, i in valid}
    genre_binary, _ = load_genres(item_metadata, items)
    return users, items, train, valid, training, validation, genre_binary, source_scores, manifest


def recommendations(scores, users, items, observed, k):
    result = {}
    for row, user in enumerate(users):
        eligible = np.flatnonzero(~observed[row])
        eligible = eligible[eligible != 0]
        result[user] = [items[i] for i in ranking(scores[row], eligible, k)]
    return result


def evaluate_ratings(recs, chosen_users, valid_ratings, history, catalog, counts, k, stage='validation'):
    if stage not in ('validation','test'):
        raise ValueError('Unknown evaluation stage')
    selected = set(chosen_users)
    all_truth, liked_truth, disliked_truth = {}, {}, {}
    for (user, item), value in valid_ratings.items():
        if user not in selected:
            continue
        all_truth.setdefault(user, set()).add(item)
        if value >= 4:
            liked_truth.setdefault(user, set()).add(item)
        elif value <= 2:
            disliked_truth.setdefault(user, set()).add(item)
    if selected != set(all_truth):
        raise ValueError('Evaluation cohort includes user without validation ratings')
    all_result = evaluate({u: recs[u] for u in selected}, all_truth, history, catalog, counts, k)
    liked_result = evaluate({u: recs[u] for u in liked_truth}, liked_truth, history, catalog, counts, k) if liked_truth else None
    dislike_hits = sum(len(set(recs[u]) & disliked_truth.get(u, set())) for u in selected)
    rated_hits = sum(len(set(recs[u]) & all_truth[u]) for u in selected)
    return {
        'stage': stage,
        'all_observed': all_result,
        'liked_ratings': liked_result,
        'denominators': {
            'all_observed_users': len(selected), 'liked_ratings_users': len(liked_truth),
            f'users_without_liked_{stage}_excluded_from_liked_metrics': len(selected)-len(liked_truth),
            f'{stage}_observations': sum(map(len, all_truth.values())),
            f'{stage}_likes': sum(map(len, liked_truth.values())),
            f'{stage}_dislikes': sum(map(len, disliked_truth.values())),
            'recommendation_slots': len(selected)*k,
            f'recommended_items_with_{stage}_rating': rated_hits,
            'recommended_known_dislikes': dislike_hits,
        },
        'known_dislike_rate_per_slot': dislike_hits / (len(selected)*k),
        f'dislike_rate_among_{stage}_rated_recommendations': dislike_hits/rated_hits if rated_hits else None,
        'caution': 'Missing ratings are unknown. Per-slot dislike rate counts only known held-out dislikes; observed-rating conditional rate is selection-biased.',
    }


def objective_value(metrics, objective, k):
    result = metrics[objective]
    if result is None:
        raise ValueError('Selection cohort has no liked validation ratings')
    return result['aggregate'][f'ndcg@{k}']


def paired_comparison(first, second, objective, seed, repetitions=2000, k=10):
    a, b = first[objective]['per_user'], second[objective]['per_user']
    if set(a) != set(b):
        raise ValueError('Paired comparison users differ')
    delta = np.asarray([a[u][f'ndcg@{k}']-b[u][f'ndcg@{k}'] for u in sorted(a)])
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(delta, (repetitions, len(delta)), replace=True), axis=1)
    return {'users': len(delta), 'mean_delta': float(delta.mean()),
            'descriptive_bootstrap_interval_95': np.quantile(means, [.025, .975]).tolist(),
            'scope': 'Frozen meta-fit choices on development users; exploratory repeated-split evidence, not independent replications or multiplicity-adjusted inference.'}


def export_run(out, source, users, items, scores, source_manifest, model, selection):
    out.mkdir(parents=True, exist_ok=False)
    for part in ('train', 'valid'):
        shutil.copyfile(source/f'{part}.tsv', out/f'{part}.tsv')
    np.savez_compressed(out/'valid-scores.npz', users=np.asarray(users), items=np.asarray(items), scores=scores)
    ids = json.loads((source/'ids.json').read_text())
    if ids.get('padding_index') != 0 or ids['items'] != items or not set(users).issubset(ids['users']):
        raise ValueError('Source internal ID mapping differs from score IDs')
    shutil.copyfile(source/'ids.json',out/'ids.json')
    exported = {
        'status': 'complete',
        'model': model, 'test_evaluated': False, 'validation_used_for_training': False,
        'predictor_kind': 'static_train_fitted_full_catalog',
        'implementation': ('Copied original RecBole EASE predictions from hash-verified source run'
                           if model=='ObservedEASE' else 'Custom transparent explicit-rating research model'),
        'score_space': 'unmasked finite scores aligned to source original IDs',
        'data_sha256': source_manifest['data_sha256'],
        'split_sha256': source_manifest['split_sha256'],
        'split_sizes': source_manifest.get('split_sizes',{}),
        'python': source_manifest.get('python'), 'versions': source_manifest.get('versions'),
        'selection': selection,
        'protocol': ('Every observed training rating is positive irrespective of value; original RecBole EASE predictions. Parameters selected only on meta-fit validation users.'
                     if model=='ObservedEASE' else 'Training ratings >=4 likes, <=2 dislikes, 3 neutral. Training-history masks include every observed rating. Parameters selected only on meta-fit validation users.'),
        'source_sha256': {p: digest(Path(__file__).parent/p) for p in ('exception_model.py','exception_experiment.py')},
        'export_sha256': {'valid-scores.npz': digest(out/'valid-scores.npz')},
        'source_run': str(source.resolve()),
        'source_manifest_sha256': digest(source/'manifest.json'),
        'settings': selection,
    }
    write_json(out/'manifest.json', exported)


def verify_reference(reference_root,seed,source_manifest):
    reference = json.loads((reference_root/str(seed)/'manifest.json').read_text())
    if reference['status']!='complete' or reference['seed']!=seed or reference['test_evaluated']:
        raise ValueError('Incomplete, wrong-seed, or test-exposed reference')
    if reference['data_sha256']!=source_manifest['data_sha256'] or any(
        reference['split_sha256'][p]!=source_manifest['split_sha256'][p] for p in ('train','valid')):
        raise ValueError('Reference split or data signatures differ')


def run_seed(source, ratings, item_metadata, out, seed, k=10, dimensions=32, features=128, revision=1):
    start = time.monotonic()
    (users, items, train, valid, training, validation, genres, ease_scores,
     source_manifest) = load_source(source, ratings, item_metadata)
    out.mkdir(parents=True, exist_ok=False)
    signed, observed = rating_matrices(users, items, training)
    fit_users, dev_users = partition_users(users, seed)
    history, counts = grouped(train), Counter(i for _, i in train)
    write_json(out/'protocol.json', {
        'seed': seed, 'status': 'protocol_written_before_model_fit',
        'hypothesis': 'Matched oriented liked-disliked contrast distributions transfer preferences beyond generic signed rating and geometry controls.',
        'selection': 'Each model independently selects on meta-fit nDCG@10; separately for all observations and likes. No development metric selects a model.',
        'grid': {'neighbors': [20, 80], 'dislike_weight': [0.0, 1.0]},
        'fixed_representation': {'dimensions': dimensions, 'fourier_features': features, 'rbf_gamma': 1.0},
        'meta_fit': sorted(fit_users), 'development': sorted(dev_users),
        'split_sha256': {p: source_manifest['split_sha256'][p] for p in ('train','valid')},
        'test_read': False, 'test_evaluated': False,
        'limits': 'Random per-user temporal-agnostic split; previous development studies used these users. No psychological emotion/exception validation; novelty unestablished.',
        'revision': revision,
        'revision_2_predeclared_reason': 'Revision1 oriented relation-user similarity lost liked-anchor location and was inferior to anchor-only. Revision2 preserves candidate similarity to liked anchors and gates it by matched dislike similarity. Exploratory response to development evidence, not independent confirmation.' if revision==2 else None,
        'revision_2_grid': {'linear_penalty':[50.,250.,1000.], 'pair_temperature':[.1,.5]} if revision==2 else None,
    })
    reps = representations(signed, genres, dimensions, features, seed)
    objectives = ('all_observed', 'liked_ratings')
    selections, evaluated, selected_scores, grid_results = {}, {}, {}, []

    def select(name, candidates):
        best = {objective: (-np.inf, None, None) for objective in objectives}
        for params,scores in candidates:
            recs = recommendations(scores, users, items, observed, k)
            metrics = evaluate_ratings(recs, fit_users, validation, history, items[1:], counts, k)
            grid_results.append({'model': name, **params, **{o: objective_value(metrics, o, k) for o in objectives}})
            for objective in objectives:
                value = objective_value(metrics, objective, k)
                if value > best[objective][0]:
                    best[objective] = (value, params, scores)
        for objective, (value, params, scores) in best.items():
            key = f'{name}:{objective}'
            selections[key] = {'model': name, **params, 'selection_objective': objective,
                               'selection_ndcg': value, 'selection_cohort': 'meta_fit', 'seed': seed}
            selected_scores[key] = scores
        print(f'{seed} selected {name}', flush=True)

    for name, similarity in reps.similarities.items():
        candidates = (({'neighbors': n,'dislike_weight': p}, neighborhood_scores(similarity,signed,n,p))
                      for n in (20,80) for p in (0.,1.) if name!='positive_knn' or p==0)
        select(name,candidates)
    if revision==2:
        for name in ('positive_ease','signed_ease','signed_channels'):
            select(name, (({'penalty':p},linear_rating_scores(signed,p,name)) for p in (50.,250.,1000.)))
        random_pairs = shuffled_partners(reps.pairs,signed,seed+991)
        for name,mode,pairs in (
            ('pair_gate','gate',reps.pairs),('pair_gate_random','gate',random_pairs),
            ('pair_anchor','anchor',reps.pairs),('pair_signed','signed',reps.pairs),
        ):
            temps = (.1,.5) if mode=='gate' else (.1,)
            select(name, (({'temperature':t},pair_candidate_scores(reps.geometry,pairs,signed,mode,t)) for t in temps))
    # Selection is committed to disk before looking at any development outcomes.
    write_json(out/'selection.json', selections)
    write_json(out/'selection-grid.json', grid_results)
    for key, scores in selected_scores.items():
        recs = recommendations(scores, users, items, observed, k)
        evaluated[key] = evaluate_ratings(recs, dev_users, validation, history, items[1:], counts, k)
    ease_recs = recommendations(ease_scores, users, items, observed, k)
    evaluated['source_EASE'] = evaluate_ratings(ease_recs, dev_users, validation, history, items[1:], counts, k)
    diagnostics = {
        'users': len(users), 'fallback_users': int(reps.fallback.sum()),
        'development_fallback_users': sum(bool(reps.fallback[j]) for j,u in enumerate(users) if u in dev_users),
        'users_without_training_dislikes': int(((signed < 0).sum(axis=1)==0).sum()),
        'pairs_total': sum(map(len, reps.pairs)),
        'users_with_matched_pairs': int((~reps.fallback).sum()),
        'pairs_per_user': {u: len(reps.pairs[j]) for j,u in enumerate(users)},
        'distinct_disliked_counterparts_per_user': {u: len({d for _,d in reps.pairs[j]}) for j,u in enumerate(users)},
        'shared_exact_pair_neighbor_fraction': float(reps.shared_pair_mask.sum()/(len(users)*(len(users)-1))),
        'matched_pairs_by_user': {u: [[items[p],items[d]] for p,d in reps.pairs[j]] for j,u in enumerate(users)},
    }
    # Falsification: can the representation transfer when neighbors sharing
    # any identical ordered item pair are forbidden? This is NOT item cold-start.
    transfer = {}
    for name in ('contrast_transfer', 'signed_geometry', 'unordered_pairs', 'random_partners'):
        params = selections[f'{name}:liked_ratings']
        scores = neighborhood_scores(reps.similarities[name], signed, params['neighbors'],
                                     params['dislike_weight'], reps.shared_pair_mask)
        recs = recommendations(scores, users, items, observed, k)
        transfer[name] = evaluate_ratings(recs, dev_users, validation, history, items[1:], counts, k)
    paired = {}
    for objective in objectives:
        target = evaluated[f'contrast_transfer:{objective}']
        for control in ('signed_knn','signed_geometry','unordered_pairs','anchor_only','first_moment','random_partners'):
            paired[f'{objective}:contrast_minus_{control}'] = paired_comparison(
                target, evaluated[f'{control}:{objective}'], objective, seed, k=k)
        if revision==2:
            target = evaluated[f'pair_gate:{objective}']
            for control in ('pair_gate_random','pair_anchor','pair_signed','positive_ease','signed_ease','signed_channels'):
                paired[f'{objective}:pair_gate_minus_{control}'] = paired_comparison(
                    target,evaluated[f'{control}:{objective}'],objective,seed,k=k)
    cohorts = {}
    cutoff = float(np.median([len(p) for p in reps.pairs]))
    for cohort_name, condition in (
        ('no_pairs', lambda j: len(reps.pairs[j]) == 0),
        ('few_pairs', lambda j: 0 < len(reps.pairs[j]) <= cutoff),
        ('many_pairs', lambda j: len(reps.pairs[j]) > cutoff),
    ):
        chosen = {u for j,u in enumerate(users) if u in dev_users and condition(j)}
        if not chosen:
            continue
        cohorts[cohort_name] = {'users': len(chosen), 'training_pair_count_median': cutoff, 'models': {}}
        for name in ('contrast_transfer','signed_geometry','signed_knn'):
            recs = recommendations(selected_scores[f'{name}:liked_ratings'], users, items, observed, k)
            cohorts[cohort_name]['models'][name] = evaluate_ratings(recs, chosen, validation, history, items[1:], counts, k)
    write_json(out/'metrics.json', evaluated)
    write_json(out/'diagnostics.json', diagnostics)
    write_json(out/'disjoint-pair-neighbor-audit.json', transfer)
    write_json(out/'paired-comparisons.json', paired)
    write_json(out/'cohorts.json', cohorts)
    for objective in objectives:
        key = f'contrast_transfer:{objective}'
        export_run(out/f'contrast-{objective}', source, users, items, selected_scores[key],
                   source_manifest, 'ContrastTransfer', selections[key])
        if revision==2:
            export_names = {'positive_ease':'PositiveEASE','signed_ease':'SignedEASE',
                            'signed_channels':'SignedChannelsLinear','pair_gate':'PairContrastGate',
                            'pair_gate_random':'PairContrastRandomPartners','pair_anchor':'PairAnchorKernel',
                            'pair_signed':'PairSignedKernel'}
            for name,model in export_names.items():
                key = f'{name}:{objective}'
                export_run(out/f'{name}-{objective}',source,users,items,selected_scores[key],
                           source_manifest,model,selections[key])
    summary = {
        key: {'all_ndcg': result['all_observed']['aggregate'][f'ndcg@{k}'],
              'like_ndcg': result['liked_ratings']['aggregate'][f'ndcg@{k}'],
              'known_dislike_rate_per_slot': result['known_dislike_rate_per_slot'],
              'liked_users': result['denominators']['liked_ratings_users']}
        for key,result in evaluated.items()
    }
    write_json(out/'summary.json', summary)
    write_json(out/'manifest.json', {
        'status': 'complete', 'seconds': time.monotonic()-start, 'seed': seed,
        'test_read': False, 'test_evaluated': False, 'source_run': str(source.resolve()),
        'source_score_sha256': digest(source/'valid-scores.npz'),
        'source_manifest_sha256': digest(source/'manifest.json'),
        'source_sha256': {p: digest(Path(__file__).parent/p) for p in ('exception_model.py','exception_experiment.py')},
        'data_sha256': source_manifest['data_sha256'],
        'split_sha256': {p: source_manifest['split_sha256'][p] for p in ('train','valid')},
    })
    return summary


def audit_implicit_controls(source_root, ratings, item_metadata, out, reference_root, seeds, k=10):
    """Equal-budget all-observed EASE comparator, using existing train-only archives.

    This supplemental audit was added after revision2 identified a comparison
    confound: its source EASE used fixed lambda250, unlike tuned explicit models.
    It changes no representation or model architecture and never reads test.
    """
    out.mkdir(parents=True,exist_ok=False)
    summary = {}
    for seed in seeds:
        target = out/str(seed)
        target.mkdir()
        candidates = []
        for run in sorted(p for p in source_root.glob(f'{seed}-EASE-*') if p.is_dir()):
            (users,items,train,valid,training,validation,genres,scores,manifest) = load_source(run,ratings,item_metadata)
            verify_reference(reference_root,seed,manifest)
            if manifest.get('validation_used_for_training',True):
                raise ValueError('Implicit control used validation for training')
            penalty = manifest['settings']['reg_weight']
            if penalty not in (50.,250.,1000.):
                raise ValueError('Unexpected implicit EASE grid')
            _,observed = rating_matrices(users,items,training)
            fit,dev = partition_users(users,seed)
            history,counts = grouped(train),Counter(i for _,i in train)
            recs = recommendations(scores,users,items,observed,k)
            fit_metrics = evaluate_ratings(recs,fit,validation,history,items[1:],counts,k)
            candidates.append((objective_value(fit_metrics,'liked_ratings',k),run,penalty,recs,
                               objective_value(fit_metrics,'all_observed',k)))
        if len(candidates)!=3 or {x[2] for x in candidates}!={50.,250.,1000.}:
            raise ValueError('Need exactly three expected EASE penalty candidates')
        # Independently check source split/catalog agreement before selecting.
        from study import load_runs
        load_runs([row[1] for row in candidates])
        best = max(candidates,key=lambda row:row[0])
        selection = {'source':str(best[1]),'penalty':best[2],'meta_fit_liked_ndcg':best[0],
                     'selection_cohort':'meta_fit','objective':'liked_ratings',
                     'grid':[{'source':str(r),'penalty':p,'meta_fit_liked_ndcg':v} for v,r,p,_,_ in candidates],
                     'test_read':False,'test_evaluated':False}
        write_json(target/'selection.json',selection)
        metrics = evaluate_ratings(best[3],dev,validation,history,items[1:],counts,k)
        references = json.loads((reference_root/str(seed)/'metrics.json').read_text())
        comparisons = {name:paired_comparison(references[f'{name}:liked_ratings'],metrics,'liked_ratings',seed,k=k)
                       for name in ('positive_ease','signed_ease','signed_channels')}
        write_json(target/'metrics.json',metrics)
        write_json(target/'paired-comparisons.json',comparisons)
        for objective,column in (('liked_ratings',0),('all_observed',4)):
            selected = max(candidates,key=lambda row:row[column])
            selected_manifest = json.loads((selected[1]/'manifest.json').read_text())
            with np.load(selected[1]/'valid-scores.npz',allow_pickle=False) as archive:
                selected_scores = archive['scores'].astype(float)
            export_run(target/f'observed_ease-{objective}',selected[1],users,items,selected_scores,
                       selected_manifest,'ObservedEASE',{'selection_cohort':'meta_fit',
                       'selection_objective':objective,'selection_ndcg':selected[column],
                       'penalty':selected[2],'seed':seed})
        summary[str(seed)] = {'selection':selection,'liked_ndcg':objective_value(metrics,'liked_ratings',k),
                              'all_ndcg':objective_value(metrics,'all_observed',k),
                              'known_dislike_rate_per_slot':metrics['known_dislike_rate_per_slot'],
                              'comparisons':comparisons}
    write_json(out/'summary.json',summary)


def audit_pair_permutation(source_root,ratings,item_metadata,out,reference_root,seeds,k=10):
    """Post-review exact pairing ablation; all counterpart identities/counts fixed."""
    out.mkdir(parents=True,exist_ok=False)
    summary = {}
    for seed in seeds:
        source = source_root/f'{seed}-EASE-1'
        users,items,train,valid,training,validation,genres,_,manifest = load_source(source,ratings,item_metadata)
        verify_reference(reference_root,seed,manifest)
        signed,observed = rating_matrices(users,items,training)
        reps = representations(signed,genres,32,128,seed)
        pairs = permuted_partners(reps.pairs,seed+991)
        fit,dev = partition_users(users,seed)
        history,counts = grouped(train),Counter(i for _,i in train)
        target = out/str(seed)
        target.mkdir()
        write_json(target/'protocol.json',{'control':'Permute each user counterpart multiset; preserve every liked anchor and each disliked item reuse count.',
                                          'grid_temperature':[.1,.5],'selection_cohort':'meta_fit','test_read':False})
        candidates = []
        for temperature in (.1,.5):
            scores = pair_candidate_scores(reps.geometry,pairs,signed,'gate',temperature)
            recs = recommendations(scores,users,items,observed,k)
            fitted = evaluate_ratings(recs,fit,validation,history,items[1:],counts,k)
            candidates.append((temperature,scores,recs,fitted))
        choices = {objective:max(candidates,key=lambda row:objective_value(row[3],objective,k))
                   for objective in ('all_observed','liked_ratings')}
        selections = {objective:{'temperature':row[0],'selection_cohort':'meta_fit',
                     'selection_objective':objective,'seed':seed,
                     'selection_ndcg':objective_value(row[3],objective,k)} for objective,row in choices.items()}
        write_json(target/'selection.json',selections)
        results = {}
        for objective,row in choices.items():
            results[objective] = evaluate_ratings(row[2],dev,validation,history,items[1:],counts,k)
            export_run(target/f'pair_gate_permuted-{objective}',source,users,items,row[1],manifest,
                       'PairContrastPermutedPartners',selections[objective])
        reference = json.loads((reference_root/str(seed)/'metrics.json').read_text())
        comparisons = {objective:paired_comparison(reference[f'pair_gate:{objective}'],results[objective],objective,seed,k=k)
                       for objective in choices}
        write_json(target/'metrics.json',results)
        write_json(target/'paired-comparisons.json',comparisons)
        summary[str(seed)]={'liked_ndcg':objective_value(results['liked_ratings'],'liked_ratings',k),
                            'known_dislike_rate_per_slot':results['liked_ratings']['known_dislike_rate_per_slot'],
                            'comparisons':comparisons}
    write_json(out/'summary.json',summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026,2027,2028])
    parser.add_argument('--revision', type=int, choices=[1,2], default=1)
    parser.add_argument('--audit-reference-root',type=Path,
                        help='Only run equal-grid implicit EASE audit against this completed revision2 directory')
    parser.add_argument('--pair-permutation-reference-root',type=Path,
                        help='Only run exact counterpart-permutation control against this completed revision2 directory')
    args = parser.parse_args()
    if args.pair_permutation_reference_root:
        audit_pair_permutation(args.source_root,args.ratings,args.items,args.out,args.pair_permutation_reference_root,args.seeds)
        return
    if args.audit_reference_root:
        audit_implicit_controls(args.source_root,args.ratings,args.items,args.out,args.audit_reference_root,args.seeds)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for seed in args.seeds:
        results[str(seed)] = run_seed(args.source_root/f'{seed}-EASE-1', args.ratings,
                                     args.items, args.out/str(seed), seed,revision=args.revision)
    write_json(args.out/'summary.json', results)


if __name__ == '__main__':
    main()
