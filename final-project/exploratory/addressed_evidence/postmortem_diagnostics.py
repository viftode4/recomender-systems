"""Post-hoc, TRAIN-only diagnosis of the completed addressed study; no fitting.

Existing development aggregates are compared after verifying matching identities,
cohorts, source data and splits. No TEST file or final-result artifact is opened.
Only the new aggregate diagnostic output is written; existing seals stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from categorical_experiment import array_digest, masked_episode
from exploratory.addressed_evidence.model import AddressedEvidenceModel
from exploratory.addressed_evidence.run_experiment import (
    deterministic_runtime, restore, runtime_signature, source_hashes, tree_digest,
)
from field_reference_comparison import identity_digest
from joint_field_experiment import predict_logits

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_files(root, hashes):
    for name, expected in hashes.items():
        path = root / name
        name_lower = path.name.lower()
        if (not path.resolve().is_relative_to(root.resolve())
                or name_lower == 'test' or name_lower.startswith(('test.', 'test-', 'final-'))):
            raise ValueError('Unexpected diagnostic input')
        if digest(path) != expected:
            raise ValueError(f'Changed sealed artifact: {path}')


def describe_terms(model, context, training, activity_group):
    """All statistics use the SAME original TRAIN-unseen candidate support.

    Center every user's term over eligible item/category outcomes before RMS,
    removing the common shift that has no effect on joint probabilities. Split
    this into across-item means and within-item category contrasts as well.
    """
    totals = {name: {'users': 0, 'outcomes': 0, 'items': 0, 'history_sum': 0.,
                    'neighbor_sum': 0., 'neighbor_pairs_sum': 0., 'cross_sums': {}, 'terms': {}}
              for name in ('all', 'activity_q1', 'activity_q2', 'activity_q3', 'activity_q4')}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), 16):
            x = context[start:start+16]
            result = model(torch.as_tensor(x, dtype=torch.long), return_diagnostics=True)
            d = result['diagnostics']
            direct = d['direct'].numpy().astype(float)
            pair = (d['distinct_pair_feature'] * d['pair_coefficients'][None]).numpy().astype(float)
            if model.variant == 'additive':
                pair[:] = 0.
            terms = {'direct': direct, 'applied_pair': pair,
                     'bias': np.broadcast_to(model.bias.detach().numpy().astype(float), direct.shape),
                     'total_logits': result['rating_logits'].numpy().astype(float)}
            neighbors = d['observed_neighbors'].numpy().astype(float)
            for local in range(len(x)):
                row = start + local
                eligible = training[row] == 0
                eligible[0] = False
                n_items = int(eligible.sum())
                values, centered = {}, {}
                for name, term in terms.items():
                    z = term[local, eligible]
                    global_centered = z-z.mean()
                    centered[name] = global_centered
                    category_centered = z-z.mean(axis=-1, keepdims=True)
                    item_centered = z.mean(axis=-1)-z.mean()
                    values[name] = {'raw_ss': float(np.square(z).sum()),
                                    'joint_centered_ss': float(np.square(global_centered).sum()),
                                    'within_item_category_ss': float(np.square(category_centered).sum()),
                                    'item_mean_ss': float(np.square(item_centered).sum())*5}
                    assert np.isclose(values[name]['joint_centered_ss'],
                        values[name]['within_item_category_ss']+values[name]['item_mean_ss'], rtol=1e-10, atol=1e-9)
                for group in ('all', f'activity_q{activity_group[row]+1}'):
                    total = totals[group]
                    total['users'] += 1
                    total['outcomes'] += n_items*5
                    total['items'] += n_items
                    total['history_sum'] += np.count_nonzero(x[local])
                    counts = neighbors[local, eligible]
                    total['neighbor_sum'] += counts.sum()
                    total['neighbor_pairs_sum'] += (counts*(counts-1)/2).sum()
                    for name, stats in values.items():
                        target = total['terms'].setdefault(name, {key: 0. for key in stats})
                        for key, value in stats.items():
                            target[key] += value
                    for left, right in (('direct', 'applied_pair'), ('direct', 'bias'), ('applied_pair', 'bias')):
                        key = left+'__'+right
                        total['cross_sums'][key] = total['cross_sums'].get(key, 0.)+float((centered[left]*centered[right]).sum())
    summaries = {}
    for group, row in totals.items():
        summaries[group] = {
            'users': row['users'], 'candidate_category_count': row['outcomes'],
            'mean_visible_history_count': row['history_sum']/row['users'],
            'mean_observed_neighbors': row['neighbor_sum']/row['items'],
            'mean_distinct_observed_neighbor_pairs': row['neighbor_pairs_sum']/row['items'],
            'joint_centered_covariances': {key: value/row['outcomes'] for key, value in row['cross_sums'].items()},
            'terms': {name: {key.replace('_ss', '_rms'): float(np.sqrt(value/row['outcomes']))
                              for key, value in stats.items()} for name, stats in row['terms'].items()}}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refuse diagnostic overwrite')
    deterministic_runtime()
    sources = source_hashes()
    research = ROOT/'runs/addressed-evidence-v1'
    evidence = ROOT/'exploratory/addressed_evidence/results-v1'
    reference_root = ROOT/'runs/field-reference-v1'
    reference_evidence = ROOT/'evidence/field-reference-v1'
    check_files(evidence, read(evidence/'SHA256.json'))
    provenance = read(evidence/'provenance.json')
    assert provenance['source_sha256'] == sources and provenance['runtime'] == runtime_signature()
    reference_manifest = read(reference_root/'manifest.json')
    assert reference_manifest['status'] == 'complete' and not reference_manifest['test_read']
    check_files(reference_evidence, reference_manifest['evidence_sha256'])
    check_files(ROOT, reference_manifest['source_sha256'])
    addressed, references = read(evidence/'aggregates.json'), read(reference_evidence/'aggregates.json')
    result = {'schema_version': 1, 'scope': 'Post-hoc TRAIN-context diagnostics and reused-development aggregate comparison; no fit or TEST access.',
        'test_read': False, 'test_results_read': False, 'new_fit': False, 'fresh_confirmation': False,
        'diagnostic_definition': {
            'contexts': 'Nested identity-only epoch-1 masks with 40/60/80 percent kept, plus full TRAIN; floor rounding is inherited from masked_episode.',
            'candidates': 'Original TRAIN-unseen nonpadding items, identical for every context size; no rating likelihood evaluated on these context interventions.',
            'centering': 'Per-user mean over all fixed eligible item/category outcomes; pooled outcome-weighted RMS.',
            'activity_groups': 'TRAIN history-count quartiles via numpy quantile and searchsorted(side=right); ties may give unequal group sizes.',
            'checkpoint_roles': 'best is selected by the original meta-fit rule; latest is an overfitting diagnostic, never a new selected model.'},
        'runtime': runtime_signature(), 'source_sha256': sources,
        'diagnostic_source_sha256': digest(Path(__file__)),
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            evidence/'aggregates.json', evidence/'provenance.json', evidence/'SHA256.json',
            reference_evidence/'aggregates.json', reference_evidence/'protocol.json', reference_root/'manifest.json')},
        'seeds': {}}
    fixed_denominators = ('all_observed_users', 'liked_ratings_users', 'validation_observations', 'validation_likes', 'validation_dislikes', 'recommendation_slots')
    for seed in (2026, 2027, 2028):
        directory = research/str(seed)
        manifest = read(directory/'manifest.json')
        assert digest(directory/'manifest.json') == provenance['seeds'][str(seed)]['manifest_sha256']
        assert manifest['status'] == 'complete' and not manifest['test_read']
        check_files(directory, manifest['output_sha256'])
        reference = references['seeds'][str(seed)]
        assert reference['data_sha256'] == manifest['data_sha256']
        assert all(reference['split_sha256'][part] == manifest['split_sha256'][part] for part in ('train', 'valid'))
        assert digest(directory/'cohorts.json') == reference['cohort_file_sha256']
        assert digest(reference_root/str(seed)/'details.json') == reference['details_sha256']
        with np.load(directory/'training-categories.npz', allow_pickle=False) as archive:
            training = archive['ratings']
            users, items = archive['users'].tolist(), archive['items'].tolist()
        assert array_digest(training) == manifest['train_categories_sha256']
        assert identity_digest(users, items) == reference['ordered_identity_sha256']
        counts = np.count_nonzero(training, axis=1)
        boundaries = np.quantile(counts, [.25, .5, .75])
        groups = np.searchsorted(boundaries, counts, side='right')
        contexts = {str(f): masked_episode(training, seed, 1, f)[0] for f in (.4, .6, .8)}
        contexts['1.0'] = training
        seed_row = {'manifest_sha256': digest(directory/'manifest.json'),
            'reference_matching_checks': 'Data, TRAIN/validation splits, ordered catalog/user IDs, exact cohort file and evaluation denominators match.',
            'activity_history_count_boundaries': boundaries.tolist(), 'variants': {},
            'references': {name: reference['rows'][name] for name in ('EASE', 'SLIMElastic', 'PositiveEASE')}}
        for variant in ('additive', 'pair'):
            folder = directory/variant
            aggregate = addressed['seeds'][str(seed)]['models'][variant]
            for endpoint, adapter in (('all_observed', 'all_observed_adapter'), ('liked_ratings', 'liked_record_adapter')):
                left = aggregate['ranking'][adapter]['denominators']
                right = reference['rows']['EASE']['endpoints'][endpoint]['denominators']
                assert all(left[key] == right[key] for key in fixed_denominators)
            selected = read(folder/'selection.json')
            model, _ = restore(folder/'best.pt')
            assert array_digest(predict_logits(model, training)) == selected['logits_array_sha256']
            latest = torch.load(folder/'latest.pt', map_location='cpu', weights_only=True)
            expected = latest.pop('integrity_sha256')
            assert tree_digest(latest) == expected
            assert latest['best_logits_array_sha256'] == selected['logits_array_sha256']
            snapshots = {'best': model, 'latest': AddressedEvidenceModel(**latest['snapshot']['config'])}
            snapshots['latest'].load_state_dict(latest['snapshot']['state_dict'], strict=True)
            trace = read(folder/'training-trace.json')
            grid = read(folder/'selection-grid.json')
            row = {'selection': selected, 'development_aggregate': aggregate,
                'curves': [{**entry, 'macro_train_joint_probe_nll': trace[entry['epoch']-1]['macro_train_joint_probe_nll']} for entry in grid],
                'checkpoints': {}}
            for role, current in snapshots.items():
                gamma = current.pair_bound*current.pair_raw.detach().tanh().numpy()[1:]
                fractions = ('.4', '.6', '.8', '1.0') if variant == 'pair' and role == 'best' else ('.8', '1.0')
                fractions = [str(float(f)) for f in fractions]
                row['checkpoints'][role] = {
                    'epoch': selected['epoch'] if role == 'best' else latest['snapshot']['epoch'],
                    'checkpoint_sha256': digest(folder/('best.pt' if role == 'best' else 'latest.pt')),
                    'pair_coefficient_mean_abs': float(np.abs(gamma).mean()),
                    'pair_coefficient_saturation_above_point95': float(np.mean(np.abs(gamma)>.95)),
                    'pair_coefficient_derivative_quantiles': np.quantile(1-(gamma/current.pair_bound)**2, [0., .25, .5, .75, 1.]).tolist(),
                    'allocated_potential_rms': float(current.potentials.detach().double().square().mean().sqrt()),
                    'contexts': {f: describe_terms(current, contexts[f], training, groups) for f in fractions}}
            seed_row['variants'][variant] = row
            print(f'Completed TRAIN-only diagnostics: seed {seed}, {variant}', flush=True)
        result['seeds'][str(seed)] = seed_row
    assert source_hashes() == sources
    check_files(evidence, read(evidence/'SHA256.json'))
    from package_project import check_aggregate_only
    check_aggregate_only(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
