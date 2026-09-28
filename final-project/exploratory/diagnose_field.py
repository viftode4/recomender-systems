"""Post-test exploratory field diagnosis using TRAIN and development only.

No optimization, model selection, or TEST access. This writes aggregate-only
diagnostics; the EASE compression is a TRAIN-derived diagnostic surrogate, not
a proposed ensemble or a new held-out result.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import logsumexp
import torch

from categorical_field import CategoricalEvidenceField
from exception_experiment import recommendations
from metrics import evaluate
from study import digest, grouped, read_pairs, write_json


def read_json(path):
    return json.loads(path.read_text())


def effective_spectrum(singular):
    energy = np.maximum(np.asarray(singular, dtype=float), 0) ** 2
    probability = energy / max(energy.sum(), 1e-30)
    positive = probability[probability > 0]
    cumulative = np.cumsum(probability)
    return {'energy_entropy_rank': float(np.exp(-np.sum(positive * np.log(positive)))),
            'energy_participation_rank': float(1 / max(np.sum(probability ** 2), 1e-30)),
            'components_for_90pct_energy': int(np.searchsorted(cumulative, .9) + 1),
            'components_for_99pct_energy': int(np.searchsorted(cumulative, .99) + 1),
            'energy_share': {str(k): float(cumulative[min(k, len(cumulative))-1])
                             for k in (1, 8, 16, 32, 64, 128)}}


def centered_svd(scores):
    values = scores[:, 1:].astype(float)
    column_mean, row_mean, grand = values.mean(0), values.mean(1), values.mean()
    centered = values - column_mean - row_mean[:, None] + grand
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    np.testing.assert_allclose((left * singular) @ right, centered, atol=2e-12, rtol=2e-12)
    return (left, singular, right, column_mean, row_mean, grand,
            {'centered_score_spectrum': effective_spectrum(singular),
             'interaction_energy_fraction_after_row_centering': float(
                 np.sum(centered ** 2) / np.sum((values - row_mean[:, None]) ** 2)),
             'scope': 'Double-centered finite full-catalog score matrix, including masked TRAIN positions; nonlinear scores need not have architecture-bounded linear rank.'})


def ranking(scores, users, items, observed, valid, chosen):
    recs = recommendations(scores, users, items, observed, 10)
    history = {u: {items[i] for i in np.flatnonzero(observed[j])} for j, u in enumerate(users)}
    counts = Counter(items[i] for _, i in np.argwhere(observed))
    truth = {u: grouped(valid)[u] for u in chosen}
    result = evaluate({u: recs[u] for u in sorted(chosen)}, truth, history, items[1:], counts, 10)
    return result['aggregate']


def inference_diagnostics(model, context, indices, expected_logits):
    metrics = {str(step): {} for step in range(model.steps)}
    exact_replay = True
    for offset in range(0, len(indices), 16):
        chosen = indices[offset:offset+16]
        ratings = torch.as_tensor(context[chosen], dtype=torch.long)
        with torch.no_grad():
            out = model(ratings, return_diagnostics=True)
            exact_replay &= np.array_equal(out['rating_logits'].numpy(), expected_logits[chosen])
            d = out['diagnostics']
            observed = ratings != 0
            eligible = ~observed
            eligible[:, 0] = False
            item = model.item_embedding.weight[None].expand(len(chosen), -1, -1)
            for step in range(model.steps):
                up, down = d['up'][:, step], d['down'][:, step]
                up_entropy = -(up.clamp_min(1e-30).log() * up).sum(-1)
                up_entropy /= observed.sum(-1).clamp_min(2).log()[:, None]
                down_entropy = -(down.clamp_min(1e-30).log() * down).sum(-1) / np.log(down.shape[-1])
                norm = up.square().sum(-1).sqrt()
                cosine = torch.bmm(up, up.transpose(1, 2)) / (norm[:, :, None] * norm[:, None, :]).clamp_min(1e-30)
                offdiag = ~torch.eye(up.shape[1], dtype=torch.bool)
                state = d['states'][:, step+1]
                decoder_pre = model.decoder[0](torch.cat((item, state, item * state), -1))[eligible]
                gates = d['fidelity'][:, step][observed]
                per_user_state_sd = torch.stack([state[j, eligible[j]].std(0).mean() for j in range(len(chosen))])
                additions = {
                    'normalized_up_entropy': up_entropy.flatten(),
                    'normalized_down_entropy_unseen': down_entropy[eligible],
                    'up_port_pair_cosine': cosine[:, offdiag].flatten(),
                    'mean_candidate_state_coordinate_sd': per_user_state_sd,
                    'decoder_tanh_abs_over_095_fraction': (decoder_pre.tanh().abs() > .95).float().flatten(),
                    'decoder_tanh_mean_derivative': (1-decoder_pre.tanh().square()).flatten(),
                    'source_gate': gates,
                    'source_gate_below_005_fraction': (gates < .05).float(),
                    'source_gate_above_095_fraction': (gates > .95).float(),
                }
                for name, values in additions.items():
                    acc = metrics[str(step)].setdefault(name, [0., 0])
                    acc[0] += float(values.double().sum())
                    acc[1] += values.numel()
    if not exact_replay:
        raise ValueError('Diagnostic selected-checkpoint predictions do not replay exactly')
    return {'users': len(indices), 'saved_logits_replay_exact': exact_replay,
            'steps': {step: {name: total/count for name, (total, count) in rows.items()}
                      for step, rows in metrics.items()}}


def run_seed(seed):
    directory = ROOT/'runs/joint-field-convergence-v1'/str(seed)
    ease = ROOT/'runs/research-v2'/f'{seed}-EASE-1'
    allowed_inputs = [directory/'training-categories.npz', directory/'cohorts.json',
                      directory/'valid.tsv', ease/'valid-scores.npz', directory/'protocol.json']
    with np.load(directory/'training-categories.npz', allow_pickle=False) as archive:
        users, items, context = archive['users'].tolist(), archive['items'].tolist(), archive['ratings']
    cohorts = read_json(directory/'cohorts.json')
    development = set(cohorts['development'])
    observed = context != 0
    valid = read_pairs(directory/'valid.tsv')
    ui, ii = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    assert all(not observed[ui[u], ii[i]] for u, i in valid)
    dev_indices = np.asarray([j for j, u in enumerate(users) if u in development])
    sample = np.sort(np.random.default_rng(seed+9101).choice(dev_indices, 128, replace=False))
    with np.load(ease/'valid-scores.npz', allow_pickle=False) as archive:
        assert archive['users'].tolist() == users and archive['items'].tolist() == items
        ease_scores = archive['scores'].astype(float)
    left, singular, right, cm, rm, grand, spectrum = centered_svd(ease_scores)
    result = {'seed': seed, 'users': len(users), 'development_users': len(development),
              'ease': {'score_diagnostics': spectrum, 'compression_development_all_observed': {}}}
    for rank in (0, 8, 16, 32, 64, 128, len(singular)):
        compressed = np.zeros_like(ease_scores)
        compressed[:, 1:] = cm + rm[:, None] - grand
        if rank:
            compressed[:, 1:] += (left[:, :rank] * singular[:rank]) @ right[:rank]
        result['ease']['compression_development_all_observed'][str(rank)] = ranking(
            compressed, users, items, observed, valid, development)
    for variant in ('adaptive', 'fixed_flow'):
        folder = directory/variant
        choice, grid, trace = (read_json(folder/name) for name in
                               ('selection.json', 'selection-grid.json', 'training-trace.json'))
        checkpoint = folder/choice['checkpoint']
        if digest(checkpoint) != choice['checkpoint_sha256']:
            raise ValueError('Selected checkpoint differs')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = CategoricalEvidenceField(**saved['config'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with np.load(folder/'selected-logits.npz', allow_pickle=False) as archive:
            assert archive['users'].tolist() == users and archive['items'].tolist() == items
            logits = archive['logits']
        scores = logsumexp(logits.astype(float), axis=-1)
        field_spectrum = centered_svd(scores)[-1]
        result[variant] = {
            'config': saved['config'], 'checkpoint_epoch': saved['epoch'],
            'parameters': sum(p.numel() for p in model.parameters()),
            'score_diagnostics': field_spectrum,
            'development_all_observed': ranking(scores, users, items, observed, valid, development),
            'selected_inference': inference_diagnostics(model, context, sample, logits),
            'meta_fit_checkpoint_curve': grid,
            'training_first20_mean': float(np.mean([r['macro_train_joint_probe_nll'] for r in trace[:20]])),
            'training_epoch281_300_mean': float(np.mean([r['macro_train_joint_probe_nll'] for r in trace[280:300]])),
            'training_last20_mean': float(np.mean([r['macro_train_joint_probe_nll'] for r in trace[-20:]])),
            'training_loss_caution': 'Changing TRAIN probe episodes and shorter contexts differ from validation inference; raw TRAIN/validation loss gaps do not isolate overfitting.'}
        allowed_inputs += [checkpoint, folder/'selected-logits.npz', folder/'selection.json',
                           folder/'selection-grid.json', folder/'training-trace.json']
    result['input_sha256'] = {p.relative_to(ROOT).as_posix(): digest(p) for p in allowed_inputs}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'exploratory/field-diagnosis-v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {'stage': 'post_test_exploratory_train_development_diagnosis',
            'test_labels_or_metrics_read': False, 'new_model_training': False,
            'seeds': [2026, 2027, 2028], 'rank_grid': [0, 8, 16, 32, 64, 128, 943],
            'diagnostic_users_per_seed': 128,
            'diagnostic_sample': 'Uniform sample of development users, generator seed split_seed+9101; no outcome selection.',
            'compression': 'SVD of double-centered TRAIN-fitted EASE full-catalog scores, retaining row/item means. Includes known-position scores in SVD; evaluated only on original TRAIN-unseen candidates.',
            'source_sha256': {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
                              'categorical_field.py': digest(ROOT/'categorical_field.py')}}
    write_json(args.out/'protocol.json', plan)
    results = {}
    for seed in plan['seeds']:
        results[str(seed)] = run_seed(seed)
        write_json(args.out/f'{seed}.json', results[str(seed)])
        print(seed, 'TRAIN/development diagnostic complete', flush=True)
    write_json(args.out/'aggregate.json', {'protocol': plan, 'seeds': results})


if __name__ == '__main__':
    main()
