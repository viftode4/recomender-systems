"""Compare completed rebuilt expert predictions with frozen numerical references.

No models are fit and no VALID/TEST relevance is read. Historical score arrays
are opened only after every fresh prediction has a completed manifest. Published
output contains aggregate comparisons and hashes, never user/item identities,
prediction arrays, or absolute paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_array_hash(array):
    value = np.ascontiguousarray(array, dtype='<f8')
    header = json.dumps({'shape': list(value.shape), 'dtype': '<f8'}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + value.tobytes()).hexdigest()


def statistics(reference, rebuilt, observed, k=10):
    reference, rebuilt = np.asarray(reference, dtype=float), np.asarray(rebuilt, dtype=float)
    require(reference.ndim == 2 and reference.shape == rebuilt.shape == observed.shape,
            'Prediction/mask shape mismatch')
    require(np.isfinite(reference).all() and np.isfinite(rebuilt).all(), 'Nonfinite prediction')
    mask = np.asarray(observed, dtype=bool).copy()
    mask[:, 0] = True
    difference = rebuilt - reference
    absolute = np.abs(difference)
    eligible = ~mask
    require(np.all(eligible.sum(axis=1) >= k), 'Insufficient eligible candidates')
    full_position_differences = full_users = displacement_sum = displacement_max = 0
    top_users = top_positions = top_set_users = top_set_substitutions = 0
    for user in range(len(reference)):
        candidates = np.flatnonzero(eligible[user])
        first = np.argsort(-reference[user, candidates], kind='stable')
        second = np.argsort(-rebuilt[user, candidates], kind='stable')
        changed = first != second
        full_users += int(changed.any())
        full_position_differences += int(changed.sum())
        first_positions, second_positions = np.empty_like(first), np.empty_like(second)
        first_positions[first] = np.arange(len(first))
        second_positions[second] = np.arange(len(second))
        displacements = np.abs(first_positions - second_positions)
        displacement_sum += int(displacements.sum())
        displacement_max = max(displacement_max, int(displacements.max(initial=0)))
        top_users += int((first[:k] != second[:k]).any())
        top_positions += int((first[:k] != second[:k]).sum())
        substitutions = len(set(first[:k]) - set(second[:k]))
        top_set_users += int(substitutions > 0)
        top_set_substitutions += substitutions
    return {
        'users': int(reference.shape[0]), 'catalog_items_including_padding': int(reference.shape[1]),
        'all_score_cells': int(reference.size), 'eligible_score_cells': int(eligible.sum()),
        'scores_numerically_identical': bool(np.array_equal(reference, rebuilt)),
        'all_scores': {'different_cells': int(np.count_nonzero(difference)),
                       'max_absolute_error': float(absolute.max(initial=0)),
                       'root_mean_squared_error': float(np.sqrt(np.mean(difference ** 2)))},
        'eligible_scores': {'different_cells': int(np.count_nonzero(difference[eligible])),
                            'max_absolute_error': float(absolute[eligible].max(initial=0)),
                            'root_mean_squared_error': float(np.sqrt(np.mean(difference[eligible] ** 2)))},
        'eligible_full_ranking': {'users_with_any_order_difference': full_users,
                                 'different_order_positions': full_position_differences,
                                 'maximum_absolute_item_rank_displacement': displacement_max,
                                 'mean_absolute_item_rank_displacement': displacement_sum / int(eligible.sum())},
        f'top{k}': {'users_with_order_difference': top_users, 'different_order_positions': top_positions,
                   'users_with_set_difference': top_set_users, 'total_item_substitutions': top_set_substitutions,
                   'all_users_order_identical': top_users == 0},
    }


def self_check():
    original = np.array([[99., 4., 3., 2., 1.], [99., 7., 3., 3., 0.]])
    changed = original.copy()
    changed[0, 0] = -99.  # PAD must have no ranking effect.
    changed[1, 1] = -99.  # Seen item must have no ranking effect.
    changed[0, 1:3] = [3., 4.]
    seen = np.zeros(original.shape, dtype=bool)
    seen[1, 1] = True
    value = statistics(original, changed, seen, k=2)
    require(value['eligible_full_ranking'] == {
        'users_with_any_order_difference': 1, 'different_order_positions': 2,
        'maximum_absolute_item_rank_displacement': 1, 'mean_absolute_item_rank_displacement': 2/7},
        'Independent rank-displacement example failed')
    require(value['top2']['users_with_order_difference'] == 1
            and value['top2']['users_with_set_difference'] == 0
            and value['eligible_scores']['max_absolute_error'] == 1.
            and np.isclose(value['eligible_scores']['root_mean_squared_error'], np.sqrt(2/7)),
            'Independent score/mask example failed')
    replacement = original.copy(); replacement[0, 3] = 10.
    value = statistics(original, replacement, seen, k=2)
    require(value['top2']['total_item_substitutions'] == 1, 'Top-k substitution example failed')


def audit(rebuild, reference, output):
    rebuild, reference, output = Path(rebuild), Path(reference), Path(output)
    require(not output.exists(), 'Refusing to overwrite comparison evidence')
    self_check()
    tick = time.monotonic()
    recipe = read_json(rebuild / 'rebuild_recipe.json')
    started = read_json(rebuild / 'REBUILD-STARTED.json')
    require(digest(rebuild / 'rebuild_recipe.json') == started['recipe_sha256'], 'Rebuild recipe changed')
    require([record['seed'] for record in recipe['seeds']] == [2026, 2027, 2028], 'Unexpected seeds')
    prepared = []
    # This entire pass finishes before any historical prediction array is opened.
    for seed_record in recipe['seeds']:
        seed = seed_record['seed']
        require(len(seed_record['experts']) == 15, 'Incomplete expert recipe')
        require([expert['name'] for expert in seed_record['experts']] == seed_record['expert_order'],
                'Recipe expert order differs')
        for expert in seed_record['experts']:
            fresh = rebuild / 'experts' / str(seed) / expert['run_directory_basename']
            manifest = read_json(fresh / 'manifest.json')
            require(manifest['status'] == 'complete' and manifest.get('test_evaluated') is False,
                    'Require completed fresh predictions without TEST evaluation')
            require(manifest['model'] == expert['model'], 'Rebuilt expert identity differs')
            require(manifest['split_sha256'] == seed_record['split_sha256']
                    and manifest['data_sha256'] == seed_record['data_sha256'], 'Split/data identity differs')
            require(digest(fresh / 'train.tsv') == seed_record['expected_train_sha256'], 'Fresh TRAIN split differs')
            require(digest(fresh / 'ids.json') == seed_record['expected_ids_sha256'], 'Fresh source catalog differs')
            require((fresh / 'valid-scores.npz').is_file(), 'Fresh prediction file missing')
            prepared.append((seed_record, expert, fresh, digest(fresh / 'manifest.json'),
                             digest(fresh / 'valid-scores.npz')))
    require(len(prepared) == 45, 'Require all 45 fresh source predictions')
    rows, reference_fingerprints = [], {}
    for seed_record in recipe['seeds']:
        seed = seed_record['seed']
        historical = reference / str(seed)
        require(digest(historical / 'freeze.json') == seed_record['freeze_sha256'], 'Historical bundle identity differs')
        frozen = read_json(historical / 'freeze.json')
        require(frozen['expert_order'] == seed_record['expert_order'], 'Historical expert order differs')
        require(digest(historical / 'frozen.npz') == frozen['artifacts_sha256']['frozen.npz'],
                'Historical prediction archive hash differs')
        require(digest(historical / 'train.tsv') == seed_record['expected_train_sha256'], 'Reference TRAIN differs')
        reference_fingerprints[str(seed)] = {
            'freeze_sha256': digest(historical / 'freeze.json'),
            'frozen_archive_sha256': digest(historical / 'frozen.npz'),
            'train_sha256': digest(historical / 'train.tsv')}
        with np.load(historical / 'frozen.npz', allow_pickle=False) as archive:
            original_users, original_items = archive['users'].tolist(), archive['items'].tolist()
            original_scores, observed = archive['raw_scores'], archive['train_mask']
        require(len(original_users) == len(set(original_users)) == 943
                and len(original_items) == len(set(original_items)) == 1683
                and original_items[0] == '[PAD]', 'Unexpected historical IDs')
        require(original_scores.shape == (15, 943, 1683), 'Unexpected historical score shape')
        for _, expert, fresh, manifest_hash, prediction_hash in [entry for entry in prepared if entry[0]['seed'] == seed]:
            require(digest(fresh / 'manifest.json') == manifest_hash
                    and digest(fresh / 'valid-scores.npz') == prediction_hash, 'Fresh predictions changed during audit')
            with np.load(fresh / 'valid-scores.npz', allow_pickle=False) as archive:
                fresh_users, fresh_items, scores = archive['users'].tolist(), archive['items'].tolist(), archive['scores']
            require(len(fresh_users) == len(set(fresh_users)) and set(fresh_users) == set(original_users)
                    and len(fresh_items) == len(set(fresh_items)) and set(fresh_items) == set(original_items),
                    'Fresh and historical prediction ID sets differ')
            require(scores.shape == (len(fresh_users), len(fresh_items)), 'Fresh score shape differs')
            ui, ii = {u: r for r, u in enumerate(fresh_users)}, {i: c for c, i in enumerate(fresh_items)}
            aligned = scores[np.ix_([ui[user] for user in original_users], [ii[item] for item in original_items])]
            target = original_scores[frozen['expert_order'].index(expert['name'])]
            stats = statistics(target, aligned, observed)
            rows.append({'seed': seed, 'expert': expert['name'], 'model': expert['model'], 'kind': expert['kind'],
                'rebuilt_manifest_sha256': manifest_hash,
                'rebuilt_prediction_file_sha256': prediction_hash,
                'original_prediction_file_sha256': expert['expected_artifacts_sha256']['valid-scores.npz'],
                'prediction_file_bytes_identical': prediction_hash == expert['expected_artifacts_sha256']['valid-scores.npz'],
                'original_score_array_canonical_sha256': canonical_array_hash(target),
                'rebuilt_aligned_score_array_canonical_sha256': canonical_array_hash(aligned),
                'source_user_order_already_identical': fresh_users == original_users,
                'source_item_order_already_identical': fresh_items == original_items,
                **stats})
        print(json.dumps({'seed': seed, 'experts_compared': sum(row['seed'] == seed for row in rows)}), flush=True)
    for _, _, fresh, manifest_hash, prediction_hash in prepared:
        require(digest(fresh / 'manifest.json') == manifest_hash
                and digest(fresh / 'valid-scores.npz') == prediction_hash, 'Fresh predictions changed before receipt')
    payload = {
        'schema_version': 1, 'status': 'complete', 'scope': 'Independent numerical and ranking agreement of 45 completed rebuilt experts',
        'verifier_source_sha256': digest(Path(__file__)), 'rebuild_recipe_sha256': started['recipe_sha256'],
        'rebuild_started_sha256': digest(rebuild / 'REBUILD-STARTED.json'),
        'original_reference_sha256': reference_fingerprints,
        'all_fresh_manifests_completed_before_reference_arrays_opened': True,
        'self_check_passed': True, 'validation_relevance_read': False, 'original_test_read': False,
        'valid_mask_array_read': False, 'cause_of_any_numerical_drift': 'Not established by this comparison.',
        'definitions': {
            'alignment': 'Equal user/item token sets, explicitly reordered to the original frozen catalog; no IDs are published.',
            'numeric_scope': 'All saved score cells including PAD and seen items; eligible-only statistics separately exclude both.',
            'ranking_scope': 'All 943 users per expert; candidates exclude TRAIN observations and PAD, retaining VALID eligibility.',
            'tie_rule': 'Descending score then original catalog index, using stable sorting.',
            'full_ranking_difference': 'Different positions in the complete eligible ordering, plus per-item absolute rank displacement.',
            'top10_substitution': 'Number of original top10 items missing from rebuilt top10, summed over users.',
            'canonical_array_hash': 'SHA256 of sorted-key JSON shape/dtype metadata, NUL, and C-order little-endian float64 values.',
            'file_hash_interpretation': 'NPZ byte equality is distinct from numerical array equality and recommendation equality.',
            'no_accuracy_claim': 'No relevance labels are opened; identical top10 order implies identical top10 rank metrics under any fixed truth.'},
        'summary': {
            'experts_compared': len(rows), 'users_per_expert': 943,
            'prediction_files_byte_identical': sum(row['prediction_file_bytes_identical'] for row in rows),
            'score_arrays_numerically_identical': sum(row['scores_numerically_identical'] for row in rows),
            'experts_with_identical_full_eligible_rankings': sum(row['eligible_full_ranking']['users_with_any_order_difference'] == 0 for row in rows),
            'experts_with_identical_top10_for_all_users': sum(row['top10']['all_users_order_identical'] for row in rows),
            'expert_user_pairs_with_top10_order_difference': sum(row['top10']['users_with_order_difference'] for row in rows),
            'expert_user_pairs_with_top10_set_difference': sum(row['top10']['users_with_set_difference'] for row in rows),
            'largest_absolute_score_difference': max(row['all_scores']['max_absolute_error'] for row in rows),
            'largest_eligible_score_rmse': max(row['eligible_scores']['root_mean_squared_error'] for row in rows)},
        'experts': rows, 'elapsed_seconds': time.monotonic() - tick}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rebuild, args.reference, args.out)
    print(json.dumps(result['summary'], sort_keys=True))
