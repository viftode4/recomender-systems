"""Training-support-preserving perturbations of observed negative labels.

These are sensitivity controls, not certified uniform conditional-null samples.
Only item column zero is padding; user row zero may contain a real user.
"""

from __future__ import annotations

import numpy as np


MAX_SWITCH_ATTEMPTS = 500_000


def _binary_matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f'{name} must be a two-dimensional binary matrix with padding column 0')
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f'{name} must contain only binary values')
    return array.astype(bool, copy=True)


def _fixed_support_pairs(support, row_group_codes):
    """Enumerate eligible row pairs using integer bitsets, not dense U²I work."""
    bitsets = []
    for row in support:
        bits = 0
        for item in np.flatnonzero(row):
            bits |= 1 << int(item)
        bitsets.append(bits)
    active = [u for u, bits in enumerate(bitsets) if bits.bit_count() >= 2]
    pairs = []
    for index, u in enumerate(active):
        for v in active[index + 1:]:
            if row_group_codes[u] != row_group_codes[v]:
                continue
            common = bitsets[u] & bitsets[v]
            if common.bit_count() < 2:
                continue
            items = []
            while common:
                bit = common & -common
                items.append(bit.bit_length() - 1)
                common ^= bit
            pairs.append((u, v, np.asarray(items, dtype=np.int64)))
    return pairs


def switch_negative_labels(negative, observed_nonpositive, seed,
                           attempts_per_negative=50, row_groups=None):
    """Perturb negative/neutral labels while exactly preserving both margins.

    ``observed_nonpositive`` contains only known TRAIN ratings <= 3, and
    ``negative`` only their <= 2 subset. The caller is responsible for supplying
    the intended training partition. Neither argument is mutated.

    Optional fixed ``row_groups`` restricts switches to users in the same group,
    preserving item margins inside every group. Groups used to separate decoder
    fitting rows must derive only from context, never from probe target labels.

    Choose a row pair uniformly from a list determined solely by fixed support,
    then two distinct common supported items uniformly. Swap only alternating
    checkerboards. This proposal is symmetric in the current label state;
    rejected proposals are counted. The fixed cap is 500,000 proposals, including
    rejections. Structural zeros may disconnect the state space, and no mixing
    time or uniform conditional-null claim is made.
    """
    original = _binary_matrix(negative, 'negative')
    support = _binary_matrix(observed_nonpositive, 'observed_nonpositive')
    if original.shape != support.shape:
        raise ValueError('negative and observed_nonpositive must have the same shape')
    if original[:, 0].any() or support[:, 0].any():
        raise ValueError('Item padding column 0 must be zero')
    if np.any(original & ~support):
        raise ValueError('Every negative must lie inside observed nonpositive support')
    if (isinstance(attempts_per_negative, (bool, np.bool_)) or
            not isinstance(attempts_per_negative, (int, np.integer)) or
            attempts_per_negative < 0):
        raise ValueError('attempts_per_negative must be a nonnegative integer')
    if row_groups is None:
        labels, group_codes = np.asarray(['all']), np.zeros(len(original), dtype=np.int64)
    else:
        groups = np.asarray(row_groups)
        if (groups.ndim != 1 or len(groups) != len(original) or
                groups.dtype.kind not in 'biufUS' or
                (groups.dtype.kind in 'f' and not np.isfinite(groups).all())):
            raise ValueError('row_groups must have one finite scalar label per user')
        labels, group_codes = np.unique(groups, return_inverse=True)

    result = original.copy()
    n_negative = int(original.sum())
    requested = n_negative * int(attempts_per_negative)
    attempt_budget = min(requested, MAX_SWITCH_ATTEMPTS)
    pairs = _fixed_support_pairs(support, group_codes)
    structurally_supported_rows = np.zeros(support.shape[0], dtype=bool)
    structurally_supported_items = np.zeros(support.shape[1], dtype=bool)
    initially_flippable_rows = np.zeros(support.shape[0], dtype=bool)
    initially_flippable_items = np.zeros(support.shape[1], dtype=bool)
    initially_flippable_pairs = 0
    for u, v, items in pairs:
        structurally_supported_rows[[u, v]] = True
        structurally_supported_items[items] = True
        differing = original[u, items] != original[v, items]
        first = original[u, items]
        if np.any(differing & first) and np.any(differing & ~first):
            initially_flippable_pairs += 1
            initially_flippable_rows[[u, v]] = True
            initially_flippable_items[items[differing]] = True

    attempts = accepted = 0
    touched_rows = np.zeros(support.shape[0], dtype=bool)
    touched_items = np.zeros(support.shape[1], dtype=bool)
    trace = []
    if pairs and n_negative and attempt_budget:
        rng = np.random.default_rng(seed)
        checkpoints = set(np.linspace(1, attempt_budget, min(10, attempt_budget),
                                      dtype=np.int64).tolist())
        for attempts in range(1, attempt_budget + 1):
            u, v, items = pairs[int(rng.integers(len(pairs)))]
            a = int(rng.integers(len(items)))
            b = int(rng.integers(len(items) - 1))
            b += b >= a
            i, j = int(items[a]), int(items[b])
            if (result[u, i] != result[u, j] and
                    result[u, i] == result[v, j] and
                    result[u, j] == result[v, i]):
                result[u, i] = not result[u, i]
                result[u, j] = not result[u, j]
                result[v, i] = not result[v, i]
                result[v, j] = not result[v, j]
                accepted += 1
                touched_rows[[u, v]] = True
                touched_items[[i, j]] = True
            if attempts in checkpoints:
                trace.append({'attempts': attempts, 'accepted': accepted,
                              'negative_overlap_fraction': float(
                                  np.count_nonzero(result & original) / n_negative)})

    group_diagnostics = []
    for code, label in enumerate(labels):
        rows = group_codes == code
        group_diagnostics.append({
            'group': label.item(),
            'users': int(rows.sum()),
            'negative_count': int(original[rows].sum()),
            'row_negative_counts_preserved': bool(np.array_equal(
                original[rows].sum(axis=1), result[rows].sum(axis=1))),
            'item_negative_counts_preserved': bool(np.array_equal(
                original[rows].sum(axis=0), result[rows].sum(axis=0))),
            'changed_label_count': int(np.count_nonzero(original[rows] != result[rows])),
        })
    invariants = {
        'row_negative_counts_preserved': bool(np.array_equal(
            original.sum(axis=1), result.sum(axis=1))),
        'item_negative_counts_preserved': bool(np.array_equal(
            original.sum(axis=0), result.sum(axis=0))),
        'negative_count_preserved': bool(result.sum() == n_negative),
        'inside_original_nonpositive_support': bool(not np.any(result & ~support)),
        'padding_zero': bool(not result[:, 0].any()),
        'group_item_negative_counts_preserved': all(
            group['item_negative_counts_preserved'] for group in group_diagnostics),
    }
    if not all(invariants.values()):
        raise RuntimeError('Negative-label switch invariant failed')
    changed = original != result
    overlap = int(np.count_nonzero(original & result))
    n_support = int(support.sum())
    diagnostics = {
        'control': 'fixed-margin observed-negative/neutral checkerboard perturbation',
        'seed': int(seed),
        'proposal': 'uniform fixed-support same-group eligible row pair, then uniform distinct common items',
        'sampling_claim': 'symmetric proposals; no global connectivity, mixing, or uniform-null guarantee',
        'attempts_per_negative': int(attempts_per_negative),
        'requested_attempts': requested,
        'proposal_cap': MAX_SWITCH_ATTEMPTS,
        'attempt_budget': attempt_budget,
        'attempts': attempts,
        'accepted': accepted,
        'acceptance_fraction': accepted / attempts if attempts else 0.0,
        'negative_count': n_negative,
        'observed_nonpositive_count': n_support,
        'initial_overlap': n_negative,
        'final_overlap': overlap,
        'initial_overlap_fraction': 1.0 if n_negative else None,
        'final_overlap_fraction': overlap / n_negative if n_negative else None,
        'moved_negative_fraction': (n_negative - overlap) / n_negative if n_negative else 0.0,
        'changed_label_fraction': float(changed.sum() / n_support) if n_support else 0.0,
        'fixed_support_row_pairs': len(pairs),
        'row_groups_supplied': row_groups is not None,
        'group_margins': group_diagnostics,
        'structurally_supported_rows': int(structurally_supported_rows.sum()),
        'structurally_supported_items': int(structurally_supported_items.sum()),
        'initially_flippable_row_pairs': initially_flippable_pairs,
        'initially_flippable_rows': int(initially_flippable_rows.sum()),
        'initially_flippable_items': int(initially_flippable_items.sum()),
        'rows_ever_switched': int(touched_rows.sum()),
        'items_ever_switched': int(touched_items.sum()),
        'rows_changed_at_end': int(np.count_nonzero(changed.any(axis=1))),
        'items_changed_at_end': int(np.count_nonzero(changed.any(axis=0))),
        'overlap_trace': trace,
        'invariants': invariants,
    }
    return result, diagnostics
