"""Rebuild the three fixed research experts from TRAIN ratings only.

This is reproduction of previously selected settings, not model selection.
The historical implementation hashes remain in the public recipe; a separate
comparison establishes whether current mathematical helpers replay their scores.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from exception_experiment import export_run
from exception_model import (linear_rating_scores, neighborhood_scores,
                             rating_matrices, representations)
from study import load_genres, read_pairs


METHODS = {'PositiveEASE': 'positive_ease',
           'SignedChannelsLinear': 'signed_channels',
           'ContrastTransfer': 'contrast_transfer'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_ratings(path, train_pairs):
    """Filter by TRAIN identities before interpreting any rating value."""
    allowed = set(train_pairs)
    if len(allowed) != len(train_pairs):
        raise ValueError('Duplicate TRAIN identities')
    found = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = (row['user_id:token'], row['item_id:token'])
            if pair not in allowed:
                continue
            if pair in found:
                raise ValueError('Duplicate TRAIN rating')
            value = float(row['rating:float'])
            if not np.isfinite(value) or not 1 <= value <= 5:
                raise ValueError('Invalid TRAIN rating')
            found[pair] = value
    if set(found) != allowed:
        raise ValueError('Missing TRAIN ratings')
    return [(user, item, found[user, item]) for user, item in train_pairs]


def rebuild_research(source, ratings, items_metadata, entries, output_root):
    """Return fixed research export directories in recipe order.

    ``entries`` contains name, model, and the historically selected settings.
    ``source`` is the freshly reproduced EASE run. Its VALID file is copied as
    an opaque input artifact by the existing exporter; no VALID outcome is
    parsed here, and no TEST file is opened.
    """
    source, ratings = Path(source), Path(ratings)
    items_metadata, output_root = Path(items_metadata), Path(output_root)
    manifest = json.loads((source / 'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or
            manifest.get('model') != 'EASE' or
            manifest.get('test_evaluated') is not False or
            manifest.get('validation_used_for_training') is not False):
        raise ValueError('Require a completed fixed-budget validation-only EASE source')
    if digest(source / 'train.tsv') != manifest['split_sha256']['train']:
        raise ValueError('TRAIN source hash mismatch')
    for path in (ratings, items_metadata):
        if digest(path) != manifest['data_sha256'].get(path.name):
            raise ValueError('Dataset differs from EASE source')
    names = [entry['name'] for entry in entries]
    if len(names) != len(set(names)) or not entries:
        raise ValueError('Require distinct fixed research exports')
    for entry in entries:
        name = entry['name']
        if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..') or
                entry['model'] not in METHODS):
            raise ValueError('Unknown research model or unsafe export name')
        if (output_root / name).exists():
            raise FileExistsError(output_root / name)
        if entry['settings'].get('model') != METHODS[entry['model']]:
            raise ValueError('Fixed model/settings mismatch')
        if entry['settings'].get('seed') != manifest['settings']['seed']:
            raise ValueError('Fixed seed differs from EASE source')
        if entry['settings'].get('selection_cohort') != 'meta_fit':
            raise ValueError('Historical settings were not selected on meta-fit')
        if entry['settings'].get('selection_objective') != 'all_observed':
            raise ValueError('Historical settings use another relevance objective')
    with np.load(source / 'valid-scores.npz', allow_pickle=False) as archive:
        # Only score-aligned IDs are needed, never the existing model scores.
        users, items = archive['users'].tolist(), archive['items'].tolist()
    if (not items or items[0] != '[PAD]' or len(set(items)) != len(items) or
            len(set(users)) != len(users) or '[PAD]' in users):
        raise ValueError('Invalid source score catalogs')
    train_pairs = read_pairs(source / 'train.tsv')
    if {user for user, _ in train_pairs} != set(users):
        raise ValueError('TRAIN users differ from source score IDs')
    if {item for _, item in train_pairs} - set(items[1:]):
        raise ValueError('TRAIN item absent from source catalog')
    training = training_ratings(ratings, train_pairs)
    signed, _ = rating_matrices(users, items, training)
    genres, _ = load_genres(items_metadata, items)
    geometry = None
    outputs = []
    for entry in entries:
        settings = dict(entry['settings'])
        method = METHODS[entry['model']]
        if method == 'contrast_transfer':
            if geometry is None:
                geometry = representations(signed, genres, dimensions=32, features=128,
                                           seed=int(settings['seed']))
            scores = neighborhood_scores(geometry.similarities[method], signed,
                                         neighbors=int(settings['neighbors']),
                                         dislike_weight=float(settings['dislike_weight']))
        else:
            scores = linear_rating_scores(signed, float(settings['penalty']), method)
        if scores.shape != (len(users), len(items)) or not np.isfinite(scores).all():
            raise ValueError('Invalid regenerated research scores')
        settings['fixed_historical_configuration_replay'] = True
        settings['selection_metric_recomputed'] = False
        # A recipe may retain original diagnostics, but never label them as a
        # metric measured by this fixed-configuration replay.
        if 'selection_ndcg' in settings:
            settings['historical_selection_ndcg'] = settings.pop('selection_ndcg')
        output = output_root / entry['name']
        export_run(output, source, users, items, scores, manifest, entry['model'], settings)
        exported = json.loads((output / 'manifest.json').read_text())
        exported['protocol'] = (
            'Fixed historical configuration replay. TRAIN ratings >=4 likes, <=2 '
            'dislikes, 3 neutral. No selection or VALID/TEST outcome parsing by this helper.')
        exported['rebuild'] = {
            'source': 'coursework_completion/rebuild_research.py',
            'source_sha256': digest(Path(__file__)),
            'training_ratings_only': True,
            'validation_file_copied_without_parsing': True,
            'original_test_file_opened': False,
            'historical_source_identity_claimed': False,
        }
        (output / 'manifest.json').write_text(json.dumps(exported, indent=2,
                                                       sort_keys=True, allow_nan=False) + '\n')
        outputs.append(output)
    return outputs
