"""TRAIN-only inputs, deterministic masked episodes and deferred DEV outcomes.

Validation streams are filtered by user before the item field is interpreted.
Only a caller-supplied global selection-barrier verifier can unlock DEV values.
No function opens TEST, cached predictions or cached training-category arrays.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from categorical_experiment import array_digest
from exploratory.categorical_reconstruction.run_experiment import load_train
from field_reference_comparison import identity_digest
from study import digest, grouped, partition_users

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "runs/research-v2"
DEFAULT_CATEGORICAL = ROOT / "runs/categorical-reconstruction-v1"
DEFAULT_RATINGS = ROOT.parent.parent / "recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter"


def _json(path):
    return json.loads(Path(path).read_text())


def _checked_hash(path, expected):
    if digest(Path(path)) != expected:
        raise ValueError(f"Input signature mismatch: {Path(path).name}")


def read_user_pairs(path, allowed_users, *, real_items, train_pairs):
    """Interpret item identities only after the user is in the allowed cohort."""
    allowed_users, real_items, train_pairs = set(allowed_users), set(real_items), set(train_pairs)
    pairs, seen = [], set()
    with Path(path).open() as stream:
        if stream.readline().rstrip("\r\n") != "user_id\titem_id":
            raise ValueError("Unexpected validation pair header")
        for line in stream:
            user, separator, remainder = line.rstrip("\r\n").partition("\t")
            if user not in allowed_users:
                continue
            if not separator or not remainder or "\t" in remainder:
                raise ValueError("Malformed allowed-cohort pair")
            pair = user, remainder
            if remainder not in real_items or pair in seen or pair in train_pairs:
                raise ValueError("Invalid, duplicate or TRAIN-overlapping allowed-cohort pair")
            seen.add(pair)
            pairs.append(pair)
    if {user for user, _ in pairs} != allowed_users:
        raise ValueError("Missing allowed-cohort validation user")
    return pairs


def load_inputs(seed, *, source_root=DEFAULT_SOURCE, ratings=DEFAULT_RATINGS,
                categorical_root=DEFAULT_CATEGORICAL, load_meta=True):
    """Load original TRAIN and optionally meta-fit pair identities, never DEV pairs.

    The categorical input signature is a hash-only anchor. Categories are rebuilt
    from raw TRAIN records, not accepted from a cache. ``load_meta=False`` does
    not open either validation pair file, even for hashing.
    """
    seed = int(seed)
    source, cohort = Path(source_root) / f"{seed}-EASE-1", Path(categorical_root) / str(seed)
    ratings = Path(ratings)
    users, items, train_pairs, original_categories, manifest = load_train(source, ratings)
    meta_users, dev_users = partition_users(users, seed)
    expected_cohorts = {"meta_fit": sorted(meta_users), "development": sorted(dev_users)}
    if _json(cohort / "cohorts.json") != expected_cohorts:
        raise ValueError("Original user cohorts differ")
    signature = {
        "source_manifest_sha256": digest(source / "manifest.json"),
        "data_sha256": manifest["data_sha256"],
        "split_sha256": {key: manifest["split_sha256"][key] for key in ("train", "valid")},
        "ids_sha256": digest(source / "ids.json"),
        "ordered_identity_sha256": identity_digest(users, items),
        "cohort_file_sha256": digest(cohort / "cohorts.json"),
        "training_categories_sha256": array_digest(original_categories),
    }
    if signature != _json(cohort / "input-signature.json"):
        raise ValueError("Original categorical input signature differs")
    _checked_hash(cohort / "train.tsv", signature["split_sha256"]["train"])
    paths = {"source_manifest": source / "manifest.json", "train": source / "train.tsv",
             "ids": source / "ids.json", "ratings": ratings,
             "cohorts": cohort / "cohorts.json", "categorical_signature": cohort / "input-signature.json"}
    input_hashes = {name: digest(path) for name, path in paths.items()}
    meta_pairs = []
    if load_meta:
        _checked_hash(source / "valid.tsv", signature["split_sha256"]["valid"])
        _checked_hash(cohort / "valid.tsv", signature["split_sha256"]["valid"])
        meta_pairs = read_user_pairs(source / "valid.tsv", meta_users,
                                    real_items=items[1:], train_pairs=train_pairs)
        input_hashes["valid"] = signature["split_sha256"]["valid"]
    categories = np.asarray(original_categories, dtype=np.uint8)
    categories.setflags(write=False)
    observed = categories != 0
    observed.setflags(write=False)
    return {"seed": seed, "users": users, "items": items, "categories": categories,
            "train_observed": observed, "train_pairs": train_pairs,
            "meta_users": meta_users, "dev_users": dev_users,
            "meta_validpairs": meta_pairs, "meta_truth": dict(grouped(meta_pairs)),
            "signature": signature, "input_signature": signature,
            "input_hashes": input_hashes,
            "source_paths": {**paths, "valid": source / "valid.tsv", "cohort_valid": cohort / "valid.tsv"},
            "meta_loaded": bool(load_meta), "development_loaded": False, "test_read": False}


def load_development(inputs, global_seal, *, verify_global_seal):
    """Unlock DEV only after the runner verifies its complete all-seed seal.

    The required verifier is called before any file opens. It must raise on any
    incomplete, missing or changed global selection artifact. This module does
    not treat the existence of a seal file as sufficient authorization.
    """
    if not callable(verify_global_seal):
        raise TypeError("A global all-seed barrier verifier is required")
    verification = verify_global_seal(global_seal)
    if verification is False:
        raise ValueError("Global all-seed selection barrier rejected")
    paths = inputs["source_paths"]
    for key, expected in inputs["input_hashes"].items():
        _checked_hash(paths[key], expected)
    _checked_hash(paths["valid"], inputs["signature"]["split_sha256"]["valid"])
    _checked_hash(paths["cohort_valid"], inputs["signature"]["split_sha256"]["valid"])
    pairs = read_user_pairs(paths["valid"], inputs["dev_users"],
                            real_items=inputs["items"][1:], train_pairs=inputs["train_pairs"])
    allowed, values = set(pairs), {}
    with Path(paths["ratings"]).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            user = row["user_id:token"]
            if user not in inputs["dev_users"]:
                continue
            pair = user, row["item_id:token"]
            if pair not in allowed:
                continue
            if pair in values:
                raise ValueError("Duplicate DEV rating")
            rating = float(row["rating:float"])
            if not np.isfinite(rating) or rating != int(rating) or not 1 <= rating <= 5:
                raise ValueError("Require finite integer DEV category 1..5")
            values[pair] = rating
    if set(values) != allowed:
        raise ValueError("Missing DEV rating")
    return {"pairs": pairs, "truth": dict(grouped(pairs)),
            "ratings": [(user, item, values[user, item]) for user, item in pairs],
            "liked_truth": dict(grouped(pair for pair in pairs if values[pair] >= 4)),
            "development_loaded": True, "test_read": False}


def _categories(values):
    matrix = np.asarray(values["categories"] if isinstance(values, dict) else values)
    if (matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or np.any(matrix > 5) or np.any(matrix != np.floor(matrix))
            or np.any(matrix[:, 0])):
        raise ValueError("Require a finite user-by-item category matrix 0..5 with empty PAD")
    return matrix


def episode_status(values):
    matrix = _categories(values)
    counts = np.count_nonzero(matrix, axis=1)
    return {"users": len(matrix), "eligible_users": int((counts >= 2).sum()),
            "skipped_tiny_history_users": int((counts < 2).sum()),
            "rule": "One equally weighted episode per user with at least two TRAIN records"}


def episodes(values, epoch, retain_fraction, seed, negatives=128):
    """Yield one fresh masked TRAIN-only query per eligible user in catalog order.

    Every hidden observed item is a target, regardless of its rating category.
    Alternatives are uniform without replacement from the complete TRAIN-absent
    real catalog. No hidden observed item can become an alternative or context.
    """
    if isinstance(retain_fraction, (tuple, list)):
        if tuple(retain_fraction) != (.8, .9):
            raise ValueError("The declared paired episode schedule is (0.8, 0.9)")
        for paired in zip(episodes(values, epoch, .8, seed, negatives),
                          episodes(values, epoch, .9, seed, negatives)):
            yield from paired
        return
    matrix = _categories(values)
    if retain_fraction not in (.8, .9) or int(epoch) != epoch or epoch < 0:
        raise ValueError("Use a nonnegative integer epoch and declared 0.8/0.9 retention")
    if int(negatives) != negatives or negatives < 1:
        raise ValueError("Require a positive integer alternative sample size")
    for user_index, row in enumerate(matrix):
        observed = np.flatnonzero(row)
        if len(observed) < 2:
            continue
        text = json.dumps(["conditional-evidence-episodes-v1", int(seed), int(epoch), user_index, float(retain_fraction)], separators=(",", ":"))
        stable_seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "little")
        rng = np.random.default_rng(stable_seed)
        n_keep = min(len(observed) - 1, max(1, int(np.floor(retain_fraction * len(observed)))))
        kept = rng.choice(observed, size=n_keep, replace=False)
        context = np.zeros_like(row, dtype=np.uint8)
        context[kept] = row[kept]
        targets = np.flatnonzero((row != 0) & (context == 0))
        absent = np.flatnonzero(row == 0)
        absent = absent[absent != 0]
        sampled = rng.choice(absent, size=min(int(negatives), len(absent)), replace=False)
        candidates = np.concatenate((targets, sampled)).astype(np.int64)
        target_mask = np.arange(len(candidates)) < len(targets)
        yield {"user_index": user_index, "context": context,
               "candidate_ids": candidates, "target_mask": target_mask,
               "unobserved_population_count": len(absent),
               "retain_fraction": float(retain_fraction), "context_count": n_keep,
               "target_count": len(targets), "sampled_alternatives": len(sampled)}


def episodes_signature(values, epoch, retain_fraction, seed, negatives=128):
    """A replay digest over all contexts, candidate orders and sample metadata."""
    result = hashlib.sha256()
    for episode in episodes(values, epoch, retain_fraction, seed, negatives):
        meta = {key: value for key, value in episode.items() if not isinstance(value, np.ndarray)}
        result.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
        for key in ("context", "candidate_ids", "target_mask"):
            result.update(array_digest(episode[key]).encode())
    return result.hexdigest()


def sampled_multinomial_loss(logits, target_mask, unobserved_population_count):
    """Equal-target cross entropy with the sampled partition correction U/m.

    A vector returns one query loss; a rectangular batch returns the mean query
    loss, so a user with many hidden targets does not receive additional weight.
    The logarithm of the sampled partition is a stochastic surrogate, not an
    assertion of an unbiased full-softmax loss or gradient.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim not in (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("Require finite vector or matrix logits")
    mask = torch.as_tensor(target_mask, device=logits.device)
    if mask.dtype != torch.bool or mask.shape != logits.shape:
        raise ValueError("Require a boolean target mask with the logits shape")
    scores = logits.unsqueeze(0) if logits.ndim == 1 else logits
    mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
    targets, sampled = mask.sum(-1), (~mask).sum(-1)
    population = torch.as_tensor(unobserved_population_count, dtype=logits.dtype, device=logits.device)
    if population.ndim == 0:
        population = population.expand(len(scores))
    if (population.shape != targets.shape or not torch.isfinite(population).all()
            or torch.any(population < sampled) or torch.any(population != population.floor())
            or torch.any(targets == 0) or torch.any((population > 0) & (sampled == 0))):
        raise ValueError("Invalid target/sample/population counts")
    # A query with U=m=0 has only observed targets; its correction is unused.
    correction = torch.log(torch.where(sampled > 0, population / sampled.clamp_min(1), torch.ones_like(population)))
    adjusted = scores + torch.where(mask, torch.zeros_like(scores), correction[:, None])
    target_mean = torch.where(mask, scores, torch.zeros_like(scores)).sum(-1) / targets
    return (torch.logsumexp(adjusted, dim=-1) - target_mean).mean()
