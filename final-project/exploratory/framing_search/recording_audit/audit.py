"""Descriptive audit of original TRAIN rating-recording bundles; no model fit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

QUANTILES = (0., .25, .5, .75, .9, .95, .99, 1.)
SHUFFLES = 100
SEED_BASE = 2026092900
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require_digest(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {Path(path).name}")
    return actual


def check_inputs(source, signature_path, train_path, ratings_path, metadata_path):
    signature = json.loads(signature_path.read_text())
    source_manifest = source / "manifest.json"
    require_digest(source_manifest, signature["source_manifest_sha256"])
    manifest = json.loads(source_manifest.read_text())
    if (manifest.get("status") != "complete" or manifest.get("test_evaluated", True)
            or manifest.get("test_read", False)
            or manifest.get("validation_used_for_training", True)
            or manifest["settings"]["seed"] != 2026):
        raise ValueError("Require original complete seed-2026 TRAIN-only source")
    if manifest["split_sha256"]["train"] != signature["split_sha256"]["train"]:
        raise ValueError("Original and categorical TRAIN signatures differ")
    paths = {"categorical_input_signature": signature_path,
             "original_source_manifest": source_manifest,
             "original_train_pairs": source / "train.tsv", "copied_train_pairs": train_path,
             "raw_interactions": ratings_path, "static_item_metadata": metadata_path}
    require_digest(source / "train.tsv", manifest["split_sha256"]["train"])
    require_digest(train_path, manifest["split_sha256"]["train"])
    for path in (ratings_path, metadata_path):
        if manifest["data_sha256"][path.name] != signature["data_sha256"][path.name]:
            raise ValueError("Original and categorical raw data signatures differ")
        require_digest(path, manifest["data_sha256"][path.name])
    return {name: {"path": str(path), "sha256": digest(path)}
            for name, path in paths.items()}, manifest["split_sizes"]["train"]


def load_train_pairs(path):
    with Path(path).open() as stream:
        rows = [(row["user_id"], row["item_id"])
                for row in csv.DictReader(stream, delimiter="\t")]
    if len(set(rows)) != len(rows):
        raise ValueError("Duplicate TRAIN pair")
    return set(rows)


def read_train_records(path, allowed):
    """Do not interpret any timestamp/rating text outside allowed TRAIN pairs."""
    retained, skipped = {}, 0
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pair = row["user_id:token"], row["item_id:token"]
            if pair not in allowed:
                skipped += 1
                continue
            if pair in retained:
                raise ValueError("Duplicate raw TRAIN pair")
            timestamp, rating = float(row["timestamp:float"]), float(row["rating:float"])
            if (not math.isfinite(timestamp) or timestamp != int(timestamp)
                    or not math.isfinite(rating) or rating not in (1., 2., 3., 4., 5.)):
                raise ValueError("Invalid TRAIN timestamp/rating")
            retained[pair] = (int(timestamp), rating)
    if set(retained) != allowed:
        raise ValueError("Missing TRAIN records")
    by_user = defaultdict(list)
    for (user, item), (timestamp, rating) in retained.items():
        by_user[user].append((item, timestamp, rating))
    return {user: sorted(by_user[user]) for user in sorted(by_user)}, skipped


def load_genres(path, allowed_items):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            item = row["item_id:token"]
            if item not in allowed_items:
                continue
            if item in result:
                raise ValueError("Duplicate metadata item")
            result[item] = frozenset(row["class:token_seq"].split())
    if set(result) != allowed_items:
        raise ValueError("Missing TRAIN item genre metadata")
    return result


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in QUANTILES} if len(values) else None


def fraction(count, denominator):
    return {"count": int(count), "denominator": int(denominator),
            "fraction": float(count / denominator) if denominator else None}


def describe(records):
    sizes, max_fractions, gaps, positive_gaps = [], [], [], []
    per_user_sizes = []
    for history in records.values():
        timestamps = sorted(row[1] for row in history)
        counts = list(Counter(timestamps).values())
        sizes.extend(counts)
        per_user_sizes.append(counts)
        max_fractions.append(max(counts) / len(history))
        gaps.extend(np.diff(timestamps).tolist())
        positive_gaps.extend(np.diff(sorted(set(timestamps))).tolist())
    events = sum(map(len, records.values()))
    gap_values, positive_values = np.asarray(gaps), np.asarray(positive_gaps)
    tied_sizes = [size for size in sizes if size >= 2]
    threshold_results = {}
    for threshold in (2, 5, 10):
        threshold_results[str(threshold)] = {
            "users": fraction(sum(any(n >= threshold for n in row) for row in per_user_sizes), len(records)),
            "groups": fraction(sum(n >= threshold for n in sizes), len(sizes)),
            "records": fraction(sum(n for n in sizes if n >= threshold), events)}
    result = {"users": len(records), "records": events, "exact_time_groups": len(sizes),
              "adjacent_event_gaps": len(gaps), "distinct_time_gaps": len(positive_gaps),
              "group_thresholds": threshold_results,
              "all_group_size_quantiles": quantiles(sizes),
              "tied_group_size_quantiles": quantiles(tied_sizes),
              "largest_group_fraction_per_user_quantiles": quantiles(max_fractions),
              "zero_adjacent_event_gaps": fraction(np.sum(gap_values == 0), len(gaps)),
              "adjacent_event_gap_at_most_seconds": {},
              "positive_distinct_time_gap_at_most_seconds": {}}
    for threshold in (1, 60, 300, 1800):
        result["adjacent_event_gap_at_most_seconds"][str(threshold)] = fraction(
            np.sum(gap_values <= threshold), len(gaps))
        result["positive_distinct_time_gap_at_most_seconds"][str(threshold)] = fraction(
            np.sum(positive_values <= threshold), len(positive_gaps))
    if len(gaps) != events - len(records) or len(positive_gaps) != len(sizes) - len(records):
        raise AssertionError("Gap accounting failed")
    return result


def genre_pair_cache(records, genres):
    """Keep full within-user item association matrix; shuffle item records by index."""
    cache = []
    for history in records.values():
        groups = defaultdict(list)
        for position, (_, timestamp, _) in enumerate(history):
            groups[timestamp].append(position)
        left, right = [], []
        for members in groups.values():
            a, b = np.triu_indices(len(members), 1)
            left.extend(np.asarray(members)[a].tolist())
            right.extend(np.asarray(members)[b].tolist())
        if not left:
            # Still consume a permutation for every user's history in every null
            # replicate, maintaining the predeclared lexicographic user order.
            cache.append((len(history), None, None, None))
            continue
        vocabulary = sorted(set().union(*(genres[item] for item, _, _ in history)))
        matrix = np.asarray([[genre in genres[item] for genre in vocabulary]
                             for item, _, _ in history], dtype=np.int32)
        intersection = matrix @ matrix.T
        cardinality = matrix.sum(axis=1)
        union = cardinality[:, None] + cardinality[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)
        cache.append((len(history), similarity, np.asarray(left), np.asarray(right)))
    return cache


def genre_statistics(cache, rng=None):
    total, pairs, means = 0., 0, []
    for length, similarity, left, right in cache:
        order = np.arange(length) if rng is None else rng.permutation(length)
        if similarity is None:
            continue
        values = similarity[order[left], order[right]]
        total += float(values.sum())
        pairs += len(values)
        means.append(float(values.mean()))
    return {"pair_weighted": total / pairs if pairs else None,
            "macro_user": float(np.mean(means)) if means else None,
            "eligible_users": len(means), "tied_pairs": pairs}


def genre_comparison(records, genres):
    cache = genre_pair_cache(records, genres)
    observed = genre_statistics(cache)
    draws = [genre_statistics(cache, np.random.default_rng(SEED_BASE + replicate))
             for replicate in range(SHUFFLES)]
    summaries = {}
    for name in ("pair_weighted", "macro_user"):
        values = np.asarray([row[name] for row in draws], dtype=float)
        summaries[name] = {"observed": observed[name], "null_values": values.tolist(),
            "null_mean": float(values.mean()), "null_std_ddof0": float(values.std()),
            "null_percentiles": dict(zip(("2.5", "50", "97.5"),
                                         np.quantile(values, (.025, .5, .975)).tolist())),
            "observed_minus_null_mean": observed[name] - float(values.mean())}
    return {"eligible_users": observed["eligible_users"], "all_users": len(records),
            "excluded_users_without_tied_pairs": len(records) - observed["eligible_users"],
            "tied_pairs": observed["tied_pairs"], "shuffle_count": SHUFFLES,
            "replicate_seed_base": SEED_BASE, "statistics": summaries}


def render(result):
    count, genre = result["recording_structure"], result["genre_comparison"]
    lines = ["# TRAIN recording-bundle audit", "",
             "Descriptive evidence only; no model fit, ranking evaluation, or held-out value access.", "",
             f"Original seed-2026 TRAIN contains {count['records']:,} records from {count['users']:,} users, "
             f"forming {count['exact_time_groups']:,} exact-time groups.", "",
             "| Minimum group size | Users / all users | Records / all TRAIN records | Groups / all groups |",
             "|---|---:|---:|---:|"]
    for threshold, row in count["group_thresholds"].items():
        def cell(key):
            entry = row[key]
            return f"{entry['count']:,}/{entry['denominator']:,} ({entry['fraction']:.2%})"
        lines.append(f"| {threshold} | {cell('users')} | {cell('records')} | {cell('groups')} |")
    lines += ["", "| Quantile | All group sizes | Tied group sizes | Largest group / user's TRAIN records |",
              "|---|---:|---:|---:|"]
    for q in QUANTILES:
        lines.append(f"| {q:g} | {count['all_group_size_quantiles'][str(q)]:g} | "
                     f"{count['tied_group_size_quantiles'][str(q)]:g} | "
                     f"{count['largest_group_fraction_per_user_quantiles'][str(q)]:.2%} |")
    lines += ["", "| Maximum adjacent gap | Including tied rows | Positive gaps between distinct times |",
              "|---|---:|---:|"]
    for threshold in (1, 60, 300, 1800):
        a = count["adjacent_event_gap_at_most_seconds"][str(threshold)]
        b = count["positive_distinct_time_gap_at_most_seconds"][str(threshold)]
        lines.append(f"| {threshold}s | {a['count']:,}/{a['denominator']:,} ({a['fraction']:.2%}) | "
                     f"{b['count']:,}/{b['denominator']:,} ({b['fraction']:.2%}) |")
    zero = count["zero_adjacent_event_gaps"]
    lines += ["", f"Exactly zero: {zero['count']:,}/{zero['denominator']:,} adjacent event gaps ({zero['fraction']:.2%}).", "",
              f"Genre coherence uses {genre['tied_pairs']:,} unordered same-user tied-time pairs across "
              f"{genre['eligible_users']:,}/{genre['all_users']:,} users. The comparison preserves each user's "
              "movies/ratings and exact timestamp multiplicities in 100 deterministic shuffles.", "",
              "| Genre Jaccard statistic | Observed | Shuffle mean | Difference | Shuffle 2.5–97.5 percentiles |",
              "|---|---:|---:|---:|---:|"]
    for name, row in genre["statistics"].items():
        lines.append(f"| {name} | {row['observed']:.6f} | {row['null_mean']:.6f} | "
                     f"{row['observed_minus_null_mean']:+.6f} | "
                     f"{row['null_percentiles']['2.5']:.6f}–{row['null_percentiles']['97.5']:.6f} |")
    lines += ["", "These are descriptive shuffle comparisons, not p-values. Timestamp ties do not identify a "
              "screen, consumption time, or causal interface effect. The random TRAIN split thins the original "
              "records. Any observed genre coherence is compatible with multiple recording processes and is "
              "not evidence of improved recommendation accuracy or a novel method.", ""]
    return "\n".join(lines)


def run(args):
    if args.output.exists():
        raise ValueError("Refusing to overwrite an audit")
    source = check_inputs(args.source, args.signature, args.train, args.ratings, args.metadata)
    inputs, expected_rows = source
    allowed = load_train_pairs(args.train)
    if len(allowed) != expected_rows:
        raise ValueError("Original TRAIN count differs")
    records, skipped = read_train_records(args.ratings, allowed)
    genres = load_genres(args.metadata, {item for _, item in allowed})
    result = {"schema_version": 1, "study": "train_only_recording_bundle_description",
              "fit_performed": False, "ranking_evaluated": False, "valid_files_read": False,
              "test_files_read": False, "held_out_values_parsed": False,
              "outside_train_rows_skipped_before_value_parsing": skipped,
              "recording_structure": describe(records),
              "genre_comparison": genre_comparison(records, genres)}
    provenance = {"inputs": inputs,
                  "code_sha256": {name: digest(HERE / name) for name in ("audit.py", "PROTOCOL.md")},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                              "numerical_threads": 1},
                  "command": "See README.md; output directory must not already exist."}
    args.output.mkdir(parents=True)
    save_json(args.output / "aggregates.json", result)
    save_json(args.output / "provenance.json", provenance)
    (args.output / "RESULTS.md").write_text(render(result))
    save_json(args.output / "SHA256.json", {name: digest(args.output / name)
              for name in ("aggregates.json", "provenance.json", "RESULTS.md")})
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "records": len(allowed), "users": len(records)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "signature", "train", "ratings", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
