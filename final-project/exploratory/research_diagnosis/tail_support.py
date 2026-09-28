"""Read-only TRAIN-support breakdown of already selected VALID recommendations.

No raw rating file, TEST payload, fitting, tuning or new model selection.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exploratory.categorical_reconstruction.verify_results import (
    array_digest, check_barrier, digest, grouping, json_read, ordered_digest,
    pairs, require, safe_path)
from exploratory.research_diagnosis.diagnostics import ranked_user

BINS = (("0", 0, 0), ("1-5", 1, 5), ("6-20", 6, 20),
        ("21-50", 21, 50), ("51+", 51, np.inf))


def mean(values):
    return float(np.mean(values)) if len(values) else None


def available_evidence(rows):
    return {"pairs": len(rows),
        "fraction_with_any_shared_TRAIN_user": mean([r["cooc_max"] > 0 for r in rows]),
        "fraction_with_max_coobservation_at_least_5": mean([r["cooc_max"] >= 5 for r in rows]),
        "mean_max_coobservation": mean([r["cooc_max"] for r in rows]),
        "median_max_coobservation": float(np.median([r["cooc_max"] for r in rows])) if rows else None,
        "mean_number_of_coobserved_context_items": mean([r["cooc_sources"] for r in rows]),
        "fraction_with_any_shared_genre": mean([r["genre_coverage"] > 0 for r in rows]),
        "mean_target_genre_coverage_by_TRAIN_history": mean([r["genre_coverage"] for r in rows]),
        "mean_max_context_item_genre_Jaccard": mean([r["genre_max_jaccard"] for r in rows])}


def summarize(rows, membership, tops):
    chosen = [r for r in rows if membership[r["item"]]]
    result = {"catalog_items": int(membership.sum()), "positive_pairs": len(chosen),
              "distinct_positive_items": len({r["item"] for r in chosen}),
              "users_with_positives": len({r["user"] for r in chosen}),
              "available_evidence_all_positives": available_evidence(chosen), "models": {}}
    for model in ("binary_expanded", "categorical"):
        hit = [r for r in chosen if r[model] <= 10]
        missed = [r for r in chosen if r[model] > 10]
        grouped = {}
        for row in chosen:
            grouped.setdefault(row["user"], []).append(row[model] <= 10)
        result["models"][model] = {
            "hits@10": len(hit), "misses@10": len(missed),
            "pooled_recall@10": len(hit)/len(chosen) if chosen else None,
            "macro_conditional_recall@10": mean([np.mean(v) for v in grouped.values()]),
            "recommendation_slots": int(membership[tops[model]].sum()),
            "median_positive_rank": float(np.median([r[model] for r in chosen])) if chosen else None,
            "available_evidence_missed_positives": available_evidence(missed)}
    result["hit_intersection"] = {
        "both": sum(r["binary_expanded"] <= 10 and r["categorical"] <= 10 for r in chosen),
        "binary_only": sum(r["binary_expanded"] <= 10 < r["categorical"] for r in chosen),
        "categorical_only": sum(r["categorical"] <= 10 < r["binary_expanded"] for r in chosen),
        "neither": sum(r["binary_expanded"] > 10 and r["categorical"] > 10 for r in chosen)}
    return result


def run(research, metadata, out):
    require(not out.exists(), "Refuse to overwrite diagnostic evidence")
    frozen, protocol, manifests, completed = check_barrier(research)
    source_hash = digest(research/"manifest.json")
    own_hash = digest(Path(__file__))
    metadata_hash = digest(metadata)
    with metadata.open() as stream:
        genres = {r["item_id:token"]: set(r["class:token_seq"].split())
                  for r in csv.DictReader(stream, delimiter="\t")}
    result = {"schema_version": 1, "stage": "reused_development_support_diagnosis",
        "test_read": False, "raw_ratings_read": False, "fitting_performed": False,
        "selection_performed": False, "fresh_confirmation": False,
        "definitions": {
            "relevance": "Every original VALID record for the unchanged472 development users; no rating threshold.",
            "head": "Top ceil(20% of real catalog) by TRAIN count, ties by original item token; tail is complement.",
            "support": "Number of distinct TRAIN users observing the target item, regardless of rating.",
            "bins": [name for name, _, _ in BINS],
            "coobservation": "Maximum over this user's TRAIN context items of the number of TRAIN users observing both source and target. The evaluated user never observes target in TRAIN.",
            "genre": "Provided catalog genre tokens; fraction of target genres present in union of TRAIN-history genres, and maximum source-item genre Jaccard. This is descriptive overlap, not a fitted content predictor.",
            "averaging": "Pooled recall is hits/positive pairs; conditional macro recall averages users with positives in that stratum. Empty strata use null.",
            "limits": "Available cooccurrence or broad genre overlap is not proof of useful identifiable preference signal. Overlapping seed splits are not independent observations."},
        "source_manifest_sha256": source_hash, "diagnostic_source_sha256": own_hash,
        "metadata_sha256": metadata_hash, "seeds": {}}
    previous = json_read(ROOT/"exploratory/research_diagnosis/results-v1/aggregates.json")
    result["earlier_diagnosis_sha256"] = digest(ROOT/"exploratory/research_diagnosis/results-v1/aggregates.json")
    for seed in (2026, 2027, 2028):
        directory = research/str(seed)
        signature = manifests[seed]["input_signature"]
        require(metadata_hash == signature["data_sha256"][metadata.name], "Item metadata changed")
        with np.load(directory/"training-categories.npz", allow_pickle=False) as archive:
            users, items, ratings = archive["users"].tolist(), archive["items"].tolist(), archive["ratings"].copy()
        require(ordered_digest(users, items) == signature["ordered_identity_sha256"]
                and array_digest(ratings) == signature["training_categories_sha256"], "TRAIN identities changed")
        x = ratings != 0
        counts = x.sum(axis=0)
        require(not x[:, 0].any(), "Padding contains observations")
        head = np.zeros(len(items), bool)
        head[sorted(range(1, len(items)), key=lambda i: (-counts[i], items[i]))[:math.ceil(.2*(len(items)-1))]] = True
        tail = ~head
        tail[0] = False
        all_items = np.ones(len(items), bool)
        all_items[0] = False
        cohorts = json_read(directory/"cohorts.json")
        dev = [users.index(user) for user in cohorts["development"]]
        truth = grouping(pairs(directory/"valid.tsv"))
        item_index = {item: i for i, item in enumerate(items)}
        selections = json_read(directory/"selection.json")
        scored = {}
        for model in ("binary_expanded", "categorical"):
            selection = selections[model]
            path = safe_path(directory, selection["scores_file"])
            require(digest(path) == selection["scores_file_sha256"], "Score file changed")
            scores = np.load(path, allow_pickle=False)
            require(array_digest(scores) == selection["scores_array_sha256"], "Score array changed")
            scored[model] = [ranked_user(scores[u], x[u], [item_index[item] for item in truth[users[u]]]) for u in dev]
        cooc = x.astype(np.float64).T @ x.astype(np.float64)
        vocabulary = sorted(set().union(*(genres[item] for item in items[1:])))
        g = np.zeros((len(items), len(vocabulary)), dtype=np.int32)
        for j, item in enumerate(items[1:], 1):
            g[j] = [value in genres[item] for value in vocabulary]
        require(np.all(g[1:].sum(axis=1) > 0), "Missing catalog genres")
        rows = []
        for n, u in enumerate(dev):
            context = np.flatnonzero(x[u])
            union = g[context].sum(axis=0) > 0
            for i in scored["binary_expanded"][n]["positives"]:
                common = cooc[i, context]
                intersection = g[context] @ g[i]
                union_sizes = g[context].sum(axis=1) + g[i].sum() - intersection
                rows.append({"user": n, "item": int(i),
                    "binary_expanded": int(scored["binary_expanded"][n]["ranks"][i]),
                    "categorical": int(scored["categorical"][n]["ranks"][i]),
                    "cooc_max": int(common.max()), "cooc_sources": int((common > 0).sum()),
                    "genre_coverage": float(np.sum(g[i]*union)/g[i].sum()),
                    "genre_max_jaccard": float(np.max(intersection/union_sizes))})
        tops = {model: np.stack([r["top"] for r in values]) for model, values in scored.items()}
        summary = {"development_users": len(dev), "recommendation_slots_per_model": len(dev)*10,
            "TRAIN_observations": int(x.sum()), "head_min_TRAIN_support": int(counts[head].min()),
            "tail_max_TRAIN_support": int(counts[tail].max()),
            "all_catalog": summarize(rows, all_items, tops), "tail": summarize(rows, tail, tops),
            "bins_all_catalog": {}, "bins_tail": {}}
        for name, low, high in BINS:
            members = all_items & (counts >= low) & (counts <= high)
            summary["bins_all_catalog"][name] = summarize(rows, members, tops)
            summary["bins_tail"][name] = summarize(rows, members & tail, tops)
        for scope in ("all_catalog", "tail"):
            bins = summary["bins_" + ("all_catalog" if scope == "all_catalog" else "tail")]
            require(sum(b["positive_pairs"] for b in bins.values()) == summary[scope]["positive_pairs"], "Bin positive counts do not partition")
            for model in scored:
                require(sum(b["models"][model]["hits@10"] for b in bins.values()) == summary[scope]["models"][model]["hits@10"], "Bin hits do not partition")
        for model in scored:
            reference = previous["seeds"][str(seed)]["models"][model]["popularity"]["tail"]
            require(summary["tail"]["positive_pairs"] == reference["positive_pairs"]
                    and summary["tail"]["models"][model]["hits@10"] == reference["hits@10"], "Earlier tail diagnosis differs")
        result["seeds"][str(seed)] = summary
    check_barrier(research)
    require(digest(research/"manifest.json") == source_hash and digest(Path(__file__)) == own_hash
            and digest(metadata) == metadata_hash, "Diagnostic inputs changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"Saved aggregate-only support diagnosis: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=ROOT/"runs/categorical-reconstruction-v1")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.research, args.items, args.out)
