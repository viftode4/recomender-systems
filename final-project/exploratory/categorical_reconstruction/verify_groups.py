"""Independent aggregate group audit of the completed reconstruction study.

No fitting or TEST access. Uses the result verifier only for hash/barrier checks;
ranking, metadata handling, user metrics and exposure arithmetic are independent
of the measured runner, group_analysis, metrics and societal modules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exploratory.categorical_reconstruction.verify_results import check_barrier, safe_path

MODELS = ("binary_original_grid", "binary_expanded", "categorical", "shuffled_categories",
          "hybrid_baseline2", "hybrid_real3", "hybrid_shuffled3")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def grouped_pairs(path):
    result = defaultdict(set)
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            user, item = row["user_id"], row["item_id"]
            if item in result[user]:
                raise ValueError("Duplicate interaction")
            result[user].add(item)
    return dict(result)


def entropy(values):
    return -sum(float(p)*math.log2(float(p)) for p in values if p > 0)


def jsd(first, second):
    return entropy((first+second)/2) - (entropy(first)+entropy(second))/2


def recompute(scores, users, items, history, truth, development, genres):
    """Return all reported numeric group fields using separate formulas."""
    k, catalog = 10, items[1:]
    counts = Counter(item for entries in history.values() for item in entries)
    limits = np.quantile([len(history[user]) for user in users], [1/3, 2/3])
    names = ("sparse", "medium", "dense")
    activity = {user: names[0 if len(history[user]) < limits[0] else
                           1 if len(history[user]) < limits[1] else 2] for user in users}
    head = set(sorted(catalog, key=lambda item: (-counts[item], item))[:math.ceil(len(catalog)/5)])
    vocabulary = sorted(set().union(*(genres[item] for item in catalog)))
    features = {item: np.array([int(genre in genres[item])/len(genres[item]) for genre in vocabulary])
                for item in catalog}
    discounts = np.array([1/math.log2(position+1) for position in range(1, k+1)])
    exposure = dict.fromkeys(catalog, 0.)
    rows, recs, exposed = {}, {}, set()
    for index, user in enumerate(users):
        if user not in development:
            continue
        eligible = [column for column in range(1, len(items)) if items[column] not in history[user]]
        if not np.isfinite(scores[index, eligible]).all():
            raise ValueError("Nonfinite candidate score")
        selected = sorted(eligible, key=lambda column: (-scores[index, column], column))[:k]
        recommendation = [items[column] for column in selected]
        if len(recommendation) != k or not truth[user] or truth[user] & history[user]:
            raise ValueError("Invalid recommendation or truth")
        recs[user] = recommendation
        hit = np.array([item in truth[user] for item in recommendation], dtype=float)
        positions = np.flatnonzero(hit)
        profile = sum((features[item] for item in history[user]), np.zeros(len(vocabulary)))/len(history[user])
        suggested = sum((features[item] for item in recommendation), np.zeros(len(vocabulary)))/k
        train_head = len(history[user] & head)/len(history[user])
        rec_head = len(set(recommendation) & head)/k
        dissimilarities = [1-len(genres[a] & genres[b])/len(genres[a] | genres[b])
                          for p, a in enumerate(recommendation) for b in recommendation[p+1:]]
        rows[user] = {
            "precision@10": float(hit.sum()/k), "recall@10": float(hit.sum()/len(truth[user])),
            "ndcg@10": float(hit@discounts/discounts[:min(k, len(truth[user]))].sum()),
            "mrr@10": 0. if not len(positions) else 1/float(positions[0]+1),
            "hit@10": float(bool(len(positions))),
            "novelty@10": sum(math.log2((sum(counts.values())+len(catalog))/(counts[item]+1))
                              for item in recommendation)/k,
            "diversity": sum(dissimilarities)/len(dissimilarities),
            "calibration_jsd": jsd(profile, suggested), "head_exposure": rec_head,
            "popularity_jsd": jsd(np.array([1-train_head, train_head]), np.array([1-rec_head, rec_head]))}
        for position, item in enumerate(recommendation):
            exposure[item] += discounts[position]/len(development)
        exposed.update(recommendation)
    if set(rows) != development:
        raise ValueError("Evaluation cohort differs")
    aggregate = {key: sum(row[key] for row in rows.values())/len(rows) for key in next(iter(rows.values()))}
    aggregate["coverage@10"] = len(exposed)/len(catalog)
    user_groups = {}
    for name in names:
        members = [user for user in development if activity[user] == name]
        user_groups[name] = {"users": len(members), **{
            key: None if not members else sum(rows[user][key] for user in members)/len(members)
            for key in next(iter(rows.values()))}}
    utilities = [value["ndcg@10"] for value in user_groups.values() if value["users"]]
    aggregate.update(activity_ndcg_gap=max(utilities)-min(utilities), worst_activity_ndcg=min(utilities),
                     head_exposure_gap=abs(aggregate["head_exposure"]-len(head)/len(catalog)))
    mass = np.array(list(exposure.values()))
    probability = mass/mass.sum()
    cumulative = np.r_[0., np.cumsum(np.sort(probability))]
    gini = 1-float(np.sum(cumulative[:-1]+cumulative[1:])/len(catalog))
    exposure_entropy = entropy(probability)
    item_groups, exposure_groups = {}, {}
    for name, members in (("head", head), ("tail", set(catalog)-head)):
        recall, positives, hits, slots = [], 0, 0, 0
        for user, recommendation in recs.items():
            relevant = truth[user] & members
            found = len(set(recommendation) & relevant)
            positives += len(relevant)
            hits += found
            slots += len(set(recommendation) & members)
            if relevant:
                recall.append(found/len(relevant))
        total = sum(exposure[item] for item in members)
        mean_exposure = total/len(members)
        share = total/mass.sum()
        conditional = None if not recall else sum(recall)/len(recall)
        item_groups[name] = {"items": len(members), "users_with_positives": len(recall),
            "heldout_positive_pairs": positives, "hits": hits, "recall@10": conditional,
            "exposure": slots/(len(development)*k), "discounted_exposure": share,
            "mean_item_exposure": mean_exposure}
        exposure_groups[name] = {"items": len(members), "mean_item_exposure": mean_exposure,
                                 "fraction_total_exposure": share}
        aggregate[name+"_recall"] = conditional
    gap = abs(exposure_groups["head"]["mean_item_exposure"]-exposure_groups["tail"]["mean_item_exposure"])
    aggregate.update(exposure_gini=gini, exposure_entropy=exposure_entropy/math.log2(len(catalog)),
                     item_group_exposure_gap=gap)
    return {"users": len(rows), "aggregate": aggregate, "user_groups": user_groups, "item_groups": item_groups,
        "item_exposure": {"gini": gini, "entropy_bits": exposure_entropy,
            "normalized_entropy": exposure_entropy/math.log2(len(catalog)), "groups": exposure_groups,
            "group_mean_exposure_gap": gap, "users": len(rows), "catalogue_items": len(catalog)},
        "definitions": {"grouping_population_users": len(users), "activity_thresholds": limits.tolist(),
                        "head_items": len(head), "catalog_items": len(catalog)}}


def compare(actual, expected, path="root"):
    errors, fields = [], 0
    if isinstance(expected, dict):
        for key, value in expected.items():
            error, count = compare(actual[key], value, path+"/"+key)
            errors.append(error); fields += count
        return max(errors, default=0.), fields
    if expected is None:
        if actual is not None:
            raise ValueError("Empty denominator mismatch at "+path)
        return 0., 1
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, atol=2e-12, rtol=2e-12):
        raise ValueError("Independent group metric differs at "+path)
    return float(np.max(np.abs(a-b))), int(b.size)


def audit(research, evidence, metadata, out):
    if out.exists():
        raise FileExistsError(out)
    frozen, _, _, completed = check_barrier(research)
    pins = {"selection_freeze_sha256": sha(research/"SELECTIONS-FROZEN.json"),
            "completed_manifest_sha256": sha(research/"manifest.json"),
            "verifier_sha256": sha(__file__), "barrier_verifier_sha256": sha(Path(__file__).with_name("verify_results.py"))}
    hashes = {name: sha(evidence/name) for name in ("aggregates.json", "protocol.json", "provenance.json")}
    seal = read_json(evidence/"SHA256.json")
    if any(hashes[name] != seal[name] for name in hashes) or sha(research/"aggregates.json") != hashes["aggregates.json"]:
        raise ValueError("Curated aggregate seal differs")
    provenance = read_json(evidence/"provenance.json")
    if (provenance["manifest_sha256"] != pins["completed_manifest_sha256"] or
            provenance["selection_freeze_sha256"] != pins["selection_freeze_sha256"]):
        raise ValueError("Provenance differs")
    published = read_json(evidence/"aggregates.json")
    with metadata.open() as stream:
        genres = {row["item_id:token"]: set(row["class:token_seq"].split())
                  for row in csv.DictReader(stream, delimiter="\t")}
    result = {"status": "complete", "test_read": False, "fresh_confirmation": False,
        "stage": "reused_development", "artifact_sha256": hashes, "source_sha256": frozen["source_sha256"],
        **pins, "seeds": {}, "scope": "All numeric user/item group, diversity, JSD, novelty and exposure fields; seven selected models, three seeds. Independent formulas and rankings; no model fitting."}
    for seed in frozen["seeds"]:
        directory = research/str(seed)
        signature = read_json(directory/"input-signature.json")
        if sha(metadata) != signature["data_sha256"][metadata.name]:
            raise ValueError("Metadata changed")
        history, truth = grouped_pairs(directory/"train.tsv"), grouped_pairs(directory/"valid.tsv")
        ids = read_json(directory/"ids.json")
        users, items = [user for user in ids["users"] if user in history], ids["items"]
        dev = set(read_json(directory/"cohorts.json")["development"])
        selection = read_json(directory/"selection.json")
        result["seeds"][str(seed)] = {}
        for model in MODELS:
            scores = np.load(safe_path(directory, selection[model]["scores_file"]), allow_pickle=False)
            computed = recompute(scores, users, items, history, truth, dev, genres)
            original = published["seeds"][str(seed)]["models"][model]["groups"]
            error, fields = compare(original, computed)
            result["seeds"][str(seed)][model] = {"numeric_fields_verified": fields,
                "maximum_absolute_error": error, "users": len(dev),
                "user_group_counts": {name: value["users"] for name, value in computed["user_groups"].items()},
                "item_group_positive_users": {name: value["users_with_positives"] for name, value in computed["item_groups"].items()}}
    check_barrier(research)
    if (sha(research/"manifest.json") != pins["completed_manifest_sha256"] or sha(__file__) != pins["verifier_sha256"]
            or sha(Path(__file__).with_name("verify_results.py")) != pins["barrier_verifier_sha256"]
            or any(sha(evidence/name) != value for name, value in hashes.items())):
        raise ValueError("Audit inputs or source changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("research", "evidence", "metadata", "out"):
        parser.add_argument("--"+flag, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.research, args.evidence, args.metadata, args.out)
    print(json.dumps({"status": result["status"], "seeds": len(result["seeds"]),
                      "models": len(MODELS), "receipt_sha256": sha(args.out)}, indent=2))


if __name__ == "__main__":
    main()
