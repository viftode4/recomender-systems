"""Read-only diagnosis of completed reconstruction scores on reused development.

No fitting, parameter selection, raw rating join, TEST split, or final-test output
is accessed. All exported data is aggregate-only. Oracle statistics deliberately
use development truth and are upper bounds, not implementable recommenders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exploratory.categorical_reconstruction.verify_results import (
    SEEDS, array_digest, check_barrier, digest, grouping, json_read, json_write,
    ordered_digest, pairs, require, safe_path,
)

MODEL_NAMES = ("binary_original_grid", "binary_expanded", "categorical",
               "shuffled_categories", "slim", "hybrid_baseline2",
               "hybrid_real3", "hybrid_shuffled3")
COMPARISONS = (("binary_expanded", "categorical"),
               ("binary_expanded", "shuffled_categories"),
               ("binary_expanded", "slim"), ("categorical", "slim"),
               ("hybrid_baseline2", "hybrid_real3"))
ACTIVITY = ("sparse", "medium", "dense")


def mean(values):
    return float(np.mean(values)) if len(values) else None


def median(values):
    return float(np.median(values)) if len(values) else None


def correlation(first, second):
    """Shift- and positive-scale-invariant; constant vectors have no correlation."""
    a = np.asarray(first, float) - np.mean(first)
    b = np.asarray(second, float) - np.mean(second)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1., 1.)) if denominator else None


def ranked_user(scores, train_mask, positives, k=10):
    scores, train_mask = np.asarray(scores, float), np.asarray(train_mask, bool)
    require(scores.ndim == 1 and scores.shape == train_mask.shape
            and np.isfinite(scores).all(), "Invalid full-catalog scores")
    eligible = np.flatnonzero(~train_mask)
    eligible = eligible[eligible != 0]
    positives = np.asarray(sorted(set(positives)), dtype=int)
    require(len(eligible) >= k and len(positives) > 0
            and np.isin(positives, eligible).all(), "Invalid eligible set or truth")
    order = eligible[np.argsort(-scores[eligible], kind="stable")]
    ranks = np.zeros(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    positive_ranks = ranks[positives]
    top = order[:k]
    hit_ranks = positive_ranks[positive_ranks <= k]
    ideal = np.sum(1 / np.log2(np.arange(1, min(k, len(positives)) + 1) + 1))
    return dict(eligible=eligible, ranks=ranks, top=top, positives=positives,
                positive_ranks=positive_ranks,
                ndcg=float(np.sum(1 / np.log2(hit_ranks + 1)) / ideal),
                mrr=float(1 / min(positive_ranks)),
                mrr_at_10=float(1 / min(positive_ranks)) if len(hit_ranks) else 0.)


def positive_summary(rows, selectors=None):
    """Conditional macro means omit users without positives in the chosen stratum."""
    ranks, percentiles, macro_ranks, macro_percentiles = [], [], [], []
    recalls = {cutoff: [] for cutoff in (10, 50, 100)}
    for position, row in enumerate(rows):
        selected = (np.ones(len(row["positives"]), dtype=bool)
                    if selectors is None else selectors[position])
        values = row["positive_ranks"][selected]
        if not len(values):
            continue
        pct = (values - 1) / max(len(row["eligible"]) - 1, 1)
        ranks.extend(values.tolist())
        percentiles.extend(pct.tolist())
        macro_ranks.append(float(np.mean(values)))
        macro_percentiles.append(float(np.mean(pct)))
        for cutoff in recalls:
            recalls[cutoff].append(float(np.mean(values <= cutoff)))
    ranks = np.asarray(ranks, dtype=int)
    result = dict(users_with_positive_pairs=len(macro_ranks), positive_pairs=len(ranks),
                  positive_pair_mean_rank=mean(ranks), positive_pair_median_rank=median(ranks),
                  macro_user_mean_positive_rank=mean(macro_ranks),
                  positive_pair_mean_rank_percentile=mean(percentiles),
                  macro_user_mean_positive_rank_percentile=mean(macro_percentiles))
    for cutoff, values in recalls.items():
        result[f"hits@{cutoff}"] = int(np.sum(ranks <= cutoff))
        result[f"macro_conditional_recall@{cutoff}"] = mean(values)
        result[f"pooled_recall@{cutoff}"] = float(np.mean(ranks <= cutoff)) if len(ranks) else None
    result["misses@10"] = int(np.sum(ranks > 10))
    return result


def model_summary(rows, activities, head, zero_count):
    result = positive_summary(rows)
    result.update(ndcg_at_10=mean([row["ndcg"] for row in rows]),
                  mrr_full_catalog=mean([row["mrr"] for row in rows]),
                  mrr_at_10=mean([row["mrr_at_10"] for row in rows]),
                  head_slot_share=mean([float(np.mean(head[row["top"]])) for row in rows]),
                  zero_train_count_slot_share=mean([float(np.mean(zero_count[row["top"]])) for row in rows]))
    total_pairs, total_misses = result["positive_pairs"], result["misses@10"]

    def group_summary(chosen_rows, selectors=None):
        value = positive_summary(chosen_rows, selectors)
        value["share_of_all_positive_pairs"] = value["positive_pairs"] / total_pairs
        value["share_of_all_misses_at_10"] = value["misses@10"] / total_misses if total_misses else 0.
        return value

    result["activity"] = {}
    for name in ACTIVITY:
        chosen = [row for row, activity in zip(rows, activities) if activity == name]
        value = group_summary(chosen)
        value["ndcg_at_10"] = mean([row["ndcg"] for row in chosen])
        value["head_slot_share"] = mean([float(np.mean(head[row["top"]])) for row in chosen])
        result["activity"][name] = value
    result["popularity"] = {}
    for name, membership in (("head", head), ("tail", ~head), ("zero_train_count", zero_count)):
        value = group_summary(rows, [membership[row["positives"]] for row in rows])
        value["catalog_items"] = int(np.sum(membership[1:]))
        result["popularity"][name] = value
    result["activity_by_popularity"] = {}
    for activity in ACTIVITY:
        chosen = [row for row, name in zip(rows, activities) if name == activity]
        result["activity_by_popularity"][activity] = {
            label: group_summary(chosen, [membership[row["positives"]] for row in chosen])
            for label, membership in (("head", head), ("tail", ~head))}
    return result


def compare_models(first_scores, second_scores, first_rows, second_rows):
    pearson, spearman, overlaps, jaccards, exact_order, exact_sets = [], [], [], [], [], []
    shifts, wins, first_ndcg, second_ndcg, oracle_ndcg, union_recalls = [], [], [], [], [], []
    both = first_only = second_only = neither = total_positives = 0
    for a, b, ra, rb in zip(first_scores, second_scores, first_rows, second_rows):
        require(np.array_equal(ra["eligible"], rb["eligible"])
                and np.array_equal(ra["positives"], rb["positives"]), "Unmatched comparison")
        eligible = ra["eligible"]
        p = correlation(a[eligible], b[eligible])
        s = correlation(rankdata(a[eligible], method="average"), rankdata(b[eligible], method="average"))
        if p is not None:
            pearson.append(p)
        if s is not None:
            spearman.append(s)
        ta, tb = set(ra["top"]), set(rb["top"])
        overlaps.append(len(ta & tb) / len(ta))
        jaccards.append(len(ta & tb) / len(ta | tb))
        exact_order.append(bool(np.array_equal(ra["top"], rb["top"])))
        exact_sets.append(ta == tb)
        # Positive shift > 0 means the second model improves its rank.
        shifts.extend((ra["positive_ranks"] - rb["positive_ranks"]).tolist())
        aa, bb = ra["positive_ranks"] <= 10, rb["positive_ranks"] <= 10
        both += int(np.sum(aa & bb))
        first_only += int(np.sum(aa & ~bb))
        second_only += int(np.sum(~aa & bb))
        neither += int(np.sum(~aa & ~bb))
        total_positives += len(aa)
        first_ndcg.append(ra["ndcg"])
        second_ndcg.append(rb["ndcg"])
        wins.append(rb["ndcg"] - ra["ndcg"])
        oracle_ndcg.append(max(ra["ndcg"], rb["ndcg"]))
        union_recalls.append(float(np.mean(aa | bb)))
    shifts, wins = np.asarray(shifts), np.asarray(wins)
    return dict(users=len(first_rows), positive_pairs=total_positives,
                score_pearson_mean=mean(pearson), score_pearson_median=median(pearson),
                score_pearson_defined_users=len(pearson),
                eligible_spearman_mean=mean(spearman), eligible_spearman_median=median(spearman),
                eligible_spearman_defined_users=len(spearman),
                top10_overlap_fraction=mean(overlaps), top10_jaccard=mean(jaccards),
                exact_ordered_top10_fraction=mean(exact_order), exact_top10_set_fraction=mean(exact_sets),
                positive_mean_rank_improvement_second=mean(shifts),
                positive_median_rank_improvement_second=median(shifts),
                positive_rank_improved_fraction=float(np.mean(shifts > 0)),
                positive_rank_worsened_fraction=float(np.mean(shifts < 0)),
                positive_rank_unchanged_fraction=float(np.mean(shifts == 0)),
                user_ndcg_improved=int(np.sum(wins > 1e-14)),
                user_ndcg_worsened=int(np.sum(wins < -1e-14)),
                user_ndcg_unchanged=int(np.sum(np.abs(wins) <= 1e-14)),
                first_ndcg_at_10=mean(first_ndcg), second_ndcg_at_10=mean(second_ndcg),
                positive_hits_both_at_10=both, positive_hits_first_only_at_10=first_only,
                positive_hits_second_only_at_10=second_only, positive_missed_both_at_10=neither,
                second_recovers_fraction_of_first_misses=second_only / (second_only + neither)
                if second_only + neither else 0.,
                oracle_whole_list_ndcg_at_10=mean(oracle_ndcg),
                oracle_whole_list_gain_over_better_fixed_model=mean(oracle_ndcg) - max(mean(first_ndcg), mean(second_ndcg)),
                union_of_two_top10_macro_recall_up_to_20_slots=mean(union_recalls),
                oracle_warning="Uses development truth to choose a whole list per user; unattainable selection upper bound. Union recall has up to 20 slots and is not recall@10.")


def load_seed(directory, manifest):
    signature = json_read(directory / "input-signature.json")
    require(signature == manifest["input_signature"], "Input signature changed")
    with np.load(directory / "training-categories.npz", allow_pickle=False) as archive:
        users, items = archive["users"].tolist(), archive["items"].tolist()
        matrix = archive["ratings"].copy()
    require(items[0] == "[PAD]" and matrix.shape == (len(users), len(items))
            and ordered_digest(users, items) == signature["ordered_identity_sha256"]
            and array_digest(matrix) == signature["training_categories_sha256"], "TRAIN/catalog mismatch")
    cohorts = json_read(directory / "cohorts.json")
    require(set(cohorts["meta_fit"]).isdisjoint(cohorts["development"])
            and set(cohorts["meta_fit"]) | set(cohorts["development"]) == set(users), "Cohort mismatch")
    user_index, item_index = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    train_pairs = pairs(directory / "train.tsv")
    require(len(train_pairs) == int(np.count_nonzero(matrix))
            and all(matrix[user_index[u], item_index[i]] for u, i in train_pairs), "TRAIN pair mismatch")
    truth = grouping(pairs(directory / "valid.tsv"))
    dev = [user_index[u] for u in cohorts["development"]]
    positives = [sorted(item_index[i] for i in truth[users[row]]) for row in dev]
    selections = json_read(directory / "selection.json")
    score_arrays = {}
    for name in MODEL_NAMES:
        if name == "slim":
            path = directory / "locked-slim-scores.npy"
        else:
            selected = selections[name]
            path = safe_path(directory, selected["scores_file"])
            require(digest(path) == selected["scores_file_sha256"], "Selected scores changed")
        value = np.load(path, allow_pickle=False)
        require(value.shape == matrix.shape and np.isfinite(value).all(), "Invalid selected score array")
        if name != "slim":
            require(array_digest(value) == selections[name]["scores_array_sha256"], "Selected score array hash changed")
        score_arrays[name] = value[dev]
    counts = np.count_nonzero(matrix, axis=0)
    thresholds = np.quantile(np.count_nonzero(matrix, axis=1), [1/3, 2/3])
    activity = [ACTIVITY[int(np.searchsorted(thresholds, np.count_nonzero(matrix[row]), side="right"))] for row in dev]
    head = np.zeros(len(items), dtype=bool)
    head[sorted(range(1, len(items)), key=lambda i: (-counts[i], items[i]))[:math.ceil(.2 * (len(items)-1))]] = True
    ranked = {name: [ranked_user(scores[n], matrix[row] != 0, positives[n]) for n, row in enumerate(dev)]
              for name, scores in score_arrays.items()}
    result = dict(cohorts={name: len(values) for name, values in cohorts.items()},
                  definitions=dict(activity_thresholds=thresholds.tolist(),
                                   head_items=int(np.sum(head)), catalog_items=len(items)-1,
                                   train_count_zero_items=int(np.sum(counts[1:] == 0))),
                  selections={name: selections[name]["candidate_id"] for name in selections},
                  models={name: model_summary(rows, activity, head, counts == 0) for name, rows in ranked.items()},
                  comparisons={f"{a}__{b}": compare_models(score_arrays[a], score_arrays[b], ranked[a], ranked[b])
                               for a, b in COMPARISONS})
    return result


def render(results):
    lines = ["# Reconstruction error diagnosis", "",
             "Post-final-test exploratory diagnosis on the already reused development cohort (472 users per seed). "
             "No fitting, retuning, TEST access, or fresh confirmation occurred. Positives are every recorded validation item, regardless of rating category.", "",
             "Full-catalog ranking excludes padding and each user's original TRAIN items. Ties follow catalog column order. "
             "All means across seeds below give each split equal weight; the three splits reuse the same dataset and are not independent replications.", "",
             "| Model | nDCG@10 | Recall@10 (macro) | Recall@100 (macro) | Positive mean rank | Head slots |",
             "|---|---:|---:|---:|---:|---:|"]
    def avg(name, key):
        return np.mean([seed["models"][name][key] for seed in results["seeds"].values()])
    for name in MODEL_NAMES:
        lines.append(f"| {name} | {avg(name, 'ndcg_at_10'):.6f} | {avg(name, 'macro_conditional_recall@10'):.4f} | {avg(name, 'macro_conditional_recall@100'):.4f} | {avg(name, 'positive_pair_mean_rank'):.1f} | {avg(name, 'head_slot_share'):.2%} |")
    lines.extend(["", "| First → second | Eligible score Pearson | Top-10 overlap | Positive mean rank improvement | Oracle whole-list gain |",
                  "|---|---:|---:|---:|---:|"])
    for a, b in COMPARISONS:
        rows = [seed["comparisons"][f"{a}__{b}"] for seed in results["seeds"].values()]
        values = [mean([row[key] for row in rows]) for key in ("score_pearson_mean", "top10_overlap_fraction", "positive_mean_rank_improvement_second", "oracle_whole_list_gain_over_better_fixed_model")]
        lines.append(f"| {a} → {b} | {values[0]:.5f} | {values[1]:.2%} | {values[2]:.2f} | {values[3]:.5f} |")
    lines.extend(["", "Pearson is computed separately over each user's eligible scores after subtracting their mean; it is invariant to positive score scaling. "
                  "Eligible Spearman uses average ranks for score ties and is stored separately in aggregates.json. "
                  "Oracle whole-list selection uses development truth and cannot be deployed as measured; its gain does not show that a learned selector can recover it. "
                  "The union-of-top-10 statistic in JSON has up to 20 slots and must not be compared with fixed-budget recall@10.", "",
                  "| Seed | Model | Head positive pairs | Head recall@10 (pooled) | Tail positive pairs | Tail recall@10 (pooled) | Tail share of misses |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for seed, payload in results["seeds"].items():
        for name in ("binary_expanded", "categorical", "slim"):
            head, tail = (payload["models"][name]["popularity"][key] for key in ("head", "tail"))
            lines.append(f"| {seed} | {name} | {head['positive_pairs']} | {head['pooled_recall@10']:.4f} | {tail['positive_pairs']} | {tail['pooled_recall@10']:.4f} | {tail['share_of_all_misses_at_10']:.2%} |")
    lines.extend(["", "Head is the top ceil(20% of the catalog) by TRAIN interaction count, with lexicographic item-token ties. "
                  "Activity groups use TRAIN history terciles over all 943 users, with ties assigned upward. "
                  "Group statistics include positive-pair and user denominators; conditional macro recall excludes users with no positives in that item group. "
                  "A high fraction of all errors is not evidence of disproportionate error unless compared with that group's fraction of all positive pairs.", "",
                  "The JSON also contains activity × popularity strata, per-model positive rank percentiles, MRR, top-10 hit intersections, and per-user nDCG win/loss counts. "
                  "Different metrics need not move together: first-hit MRR can worsen while multi-positive nDCG improves. "
                  "These random per-user splits do not establish changing taste, chronological generalization, or a causal benefit from an architectural feature.", ""])
    return "\n".join(lines)


def run(research, output):
    research, output = research.resolve(), output.resolve()
    require(not output.exists(), "Refusing to overwrite diagnostic output")
    require(research.is_relative_to(ROOT), "Input must belong to this personal project")
    require(not any(p.lower().startswith(("test", "final-")) for p in research.relative_to(ROOT).parts), "Forbidden input directory")
    started = time.perf_counter()
    frozen, protocol, manifests, completed = check_barrier(research)
    source_manifest_sha = digest(research / "manifest.json")
    original = json_read(research / "aggregates.json")
    result = dict(schema_version=1, stage="reused_development", study_kind="post_final_test_error_diagnosis",
                  test_read=False, fitting_performed=False, selection_performed=False,
                  dataset_test_previously_evaluated=True, fresh_confirmation=False, seeds={})
    for seed in SEEDS:
        measured = load_seed(research / str(seed), manifests[seed])
        for name in MODEL_NAMES:
            expected = (original["seeds"][str(seed)]["references"]["SLIMElastic"]["endpoints"]["all_observed"]["metrics"]["ndcg@10"]
                        if name == "slim" else original["seeds"][str(seed)]["models"][name]["all_observed"]["ndcg@10"])
            require(abs(measured["models"][name]["ndcg_at_10"] - expected) < 2e-12,
                    f"Independent nDCG mismatch: {seed}/{name}")
        result["seeds"][str(seed)] = measured
        print(f"Diagnosed seed {seed}: identical evaluation nDCG verified for all {len(MODEL_NAMES)} score arrays", flush=True)
    require(digest(research / "manifest.json") == source_manifest_sha, "Completed source changed during diagnosis")
    # Recheck every sealed payload, not merely its index, before exporting claims.
    check_barrier(research)
    provenance = dict(schema_version=1, stage=result["stage"], test_read=False,
                      fitting_performed=False, selection_performed=False, fresh_confirmation=False,
                      source_study_manifest_sha256=source_manifest_sha,
                      source_selection_freeze_sha256=digest(research / "SELECTIONS-FROZEN.json"),
                      source_aggregates_sha256=digest(research / "aggregates.json"),
                      source_selection_manifest_sha256=frozen["selection_manifest_sha256"],
                      source_training_and_selection_sha256=protocol["source_sha256"],
                      diagnostic_source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
                          "exploratory/categorical_reconstruction/verify_results.py": digest(ROOT / "exploratory/categorical_reconstruction/verify_results.py")},
                      runtime=frozen["runtime"], elapsed_seconds=time.perf_counter()-started,
                      input_boundary="Only sealed source code/metadata, TRAIN categories/pairs, VALID pair IDs and already selected development scores. No raw .inter join, TEST split, or final-test output.",
                      independent_ndcg_replay="All eight selected score arrays reproduce saved development nDCG@10 within 2e-12 for every seed.")
    output.mkdir(parents=True)
    json_write(output / "aggregates.json", result)
    json_write(output / "provenance.json", provenance)
    (output / "RESULTS.md").write_text(render(result))
    json_write(output / "SHA256.json", {name: digest(output / name) for name in ("aggregates.json", "provenance.json", "RESULTS.md")})
    print(f"Wrote aggregate-only diagnosis to {output} in {provenance['elapsed_seconds']:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.research, args.out)


if __name__ == "__main__":
    main()
