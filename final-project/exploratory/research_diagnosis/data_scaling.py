"""Fixed query histories, nested disjoint donor rows; no TEST access or old edits."""
from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exploratory.categorical_reconstruction import model
from exploratory.research_diagnosis.diagnostics import (
    ACTIVITY, array_digest, check_barrier, digest, json_read, json_write,
    model_summary, ordered_digest, ranked_user, require,
)

SEEDS = (2026, 2027, 2028)
FRACTIONS = (.25, .5, .75, 1.)
LAMBDAS = (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)
SOURCE_FILES = ("exploratory/research_diagnosis/data_scaling.py",
                "exploratory/research_diagnosis/PROTOCOL_DATA_SCALING.md",
                "exploratory/research_diagnosis/diagnostics.py",
                "exploratory/categorical_reconstruction/model.py",
                "exploratory/categorical_reconstruction/verify_results.py")


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def split_donors(meta_users, development_users, seed):
    require(len(meta_users) == len(set(meta_users)) and set(meta_users).isdisjoint(development_users),
            "Duplicate meta users or overlap with development")
    order = np.array(sorted(meta_users))
    np.random.default_rng(seed + 30011).shuffle(order)
    inner_count = math.ceil(len(order) / 4)
    inner, donors = order[:inner_count].tolist(), order[inner_count:].tolist()
    require(inner and len(donors) >= 4, "Too few query/donor users")
    return inner, donors


def read_truth(directory, chosen_users):
    """Only retain permitted users' validation pair identities; no rating labels."""
    chosen = set(chosen_users)
    result = {user: set() for user in chosen}
    with (directory / "valid.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["user_id"] in chosen:
                result[row["user_id"]].add(row["item_id"])
    require(all(result.values()), "Every chosen query requires validation truth")
    return result


def inputs(directory, seed):
    with np.load(directory / "training-categories.npz", allow_pickle=False) as archive:
        users, items, matrix = archive["users"].tolist(), archive["items"].tolist(), archive["ratings"].copy()
    signature = json_read(directory / "input-signature.json")
    require(matrix.shape == (len(users), len(items)) and items[0] == "[PAD]"
            and array_digest(matrix) == signature["training_categories_sha256"]
            and ordered_digest(users, items) == signature["ordered_identity_sha256"], "Changed TRAIN identity")
    cohorts = json_read(directory / "cohorts.json")
    require(set(cohorts["meta_fit"]) | set(cohorts["development"]) == set(users), "Incomplete cohorts")
    inner, donors = split_donors(cohorts["meta_fit"], cohorts["development"], seed)
    return users, items, matrix, cohorts, inner, donors


def metric(scores, query_rows, users, items, matrix, truth):
    lookup = {item: index for index, item in enumerate(items)}
    rows = [ranked_user(scores[n], matrix[row] != 0,
                        [lookup[item] for item in truth[users[row]]])
            for n, row in enumerate(query_rows)]
    return float(np.mean([row["ndcg"] for row in rows]))


def select_seed(source, output, seed, signature):
    output.mkdir()
    users, items, matrix, cohorts, inner, donors = inputs(source, seed)
    user_index = {user: index for index, user in enumerate(users)}
    query_rows = np.array([user_index[user] for user in inner])
    truth = read_truth(source, inner)
    split = dict(seed=seed, inner_queries=inner, donors_ordered=donors,
                 development_queries=cohorts["development"],
                 own_fitting_row_present_inner=0, own_fitting_row_present_development=0)
    json_write(output / "cohorts.json", split)
    selections, grid = {}, []
    started = time.perf_counter()
    for fraction in FRACTIONS:
        count = math.floor(len(donors) * fraction)
        key = str(count)
        donor_rows = np.array([user_index[user] for user in donors[:count]])
        prepared = model.prepare_features(matrix[donor_rows], smoothing=20.)
        best = None
        for penalty in LAMBDAS:
            fit_started = time.perf_counter()
            fitted = model.fit(prepared, penalty, category_ratio=None)
            scores = fitted.predict(matrix[query_rows])
            value = metric(scores, query_rows, users, items, matrix, truth)
            candidate = dict(donors=count, fraction=fraction, penalty=penalty,
                             inner_query_ndcg_at_10=value,
                             inner_scores_array_sha256=array_digest(scores),
                             fit_prediction_selection_seconds=time.perf_counter()-fit_started,
                             solver_fallback_count=fitted.diagnostics["fallback_count"])
            grid.append(candidate)
            if best is None or value > best["inner_query_ndcg_at_10"]:
                best, winner = candidate.copy(), fitted
            print(f"seed={seed} donors={count} lambda={penalty:g} inner_ndcg={value:.6f}", flush=True)
        predictions = winner.predict(matrix)
        refit = model.fit(prepared, best["penalty"], category_ratio=None)
        require(np.array_equal(predictions, refit.predict(matrix)), "Selected refit differs")
        checkpoint = output / f"model-donors-{count}.npz"
        winner.save(checkpoint)
        require(np.array_equal(predictions, model.load(checkpoint).predict(matrix)), "Serialized model replay differs")
        score_path = output / f"scores-donors-{count}.npy"
        np.save(score_path, predictions, allow_pickle=False)
        require(np.array_equal(np.load(score_path, allow_pickle=False), predictions), "Saved scores differ")
        best.update(scores_file=score_path.name, scores_file_sha256=digest(score_path),
                    scores_array_sha256=array_digest(predictions), model_file=checkpoint.name,
                    model_file_sha256=digest(checkpoint), replay_exact=True,
                    donor_training_categories_sha256=array_digest(matrix[donor_rows]),
                    donor_observations=int(np.count_nonzero(matrix[donor_rows])),
                    donor_observed_items=int(np.sum(np.any(matrix[donor_rows, 1:] != 0, axis=0))))
        selections[key] = best
    require(source_hashes() == signature["source_sha256"], "Source changed while selecting")
    json_write(output / "candidate-grid.json", grid)
    json_write(output / "selection.json", selections)
    hashes = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    manifest = dict(seed=seed, stage="selected", development_evaluated=False, test_read=False,
                    source_sha256=signature["source_sha256"], protocol_sha256=signature["protocol_sha256"],
                    source_study_manifest_sha256=signature["source_study_manifest_sha256"],
                    inner_query_users=len(inner), development_users=len(cohorts["development"]),
                    available_donors=len(donors), candidate_fits=len(grid),
                    selection_seconds=time.perf_counter()-started, output_sha256=hashes)
    json_write(output / "selection-manifest.json", manifest)
    return manifest


def verify_selection(directory, manifest):
    for name, expected in manifest["output_sha256"].items():
        require(Path(name).name == name and digest(directory / name) == expected, "Changed selected payload")
    grid = json_read(directory / "candidate-grid.json")
    selection = json_read(directory / "selection.json")
    counts = [math.floor(manifest["available_donors"] * fraction) for fraction in FRACTIONS]
    require(len(grid) == len(counts) * len(LAMBDAS) and set(selection) == set(map(str, counts)), "Incomplete curve grid")
    require([(row["donors"], row["penalty"]) for row in grid] == [(count, penalty) for count in counts for penalty in LAMBDAS],
            "Changed grid order or candidates")
    for count in counts:
        candidates = [row for row in grid if row["donors"] == count]
        require(all(np.isfinite(row["inner_query_ndcg_at_10"]) for row in candidates), "Nonfinite selection metric")
        winner = max(candidates, key=lambda row: row["inner_query_ndcg_at_10"])
        actual = selection[str(count)]
        require(all(actual[key] == value for key, value in winner.items()), "Selection is not exact first argmax")
        scores = np.load(directory / actual["scores_file"], allow_pickle=False)
        require(digest(directory / actual["scores_file"]) == actual["scores_file_sha256"]
                and np.isfinite(scores).all() and array_digest(scores) == actual["scores_array_sha256"], "Invalid selected score file")
    return selection


def freeze(output, manifests, signature):
    require(source_hashes() == signature["source_sha256"], "Source changed before global seal")
    for seed, manifest in manifests.items():
        verify_selection(output / str(seed), manifest)
    require(tuple(manifests) == SEEDS, "Missing study seed")
    frozen = dict(status="all_selections_frozen", seeds=list(SEEDS), test_read=False,
                  development_evaluated=False, selection_count=len(SEEDS)*len(FRACTIONS),
                  **signature, selection_manifest_sha256={str(seed): digest(output / str(seed) / "selection-manifest.json") for seed in SEEDS})
    json_write(output / "SELECTIONS-FROZEN.json", frozen)
    return frozen


def evaluate_seed(source, directory, seed, frozen, root):
    require((root / "SELECTIONS-FROZEN.json").is_file()
            and json_read(root / "SELECTIONS-FROZEN.json") == frozen
            and frozen["selection_count"] == 12
            and tuple(frozen["seeds"]) == SEEDS, "Global selection barrier absent")
    require(digest(directory / "selection-manifest.json") == frozen["selection_manifest_sha256"][str(seed)], "Selection manifest changed")
    selected = verify_selection(directory, json_read(directory / "selection-manifest.json"))
    users, items, matrix, cohorts, inner, donors = inputs(source, seed)
    user_index, item_index = {user: n for n, user in enumerate(users)}, {item: n for n, item in enumerate(items)}
    dev_rows = [user_index[user] for user in cohorts["development"]]
    truth = read_truth(source, cohorts["development"])
    counts = np.count_nonzero(matrix, axis=0)
    thresholds = np.quantile(np.count_nonzero(matrix, axis=1), [1/3, 2/3])
    activity = [ACTIVITY[int(np.searchsorted(thresholds, np.count_nonzero(matrix[row]), side="right"))] for row in dev_rows]
    head = np.zeros(len(items), dtype=bool)
    head[sorted(range(1, len(items)), key=lambda i: (-counts[i], items[i]))[:math.ceil(.2*(len(items)-1))]] = True
    results = {}
    for key, chosen in selected.items():
        scores = np.load(directory / chosen["scores_file"], allow_pickle=False)
        rows = [ranked_user(scores[row], matrix[row] != 0, [item_index[item] for item in truth[users[row]]]) for row in dev_rows]
        summary = model_summary(rows, activity, head, counts == 0)
        donor_rows = [user_index[user] for user in donors[:chosen["donors"]]]
        supported = np.any(matrix[donor_rows] != 0, axis=0)
        unsupported_positives = sum(int(np.sum(~supported[row["positives"]])) for row in rows)
        unsupported_histories = [float(np.mean(~supported[np.flatnonzero(matrix[row])])) for row in dev_rows]
        summary.update(donors=chosen["donors"], fraction=chosen["fraction"], selected_penalty=chosen["penalty"],
                       inner_query_ndcg_at_10=chosen["inner_query_ndcg_at_10"],
                       donor_observations=chosen["donor_observations"], donor_observed_items=chosen["donor_observed_items"],
                       own_fitting_row_present_development=0, development_users=len(dev_rows),
                       inner_query_users=len(inner), own_fitting_row_present_inner=0,
                       unsupported_positive_pairs=unsupported_positives,
                       unsupported_positive_fraction=unsupported_positives/summary["positive_pairs"],
                       macro_query_history_fraction_without_donor_support=float(np.mean(unsupported_histories)))
        results[key] = summary
    return results


def render(aggregates):
    lines = ["# Other-user donor-data sensitivity", "",
             "Reused-development exploratory result, following the previously evaluated original TEST. "
             "No TEST access or fresh confirmation. Every query retains its complete original TRAIN history; none of the 118 inner or 472 development users contributes a fitting row. "
             "Lambda was selected separately per donor prefix on the fixed inner query users, and all twelve choices were globally sealed before development evaluation.", "",
             "| Seed | Donors | Lambda | Inner nDCG@10 | DEV nDCG@10 | DEV recall@10 | DEV recall@100 | Donor-observed items | Unsupported DEV positives |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, curves in aggregates["seeds"].items():
        for count, result in sorted(curves.items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {seed} | {count} | {result['selected_penalty']:g} | {result['inner_query_ndcg_at_10']:.5f} | {result['ndcg_at_10']:.5f} | {result['macro_conditional_recall@10']:.4f} | {result['macro_conditional_recall@100']:.4f} | {result['donor_observed_items']} | {result['unsupported_positive_pairs']}/{result['positive_pairs']} |")
    lines.extend(["", "| Donors | Equal-seed mean DEV nDCG@10 | Equal-seed mean DEV recall@10 |",
                  "|---|---:|---:|"])
    keys = sorted(next(iter(aggregates["seeds"].values())), key=int)
    for key in keys:
        rows = [seed[key] for seed in aggregates["seeds"].values()]
        lines.append(f"| {key} | {np.mean([row['ndcg_at_10'] for row in rows]):.6f} | {np.mean([row['macro_conditional_recall@10'] for row in rows]):.6f} |")
    lines.extend(["", "More donor rows change both estimation and item coverage, so this experiment cannot separate those effects. "
                  "The observed range ends at 353 other-user donors; its absolute metrics are not comparable to the prior 943-row full-system study. "
                  "No extrapolation, new-data guarantee, metadata conclusion, or claim of an information-theoretic ceiling follows. "
                  "The three splits reuse the same dataset and are not independent replications. All activity/popularity definitions are fixed from the complete original TRAIN population.", ""])
    return "\n".join(lines)


def run(source, output, evidence):
    source, output, evidence = source.resolve(), output.resolve(), evidence.resolve()
    require(source.is_relative_to(ROOT) and not any(part.startswith(("final-", "test")) for part in source.relative_to(ROOT).parts), "Forbidden source root")
    require(not output.exists() and not evidence.exists(), "Refusing overwrite")
    frozen_original, original_protocol, _, _ = check_barrier(source)
    hashes = source_hashes()
    protocol = dict(schema_version=1, study_kind="post_final_test_fixed_query_donor_sensitivity",
                    test_read=False, fresh_confirmation=False, seeds=list(SEEDS), fractions=list(FRACTIONS),
                    donor_counts=[88,176,264,353], lambdas=list(LAMBDAS), split_rng_offset=30011,
                    inner_query_users=118, development_users=472, selection_metric="inner_query_all_observed_ndcg_at_10",
                    selection_tie="first declared lambda", candidate_fits=108,
                    query_history="complete original TRAIN, unchanged at every donor count",
                    source_sha256=hashes, runtime=frozen_original["runtime"],
                    source_study_manifest_sha256=digest(source / "manifest.json"),
                    source_selection_freeze_sha256=digest(source / "SELECTIONS-FROZEN.json"),
                    original_training_source_sha256=original_protocol["source_sha256"],
                    input_signature_sha256={str(seed): digest(source / str(seed) / "input-signature.json") for seed in SEEDS})
    output.mkdir(parents=True)
    json_write(output / "protocol.json", protocol)
    signature = dict(source_sha256=hashes, protocol_sha256=digest(output / "protocol.json"),
                     source_study_manifest_sha256=protocol["source_study_manifest_sha256"])
    started = time.perf_counter()
    manifests = {seed: select_seed(source / str(seed), output / str(seed), seed, signature) for seed in SEEDS}
    # A changed source payload cannot pass the development gate.
    check_barrier(source)
    require(digest(source / "manifest.json") == protocol["source_study_manifest_sha256"], "Source study changed")
    frozen = freeze(output, manifests, signature)
    json_write(output / "DEVELOPMENT-OPENED.json", dict(status="evaluating", test_read=False,
               selection_freeze_sha256=digest(output / "SELECTIONS-FROZEN.json")))
    aggregates = dict(schema_version=1, stage="reused_development", fresh_confirmation=False,
                      study_kind=protocol["study_kind"], test_read=False,
                      dataset_test_previously_evaluated=True,
                      seeds={str(seed): evaluate_seed(source / str(seed), output / str(seed), seed, frozen, output) for seed in SEEDS})
    require(source_hashes() == hashes, "Source changed during development evaluation")
    check_barrier(source)
    json_write(output / "aggregates.json", aggregates)
    json_write(output / "DEVELOPMENT-OPENED.json", dict(status="complete", test_read=False,
               selection_freeze_sha256=digest(output / "SELECTIONS-FROZEN.json")))
    manifest = dict(schema_version=1, status="complete", test_read=False, fresh_confirmation=False,
                    candidate_fits=108, selected_refits=12, selected_serialization_replays=12,
                    elapsed_seconds=time.perf_counter()-started,
                    protocol_sha256=digest(output / "protocol.json"), source_sha256=hashes,
                    selection_freeze_sha256=digest(output / "SELECTIONS-FROZEN.json"),
                    output_sha256={str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*')) if path.is_file()})
    json_write(output / "manifest.json", manifest)
    evidence.mkdir(parents=True)
    for name in ("aggregates.json", "protocol.json", "manifest.json", "SELECTIONS-FROZEN.json"):
        (evidence / name).write_bytes((output / name).read_bytes())
    (evidence / "RESULTS.md").write_text(render(aggregates))
    json_write(evidence / "SHA256.json", {path.name: digest(path) for path in sorted(evidence.iterdir()) if path.is_file()})
    print(f"Completed fixed-query donor curve in {manifest['elapsed_seconds']:.2f}s; {evidence}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out, args.evidence)


if __name__ == "__main__":
    main()
