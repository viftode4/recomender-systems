"""Independent post-run audit of the sealed categorical reconstruction study.

No runner, ranking, calibration, hybrid, or evaluation helper is imported.
Only the model is reused for deterministic TRAIN replay, after the global seal
and completed output manifest pass. Raw rating values are parsed solely for
declared TRAIN/validation pairs; no TEST split or final-result file is opened.
All emitted evidence is aggregate-only. This is reused-development verification,
not another selection stage or fresh assessment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (2026, 2027, 2028)
LAMBDAS = (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)
ORIGINAL = (50., 250., 1000.)
RATIOS = (.1, 1., 10., 100.)
FAMILIES = ("binary_original_grid", "binary_expanded", "categorical", "shuffled_categories")
ARMS = {"baseline2": ("binary", "slim"), "real3": ("binary", "slim", "real"),
        "shuffled3": ("binary", "slim", "shuffled")}
PENALTIES = (.001, .01, .1, 1.)
K = 10


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def array_digest(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256(f"{array.dtype}:{array.shape}:".encode())
    result.update(memoryview(array).cast("B"))
    return result.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=2e-12, rtol=2e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), f"Mismatch: {label}")
    return float(np.max(np.abs(a-b), initial=0.))


def safe_path(directory, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Absolute sealed path")
    parts = Path(relative).parts
    require(not any(p.lower().startswith(("test.", "test-", "final-")) for p in parts),
            "Forbidden TEST/final artifact")
    path = (Path(directory)/relative).resolve()
    require(path.is_relative_to(Path(directory).resolve()), "Sealed path escapes its root")
    return path


def check_hashes(directory, signatures):
    require(isinstance(signatures, dict) and signatures, "Missing sealed hashes")
    for name, expected in signatures.items():
        require(digest(safe_path(directory, name)) == expected, f"Hash mismatch: {name}")


def check_barrier(research):
    """This MUST precede score or pair loading, including meta-only loading."""
    path = research/"SELECTIONS-FROZEN.json"
    require(path.is_file(), "Global selection seal is absent; refusing label access")
    frozen = json_read(path)
    require(frozen.get("status") == "all_selections_frozen"
            and frozen.get("test_read") is False
            and frozen.get("development_evaluated") is False
            and tuple(frozen.get("seeds", [])) == SEEDS, "Invalid global selection seal")
    require(digest(research/"protocol.json") == frozen["protocol_sha256"], "Protocol changed after freeze")
    protocol = json_read(research/"protocol.json")
    require(tuple(protocol["seeds"]) == SEEDS and tuple(protocol["lambdas"]) == LAMBDAS
            and tuple(protocol["category_ratios"]) == RATIOS
            and tuple(protocol["original_binary_lambdas"]) == ORIGINAL
            and protocol["smoothing"] == 20. and protocol["test_read"] is False,
            "Unexpected predeclared study grid")
    require(protocol["source_sha256"] == frozen["source_sha256"], "Protocol/source seal mismatch")
    check_hashes(ROOT, frozen["source_sha256"])
    runtime = frozen["runtime"]
    require(runtime["python"] == platform.python_version() and runtime["numpy"] == np.__version__
            and runtime["scipy"] == scipy.__version__ and runtime["numerical_threads"] == 1
            and runtime["machine"] == platform.machine() and runtime["system"] == platform.system(),
            "Use the frozen numerical runtime for exact replay")
    selections = {}
    for seed in SEEDS:
        directory = research/str(seed)
        require(digest(directory/"selection-manifest.json") == frozen["selection_manifest_sha256"][str(seed)],
                "A seed selection changed after the global seal")
        manifest = json_read(directory/"selection-manifest.json")
        require(manifest["status"] == "selected" and manifest["seed"] == seed
                and manifest["test_read"] is False and manifest["development_evaluated"] is False
                and manifest["source_sha256"] == frozen["source_sha256"]
                and manifest["runtime"] == runtime and manifest["unique_fits"] == 81,
                "Invalid seed selection manifest")
        check_hashes(directory, manifest["output_sha256"])
        selections[seed] = manifest
    require((research/"manifest.json").is_file(), "Completed study manifest is absent")
    completed = json_read(research/"manifest.json")
    require(completed["status"] == "complete" and completed["test_read"] is False
            and completed["test_evaluated"] is False and completed["fresh_confirmation"] is False
            and completed["source_sha256"] == frozen["source_sha256"]
            and completed["runtime"] == runtime
            and completed["selection_freeze_sha256"] == digest(path), "Invalid completed study manifest")
    check_hashes(research, completed["output_sha256"])
    marker = json_read(research/"DEVELOPMENT-OPENED.json")
    require(marker["selection_freeze_sha256"] == digest(path) and marker["test_read"] is False,
            "Development stage does not point to the global seal")
    return frozen, protocol, selections, completed


def pairs(path):
    with Path(path).open() as stream:
        result = [(row["user_id"], row["item_id"]) for row in csv.DictReader(stream, delimiter="\t")]
    require(len(set(result)) == len(result), "Duplicate source interaction")
    return result


def grouping(values):
    result = defaultdict(set)
    for user, item in values:
        result[user].add(item)
    return dict(result)


def ordered_digest(users, items):
    return hashlib.sha256(json.dumps({"users": users, "items": items}, sort_keys=True).encode()).hexdigest()


def load_inputs(directory, selection_manifest, data_path, source_root, seed):
    signature = json_read(directory/"input-signature.json")
    require(signature == selection_manifest["input_signature"], "Input signature changed")
    for name, expected in signature["data_sha256"].items():
        require(Path(name).name == name and Path(name).suffix in (".inter", ".item", ".user"),
                "Unexpected raw dataset file")
        require(digest(data_path/name) == expected, "Raw dataset digest differs")
    for stage in ("train", "valid"):
        require(digest(directory/f"{stage}.tsv") == signature["split_sha256"][stage], "Split changed")
    require(digest(directory/"ids.json") == signature["ids_sha256"], "ID mapping changed")
    require(digest(directory/"cohorts.json") == signature["cohort_file_sha256"], "Cohorts changed")
    source = source_root/f"{seed}-EASE-1"
    require(digest(source/"manifest.json") == signature["source_manifest_sha256"], "Original source changed")
    with np.load(directory/"training-categories.npz", allow_pickle=False) as archive:
        users, items = archive["users"].tolist(), archive["items"].tolist()
        matrix = archive["ratings"].copy()
    require(len(set(users)) == len(users) and len(set(items)) == len(items)
            and items[0] == "[PAD]" and ordered_digest(users, items) == signature["ordered_identity_sha256"],
            "Ordered source identities differ")
    require(matrix.shape == (len(users), len(items))
            and array_digest(matrix) == signature["training_categories_sha256"], "TRAIN categories changed")
    ids = json_read(directory/"ids.json")
    require(ids["items"] == items and [u for u in ids["users"] if u in set(users)] == users,
            "Score identities differ from the copied internal mapping")
    train, valid = pairs(directory/"train.tsv"), pairs(directory/"valid.tsv")
    require(not set(train) & set(valid), "TRAIN/validation overlap")
    user_index, item_index = {u: i for i, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    train_set, valid_set = set(train), set(valid)
    rebuilt = np.zeros(matrix.shape, dtype=matrix.dtype)
    validation = {}
    retained = set()
    rating_files = [name for name in signature["data_sha256"] if name.endswith(".inter")]
    require(len(rating_files) == 1, "Expected exactly one raw interaction file")
    with (data_path/rating_files[0]).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pair = row["user_id:token"], row["item_id:token"]
            if pair not in train_set and pair not in valid_set:
                continue  # Never parse the rating label of any other split.
            require(pair not in retained and pair[0] in user_index and pair[1] in item_index,
                    "Duplicate or unknown authorized rating")
            retained.add(pair)
            rating = float(row["rating:float"])
            require(np.isfinite(rating) and rating == int(rating) and 1 <= rating <= 5,
                    "Invalid authorized rating category")
            if pair in train_set:
                rebuilt[user_index[pair[0]], item_index[pair[1]]] = int(rating)
            else:
                validation[pair] = int(rating)
    require(retained == train_set | valid_set and np.array_equal(rebuilt, matrix),
            "Stored TRAIN matrix does not match authorized raw rating values")
    cohorts = json_read(directory/"cohorts.json")
    shuffled = np.asarray(sorted(users))
    np.random.default_rng(seed).shuffle(shuffled)
    require(cohorts == {"meta_fit": sorted(shuffled[:len(users)//2].tolist()),
                        "development": sorted(shuffled[len(users)//2:].tolist())},
            "Cohort partition is not the declared seeded split")
    history = grouping(train)
    eligible = [np.array([j for j, item in enumerate(items) if j and item not in history[user]], dtype=int)
                for user in users]
    require(all(len(row) >= K for row in eligible), "Too few recommendation candidates")
    truth = grouping(valid)
    likes = grouping(pair for pair, rating in validation.items() if rating >= 4)
    return dict(signature=signature, users=users, items=items, matrix=matrix, train=train,
                valid=valid, history=history, truth=truth, likes=likes, cohorts=cohorts,
                user_index=user_index, item_index=item_index, eligible=eligible)


def rank(scores, eligible):
    result = []
    for row, columns in zip(scores, eligible):
        require(np.isfinite(row[columns]).all(), "Nonfinite eligible score")
        result.append(columns[np.lexsort((columns, -row[columns]))[:K]])
    return np.asarray(result)


def independent_metrics(scores, inputs, cohort, truth=None):
    truth = inputs["truth"] if truth is None else truth
    recommendations = rank(scores, inputs["eligible"])
    values, rows = [], {}
    discounts = 1/np.log2(np.arange(2, K+2))
    for user in sorted(cohort):
        relevant = truth.get(user, set())
        if not relevant:
            continue
        columns = recommendations[inputs["user_index"][user]]
        hit = np.array([inputs["items"][j] in relevant for j in columns], dtype=float)
        record = {"ndcg@10": float(hit @ discounts / discounts[:min(K, len(relevant))].sum()),
                  "recall@10": float(hit.sum()/len(relevant)), "precision@10": float(hit.mean())}
        rows[user] = record
        values.append([record[key] for key in ("ndcg@10", "recall@10", "precision@10")])
    require(values, "No eligible truth-bearing users")
    return {"users": len(values), **dict(zip(("ndcg@10", "recall@10", "precision@10"),
                                              np.mean(values, axis=0).tolist()))}, rows


def shuffled_categories(matrix, seed):
    result = matrix.copy()
    generator = np.random.default_rng(seed+17011)
    for item in range(1, matrix.shape[1]):
        observed = np.flatnonzero(matrix[:, item])
        result[observed, item] = generator.permutation(matrix[observed, item])
    require(np.array_equal(result != 0, matrix != 0), "Shuffle changed observation mask")
    for rating in range(1, 6):
        require(np.array_equal((result == rating).sum(axis=0), (matrix == rating).sum(axis=0)),
                "Shuffle changed item-category histogram")
    return result


def identifier(kind, penalty, ratio):
    return f"{kind}-lambda-{penalty:g}" + ("" if ratio is None else f"-ratio-{ratio:g}")


def check_grid(directory):
    grid, chosen = json_read(directory/"candidate-grid.json"), json_read(directory/"selection.json")
    specs = [("binary", penalty, None) for penalty in LAMBDAS]
    specs += [(kind, penalty, ratio) for kind in ("categorical", "shuffled_categories")
              for penalty in LAMBDAS for ratio in RATIOS]
    require(len(grid) == len(specs) == 81, "Wrong unique candidate count")
    for row, spec in zip(grid, specs):
        require((row["feature_kind"], row["penalty"], row["category_ratio"]) == spec
                and row["candidate_id"] == identifier(*spec), "Candidate order or identity differs")
        require(math.isfinite(row["meta_fit_all_observed_ndcg@10"])
                and 0 <= row["meta_fit_all_observed_ndcg@10"] <= 1, "Invalid candidate metric")
    pools = {"binary_original_grid": [r for r in grid if r["feature_kind"] == "binary" and r["penalty"] in ORIGINAL],
             "binary_expanded": [r for r in grid if r["feature_kind"] == "binary"]}
    pools.update({name: [r for r in grid if r["feature_kind"] in ("binary", name)]
                  for name in ("categorical", "shuffled_categories")})
    require(set(chosen) == set(FAMILIES) | {"hybrid_"+name for name in ARMS}, "Unexpected selected families")
    for family, count in zip(FAMILIES, (3, 9, 45, 45)):
        require(len(pools[family]) == count and chosen[family]["candidate_count"] == count, "Wrong selection-pool size")
        winner = max(pools[family], key=lambda row: row["meta_fit_all_observed_ndcg@10"])
        require(chosen[family]["candidate_id"] == winner["candidate_id"], "Selected row is not the first exact maximum")
        for field in ("feature_kind", "penalty", "category_ratio", "scores_array_sha256", "meta_fit_all_observed_ndcg@10"):
            require(chosen[family][field] == winner[field], "Selected candidate metadata changed")
        require(chosen[family]["selection_cohort"] == "reused_meta_fit", "Unexpected selection cohort")
    return grid, chosen


def selected_arrays(directory, selections, shape):
    arrays = {}
    for name, row in selections.items():
        path = safe_path(directory, row["scores_file"])
        require(digest(path) == row["scores_file_sha256"], "Selected score file changed")
        array = np.load(path, allow_pickle=False)
        require(array.shape == shape and np.isfinite(array).all()
                and array_digest(array) == row["scores_array_sha256"], "Selected score array changed")
        arrays[name] = array
    return arrays


def load_reference(directory, inputs, expected_model, penalty=None):
    """Read old validation scores only; never inspect a TEST split or final file."""
    manifest = json_read(directory/"manifest.json")
    require(manifest["status"] == "complete" and manifest["model"] == expected_model
            and manifest.get("test_evaluated") is False and not manifest.get("test_read", False)
            and manifest.get("validation_used_for_training") is False, "Invalid locked reference")
    if penalty is not None:
        require(manifest["settings"]["reg_weight"] == penalty, "Original EASE penalty differs")
    signature = inputs["signature"]
    require(manifest["data_sha256"] == signature["data_sha256"], "Reference dataset differs")
    for stage in ("train", "valid"):
        require(manifest["split_sha256"][stage] == signature["split_sha256"][stage]
                and digest(directory/f"{stage}.tsv") == signature["split_sha256"][stage], "Reference split differs")
    require(digest(directory/"ids.json") == signature["ids_sha256"], "Reference internal IDs differ")
    path = directory/"valid-scores.npz"
    pinned = manifest.get("export_sha256", {}).get("valid-scores.npz")
    require(pinned is None or digest(path) == pinned, "Reference prediction file changed")
    with np.load(path, allow_pickle=False) as archive:
        require(archive["users"].tolist() == inputs["users"] and archive["items"].tolist() == inputs["items"],
                "Reference ordered IDs differ")
        scores = archive["scores"].astype(np.float64)
    require(scores.shape == inputs["matrix"].shape and np.isfinite(scores).all(), "Invalid reference scores")
    return scores, {"manifest_sha256": digest(directory/"manifest.json"),
                    "scores_sha256": digest(path), "preexisting_export_hash_available": pinned is not None}


def replay_reconstruction(directory, inputs, selection_manifest, grid, selections, arrays, source_root, seed):
    from exploratory.categorical_reconstruction.model import prepare_features, fit
    matrices = {"binary": inputs["matrix"], "categorical": inputs["matrix"],
                "shuffled_categories": shuffled_categories(inputs["matrix"], seed)}
    require(array_digest(matrices["shuffled_categories"]) == selection_manifest["shuffled_training_categories_sha256"],
            "Shuffled categories differ from sealed fit")
    prepared = {"categorical": prepare_features(matrices["categorical"], smoothing=20.),
                "shuffled_categories": prepare_features(matrices["shuffled_categories"], smoothing=20.)}
    prepared["binary"] = prepared["categorical"]
    replay, report = {}, {"selected_replays": {}, "original_ease": {}}
    requested = {(row["feature_kind"], row["penalty"], row["category_ratio"])
                 for name, row in selections.items() if name in FAMILIES}
    requested |= {("binary", penalty, None) for penalty in ORIGINAL}
    for kind, penalty, ratio in sorted(requested, key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2])):
        key = identifier(kind, penalty, ratio)
        scores = fit(prepared[kind], lambda_binary=penalty, category_ratio=ratio).predict()
        replay[key] = scores
        row = next(row for row in grid if row["candidate_id"] == key)
        require(array_digest(scores) == row["scores_array_sha256"], "Deterministic model replay differs from candidate hash")
        aggregate, _ = independent_metrics(scores, inputs, inputs["cohorts"]["meta_fit"])
        close(aggregate["ndcg@10"], row["meta_fit_all_observed_ndcg@10"], "Replayed candidate meta nDCG")
    for name in FAMILIES:
        row = selections[name]
        require(np.array_equal(arrays[name], replay[row["candidate_id"]]), "Selected prediction is not exact fit replay")
        report["selected_replays"][name] = {"candidate_id": row["candidate_id"], "exact": True,
                                           "maximum_absolute_error": 0.}
    report["unique_replayed_configurations"] = len(replay)
    for index, penalty in enumerate(ORIGINAL):
        old, provenance = load_reference(source_root/f"{seed}-EASE-{index}", inputs, "EASE", penalty)
        current = replay[identifier("binary", penalty, None)]
        error = close(current, old, "Binary/locked EASE predictions", atol=5e-5, rtol=5e-5)
        first, second = rank(current, inputs["eligible"]), rank(old, inputs["eligible"])
        exact_rows = np.all(first == second, axis=1)
        same_sets = np.array([set(a) == set(b) for a, b in zip(first, second)])
        ref_aggregate, _ = independent_metrics(old, inputs, inputs["cohorts"]["meta_fit"])
        fit_aggregate, _ = independent_metrics(current, inputs, inputs["cohorts"]["meta_fit"])
        report["original_ease"][str(penalty)] = {
            "maximum_absolute_score_error": error, "score_tolerance_atol_rtol": [5e-5, 5e-5],
            "exact_top10_order_users": int(exact_rows.sum()), "same_top10_set_users": int(same_sets.sum()),
            "users": len(first), "meta_ndcg_difference": fit_aggregate["ndcg@10"]-ref_aggregate["ndcg@10"],
            **provenance}
    return report


def pipeline_reference(features, target, penalty):
    """Independent affine fit and an augmented KKT solve for sum-one ridge."""
    mean_x, mean_y = features.mean(axis=0), float(target.mean())
    centered = features-mean_x
    variance = np.sum(centered*centered, axis=0)
    covariance = centered.T @ (target-mean_y)
    slopes = np.maximum(np.divide(covariance, variance, out=np.zeros_like(covariance), where=variance > 0), 0)
    offsets = mean_y-slopes*mean_x
    calibrated = features*slopes+offsets
    mean_c = calibrated.mean(axis=0)
    z = calibrated-mean_c
    dimensions = features.shape[1]
    kkt = np.zeros((dimensions+1, dimensions+1))
    kkt[:dimensions, :dimensions] = z.T@z/len(target)+penalty*np.eye(dimensions)
    kkt[:dimensions, dimensions] = 1
    kkt[dimensions, :dimensions] = 1
    rhs = np.r_[z.T@(target-mean_y)/len(target), 1.]
    weights = np.linalg.solve(kkt, rhs)[:dimensions]
    intercept = float(mean_y-mean_c@weights)
    return dict(calibration_slopes=slopes, calibration_offsets=offsets, weights=weights,
                intercept=intercept, effective_weights=weights*slopes,
                effective_intercept=float(weights@offsets+intercept))


def replay_hybrids(directory, inputs, selections, arrays, slim_root, seed, selection_manifest):
    selection_file = slim_root/f"expert-selection-{seed}.json"
    record = json_read(selection_file)["SLIMElastic"]
    winner = max(record["candidates"], key=lambda row: row["meta_fit_ndcg"])
    require(record["selected"] == winner, "Locked SLIM selection is not its first exact maximum")
    slim_path = slim_root/Path(winner["path"]).name
    slim, provenance = load_reference(slim_path, inputs, "SLIMElastic")
    sealed_slim = selection_manifest["locked_slim_source"]
    require(provenance["manifest_sha256"] == sealed_slim["manifest_sha256"]
            and provenance["scores_sha256"] == sealed_slim["score_file_sha256"], "Locked SLIM prediction changed")
    require(digest(selection_file) == sealed_slim["selection"]["selection_file_sha256"]
            and winner["settings"] == sealed_slim["selection"]["settings"], "Locked SLIM selection changed")
    copied_slim = np.load(directory/"locked-slim-scores.npy", allow_pickle=False)
    require(np.array_equal(copied_slim, slim), "Copied and original locked SLIM scores differ")
    raw = {"binary": arrays["binary_expanded"], "slim": slim,
           "real": arrays["categorical"], "shuffled": arrays["shuffled_categories"]}
    standardized = {}
    for name, scores in raw.items():
        normalized = np.zeros(scores.shape)
        for row, candidates in enumerate(inputs["eligible"]):
            values = scores[row, candidates]
            normalized[row, candidates] = (values-values.mean())/max(values.std(), 1e-8)
        standardized[name] = normalized
    metadata = json_read(directory/"hybrid-selection.json")
    require(tuple(metadata["penalty_grid"]) == PENALTIES, "Hybrid penalty grid differs")
    meta = sorted(inputs["cohorts"]["meta_fit"])
    rows, columns, target = [], [], []
    rng = np.random.default_rng(seed+24011)
    for user in meta:
        row = inputs["user_index"][user]
        positive = np.array(sorted(inputs["item_index"][i] for i in inputs["truth"][user]), dtype=int)
        available = np.setdiff1d(inputs["eligible"][row], positive, assume_unique=True)
        negative = np.sort(rng.choice(available, min(5*len(positive), len(available)), replace=False))
        selected = np.r_[positive, negative]
        rows.extend([row]*len(selected)); columns.extend(selected.tolist())
        target.extend([1.]*len(positive)+[0.]*len(negative))
    rows, columns, target = np.asarray(rows), np.asarray(columns), np.asarray(target)
    cells = [[inputs["users"][row], inputs["items"][column], int(label)]
             for row, column, label in zip(rows, columns, target)]
    cell_digest = hashlib.sha256(json.dumps(cells, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    require(cell_digest == metadata["sampling"]["sampled_cells_sha256"], "Hybrid sampled examples differ")
    shuffled = np.asarray(meta)
    np.random.default_rng(seed+24012).shuffle(shuffled)
    coefficient = sorted(shuffled[:len(meta)//2].tolist())
    selection = sorted(shuffled[len(meta)//2:].tolist())
    require(metadata["cohorts"]["meta_fit"] == meta
            and metadata["cohorts"]["coefficient_fit"] == coefficient
            and metadata["cohorts"]["penalty_selection"] == selection, "Hybrid inner cohorts differ")
    fitting = np.isin(rows, [inputs["user_index"][u] for u in coefficient])
    report = {}
    for arm, experts in ARMS.items():
        original = metadata["arms"][arm]
        require(original["experts"] == list(experts) and len(original["candidates"]) == 4, "Hybrid arm differs")
        features = np.stack([standardized[name] for name in experts], axis=-1)
        sampled = features[rows, columns]
        values, maximum_parameter_error = [], 0.
        for penalty, candidate in zip(PENALTIES, original["candidates"]):
            require(candidate["penalty"] == penalty, "Hybrid penalty order differs")
            pipeline = pipeline_reference(sampled[fitting], target[fitting], penalty)
            for field, value in pipeline.items():
                maximum_parameter_error = max(maximum_parameter_error,
                    close(value, candidate["coefficient_fit"][field], "Inner hybrid "+field, atol=2e-9, rtol=2e-9))
            scores = features@pipeline["effective_weights"]+pipeline["effective_intercept"]
            value = independent_metrics(scores, inputs, selection)[0]["ndcg@10"]
            close(value, candidate["selection_ndcg@10"], "Inner hybrid selection metric")
            values.append((value, penalty))
        selected_penalty = max(values)[1]
        require(original["selected_penalty"] == selected_penalty
                and selections["hybrid_"+arm]["penalty"] == selected_penalty,
                "Hybrid selection does not use maximum then larger-penalty tie")
        pipeline = pipeline_reference(sampled, target, selected_penalty)
        for field, value in pipeline.items():
            maximum_parameter_error = max(maximum_parameter_error,
                close(value, original["final"][field], "Final hybrid "+field, atol=2e-9, rtol=2e-9))
        scores = features@pipeline["effective_weights"]+pipeline["effective_intercept"]
        masked = np.zeros(scores.shape)
        for row, candidates in enumerate(inputs["eligible"]):
            masked[row, candidates] = scores[row, candidates]
        score_error = close(masked, arrays["hybrid_"+arm], "Hybrid score assembly", atol=2e-9, rtol=2e-9)
        require(np.array_equal(rank(masked, inputs["eligible"]), rank(arrays["hybrid_"+arm], inputs["eligible"])),
                "Independent hybrid fit changes top10 rankings")
        report[arm] = {"selected_penalty": selected_penalty, "candidates": len(values),
                       "maximum_parameter_error": maximum_parameter_error, "maximum_score_error": score_error,
                       "sampled_examples": len(target), "sampled_cells_hash_verified": True,
                       "exact_top10_order": True}
    return report


def audit(args):
    started = time.monotonic()
    research = args.research.resolve()
    require(not args.out.exists(), "Refuse to overwrite audit evidence")
    frozen, protocol, manifests, completed = check_barrier(research)
    freeze_hash, completed_hash = digest(research/"SELECTIONS-FROZEN.json"), digest(research/"manifest.json")
    audit_source_hash = digest(Path(__file__))
    artifact_hashes = {}
    if args.evidence is not None:
        check_hashes(args.evidence, json_read(args.evidence/"SHA256.json"))
        for name in ("aggregates.json", "protocol.json"):
            require(digest(args.evidence/name) == digest(research/name), "Curated evidence differs from sealed raw output")
        provenance = json_read(args.evidence/"provenance.json")
        require(provenance["manifest_sha256"] == completed_hash
                and provenance["selection_freeze_sha256"] == freeze_hash
                and provenance["source_sha256"] == frozen["source_sha256"], "Curated provenance differs")
        artifact_hashes = {name: digest(args.evidence/name) for name in
                           ("aggregates.json", "protocol.json", "provenance.json")}
    published = json_read(research/"aggregates.json")
    result = {"schema_version": 1, "status": "complete", "stage": "reused_development",
              "fresh_confirmation": False, "test_read": False, "test_evaluated": False,
              "audit_source_sha256": audit_source_hash, "source_sha256": frozen["source_sha256"],
              "selection_freeze_sha256": freeze_hash, "completed_manifest_sha256": completed_hash,
              "artifact_sha256": artifact_hashes,
              "raw_aggregate_sha256": digest(research/"aggregates.json"),
              "seeds": {}}
    total_candidates = 0
    for seed in SEEDS:
        directory = research/str(seed)
        inputs = load_inputs(directory, manifests[seed], args.data_path, args.source_root, seed)
        grid, chosen = check_grid(directory)
        total_candidates += len(grid)
        arrays = selected_arrays(directory, chosen, inputs["matrix"].shape)
        replay = replay_reconstruction(directory, inputs, manifests[seed], grid, chosen, arrays, args.source_root, seed)
        slim_root = args.slim_root if seed == 2026 else args.slim_more_root
        hybrids = replay_hybrids(directory, inputs, chosen, arrays, slim_root, seed, manifests[seed])
        details = json_read(directory/"development-metrics.json")
        metrics = {}
        maximum_metric_error = 0.
        for name, scores in arrays.items():
            metrics[name] = {}
            for endpoint, truth in (("all_observed", inputs["truth"]), ("liked_ratings", inputs["likes"])):
                aggregate, per_user = independent_metrics(scores, inputs, inputs["cohorts"]["development"], truth)
                original = details[name][endpoint]
                require(original is not None and original["users"] == aggregate["users"]
                        and set(original["per_user"]) == set(per_user), "Development metric cohort differs")
                for metric in ("ndcg@10", "recall@10", "precision@10"):
                    maximum_metric_error = max(maximum_metric_error,
                        close(aggregate[metric], original["aggregate"][metric], "Detailed development "+metric),
                        close(aggregate[metric], published["seeds"][str(seed)]["models"][name][endpoint][metric],
                              "Published development "+metric))
                    for user, row in per_user.items():
                        close(row[metric], original["per_user"][user][metric], "Per-user development "+metric)
                metrics[name][endpoint] = aggregate
        result["seeds"][str(seed)] = {"candidate_rows": len(grid), "selection_pools": {name: chosen[name]["candidate_count"] for name in FAMILIES},
            "meta_users": len(inputs["cohorts"]["meta_fit"]), "development_users": len(inputs["cohorts"]["development"]),
            "reconstruction": replay, "hybrids": hybrids, "verified_development_metrics": metrics,
            "maximum_metric_absolute_error": maximum_metric_error}
        print(f"Verified seed {seed}: {len(grid)} candidates, selected score replays, references, hybrids and development metrics", flush=True)
    require(total_candidates == 243, "Study does not contain243 declared unique fits")
    result["unique_candidate_rows_verified"] = total_candidates
    result["elapsed_seconds"] = time.monotonic()-started
    result["limits"] = ["Only selected unique configurations and original-grid binary configurations are refitted; all243 grid rows and selection maxima are checked.",
        "Independent metric and hybrid formulas; model refits deliberately reuse the already algebra-tested solver.",
        "Original reference exports may lack a historical prediction-file hash; current hashes and available producer seals are recorded.",
        "Numerical agreement is separated from exact ranking agreement; old float32 scores are checked at atol=rtol=5e-5.",
        "Reused development evidence after earlier TEST exposure; not fresh held-out confirmation."]
    check_barrier(research)
    require(digest(research/"SELECTIONS-FROZEN.json") == freeze_hash
            and digest(research/"manifest.json") == completed_hash
            and digest(Path(__file__)) == audit_source_hash, "Audit inputs or source changed during execution")
    if args.evidence is not None:
        check_hashes(args.evidence, artifact_hashes)
    args.out.mkdir(parents=True, exist_ok=False)
    json_write(args.out/"audit.json", result)
    lines = ["# Independent reconstruction results audit", "", "All checks passed on the completed, globally sealed study. No TEST split or final-result artifact was opened.", "",
             "243 distinct candidate rows, all selection maxima/ties, selected predictions, original-grid EASE compatibility and three hybrid arms were verified.",
             "Development nDCG@10, Recall@10 and Precision@10 were recomputed with independent ranking/metric code, including per-user agreement.", "",
             "| Seed | Selected configurations plus EASE grid replayed | Maximum metric error |", "|---|---:|---:|"]
    for seed, record in result["seeds"].items():
        lines.append(f"| {seed} | {record['reconstruction']['unique_replayed_configurations']} | {record['maximum_metric_absolute_error']:.3g} |")
    lines += ["", "Hybrid verification independently reconstructs sampled cells, user-wise score normalization, monotone affine calibration, sum-one ridge KKT systems, penalty selection and final score assembly.", "",
              "Full score/ranking tolerances, cohort denominators and hashes are in audit.json. This verifies exploratory reused-development results; it does not create a new assessment sample.", ""]
    (args.out/"RESULTS.md").write_text("\n".join(lines))
    json_write(args.out/"SHA256.json", {p.name: digest(p) for p in sorted(args.out.iterdir())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True, help="Directory containing the frozen ml-100k.inter/.item/.user files")
    parser.add_argument("--source-root", type=Path, default=ROOT/"runs/research-v2")
    parser.add_argument("--slim-root", type=Path, default=ROOT/"runs/coverage-v1")
    parser.add_argument("--slim-more-root", type=Path, default=ROOT/"runs/coverage-v1-more")
    parser.add_argument("--evidence", type=Path, help="Optional curated output; explicitly hash-bind it for the report renderer")
    parser.add_argument("--out", type=Path, required=True, help="New aggregate-only audit directory")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
