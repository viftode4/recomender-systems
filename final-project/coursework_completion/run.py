"""Complete explicit lecture hybrids without reopening original TEST evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy

from coursework_completion.models import (rank_scores, mixed_rankings, fit_meta_level,
                                          fit_switch, switch_scores, rrf_scores)
from freeze import score_models
from study import load_genres, read_pairs, grouped
from metrics import evaluate
from exploratory.categorical_reconstruction.group_analysis import analyze_groups

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
FAMILIES = ("mixed", "meta", "tuned_switch", "tuned_rrf")
REFERENCES = ("EASE", "SLIM", "context", "fixed_switch", "fixed_rrf")
ROLES = (*REFERENCES, *FAMILIES)
GRIDS = {"mixed": [(6, 2, 2), (4, 4, 2), (4, 2, 4)], "meta": [.1, 1., 10., 100.],
         "tuned_switch": [2, 3, 4], "tuned_rrf": [10, 60, 100]}
SOURCE_FILES = ("coursework_completion/__init__.py", "coursework_completion/models.py",
                "coursework_completion/run.py", "coursework_completion/PROTOCOL.md",
                "freeze.py", "study.py", "metrics.py", "societal.py", "hybrid_constraints.py",
                "exploratory/categorical_reconstruction/group_analysis.py")
SAFE_ARRAYS = ("users", "items", "raw_scores", "train_mask", "score_mean", "score_scale",
               "activity", "entropy", "popularity", "raw_activity", "genre_binary",
               "genre_fraction", "groups", "training_counts")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "system": platform.system(), "machine": platform.machine(), "numerical_threads": 1}


def safe_child(directory, name):
    directory = Path(directory).resolve()
    path = directory / name
    if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or
            not path.resolve().is_relative_to(directory) or not path.is_file()):
        raise ValueError("Unsafe or missing frozen artifact")
    return path


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {Path(path).name}")


def read_valid_subset(path, allowed_users, item_set=None):
    """Skip other users before interpreting their item field, including PAD tests."""
    allowed = set(allowed_users)
    truth = {}
    with Path(path).open() as stream:
        if stream.readline().rstrip("\r\n") != "user_id\titem_id":
            raise ValueError("Unexpected validation header")
        for line in stream:
            user, separator, remainder = line.partition("\t")
            if user not in allowed:
                continue
            if not separator:
                raise ValueError("Malformed authorized validation row")
            item = remainder.rstrip("\r\n")
            if not item or "\t" in item or item == "[PAD]" or (item_set is not None and item not in item_set):
                raise ValueError("Invalid authorized validation item")
            if item in truth.setdefault(user, set()):
                raise ValueError("Duplicate authorized validation pair")
            truth[user].add(item)
    if set(truth) != allowed:
        raise ValueError("Validation truth is missing authorized users")
    return truth


def load_safe_arrays(path):
    """Never access valid_mask; NPZ integrity is checked separately as opaque bytes."""
    with np.load(path, allow_pickle=False) as archive:
        if not set(SAFE_ARRAYS).issubset(archive.files):
            raise ValueError("Frozen prediction archive is incomplete")
        return {name: archive[name] for name in SAFE_ARRAYS}


def load_seed(source_root, seed, items_path):
    source_root, items_path = Path(source_root).resolve(), Path(items_path).resolve()
    root_manifest = source_root / "manifest.json"
    require_hash(root_manifest, (source_root / "manifest.sha256").read_text().strip())
    master = json.loads(root_manifest.read_text())
    if master.get("status") != "complete" or master.get("test_read") is not False:
        raise ValueError("Require a complete original frozen assembly")
    directory = source_root / str(seed)
    freeze_path = directory / "freeze.json"
    require_hash(freeze_path, (directory / "freeze.sha256").read_text().strip())
    require_hash(freeze_path, master["bundles"][str(seed)]["freeze_sha256"])
    bundle = json.loads(freeze_path.read_text())
    if bundle.get("status") != "frozen" or bundle.get("test_read") is not False or bundle["seed"] != seed:
        raise ValueError("Frozen seed/stage mismatch")
    inputs = {str(root_manifest): digest(root_manifest), str(freeze_path): digest(freeze_path)}
    for name, expected in bundle["code_sha256"].items():
        require_hash(safe_child(ROOT, name), expected)
    for name, expected in bundle["artifacts_sha256"].items():
        path = safe_child(directory, name)
        if path.name == "test.tsv":
            raise ValueError("TEST files are outside this experiment")
        require_hash(path, expected)
        inputs[str(path)] = expected
    study_manifest = json.loads((directory / "provenance/manifest.json").read_text())
    require_hash(items_path, study_manifest["metadata_sha256"])
    inputs[str(items_path)] = digest(items_path)
    arrays = load_safe_arrays(directory / "frozen.npz")
    users, items = arrays["users"].tolist(), arrays["items"].tolist()
    if (len(set(users)) != len(users) or "[PAD]" in users or len(set(items)) != len(items) or
            items[0] != "[PAD]" or arrays["raw_scores"].shape != (len(bundle["expert_order"]), len(users), len(items))):
        raise ValueError("Frozen catalog/prediction shape mismatch")
    train = read_pairs(directory / "train.tsv")
    ui, ii = {user: row for row, user in enumerate(users)}, {item: col for col, item in enumerate(items)}
    observed = np.zeros((len(users), len(items)), dtype=bool)
    for user, item in train:
        if user not in ui or item not in ii or item == "[PAD]" or observed[ui[user], ii[item]]:
            raise ValueError("TRAIN IDs differ from frozen catalog")
        observed[ui[user], ii[item]] = True
    if not np.array_equal(observed, arrays["train_mask"]):
        raise ValueError("TRAIN matrix differs from frozen source")
    if not np.array_equal(observed.sum(axis=1), arrays["raw_activity"]) or not np.array_equal(observed.sum(axis=0), arrays["training_counts"]):
        raise ValueError("Frozen TRAIN summaries differ")
    genre_binary, genre_fraction = load_genres(items_path, items)
    if not np.array_equal(genre_binary, arrays["genre_binary"]) or not np.array_equal(genre_fraction, arrays["genre_fraction"]):
        raise ValueError("Frozen catalog metadata differs")
    cohorts = json.loads((directory / "provenance/cohorts.json").read_text())
    sets = [set(cohorts[name]) for name in ("meta_fit", "development", "calibration")]
    if set.union(*sets) != set(users) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Source user cohorts do not partition the catalog")
    authorized = sets[0] | sets[1]
    truth = read_valid_subset(directory / "valid.tsv", authorized, set(items[1:]))
    history = grouped(train)
    if any(truth[user] & set(history[user]) for user in truth):
        raise ValueError("VALID labels overlap TRAIN")
    counts = Counter(item for _, item in train)
    genres = {item: {str(index) for index in np.flatnonzero(genre_binary[col])} for col, item in enumerate(items) if col}
    return {"seed": seed, "directory": directory, "bundle": bundle, "arrays": arrays,
            "users": users, "items": items, "cohorts": cohorts, "truth": truth,
            "history": history, "counts": counts, "genres": genres, "inputs": inputs,
            "signature": {"freeze_sha256": digest(freeze_path), "root_manifest_sha256": digest(root_manifest),
                          "metadata_sha256": digest(items_path), "users_sha256": hashlib.sha256(json.dumps(users).encode()).hexdigest(),
                          "items_sha256": hashlib.sha256(json.dumps(items).encode()).hexdigest(),
                          "train_sha256": digest(directory / "train.tsv"), "valid_sha256": digest(directory / "valid.tsv")}}


def frozen_context_rrf(data, chunk_size=64):
    """Replay exactly the original valid-phase functions in bounded user chunks."""
    bundle, arrays = data["bundle"], data["arrays"]
    context = bundle["selection"]["families"]["context"]
    selected = dict(bundle)
    selected["models"] = {context: bundle["models"][context], "rrf": bundle["models"]["rrf"]}
    context_scores = np.empty_like(arrays["train_mask"], dtype=float)
    rank_scores_rrf = np.empty_like(context_scores)
    row_arrays = {"users", "train_mask", "activity", "entropy", "raw_activity", "groups"}
    for start in range(0, len(data["users"]), chunk_size):
        part = slice(start, start + chunk_size)
        chunk = {name: value[part] if name in row_arrays else value for name, value in arrays.items()}
        for name in ("raw_scores", "score_mean", "score_scale"):
            chunk[name] = arrays[name][:, part]
        scores, _ = score_models(selected, chunk, "valid")
        context_scores[part] = scores[context]
        rank_scores_rrf[part] = scores["rrf"]
    return context_scores, rank_scores_rrf


def evaluate_rankings(ranks, data, truth):
    recommendations = {user: [data["items"][item] for item in ranks[row]]
                       for row, user in enumerate(data["users"]) if user in truth}
    return evaluate(recommendations, truth, data["history"], data["items"][1:], data["counts"], 10)


def seed_candidates(data, out):
    out.mkdir()
    arrays, bundle, users = data["arrays"], data["bundle"], data["users"]
    if [len(data["cohorts"][name]) for name in ("meta_fit", "development", "calibration")] != [471, 236, 236]:
        raise ValueError("Unexpected declared cohort sizes")
    experts = dict(zip(bundle["expert_order"], arrays["raw_scores"]))
    active = bundle["active_experts"]
    if len(active) != 14 or any(bundle["sources"][name]["manifest"]["model"] == "Random" for name in active):
        raise ValueError("Expected the frozen fourteen non-Random experts")
    aliases = {}
    for family in ("EASE", "SLIMElastic", "GenreContent", "ExactPop"):
        matches = [name for name in active if bundle["sources"][name]["manifest"]["model"] == family]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous locked expert")
        aliases[family] = matches[0]
    observed, activity = arrays["train_mask"], arrays["raw_activity"]
    active_scores = np.stack([experts[name] for name in active])
    meta_truth = {u: data["truth"][u] for u in data["cohorts"]["meta_fit"]}
    dev_truth = {u: data["truth"][u] for u in data["cohorts"]["development"]}
    fit_indices = np.asarray([i for i, user in enumerate(users) if user in meta_truth])
    expert_ranks = {name: rank_scores(experts[name], observed) for name in active}
    fit_values = []
    for name in active:
        detail = evaluate_rankings(expert_ranks[name], data, meta_truth)
        fit_values.append([detail["per_user"][users[index]]["ndcg@10"] for index in fit_indices])
    context, fixed_rrf = frozen_context_rrf(data)
    fixed_switch = np.stack([experts[bundle["models"]["group-switch"]["experts"][str(group)]][row]
                             for row, group in enumerate(arrays["groups"])])
    predictions = {"EASE": expert_ranks[aliases["EASE"]], "SLIM": expert_ranks[aliases["SLIMElastic"]],
                   "context": rank_scores(context, observed), "fixed_rrf": rank_scores(fixed_rrf, observed),
                   "fixed_switch": rank_scores(fixed_switch, observed)}
    references = {"EASE": aliases["EASE"], "SLIM": aliases["SLIMElastic"],
                  "context": bundle["selection"]["families"]["context"],
                  "fixed_rrf": bundle["models"]["rrf"], "fixed_switch": bundle["models"]["group-switch"]}
    candidates, selections, candidate_predictions = [], {}, {}
    for family in FAMILIES:
        winner = None
        for index, setting in enumerate(GRIDS[family]):
            tick = time.monotonic()
            identifier = f"{family}-{index}"
            diagnostics = {}
            if family == "mixed":
                ranks, diagnostics = mixed_rankings(np.stack([experts[aliases[name]] for name in
                                                              ("EASE", "GenreContent", "ExactPop")]), observed, setting)
                settings = {"quotas": list(setting), "expert_order": [aliases[name] for name in ("EASE", "GenreContent", "ExactPop")]}
            elif family == "meta":
                model = fit_meta_level(observed[:, 1:], arrays["genre_fraction"][1:], setting)
                scores = np.zeros(observed.shape, dtype=float)
                scores[:, 1:] = model["scores"]
                ranks = rank_scores(scores, observed)
                settings = {"lambda": setting, "profile_dimension": arrays["genre_fraction"].shape[1],
                            "fit_labels": "binary TRAIN only"}
            elif family == "tuned_switch":
                settings = fit_switch(activity, np.asarray(fit_values), fit_indices, setting, active)
                ranks = rank_scores(switch_scores(experts, activity, settings), observed)
            else:
                settings = {"offset": setting, "expert_order": active}
                scores = rrf_scores(active_scores, observed, setting)
                ranks = rank_scores(scores, observed)
                if setting == 60 and not np.array_equal(ranks, predictions["fixed_rrf"]):
                    raise AssertionError("RRF60 does not replay locked original rankings")
            measured = evaluate_rankings(ranks, data, dev_truth)
            candidate = {"family": family, "candidate_id": identifier, "settings": settings,
                         "development": {"users": len(dev_truth), "aggregate": measured["aggregate"]},
                         "seconds": time.monotonic() - tick, "diagnostics": diagnostics}
            candidates.append(candidate)
            candidate_predictions[identifier] = ranks
            if winner is None or candidate["development"]["aggregate"]["ndcg@10"] > winner["development"]["aggregate"]["ndcg@10"]:
                winner = candidate
                predictions[family] = ranks
            print(json.dumps({"seed": data["seed"], "candidate": identifier,
                              "development_ndcg": candidate["development"]["aggregate"]["ndcg@10"]}), flush=True)
        selections[family] = winner
    write_json(out / "ids.json", {"users": users, "items": data["items"]})
    write_json(out / "cohorts.json", data["cohorts"])
    write_json(out / "input-signature.json", data["signature"])
    write_json(out / "candidate-grid.json", candidates)
    write_json(out / "selections.json", {"families": selections, "references": references})
    np.savez_compressed(out / "predictions.npz", **predictions)
    np.savez_compressed(out / "candidate-predictions.npz", **candidate_predictions)
    manifest = {"seed": data["seed"], "roles": list(ROLES), "candidate_count": len(candidates),
                "selection_cohort": "development", "assessment_cohort": "reused calibration",
                "assessment_labels_parsed": False,
                "artifact_sha256": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "selection-manifest.json", manifest)
    return {"cohort_sizes": {name: len(data["cohorts"][name]) for name in ("meta_fit", "development", "calibration")},
            "input_signature": data["signature"], "candidates": candidates, "selections": selections,
            "references": references}


def check_choices(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "selection-manifest.json").read_text())
    expected_artifacts = {"ids.json", "cohorts.json", "input-signature.json", "candidate-grid.json",
                          "selections.json", "predictions.npz", "candidate-predictions.npz"}
    if (set(manifest["artifact_sha256"]) != expected_artifacts or manifest["seed"] != int(directory.name) or
            manifest["roles"] != list(ROLES) or manifest["selection_cohort"] != "development" or
            manifest["assessment_cohort"] != "reused calibration" or manifest["assessment_labels_parsed"] is not False):
        raise ValueError("Incomplete or inconsistent selection manifest")
    for name, expected in manifest["artifact_sha256"].items():
        require_hash(safe_child(directory, name), expected)
    choices = json.loads((directory / "selections.json").read_text())
    grid = json.loads((directory / "candidate-grid.json").read_text())
    if len(grid) != 13 or manifest["candidate_count"] != 13 or set(choices["families"]) != set(FAMILIES):
        raise ValueError("Incomplete bounded grid")
    if set(choices["references"]) != set(REFERENCES):
        raise ValueError("Missing fixed reference roles")
    for family in FAMILIES:
        pool = [row for row in grid if row["family"] == family]
        if [row["candidate_id"] for row in pool] != [f"{family}-{i}" for i in range(len(GRIDS[family]))]:
            raise ValueError("Candidate order differs")
        parameter = {"mixed": "quotas", "meta": "lambda", "tuned_switch": "group_count", "tuned_rrf": "offset"}[family]
        for candidate, setting in zip(pool, GRIDS[family]):
            expected_setting = list(setting) if family == "mixed" else setting
            if candidate["settings"][parameter] != expected_setting:
                raise ValueError("Candidate setting differs from fixed grid")
            metrics = candidate["development"]["aggregate"]
            if candidate["development"]["users"] != 236 or not metrics or not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("Development metric is nonfinite or uses the wrong cohort")
        expected = max(pool, key=lambda row: row["development"]["aggregate"]["ndcg@10"])
        if choices["families"][family] != expected:
            raise ValueError("Choice is not first exact development maximum")
    with np.load(directory / "predictions.npz", allow_pickle=False) as arrays, np.load(
            directory / "candidate-predictions.npz", allow_pickle=False) as candidates:
        if set(arrays.files) != set(ROLES):
            raise ValueError("Missing selected/reference prediction roles")
        if set(candidates.files) != {row["candidate_id"] for row in grid}:
            raise ValueError("Missing candidate predictions")
        ids = json.loads((directory / "ids.json").read_text())
        for ranks in [*(arrays[role] for role in ROLES), *(candidates[name] for name in candidates.files)]:
            if ranks.shape != (len(ids["users"]), 10) or not np.issubdtype(ranks.dtype, np.integer):
                raise ValueError("Prediction shape/type differs")
            if (ranks <= 0).any() or (ranks >= len(ids["items"])).any() or any(len(set(row)) != 10 for row in ranks):
                raise ValueError("Prediction contains PAD, invalid or duplicate items")
        for family in FAMILIES:
            if not np.array_equal(arrays[family], candidates[choices["families"][family]["candidate_id"]]):
                raise ValueError("Selected predictions differ from selected candidate")
    return manifest


def check_execution(protocol):
    if protocol["source_sha256"] != source_hashes() or protocol["runtime"] != runtime():
        raise ValueError("Sealed source/runtime changed")
    for path, expected in protocol["input_artifacts_sha256"].items():
        require_hash(path, expected)


def write_barrier(out):
    out = Path(out)
    target = out / "SELECTIONS-FROZEN.json"
    if target.exists():
        raise FileExistsError(target)
    protocol = json.loads((out / "protocol.json").read_text())
    check_execution(protocol)
    for seed in SEEDS:
        check_choices(out / str(seed))
    seal = {"status": "all_selections_frozen", "seeds": list(SEEDS),
            "source_sha256": protocol["source_sha256"], "protocol_sha256": digest(out / "protocol.json"),
            "selection_manifest_sha256": {str(seed): digest(out / str(seed) / "selection-manifest.json") for seed in SEEDS},
            "assessment_labels_parsed": False}
    write_json(target, seal)
    (out / "SELECTIONS-FROZEN.sha256").write_text(digest(target) + "\n")
    return seal


def verify_barrier(out):
    out = Path(out)
    require_hash(out / "SELECTIONS-FROZEN.json", (out / "SELECTIONS-FROZEN.sha256").read_text().strip())
    seal = json.loads((out / "SELECTIONS-FROZEN.json").read_text())
    if (seal["status"] != "all_selections_frozen" or seal["seeds"] != list(SEEDS) or
            set(seal["selection_manifest_sha256"]) != {str(seed) for seed in SEEDS}):
        raise ValueError("Incomplete global selection barrier")
    require_hash(out / "protocol.json", seal["protocol_sha256"])
    protocol = json.loads((out / "protocol.json").read_text())
    if seal["source_sha256"] != protocol["source_sha256"] or seal["assessment_labels_parsed"] is not False:
        raise ValueError("Invalid source or stage in global seal")
    check_execution(protocol)
    for seed in SEEDS:
        require_hash(out / str(seed) / "selection-manifest.json", seal["selection_manifest_sha256"][str(seed)])
        check_choices(out / str(seed))
    return protocol


def read_assessment(out, seed):
    out = Path(out)
    protocol = verify_barrier(out)
    cohorts = json.loads((out / str(seed) / "cohorts.json").read_text())
    ids = json.loads((out / str(seed) / "ids.json").read_text())
    return read_valid_subset(Path(protocol["source_root"]) / str(seed) / "valid.tsv",
                             cohorts["calibration"], set(ids["items"][1:]))


def experiment(args):
    if args.out.exists() or args.evidence.exists():
        raise FileExistsError("Use new private and public output directories")
    started = time.monotonic()
    # Collect source manifests without parsing any calibration item values.
    data_by_seed, input_hashes = {}, {}
    for seed in SEEDS:
        data = load_seed(args.source_root, seed, args.items)
        input_hashes.update(data["inputs"])
        # Do not retain three full expert-score cubes in memory at once.
        data["arrays"].pop("raw_scores")
        data_by_seed[seed] = data
    protocol = {"schema_version": 1, "study": "coursework_hybrid_completion_reused_calibration",
                "seeds": list(SEEDS), "grids": GRIDS, "roles": list(ROLES), "candidate_count": 39,
                "source_root": str(args.source_root.resolve()), "items": str(args.items.resolve()),
                "source_sha256": source_hashes(), "runtime": runtime(), "input_artifacts_sha256": input_hashes,
                "original_test_read": False, "fresh_holdout": False,
                "assessment_stage": "exploratory reused validation calibration cohort",
                "prior_exposure": "These labels previously fit societal policies and were reused in exploratory studies.",
                "command": sys.argv}
    args.out.mkdir(parents=True)
    write_json(args.out / "protocol.json", protocol)
    summaries = {}
    for seed in SEEDS:
        check_execution(protocol)
        with np.load(data_by_seed[seed]["directory"] / "frozen.npz", allow_pickle=False) as archive:
            data_by_seed[seed]["arrays"]["raw_scores"] = archive["raw_scores"]
        summaries[str(seed)] = seed_candidates(data_by_seed[seed], args.out / str(seed))
        data_by_seed[seed]["arrays"].pop("raw_scores")
    write_barrier(args.out)
    verify_barrier(args.out)
    write_json(args.out / "ASSESSMENT-OPENED.json", {"status": "started", "selection_seal_sha256": digest(args.out / "SELECTIONS-FROZEN.json"),
                                                   "all_seed_choices_frozen": True, "original_test_read": False})
    for seed in SEEDS:
        data = data_by_seed[seed]
        truth = read_assessment(args.out, seed)
        assessed, individual = {}, {}
        with np.load(args.out / str(seed) / "predictions.npz", allow_pickle=False) as archive:
            for role in ROLES:
                ranks = archive[role]
                recommendations = {user: [data["items"][item] for item in ranks[row]]
                                   for row, user in enumerate(data["users"]) if user in truth}
                assessed[role] = analyze_groups(recommendations, truth, data["history"], data["items"][1:],
                                               data["counts"], data["genres"], k=10, group_users=data["users"])
                assessed[role]["positive_pairs"] = sum(map(len, truth.values()))
                individual[role] = evaluate_rankings(ranks, data, truth)["per_user"]
        write_json(args.out / str(seed) / "assessment-metrics.json", assessed)
        write_json(args.out / str(seed) / "assessment-per-user.json", individual)
        summaries[str(seed)]["assessment"] = assessed
    verify_barrier(args.out)
    means = {role: {metric: float(np.mean([summaries[str(seed)]["assessment"][role]["aggregate"][metric] for seed in SEEDS]))
                    for metric in summaries[str(SEEDS[0])]["assessment"][role]["aggregate"]} for role in ROLES}
    result = {"schema_version": 1, "status": "complete", "study": protocol["study"],
              "original_test_read": False, "fresh_holdout": False, "assessment_stage": protocol["assessment_stage"],
              "prior_exposure": protocol["prior_exposure"], "seeds": summaries, "mean_metrics": means,
              "timing_seconds": time.monotonic() - started,
              "limitations": ["Calibration cohorts were previously exposed; this is exploratory reuse.",
                              "Overlapping split seeds are not independent population replications.",
                              "Original frozen TEST tables have different cohorts/masks and are not matched comparison values.",
                              "Family coverage is the aim; improvements and novelty are not required or claimed."]}
    write_json(args.out / "aggregates.json", result)
    write_json(args.out / "ASSESSMENT-OPENED.json", {"status": "complete", "selection_seal_sha256": digest(args.out / "SELECTIONS-FROZEN.json"),
                                                   "all_seed_choices_frozen": True, "original_test_read": False})
    manifest = {"status": "complete", "source_sha256": protocol["source_sha256"], "runtime": protocol["runtime"],
                "protocol_sha256": digest(args.out / "protocol.json"), "aggregate_sha256": digest(args.out / "aggregates.json"),
                "selection_seal_sha256": digest(args.out / "SELECTIONS-FROZEN.json"),
                "original_test_read": False, "fresh_holdout": False, "assessment_after_all_seed_selections": True,
                "output_sha256": {str(p.relative_to(args.out)): digest(p) for p in sorted(args.out.rglob("*")) if p.is_file()}}
    write_json(args.out / "manifest.json", manifest)
    args.evidence.mkdir(parents=True)
    for name, value in (("aggregates.json", result), ("protocol.json", protocol), ("provenance.json", manifest)):
        write_json(args.evidence / name, value)
    write_json(args.evidence / "SHA256.json", {name: digest(args.evidence / name) for name in
                                               ("aggregates.json", "protocol.json", "provenance.json")})
    print(json.dumps({"status": "complete", "seconds": result["timing_seconds"], "mean_metrics": means}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "items", "out", "evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    experiment(parser.parse_args())


if __name__ == "__main__":
    main()
