"""Compare completed replay artifacts without fitting, changing or replacing them.

The reference is verification-only. Outputs contain aggregate differences, never
user/item identities or predictions. A successful comparison does not mean that
the predictions are identical: all observed differences are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_run(directory):
    manifest = read(directory / "manifest.json")
    if manifest["status"] != "complete" or manifest["original_test_read"]:
        raise ValueError("Expected completed validation-only experiment")
    for name, expected in manifest["output_sha256"].items():
        child = (directory / name).resolve()
        if not child.is_relative_to(directory.resolve()) or digest(child) != expected:
            raise ValueError(f"Invalid artifact: {name}")
    opened = read(directory / "ASSESSMENT-OPENED.json")
    if not opened["all_seed_choices_frozen"] or opened["selection_seal_sha256"] != manifest["selection_seal_sha256"]:
        raise ValueError("Assessment seal mismatch")
    return manifest, read(directory / "aggregates.json")


def differences(reference, replay, path=""):
    """Retain every changed leaf, including structure and nonnumeric values."""
    if isinstance(reference, dict) and isinstance(replay, dict):
        result = []
        for key in sorted(set(reference) | set(replay)):
            if key not in reference or key not in replay:
                result.append({"field": path + "/" + key, "reference": reference.get(key), "replay": replay.get(key)})
            else:
                result.extend(differences(reference[key], replay[key], path + "/" + key))
        return result
    if reference == replay:
        return []
    row = {"field": path, "reference": reference, "replay": replay}
    if isinstance(reference, (int, float)) and isinstance(replay, (int, float)):
        row["replay_minus_reference"] = replay - reference
    return [row]


def compare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    old_manifest, old = verify_run(args.reference)
    new_manifest, new = verify_run(args.replay)
    if old_manifest["source_sha256"] != new_manifest["source_sha256"]:
        raise ValueError("Scientific source differs; this is not the declared replay")
    rebuilt = read(args.rebuild_receipt)
    if rebuilt["status"] != "PASS" or rebuilt["expert_count"] != 45 or rebuilt["original_test_evaluated"]:
        raise ValueError("Incomplete or out-of-scope rebuild")
    seed_rows = {}
    for seed in old["seeds"]:
        a, b = old["seeds"][seed], new["seeds"][seed]
        if a["cohort_sizes"] != b["cohort_sizes"]:
            raise ValueError("Cohort size changed")
        if read(args.reference / seed / "ids.json") != read(args.replay / seed / "ids.json"):
            raise ValueError("Catalog or user order changed")
        rankings = {}
        with np.load(args.reference / seed / "predictions.npz", allow_pickle=False) as left, np.load(args.replay / seed / "predictions.npz", allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise ValueError("Different prediction roles")
            for role in left.files:
                x, y = left[role], right[role]
                if x.shape != y.shape:
                    raise ValueError("Different ranking shapes")
                rankings[role] = {
                    "users": len(x), "positions": int(x.size),
                    "users_with_order_difference": int(np.any(x != y, axis=1).sum()),
                    "different_order_positions": int((x != y).sum()),
                    "users_with_set_difference": sum(set(p) != set(q) for p, q in zip(x, y)),
                }
        selections = {}
        for role, choice in a["selections"].items():
            other = b["selections"][role]
            selections[role] = {
                "reference_candidate": choice["candidate_id"],
                "replay_candidate": other["candidate_id"],
                "candidate_identical": choice["candidate_id"] == other["candidate_id"],
                "settings_differences": differences(choice["settings"], other["settings"]),
                "development_differences": differences(choice["development"], other["development"]),
            }
        candidate_rows = []
        if len(a["candidates"]) != len(b["candidates"]):
            raise ValueError("Candidate count changed")
        for left, right in zip(a["candidates"], b["candidates"]):
            if left["candidate_id"] != right["candidate_id"]:
                raise ValueError("Candidate order changed")
            candidate_rows.append({"candidate_id": left["candidate_id"],
                "settings_differences": differences(left["settings"], right["settings"]),
                "development_differences": differences(left["development"], right["development"])})
        seed_rows[seed] = {"cohort_sizes_identical": True, "ordered_ids_identical": True,
            "selected_candidates": selections, "all_candidates": candidate_rows,
            "all_user_top10_comparison": rankings,
            "assessment_differences": differences(a["assessment"], b["assessment"])}
    rows = [{"role": role, "metric": metric, "reference": value,
             "replay": new["mean_metrics"][role][metric],
             "replay_minus_reference": new["mean_metrics"][role][metric] - value}
            for role, metrics in old["mean_metrics"].items() for metric, value in metrics.items()]
    result = {"schema_version": 1, "status": "comparison_complete_with_reported_drift",
        "reference_used_only_after_replay_completed": True,
        "source_archive_sha256": digest(args.source_archive),
        "comparator_sha256": digest(__file__),
        "reference_manifest_sha256": digest(args.reference / "manifest.json"),
        "replay_manifest_sha256": digest(args.replay / "manifest.json"),
        "reference_aggregate_sha256": digest(args.reference / "aggregates.json"),
        "replay_aggregate_sha256": digest(args.replay / "aggregates.json"),
        "rebuild_receipt_sha256": digest(args.rebuild_receipt),
        "scientific_source_sha256": old_manifest["source_sha256"],
        "scientific_sources_identical": True,
        "all_12_selected_candidates_identical": all(v["candidate_identical"] for s in seed_rows.values() for v in s["selected_candidates"].values()),
        "all_mean_metrics_identical": all(v["replay_minus_reference"] == 0 for v in rows),
        "original_test_evaluated": False,
        "fresh_population_assessment": False,
        "seconds": {"source_rebuild": rebuilt["seconds"], "completion_replay": new["timing_seconds"]},
        "prediction_file_agreement": {"identical": sum(r["prediction_file_sha256_matches_original"] for r in rebuilt["experts"]), "total": len(rebuilt["experts"])},
        "mean_metrics": rows, "seeds": seed_rows,
        "limitations": [
            "Material NGCF score/ranking drift is independently documented in source-scores-replay-v1.json; its cause is not established here.",
            "Changing file hashes, identical score arrays, identical recommendations and identical metrics are distinct claims.",
            "Switch settings contain fitted group means; equality of chosen candidate IDs does not imply equality of all fitted diagnostics.",
            "This retrains fixed historical expert configurations, not their earlier full search grids.",
            "The original calibration labels were previously exposed; no fresh generalization or breakthrough claim follows.",
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "all_12_selected_candidates_identical": result["all_12_selected_candidates_identical"],
                      "all_mean_metrics_identical": result["all_mean_metrics_identical"], "sha256": digest(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--rebuild-receipt", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args())
