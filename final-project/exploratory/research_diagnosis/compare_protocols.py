"""Compare frozen TRAIN/VALID protocols; never open TEST payloads or fit models.

Earlier TEST values are read only from previously published aggregate evidence.
The new replay is descriptive and explicitly records hybrid selection overlap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def ranks(scores, mask):
    return np.stack([np.flatnonzero(~row)[np.lexsort((np.flatnonzero(~row),
                    -scores[u, ~row]))[:10]] for u, row in enumerate(mask)])


def metrics(order, truth, rows):
    discount = 1 / np.log2(np.arange(2, 12))
    hit = np.take_along_axis(truth, order, axis=1)
    relevant = truth.sum(axis=1)
    ideal = np.array([discount[:min(10, int(n))].sum() for n in relevant])
    assert np.all(ideal > 0)
    return {"users": len(rows), "validation_observations": int(relevant[rows].sum()),
            "ndcg@10": float(((hit * discount).sum(axis=1) / ideal)[rows].mean()),
            "recall@10": float((hit.sum(axis=1) / relevant)[rows].mean()),
            "precision@10": float(hit.sum(axis=1)[rows].mean() / 10)}


def main(out):
    if out.exists():
        raise ValueError("Use a new diagnostic output")
    provenance = {}
    def document(relative):
        path = ROOT / relative
        provenance[relative] = sha(path)
        return read(path)
    final_ref = document("evidence/final-field-references-v1/aggregates.json")
    final_main = document("evidence/final-primary-v3/aggregates.json")
    old_dev = document("evidence/research-v3/aggregates.json")
    new = document("exploratory/categorical_reconstruction/results-v1/aggregates.json")
    original = document("evidence/field-reference-convergence-v1/aggregates.json")
    del original  # Hash the published matched-reference provenance; values are replayed below.
    frozen_root = document("runs/frozen-v3/manifest.json")
    result = {"stage": "post_test_exploratory_protocol_diagnosis", "test_payload_read": False,
              "published_test_aggregates_read": True, "models_fitted": 0, "seeds": {}}
    for seed in (2026, 2027, 2028):
        tag = str(seed)
        directory = ROOT / "runs/frozen-v3" / tag
        bundle = read(directory / "freeze.json")
        assert sha(directory / "freeze.json") == frozen_root["bundles"][tag]["freeze_sha256"]
        for name in ("frozen.npz", "provenance/cohorts.json"):
            assert sha(directory / name) == bundle["artifacts_sha256"][name]
            provenance[f"runs/frozen-v3/{tag}/{name}"] = sha(directory / name)
        before = read(directory / "provenance/cohorts.json")
        after = document(f"runs/categorical-reconstruction-v1/{tag}/cohorts.json")
        assert set(after["meta_fit"]) == set(before["meta_fit"])
        assert set(after["development"]) == set(before["development"]) | set(before["calibration"])
        with np.load(directory / "frozen.npz", allow_pickle=False) as archive:
            users = archive["users"].tolist()
            train, valid = archive["train_mask"].copy(), archive["valid_mask"].copy()
            assert not np.any(train & valid) and not train[:, 0].any() and not valid[:, 0].any()
            raw = archive["raw_scores"]
            ease = raw[bundle["expert_order"].index(f"{seed}-EASE-1")].astype(float)
            active = bundle["active_experts"]
            base = np.stack([(raw[bundle["expert_order"].index(name)] - archive["score_mean"][j, :, None])
                             / archive["score_scale"][j, :, None] for j, name in enumerate(active)], axis=-1)
        label = bundle["selection"]["families"]["disagreement"]
        spec = bundle["models"][label]
        assert spec["variant"] == "disagreement"
        names = active + ["expert_disagreement"]
        weights = np.array([spec["coefficients"]["weights"][name] for name in names])
        features = np.concatenate((base, base.std(axis=2, keepdims=True)), axis=2)
        hybrid = features @ weights + spec["coefficients"]["intercept"]
        del base, features, raw
        mask = train.copy()
        mask[:, 0] = True
        additional_mask = mask | valid
        cohorts = {"all_users": np.arange(len(users))}
        for name, members in (("new_development", after["development"]),
                              ("old_model_selection", before["development"]),
                              ("old_policy_calibration", before["calibration"]),
                              ("meta_fit", after["meta_fit"])):
            cohorts[name] = np.array([u for u, user in enumerate(users) if user in set(members)])
        models = {}
        for name, scores in (("EASE", ease), ("old_disagreement_hybrid", hybrid)):
            ordering = ranks(scores, mask)
            with_extra_mask = ranks(scores, additional_mask)
            changed = np.any(ordering != with_extra_mask, axis=1)
            removed = np.take_along_axis(valid, ordering, axis=1).sum(axis=1)
            models[name] = {"validation_metrics": {cohort: metrics(ordering, valid, indices)
                           for cohort, indices in cohorts.items()},
                           "mask_only_diagnostics": {cohort: {
                               "users": len(indices),
                               "mean_validation_items_removed_from_top10": float(removed[indices].mean()),
                               "fraction_top10_lists_changed": float(changed[indices].mean()),
                               "mean_train_only_candidates": float((~mask)[indices].sum(axis=1).mean()),
                               "mean_train_plus_valid_candidates": float((~additional_mask)[indices].sum(axis=1).mean())}
                               for cohort, indices in cohorts.items()}}
        # Compare the independently replayed old hybrid on its original choice cohort.
        assert abs(models["old_disagreement_hybrid"]["validation_metrics"]["old_model_selection"]["ndcg@10"]
                   - old_dev[tag][label]["aggregate"]["ndcg@10"]) < 1e-12
        assert abs(models["EASE"]["validation_metrics"]["new_development"]["ndcg@10"]
                   - new["seeds"][tag]["models"]["binary_original_grid"]["all_observed"]["ndcg@10"]) < 1e-12
        result["seeds"][tag] = {"active_hybrid_experts": len(active), "hybrid_label": label,
            "hybrid_validation_choice_overlap_with_new_development": len(set(before["development"]) & set(after["development"])),
            "hybrid_coefficient_fit_overlap_with_new_development": len(set(before["meta_fit"]) & set(after["development"])),
            "train_observations": int(train.sum()), "valid_observations": int(valid.sum()),
            "catalog_items_excluding_padding": train.shape[1]-1,
            "published_final_EASE": final_ref["seeds"][tag]["EASE"],
            "published_final_hybrid_all_observed": final_main["seeds"][tag]["models"][label]["aggregate"]["ndcg@10"],
            "published_final_primary_denominators": final_main["seeds"][tag]["denominators"],
            "models": models}
    result["provenance_sha256"] = provenance
    result["diagnostic_source_sha256"] = sha(Path(__file__))
    for relative, expected in provenance.items():
        assert sha(ROOT / relative) == expected
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote aggregate-only comparability diagnosis to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
