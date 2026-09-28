"""Bounded synthetic proof, including an intentionally misspecified XOR task.

Run from the project root with: python -m exploratory.structural_program.synthetic_proof --out ...
The evaluation sample is constructed only after choices have been written.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .core import (QueryAtoms, SearchConfig, atom, calibrate, evaluate, fit_pool,
                   log_loss, merge, product, select)


def sample(kind: str, n: int, seed: int, partition: str):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (n, 3)).astype(float)
    x = np.zeros((n, 3, 5))
    x[:, 0, 0] = bits[:, 0]
    x[:, 1, 1] = bits[:, 1]
    x[:, 2, 2] = bits[:, 2]
    x[:, :, 3:] = rng.binomial(1, .2, (n, 3, 2))
    if kind == "conjunction":
        truth = bits[:, 0] * bits[:, 1]
    elif kind == "redundancy":
        x[:, :, 1] = x[:, :, 0]
        truth = bits[:, 0]
    elif kind == "xor_misspecified":
        truth = (bits[:, 0] != bits[:, 1]).astype(float)
    else:
        raise ValueError(kind)
    probability = .06 + .88 * truth
    labels = rng.binomial(1, probability).astype(float)
    batch = QueryAtoms(x, tuple(f"{partition}-{j}" for j in range(n)), ("support-A", "support-B", "support-C"))
    return batch, labels, probability


def linear_features(batch):
    k = batch.values.shape[2]
    terms = [atom(a) for a in range(k)] + [product(a, b) for a in range(k) for b in range(a, k)]
    return np.column_stack([np.ones(len(batch.candidate_ids))] + [evaluate(t, batch) for t in terms])


def fit_linear_pairs(fit, labels):
    """Strong fixed-grammar reference: signed coefficients, all unary/pair terms."""
    x = linear_features(fit)
    ridge = .001
    def objective(w):
        logits = x @ w
        return log_loss(logits, labels) + .5 * ridge * np.dot(w, w), x.T @ (expit(logits) - labels) / len(labels) + ridge * w
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def score(logits, labels, true_probability):
    p = expit(logits)
    return {"log_loss": log_loss(logits, labels), "accuracy": float(np.mean((p >= .5) == labels)),
            "probability_mse": float(np.mean((p - true_probability) ** 2))}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    config = SearchConfig(rounds=3, beam_width=6, max_nodes=7, max_depth=3, max_candidates=320)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    protocol = {"stage": "synthetic_exploration", "movielens_read": False, "old_test_read": False,
                "claim": "Selection among supplied operators, not discovery of logic or recommender superiority",
                "grammar": "atom / distinct-witness pair product / max merge",
                "selection": "FIT-only candidate pool and affine calibration; one PROBE choice; EVALUATION generated afterward",
                "candidate_counts": {"fit": 600, "probe": 600, "evaluation": 5000},
                "noise_probability": .06, "config": asdict(config), "source_sha256": source_hashes,
                "reference": "signed logistic regression on all supplied unary and pair features, fixed ridge .001",
                "limitations": ["All tree operations and their witness semantics are supplied by us",
                                "Positive root scale cannot express mixed support and veto or XOR",
                                "Synthetic atoms are supplied perfect latent features plus noise",
                                "No claim that a MovieLens atom builder can supply these features"]}
    dump(out / "protocol.json", protocol)
    results = {}
    for index, kind in enumerate(("conjunction", "redundancy", "xor_misspecified")):
        fit, fit_y, _ = sample(kind, 600, 41 + index, "fit")
        probe, probe_y, _ = sample(kind, 600, 141 + index, "probe")
        initial = merge(atom(0), atom(1)) if kind == "redundancy" else atom(3)
        pool = fit_pool(fit, fit_y, initial, config)
        chosen, selection = select(pool, probe, probe_y)
        fixed = calibrate(initial, fit, fit_y, config.calibration_ridge)
        coefficients = fit_linear_pairs(fit, fit_y)
        frozen_choice = {"selected": chosen.to_dict(), "selection": selection, "initial": initial.to_dict(),
                         "fixed_structure": fixed.to_dict(), "linear_pair_coefficients": coefficients.tolist(),
                         "fit_sha256": fit.sha256, "probe_sha256": probe.sha256}
        choice_path = out / f"{kind}-selection.json"
        dump(choice_path, frozen_choice)
        # The evaluator data and labels do not exist until after this persisted decision.
        heldout, heldout_y, probability = sample(kind, 5000, 1041 + index, "evaluation")
        results[kind] = {"selected_expression": chosen.tree.key, "selected_nodes": chosen.tree.nodes,
                         "candidate_count": len(pool.candidates), "selection_sha256": hashlib.sha256(choice_path.read_bytes()).hexdigest(),
                         "selected_program": score(chosen.logits(heldout), heldout_y, probability),
                         "fixed_structure": score(fixed.logits(heldout), heldout_y, probability),
                         "linear_and_pairs": score(linear_features(heldout) @ coefficients, heldout_y, probability)}
    dump(out / "results.json", results)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
