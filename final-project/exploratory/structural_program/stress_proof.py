"""Prospective bounded synthetic stress test; old core/proof remain unchanged."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from .core import (CandidatePool, FittedTree, QueryAtoms, SearchConfig, Tree, atom,
                   calibrate, evaluate, labels_array, log_loss, merge, product, propose_edits, select)
from .synthetic_proof import fit_linear_pairs, linear_features, score

SEEDS = (3101, 3102, 3103, 3104, 3105)
SIZES = (16, 64, 256)
TASKS = ("clean_conjunction", "noisy_conjunction", "correlated_distractors",
         "contradictory_correlates", "xor_misspecified", "no_signal")
METHODS = ("beam", "random_grammar", "selected_unary", "fixed_structure", "linear_and_pairs", "intercept")
EVALUATION_N = 5000
CONFIG = SearchConfig(rounds=3, beam_width=6, max_nodes=7, max_depth=3,
                      max_candidates=320, size_penalty=.003, calibration_ridge=.0001)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(values):
    x = np.ascontiguousarray(values)
    return hashlib.sha256(str(x.shape).encode() + str(x.dtype).encode() + x.tobytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sample(task: str, n: int, seed: int, size: int, partition: str):
    task_index = TASKS.index(task)
    stream = {"fit": 0, "probe": 1, "evaluation": 2}[partition]
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_index, size, stream]))
    latent = rng.integers(0, 2, (n, 2)).astype(float)
    noisy = task in {"noisy_conjunction", "correlated_distractors", "contradictory_correlates"}
    observed = np.logical_xor(latent, rng.random((n, 2)) < (.15 if noisy else 0.)).astype(float)
    x = np.zeros((n, 4, 7))
    x[:, 0, 0], x[:, 1, 1] = observed[:, 0], observed[:, 1]
    x[:, 2, 2] = rng.integers(0, 2, n)
    if task in {"correlated_distractors", "contradictory_correlates"}:
        proxies = np.logical_xor(observed, rng.random((n, 2)) < .05).astype(float)
        if task == "contradictory_correlates":
            proxies = 1. - proxies
        x[:, 2, 3], x[:, 3, 4] = proxies[:, 0], proxies[:, 1]
    else:
        x[:, :, 3:5] = rng.binomial(1, .15, (n, 4, 2))
    x[:, :, 5:] = rng.binomial(1, .15, (n, 4, 2))
    if task == "no_signal":
        probability = np.full(n, .5)
    else:
        truth = (latent[:, 0] != latent[:, 1]) if task == "xor_misspecified" else latent[:, 0] * latent[:, 1]
        probability = .06 + .88 * truth
    y = rng.binomial(1, probability).astype(float)  # after all atom generation
    candidates = tuple(f"{partition}-{i}" for i in range(n))
    return QueryAtoms(x, candidates, ("witness-0", "witness-1", "witness-2", "witness-3")), y, probability


def grammar_space(n_atoms: int, config=CONFIG):
    """Enumerate every canonical syntax tree in the unchanged bounded grammar."""
    by_nodes = {n: set() for n in range(1, config.max_nodes + 1)}
    by_nodes[1].update(atom(a) for a in range(n_atoms))
    if config.max_nodes >= 3 and config.max_depth >= 2:
        by_nodes[3].update(product(a, b) for a in range(n_atoms) for b in range(a, n_atoms))
    for total in range(3, config.max_nodes + 1):
        for left_n in range(1, total - 1):
            right_n = total - 1 - left_n
            for left in by_nodes[left_n]:
                for right in by_nodes[right_n]:
                    candidate = merge(left, right)
                    if candidate.depth <= config.max_depth:
                        by_nodes[total].add(candidate)
    return tuple(sorted(set().union(*by_nodes.values()), key=lambda t: (t.nodes, t.key)))


def random_trees(space, initial, count: int, seed: int):
    if initial not in space or count < 1 or count > len(space):
        raise ValueError("invalid matched random pool budget")
    others = tuple(t for t in space if t != initial)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(others), size=count - 1, replace=False)
    return (initial,) + tuple(others[int(i)] for i in indices)


def robust_calibrate(tree, fit, labels, ridge, fallback_events):
    """Preserve the original solver; repair numerical failures only, same objective."""
    try:
        return calibrate(tree, fit, labels, ridge)
    except RuntimeError as original_error:
        y = labels_array(labels, len(fit.candidate_ids))
        features = evaluate(tree, fit)
        def objective(v):
            logits = v[0] + v[1] * features
            residual = expit(logits) - y
            return (log_loss(logits, y) + ridge * np.dot(v, v) / 2,
                    np.array([residual.mean(), np.mean(residual * features)]) + ridge * v)
        result = minimize(objective, np.array([0., 1.]), jac=True, method="SLSQP",
                          bounds=[(-20., 20.), (0., 40.)], options={"maxiter": 300, "ftol": 1e-13})
        _, gradient = objective(result.x)
        projected_gradient = result.x - np.clip(result.x - gradient, [-20., 0.], [20., 40.])
        residual = float(np.max(np.abs(projected_gradient)))
        if not np.isfinite(result.x).all() or residual >= 1e-5:
            raise RuntimeError(f"same-objective fallback failed, projected residual={residual}") from original_error
        fallback_events.append({"tree": tree.key, "original_error": str(original_error),
                                "solver": "SLSQP", "solver_success": bool(result.success),
                                "projected_gradient_residual": residual, "objective_unchanged": True})
        return FittedTree(tree, float(result.x[0]), float(result.x[1]),
                          log_loss(result.x[0] + result.x[1] * features, y))


def stress_fit_pool(fit, labels, initial, fallback_events, config=CONFIG):
    """Unchanged FIT beam algorithm, with explicit stress-only solver injection."""
    y = labels_array(labels, len(fit.candidate_ids))
    cache = {initial: robust_calibrate(initial, fit, y, config.calibration_ridge, fallback_events)}
    beam = [initial]
    rank = lambda t: (cache[t].fit_loss + config.size_penalty * t.nodes, t.nodes, t.key)
    for _ in range(config.rounds):
        proposals = set(beam)
        for parent in beam:
            for tree in propose_edits(parent, fit.values.shape[2], config):
                if tree not in cache and len(cache) < config.max_candidates:
                    cache[tree] = robust_calibrate(tree, fit, y, config.calibration_ridge, fallback_events)
                if tree in cache:
                    proposals.add(tree)
        beam = sorted(proposals, key=rank)[:config.beam_width]
    candidates = tuple(cache[t] for t in sorted(cache, key=lambda t: (t.nodes, t.key)))
    return CandidatePool(candidates, config, fit.sha256, fit.candidate_ids, fit.evidence_ids,
                         fit.values.shape[2], config.rounds)


def make_pool(trees, fit, labels, fallback_events, config=CONFIG):
    fitted = tuple(robust_calibrate(t, fit, labels, config.calibration_ridge, fallback_events) for t in trees)
    return CandidatePool(fitted, config, fit.sha256, fit.candidate_ids, fit.evidence_ids,
                         fit.values.shape[2], 0)


def frozen_tree(fitted):
    return {"kind": "tree", **fitted.to_dict()}


def replay_logits(record, batch):
    if record["kind"] == "tree":
        fitted = FittedTree(Tree.from_dict(record["tree"]), record["bias"], record["scale"], record["fit_loss"])
        return fitted.logits(batch)
    if record["kind"] == "linear_and_pairs":
        return linear_features(batch) @ np.asarray(record["coefficients"])
    if record["kind"] == "intercept":
        return np.full(len(batch.candidate_ids), record["bias"])
    raise ValueError("unknown frozen predictor")


def run_case(case_dir: Path, task: str, size: int, seed: int, space, evaluation_n=EVALUATION_N):
    case_dir.mkdir(parents=True, exist_ok=False)
    fit, fit_y, _ = sample(task, size, seed, size, "fit")
    probe, probe_y, _ = sample(task, size, seed, size, "probe")
    initial = atom(5)
    timings, records, selections, candidates = {}, {}, {}, {}
    fallback_events = []
    started = time.perf_counter()
    beam_pool = stress_fit_pool(fit, fit_y, initial, fallback_events)
    beam, selections["beam"] = select(beam_pool, probe, probe_y)
    timings["beam"] = time.perf_counter() - started
    records["beam"] = frozen_tree(beam)
    candidates["beam"] = [f.to_dict() for f in beam_pool.candidates]
    started = time.perf_counter()
    random_seed = int(np.random.SeedSequence([seed, TASKS.index(task), size, 93]).generate_state(1)[0])
    drawn = random_trees(space, initial, len(beam_pool.candidates), random_seed)
    random_pool = make_pool(drawn, fit, fit_y, fallback_events)
    random_fit, selections["random_grammar"] = select(random_pool, probe, probe_y)
    timings["random_grammar"] = time.perf_counter() - started
    records["random_grammar"] = frozen_tree(random_fit)
    candidates["random_grammar"] = [f.to_dict() for f in random_pool.candidates]
    started = time.perf_counter()
    unary_pool = make_pool(tuple(atom(a) for a in range(fit.values.shape[2])), fit, fit_y, fallback_events)
    unary, selections["selected_unary"] = select(unary_pool, probe, probe_y)
    timings["selected_unary"] = time.perf_counter() - started
    records["selected_unary"] = frozen_tree(unary)
    started = time.perf_counter()
    fixed = robust_calibrate(initial, fit, fit_y, CONFIG.calibration_ridge, fallback_events)
    records["fixed_structure"] = frozen_tree(fixed)
    timings["fixed_structure"] = time.perf_counter() - started
    started = time.perf_counter()
    coefficients = fit_linear_pairs(fit, fit_y)
    records["linear_and_pairs"] = {"kind": "linear_and_pairs", "coefficients": coefficients.tolist(), "ridge": .001}
    timings["linear_and_pairs"] = time.perf_counter() - started
    started = time.perf_counter()
    rate = (float(fit_y.sum()) + .5) / (len(fit_y) + 1.)
    records["intercept"] = {"kind": "intercept", "bias": float(np.log(rate / (1 - rate))), "beta_prior": [.5, .5]}
    timings["intercept"] = time.perf_counter() - started
    frozen = {"task": task, "size": size, "seed": seed, "models": records,
              "selection_details": selections, "candidate_pools": candidates,
              "random_seed": random_seed, "random_count_exactly_matches_beam": len(drawn) == len(beam_pool.candidates),
              "grammar_space_size": len(space), "fit_sha256": fit.sha256, "probe_sha256": probe.sha256,
              "fit_labels_sha256": array_sha256(fit_y), "probe_labels_sha256": array_sha256(probe_y),
              "fitting_elapsed_seconds": timings, "evaluation_generated": False}
    frozen["numerical_fallback_events"] = fallback_events
    decision_path = case_dir / "selection.json"
    dump(decision_path, frozen)
    decision_sha = sha256(decision_path)
    # Replay persisted choices. Evaluation data are generated only after the file exists.
    persisted = json.loads(decision_path.read_text())
    evaluation, evaluation_y, probability = sample(task, evaluation_n, seed, size, "evaluation")
    result = {"task": task, "size": size, "seed": seed, "selection_sha256": decision_sha,
              "evaluation_n": evaluation_n, "evaluation_sha256": evaluation.sha256,
              "evaluation_labels_sha256": array_sha256(evaluation_y), "models": {}}
    for method in METHODS:
        record = persisted["models"][method]
        fit_loss = log_loss(replay_logits(record, fit), fit_y)
        probe_loss = log_loss(replay_logits(record, probe), probe_y)
        evaluated = score(replay_logits(record, evaluation), evaluation_y, probability)
        result["models"][method] = {**evaluated, "fit_nll": fit_loss, "probe_nll": probe_loss,
                                     "evaluation_minus_probe_nll": evaluated["log_loss"] - probe_loss,
                                     "evaluation_minus_fit_nll": evaluated["log_loss"] - fit_loss,
                                     "candidate_count": len(candidates[method]) if method in candidates else 7 if method == "selected_unary" else 1,
                                     "elapsed_seconds": timings[method],
                                     "expression": record.get("expression"),
                                     "nodes": Tree.from_dict(record["tree"]).nodes if record["kind"] == "tree" else None}
    if sha256(decision_path) != decision_sha:
        raise RuntimeError("frozen choices changed during evaluation")
    dump(case_dir / "results.json", result)
    return result


def aggregate(rows):
    summary = {}
    for task in TASKS:
        summary[task] = {}
        for size in SIZES:
            subset = sorted((r for r in rows if r["task"] == task and r["size"] == size), key=lambda r: r["seed"])
            if len(subset) != len(SEEDS) or tuple(r["seed"] for r in subset) != SEEDS:
                raise ValueError("incomplete predeclared case set")
            cell = {"models": {}, "paired_beam_minus_reference_nll": {}}
            for method in METHODS:
                values = {}
                for metric in ("log_loss", "accuracy", "probability_mse", "evaluation_minus_probe_nll", "candidate_count", "elapsed_seconds"):
                    arr = np.array([r["models"][method][metric] for r in subset])
                    values[metric] = {"mean": float(arr.mean()), "sample_std": float(arr.std(ddof=1)), "values_by_seed": arr.tolist()}
                cell["models"][method] = values
            for reference in METHODS[1:]:
                difference = [r["models"]["beam"]["log_loss"] - r["models"][reference]["log_loss"] for r in subset]
                cell["paired_beam_minus_reference_nll"][reference] = {"values_by_seed": difference, "mean": float(np.mean(difference)),
                                                                       "beam_worse_seeds": int(np.sum(np.array(difference) > 0))}
            summary[task][str(size)] = cell
    return summary


def write_report(out, summary):
    lines = ["# Structural-program stress test", "", "All 90 predeclared cases completed. Values below are evaluation NLL means over five seeds; lower is better.",
             "No novelty or recommender-quality claim follows from these synthetic tasks.", "",
             "| Task | FIT / PROBE each | Beam | Random grammar | Selected unary | Signed linear/pairs | Intercept | Beam eval minus probe |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for size in SIZES:
            models = summary[task][str(size)]["models"]
            losses = [models[m]["log_loss"]["mean"] for m in ("beam", "random_grammar", "selected_unary", "linear_and_pairs", "intercept")]
            optimism = models["beam"]["evaluation_minus_probe_nll"]["mean"]
            lines.append(f"| {task} | {size} | " + " | ".join(f"{v:.4f}" for v in losses) + f" | {optimism:+.4f} |")
    lines += ["", "The complete JSON retains every task, size, seed, reference, selected expression, pool size, runtime and failure. The fixed a5 control is retained there; it is weaker than selecting the best unary atom.",
              "", "Random and beam searches calibrate exactly the same number of unique syntax candidates per case, then use the same PROBE objective. Their size, depth, and functional priors differ, so a beam advantage would not isolate adaptive reasoning. The signed linear/pair reference has different coefficient capacity and ridge, so small NLL differences do not establish controlled superiority.",
              "", "These are supplied synthetic atoms with fully observed binary outcomes. Four witness identities remain fixed; 16/64/256 denotes calibration sample count. Correlated proxies carry no additional target information conditional on the observed signal atoms. Neither a real atom learner nor a recommendation loss is tested."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def verify_prior_case(case_dir: Path, prior_dir: Path):
    """Require exact earlier successful behavior; elapsed time is not deterministic."""
    old_selection = json.loads((prior_dir / "selection.json").read_text())
    new_selection = json.loads((case_dir / "selection.json").read_text())
    for key in ("models", "selection_details", "candidate_pools", "fit_sha256", "probe_sha256",
                "fit_labels_sha256", "probe_labels_sha256", "random_seed"):
        if old_selection[key] != new_selection[key]:
            raise RuntimeError(f"prior case did not replay exactly: {case_dir.name}/{key}")
    old = json.loads((prior_dir / "results.json").read_text())
    new = json.loads((case_dir / "results.json").read_text())
    for obj in (old, new):
        obj.pop("selection_sha256")
        for model in obj["models"].values():
            model.pop("elapsed_seconds")
    if old != new:
        raise RuntimeError(f"prior case metrics changed: {case_dir.name}")
    return {"case": case_dir.name, "predictors_pools_and_metrics_exact": True,
            "prior_selection_sha256": sha256(prior_dir / "selection.json"),
            "prior_results_sha256": sha256(prior_dir / "results.json")}


def run(out: Path, replay_prefix_root: Path | None = None):
    out.mkdir(parents=True, exist_ok=False)
    folder = Path(__file__).parent
    sources = ("core.py", "synthetic_proof.py", "stress_proof.py", "stress_protocol.md")
    source_hashes = {name: sha256(folder / name) for name in sources}
    protocol = {"stage": "prospective_synthetic_stress", "created_utc": datetime.now(timezone.utc).isoformat(),
                "movielens_read": False, "old_test_read": False, "seeds": SEEDS, "sizes": SIZES, "tasks": TASKS,
                "evaluation_n": EVALUATION_N, "config": asdict(CONFIG), "source_sha256": source_hashes,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "platform": platform.platform(), "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}}
    prior_cases = {}
    if replay_prefix_root is not None:
        for completed in sorted(replay_prefix_root.glob("*/results.json")):
            prior_cases[completed.parent.name] = {"selection_sha256": sha256(completed.parent / "selection.json"),
                                                  "results_sha256": sha256(completed)}
        protocol["technical_repair_prior_cases"] = prior_cases
    dump(out / "protocol.json", protocol)
    space = grammar_space(7)
    rows, replay_proofs, fallback_cases = [], [], []
    started = time.perf_counter()
    for task in TASKS:
        for size in SIZES:
            for seed in SEEDS:
                result = run_case(out / f"{task}-{size}-{seed}", task, size, seed, space)
                rows.append(result)
                name = f"{task}-{size}-{seed}"
                if name in prior_cases:
                    proof = verify_prior_case(out / name, replay_prefix_root / name)
                    if (proof["prior_selection_sha256"] != prior_cases[name]["selection_sha256"]
                            or proof["prior_results_sha256"] != prior_cases[name]["results_sha256"]):
                        raise RuntimeError(f"prior evidence changed during rerun: {name}")
                    replay_proofs.append(proof)
                saved = json.loads((out / name / "selection.json").read_text())
                if saved["numerical_fallback_events"]:
                    fallback_cases.append({"case": name, "events": saved["numerical_fallback_events"]})
            print(f"Completed {task}, FIT/PROBE={size}, 5 seeds", flush=True)
    for name, original in source_hashes.items():
        if sha256(folder / name) != original:
            raise RuntimeError(f"source changed during stress run: {name}")
    summary = aggregate(rows)
    dump(out / "all-results.json", rows)
    dump(out / "aggregates.json", summary)
    dump(out / "prefix-replay.json", {"cases": replay_proofs, "all_prior_cases_replayed": len(replay_proofs) == len(prior_cases)})
    dump(out / "numerical-fallbacks.json", fallback_cases)
    write_report(out, summary)
    with (out / "RESULTS.md").open("a") as report:
        report.write(f"\nTechnical repair: {len(replay_proofs)} prior completed cases replayed exactly; the identical-objective fallback was used in {len(fallback_cases)} cases. The incomplete first execution and its sources remain preserved. See protocol amendment, prefix-replay.json and numerical-fallbacks.json.\n")
    dump(out / "manifest.json", {"completed_cases": len(rows), "elapsed_seconds": time.perf_counter() - started,
                                  "grammar_space_size": len(space), "source_sha256": source_hashes,
                                  "outputs_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print(f"Complete: {len(rows)} cases in {time.perf_counter() - started:.1f}s; {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-prefix-root", type=Path)
    args = parser.parse_args()
    run(args.out, args.replay_prefix_root)
