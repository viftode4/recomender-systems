"""Run the declared finite library-query experiment in a new output directory.

The oracle owns hidden training functions. Query policies receive only Evidence.
Transfer evaluation chooses one hard program from each support set, then scores
only its complement. This module does not run experiments when imported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np


SEEDS = tuple(range(4100, 4110))
BUDGETS = (0, 2, 4, 8, 12)
ARMS = ("entropy", "library_information", "random")
SUPPORT_SIZES = (2, 4)
N_TASKS = 3
N_INPUTS = 8
N_TABLES = 16
N_PROGRAMS = 18
ROOT = Path(__file__).resolve().parent


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      default=json_value, separators=(",", ":"))


def write_json(path, value, *, replace=False):
    mode = "w" if replace else "x"
    with Path(path).open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True,
                                allow_nan=False, default=json_value) + "\n")


def append_record(handle, value):
    handle.write(encoded(value) + "\n")
    handle.flush()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def generator(seed, regime, table_key, stream):
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(regime), int(table_key), int(stream)]))


def load_core(path):
    spec = importlib.util.spec_from_file_location("library_query_experiment_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load core module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def truth_key(values):
    return tuple(int(x) for x in values)


def make_world(core, seed, regime, table_key):
    """The oracle alone uses the sampled descriptors and complete truth vectors."""
    programs = generator(seed, regime, table_key, 0).integers(0, N_PROGRAMS, N_TASKS)
    initial_inputs = generator(seed, regime, table_key, 1).integers(0, N_INPUTS, N_TASKS)
    random_order = generator(seed, regime, table_key, 2).permutation(N_TASKS * N_INPUTS)
    if regime == 0:
        tables = np.full(N_TASKS, table_key, dtype=np.int64)
    else:
        tables = generator(seed, regime, table_key, 3).integers(0, N_TABLES, N_TASKS)
    labels = np.asarray(core.PREDICTIONS)[tables, programs].copy()
    return {"seed": int(seed), "regime": "shared" if regime == 0 else "no_sharing",
            "table_key": int(table_key), "training_tables": tables.tolist(),
            "training_programs": programs.tolist(), "initial_inputs": initial_inputs.tolist(),
            "random_order": random_order.tolist(), "oracle_labels": labels}


def choose_from_observations(core, posterior, observed, arm, random_order):
    """No hidden labels, target table or descriptor identities enter a policy."""
    available = [(task, x) for task in range(N_TASKS) for x in range(N_INPUTS)
                 if not observed[task, x]]
    if not available:
        raise RuntimeError("The declared budget exceeds the unobserved input pairs")
    if not posterior.valid:
        return available[0], None, 0, "empty_joint_posterior_lexicographic_fallback"
    if arm == "random":
        for pair in random_order:
            task, x = divmod(int(pair), N_INPUTS)
            if not observed[task, x]:
                return (task, x), None, 0, "random_permutation"
        raise RuntimeError("Random query permutation has no remaining input")
    if arm not in ("entropy", "library_information"):
        raise ValueError(f"Unknown arm: {arm}")
    candidates = []
    for task, x in available:
        stats = core.query_stats(posterior, task, x)
        score = float(stats.output_entropy if arm == "entropy" else stats.library_information)
        if not np.isfinite(score):
            raise FloatingPointError("A query acquisition score is nonfinite")
        candidates.append(((task, x), score, stats))
    maximum = max(row[1] for row in candidates)
    best, _, best_stats = next(row for row in candidates
                               if maximum - row[1] <= core.QUERY_TIE_TOLERANCE)
    return best, {"p_y1": float(best_stats.p_y1),
                  "output_entropy": float(best_stats.output_entropy),
                  "library_information": float(best_stats.library_information)}, len(available), arm


def acquire(core, world, arm, failures):
    """Run all 12 extra queries once and freeze MAP choices at fixed budgets."""
    began = time.perf_counter()
    evidence = core.Evidence.empty(N_TASKS)
    observed = np.zeros((N_TASKS, N_INPUTS), dtype=bool)
    queries = []
    for task, x in enumerate(world["initial_inputs"]):
        label = int(world["oracle_labels"][task, x])
        evidence = evidence.observe(task, x, label)
        observed[task, x] = True
        queries.append({"task": task, "input": int(x), "label": label, "kind": "initial"})
    posterior = core.exact_posterior(evidence)
    counters = {"posterior_requests": 1, "query_statistics_requests": 0,
                "logical_predictive_entries_requested": 0,
                "logical_observed_candidate_comparisons": N_TABLES * N_PROGRAMS * N_TASKS,
                "posterior_prediction_entries_scanned": N_TABLES * N_PROGRAMS * N_TASKS * N_INPUTS,
                "predictive_entropy_requests": 0, "canonical_conditional_entropy_requests": 0,
                "concrete_conditional_entropy_requests": 0}
    checkpoints = {}
    first_failure = None

    def capture(extra):
        selected = int(core.select_map_library(posterior)) if posterior.valid else None
        probabilities = np.asarray(posterior.p_library, dtype=float)
        if probabilities.shape != (N_TABLES,) or not np.isfinite(probabilities).all():
            raise ValueError("Core returned invalid concrete-table probabilities")
        canonical = defaultdict(float)
        for table, probability in enumerate(probabilities):
            canonical[int(core.canonical_table(table))] += float(probability)
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * np.log2(positive)).sum()) if posterior.valid else None
        class_positive = np.asarray([mass for mass in canonical.values() if mass > 0])
        class_entropy = float(-(class_positive * np.log2(class_positive)).sum()) if posterior.valid else None
        checkpoints[extra] = {"extra_queries": extra, "oracle_queries": N_TASKS + extra,
                              "abstained": not bool(posterior.valid), "selected_table": selected,
                              "table_probabilities": probabilities.tolist(),
                              "canonical_probabilities": dict(canonical),
                              "table_entropy_bits": entropy, "canonical_entropy_bits": class_entropy,
                              "first_failure_extra_queries": first_failure, "queries": list(queries),
                              "cost": dict(counters), "elapsed_seconds": time.perf_counter() - began}

    def record_empty(extra):
        nonlocal first_failure
        if not posterior.valid:
            if world["regime"] == "shared":
                raise RuntimeError("Shared generating hypothesis disappeared from the exact posterior")
            if first_failure is None:
                first_failure = extra
            append_record(failures, {"kind": "empty_joint_posterior", "seed": world["seed"],
                          "regime": world["regime"], "table_key": world["table_key"],
                          "arm": arm, "extra_queries": extra, "bounded_loss_on_abstention": 1,
                          "action": "abstain; further queries use the first unobserved pair"})

    record_empty(0)
    capture(0)
    for extra in range(1, max(BUDGETS) + 1):
        query_started = time.perf_counter()
        pair, stats, requests, reason = choose_from_observations(
            core, posterior, observed, arm, world["random_order"])
        task, x = pair
        # Only this oracle boundary accesses the full hidden training function.
        label = int(world["oracle_labels"][task, x])
        evidence = evidence.observe(task, x, label)
        observed[task, x] = True
        posterior = core.exact_posterior(evidence)
        counters["posterior_requests"] += 1
        counters["query_statistics_requests"] += requests
        counters["logical_predictive_entries_requested"] += requests * N_TABLES * N_PROGRAMS
        counters["logical_observed_candidate_comparisons"] += int(observed.sum()) * N_TABLES * N_PROGRAMS
        counters["posterior_prediction_entries_scanned"] += N_TABLES * N_PROGRAMS * N_TASKS * N_INPUTS
        for name in ("predictive_entropy_requests", "canonical_conditional_entropy_requests",
                     "concrete_conditional_entropy_requests"):
            counters[name] += requests
        queries.append({"task": task, "input": x, "label": label, "kind": "acquired",
                        "reason": reason, "statistics": stats, "statistics_requests": requests,
                        "logical_predictive_entries_requested": requests * N_TABLES * N_PROGRAMS,
                        "elapsed_seconds": time.perf_counter() - query_started})
        record_empty(extra)
        if extra in BUDGETS:
            capture(extra)
    return checkpoints


class TransferEvaluator:
    """Exact support-complement risk, cached only by frozen model and target."""

    def __init__(self, core):
        self.predictions = np.asarray(core.PREDICTIONS, dtype=np.uint8)
        if self.predictions.shape != (N_TABLES, N_PROGRAMS, N_INPUTS):
            raise ValueError("Core prediction shape differs from the declared grammar")
        if not np.isin(self.predictions, [0, 1]).all():
            raise ValueError("The finite grammar must output Boolean truth vectors")
        self.calls = np.asarray([program.calls for program in core.PROGRAMS], dtype=np.int64)
        if self.calls.shape != (N_PROGRAMS,):
            raise ValueError("Core program count differs from the declared grammar")
        self.supports = {size: np.asarray(list(itertools.combinations(range(N_INPUTS), size)),
                                         dtype=np.int64) for size in SUPPORT_SIZES}
        self.models = {}
        for table in range(N_TABLES):
            order = sorted(range(N_PROGRAMS), key=lambda p: (
                int(self.calls[p]), truth_key(self.predictions[table, p]), p))
            self.models[table] = (self.predictions[table, order],
                                  [(table, p) for p in order], self.calls[order])
        all_pairs = sorted(itertools.product(range(N_TABLES), range(N_PROGRAMS)),
                           key=lambda tp: (int(self.calls[tp[1]]),
                                           truth_key(self.predictions[tp[0], tp[1]]), tp[0], tp[1]))
        self.models["all_tables"] = (
            np.asarray([self.predictions[t, p] for t, p in all_pairs]), all_pairs,
            np.asarray([self.calls[p] for _, p in all_pairs]))
        self.cache = {}
        self.cache_requests = 0
        self.cache_hits = 0
        self.actual_candidate_support_entries = 0
        self.actual_evaluated_support_sets = 0

    def fit_supports(self, model, support_indices, support_labels):
        """Select programs using only support labels and fixed model tie order."""
        predictions = self.models[model][0]
        indices = np.asarray(support_indices, dtype=np.int64)
        labels = np.asarray(support_labels)
        if indices.ndim != 2 or labels.shape != indices.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Support indices and Boolean labels must have equal two-dimensional shapes")
        support_errors = (predictions[:, indices] != labels[None, :, :]).sum(axis=2)
        selected = np.argmin(support_errors, axis=0)
        return selected, support_errors[selected, np.arange(len(indices))]

    def evaluate(self, model, truth, size):
        self.cache_requests += 1
        key = (model, truth_key(truth), int(size))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        supports = self.supports[size]
        if model is None:
            result = {"bounded_loss": 1.0, "conditional_error": None,
                      "abstained": True, "support_sets": len(supports),
                      "support_error": None, "exact_complement_accuracy": None,
                      "support_error_count_total": None, "mean_support_error_count": None,
                      "fits_with_nonzero_support_error": None,
                      "mean_program_calls": None}
        else:
            predictions, descriptors, calls = self.models[model]
            mismatch = predictions != np.asarray(truth, dtype=np.uint8)[None, :]
            selected, selected_support_errors = self.fit_supports(
                model, supports, np.asarray(truth, dtype=np.uint8)[supports])
            remaining_errors = mismatch.sum(axis=1)[selected] - selected_support_errors
            risks = remaining_errors / (N_INPUTS - size)
            first_table, first_program = descriptors[int(selected[0])]
            result = {"bounded_loss": float(risks.mean()), "conditional_error": float(risks.mean()),
                      "abstained": False, "support_sets": len(supports),
                      "support_error": float((selected_support_errors / size).mean()),
                      "support_error_count_total": int(selected_support_errors.sum()),
                      "mean_support_error_count": float(selected_support_errors.mean()),
                      "fits_with_nonzero_support_error": int((selected_support_errors > 0).sum()),
                      "exact_complement_accuracy": float((remaining_errors == 0).mean()),
                      "mean_program_calls": float(calls[selected].mean()),
                      "first_support_example": {"support_indices": supports[0].tolist(),
                          "support_labels": np.asarray(truth)[supports[0]].tolist(),
                          "selected_table": int(first_table), "selected_program": int(first_program),
                          "selected_predictions": predictions[int(selected[0])].tolist(),
                          "remaining_error_count": int(remaining_errors[0])}}
            self.actual_candidate_support_entries += int(len(predictions) * supports.size)
            self.actual_evaluated_support_sets += int(len(supports))
        self.cache[key] = result
        return result

    def universe(self, true_table, training_truths):
        unique = {truth_key(row) for row in self.predictions[true_table]}
        excluded = {truth_key(row) for row in training_truths}
        return sorted(unique - excluded), len(unique), len(unique & excluded)

    def aggregate(self, selected_table, true_table, truths, size):
        if not truths:
            return {"truth_functions": 0, "status": "empty_transfer_universe",
                    "learned": None, "all_tables": None, "oracle": None}
        output = {"truth_functions": len(truths), "status": "evaluated"}
        for name, model in (("learned", selected_table), ("all_tables", "all_tables"),
                            ("oracle", int(true_table))):
            rows = [self.evaluate(model, truth, size) for truth in truths]
            averages = {}
            for field in ("bounded_loss", "conditional_error", "support_error",
                          "mean_support_error_count", "exact_complement_accuracy", "mean_program_calls"):
                values = [row[field] for row in rows if row[field] is not None]
                averages[field] = float(np.mean(values)) if values else None
            averages.update({"abstained": bool(rows[0]["abstained"]),
                             "support_sets_per_function": len(self.supports[size]),
                             "support_sets_total": len(rows) * len(self.supports[size]),
                             "support_error_count_total": sum(row["support_error_count_total"] for row in rows)
                                 if not rows[0]["abstained"] else None,
                             "fits_with_nonzero_support_error": sum(row["fits_with_nonzero_support_error"] for row in rows)
                                 if not rows[0]["abstained"] else None,
                             "prediction_coverage": 0.0 if rows[0]["abstained"] else 1.0})
            output[name] = averages
        return output

    def costs(self):
        return {"cache_requests": self.cache_requests, "cache_hits": self.cache_hits,
                "unique_cached_evaluations": len(self.cache),
                "actual_candidate_support_entries": self.actual_candidate_support_entries,
                "actual_evaluated_support_sets": self.actual_evaluated_support_sets,
                "definition": "Cached support fitting uses fixed prediction lookups, not new operator execution"}


def table_weighted_mean(rows, value):
    grouped = defaultdict(list)
    for row in rows:
        item = value(row)
        if item is not None:
            grouped[int(row["true_table"])].append(float(item))
    table_means = {str(table): float(np.mean(values)) for table, values in sorted(grouped.items())}
    return {"mean": float(np.mean(list(table_means.values()))) if table_means else None,
            "available_tables": len(table_means), "table_means": table_means,
            "cells_per_table": {str(table): len(values) for table, values in sorted(grouped.items())}}


def summarize(records, worlds, evaluator, elapsed):
    groups = defaultdict(list)
    lookup = {}
    for record in records:
        key = (record["regime"], record["arm"], record["extra_queries"])
        groups[key].append(record)
        lookup[(record["regime"], record["seed"], record["true_table"],
                record["arm"], record["extra_queries"])] = record
    metrics = []
    comparisons = []
    for (regime, arm, budget), rows in sorted(groups.items()):
        for size in SUPPORT_SIZES:
            support = str(size)
            eligible = [row for row in rows if row["transfer"][support]["truth_functions"]]
            entry = {"regime": regime, "arm": arm, "extra_queries": budget, "support_size": size,
                     "cells": len(rows), "available_transfer_cells": len(eligible),
                     "empty_transfer_cells": len(rows) - len(eligible),
                     "empty_joint_posterior_cells": sum(row["abstained"] for row in rows),
                     "abstention_rate_all_cells": float(np.mean([row["abstained"] for row in rows])),
                     "controls": {}}
            unique_worlds = {(row["seed"], row["training_world_key"]): row for row in rows}
            entry["acquisition_training_worlds"] = len(unique_worlds)
            entry["failed_training_worlds"] = sum(row["abstained"] for row in unique_worlds.values())
            for name in ("learned", "all_tables", "oracle"):
                entry["controls"][name] = {
                    field: table_weighted_mean(eligible, lambda row, field=field, name=name:
                                               row["transfer"][support][name][field])
                    for field in ("bounded_loss", "conditional_error", "prediction_coverage",
                                  "support_error", "mean_support_error_count",
                                  "exact_complement_accuracy", "mean_program_calls")}
            metrics.append(entry)
    for regime in ("shared", "no_sharing"):
        for budget in BUDGETS:
            focused = groups[(regime, "library_information", budget)]
            for size in SUPPORT_SIZES:
                support = str(size)
                differences = []
                for row in focused:
                    if not row["transfer"][support]["truth_functions"]:
                        continue
                    other = lookup[(regime, row["seed"], row["true_table"], "entropy", budget)]
                    differences.append({"true_table": row["true_table"],
                        "difference": row["transfer"][support]["learned"]["bounded_loss"] -
                                      other["transfer"][support]["learned"]["bounded_loss"]})
                comparisons.append({"regime": regime, "extra_queries": budget, "support_size": size,
                    "contrast": "library_information_minus_entropy_bounded_loss",
                    "paired_cells": len(differences),
                    **table_weighted_mean(differences, lambda row: row["difference"])})
    final_cost = {"posterior_requests": 0, "query_statistics_requests": 0,
                  "logical_predictive_entries_requested": 0,
                  "logical_observed_candidate_comparisons": 0, "oracle_queries": 0,
                  "posterior_prediction_entries_scanned": 0, "predictive_entropy_requests": 0,
                  "canonical_conditional_entropy_requests": 0,
                  "concrete_conditional_entropy_requests": 0}
    for world in worlds:
        for arm in ARMS:
            final = world["arms"][arm][max(BUDGETS)]
            for name in final_cost:
                final_cost[name] += final["oracle_queries"] if name == "oracle_queries" else final["cost"][name]
    return {"schema_version": 1, "status": "complete", "seeds": list(SEEDS),
            "budgets": list(BUDGETS), "support_sizes": list(SUPPORT_SIZES),
            "worlds": {"shared": 160, "no_sharing": 10,
                       "no_sharing_transfer_cells": 160,
                       "note": "The 16 no-sharing transfer-table cells reuse each seed's one training world"},
            "weighting": "Equal functions and support subsets within a cell; mean available seeds per target table; equal available target tables",
            "empty_universes": "N/A with explicit coverage; never counted as successful predictions",
            "abstention": "Loss one in the bounded-loss metric; excluded only from separately reported conditional error",
            "uncertainty": "No confidence intervals; reused transfer cells are not independent training worlds",
            "metrics": metrics, "focused_minus_entropy": comparisons,
            "costs": {"common_precomputation": {"candidate_output_evaluations": 2304,
                         "operator_calls": 3840, "computed_once_per_process": True},
                      "actual_acquisition_runs": len(worlds) * len(ARMS),
                      "acquisition_totals": final_cost, "transfer": evaluator.costs(),
                      "elapsed_seconds": elapsed,
                      "comparison_scope": "Equal oracle acquisition budgets only; total compute is not matched",
                      "support_label_budget": "Separate from acquisition: every transfer fit receives exactly two or four labels",
                      "statistics_overhead": "Both active arms request the same full QueryStats routine: predictive entropy plus canonical and concrete conditional entropy diagnostics, including probability validation; no timing-efficiency claim",
                      "logical_count_note": "Prediction-entry counters count requested finite-table entries, not measured CPU instructions"}}


def execute(core, out):
    evaluator = TransferEvaluator(core)
    if int(evaluator.calls.sum()) * N_TABLES * N_INPUTS != 3840:
        raise ValueError("Grammar call counts differ from the declared common precomputation")
    began = time.perf_counter()
    records, worlds = [], []
    with (out / "records.jsonl").open("x", encoding="utf-8") as results, \
         (out / "worlds.jsonl").open("x", encoding="utf-8") as world_log, \
         (out / "failures.jsonl").open("x", encoding="utf-8") as failures:
        for regime in (0, 1):
            for seed in SEEDS:
                training_keys = range(N_TABLES) if regime == 0 else (0,)
                for table_key in training_keys:
                    world = make_world(core, seed, regime, table_key)
                    arms = {arm: acquire(core, world, arm, failures) for arm in ARMS}
                    world_public = {key: value for key, value in world.items() if key != "oracle_labels"}
                    world_public["training_truth_vectors"] = world["oracle_labels"].tolist()
                    world_public["arms"] = arms
                    append_record(world_log, world_public)
                    worlds.append(world_public)
                    target_tables = (table_key,) if regime == 0 else range(N_TABLES)
                    for true_table in target_tables:
                        truths, universe_size, excluded = evaluator.universe(true_table, world["oracle_labels"])
                        for arm in ARMS:
                            for budget in BUDGETS:
                                checkpoint = arms[arm][budget]
                                selected = checkpoint["selected_table"]
                                record = {"regime": world["regime"], "seed": seed,
                                    "training_world_key": table_key, "true_table": true_table,
                                    "arm": arm, "extra_queries": budget,
                                    "acquisition_queries": checkpoint["oracle_queries"],
                                    "selected_table": selected, "abstained": checkpoint["abstained"],
                                    "table_entropy_bits": checkpoint["table_entropy_bits"],
                                    "canonical_entropy_bits": checkpoint["canonical_entropy_bits"],
                                    "first_failure_extra_queries": checkpoint["first_failure_extra_queries"],
                                    "target_concrete_table_posterior": checkpoint["table_probabilities"][true_table],
                                    "target_canonical_table_posterior": checkpoint["canonical_probabilities"].get(
                                        int(core.canonical_table(true_table)), 0.0),
                                    "all_unique_target_truth_functions": universe_size,
                                    "excluded_training_truth_functions": excluded,
                                    "transfer_truth_functions": len(truths),
                                    "transfer": {str(size): evaluator.aggregate(selected, true_table, truths, size)
                                                 for size in SUPPORT_SIZES}}
                                append_record(results, record)
                                records.append(record)
                    print(encoded({"completed_training_worlds": len(worlds), "regime": world["regime"],
                                   "seed": seed, "table_key": table_key}), flush=True)
    return summarize(records, worlds, evaluator, time.perf_counter() - began)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite an existing output path: {out}")
    source_paths = {"core.py": ROOT / "core.py", "run_experiment.py": Path(__file__).resolve(),
                    "PROTOCOL.md": ROOT / "PROTOCOL.md"}
    for name, path in source_paths.items():
        if not path.is_file():
            parser.error(f"Required declared source is missing: {name}")
    sources = {name: sha256(path) for name, path in source_paths.items()}
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "started_at": utc_now(),
                "source_sha256": sources, "python": sys.version, "numpy": np.__version__,
                "platform": {"system": platform.system(), "machine": platform.machine()},
                "seeds": list(SEEDS), "budgets": list(BUDGETS), "arms": list(ARMS),
                "support_sizes": list(SUPPORT_SIZES), "manifest_written_before_core_import": True,
                "fixture_encoding": {"table_output": "(table >> (2*a+b)) & 1",
                    "input_order": "index = 4*x0 + 2*x1 + x2, ascending 0..7",
                    "program_order": "Six direct ordered-distinct pairs lexicographic, six left-nested permutations lexicographic, six right-nested permutations lexicographic",
                    "canonical_table": "min(table, argument-transposed table)",
                    "query_ties": "Within absolute 1e-12 bits of the global maximum, then lexicographic task/input"},
                "rng": "default_rng(SeedSequence([seed, regime_code, table_key, stream_code])); streams programs=0,initial=1,random=2,no-sharing tables=3",
                "test_data_access": False, "synthetic_finite_experiment": True}
    write_json(out / "manifest.json", manifest)
    write_json(out / "sourcehashes.json", sources)
    began = time.perf_counter()
    try:
        core = load_core(source_paths["core.py"])
        summary = execute(core, out)
        current = {name: sha256(path) for name, path in source_paths.items()}
        if current != sources:
            raise RuntimeError("A declared source changed during the experiment")
        summary["source_sha256"] = sources
        summary["costs"]["total_elapsed_seconds_including_core_import"] = time.perf_counter() - began
        write_json(out / "summary.json", summary)
        manifest.update(status="complete", finished_at=utc_now(),
                        elapsed_seconds=time.perf_counter() - began,
                        output_sha256={p.name: sha256(p) for p in sorted(out.iterdir())
                                       if p.is_file() and p.name != "manifest.json"})
        write_json(out / "manifest.json", manifest, replace=True)
    except BaseException as error:
        failure = {"kind": "runner_exception", "exception_type": type(error).__name__,
                   "message": str(error), "traceback": traceback.format_exc(), "time": utc_now()}
        with (out / "failures.jsonl").open("a", encoding="utf-8") as handle:
            append_record(handle, failure)
        manifest.update(status="failed", finished_at=utc_now(), error=failure,
                        elapsed_seconds=time.perf_counter() - began)
        write_json(out / "manifest.json", manifest, replace=True)
        raise


if __name__ == "__main__":
    main()
