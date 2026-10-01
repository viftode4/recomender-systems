"""Bounded TRAIN-only builder profiling; never edits or patches the live runner."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from exploratory.conditional_evidence import data, model, run_experiment
from exploratory.conditional_evidence.native_scramble import runtime_metadata


class SkipUnusedSummaryBank(model.EvidenceBank):
    @staticmethod
    def _summary(panel, candidate_ratings, weights, h):
        # GraphReader never consumes features9. SummaryReader is never used here.
        return np.zeros(9, dtype=np.float64)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def create(bank, episode, arm, seed):
    return bank.build_query(episode["context"], episode["user_index"], episode["candidate_ids"],
                            variant=arm, seed=seed, materialize_graph=arm != "summary")


def run_builders(bank, episodes, arm, seed, chunk):
    start = time.perf_counter()
    calls = graphs = edges = nodes = 0
    for episode in episodes:
        graph_seed = run_experiment.episode_seed(seed, episode, epoch=1)
        candidates = episode["candidate_ids"]
        for offset in range(0, len(candidates), chunk):
            batch = create(bank, {**episode, "candidate_ids": candidates[offset:offset + chunk]}, arm, graph_seed)
            calls += 1
            graphs += batch.num_graphs
            edges += batch.rating.numel()
            nodes += batch.x.shape[0]
    elapsed = time.perf_counter() - start
    return {"seconds": elapsed, "graphs": graphs, "graphs_per_second": graphs / elapsed,
            "chunks": calls, "nodes": nodes, "directed_edges": edges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.repetitions <= 3:
        raise ValueError("Use a new output and1..3 bounded repetitions")
    run_experiment.configure_runtime(2026) if hasattr(run_experiment, "configure_runtime") else torch.set_num_threads(1)
    seed = 2026
    paths = [Path(data.__file__), Path(model.__file__), Path(run_experiment.__file__), Path(__file__),
             ROOT / "exploratory/conditional_evidence/native_scramble.py",
             ROOT / "exploratory/conditional_evidence/native_scramble.c"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    inputs = data.load_inputs(seed, load_meta=False)
    counts = np.count_nonzero(inputs["categories"], axis=1)
    order = np.lexsort((np.arange(len(counts)), counts))
    chosen = set(int(order[int(q * (len(order) - 1))]) for q in (.1, .4, .7, .9))
    episodes = [episode for episode in data.episodes(inputs, 1, (.8, .9), seed) if episode["user_index"] in chosen]
    bank = model.EvidenceBank(inputs["categories"])
    optimized = SkipUnusedSummaryBank(inputs["categories"])
    native = runtime_metadata()
    # Warm compiler and dispatch before timing; source/library remain guarded.
    create(bank, {**episodes[0], "candidate_ids": episodes[0]["candidate_ids"][:2]}, "scrambled", 1)
    before_rss = rss()
    timing = {}
    for arm in ("raw", "scrambled", "summary"):
        timing[arm] = {"baseline": [run_builders(bank, episodes, arm, seed, 64) for _ in range(args.repetitions)]}
        if arm != "summary":
            timing[arm]["skip_unused_summary"] = [run_builders(optimized, episodes, arm, seed, 64) for _ in range(args.repetitions)]
    profiles = {}
    for arm in ("raw", "scrambled", "summary"):
        profiler = cProfile.Profile()
        profiler.enable()
        run_builders(bank, episodes, arm, seed, 64)
        profiler.disable()
        rows = []
        for (filename, line, function), (primitive, total, self_time, cumulative, callers) in pstats.Stats(profiler).stats.items():
            rows.append({"function": function, "file": Path(filename).name, "line": line,
                         "calls": total, "self_seconds": self_time, "cumulative_seconds": cumulative})
        profiles[arm] = sorted(rows, key=lambda x: x["cumulative_seconds"], reverse=True)[:25]
    reader = model.GraphReader(seed=seed)
    checked = 0
    max_output_difference = 0.
    max_chunk_bytes = 0
    for arm in ("raw", "scrambled"):
        for episode in episodes:
            graph_seed = run_experiment.episode_seed(seed, episode, epoch=1)
            # One real chunk per episode/arm is enough for this local prototype.
            part = {**episode, "candidate_ids": episode["candidate_ids"][:64]}
            a, b = create(bank, part, arm, graph_seed), create(optimized, part, arm, graph_seed)
            for key in ("x", "edge_index", "rating", "query_idx", "candidate_idx", "donor_idx", "donor_graph_idx", "candidate_ids"):
                if not torch.equal(getattr(a, key), getattr(b, key)):
                    raise AssertionError(f"Graph input changed:{key}")
            if a.donor_rows != b.donor_rows or a.metadata != b.metadata:
                raise AssertionError("Graph donor or evidence metadata changed")
            with torch.no_grad():
                difference = float((reader(a) - reader(b)).abs().max())
            max_output_difference = max(max_output_difference, difference)
            max_chunk_bytes = max(max_chunk_bytes, sum(v.numel() * v.element_size() for v in vars(a).values() if isinstance(v, torch.Tensor)))
            checked += 1
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in paths} or native != runtime_metadata():
        raise ValueError("Frozen source/runtime changed during benchmark")
    result = {"status": "complete", "validation_read": False, "test_read": False,
        "live_runner_mutated": False, "source_sha256": hashes, "native_runtime": native,
        "episodes": len(episodes), "users": len(chosen), "activity_quantiles": [.1, .4, .7, .9],
        "history_counts_sorted": sorted(int(counts[index]) for index in chosen),
        "candidate_chunk": 64, "timing": timing, "profile": profiles,
        "parity": {"real_graph_chunks": checked, "graph_tensors_donors_metadata_exact": True,
                   "maximum_reader_output_difference": max_output_difference,
                   "unused_features9_changed": True},
        "memory": {"peak_rss_before_timing_bytes": before_rss, "peak_rss_after_bytes": rss(),
                   "largest_checked_chunk_tensor_bytes": max_chunk_bytes},
        "limitations": ["Concurrent training can affect wall-clock timing; these are bounded samples, not guaranteed study runtimes.",
                        "cProfile runs are separate from uninstrumented throughput trials.",
                        "Skip-summary changes an unused diagnostic tensor; GraphReader inputs/output remain exact in this pilot.",
                        "No source edit or monkeypatch was applied to any live process."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for arm, values in timing.items():
        print(arm, {name: round(float(np.median([row["seconds"] for row in rows])), 6) for name, rows in values.items()})
    print(json.dumps({"status": result["status"], "parity": result["parity"], "memory": result["memory"]}))


if __name__ == "__main__":
    main()
