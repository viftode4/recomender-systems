"""Compare the accelerator with copied live checkpoints using TRAIN only.

This never writes to the study, reads validation outcomes or updates a live
worker. Timings include competition with whatever else is running locally.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import statistics

from exploratory.conditional_evidence import run_experiment as run

import numpy as np
import torch

from exploratory.conditional_evidence import data, model
from operations import conditional_evidence_acceleration as acceleration


def equal_state(left, right, where="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right), where
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right), where
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), where
        for key in left:
            equal_state(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), where
        for index, (a, b) in enumerate(zip(left, right)):
            equal_state(a, b, f"{where}[{index}]")
    else:
        assert left == right, where


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=run.ROOT / "runs/conditional-evidence-v1")
    parser.add_argument("--out", type=Path, default=run.ROOT / "runs/conditional-evidence-speedup-resume-verification.json")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing verification receipt")
    if args.episodes < 8:
        parser.error("At least one complete optimizer batch is required")
    if not 1 <= args.repeats <= 5:
        parser.error("Use 1..5 bounded repeat pairs")
    run.numerical_setup(2026)
    plan = json.loads((args.study / "plan.json").read_text())
    run.check_hashes(plan["source_sha256"])
    inputs = data.load_inputs(2026, load_meta=False)
    run.check_hashes(run.input_hashes(inputs))
    baseline = run.two_pass_backward
    acceleration_hash = run.digest(Path(acceleration.__file__))
    verifier_hash = run.digest(Path(__file__))
    acceleration_receipt = acceleration.install()
    accelerated = run.two_pass_backward
    results = {}
    try:
        for arm in ("raw", "scrambled", "summary"):
            checkpoint = args.study / "2026" / f"{arm}-lr0" / "latest.pt"
            # Opening an atomically replaced checkpoint yields one complete version.
            blob = checkpoint.read_bytes()
            saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            assert saved["guard"]["source_sha256"] == plan["source_sha256"]
            assert saved["guard"]["runtime"] == run.runtime()
            assert saved["guard"]["config"] == run.config_spec(2026, arm, 0.001)
            for name, expected in run.input_hashes(inputs).items():
                assert saved["guard"]["input_sha256"][name] == expected
            times = {"baseline": [], "accelerated": []}
            for repeat in range(args.repeats):
                variants = {}
                order = [("baseline", baseline), ("accelerated", accelerated)]
                if repeat % 2:
                    order.reverse()
                for name, function in order:
                    reader = run.make_reader(model, arm, 2026)
                    optimizer = torch.optim.Adam(reader.parameters(), lr=0.001, weight_decay=1e-4)
                    reader.load_state_dict(saved["model"])
                    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                    run.restore_rng(saved["rng"])
                    run.two_pass_backward = function
                    result = run.train_epoch(reader, model.EvidenceBank(inputs["categories"]),
                        inputs, data, optimizer, saved["epoch"] + 1, 2026, arm,
                        episode_limit=args.episodes)
                    variants[name] = {
                        "row": result, "model": copy.deepcopy(reader.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "gradients": [p.grad.clone() if p.grad is not None else None
                                      for p in reader.parameters()],
                        "rng": run.rng_state(),
                    }
                    times[name].append(result["seconds"])
                before, after = variants["baseline"], variants["accelerated"]
                for key in ("model", "optimizer", "gradients", "rng"):
                    equal_state(before[key], after[key], f"{arm}.{key}")
                for key in ("epoch", "queries", "mean_query_loss"):
                    equal_state(before["row"][key], after["row"][key], f"{arm}.{key}")
            a, b = (statistics.median(times[name]) for name in ("baseline", "accelerated"))
            results[arm] = {
                "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
                "starting_epoch": saved["epoch"], "episodes": args.episodes,
                "baseline_seconds": a, "accelerated_seconds": b, "speedup": a / b,
                "repeat_seconds": times, "repeats": args.repeats,
                "exact_loss_model_gradients_optimizer_rng": True,
            }
            print(json.dumps({"arm": arm, **results[arm]}), flush=True)
    finally:
        run.two_pass_backward = baseline
    run.check_hashes(plan["source_sha256"])
    run.check_hashes(run.input_hashes(inputs))
    assert acceleration_hash == run.digest(Path(acceleration.__file__))
    assert verifier_hash == run.digest(Path(__file__))
    assert acceleration_receipt == acceleration.receipt()
    run.atomic_json(args.out, {
        "status": "passed", "study_modified": False, "train_only": True,
        "meta_labels_read": False, "development_read": False, "original_test_read": False,
        "accelerator_sha256": acceleration_hash,
        "verifier_sha256": verifier_hash, "results": results,
        "timing_scope": "Median bounded continuations per arm with alternating order; competing live CPU work; not whole-study speedup.",
    })


if __name__ == "__main__":
    main()
