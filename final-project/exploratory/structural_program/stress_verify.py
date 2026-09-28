"""Verify curated stress artifacts and all local saved choices without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

from .stress_proof import SEEDS, SIZES, TASKS, aggregate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle: Path, run: Path):
    provenance = json.loads((bundle / "provenance.json").read_text())
    for name, expected in provenance["artifact_sha256"].items():
        if digest(bundle / name) != expected:
            raise ValueError(f"curated artifact changed: {name}")
    source_folder = Path(__file__).parent
    for name, expected in provenance["source_sha256"].items():
        if digest(source_folder / name) != expected:
            raise ValueError(f"source changed: {name}")
    if digest(run / "manifest.json") != provenance["run_manifest_sha256"]:
        raise ValueError("original run manifest changed")
    protocol = json.loads((bundle / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        if digest(source_folder / name) != expected:
            raise ValueError(f"run source changed: {name}")
    rows = json.loads((bundle / "all-results.json").read_text())
    expected_cases = {(task, size, seed) for task in TASKS for size in SIZES for seed in SEEDS}
    if len(rows) != len(expected_cases) or {(r["task"], r["size"], r["seed"]) for r in rows} != expected_cases:
        raise ValueError("missing or duplicate predeclared case")
    for row in rows:
        case = run / f'{row["task"]}-{row["size"]}-{row["seed"]}'
        if digest(case / "selection.json") != row["selection_sha256"]:
            raise ValueError(f"saved selection changed: {case.name}")
        if json.loads((case / "results.json").read_text()) != row:
            raise ValueError(f"case metrics differ from curated row: {case.name}")
        if row["models"]["beam"]["candidate_count"] != row["models"]["random_grammar"]["candidate_count"]:
            raise ValueError("random budget did not match")
    if aggregate(rows) != json.loads((bundle / "aggregates.json").read_text()):
        raise ValueError("aggregate values do not reproduce")
    replay = json.loads((bundle / "prefix-replay.json").read_text())
    if not replay["all_prior_cases_replayed"] or len(replay["cases"]) != provenance["prior_cases_replayed"]:
        raise ValueError("incomplete prefix replay proof")
    fallbacks = json.loads((bundle / "numerical-fallbacks.json").read_text())
    if any(not event["objective_unchanged"] or event["projected_gradient_residual"] >= 1e-5
           for case in fallbacks for event in case["events"]):
        raise ValueError("invalid numerical fallback certificate")
    print(f"Verified {len(rows)} cases, exact candidate budgets, saved-choice hashes, aggregates, sources, and {len(replay['cases'])} prefix proofs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    verify(args.bundle, args.run)
