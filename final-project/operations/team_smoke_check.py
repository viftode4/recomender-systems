"""Fast teammate check using synthetic tests and existing public evidence only.

Run with Python 3.11 and NumPy. This command does not train models, read ratings,
resume a workflow, install dependencies, or contact a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


TEST_DRIVER = r'''
import json
import unittest
import numpy

suite = unittest.TestSuite()
for pattern in ("test_metrics.py", "test_hybrid_constraints.py", "test_societal.py"):
    suite.addTests(unittest.TestLoader().discover("tests", pattern=pattern))
suite.addTests(unittest.TestLoader().loadTestsFromName("coursework_completion.test_models"))
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({"tests": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "skipped": len(result.skipped),
                  "numpy": numpy.__version__}))
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
'''


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or invalid artifact: {relative}")
    return path


def check_artifacts(project: Path) -> dict:
    current_report = next((project / "reports" / version for version in
                           ("coursework-complete-v2", "coursework-complete-v1")
                           if (project / "reports" / version / "report.pdf").is_file()), None)
    if current_report is not None:
        layout = "repository"
        report_root = current_report
        evidence_root = project / "coursework_completion/results-v1"
    elif project.name == "code" and (project.parent / "PACKAGE-MANIFEST.json").is_file():
        layout = "extracted_coursework_archive"
        report_root = project.parent
        evidence_root = project.parent / "evidence/coursework-completion-v1"
    else:
        raise ValueError("Use the project repository root or the extracted 24.zip code/ directory")

    hashes = json.loads((evidence_root / "SHA256.json").read_text())
    required = {"aggregates.json", "protocol.json", "provenance.json"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("Completion evidence hash manifest is incomplete")
    for relative, expected in hashes.items():
        if digest(checked_path(evidence_root, relative)) != expected:
            raise ValueError(f"Evidence hash mismatch: {relative}")

    aggregate = json.loads((evidence_root / "aggregates.json").read_text())
    if (aggregate.get("status") != "complete" or aggregate.get("fresh_holdout") is not False
            or aggregate.get("original_test_read") is not False
            or set(aggregate.get("seeds", {})) != {"2026", "2027", "2028"}):
        raise ValueError("Completion evidence does not match the documented study scope")

    report = checked_path(report_root, "report.pdf")
    manifest = json.loads((report_root / "REPORT-MANIFEST.json").read_text())
    report_hash = digest(report)
    if manifest.get("artifact_sha256", {}).get("report.pdf") != report_hash:
        raise ValueError("Report hash differs from the recorded report manifest")
    with report.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("Report is missing its PDF header")
    if manifest.get("pages") != 5:
        raise ValueError("Expected the recorded five-page coursework draft")
    return {
        "project_layout": layout,
        "completion_evidence_files_verified": len(hashes),
        "completion_seeds": sorted(aggregate["seeds"]),
        "completion_assessment": aggregate["assessment_stage"],
        "completion_fresh_holdout": False,
        "report_sha256": report_hash,
        "report_pages_recorded_in_manifest": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", "--root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="Project directory (final-project/ in the repository) or extracted coursework archive's code/ directory")
    args = parser.parse_args()
    start = time.perf_counter()
    receipt = {"status": "failed", "training_started": False,
               "raw_dataset_required": False, "python": sys.version.split()[0]}
    try:
        if sys.version_info < (3, 11):
            raise ValueError("Use Python 3.11 or newer; the recorded environment is Python 3.11")
        project = args.project_root.expanduser().resolve()
        receipt.update(check_artifacts(project))
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Keep imports rooted in the selected checkout, not a caller's PYTHONPATH.
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-c", TEST_DRIVER], cwd=project,
                                env=env, capture_output=True, text=True, timeout=50)
        sys.stderr.write(result.stderr)
        if result.stdout.strip():
            receipt["synthetic_tests"] = json.loads(result.stdout)
        if result.returncode:
            raise ValueError("Synthetic checks failed; inspect the test output above")
        if receipt.get("synthetic_tests", {}).get("tests", 0) < 1:
            raise ValueError("No synthetic tests were executed")
        receipt["status"] = "passed"
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        receipt["error"] = str(error)
    receipt["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
