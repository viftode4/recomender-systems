"""Build a local, deterministic group ZIP from an explicit source whitelist.

No upload or submission is performed. Dataset files, raw runs, checkpoints,
credentials and the copyrighted assignment attachment are excluded. Draft reports
require --allow-draft and remain labelled DRAFT inside the ZIP.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


CODE_FILES = (
    "README.md", "RESEARCH.md", "PLAN.md", "HANDOFF.md", "AUDIT.md", "config.json",
    "ADAPTIVE_RESEARCH.md", "REJECTION_RESEARCH.md", "NEGATIVE_INFORMATION_PROTOCOL.md", "JOINT_FIELD_PROTOCOL.md",
    "requirements.txt", "requirements.lock.txt", "pyproject.toml", "uv.lock",
    "requirements-report.txt",
    "run.py", "metrics.py", "study.py", "experiment.py", "summarize.py",
    "exception_model.py", "exception_experiment.py", "societal.py", "freeze.py",
    "final_evaluate.py", "hybrid_constraints.py", "verification.py",
    "audit_study.py", "compare_frozen.py", "freeze_exception.py", "evaluate_exception.py",
    "assemble_study.py", "temporal_experiment.py", "exception_evaluation.py", "verify_environment.py", "audit_slim.py",
    "categorical_field.py", "categorical_experiment.py", "categorical_demo.py", "joint_field_experiment.py", "negative_information.py",
    "negative_information_experiment.py", "negative_information_report.py", "negative_controls.py",
    "categorical_evaluation.py", "joint_evaluation.py", "field_reference_comparison.py",
    "joint_field_convergence.py", "joint_convergence_evaluation.py", "JOINT_FIELD_CONVERGENCE_PROTOCOL.md",
    "field_reference_evaluation.py", "compare_fields_final.py", "field_learning_curves.py", "run_final_batch.py",
    "curate_final_evidence.py",
    "report.py", "package_project.py",
)
REQUIRED_CODE = {"README.md", "config.json", "run.py", "metrics.py", "study.py", "report.py", "package_project.py"}
AGGREGATE_FILES = {
    "RESULTS.md", "aggregates.json", "aggregate.json", "selected.json", "candidate-budgets.json",
    "candidate-feasibility.json", "group-policies.json", "research-figures.png", "research-figures.pdf",
    "provenance.json",
    "summary.json", "individual-models.json", "cohort-counts.json",
    "protocol.json", "selection.json", "paired-comparisons.json", "SHA256.json", "manifest.json",
    "development-results.png", "development-results.pdf", "reranker-tradeoffs.png", "reranker-tradeoffs.pdf",
    "learning-curves.png", "learning-curves.pdf", "SUMMARY.md", "per-seed-comparisons.json",
    "training-taste-groups.json", "societal-audit.json",
}
REPORT_FILES = ("report.pdf", "report.md", "report-content.json", "REPORT-MANIFEST.json")
DEMO_FILES = ("index.html", "predictions.json", "manifest.json", "validation.json")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_read(root, path):
    """Read a regular file under its explicit root; reject symlinks at every level."""
    root, path = Path(root).absolute(), Path(path).absolute()
    relative = path.relative_to(root)
    current = root
    if root.is_symlink():
        raise ValueError(f"Symlinks are excluded: {root.name}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are excluded: {relative}")
    if not path.is_file():
        raise ValueError(f"Required regular file is missing: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("File resolves outside its declared source root")
    return path.read_bytes()


def check_aggregate_only(value, location="evidence"):
    """Fail on known individual-record fields even inside an allowed JSON file."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"per_user", "matched_pairs_by_user", "pairs_per_user", "recommendations",
                       "histories", "user_ids", "item_ids"}:
                raise ValueError(f"Individual records are excluded from packages: {location}/{key}")
            if key in {"calibration_users", "meta_fit", "development", "users"} and isinstance(child, (list, dict)):
                raise ValueError(f"User identifier lists are excluded from packages: {location}/{key}")
            check_aggregate_only(child, location + "/" + key)
    elif isinstance(value, list):
        for child in value:
            check_aggregate_only(child, location)


def demo_payload(directory, source_payload):
    """Only the named, source-bound synthetic demonstration may accompany a ZIP."""
    directory = Path(directory)
    data = {name: safe_read(directory, directory / name) for name in DEMO_FILES}
    manifest = json.loads(data["manifest.json"])
    predictions = json.loads(data["predictions.json"])
    validation = json.loads(data["validation.json"])
    if (manifest.get("profile_kind") != "hand_authored_synthetic" or
            predictions.get("profile_kind") != "hand_authored_synthetic" or
            manifest.get("real_user_history_read") is not False or manifest.get("test_read") is not False):
        raise ValueError("Demo requires a declared synthetic profile with no real history or test access")
    check_aggregate_only(predictions, "synthetic-demo")
    if manifest.get("provenance") != predictions.get("provenance"):
        raise ValueError("Demo prediction and manifest provenance differ")
    expected = manifest.get("files_sha256", {})
    if set(expected) != {"index.html", "predictions.json"}:
        raise ValueError("Demo manifest may authorize only index.html and predictions.json")
    for name, expected_hash in expected.items():
        if sha256(data[name]) != expected_hash:
            raise ValueError(f"Demo file hash mismatch: {name}")
    for filename, expected_hash in [
            ("categorical_demo.py", manifest.get("generator_sha256")),
            ("categorical_field.py", manifest.get("provenance", {}).get("model_source_sha256"))]:
        if (f"code/{filename}" not in source_payload or
                sha256(source_payload[f"code/{filename}"]) != expected_hash):
            raise ValueError(f"Demo source hash mismatch: {filename}")
    if validation.get("status") != "PASS":
        raise ValueError("Demo requires its recorded functional validation artifact")
    return {f"demo/{name}": value for name, value in data.items()}


def package_project(root, report_dir, out, group, evidence_dirs=(), allow_draft=False, demo_dir=None):
    root, report_dir, out = Path(root), Path(report_dir), Path(out)
    group = str(group)
    if not re.fullmatch(r"[1-9][0-9]*", group):
        raise ValueError("An explicit positive integer group number is required")
    if out.name != f"{group}.zip":
        raise ValueError(f"The assignment requires the archive filename {group}.zip")
    if out.exists():
        raise FileExistsError(out)
    report_manifest = json.loads(safe_read(report_dir, report_dir / "REPORT-MANIFEST.json"))
    if report_manifest.get("group") is not None and str(report_manifest["group"]) != group:
        raise ValueError("Group number differs between the report and requested archive")
    status = report_manifest.get("status")
    if status not in {"DRAFT", "FINAL"}:
        raise ValueError("Report status must be DRAFT or FINAL")
    if status == "DRAFT" and not allow_draft:
        raise ValueError("The report is DRAFT; --allow-draft is required for a development package")
    if status == "FINAL" and (str(report_manifest.get("group")) != group or
                              not report_manifest.get("contributors_supplied")):
        raise ValueError("Final report must contain the actual group and contributors")
    payload = {}
    for filename in REPORT_FILES:
        data = safe_read(report_dir, report_dir / filename)
        if filename != "REPORT-MANIFEST.json" and report_manifest.get("artifact_sha256", {}).get(filename) != sha256(data):
            raise ValueError(f"Report artifact hash mismatch: {filename}")
        payload[filename] = data
    for filename in CODE_FILES:
        path = root / filename
        if path.exists() or path.is_symlink():
            payload[f"code/{filename}"] = safe_read(root, path)
        elif filename in REQUIRED_CODE:
            raise ValueError(f"Required runnable code is missing: {filename}")
    # Only source tests, never caches, fixtures containing histories, or arbitrary subtrees.
    for path in sorted((root / "tests").glob("test_*.py")):
        payload[f"code/tests/{path.name}"] = safe_read(root, path)
    if (root / "evidence/EVALUATION_PROTOCOL.md").exists():
        payload["evidence/EVALUATION_PROTOCOL.md"] = safe_read(root, root / "evidence/EVALUATION_PROTOCOL.md")
    source_hashes = {name: {sha256(data)} for name, data in payload.items() if name.startswith("code/")}
    for directory in evidence_dirs:
        directory = Path(directory)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", directory.name):
            raise ValueError("Evidence directory requires a simple nonempty name")
        for path in sorted(directory.iterdir()):
            allowed = path.name in AGGREGATE_FILES or bool(re.fullmatch(
                r"[0-9]+-(coefficients|selection|paired-comparisons|switching)\.json", path.name))
            if not allowed:
                continue
            name = f"evidence/{directory.name}/{path.name}"
            if name in payload:
                raise ValueError(f"Duplicate evidence path: {name}")
            data = safe_read(directory, path)
            if path.suffix == ".json":
                check_aggregate_only(json.loads(data), path.name)
            payload[name] = data
            source_hashes.setdefault(path.name, set()).add(sha256(data))
            source_hashes.setdefault(f"{directory.name}/{path.name}", set()).add(sha256(data))
    expected_sources = report_manifest.get("source_sha256", {})
    missing_sources = [name for name, expected in expected_sources.items() if expected not in source_hashes.get(name, set())]
    if missing_sources:
        raise ValueError("Supply the matching aggregate evidence used by the report: " + ", ".join(missing_sources))
    if demo_dir is not None:
        payload.update(demo_payload(demo_dir, payload))
    note = (
        f"Group {group} project package\nStatus: {status}\n\n"
        "Open report.pdf for the task-structured report and code/README.md for reproduction.\n"
        "PACKAGE-MANIFEST.json lists SHA-256 hashes for every other archive entry.\n"
        "Dataset, raw runs, checkpoints and the assignment PDF are intentionally excluded.\n"
        "Obtain MovieLens 100K and the instructor's RecBole fork as documented.\n"
        "This tool only prepared a local ZIP. Nothing was submitted to Brightspace.\n"
        "Each student must separately complete the required peer-feedback workbook.\n"
    )
    if status == "DRAFT":
        note += "\nDRAFT: contributor metadata or final review is incomplete; the report labels each evidence stage.\n"
    if demo_dir is not None:
        note += ("\nOptional offline demonstration: open demo/index.html. Its history is hand-authored and synthetic.\n"
                 "demo/validation.json records the checks actually performed and their scope.\n"
                 "The demonstration shows model sensitivities, not measured recommendation quality.\n")
    payload["PACKAGE-README.txt"] = note.encode()
    manifest = {
        "schema_version": 1, "group": group, "status": status,
        "demo_included": demo_dir is not None,
        "artifact_sha256": {name: sha256(data) for name, data in sorted(payload.items())},
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "exclusions": ["datasets", "raw runs", "user histories", "checkpoints", "credentials", "assignment PDF"],
    }
    payload["PACKAGE-MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data, compresslevel=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(buffer.getvalue())
    return {"archive": str(out), "status": status, "files": len(payload),
            "bytes": out.stat().st_size, "sha256": sha256(buffer.getvalue())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--demo", type=Path, help="optional source-bound synthetic demo; only four fixed files are copied")
    args = parser.parse_args()
    print(json.dumps(package_project(args.root, args.report_dir, args.out, args.group,
                                     args.evidence, args.allow_draft, args.demo)))


if __name__ == "__main__":
    main()
