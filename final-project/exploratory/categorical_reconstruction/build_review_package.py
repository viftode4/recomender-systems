"""Build a new review ZIP while preserving the verified original study archive.

No upload occurs. Only original audited code/aggregate evidence and explicitly
named new research artifacts may enter this archive; raw runs are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exploratory.categorical_reconstruction.render_supplement import (
    EVIDENCE_FILES, aggregate_only, digest, load_evidence, safe_bytes)


BASE_SHA256 = "7e1d86708471b0d70a4be375ff7211f69a12505802d626e951f2a0fdbc4fd0bb"
ORIGINAL_PDF_SHA256 = "08888a2ce58b6fe9ae8200c785e185bdaa0acba247c26f8e719123a8759d2fe0"
STUDY_PATH = "exploratory/categorical_reconstruction"
LOCAL_FILES = ("model.py", "run_experiment.py", "hybrid.py", "group_analysis.py",
               "render_supplement.py", "build_review_package.py", "verify_results.py", "verify_groups.py", "PROTOCOL.md", "DERIVATION.md",
               "README.md", "RELATED_WORK.md", "ASSIGNMENT_FIT.md", "REPORT_PLAN.md", "FINDINGS.md")
OPTIONAL_FILES = ("TRAIN_TIMING.json",)
REPORT_FILES = ("research-supplement.pdf", "research-supplement.md", "report-content.json",
                "evidence-audit.json", "SUPPLEMENT-MANIFEST.json")
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".tsv", ".csv", ".pkl", ".pickle", ".pt", ".pth", ".ckpt"}


def bytes_hash(data):
    return hashlib.sha256(data).hexdigest()


def valid_archive_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts
            or any(part in {"", ".", "__pycache__", ".git", "runs"} for part in path.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES):
        raise ValueError(f"Excluded archive path: {name}")


def original_payload(archive_path):
    raw = safe_bytes(Path(archive_path).parent, Path(archive_path).name)
    if bytes_hash(raw) != BASE_SHA256:
        raise ValueError("Base archive is not the immutable verified original 24.zip")
    import io
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.testzip() is not None:
            raise ValueError("Base archive CRC check failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate entries in original archive")
        manifest = json.loads(archive.read("PACKAGE-MANIFEST.json"))
        expected = manifest["artifact_sha256"]
        if set(expected) != set(names) - {"PACKAGE-MANIFEST.json"}:
            raise ValueError("Original archive manifest does not cover its entries exactly")
        payload = {}
        for info in archive.infolist():
            valid_archive_name(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlink entry in base archive")
            data = archive.read(info.filename)
            if info.filename != "PACKAGE-MANIFEST.json" and bytes_hash(data) != expected[info.filename]:
                raise ValueError("Original archive entry hash mismatch")
            if (info.filename.startswith("code/") or info.filename.startswith("evidence/")
                    or info.filename in {"report.pdf", "report.md", "report-content.json", "REPORT-MANIFEST.json"}):
                if info.filename.endswith(".json") and not info.filename.startswith("code/"):
                    aggregate_only(json.loads(data), info.filename)
                payload[info.filename] = data
        if bytes_hash(payload["report.pdf"]) != ORIGINAL_PDF_SHA256:
            raise ValueError("Original report PDF differs")
        payload["original-package-manifest.json"] = archive.read("PACKAGE-MANIFEST.json")
    return payload


def readme(evidence, supplement_manifest):
    runtime = evidence["provenance"].get("runtime", {})
    return f"""# Group 24 research review bundle

Status: REVIEW DRAFT. This archive is not a submission receipt. Four member
names/contributions remain deferred unless supplied in the accompanying report;
no peer contributions or assessments are invented.

Read `report.pdf` for the original, frozen coursework study. Its exact bytes
are preserved from the independently verified original group archive.
Read `research-supplement.pdf` for later categorical-reconstruction research.
The supplement uses reused development data after the original dataset TEST
was viewed; it is not new held-out confirmation and does not amend the original
test tables. Detailed new tables are in `research/research-supplement.md` and
`evidence/categorical-reconstruction-v1/RESULTS.md`.

## Contents and verification

- `code/`: original runnable project source plus the separate new study under
  `exploratory/categorical_reconstruction/`.
- `evidence/`: curated aggregate evidence only; no user histories or scores.
- `research/`: exact supplement artifacts, audit receipt and report manifest.
- `research/group-audit.json`: independent replay of every reported user/item
  group metric, bound to the measured aggregate files and verifier source.
- `PACKAGE-MANIFEST.json`: SHA-256 of every other archive entry.
- `original-package-manifest.json`: historical seal of the original archive;
  it describes that original archive, including entries not carried here.

Both PDF files are review drafts. The new supplement has one page, Times New
Roman 12pt, spacing 1.15, and {supplement_manifest['discussion_words']} discussion words.
Original PDF SHA-256: {ORIGINAL_PDF_SHA256}
Original archive SHA-256: {BASE_SHA256}

## Run the independent new-model tests

Use a Python environment with NumPy and SciPy. The measured fitting environment
was Python {runtime.get('python', 'recorded in provenance.json')}, NumPy
{runtime.get('numpy', 'recorded in provenance.json')} and SciPy
{runtime.get('scipy', 'recorded in provenance.json')}. Rendering additionally
requires `code/requirements-report.txt` and locally installed Times New Roman
regular/bold fonts. Font files and third-party datasets are not redistributed.

From the extracted archive root:

```sh
cd code
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m unittest discover -s exploratory/categorical_reconstruction -p 'test_*.py' -v
```

## Rebuild the measured supplement

From `code/`, with the same source files and a rendering environment:

```sh
python exploratory/categorical_reconstruction/render_supplement.py --evidence ../evidence/categorical-reconstruction-v1 --audit ../research/evidence-audit.json --out ../rebuilt-supplement --group 24 --member 'Vlad George Iftode'
```

Supply only actual contributor names. Output directories must be new. The
renderer verifies the evidence and source hashes before drawing any results;
it never fits models or reads ratings. `research/report-content.json` binds the
displayed numbers to the curated evidence. Font hashes are in the report seal.

## Repeat fitting

The standalone model API and exact runner commands are documented in
`code/exploratory/categorical_reconstruction/README.md`; the protocol and
derivation are alongside it. Full study reproduction first requires the public
MovieLens data, the original TRAIN/VALID partitions, cohort definitions and
locked reference runs described by the project instructions in `code/README.md`.
Those raw inputs and reference prediction arrays are deliberately not packaged.
They must be regenerated through the declared source pipeline, with identities
and fingerprints verified; aggregate tables alone cannot reconstruct them.
Use fresh output paths and keep the exploration label. Rebuilding code does
not restore the original TEST to an unexamined state.

Existing source-document links were written relative to the complete repository.
In this archive, source is under `code/`, evidence is its sibling, and reports
are at the archive root. Use the entry points above for the extracted layout.
The package builder performs no network upload or course submission.
"""


def build(base_archive, evidence_dir, report_dir, out, root=ROOT, group_audit=None):
    root, report_dir, evidence_dir, out = map(Path, (root, report_dir, evidence_dir, out))
    if out.exists() or out.with_suffix(".receipt.json").exists():
        raise FileExistsError(out)
    if out.name != "24-review.zip":
        raise ValueError("Use the explicit research-review filename 24-review.zip")
    evidence = load_evidence(evidence_dir, report_dir / "evidence-audit.json", root)
    group_audit = Path(group_audit) if group_audit else root / STUDY_PATH / "group-audit-v1.json"
    group_audit_bytes = safe_bytes(group_audit.parent, group_audit.name)
    group_receipt = json.loads(group_audit_bytes)
    aggregate_only(group_receipt, "group-audit")
    expected_evidence = {name: evidence["artifact_sha256"][name]
                         for name in ("aggregates.json", "protocol.json", "provenance.json")}
    if (group_receipt.get("status") != "complete" or group_receipt.get("stage") != "reused_development"
            or group_receipt.get("fresh_confirmation") is not False or group_receipt.get("test_read") is not False
            or group_receipt.get("artifact_sha256") != expected_evidence
            or group_receipt.get("completed_manifest_sha256") != evidence["provenance"]["manifest_sha256"]
            or group_receipt.get("selection_freeze_sha256") != evidence["provenance"]["selection_freeze_sha256"]
            or group_receipt.get("source_sha256") != evidence["provenance"]["source_sha256"]
            or group_receipt.get("barrier_verifier_sha256") != digest(root / STUDY_PATH / "verify_results.py")
            or group_receipt.get("verifier_sha256") != digest(root / STUDY_PATH / "verify_groups.py")):
        raise ValueError("Group audit does not bind these exact sources and measured aggregates")
    report_data = {name: safe_bytes(report_dir, name) for name in REPORT_FILES}
    report_manifest = json.loads(report_data["SUPPLEMENT-MANIFEST.json"])
    if report_manifest.get("status") != "REVIEW DRAFT" or report_manifest.get("group") != "24":
        raise ValueError("The supplement must be the group 24 review draft")
    if (report_manifest.get("layout", {}).get("pages") != 1
            or report_manifest.get("discussion_words", 201) > 200
            or report_manifest.get("evidence_sha256") != evidence["artifact_sha256"]
            or report_manifest.get("audit_sha256") != evidence["audit_sha256"]):
        raise ValueError("Report limits or evidence bindings differ")
    expected_reports = set(REPORT_FILES) - {"SUPPLEMENT-MANIFEST.json"}
    if set(report_manifest.get("artifact_sha256", {})) != expected_reports:
        raise ValueError("Report manifest does not cover exactly the expected artifacts")
    if any(bytes_hash(report_data[name]) != expected
           for name, expected in report_manifest["artifact_sha256"].items()):
        raise ValueError("Supplement artifact hash mismatch")
    if report_manifest.get("generator_sha256") != digest(root / STUDY_PATH / "render_supplement.py"):
        raise ValueError("Renderer source differs from the generated report")
    payload = original_payload(base_archive)
    local = root / STUDY_PATH
    for name in LOCAL_FILES:
        payload[f"code/{STUDY_PATH}/{name}"] = safe_bytes(local, name)
    for name in OPTIONAL_FILES:
        if (local / name).exists():
            data = safe_bytes(local, name)
            if name.endswith(".json"):
                aggregate_only(json.loads(data), name)
            payload[f"code/{STUDY_PATH}/{name}"] = data
    for path in sorted(local.glob("test_*.py")):
        payload[f"code/{STUDY_PATH}/{path.name}"] = safe_bytes(local, path.name)
    for name, expected in evidence["provenance"]["source_sha256"].items():
        key = "code/" + name
        if key not in payload or bytes_hash(payload[key]) != expected:
            raise ValueError(f"Packaged source differs from new-study provenance: {name}")
    for name in EVIDENCE_FILES:
        payload["evidence/categorical-reconstruction-v1/" + name] = safe_bytes(evidence_dir, name)
    for name, data in report_data.items():
        if name.endswith(".json"):
            aggregate_only(json.loads(data), "research/" + name)
        payload["research/" + name] = data
    payload["research-supplement.pdf"] = report_data["research-supplement.pdf"]
    payload["research/group-audit.json"] = group_audit_bytes
    payload["README.md"] = readme(evidence, report_manifest).encode("utf-8")
    for name in payload:
        valid_archive_name(name)
    package_manifest = {"schema_version": 1, "status": "REVIEW DRAFT", "group": "24",
        "original_archive_sha256": BASE_SHA256, "original_report_sha256": ORIGINAL_PDF_SHA256,
        "generator_sha256": digest(__file__), "supplement_manifest_sha256": bytes_hash(report_data["SUPPLEMENT-MANIFEST.json"]),
        "scope": "Original frozen report plus a separate post-test exploratory development supplement",
        "exclusions": ["raw ratings", "user histories", "individual predictions", "selected scores", "checkpoints",
                       "credentials", "assignment PDF", "font binaries"],
        "artifact_sha256": {name: bytes_hash(data) for name, data in sorted(payload.items())}}
    payload["PACKAGE-MANIFEST.json"] = (json.dumps(package_manifest, indent=2, sort_keys=True) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload):
            raise ValueError("Generated ZIP verification failed")
        if any(bytes_hash(archive.read(name)) != expected
               for name, expected in package_manifest["artifact_sha256"].items()):
            raise ValueError("Generated ZIP entry hash mismatch")
    receipt = {"status": "REVIEW DRAFT", "archive": out.name, "bytes": out.stat().st_size,
               "sha256": digest(out), "entries": len(payload),
               "original_report_unchanged": True, "raw_data_included": False,
               "supplement_manifest_sha256": package_manifest["supplement_manifest_sha256"]}
    receipt_path = out.with_suffix(".receipt.json")
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=ROOT / "packages/final-review-v1/24.zip")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.base_archive, args.evidence, args.report_dir, args.out,
                           group_audit=args.group_audit), indent=2))


if __name__ == "__main__":
    main()
