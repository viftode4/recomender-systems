"""Build the aggregate-only framing review; never fit or read interaction data.

The scientific experiment and its review must finish before this builder runs.
The previous archive is immutable, and all outputs are created without overwrite.
Run with the report Python environment (Matplotlib, pypdf and Times New Roman).
Use --runtime-python for independent imports from the extracted numerical code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
STUDY = "exploratory/framing_search"
BASE_SHA256 = "61c89bc8411ee196c7679b860647a80bf199c31155258db89e1c5be3e3b48de3"
ORIGINAL_PDF_SHA256 = "08888a2ce58b6fe9ae8200c785e185bdaa0acba247c26f8e719123a8759d2fe0"
CATEGORICAL_PDF_SHA256 = "113bb2eddd22d2424680dadf880fb57e15a8d55b282d525fa0fa849169785830"
ROLES = ("EASE", "bag", "true_group", "shuffle_0", "shuffle_1", "shuffle_2")
LABELS = {"EASE": "Tuned EASE", "bag": "Unordered pairs", "true_group": "Recording groups",
          "shuffle_0": "Shuffle 1", "shuffle_1": "Shuffle 2", "shuffle_2": "Shuffle 3"}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".tsv", ".csv", ".pkl", ".pickle", ".pt", ".pth", ".ckpt"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or not name or "\\" in name or
            any(p in {"..", ".", "__pycache__", ".git", "runs"} for p in path.parts) or
            path.suffix.lower() in FORBIDDEN_SUFFIXES):
        raise ValueError(f"Unsafe or excluded archive name: {name}")
    return path


def read_safe(directory, name):
    relative = safe_name(name)
    directory = Path(directory).absolute()
    if directory.is_symlink():
        raise ValueError("Symlink directory is excluded")
    path = directory
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Symlink is excluded: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Missing regular file: {name}")
    return path.read_bytes()


def aggregate_only(value, location="root"):
    """Additional guard on explicitly allowlisted public files, not anonymization."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {"per_user", "recommendations", "histories", "user_ids", "item_ids",
                              "pairs_per_user", "matched_pairs_by_user", "cohort_ids", "train_pairs"}:
                raise ValueError(f"Individual data excluded: {location}/{key}")
            aggregate_fraction = (isinstance(child, dict) and set(child) == {"count", "denominator", "fraction"}
                                  and all(isinstance(v, (int, float)) for v in child.values()))
            if normalized in {"users", "items", "ratings", "scores", "predictions", "selected_scores",
                              "raw_scores", "fit_users", "assessment_users"} and isinstance(child, (list, dict)):
                if not aggregate_fraction:
                    raise ValueError(f"Individual collection excluded: {location}/{key}")
            aggregate_only(child, location + "/" + key)
    elif isinstance(value, list):
        for child in value:
            aggregate_only(child, location)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite public value: {location}")


def checked_archive(path):
    raw = Path(path).read_bytes()
    if sha(raw) != BASE_SHA256:
        raise ValueError("Base archive is not the verified categorical review")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Duplicate entries or failed ZIP CRC")
        for entry in archive.infolist():
            safe_name(entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlink excluded")
        payload = {name: archive.read(name) for name in names}
    manifest = json.loads(payload["PACKAGE-MANIFEST.json"])
    if set(manifest["artifact_sha256"]) != set(payload) - {"PACKAGE-MANIFEST.json"}:
        raise ValueError("Base manifest does not cover its exact payload")
    for name, expected in manifest["artifact_sha256"].items():
        if sha(payload[name]) != expected:
            raise ValueError(f"Base hash mismatch: {name}")
    if sha(payload["report.pdf"]) != ORIGINAL_PDF_SHA256:
        raise ValueError("Original coursework PDF changed")
    if sha(payload["research-supplement.pdf"]) != CATEGORICAL_PDF_SHA256:
        raise ValueError("Categorical supplement changed")
    return payload


def checked_evidence(directory, required):
    """Public SHA index must cover exactly the declared aggregate artifact set."""
    payload = {name: read_safe(directory, name) for name in (*required, "SHA256.json")}
    seal = json.loads(payload["SHA256.json"])
    if set(seal) != set(required):
        raise ValueError("Unexpected public evidence file set")
    for name, expected in seal.items():
        if sha(payload[name]) != expected:
            raise ValueError(f"Public evidence hash mismatch: {name}")
        if name.endswith(".json"):
            aggregate_only(json.loads(payload[name]))
    return payload


def check_sources(sources, root=ROOT, relative_to=""):
    normalized = {}
    for name, expected in sources.items():
        relative = str(PurePosixPath(relative_to) / name)
        raw = read_safe(root, relative)
        if sha(raw) != expected:
            raise ValueError(f"Scientific source changed: {relative}")
        normalized[relative] = expected
    return normalized


def recording_evidence(directory, root=ROOT):
    payload = checked_evidence(directory, ("aggregates.json", "provenance.json", "RESULTS.md"))
    data = json.loads(payload["aggregates.json"])
    provenance = json.loads(payload["provenance.json"])
    if any(data.get(key) is not False for key in
           ("fit_performed", "ranking_evaluated", "valid_files_read", "test_files_read", "held_out_values_parsed")):
        raise ValueError("Recording description must remain TRAIN-only")
    sources = check_sources(provenance["code_sha256"], root, STUDY + "/recording_audit")
    return {"payload": payload, "data": data, "provenance": provenance, "sources": sources}


def predictive_evidence(directory, root=ROOT, audit_dir=None):
    """Check the published experiment without opening private predictions or labels."""
    payload = checked_evidence(directory, ("aggregates.json", "provenance.json", "protocol.json"))
    data, provenance, protocol = (json.loads(payload[name]) for name in
                                  ("aggregates.json", "provenance.json", "protocol.json"))
    for value in (data, provenance, protocol):
        if any(value.get(key) is not False for key in ("original_validation_read", "original_test_read")):
            raise ValueError("Predictive evidence must not reopen original held-out splits")
    if any(value.get("fresh_population_test") is not False for value in (data, protocol)):
        raise ValueError("Nested exploration must not be presented as a fresh population test")
    if any(value.get("status") != "complete" for value in (data, provenance)):
        raise ValueError("Predictive producer is incomplete")
    if data["study"] != "nested_original_train_exploratory" or protocol["study"] != data["study"]:
        raise ValueError("Unexpected experiment kind")
    if provenance["aggregate_sha256"] != sha(payload["aggregates.json"]) or provenance["protocol_sha256"] != sha(payload["protocol.json"]):
        raise ValueError("Producer provenance does not bind public evidence")
    if provenance["source_sha256"] != protocol["source_sha256"] or provenance["runtime"] != protocol["runtime"]:
        raise ValueError("Frozen producer source/runtime differs")
    sources = check_sources(provenance["source_sha256"], root)
    if (protocol["candidate_count"] != 39 or set(protocol["roles"]) != set(ROLES) or
            protocol["pair_lambdas"] != [50., 250., 1000.] or protocol["betas"] != [.1, 1.] or
            protocol["ease_lambdas"] != [10., 30., 50., 100., 250., 300., 1000., 3000., 10000.]):
        raise ValueError("Unexpected declared grid")
    if data["split_counts"] != protocol["split_counts"] or set(data["selected"]) != set(ROLES) or set(data["assessment"]) != set(ROLES):
        raise ValueError("Split or model coverage mismatch")
    counts = data["split_counts"]
    if counts["users"] != 943 or counts["catalog"] != 1682 or sum(counts[k] for k in ("fit", "development", "assessment")) != 80808:
        raise ValueError("Unexpected nested TRAIN population")
    candidates = data["candidates"]
    if len(candidates) != 39 or len({row["candidate_id"] for row in candidates}) != 39:
        raise ValueError("Incomplete or duplicated candidate grid")
    raw_hashes = provenance["raw_run_files_sha256"]
    if raw_hashes["SELECTIONS-FROZEN.json"] != provenance["selection_seal_sha256"] or raw_hashes["protocol.json"] != provenance["protocol_sha256"]:
        raise ValueError("Published selection-seal chain differs")
    for role in ROLES:
        pool = [row for row in candidates if row["role"] == role]
        expected = ([(value, 0.) for value in protocol["ease_lambdas"]] if role == "EASE" else
                    [(value, beta) for value in protocol["pair_lambdas"] for beta in protocol["betas"]])
        if [(row["lambda"], row["beta"]) for row in pool] != expected:
            raise ValueError(f"Grid order differs: {role}")
        best = max(pool, key=lambda row: row["development"]["aggregate"]["ndcg@10"])
        selected = data["selected"][role]
        if any(selected.get(key) != value for key, value in best.items()):
            raise ValueError(f"Selection is not first exact development maximum: {role}")
        if raw_hashes[selected["scores_file"]] != selected["scores_file_sha256"]:
            raise ValueError(f"Selected score hash differs: {role}")
        primary = data["assessment"][role]["all_recorded"]
        if primary["users"] != counts["users"] or primary["positive_pairs"] != counts["assessment"]:
            raise ValueError("Primary assessment denominator differs")
        for metric in ("ndcg@10", "recall@10"):
            if not 0 <= primary["aggregate"][metric] <= 1:
                raise ValueError("Invalid bounded ranking metric")
    for comparison, other_roles in (("true_minus_bag", ["bag"]), ("true_minus_EASE", ["EASE"]),
                                    ("true_minus_mean_shuffle", ["shuffle_0", "shuffle_1", "shuffle_2"])):
        for metric in ("ndcg@10", "recall@10"):
            entry = data["paired_comparisons"][comparison][metric]
            expected = data["assessment"]["true_group"]["all_recorded"]["aggregate"][metric] - sum(
                data["assessment"][role]["all_recorded"]["aggregate"][metric] for role in other_roles) / len(other_roles)
            if not math.isclose(entry["mean_delta"], expected, abs_tol=1e-12) or entry["users"] != counts["users"]:
                raise ValueError("Paired contrast does not match common-cohort aggregates")
            if entry["replicates"] != 2000 or len(entry["ci95"]) != 2 or entry["ci95"][0] > entry["ci95"][1]:
                raise ValueError("Invalid declared paired interval")
    values = {role: data["assessment"][role]["all_recorded"]["aggregate"]["ndcg@10"] for role in ROLES}
    gain = values["true_group"] / values["EASE"] - 1
    shuffle_mean = sum(values[role] for role in ("shuffle_0", "shuffle_1", "shuffle_2")) / 3
    expected = {"threshold_relative_gain": .1, "beats_bag": values["true_group"] > values["bag"],
                "beats_shuffle_mean": values["true_group"] > shuffle_mean,
                "declared_success": gain >= .1 and values["true_group"] > values["bag"] and values["true_group"] > shuffle_mean}
    if any(data["success"][key] != value for key, value in expected.items()) or not math.isclose(data["success"]["relative_gain_over_ease"], gain, abs_tol=1e-12):
        raise ValueError("Success-rule arithmetic differs")
    if audit_dir is None:
        audit_dir = Path(directory).parent / "audit-v1"
    audit_payload = checked_evidence(audit_dir, ("audit.json",))
    audit = json.loads(audit_payload["audit.json"])
    audit_relative = STUDY + "/predictive/verify_results.py"
    if (audit.get("status") != "PASS" or audit["audit_source_sha256"] != digest(root / audit_relative) or
            audit["run_manifest_sha256"] != sha(payload["provenance.json"]) or
            audit["selection_seal_sha256"] != provenance["selection_seal_sha256"] or
            any(audit.get(key) is not False for key in ("original_test_read", "original_validation_read", "fresh_population_test", "new_model_search"))):
        raise ValueError("Independent audit is missing, stale or outside the declared scope")
    if audit["checks"]["maximum_metric_error"] > 1e-10 or audit["checks"]["declared_candidates_and_selection_choices"] != 39:
        raise ValueError("Independent metric/grid audit did not pass")
    if any(not math.isclose(audit["primary_ndcg"][role], values[role], abs_tol=1e-12) for role in ROLES):
        raise ValueError("Independent primary metric replay differs")
    sources[audit_relative] = audit["audit_source_sha256"]
    return {"payload": payload, "data": data, "provenance": provenance, "protocol": protocol, "sources": sources,
            "audit": audit, "audit_payload": audit_payload,
            "verification": {"candidate_rows": 39, "selections_checked": 6, "paired_means_checked": 6,
                             "selected_hashes_checked": 6, "private_score_or_label_files_read": False}}


def main_content(path, base, recording):
    raw = Path(path).read_bytes()
    content = json.loads(raw)
    import report
    counts = report.validate_content(content)
    if content["status"] != "DRAFT" or str(content["group"]) != "24" or content.get("appendices"):
        raise ValueError("Require group-24 draft core with three tasks and no appendices")
    old = json.loads(base["report-content.json"])
    for key in ("aggregate_evidence", "canonical_roles", "source_sha256", "members", "final_evaluation"):
        if content.get(key) != old.get(key):
            raise ValueError(f"Core scientific or contributor metadata changed: {key}")
    revision = content["editorial_revision"]
    expected = {"original_content_sha256": sha(base["report-content.json"]),
                "original_pdf_sha256": ORIGINAL_PDF_SHA256,
                "original_report_manifest_sha256": sha(base["REPORT-MANIFEST.json"]),
                "recording_audit_aggregate_sha256": sha(recording["payload"]["aggregates.json"])}
    if any(revision.get(k) != v for k, v in expected.items()):
        raise ValueError("Editorial revision is not bound to original evidence")
    return content, raw, counts


def build_grouping_content(recording, predictive):
    data = predictive["data"]
    structure = recording["data"]["recording_structure"]
    genre = recording["data"]["genre_comparison"]["statistics"]["pair_weighted"]
    rows = []
    for role in ROLES:
        selected = data["selected"][role]
        result = data["assessment"][role]["all_recorded"]["aggregate"]
        rows.append([LABELS[role], f"{selected['lambda']:g} / {selected['beta']:g}",
                     f"{result['ndcg@10']:.4f}", f"{result['recall@10']:.4f}"])
    contrasts = []
    for name, label in (("true_minus_EASE", "vs EASE"), ("true_minus_bag", "vs unordered"),
                        ("true_minus_mean_shuffle", "vs shuffle mean")):
        entry = data["paired_comparisons"][name]["ndcg@10"]
        contrasts.append([label, f"{entry['mean_delta']:+.4f}",
                          f"[{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}]"])
    success, counts = data["success"], data["split_counts"]
    decision = "met" if success["declared_success"] else "did not meet"
    discussion = (
        f"Recording groups {decision} the declared research-priority rule: at least 10% relative nDCG gain over "
        f"tuned EASE and higher means than unordered pairs and shuffled groups. The measured EASE-relative change "
        f"was {100 * success['relative_gain_over_ease']:+.2f}%. Coherent recording groups alone do not establish "
        "predictive value. The unordered control tests the grouping restriction; timestamp shuffles retain each "
        "user's movies and group sizes. They do not identify exposure, interface design, or watching sessions. "
        "Pairwise linear recommendation is established; the contribution here is a controlled input-structure test. "
        "This single nested split follows earlier dataset exploration. Its assessment is separated from model "
        "selection, but is not fresh population confirmation or directly comparable with the preceding test tables.")
    content = {
        "schema_version": 1, "title": "Appendix: do rating-recording groups improve ranking?",
        "scope": "REVIEW DRAFT | Exploratory nested TRAIN split. Original VALID/TEST not accessed in this experiment.",
        "method": (f"Motivation: {100 * structure['group_thresholds']['2']['records']['fraction']:.2f}% of original TRAIN "
                   f"ratings share a user's timestamp. Within-group genre Jaccard is {genre['observed']:.3f} versus "
                   f"{genre['null_mean']:.3f} across 100 within-user shuffles. This is descriptive evidence."),
        "evaluation": (f"Known constrained ridge, singleton + pair features; own-target features excluded. "
                       f"F/D/A = {counts['fit']:,}/{counts['development']:,}/{counts['assessment']:,} records; "
                       f"{counts['users']} users. All 39 candidates use F context; D selects, A assesses after sealing. "
                       "All 1,682 items eligible except F observations; no refit or target timestamps."),
        "tables": [
            {"caption": "All-recorded assessment at ten (every selected arm)",
             "columns": ["Model", "lambda / beta", "nDCG", "Recall"], "rows": rows,
             "widths": [.36, .22, .21, .21]},
            {"caption": "Recording groups minus control: paired nDCG",
             "columns": ["Control", "Mean change", "95% interval"], "rows": contrasts,
             "widths": [.36, .25, .39]}],
        "note": "Intervals: 2,000 paired-user resamples, descriptive and not multiplicity adjusted. Shuffle comparison averages metrics, never scores. Full subgroup/secondary metrics remain in aggregate evidence.",
        "discussion": discussion,
        "source_note": "Prior: Steck & Liang (2021), Negative Interactions for Improved Collaborative Filtering: Don’t go Deeper, go Higher. doi:10.1145/3460231.3474273. Protocol, source hashes and all candidates accompany this report.",
        "evidence_stage": "exploratory nested original TRAIN assessment",
        "recording_evidence_sha256": {name: sha(raw) for name, raw in recording["payload"].items()},
        "predictive_evidence_sha256": {name: sha(raw) for name, raw in predictive["payload"].items()},
        "predictive_audit_sha256": {name: sha(raw) for name, raw in predictive["audit_payload"].items()},
        "source_sha256": {**recording["sources"], **predictive["sources"]},
    }
    if len(discussion.split()) > 200:
        raise ValueError("Generated discussion is too long")
    return content


def render_grouping(content, output, font_dir=None):
    """One A4 page, 12pt real Times New Roman and fixed 13.8pt line spacing."""
    import report
    if len(content["discussion"].split()) > 200:
        raise ValueError("Grouping discussion exceeds 200 words")
    regular, bold = report.find_fonts(font_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "framing-review-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    width, height, margin, leading = report.PAGE_WIDTH, report.PAGE_HEIGHT, 40., 13.8
    fonts = {False: FontProperties(fname=str(regular), size=12),
             True: FontProperties(fname=str(bold), size=12)}
    figure = Figure(figsize=(width / 72, height / 72))
    measure, y = TextToPath(), height - margin

    def wrap(value, limit, strong=False):
        lines, line = [], ""
        for word in str(value).split():
            if measure.get_text_width_height_descent(word, fonts[strong], False)[0] > limit:
                raise ValueError(f"Unwrappable word: {word}")
            proposed = word if not line else line + " " + word
            if line and measure.get_text_width_height_descent(proposed, fonts[strong], False)[0] > limit:
                lines.append(line)
                line = word
            else:
                line = proposed
        return lines + ([line] if line else [])

    def draw(text, x=margin, strong=False):
        if y < margin + leading:
            raise ValueError("Grouping page overflows at required font size")
        figure.text(x / width, y / height, text, fontproperties=fonts[strong], va="top", parse_math=False)

    def paragraph(value, strong=False):
        nonlocal y
        for line in wrap(value, width - 2 * margin, strong):
            draw(line, strong=strong)
            y -= leading
        y -= 4

    paragraph(content["title"], True)
    for key in ("scope", "method", "evaluation"):
        paragraph(content[key])
    for table in content["tables"]:
        paragraph(table["caption"], True)
        if abs(sum(table["widths"]) - 1) > 1e-8:
            raise ValueError("Invalid table widths")
        for row_index, row in enumerate([table["columns"], *table["rows"]]):
            if len(row) != len(table["widths"]):
                raise ValueError("Table shape mismatch")
            cells = [wrap(cell, (width - 2 * margin) * share - 6, row_index == 0)
                     for cell, share in zip(row, table["widths"])]
            start, x = y, margin
            extent = max(map(len, cells)) * leading + 2
            for cell, share in zip(cells, table["widths"]):
                for offset, line in enumerate(cell):
                    y = start - offset * leading
                    draw(line, x, row_index == 0)
                x += (width - 2 * margin) * share
            y = start - extent
        y -= 3
    paragraph(content["note"])
    paragraph("Discussion", True)
    paragraph(content["discussion"])
    paragraph(content["source_note"])
    figure.text(margin / width, 25 / height, "REVIEW DRAFT | Exploratory nested TRAIN assessment | 1/1",
                fontproperties=fonts[False])
    metadata = {"Title": content["title"], "Author": "Vlad George Iftode",
                "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
                "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc), "Creator": STUDY + "/build_review.py"}
    with matplotlib.rc_context({"pdf.fonttype": 42, "pdf.compression": 6}):
        with PdfPages(output, metadata=metadata) as pdf:
            pdf.savefig(figure)
    return {"pages": 1, "font": "Times New Roman", "font_size": 12, "line_spacing": 1.15,
            "bottom_pt": round(y, 2), "font_sha256": {"regular": digest(regular), "bold": digest(bold)}}


def merge_report(core, grouping, output, font_dir=None):
    """Keep source PDFs unchanged; overlay only page footers in the merged copy."""
    import report
    from pypdf import PdfReader, PdfWriter
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    import matplotlib
    reader_core, reader_group = PdfReader(core), PdfReader(grouping)
    if len(reader_core.pages) != 4 or len(reader_group.pages) != 1:
        raise ValueError("Expected four core pages and one grouping page")
    regular, _ = report.find_fonts(font_dir)
    font = FontProperties(fname=str(regular), size=12)
    width, height = report.PAGE_WIDTH, report.PAGE_HEIGHT
    overlay = io.BytesIO()
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        with PdfPages(overlay) as pdf:
            for number in range(1, 6):
                figure = Figure(figsize=(width / 72, height / 72))
                figure.patch.set_alpha(0)
                from matplotlib.patches import Rectangle
                figure.add_artist(Rectangle((0, 0), 1, 42 / height, transform=figure.transFigure,
                                            facecolor="white", edgecolor="none"))
                stage = "Original frozen evaluation" if number < 5 else "Exploratory nested TRAIN assessment"
                figure.text(40 / width, 25 / height, f"REVIEW DRAFT | {stage} | {number}/5", fontproperties=font)
                pdf.savefig(figure, transparent=True)
    overlay.seek(0)
    footer = PdfReader(overlay)
    writer = PdfWriter()
    for index, page in enumerate([*reader_core.pages, *reader_group.pages]):
        page.merge_page(footer.pages[index])
        writer.add_page(page)
    writer.add_metadata({"/Title": "Group 24: recommender systems review", "/Author": "Vlad George Iftode",
                         "/Creator": STUDY + "/build_review.py", "/CreationDate": "D:20000101000000Z"})
    with Path(output).open("xb") as stream:
        writer.write(stream)
    if len(PdfReader(output).pages) != 5:
        raise ValueError("Merged page count mismatch")


def markdown_grouping(content):
    lines = ["# " + content["title"], "", content["scope"], "", content["method"], "", content["evaluation"], ""]
    for table in content["tables"]:
        lines += ["## " + table["caption"], "", "| " + " | ".join(table["columns"]) + " |",
                  "| " + " | ".join(["---"] * len(table["columns"])) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in table["rows"]]
        lines.append("")
    lines += [content["note"], "", "## Discussion", "", content["discussion"], "", content["source_note"], ""]
    return "\n".join(lines)


def source_payload(root, base, sources):
    payload = {name: raw for name, raw in base.items() if name.startswith(("code/", "evidence/"))}
    names = {STUDY + "/build_review.py", *sources}
    for subdir in ("recording_audit", "predictive"):
        directory = root / STUDY / subdir
        names.update(str(path.relative_to(root)) for path in directory.iterdir()
                     if path.is_file() and path.suffix in {".py", ".md"})
    for relative in sorted(names):
        raw = read_safe(root, relative)
        key = "code/" + relative
        if key in payload and payload[key] != raw:
            raise ValueError(f"Previously packaged dependency changed: {relative}")
        payload[key] = raw
    for relative, expected in sources.items():
        if sha(payload["code/" + relative]) != expected:
            raise ValueError(f"Packaged source is not exact: {relative}")
    return payload


def navigation_payload(root, payload):
    """Current navigation is explicitly separate from archived scientific sources."""
    for name in ("README.md", "HANDOFF.md", "PLAN.md", "exploratory/README.md",
                 STUDY + "/README.md", STUDY + "/COMPLETION_CHECK.md"):
        raw = read_safe(root, name)
        key = "code/" + name
        if key in payload and payload[key] != raw:
            payload["archive/previous-navigation/" + name] = payload[key]
        payload[key] = raw
    return payload


def verify_imports(payload, runtime_python):
    """Import newly packaged modules from an isolated extraction, without datasets."""
    modules = sorted(name[5:-3].replace("/", ".") for name in payload
                     if name.startswith("code/" + STUDY + "/") and name.endswith(".py")
                     and not Path(name).name.startswith("test_") and not name.endswith("/__init__.py"))
    with tempfile.TemporaryDirectory(prefix="framing-review-import-") as temporary:
        extraction = Path(temporary)
        for name, raw in payload.items():
            if name.startswith("code/"):
                destination = extraction / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
        snippet = "import importlib,json; names=" + repr(modules) + "; [importlib.import_module(n) for n in names]; print(json.dumps(names))"
        env = os.environ.copy()
        env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(extraction / "code")
        result = subprocess.run([str(runtime_python), "-c", snippet], cwd=extraction / "code", env=env,
                                capture_output=True, text=True, check=True, timeout=60)
        imported = json.loads(result.stdout.strip())
        if imported != modules:
            raise ValueError("Extracted import verification mismatch")
    return {"status": "complete", "modules": modules, "python": str(runtime_python), "dataset_access": False}


def write_zip(path, payload):
    if Path(path).exists():
        raise FileExistsError(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(payload.items()):
            safe_name(name)
            entry = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, raw)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload):
            raise ValueError("Final ZIP CRC or entry-set mismatch")
        for name, raw in payload.items():
            if sha(archive.read(name)) != sha(raw):
                raise ValueError(f"Final archived hash mismatch: {name}")


def build(args):
    root = ROOT
    if args.out.name != "24.zip":
        raise ValueError("Coursework archive must be named 24.zip")
    if args.out.exists() or (args.out.parent / "verification.json").exists():
        raise FileExistsError(args.out)
    report_dir = args.report_dir
    generated = ("main-report.pdf", "grouping-supplement.pdf", "grouping-supplement.md", "grouping-content.json",
                 "report.pdf", "report.md", "report-content.json", "REPORT-MANIFEST.json")
    if any((report_dir / name).exists() for name in generated):
        raise FileExistsError("Generated report artifacts already exist")
    self_hash = digest(__file__)
    base = checked_archive(args.base_archive)
    recording = recording_evidence(args.recording_evidence)
    predictive = predictive_evidence(args.predictive_evidence, audit_dir=args.predictive_audit)
    core, core_raw, counts = main_content(args.main_content, base, recording)
    grouping = build_grouping_content(recording, predictive)
    sources = {**recording["sources"], **predictive["sources"], "report.py": digest(root / "report.py"),
               STUDY + "/build_review.py": self_hash}
    payload = navigation_payload(root, source_payload(root, base, sources))
    for name, raw in base.items():
        if name.startswith(("code/", "evidence/")):
            continue
        if name in {"report.pdf", "report.md", "report-content.json", "REPORT-MANIFEST.json"}:
            destination = "archive/original-coursework/" + name
        else:
            destination = "archive/categorical-review/" + name
        payload[destination] = raw
    for label, evidence in (("recording-audit-v1", recording), ("framing-predictive-v1", predictive)):
        for name, raw in evidence["payload"].items():
            payload["evidence/" + label + "/" + name] = raw
    for name, raw in predictive["audit_payload"].items():
        payload["evidence/framing-predictive-audit-v1/" + name] = raw
    # Only numerical imports are run; imported code receives no dataset or run files.
    # Keep the venv symlink path: resolving it would bypass its site-packages.
    imports = verify_imports(payload, args.runtime_python.absolute())
    report_dir.mkdir(parents=True, exist_ok=True)
    import report
    core_layout = report.render_pdf(core, report_dir / "main-report.pdf", args.font_dir)
    group_layout = render_grouping(grouping, report_dir / "grouping-supplement.pdf", args.font_dir)
    merge_report(report_dir / "main-report.pdf", report_dir / "grouping-supplement.pdf", report_dir / "report.pdf", args.font_dir)
    (report_dir / "grouping-supplement.md").write_text(markdown_grouping(grouping), encoding="utf-8")
    (report_dir / "grouping-content.json").write_bytes(json_bytes(grouping))
    (report_dir / "report.md").write_text(report.render_markdown(core) + "\n\n" + markdown_grouping(grouping), encoding="utf-8")
    combined = {"schema_version": 1, "status": "REVIEW DRAFT", "group": "24", "main": core, "grouping": grouping}
    (report_dir / "report-content.json").write_bytes(json_bytes(combined))
    # Source, aggregate and navigation snapshots must remain unchanged throughout assembly.
    check_sources(sources)
    for name, raw in payload.items():
        if name.startswith("code/") and name[5:] in {
                "README.md", "HANDOFF.md", "PLAN.md", "exploratory/README.md", STUDY + "/README.md", STUDY + "/COMPLETION_CHECK.md"}:
            if read_safe(root, name[5:]) != raw:
                raise ValueError("Navigation changed while packaging")
    for directory, evidence in ((args.recording_evidence, recording), (args.predictive_evidence, predictive)):
        if any(read_safe(directory, name) != raw for name, raw in evidence["payload"].items()):
            raise ValueError("Aggregate evidence changed while rendering")
    audit_dir = args.predictive_audit or args.predictive_evidence.parent / "audit-v1"
    if any(read_safe(audit_dir, name) != raw for name, raw in predictive["audit_payload"].items()):
        raise ValueError("Independent audit changed while rendering")
    if args.main_content.read_bytes() != core_raw:
        raise ValueError("Main content changed while rendering")
    report_manifest = {
        "schema_version": 1, "status": "REVIEW DRAFT", "group": "24", "contributors_supplied": len(core["members"]) == 5,
        "pages": 5, "main_evidence_stage": "previously frozen held-out test", "appendix_evidence_stage": grouping["evidence_stage"],
        "layout": {"core": core_layout, "grouping": group_layout, "merged_pages": 5},
        "discussion_words": {**counts, "grouping": len(grouping["discussion"].split())},
        "original_report_sha256": ORIGINAL_PDF_SHA256, "categorical_supplement_sha256": CATEGORICAL_PDF_SHA256,
        "main_content_sha256": sha(core_raw), "source_sha256": sources,
        "recording_evidence_sha256": grouping["recording_evidence_sha256"],
        "predictive_evidence_sha256": grouping["predictive_evidence_sha256"],
        "predictive_audit_sha256": grouping["predictive_audit_sha256"],
        "producer_checks": predictive["verification"], "extracted_source_imports": imports,
        "artifact_sha256": {name: digest(report_dir / name) for name in generated if name != "REPORT-MANIFEST.json"}}
    (report_dir / "REPORT-MANIFEST.json").write_bytes(json_bytes(report_manifest))
    payload["main-report-content.json"] = core_raw
    for name in generated:
        payload[name] = (report_dir / name).read_bytes()
    payload["README.md"] = package_readme().encode()
    package_manifest = {"schema_version": 1, "status": "REVIEW DRAFT", "group": "24",
        "entrypoint": "report.pdf", "pages": 5, "contributors_pending": len(core["members"]) != 5,
        "base_archive_sha256": BASE_SHA256, "original_report_sha256": ORIGINAL_PDF_SHA256,
        "categorical_supplement_sha256": CATEGORICAL_PDF_SHA256, "generator_sha256": self_hash,
        "report_manifest_sha256": sha(payload["REPORT-MANIFEST.json"]),
        "source_sha256": sources, "exclusions": ["raw ratings", "user IDs", "histories", "individual predictions",
        "scores", "checkpoints", "credentials", "font binaries", "coursework brief"],
        "artifact_sha256": {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload["PACKAGE-MANIFEST.json"] = json_bytes(package_manifest)
    write_zip(args.out, payload)
    verification = {"schema_version": 1, "status": "complete", "archive": str(args.out),
        "archive_sha256": digest(args.out), "bytes": args.out.stat().st_size, "entries": len(payload),
        "zip_crc_valid": True, "all_artifact_hashes_valid": True,
        "original_report_unchanged": sha(payload["archive/original-coursework/report.pdf"]) == ORIGINAL_PDF_SHA256,
        "categorical_supplement_unchanged": sha(payload["archive/categorical-review/research-supplement.pdf"]) == CATEGORICAL_PDF_SHA256,
        "source_import_verification": imports, "report_manifest_sha256": digest(report_dir / "REPORT-MANIFEST.json"),
        "raw_data_included": False, "report_pages": 5, "status_of_submission": "REVIEW DRAFT; not submitted",
        "verification_limits": "Public hashes/grid/aggregate arithmetic and isolated imports; no model refit or raw-label audit by this builder."}
    (args.out.parent / "verification.json").write_bytes(json_bytes(verification))
    return verification


def package_readme():
    return """# Group 24 review bundle

Start with **report.pdf**: cover, three required coursework tasks and one exploratory grouping appendix (five pages).
This is a REVIEW DRAFT. Four contributor names and each student's honest peer evaluation remain human tasks.
No submission has been made. Names or contributions were not invented.

The core reports the already frozen evaluation. The new appendix reports a nested experiment inside original TRAIN after prior dataset exploration; it does not reopen original validation/test and is not a fresh population test. The two stages are labelled separately and their scores should not be compared directly.

The original coursework PDF is preserved byte for byte at `archive/original-coursework/report.pdf`.
The earlier categorical supplement remains byte for byte at `archive/categorical-review/research-supplement.pdf`, with its report/audit files below `archive/categorical-review/research/`.
Current source/navigation is under `code/`; older overwritten navigation is retained under `archive/previous-navigation/`.
Aggregate evidence is under `evidence/`, including every grouping candidate, all six selected arms, secondary endpoints and subgroup results. No raw ratings, identity lists, score arrays or checkpoints are distributed.

## Reproduce and review

Use `code/README.md` for the core experiment, `code/exploratory/categorical_reconstruction/README.md` for the categorical extension, and `code/exploratory/framing_search/predictive/README.md` for the nested grouping experiment. The latter requires the separately obtained MovieLens100K data and original split/hash prerequisites; the originals are intentionally not redistributed. The exact experiment invocation and source/runtime hashes are in `evidence/framing-predictive-v1/protocol.json`.

From `code/`, run the synthetic tests with the verified numerical environment:

```sh
python -m unittest discover -s exploratory/framing_search/predictive -p 'test_*.py' -v
python -m unittest exploratory.framing_search.recording_audit.test_audit -v
```

To regenerate the five-page PDF from the packaged public evidence/content, use a Python environment with Matplotlib and pypdf, installed Times New Roman regular/bold fonts, and a new output directory:

```sh
python exploratory/framing_search/build_review.py --render-packaged .. --report-dir ../rerendered
```

This reads only packaged aggregate evidence/content. It does not fit or evaluate a recommender. `PACKAGE-MANIFEST.json` binds every other entry; `REPORT-MANIFEST.json` records font digests, exact discussion counts and separate evidence stages. ZIP CRC and extracted-source imports were checked during assembly. Fonts, assignment slides and external datasets are not bundled.
"""


def render_packaged(directory, out, font_dir=None):
    """Portable report replay from a verified extracted bundle, with no data access."""
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(read_safe(directory, "PACKAGE-MANIFEST.json"))
    for name, expected in manifest["artifact_sha256"].items():
        if sha(read_safe(directory, name)) != expected:
            raise ValueError(f"Packaged artifact changed: {name}")
    if manifest["generator_sha256"] != digest(__file__):
        raise ValueError("Report replay builder differs from the packaged version")
    check_sources(manifest["source_sha256"], directory / "code")
    core = json.loads(read_safe(directory, "main-report-content.json"))
    grouping = json.loads(read_safe(directory, "grouping-content.json"))
    recording = recording_evidence(directory / "evidence/recording-audit-v1", directory / "code")
    predictive = predictive_evidence(directory / "evidence/framing-predictive-v1", directory / "code",
                                     directory / "evidence/framing-predictive-audit-v1")
    if grouping != build_grouping_content(recording, predictive):
        raise ValueError("Grouping content does not reproduce from public aggregates")
    import report
    out.mkdir(parents=True)
    core_layout = report.render_pdf(core, out / "main-report.pdf", font_dir)
    grouping_layout = render_grouping(grouping, out / "grouping-supplement.pdf", font_dir)
    merge_report(out / "main-report.pdf", out / "grouping-supplement.pdf", out / "report.pdf", font_dir)
    result = {"status": "complete", "pages": 5, "core": core_layout, "grouping": grouping_layout,
              "original_report_pdf_sha256": manifest["artifact_sha256"]["report.pdf"],
              "replayed_report_pdf_sha256": digest(out / "report.pdf"), "data_access": False}
    (out / "replay.json").write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=ROOT / "packages/categorical-reconstruction-v1/24-review.zip")
    parser.add_argument("--main-content", type=Path, default=ROOT / "reports/framing-review-v1/main-report-content.json")
    parser.add_argument("--recording-evidence", type=Path, default=ROOT / STUDY / "recording_audit/results-v1")
    parser.add_argument("--predictive-evidence", type=Path)
    parser.add_argument("--predictive-audit", type=Path, help="Defaults to audit-v1 next to predictive results")
    parser.add_argument("--render-packaged", type=Path, help="Replay reports from an extracted verified review bundle")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/framing-review-v1")
    parser.add_argument("--out", type=Path, default=ROOT / "packages/framing-review-v1/24.zip")
    parser.add_argument("--runtime-python", type=Path, default=ROOT / "runs/environment-check/.venv/bin/python")
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.render_packaged:
        result = render_packaged(args.render_packaged, args.report_dir, args.font_dir)
    else:
        if args.predictive_evidence is None:
            parser.error("--predictive-evidence is required for building a review archive")
        result = build(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
