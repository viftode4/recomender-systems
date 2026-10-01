"""Build a new review from public evidence, preserving every scientific freeze.

The prior ZIP is a verified input, never modified. Rendering from an extracted
archive uses only its aggregate evidence and source; no old ZIP or dataset is
needed. Experimental replay is a separate operation documented in REPRODUCE.md.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exploratory.framing_search import build_review as previous

BASE_SHA256 = "c4933e0338bec15d2b03bcfbac27d66de9a47abff8192fbf9d3bd4b835c4d639"
BASE_REPORT_SHA256 = "491ae63f52423d0678daafe8451d1a42913b46da8ec053521d739b3e12103399"
STUDY = "coursework_completion"
SEEDS = ("2026", "2027", "2028")
FAMILIES = ("mixed", "meta", "tuned_switch", "tuned_rrf")
ROLES = ("EASE", "SLIM", "context", "fixed_switch", "fixed_rrf", *FAMILIES)
GRIDS = {"mixed": [[6, 2, 2], [4, 4, 2], [4, 2, 4]], "meta": [.1, 1., 10., 100.],
         "tuned_switch": [2, 3, 4], "tuned_rrf": [10, 60, 100]}
sha, digest, json_bytes = previous.sha, previous.digest, previous.json_bytes
read_safe, safe_name = previous.read_safe, previous.safe_name


def checked_base(path):
    raw = Path(path).read_bytes()
    if sha(raw) != BASE_SHA256:
        raise ValueError("Require the verified framing-review-v1 archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or archive.testzip() is not None:
            raise ValueError("Invalid base archive")
        for entry in archive.infolist():
            safe_name(entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlink excluded")
        payload = {name: archive.read(name) for name in names}
    manifest = json.loads(payload["PACKAGE-MANIFEST.json"])
    if set(manifest["artifact_sha256"]) != set(payload)-{"PACKAGE-MANIFEST.json"}:
        raise ValueError("Base manifest coverage differs")
    for name, expected in manifest["artifact_sha256"].items():
        if sha(payload[name]) != expected:
            raise ValueError(f"Base artifact changed: {name}")
    if sha(payload["report.pdf"]) != BASE_REPORT_SHA256:
        raise ValueError("Prior report changed")
    return payload


def check_sources(sources, root=ROOT):
    if not sources:
        raise ValueError("Scientific source hashes are required")
    return previous.check_sources(sources, root)


def checked_public(directory):
    """Read exactly the producer's allowlisted public artifacts, with no runs."""
    allowed = {"aggregates.json", "protocol.json", "provenance.json", "RESULTS.md"}
    index = json.loads(read_safe(directory, "SHA256.json"))
    if not {"aggregates.json", "protocol.json", "provenance.json"} <= set(index) <= allowed:
        raise ValueError("Unexpected completion evidence files")
    result = {"SHA256.json": read_safe(directory, "SHA256.json")}
    for name, expected in index.items():
        raw = read_safe(directory, name)
        if sha(raw) != expected:
            raise ValueError(f"Completion evidence changed: {name}")
        if name.endswith(".json"):
            previous.aggregate_only(json.loads(raw))
        result[name] = raw
    return result


def validate_completion(data, protocol):
    """Validate roles, ordered tuning decisions and equally weighted means."""
    if (data.get("status") != "complete" or data.get("original_test_read") is not False or
            data.get("fresh_holdout") is not False or protocol.get("original_test_read") is not False or
            protocol.get("fresh_holdout") is not False or
            data.get("assessment_stage") != "exploratory reused validation calibration cohort" or
            data["assessment_stage"] != protocol.get("assessment_stage") or
            tuple(map(str, protocol.get("seeds", []))) != SEEDS or set(data.get("seeds", {})) != set(SEEDS) or
            protocol.get("grids") != GRIDS or protocol.get("candidate_count") != 39 or
            tuple(protocol.get("roles", [])) != ROLES):
        raise ValueError("Completion scope, roles or declared grids differ")
    for seed in SEEDS:
        record = data["seeds"][seed]
        if (record["cohort_sizes"] != {"meta_fit": 471, "development": 236, "calibration": 236} or
                set(record["assessment"]) != set(ROLES) or set(record["selections"]) != set(FAMILIES) or
                set(record["references"]) != set(ROLES[:5])):
            raise ValueError("Completion seed/cohort/role coverage differs")
        candidates = record["candidates"]
        if len(candidates) != 13 or len({row["candidate_id"] for row in candidates}) != 13:
            raise ValueError("Completion candidate grid is incomplete")
        for family in FAMILIES:
            rows = [row for row in candidates if row["family"] == family]
            key = {"mixed": "quotas", "meta": "lambda", "tuned_switch": "group_count", "tuned_rrf": "offset"}[family]
            if ([row["settings"][key] for row in rows] != GRIDS[family] or
                    [row["candidate_id"] for row in rows] != [f"{family}-{index}" for index in range(len(rows))] or
                    any(row["development"]["users"] != 236 for row in rows)):
                raise ValueError("Completion grid order or development cohort differs")
            if record["selections"][family] != max(rows, key=lambda row: row["development"]["aggregate"]["ndcg@10"]):
                raise ValueError("Completion choice is not the first exact development maximum")
        pairs = set()
        for role in ROLES:
            row = record["assessment"][role]
            if row["users"] != 236 or row["positive_pairs"] <= 0:
                raise ValueError("Completion assessment denominator differs")
            pairs.add(row["positive_pairs"])
            if any(not 0 <= row["aggregate"][metric] <= 1 for metric in ("ndcg@10", "recall@10", "mrr@10")):
                raise ValueError("Invalid completion ranking metric")
        if len(pairs) != 1:
            raise ValueError("Completion roles use different positive-pair counts")
    if set(data["mean_metrics"]) != set(ROLES):
        raise ValueError("Completion mean role coverage differs")
    means = {}
    for role in ROLES:
        metrics = data["seeds"][SEEDS[0]]["assessment"][role]["aggregate"]
        means[role] = {metric: math.fsum(data["seeds"][seed]["assessment"][role]["aggregate"][metric]
                                       for seed in SEEDS)/3 for metric in metrics}
        if set(data["mean_metrics"][role]) != set(metrics) or any(not math.isclose(value,
                data["mean_metrics"][role][metric], rel_tol=0, abs_tol=1e-12) for metric, value in means[role].items()):
            raise ValueError("Published completion mean does not equal three split means")
    return means


def completion_evidence(directory, audit_dir, root=ROOT):
    payload = checked_public(directory)
    data, protocol, provenance = [json.loads(payload[name]) for name in
                                 ("aggregates.json", "protocol.json", "provenance.json")]
    means = validate_completion(data, protocol)
    if (provenance.get("status") != "complete" or provenance.get("original_test_read") is not False or
            provenance.get("fresh_holdout") is not False or provenance.get("assessment_after_all_seed_selections") is not True or
            provenance["protocol_sha256"] != sha(payload["protocol.json"]) or
            provenance["aggregate_sha256"] != sha(payload["aggregates.json"]) or
            provenance["source_sha256"] != protocol["source_sha256"] or provenance["runtime"] != protocol["runtime"] or
            provenance["output_sha256"]["SELECTIONS-FROZEN.json"] != provenance["selection_seal_sha256"]):
        raise ValueError("Completion provenance does not bind the assessment to its selection barrier")
    sources = check_sources(protocol["source_sha256"], root)
    audit_payload = {"SHA256.json": read_safe(audit_dir, "SHA256.json")}
    index = json.loads(audit_payload["SHA256.json"])
    if "audit.json" not in index or not set(index) <= {"audit.json", "RESULTS.md", "REVIEW.md", "verify.py"}:
        raise ValueError("Unexpected independent-audit artifact set")
    for name, expected in index.items():
        # The verifier indexes its own source without copying it into the audit
        # directory. Include that exact source copy beside the unchanged index.
        audit_payload[name] = (read_safe(root, STUDY+"/verify.py") if name == "verify.py"
                               else read_safe(audit_dir, name))
        if sha(audit_payload[name]) != expected:
            raise ValueError("Independent audit artifact changed")
    audit = json.loads(audit_payload["audit.json"])
    previous.aggregate_only(audit)
    artifacts = {Path(name).name: value for name, value in audit["scientific_public_artifacts"].items()}
    expected = {name: sha(payload[name]) for name in ("aggregates.json", "protocol.json", "provenance.json")}
    if (len(artifacts) != len(audit["scientific_public_artifacts"]) or artifacts != expected or
            audit.get("status") != "passed" or audit["scientific_source_sha256"] != protocol["source_sha256"] or
            audit["selection_barrier_sha256"] != provenance["selection_seal_sha256"] or
            audit["assessment_manifest_sha256"] != sha(payload["provenance.json"]) or
            audit["maximum_absolute_error"] > 1e-9 or audit["checks_completed"] <= 0):
        raise ValueError("Independent audit is incomplete or does not bind this exact run")
    verifier = audit["verifier_source_sha256"]
    verifier_sources = {STUDY+"/verify.py": verifier} if isinstance(verifier, str) else verifier
    if STUDY+"/verify.py" not in verifier_sources:
        raise ValueError("Independent verifier source is not bound")
    sources.update(check_sources(verifier_sources, root))
    return {"payload": payload, "audit_payload": audit_payload, "data": data, "means": means,
            "sources": sources, "audit": audit}


def build_main_content(original, completion):
    """Preserve frozen numerical tables; add clearly separate completion results."""
    core = copy.deepcopy(original)
    if core["status"] != "DRAFT" or str(core["group"]) != "24" or core.get("appendices"):
        raise ValueError("Expected the verified three-task group-24 draft")
    task = core["sections"][0]
    means = completion["means"]
    number = lambda role: f"{means[role]['ndcg@10']:.4f}"
    core["cover_notes"][0] = (
        "The original tables average frozen test evaluations for seeds 2026, 2027 and 2028. "
        "Task 1 separately labels later hybrid completion on reused validation-calibration users. "
        "The grouping appendix uses a different nested TRAIN assessment. These stages are not one leaderboard.")
    core["cover_notes"][-1] = (
        "The lecture's seven hybrid families are mapped to explicit implementations. Earlier reports and negative "
        "research evidence are preserved separately; added coverage does not establish novelty or guarantee improvement.")
    core["references"].append("[11] Course W4S2-Hybrid lecture, pp. 6–12: weighted, switching, mixed, feature combination, feature augmentation, cascade and meta-level hybrids.")
    task["paragraphs"] = [
        "MovieLens 100K; k=10; all recorded ratings relevant. Frozen test: three overlapping 943-user splits, "
        "each with 9,596 held-out records; TRAIN+validation items masked. Seeds reuse people and observations.",
        "Lecture families [11]: weighted regression; activity switching; mixed lists; score interactions with activity, "
        "genre entropy and popularity (feature combination and augmentation); societal reranking (cascade); "
        "genre profiles feeding a collaborative decoder (meta-level). Labels can overlap.",
        "Ridge uses standardized expert scores: static is unconstrained; constrained weights sum to one; calibrated "
        "applies that constraint after response alignment. Pairwise fits score differences; RRF sums reciprocal ranks. "
        "Original ridge penalties: .001/.01/.1/1.",
        "Completion: 471 meta-fit/236 selection/236 previously exposed calibration users per split; TRAIN-only masking. "
        "Grids: mixed quotas 6/2/2, 4/4/2, 4/2/4; meta penalties .1/1/10/100; switch groups 2/3/4; RRF offsets "
        "10/60/100. All choices sealed before assessment.",
    ]
    task["tables"].append({
        "caption": "Separate exploratory completion: mean assessment nDCG@10",
        "columns": ["Model", "nDCG", "Matched model", "nDCG"],
        "widths": [.36, .14, .36, .14],
        "rows": [["Mixed lists", number("mixed"), "Meta-level", number("meta")],
                 ["Tuned switch", number("tuned_switch"), "Fixed switch", number("fixed_switch")],
                 ["Tuned RRF", number("tuned_rrf"), "Fixed RRF", number("fixed_rrf")],
                 ["EASE", number("EASE"), "Context regression", number("context")]],
    })
    task["discussion"] = (
        "Frozen disagreement fusion scores 0.3370 versus 0.3161 for the validation-selected standalone reference "
        "(+6.6% relative); static fusion scores 0.3356. Response alignment raises constrained nDCG from 0.268 "
        "to 0.334, exposing a score-scale cost; negative coefficients are allowed, not probabilities. "
        "FISMCorrected repairs supplied loss/history errors, not the paper's exact objective. LightGCN's budget "
        "extension followed improving validation curves. Mixed and meta-level score below matched EASE; switching "
        "is unchanged and RRF tuning changes "
        "utility little. Reused assessment cannot establish fresh generalization or novelty.")
    # Scientific values are not rewritten when improving the historical narrative.
    old_sentence = "It motivates the separately declared grouping study while keeping hidden target timestamps unavailable."
    new_sentence = "The separately declared grouping study kept target timestamps hidden and found no ranking advantage (appendix)."
    core["sections"][1]["discussion"] = core["sections"][1]["discussion"].replace(old_sentence, new_sentence)
    old_policy_sentence = "The independent calibration target is not a held-out guarantee."
    policy_description = (
        "Per TRAIN-activity group, independent calibration selects exposure strength to minimize the catalog-share gap, "
        "requiring a nonnegative Bonferroni-bootstrap lower bound on mean nDCG minus 95% of baseline nDCG. "
        "This is not a held-out guarantee.")
    core["sections"][2]["discussion"] = core["sections"][2]["discussion"].replace(
        old_policy_sentence, policy_description)
    for table in core["sections"][1]["tables"]:
        if table["columns"] == ["Model", "Sparse", "Medium", "Dense", "Head", "Tail"]:
            # Separate user-activity columns from the item-popularity columns.
            # This is layout only; every historical row remains unchanged.
            table["widths"] = [.24, .14, .16, .18, .14, .14]
    core["completion_evidence"] = {"stage": "exploratory reused validation calibration cohort",
        "aggregates": completion["data"], "source_sha256": completion["sources"],
        "artifact_sha256": {name: sha(raw) for name, raw in completion["payload"].items()},
        "audit_sha256": {name: sha(raw) for name, raw in completion["audit_payload"].items()}}
    if task["tables"][0] != original["sections"][0]["tables"][0]:
        raise ValueError("Frozen test table changed")
    for key in ("aggregate_evidence", "canonical_roles", "source_sha256", "members", "final_evaluation"):
        if core[key] != original[key]:
            raise ValueError("Original scientific or contributor metadata changed")
    import report
    report.validate_content(core)
    return core


def resolve_packaged_target(source_document, destination, payload):
    """Map repository-relative navigation to explicit archive destinations."""
    target, marker, anchor = destination.partition("#")
    suffix = marker+anchor
    if not target:
        return destination
    original = posixpath.normpath(posixpath.join(posixpath.dirname(source_document), target))
    mappings = (
        ("reports/coursework-complete-v2/", ""),
        ("reports/coursework-complete-v1/", ""),
        ("reports/framing-review-v1/", "archive/framing-review/"),
        ("reports/final-review-v1/", "archive/original-coursework/"),
        ("reports/categorical-reconstruction-v1/", "archive/categorical-review/"),
        ("reports/demo/categorical-field/", "demo/"),
        ("coursework_completion/results-v1/", "evidence/coursework-completion-v1/"),
        ("coursework_completion/audit-v1/", "evidence/coursework-completion-audit-v1/"),
        ("exploratory/framing_search/predictive/results-v1/", "evidence/framing-predictive-v1/"),
        ("exploratory/framing_search/predictive/audit-v1/", "evidence/framing-predictive-audit-v1/"),
        ("exploratory/framing_search/recording_audit/results-v1/", "evidence/recording-audit-v1/"),
        ("evidence/", "evidence/"),
    )
    candidates = ["code/"+original, original]
    for prefix, replacement in mappings:
        if original.startswith(prefix):
            candidates.insert(0, replacement+original[len(prefix):])
    match = next((candidate for candidate in candidates if candidate in payload), None)
    if match is None:
        return None
    return posixpath.relpath(match, posixpath.dirname("code/"+source_document))+suffix


def portable_document(raw, relative, payload):
    """Rewrite navigation copies, without altering sealed scientific sources.

    An intentionally unbundled source-checkout link becomes explicit plain text;
    it is never left as a broken archive hyperlink or silently misdirected.
    """
    text = raw.decode("utf-8")
    changes = []
    def replace(match):
        label, destination = match.group(1), match.group(2)
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", destination) or destination.startswith("#"):
            return match.group(0)
        mapped = resolve_packaged_target(relative, destination, payload)
        if mapped is None:
            replacement = f"{label} (source-checkout reference: `{destination}`)"
        else:
            replacement = f"[{label}]({mapped})"
        if replacement != match.group(0):
            changes.append({"from": destination, "to": mapped,
                            "status": "mapped" if mapped is not None else "unbundled reference labelled"})
        return replacement
    converted = re.sub(r"\[([^]\n]+)\]\(([^)\n]+)\)", replace, text)
    return converted.encode(), changes


def check_document_links(payload, documents):
    failures = []
    for name in documents:
        for target in re.findall(r"\]\(([^)\n]+)\)", payload[name].decode()):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = posixpath.normpath(posixpath.join(posixpath.dirname(name), target.partition("#")[0]))
            if path not in payload:
                failures.append([name, target])
    if failures:
        raise ValueError(f"Broken packaged navigation: {failures}")
    return {"documents": len(documents), "unresolved_internal_links": 0}


def merge_report(core, grouping, output, font_dir=None):
    """Five pages, explicitly distinguishing completion and original evidence."""
    import report
    from pypdf import PdfReader, PdfWriter
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    import matplotlib
    first, last = PdfReader(core), PdfReader(grouping)
    if len(first.pages) != 4 or len(last.pages) != 1:
        raise ValueError("Require cover, three task pages and one grouping appendix")
    regular, _ = report.find_fonts(font_dir)
    font = FontProperties(fname=str(regular), size=12)
    width, height = report.PAGE_WIDTH, report.PAGE_HEIGHT
    stages = ("Coursework review", "Frozen test + exploratory completion",
              "Original frozen evaluation", "Original frozen evaluation", "Exploratory nested TRAIN assessment")
    overlay = io.BytesIO()
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        with PdfPages(overlay) as pdf:
            for number, stage in enumerate(stages, 1):
                figure = Figure(figsize=(width/72, height/72))
                figure.patch.set_alpha(0)
                figure.add_artist(Rectangle((0, 0), 1, 42/height, transform=figure.transFigure,
                                            facecolor="white", edgecolor="none"))
                figure.text(40/width, 25/height, f"REVIEW DRAFT | {stage} | {number}/5", fontproperties=font)
                pdf.savefig(figure, transparent=True)
    overlay.seek(0)
    footer, writer = PdfReader(overlay), PdfWriter()
    for index, page in enumerate([*first.pages, *last.pages]):
        page.merge_page(footer.pages[index])
        writer.add_page(page)
    writer.add_metadata({"/Title": "Group 24: coursework completion review", "/Author": "Vlad George Iftode",
                         "/Creator": STUDY+"/build_review.py", "/CreationDate": "D:20000101000000Z"})
    with Path(output).open("xb") as stream:
        writer.write(stream)


def package_readme():
    return """# Group 24 coursework review

Start with [report.pdf](report.pdf): cover, three required task pages and one grouping appendix.
This remains a review draft while actual contributor details and individual peer feedback are deferred.
Nothing has been uploaded or submitted.

The original frozen test results are unchanged. Task 1 separately labels the later coursework
completion study on reused validation cohorts. The grouping appendix uses its own nested original-TRAIN
assessment. These stages cannot be compared as a single leaderboard or fresh confirmations.

Follow [REPRODUCE.md](code/REPRODUCE.md) for dependencies, data prerequisites and exact commands.
[Coverage](code/coursework_completion/COVERAGE.md) maps the lecture families and assignment tasks.
[Team review](code/coursework_completion/TEAM_REVIEW.md) records the human handoff.
Public aggregate evidence is under `evidence/`; executable sources are under `code/`.
The prior five-page report is preserved at [archive/framing-review/report.pdf](archive/framing-review/report.pdf),
and the original seven-page report at [archive/original-coursework/report.pdf](archive/original-coursework/report.pdf).

From `code/`, regenerate the current PDF without data, model fitting or any previous archive:

```sh
python coursework_completion/build_review.py --render-packaged .. --report-dir ../rerendered
```

This requires the pinned rendering dependencies and installed Times New Roman regular/bold fonts.
It verifies public evidence and all package hashes before rendering. Use a fresh output directory.
`PACKAGE-MANIFEST.json` binds every included artifact. Raw ratings, user histories, prediction arrays,
model checkpoints, external font files and the copyrighted course slides are deliberately excluded.
The separate `verification.json` supplied beside the ZIP records extraction, test and PDF-replay checks.
It is outside the ZIP so its archive hash does not create a circular dependency.
Older narratives are retained for provenance; their explicitly labelled source-checkout references
do not imply that omitted historical artifacts are included in this archive.
"""


def scientific_source_payload(payload, names, sources, root=ROOT):
    """Include declared files and their statically visible local Python imports."""
    pending, visited = list(names), set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        raw = read_safe(root, relative)
        key = "code/"+relative
        if key in payload and payload[key] != raw:
            raise ValueError(f"Previously packaged scientific source changed: {relative}")
        payload[key] = raw
        if relative.endswith(".py"):
            tree = ast.parse(raw, filename=relative)
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules.append(node.module)
                    modules.extend(node.module+"."+alias.name for alias in node.names if alias.name != "*")
            for module in modules:
                for candidate in (module.replace(".", "/")+".py", module.replace(".", "/")+"/__init__.py"):
                    if (root/candidate).is_file():
                        pending.append(candidate)
    for relative, expected in sources.items():
        if sha(payload["code/"+relative]) != expected:
            raise ValueError(f"Packaged scientific source hash differs: {relative}")
    return visited


def reproduction_payload(payload, root=ROOT):
    """Add only the declared recipe and aggregate-only replay receipts."""
    names = [STUDY+"/rebuild_recipe.json"]
    directory = root/STUDY/"reproduction"
    if directory.exists():
        names += [str(path.relative_to(root)) for path in sorted(directory.rglob("*.json"))]
    snapshots = {}
    for name in names:
        raw = read_safe(root, name)
        previous.aggregate_only(json.loads(raw))
        payload["code/"+name] = raw
        snapshots[name] = raw
    return snapshots


def operational_payload(payload, root=ROOT):
    """Copy the small public quick-check entry point and its dependency list."""
    names = ("operations/__init__.py", "operations/team_smoke_check.py", "requirements-smoke.txt")
    snapshot = {name: read_safe(root, name) for name in names}
    for name, raw in snapshot.items():
        payload["code/"+name] = raw
    return snapshot


def initial_payload(base):
    payload = {}
    for name, raw in base.items():
        if "/" in name:
            payload[name] = raw
        else:
            payload["archive/framing-review/"+name] = raw
    return payload


def add_navigation(payload, root=ROOT, sealed_sources=()):
    names = ["README.md", "HANDOFF.md", "PLAN.md", "REPRODUCE.md",
             "tests/README.md", "evidence/README.md", "reports/README.md", "packages/README.md",
             STUDY+"/README.md", STUDY+"/COVERAGE.md", STUDY+"/TEAM_REVIEW.md"]
    for directory in (root/"docs", root/"docs/team"):
        names.extend(str(path.relative_to(root)) for path in sorted(directory.glob("*.md")))
    names += [str(path.relative_to(root)) for path in (root/STUDY).rglob("*.md")
              if str(path.relative_to(root)) not in names and
              str(path.relative_to(root)) not in sealed_sources and
              (path.parent == root/STUDY or path.parent == root/STUDY/"legacy" or
               root/STUDY/"reproduction" in path.parents)]
    raw_documents = {name: read_safe(root, name) for name in names}
    if set(raw_documents) & set(sealed_sources):
        raise ValueError("Cannot rewrite a scientifically sealed navigation file")
    # Declare every destination before resolving links between the documents.
    for name, raw in raw_documents.items():
        key = "code/"+name
        if key in payload and payload[key] != raw:
            payload["archive/framing-navigation/"+name] = payload[key]
        payload[key] = raw
    changes = {}
    for name, raw in raw_documents.items():
        rendered, differences = portable_document(raw, name, payload)
        payload["code/"+name] = rendered
        changes[name] = {"source_sha256": sha(raw), "packaged_sha256": sha(rendered), "links": differences}
    return changes, ["code/"+name for name in names], raw_documents


def verify_extracted(payload, runtime_python, render_python):
    """Exercise all bundled test directories and rebuild the PDF without data."""
    directories = sorted({str(PurePosixPath(name).parent)[5:] for name in payload
        if name.startswith("code/") and PurePosixPath(name).name.startswith("test_") and name.endswith(".py")})
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    checks, logs = [], {}
    with tempfile.TemporaryDirectory(prefix="coursework-review-extracted-") as temporary:
        extracted = Path(temporary)
        for name, raw in payload.items():
            target = extracted/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        environment["PYTHONPATH"] = str(extracted/"code")
        commands = [[str(runtime_python), "-m", "unittest", "discover", "-s", directory, "-p", "test_*.py", "-v"]
                    for directory in directories]
        commands += [[str(render_python), "-m", "unittest", "discover", "-s", "tests", "-p", "test_report.py", "-v"],
                     [str(render_python), "-m", "unittest", "discover", "-s", STUDY, "-p", "test_build_review.py", "-v"],
                     [str(render_python), STUDY+"/build_review.py", "--render-packaged", "..", "--report-dir", "../rerendered"]]
        for index, command in enumerate(commands):
            result = subprocess.run(command, cwd=extracted/"code", env=environment,
                                    capture_output=True, text=True, timeout=300)
            log = f"extracted-check-{index}.log"
            logs[log] = (result.stdout+result.stderr).encode()
            checks.append({"command": command, "exit_code": result.returncode, "log": log})
            print(json.dumps({"check": index+1, "total": len(commands), "exit_code": result.returncode,
                              "command": command[1:]}), flush=True)
            if result.returncode:
                raise RuntimeError(f"Extracted verification failed: {command}\n{result.stdout}\n{result.stderr}")
        original = sha(payload["report.pdf"])
        rebuilt = digest(extracted/"rerendered/report.pdf")
        if original != rebuilt:
            raise ValueError("Extracted PDF replay is not byte-identical")
    return {"status": "complete", "checks": checks, "raw_data_available": False,
            "report_replay_byte_identical": True, "report_sha256": original,
            "limitations": "Uses installed local environments; no clean network installation or model training is claimed."}, logs


def render_content(core, grouping, out, font_dir=None):
    import report
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    counts = report.validate_content(core)
    core_layout = report.render_pdf(core, out/"main-report.pdf", font_dir)
    grouping_layout = previous.render_grouping(grouping, out/"grouping-supplement.pdf", font_dir)
    merge_report(out/"main-report.pdf", out/"grouping-supplement.pdf", out/"report.pdf", font_dir)
    (out/"main-report-content.json").write_bytes(json_bytes(core))
    (out/"grouping-content.json").write_bytes(json_bytes(grouping))
    (out/"report-content.json").write_bytes(json_bytes({"schema_version": 1, "status": "REVIEW DRAFT",
        "group": "24", "main": core, "grouping": grouping}))
    (out/"report.md").write_text(report.render_markdown(core)+"\n\n"+previous.markdown_grouping(grouping))
    return {"core": core_layout, "grouping": grouping_layout, "merged_pages": 5}, {
        **counts, "grouping": len(grouping["discussion"].split())}


def render_packaged(directory, out, font_dir=None):
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(read_safe(directory, "PACKAGE-MANIFEST.json"))
    if manifest["generator_sha256"] != digest(__file__):
        raise ValueError("Packaged report generator differs")
    for name, expected in manifest["artifact_sha256"].items():
        if sha(read_safe(directory, name)) != expected:
            raise ValueError(f"Packaged artifact changed: {name}")
    check_sources(manifest["source_sha256"], directory/"code")
    study = completion_evidence(directory/"evidence/coursework-completion-v1",
        directory/"evidence/coursework-completion-audit-v1", directory/"code")
    prior_core = json.loads(read_safe(directory, "archive/framing-review/main-report-content.json"))
    expected = build_main_content(prior_core, study)
    core = json.loads(read_safe(directory, "main-report-content.json"))
    if core != expected:
        raise ValueError("Completion report does not reproduce from audited aggregates")
    recording = previous.recording_evidence(directory/"evidence/recording-audit-v1", directory/"code")
    predictive = previous.predictive_evidence(directory/"evidence/framing-predictive-v1", directory/"code",
                                             directory/"evidence/framing-predictive-audit-v1")
    grouping = json.loads(read_safe(directory, "grouping-content.json"))
    if grouping != previous.build_grouping_content(recording, predictive):
        raise ValueError("Grouping appendix does not reproduce from public evidence")
    layout, counts = render_content(core, grouping, out, font_dir)
    result = {"status": "complete", "layout": layout, "discussion_words": counts,
        "source_report_sha256": manifest["artifact_sha256"]["report.pdf"],
        "replayed_report_sha256": digest(out/"report.pdf"), "dataset_access": False}
    (out/"replay.json").write_bytes(json_bytes(result))
    return result


def build(args):
    if args.out.name != "24.zip":
        raise ValueError("The coursework review archive must be named 24.zip")
    if args.out.exists() or args.out.parent.exists() or args.report_dir.exists():
        raise FileExistsError("Use new report and package directories")
    base = checked_base(args.base_archive)
    study = completion_evidence(args.evidence, args.audit)
    old_core = json.loads(base["main-report-content.json"])
    core = build_main_content(old_core, study)
    grouping = json.loads(base["grouping-content.json"])
    base_manifest = json.loads(base["PACKAGE-MANIFEST.json"])
    sources = {**base_manifest["source_sha256"], **study["sources"],
               STUDY+"/build_review.py": digest(__file__)}
    check_sources(sources)
    payload = initial_payload(base)
    names = {STUDY+"/"+path.name for path in (ROOT/STUDY).iterdir()
             if path.is_file() and path.suffix in {".py", ".md"}}
    names.update(str(path.relative_to(ROOT)) for path in (ROOT/STUDY/"legacy").rglob("*")
                 if path.is_file() and path.suffix in {".py", ".md"})
    names.update(str(path.relative_to(ROOT)) for path in (ROOT/STUDY/"reproduction").rglob("*")
                 if path.is_file() and path.suffix in {".py", ".md"})
    included_sources = scientific_source_payload(payload, names | set(sources), sources)
    source_snapshot = {name: read_safe(ROOT, name) for name in included_sources}
    reproduction_snapshot = reproduction_payload(payload)
    operational_snapshot = operational_payload(payload)
    for label, public in (("coursework-completion-v1", study["payload"]),
                           ("coursework-completion-audit-v1", study["audit_payload"])):
        for name, raw in public.items():
            payload["evidence/"+label+"/"+name] = raw
    layout, counts = render_content(core, grouping, args.report_dir, args.font_dir)
    for path in args.report_dir.iterdir():
        payload[path.name] = path.read_bytes()
    payload["README.md"] = package_readme().encode()
    navigation, documents, source_navigation = add_navigation(payload, sealed_sources=sources)
    links = check_document_links(payload, ["README.md", *documents])
    manifest = {"schema_version": 1, "status": "REVIEW DRAFT", "group": "24", "pages": 5,
        "contributors_supplied": len(core["members"]) == 5,
        "layout": layout, "discussion_words": counts,
        "evidence_stages": {"original": "previously frozen held-out test",
            "completion": "exploratory assessment on reused validation-calibration users",
            "grouping": grouping["evidence_stage"]},
        "base_archive_sha256": BASE_SHA256, "previous_report_sha256": BASE_REPORT_SHA256,
        "completion_evidence_sha256": {name: sha(raw) for name, raw in study["payload"].items()},
        "completion_audit_sha256": {name: sha(raw) for name, raw in study["audit_payload"].items()},
        "source_sha256": sources, "navigation_links": links,
        "artifact_sha256": {path.name: digest(path) for path in args.report_dir.iterdir()}}
    (args.report_dir/"REPORT-MANIFEST.json").write_bytes(json_bytes(manifest))
    payload["REPORT-MANIFEST.json"] = json_bytes(manifest)
    package = {"schema_version": 1, "status": "REVIEW DRAFT", "group": "24", "entrypoint": "report.pdf",
        "pages": 5, "base_archive_sha256": BASE_SHA256, "generator_sha256": digest(__file__),
        "source_sha256": sources, "navigation_copies": navigation,
        "original_report_sha256": previous.ORIGINAL_PDF_SHA256,
        "previous_report_sha256": BASE_REPORT_SHA256,
        "exclusions": ["raw ratings", "user/item identity lists", "prediction arrays", "model checkpoints",
                       "private run directories", "font binaries", "course slides"],
        "artifact_sha256": {name: sha(raw) for name, raw in sorted(payload.items())}}
    payload["PACKAGE-MANIFEST.json"] = json_bytes(package)
    verification, logs = verify_extracted(payload, args.runtime_python.absolute(), Path(sys.executable).absolute())
    check_sources(sources)
    for name, raw in {**source_snapshot, **reproduction_snapshot, **operational_snapshot}.items():
        if read_safe(ROOT, name) != raw:
            raise ValueError(f"Packaged source or reproduction receipt changed during verification: {name}")
    for name, raw in source_navigation.items():
        if read_safe(ROOT, name) != raw:
            raise ValueError("Navigation changed during package verification")
    if completion_evidence(args.evidence, args.audit)["payload"] != study["payload"]:
        raise ValueError("Completion evidence changed while building")
    previous.write_zip(args.out, payload)
    verification.update({"archive_sha256": digest(args.out), "entries": len(payload),
        "hashed_artifacts": len(package["artifact_sha256"]), "bytes": args.out.stat().st_size,
        "zip_crc_valid": True, "all_artifact_hashes_valid": True,
        "previous_report_unchanged": sha(payload["archive/framing-review/report.pdf"]) == BASE_REPORT_SHA256,
        "original_report_unchanged": sha(payload["archive/original-coursework/report.pdf"]) == previous.ORIGINAL_PDF_SHA256,
        "navigation_links": links, "status_of_submission": "Review draft; not submitted"})
    for name, raw in logs.items():
        (args.out.parent/name).write_bytes(raw)
    (args.out.parent/"verification.json").write_bytes(json_bytes(verification))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=ROOT/"packages/framing-review-v1/24.zip")
    parser.add_argument("--evidence", type=Path, default=ROOT/STUDY/"results-v1")
    parser.add_argument("--audit", type=Path, default=ROOT/STUDY/"audit-v1")
    parser.add_argument("--render-packaged", type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT/"reports/coursework-complete-v1")
    parser.add_argument("--out", type=Path, default=ROOT/"packages/coursework-complete-v1/24.zip")
    parser.add_argument("--runtime-python", type=Path, default=ROOT/"runs/environment-check/.venv/bin/python")
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    result = (render_packaged(args.render_packaged, args.report_dir, args.font_dir)
              if args.render_packaged else build(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
