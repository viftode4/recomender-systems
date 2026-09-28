"""Render one measured research page from audited, aggregate-only evidence.

This renderer never fits a model, reads rating files or changes the frozen
coursework report. Actual rendering requires an evidence-bound completed audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
STUDY = "exploratory/categorical_reconstruction"
SEEDS = ("2026", "2027", "2028")
MODELS = ("binary_original_grid", "binary_expanded", "categorical", "shuffled_categories",
          "hybrid_baseline2", "hybrid_real3", "hybrid_shuffled3")
LABELS = {"binary_original_grid": "Binary: original grid", "binary_expanded": "Binary: expanded grid",
          "categorical": "Categorical", "shuffled_categories": "Shuffled categories",
          "hybrid_baseline2": "Hybrid: binary + SLIM", "hybrid_real3": "Hybrid + categorical",
          "hybrid_shuffled3": "Hybrid + shuffled", "EASE": "Locked EASE",
          "SLIMElastic": "Locked SLIM", "PositiveEASE": "Locked PositiveEASE"}
REFERENCES = ("EASE", "SLIMElastic", "PositiveEASE")
EVIDENCE_FILES = ("aggregates.json", "protocol.json", "provenance.json", "RESULTS.md", "SHA256.json")
WIDTH, HEIGHT, MARGIN, FONT_SIZE, LEADING = 595.2756, 841.8898, 40., 12., 13.8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_bytes(directory, name):
    directory = Path(directory).absolute()
    if directory.is_symlink() or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Unsafe source path")
    path = directory
    for part in Path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink source is excluded: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Required regular file is missing: {name}")
    return path.read_bytes()


def aggregate_only(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {"per_user", "recommendations", "histories", "user_ids", "item_ids", "matched_pairs_by_user",
                              "pairs_per_user", "cohort_ids"}:
                raise ValueError(f"Individual records are excluded: {location}/{key}")
            if normalized in {"selected_scores", "raw_scores", "predictions"} and isinstance(child, (list, dict)):
                raise ValueError(f"Individual arrays are excluded: {location}/{key}")
            if normalized in {"meta_fit", "development", "users", "fit_users", "selection_users",
                              "calibration_users"} and isinstance(child, (list, dict)):
                raise ValueError(f"Identifier collection is excluded: {location}/{key}")
            aggregate_only(child, location + "/" + key)
    elif isinstance(value, list):
        for child in value:
            aggregate_only(child, location)


def load_evidence(directory, audit_path, root=ROOT):
    directory, root = Path(directory), Path(root)
    payload = {name: safe_bytes(directory, name) for name in EVIDENCE_FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    seal = json.loads(payload["SHA256.json"])
    if set(seal) != set(EVIDENCE_FILES) - {"SHA256.json"}:
        raise ValueError("Evidence seal must bind exactly the four declared aggregate artifacts")
    if any(hashes[name] != expected for name, expected in seal.items()):
        raise ValueError("Curated evidence artifact hash mismatch")
    data = json.loads(payload["aggregates.json"])
    protocol = json.loads(payload["protocol.json"])
    provenance = json.loads(payload["provenance.json"])
    audit_bytes = safe_bytes(Path(audit_path).parent, Path(audit_path).name)
    audit = json.loads(audit_bytes)
    for name, value in (("aggregates", data), ("protocol", protocol), ("provenance", provenance), ("audit", audit)):
        aggregate_only(value, name)
    if (audit.get("status") != "complete" or audit.get("stage") != "reused_development"
            or audit.get("test_read") is not False or audit.get("test_evaluated") is not False
            or audit.get("fresh_confirmation") is not False
            or audit.get("unique_candidate_rows_verified") != 243
            or set(audit.get("seeds", {})) != set(SEEDS)):
        raise ValueError("Rendering requires the completed independent evidence audit")
    if any(audit.get("artifact_sha256", {}).get(name) != hashes[name]
           for name in ("aggregates.json", "protocol.json", "provenance.json")):
        raise ValueError("Audit receipt is not bound to these exact aggregate artifacts")
    if (audit.get("selection_freeze_sha256") != provenance.get("selection_freeze_sha256")
            or audit.get("completed_manifest_sha256") != provenance.get("manifest_sha256")
            or audit.get("source_sha256") != provenance.get("source_sha256")
            or audit.get("audit_source_sha256") != hashlib.sha256(
                safe_bytes(root, STUDY + "/verify_results.py")).hexdigest()):
        raise ValueError("Audit source, scientific source or selection-barrier bindings differ")
    for value in (data, protocol):
        if (value.get("stage") != "reused_development" or value.get("test_read") is not False
                or value.get("fresh_confirmation") is not False
                or value.get("dataset_test_previously_evaluated") is not True):
            raise ValueError("Evidence must disclose reused development and earlier project TEST exposure")
    if set(data.get("seeds", {})) != set(SEEDS) or tuple(map(str, protocol.get("seeds", []))) != SEEDS:
        raise ValueError("All three declared seeds are required")
    sources = provenance.get("source_sha256", {})
    if not sources or sources != protocol.get("source_sha256"):
        raise ValueError("Protocol and provenance source seals differ")
    for name, expected in sources.items():
        if hashlib.sha256(safe_bytes(root, name)).hexdigest() != expected:
            raise ValueError(f"Source differs from the measured study: {name}")
    for name in ("manifest_sha256", "selection_freeze_sha256", "reference_manifest_sha256", "reference_aggregate_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(provenance.get(name, ""))):
            raise ValueError(f"Missing provenance seal: {name}")
    for seed in SEEDS:
        row = data["seeds"][seed]
        if set(row.get("models", {})) != set(MODELS) or set(row.get("selections", {})) != set(MODELS):
            raise ValueError(f"Missing or unexpected model roles for seed {seed}")
        if not set(REFERENCES).issubset(row.get("references", {})):
            raise ValueError(f"Missing matched references for seed {seed}")
        if row.get("cohorts") != {"meta_fit": 471, "development": 472}:
            raise ValueError("Unexpected study cohorts; revise the declared report design explicitly")
        for name in MODELS:
            model = row["models"][name]
            if model["denominators"]["all_observed_users"] != 472:
                raise ValueError("All-observed model denominators differ")
            metric(model["all_observed"], "ndcg@10")
            groups = group_result(model)
            if set(groups["user_groups"]) != {"sparse", "medium", "dense"} or set(groups["item_groups"]) != {"head", "tail"}:
                raise ValueError("Required user/item group analysis is incomplete")
            if sum(g["users"] for g in groups["user_groups"].values()) != 472:
                raise ValueError("Activity group denominators do not sum to the development cohort")
    return {"data": data, "protocol": protocol, "provenance": provenance, "audit": audit,
            "artifact_sha256": hashes, "audit_sha256": hashlib.sha256(audit_bytes).hexdigest()}


def group_result(model):
    keys = [key for key in ("groups", "group_analysis") if key in model]
    if len(keys) != 1:
        raise ValueError("One explicit aggregate group-analysis result is required per model")
    return model[keys[0]]


def metric(values, key):
    result = values[key]
    if result is None:
        return None
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError(f"Invalid aggregate metric: {key}")
    return float(result)


def mean(values):
    values = list(values)
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def show(value, places=4):
    return "n/a" if value is None else f"{value:.{places}f}"


def model_metric(evidence, role, endpoint, key):
    values = []
    for seed in SEEDS:
        row = evidence["data"]["seeds"][seed]
        metrics = (row["models"][role][endpoint] if role in MODELS else
                   row["references"][role]["endpoints"][endpoint]["metrics"])
        values.append(None if metrics is None else metric(metrics, key))
    return mean(values)


def group_metric(evidence, role, kind, group, key):
    return mean(metric(group_result(evidence["data"]["seeds"][seed]["models"][role])[kind][group], key)
                for seed in SEEDS)


def build_content(evidence, group, members):
    if str(group) != "24" or not members or any(not str(member).strip() for member in members):
        raise ValueError("Use verified group 24 and only actual supplied contributor names")
    if len(set(members)) != len(members) or len(members) > 5:
        raise ValueError("Contributor names must be distinct, with at most five")
    values = {role: {endpoint: model_metric(evidence, role, endpoint, "ndcg@10")
                     for endpoint in ("all_observed", "liked_ratings")}
              for role in MODELS + REFERENCES}
    table_rows = [[LABELS[role], show(values[role]["all_observed"]), show(values[role]["liked_ratings"]),
                   show(100 * model_metric(evidence, role, "all_observed", "coverage@10"), 1)]
                  for role in MODELS + REFERENCES]
    roles = ("binary_expanded", "categorical", "hybrid_baseline2", "hybrid_real3")
    group_rows = []
    for name in ("sparse", "medium", "dense"):
        group_rows.append([name.capitalize() + " nD/div"] + [
            show(group_metric(evidence, role, "user_groups", name, "ndcg@10"), 3) + "/" +
            show(group_metric(evidence, role, "user_groups", name, "diversity"), 3) for role in roles])
    for name in ("head", "tail"):
        group_rows.append([name.capitalize() + " R/exp"] + [
            show(group_metric(evidence, role, "item_groups", name, "recall@10"), 3) + "/" +
            show(group_metric(evidence, role, "item_groups", name, "exposure"), 3) for role in roles])
    seeds = evidence["data"]["seeds"]
    denominators = {kind: {name: [group_result(seeds[seed]["models"]["binary_expanded"])[kind][name][field]
                                 for seed in SEEDS] for name in names}
                    for kind, names, field in (("user_groups", ("sparse", "medium", "dense"), "users"),
                                                ("item_groups", ("head", "tail"), "users_with_positives"))}
    group_counts = "; ".join(name + " " + "/".join(map(str, denominators["user_groups"][name]))
                              for name in ("sparse", "medium", "dense"))
    item_counts = "; ".join(name + " " + "/".join(map(str, denominators["item_groups"][name]))
                             for name in ("head", "tail"))
    likes = "/".join(str(seeds[seed]["models"]["binary_expanded"]["denominators"]["liked_ratings_users"])
                     for seed in SEEDS)
    fallback = sum(seeds[seed]["selections"]["categorical"]["category_ratio"] is None for seed in SEEDS)
    cat_delta = values["categorical"]["all_observed"] - values["binary_expanded"]["all_observed"]
    shuffle_delta = values["categorical"]["all_observed"] - values["shuffled_categories"]["all_observed"]
    hybrid_delta = values["hybrid_real3"]["all_observed"] - values["hybrid_baseline2"]["all_observed"]
    hybrid_shuffle = values["hybrid_shuffled3"]["all_observed"] - values["hybrid_baseline2"]["all_observed"]
    binary_mrr = model_metric(evidence, "binary_expanded", "all_observed", "mrr@10")
    category_mrr = model_metric(evidence, "categorical", "all_observed", "mrr@10")
    discussion = (
        f"The standalone nDCG gain over expanded binary is small ({cat_delta:+.4f}); "
        f"real minus shuffled categories is {shuffle_delta:+.4f}. "
        f"MRR declines from {binary_mrr:.4f} to {category_mrr:.4f}. "
        f"The real-category hybrid gain is negligible ({hybrid_delta:+.5f}); "
        f"shuffled augmentation gives {hybrid_shuffle:+.5f}. "
        "Real categories do not consistently beat the shuffled hybrid across splits. "
        f"Real categories select binary fallback in {fallback}/3 splits; an added duplicate expert can change regularization. "
        "The 45-choice categorical searches have more tuning opportunities than the nine-choice binary search. "
        "Self-item exclusion prevents copying the target; itemwise shuffling preserves item rating histograms. "
        "Group results describe accuracy and exposure, not causal fairness. "
        "All choices were sealed before this evaluation, but validation reuse after earlier TEST exposure precludes fresh confirmation."
    )
    content = {"schema_version": 1, "status": "REVIEW DRAFT", "group": str(group), "members": list(members),
        "title": "Task 2 supplement: categorical reconstruction",
        "scope": "Post-test exploratory research; reused development only",
        "method": "One ridge model reconstructs binary record presence from binary history and five TRAIN-centered rating channels; all six target-item channels are excluded. Binary-only is EASE. Hybrids use calibrated sum-to-one ridge; weights may be negative.",
        "evaluation": f"Equal means of three overlapping splits; 471 meta-fit / 472 development users each. All ratings are relevant for All; rating >=4 defines Liked ({likes} users). Full-catalog top-10 excludes TRAIN. References retain their original tuning; no score probabilities are claimed.",
        "tables": [{"caption": "Task 2.4/2.6: standalone and hybrid controls",
                    "columns": ["Model", "All nDCG", "Liked nDCG", "Coverage %"],
                    "rows": table_rows, "widths": [.46, .18, .18, .18]},
                   {"caption": "Task 2.5: TRAIN activity / item popularity groups",
                    "columns": ["Group / metric", "Binary", "Category", "Hybrid 2", "Hybrid 3"],
                    "rows": group_rows, "widths": [.28, .18, .18, .18, .18]}],
        "group_note": "nD/div: nDCG and genre-Jaccard diversity. R/exp: conditional recall and exposure share. TRAIN activity terciles; head = top 20% by frequency. User counts (seed order): " + group_counts + ". Item-group positive users: " + item_counts + ".",
        "group_denominators": denominators,
        "discussion": discussion,
        "source_note": "Independent implementation; EASE/FEASE and categorical-feedback prior art are documented in RELATED_WORK.md. Audit and source fingerprints accompany the report.",
        "evidence_sha256": evidence["artifact_sha256"], "audit_sha256": evidence["audit_sha256"],
        "source_sha256": evidence["provenance"]["source_sha256"],
        "selection_freeze_sha256": evidence["provenance"]["selection_freeze_sha256"]}
    if len(discussion.split()) > 200:
        raise ValueError("Discussion exceeds 200 words")
    return content


def find_fonts(directory=None):
    paths = [Path(directory)] if directory else [Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts"), Path("C:/Windows/Fonts")]
    for path in paths:
        for regular, bold in (("Times New Roman.ttf", "Times New Roman Bold.ttf"), ("times.ttf", "timesbd.ttf")):
            if (path / regular).is_file() and (path / bold).is_file():
                return path / regular, path / bold
    raise FileNotFoundError("Times New Roman regular and bold are required; use --font-dir")


def render_pdf(content, output, font_dir=None):
    regular, bold = find_fonts(font_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recsys-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    fonts = {False: FontProperties(fname=str(regular), size=FONT_SIZE),
             True: FontProperties(fname=str(bold), size=FONT_SIZE)}
    measure = TextToPath()
    figure = Figure(figsize=(WIDTH / 72, HEIGHT / 72))
    y = HEIGHT - MARGIN

    def wrap(text, width, strong=False):
        result, line = [], ""
        for word in text.split():
            if measure.get_text_width_height_descent(word, fonts[strong], False)[0] > width:
                raise ValueError(f"Unbreakable text exceeds column width: {word}")
            candidate = f"{line} {word}".strip()
            if line and measure.get_text_width_height_descent(candidate, fonts[strong], False)[0] > width:
                result.append(line)
                line = word
            else:
                line = candidate
        return result + ([line] if line else [])

    def draw(text, x=MARGIN, strong=False):
        if y < MARGIN + LEADING:
            raise ValueError("One-page supplement overflows at 12pt; shorten content")
        figure.text(x / WIDTH, y / HEIGHT, text, fontproperties=fonts[strong], va="top", parse_math=False)

    def paragraph(text, strong=False, gap=3):
        nonlocal y
        for line in wrap(text, WIDTH - 2 * MARGIN, strong):
            draw(line, strong=strong)
            y -= LEADING
        y -= gap

    paragraph(content["title"], True)
    paragraph(content["status"] + " | " + content["scope"])
    paragraph("Group " + content["group"] + " | Supplied contributor: " + "; ".join(content["members"]))
    paragraph(content["method"])
    paragraph(content["evaluation"])
    for table in content["tables"]:
        paragraph(table["caption"], True)
        for index, row in enumerate([table["columns"], *table["rows"]]):
            cells = [wrap(cell, (WIDTH - 2 * MARGIN) * share - 6, index == 0)
                     for cell, share in zip(row, table["widths"])]
            start, x = y, MARGIN
            height = max(map(len, cells)) * LEADING + 2
            if y - height < MARGIN + LEADING:
                raise ValueError("Table overflows the one-page research supplement")
            for lines, share in zip(cells, table["widths"]):
                for offset, line in enumerate(lines):
                    y = start - offset * LEADING
                    draw(line, x, index == 0)
                x += (WIDTH - 2 * MARGIN) * share
            y = start - height
        y -= 3
    paragraph(content["group_note"])
    paragraph("Discussion", True)
    paragraph(content["discussion"])
    paragraph(content["source_note"])
    figure.text(MARGIN / WIDTH, 20 / HEIGHT, "REVIEW DRAFT | Exploratory development, not fresh confirmation | 1/1",
                fontproperties=fonts[False])
    metadata = {"Title": content["title"], "Author": "; ".join(content["members"]),
                "CreationDate": datetime(2000, 1, 1), "ModDate": datetime(2000, 1, 1),
                "Creator": "categorical_reconstruction/render_supplement.py"}
    with matplotlib.rc_context({"pdf.fonttype": 42, "pdf.compression": 6, "text.usetex": False}):
        with PdfPages(output, metadata=metadata) as pdf:
            pdf.savefig(figure)
    return {"pages": 1, "font": "Times New Roman", "font_size": 12, "line_spacing": 1.15,
            "bottom_pt": round(y, 2), "font_sha256": {"regular": digest(regular), "bold": digest(bold)}}


def markdown(content, evidence):
    lines = ["# " + content["status"] + ": " + content["title"], "", content["scope"], "",
             "Group " + content["group"] + "; supplied names: " + "; ".join(content["members"]), "",
             content["method"], "", content["evaluation"], ""]
    for table in content["tables"]:
        lines += ["## " + table["caption"], "", "| " + " | ".join(table["columns"]) + " |",
                  "| " + " | ".join(["---"] * len(table["columns"])) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in table["rows"]]
        lines.append("")
    lines += [content["group_note"], "", "## Discussion", "", content["discussion"], "", content["source_note"], "",
              "## Per-seed details and group denominators", ""]
    for seed in SEEDS:
        row = evidence["data"]["seeds"][seed]
        lines += ["### Seed " + seed, "", "| Model | Features | Penalty | Ratio | All nDCG | Liked nDCG |",
                  "|---|---|---:|---:|---:|---:|"]
        for role in MODELS:
            selected, model = row["selections"][role], row["models"][role]
            lines.append("| " + " | ".join([LABELS[role], str(selected["feature_kind"]), str(selected["penalty"]),
                str(selected.get("category_ratio")), show(metric(model["all_observed"], "ndcg@10")),
                show(None if model["liked_ratings"] is None else metric(model["liked_ratings"], "ndcg@10"))]) + " |")
        lines += ["", "| Model / group | Users / positive users | nDCG or recall | Diversity or exposure |",
                  "|---|---:|---:|---:|"]
        for role in MODELS:
            groups = group_result(row["models"][role])
            for name, values in groups["user_groups"].items():
                lines.append(f"| {LABELS[role]} / {name} | {values['users']} | {show(values['ndcg@10'])} | {show(values['diversity'])} |")
            for name, values in groups["item_groups"].items():
                lines.append(f"| {LABELS[role]} / {name} | {values['users_with_positives']} | {show(values['recall@10'])} | {show(values['exposure'])} |")
        lines.append("")
    lines += ["Full selection diagnostics, calibrated coefficients, all metric definitions and paired descriptive differences remain in the hash-verified aggregates.json; no user-level records are included.", ""]
    return "\n".join(lines)


def generate(evidence_dir, audit_path, out, group, members, font_dir=None, root=ROOT):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    evidence = load_evidence(evidence_dir, audit_path, root)
    content = build_content(evidence, group, members)
    out.mkdir(parents=True, exist_ok=False)
    layout = render_pdf(content, out / "research-supplement.pdf", font_dir)
    (out / "research-supplement.md").write_text(markdown(content, evidence), encoding="utf-8")
    (out / "report-content.json").write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "evidence-audit.json").write_bytes(safe_bytes(Path(audit_path).parent, Path(audit_path).name))
    manifest = {"schema_version": 1, "status": "REVIEW DRAFT", "group": str(group),
                "contributors_supplied": len(members) == 5, "evidence_stage": content["scope"],
                "layout": layout, "discussion_words": len(content["discussion"].split()),
                "evidence_sha256": evidence["artifact_sha256"], "audit_sha256": evidence["audit_sha256"],
                "source_sha256": content["source_sha256"], "generator_sha256": digest(__file__),
                "artifact_sha256": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "SUPPLEMENT-MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    result = generate(args.evidence, args.audit, args.out, args.group, args.member, args.font_dir)
    print(json.dumps({"status": result["status"], "layout": result["layout"],
                      "discussion_words": result["discussion_words"]}, indent=2))


if __name__ == "__main__":
    main()
