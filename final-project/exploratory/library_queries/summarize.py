"""Render completed finite-query aggregates; never imports the learner or oracle.

Produces a four-panel PNG/PDF, RESULTS.md and a provenance manifest. No fitting,
selection, confidence intervals or interaction with a task oracle occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform


BUDGETS = (0, 2, 4, 8, 12)
SUPPORTS = (2, 4)
REGIMES = ("shared", "no_sharing")
METHODS = ("entropy", "focused", "random", "local", "oracle")
LABELS = {
    "entropy": "Output entropy",
    "focused": "Operator information",
    "random": "Random queries",
    "local": "Local program search",
    "oracle": "True-table reference",
}
COLORS = {
    "entropy": "#1D4ED8", "focused": "#C2410C", "random": "#64748B",
    "local": "#7E22CE", "oracle": "#047857",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def validated_rows(rows):
    """Require the complete protocol grid, with finite bounded risks or explicit N/A."""
    required = {(r, s, b, m) for r in REGIMES for s in SUPPORTS
                for b in BUDGETS for m in METHODS}
    indexed = {}
    for row in rows:
        key = (row["regime"], row["support_size"], row["budget"], row["method"])
        if key not in required or key in indexed:
            raise ValueError(f"Unexpected or duplicate result cell: {key}")
        value = row["error"]
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)
                                  or value < 0 or value > 1):
            raise ValueError(f"Invalid bounded risk: {key}")
        indexed[key] = row
    if indexed.keys() != required:
        raise ValueError(f"Incomplete result grid: {len(required-indexed.keys())} missing cells")
    for regime in REGIMES:
        for support in SUPPORTS:
            for budget in BUDGETS:
                missing = [indexed[regime, support, budget, m]["error"] is None for m in METHODS]
                if any(missing) != all(missing):
                    raise ValueError("Methods must use a common transfer-availability set")
            for method in ("local", "oracle"):
                values = [indexed[regime, support, b, method]["error"] for b in BUDGETS]
                if any(v is not None for v in values) and (values[0] is None or any(
                        v is None or abs(v-values[0]) > 1e-12 for v in values)):
                    raise ValueError("Support-only references must not vary with acquisition budget")
    return indexed


def draw_plot(indexed, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    markers = {"entropy": "o", "focused": "D", "random": "s", "local": "^", "oracle": "v"}
    lines = {"entropy": "-", "focused": "-", "random": ":", "local": "--", "oracle": "-."}
    for i, regime in enumerate(REGIMES):
        for j, support in enumerate(SUPPORTS):
            axis = axes[i, j]
            for method in METHODS:
                values = [indexed[regime, support, b, method]["error"] for b in BUDGETS]
                axis.plot([b+3 for b in BUDGETS],
                          [float("nan") if v is None else v for v in values],
                          color=COLORS[method], linestyle=lines[method], marker=markers[method],
                          linewidth=1.8, markersize=4.5, label=LABELS[method])
            title = "Shared operator" if regime == "shared" else "No shared operator"
            axis.set_title(f"{title} · {support} support labels", loc="left", weight="semibold")
            axis.set_xticks([b+3 for b in BUDGETS])
            axis.set_xlim(2.5, 15.5)
            axis.set_ylim(0, 1.025)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(axis="y", color="#E2E8F0", linewidth=.65)
            axis.set_axisbelow(True)
            if not any(indexed[regime, support, b, "entropy"]["error"] is not None for b in BUDGETS):
                axis.text(.5, .5, "No available transfer functions", transform=axis.transAxes,
                          ha="center", va="center", color="#475569")
    figure.suptitle("Which questions help an operation transfer?", x=.085, y=.985,
                   ha="left", weight="bold", fontsize=17)
    figure.text(.085, .945, "Exact finite Boolean prototype · All five acquisition budgets · Lower error is better",
                fontsize=10.5, color="#475569")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(.072, .919),
                  ncol=3, frameon=False, fontsize=9)
    figure.supxlabel("Training oracle calls = 3 initial + acquired queries (support labels are additional)",
                     y=.105, fontsize=10.5)
    figure.supylabel("Transfer error; abstention after inconsistency receives loss 1", x=.013, fontsize=10.5)
    figure.text(.085, .049,
                "Means: support subsets → unique unseen functions → available seeds per true table → equal tables.\n"
                "Empty transfer universes are N/A. No-sharing cells reuse 10 training worlds. No confidence bands.\n"
                "A rise can reflect earlier detection of an inconsistent sharing assumption and abstention, not more confident errors.\n"
                "Oracle calls are matched; computation is measured separately. True-table reference has privileged information.",
                fontsize=8.0, color="#475569", linespacing=1.4)
    figure.subplots_adjust(left=.09, right=.975, top=.79, bottom=.18, hspace=.33, wspace=.13)
    figure.savefig(out/"query-transfer.png", dpi=180, facecolor="white")
    figure.savefig(out/"query-transfer.pdf", facecolor="white",
                   metadata={"Title": "Finite shared-operator query experiment",
                             "CreationDate": None, "ModDate": None})
    plt.close(figure)
    return matplotlib.__version__


def number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def result_table(indexed):
    lines = ["| Regime | Support | Total oracle calls | Entropy | Focused | Random | Local | True table | Focused − entropy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for regime in REGIMES:
        for support in SUPPORTS:
            for budget in BUDGETS:
                values = [indexed[regime, support, budget, m]["error"] for m in METHODS]
                difference = None if values[0] is None else values[1]-values[0]
                lines.append(f"| {regime} | {support} | {budget+3} | "
                             + " | ".join(number(v) for v in values + [difference]) + " |")
    return "\n".join(lines)


def render_markdown(indexed, coverage_text, failure_text, conditional_text, cost_text, source_manifest_hash):
    return f"""# Exact shared-operator query experiment

This is a completed finite Boolean experiment. It learns a four-bit operation
by exhaustive hypothesis selection in a supplied 18-program grammar. It does
not establish a new learning objective, general architecture or recommendation
result. The target-versus-nuisance information distinction is established in
[Sloman et al., UAI 2024](https://arxiv.org/abs/2310.14968).

![All regimes, support sizes and budgets](query-transfer.png)

[Standalone PDF figure](query-transfer.pdf)

## Every predeclared result

Values are transfer error, so lower is better. Negative focused-minus-entropy
values favor focused acquisition. The grid reports every budget and support size;
no favorable cell is selected as the result. Failed shared-model inference abstains
and receives loss **1**, including in the no-sharing regime.

{result_table(indexed)}

The local search uses support labels only and searches all tables/programs.
The true-table reference receives privileged information. It is a reference,
not a guaranteed performance bound under finite support and deterministic ties.

## Transfer coverage and weighting

{coverage_text}

Each risk averages unseen-input errors, then all support subsets, then unique
unseen semantic functions, then available seeds within each true transfer table,
then available tables equally. Empty semantic universes are N/A, not successes.
The no-sharing cells reuse only ten training worlds. Exhaustive support subsets
are correlated evaluation episodes, not independent experimental replications.
There are no confidence intervals or claims of statistical significance here.

## Inconsistency and abstention

{failure_text}

Inconsistent no-sharing worlds test the current model's inability to replace
its sharing assumption. This prototype does not learn when to share. Failures
are retained in the bounded-risk score; conditional metrics must be read together
with their coverage and failure denominator.

Earlier inconsistency detection can therefore increase the plotted loss by
triggering abstention. It does not, by itself, establish worse confident
predictions. Conversely, continued prediction under an inconsistent generative
assumption is not evidence that the sharing assumption is correct.

## Conditional prediction error and coverage

{conditional_text}

## Information and computation costs

{cost_text}

Acquisition budgets match oracle calls: three initial observations plus the
listed additional queries. Every transfer episode separately supplies two or
four support labels. Candidate scoring, entropy, posterior updates and transfer
search are additional computational work. Matching oracle calls does not match
total compute; the common prediction cache does not erase these differences.

## Scope and provenance

Transfer functions are absent from the three training-task truth vectors but
remain inside the same depth-at-most-two grammar. This is semantic transfer
within a supplied finite family, not greater-depth or cross-domain validation.
The predictor receives only training observations and each episode's support
labels; the assessor's remaining labels are used solely for these scores.
No MovieLens data or final assessment enters this renderer.

Completed experiment manifest SHA-256: `{source_manifest_hash}`.
The renderer's manifest records its sources, aggregate input hashes and output hashes.
"""


def same_number(first, second):
    return first is None and second is None or (
        first is not None and second is not None and abs(first-second) <= 1e-12)


def normalize_summary(summary):
    if (summary.get("status") != "complete" or summary.get("schema_version") != 1
            or tuple(summary.get("budgets", [])) != BUDGETS
            or tuple(summary.get("support_sizes", [])) != SUPPORTS
            or tuple(summary.get("seeds", [])) != tuple(range(4100, 4110))):
        raise ValueError("Summary does not match the completed version-1 protocol")
    arm_names = {"entropy": "entropy", "library_information": "focused", "random": "random"}
    metrics = {}
    required = {(r, a, b, s) for r in REGIMES for a in arm_names for b in BUDGETS for s in SUPPORTS}
    for row in summary["metrics"]:
        key = (row["regime"], row["arm"], row["extra_queries"], row["support_size"])
        if key not in required or key in metrics:
            raise ValueError(f"Duplicate or unexpected source metric: {key}")
        if row["cells"] != 160 or row["available_transfer_cells"] + row["empty_transfer_cells"] != 160:
            raise ValueError("Invalid transfer-cell denominator")
        if not 0 <= row["empty_joint_posterior_cells"] <= 160:
            raise ValueError("Invalid failed-cell count")
        metrics[key] = row
    if metrics.keys() != required:
        raise ValueError("Source summary is missing protocol cells")
    normalized = []
    for regime in REGIMES:
        anchor = metrics[regime, "entropy", 0, 2]
        for support in SUPPORTS:
            for budget in BUDGETS:
                reference = metrics[regime, "entropy", budget, support]
                for arm, label in arm_names.items():
                    row = metrics[regime, arm, budget, support]
                    for count in ("cells", "available_transfer_cells", "empty_transfer_cells"):
                        if row[count] != anchor[count]:
                            raise ValueError("Transfer availability changes between arms or budgets")
                    normalized.append({"regime": regime, "support_size": support, "budget": budget,
                                       "method": label, "error": row["controls"]["learned"]["bounded_loss"]["mean"]})
                    for control in ("all_tables", "oracle"):
                        value = row["controls"][control]["bounded_loss"]
                        base = reference["controls"][control]["bounded_loss"]
                        if not same_number(value["mean"], base["mean"]) or value["cells_per_table"] != base["cells_per_table"]:
                            raise ValueError("Support-only reference differs between acquisition arms")
                for control, label in (("all_tables", "local"), ("oracle", "oracle")):
                    normalized.append({"regime": regime, "support_size": support, "budget": budget,
                                       "method": label, "error": reference["controls"][control]["bounded_loss"]["mean"]})
    indexed = validated_rows(normalized)
    seen_contrasts = set()
    for row in summary["focused_minus_entropy"]:
        key = (row["regime"], row["support_size"], row["extra_queries"])
        if key in seen_contrasts or key not in {(r, s, b) for r in REGIMES for s in SUPPORTS for b in BUDGETS}:
            raise ValueError("Invalid paired-contrast grid")
        seen_contrasts.add(key)
        first = indexed[key + ("focused",)]["error"]
        second = indexed[key + ("entropy",)]["error"]
        expected = None if first is None else first-second
        if not same_number(row["mean"], expected):
            raise ValueError("Paired contrast disagrees with displayed bounded-risk means")
    if len(seen_contrasts) != len(REGIMES)*len(SUPPORTS)*len(BUDGETS):
        raise ValueError("Incomplete paired-contrast grid")
    return indexed, metrics


def semantic_coverage(records_path: Path, metrics):
    """Read counts only; no task outputs or histories are needed for this report."""
    cells = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["arm"] != "entropy" or row["extra_queries"] != 0:
                continue
            key = (row["regime"], row["seed"], row["true_table"])
            if key in cells:
                raise ValueError("Duplicate semantic-coverage cell")
            total, removed, remaining = (row[name] for name in (
                "all_unique_target_truth_functions", "excluded_training_truth_functions", "transfer_truth_functions"))
            if not all(type(v) is int and v >= 0 for v in (total, removed, remaining)) or total-removed != remaining:
                raise ValueError("Invalid semantic-universe counts")
            cells[key] = {"full": total, "removed": removed, "remaining": remaining}
    expected = {(r, s, t) for r in REGIMES for s in range(4100, 4110) for t in range(16)}
    if cells.keys() != expected:
        raise ValueError("Semantic coverage does not contain all320 world/table cells")
    lines = ["| Regime | Training worlds | Transfer cells | Available cells | Empty cells | Retained function instances | Support episodes: size2 / size4 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for regime in REGIMES:
        rows = [value for key, value in cells.items() if key[0] == regime]
        available = sum(row["remaining"] > 0 for row in rows)
        if available != metrics[regime, "entropy", 0, 2]["available_transfer_cells"]:
            raise ValueError("Summary coverage disagrees with record counts")
        retained = sum(row["remaining"] for row in rows)
        lines.append(f"| {regime} | {160 if regime == 'shared' else 10} | 160 | {available} | {160-available} | "
                     f"{retained} | {retained*28} / {retained*70} |")
    lines.extend(["", "Function instances count a function separately in each world/table cell. The episode counts",
                  "are per method and acquisition snapshot; they are not independent data points.", "",
                  "| True table index | Full unique-function universe | Shared available seeds /10 | Shared retained functions, sum over seeds | No-sharing available seeds /10 | No-sharing retained functions, sum over seeds |",
                  "|---|---:|---:|---:|---:|---:|"])
    for table in range(16):
        full = {cells[r, s, table]["full"] for r in REGIMES for s in range(4100, 4110)}
        if len(full) != 1:
            raise ValueError("A table's full semantic universe changes across worlds")
        row = [str(table), str(next(iter(full)))]
        for regime in REGIMES:
            group = [cells[regime, s, table]["remaining"] for s in range(4100, 4110)]
            row.extend([str(sum(value > 0 for value in group)), str(sum(group))])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def failure_summary(metrics):
    lines = ["Counts below are failed **training worlds**, including worlds without transfer targets.", "",
             "| Regime | Total oracle calls | Entropy failures | Focused failures | Random failures |",
             "|---|---:|---:|---:|---:|"]
    for regime in REGIMES:
        divisor, worlds = (1, 160) if regime == "shared" else (16, 10)
        for budget in BUDGETS:
            counts = []
            for arm in ("entropy", "library_information", "random"):
                failed = metrics[regime, arm, budget, 2]["empty_joint_posterior_cells"]
                if failed % divisor:
                    raise ValueError("Repeated no-sharing cells do not imply an integer failed-world count")
                if failed != metrics[regime, arm, budget, 4]["empty_joint_posterior_cells"]:
                    raise ValueError("Acquisition failure depends on transfer support size")
                row = metrics[regime, arm, budget, 2]
                if (row["failed_training_worlds"] != failed//divisor
                        or row["acquisition_training_worlds"] != worlds):
                    raise ValueError("Explicit failed-world counts disagree with transfer cells")
                counts.append(f"{failed//divisor}/{worlds}")
            lines.append(f"| {regime} | {budget+3} | " + " | ".join(counts) + " |")
    lines.extend(["", "The figure uses bounded error including abstentions. The next table separately shows",
                  "prediction error among non-abstaining cells and the retained prediction coverage."])
    return "\n".join(lines)


def conditional_summary(metrics):
    lines = ["Each policy entry is **conditional error / weighted coverage (predicting cells; tables)**.",
             "The cell count includes only available transfer universes and excludes abstentions.",
             "Coverage uses the same table weighting as bounded risk; it is not necessarily the raw",
             "fraction of cells. Conditional error averages available predicting seeds within each",
             "table and then available tables. Its cohort can differ across policies, so a lower",
             "conditional error alone does not establish a better policy. N/A means no predictions.", "",
             "| Regime | Support | Total oracle calls | Entropy: error / coverage (cells; tables) | Focused: error / coverage (cells; tables) | Random: error / coverage (cells; tables) |",
             "|---|---:|---:|---|---|---|"]
    for regime in REGIMES:
        for support in SUPPORTS:
            for budget in BUDGETS:
                columns = []
                for arm in ("entropy", "library_information", "random"):
                    row = metrics[regime, arm, budget, support]
                    controls = row["controls"]["learned"]
                    conditional = controls["conditional_error"]
                    coverage = controls["prediction_coverage"]
                    risk = controls["bounded_loss"]
                    count = sum(conditional["cells_per_table"].values())
                    if not 0 <= count <= row["available_transfer_cells"]:
                        raise ValueError("Invalid conditional prediction count")
                    if coverage["cells_per_table"] != risk["cells_per_table"]:
                        raise ValueError("Coverage and bounded loss have different availability")
                    for value in (conditional["mean"], coverage["mean"]):
                        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                            raise ValueError("Invalid conditional error or coverage")
                    if ((conditional["mean"] is None) != (count == 0)
                            or conditional["available_tables"] != len(conditional["cells_per_table"])):
                        raise ValueError("Conditional mean and denominator disagree")
                    weighted_coverage = "N/A" if coverage["mean"] is None else f"{coverage['mean']:.1%}"
                    columns.append(f"{number(conditional['mean'])} / {weighted_coverage} "
                                   f"({count}; {conditional['available_tables']})")
                lines.append(f"| {regime} | {support} | {budget+3} | " + " | ".join(columns) + " |")
    lines.extend(["", "The local and true-table controls always fit from the same support labels and do not",
                  "abstain: their conditional errors equal their bounded errors, with full prediction",
                  "coverage whenever the transfer universe is available. These are descriptive",
                  "conditional means, not comparisons restricted to a common non-abstaining cohort."])
    return "\n".join(lines)


def cost_summary(costs):
    precomputation = costs["common_precomputation"]
    if precomputation["candidate_output_evaluations"] != 2304 or precomputation["operator_calls"] != 3840:
        raise ValueError("Common cache differs from the declared finite grammar")
    lines = ["| Work category | Recorded amount |", "|---|---:|",
             f"| Common program-output evaluations, once per process | {precomputation['candidate_output_evaluations']:,} |",
             f"| Common primitive-operation calls, once per process | {precomputation['operator_calls']:,} |",
             f"| Actual acquisition runs | {costs['actual_acquisition_runs']:,} |"]
    for key, value in costs["acquisition_totals"].items():
        lines.append(f"| Acquisition: {key.replace('_', ' ')} | {value:,} |")
    for key, value in costs["transfer"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lines.append(f"| Transfer: {key.replace('_', ' ')} | {value:,} |")
    lines.append(f"| Experiment elapsed seconds, including core import | {costs['total_elapsed_seconds_including_core_import']:.3f} |")
    lines.extend(["", "Counts cover the complete experiment, including all acquisition policies and reused",
                  "transfer computations. Cached lookup counters are logical requested entries, not CPU",
                  "instructions. Wall time is observed, machine-dependent duration. These totals do not",
                  "establish which policy is fastest or an equal-total-compute comparison. Both informed",
                  "arms compute output, canonical conditional and concrete conditional entropy in the same",
                  "statistics routine. Transfer memoization is shared across policies, so the measured",
                  "runtime does not isolate either policy's intrinsic incremental computational cost."])
    return "\n".join(lines)


def load_completed(run: Path):
    manifest_path = run/"manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "complete" or manifest.get("schema_version") != 1
            or manifest.get("synthetic_finite_experiment") is not True
            or manifest.get("test_data_access") is not False):
        raise ValueError("Require a completed synthetic experiment with no real test access")
    inputs = {"manifest.json": sha256(manifest_path)}
    for name in ("summary.json", "records.jsonl"):
        path = run/name
        inputs[name] = sha256(path)
        if inputs[name] != manifest["output_sha256"].get(name):
            raise ValueError(f"Completed manifest does not verify {name}")
    summary = read_json(run/"summary.json")
    if summary["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Summary and manifest disagree on experiment source hashes")
    indexed, metrics = normalize_summary(summary)
    coverage = semantic_coverage(run/"records.jsonl", metrics)
    return indexed, metrics, summary, coverage, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Completed runner output directory")
    parser.add_argument("--out", required=True, type=Path, help="New report directory; never overwritten")
    args = parser.parse_args(argv)
    run, out = args.run.expanduser().resolve(), args.out.expanduser().resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite {out}")
    indexed, metrics, summary, coverage, inputs = load_completed(run)
    failure, costs = failure_summary(metrics), cost_summary(summary["costs"])
    conditional = conditional_summary(metrics)
    content = render_markdown(indexed, coverage, failure, conditional, costs, inputs["manifest.json"])
    out.mkdir(parents=True, exist_ok=False)
    renderer_hash = sha256(Path(__file__).resolve())
    report_manifest = {"status": "rendering", "created_at": datetime.now(timezone.utc).isoformat(),
                       "renderer_sha256": renderer_hash, "input_sha256": inputs,
                       "experiment_source_sha256": summary["source_sha256"],
                       "python": platform.python_version(), "fitting": False,
                       "oracle_access": False, "confidence_intervals": False}
    write_json(out/"manifest.json", report_manifest)
    try:
        plot_version = draw_plot(indexed, out)
        (out/"RESULTS.md").write_text(content, encoding="utf-8")
        if {name: sha256(run/name) for name in inputs} != inputs:
            raise RuntimeError("Experiment inputs changed during rendering")
        if sha256(Path(__file__).resolve()) != renderer_hash:
            raise RuntimeError("Renderer changed during execution")
        report_manifest.update(status="complete", matplotlib=plot_version,
                               output_sha256={name: sha256(out/name) for name in (
                                   "RESULTS.md", "query-transfer.png", "query-transfer.pdf")})
        write_json(out/"manifest.json", report_manifest)
    except BaseException as error:
        report_manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(out/"manifest.json", report_manifest)
        raise
    print(json.dumps({"status": "complete", "report": str(out/"RESULTS.md"),
                      "png": str(out/"query-transfer.png"), "pdf": str(out/"query-transfer.pdf")}, sort_keys=True))


if __name__ == "__main__":
    main()
