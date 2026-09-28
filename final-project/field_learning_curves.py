"""Plot the complete, prespecified joint-field checkpoint grid from sealed runs.

Reads aggregate meta-fit losses only. Does not load TEST or choose a new model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, seeds):
    result = {"stage": "meta_fit_validation", "test_read": False, "seeds": {}}
    expected_epochs = [10, 30, 60, 100, 200, 300, 400]
    for seed in seeds:
        folder = root / str(seed)
        manifest = json.loads((folder / "manifest.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        if (manifest.get("status") != "complete" or manifest.get("test_read") or
                manifest.get("test_evaluated") or protocol.get("checkpoints") != expected_epochs):
            raise ValueError("Require complete validation-only convergence runs")
        if digest(folder / "protocol.json") != manifest["protocol_sha256"]:
            raise ValueError("Protocol digest differs")
        variants = {}
        for variant in ("adaptive", "fixed_flow"):
            name = f"{variant}/selection-grid.json"
            if digest(folder / name) != manifest["output_sha256"][name]:
                raise ValueError("Checkpoint grid digest differs")
            grid = json.loads((folder / name).read_text())
            if ([row["epoch"] for row in grid] != expected_epochs or
                    any(not math.isfinite(row["meta_fit_macro_joint_nll"]) for row in grid)):
                raise ValueError("Checkpoint grid differs from the declared finite grid")
            selection_name = f"{variant}/selection.json"
            if digest(folder / selection_name) != manifest["output_sha256"][selection_name]:
                raise ValueError("Selection digest differs")
            selection = json.loads((folder / selection_name).read_text())
            minimum = min(grid, key=lambda row: row["meta_fit_macro_joint_nll"])
            if any(selection[key] != minimum[key] for key in minimum):
                raise ValueError("Stored selection differs from the declared minimum")
            variants[variant] = {"grid": grid, "selected_epoch": selection["epoch"]}
        result["seeds"][str(seed)] = {"variants": variants,
            "research_manifest_sha256": digest(folder / "manifest.json")}
    return result


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    (output / "aggregate.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    figure, axes = plt.subplots(1, len(data["seeds"]), figsize=(11.4, 4.2), sharey=True, squeeze=False)
    for axis, (seed, record) in zip(axes.flat, data["seeds"].items()):
        for variant, color, label in (("adaptive", "#176859", "Adaptive"),
                                      ("fixed_flow", "#b46726", "Fixed flow")):
            values = record["variants"][variant]
            grid = values["grid"]
            axis.plot([r["epoch"] for r in grid], [r["meta_fit_macro_joint_nll"] for r in grid],
                      marker="o", color=color, label=label, linewidth=1.6, markersize=4)
            selected = next(r for r in grid if r["epoch"] == values["selected_epoch"])
            axis.scatter([selected["epoch"]], [selected["meta_fit_macro_joint_nll"]],
                         s=100, marker="*", color=color, zorder=4)
        axis.axvline(100, linestyle="--", color="#8d9699", linewidth=.8)
        axis.set(title=f"Split seed {seed}", xlabel="Training epoch", xticks=[0, 100, 200, 300, 400])
        axis.grid(axis="y", color="#dddddd", linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[0].set_ylabel("Macro-user joint NLL (lower is better)")
    axes.flat[-1].legend(frameon=False, loc="upper right")
    figure.suptitle("Does the 100-epoch budget stop learning too early?", fontsize=14, y=.98)
    figure.text(.08, .035, "Meta-fit validation only. Dashed line: original budget. Stars: stored checkpoint choices.\n"
                "All seven checkpoints are shown; overlapping splits are not independent replications.", fontsize=9)
    figure.tight_layout(rect=(0, .13, 1, .93))
    figure.savefig(output / "learning-curves.png", dpi=180)
    figure.savefig(output / "learning-curves.pdf")
    plt.close(figure)
    (output / "RESULTS.md").write_text(
        "# Joint-field optimization-budget sensitivity\n\n"
        "![Complete meta-fit checkpoint curves](learning-curves.png)\n\n"
        "The figures show all seven predeclared checkpoints for both variants and all three splits. "
        "Stars mark the checkpoint already selected by the runner. These are validation selection "
        "losses, not held-out ranking scores or evidence of convergence at the final budget.\n")
    hashes = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error("Require distinct nonempty seeds")
    render(collect(args.research_root, args.seeds), args.out)


if __name__ == "__main__":
    main()
