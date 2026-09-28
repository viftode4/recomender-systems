"""Plot every predeclared seed's beam-minus-intercept NLL, without selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot(results: Path, out: Path):
    rows = json.loads(results.read_text())
    protocol = json.loads(results.with_name("protocol.json").read_text())
    SEEDS, SIZES, TASKS = tuple(protocol["seeds"]), tuple(protocol["sizes"]), tuple(protocol["tasks"])
    expected = {(task, size, seed) for task in TASKS for size in SIZES for seed in SEEDS}
    observed = {(r["task"], r["size"], r["seed"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("plot requires exactly all 90 predeclared cases")
    lookup = {(r["task"], r["size"], r["seed"]): r for r in rows}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=True, sharey=True)
    titles = {"clean_conjunction": "Clean conjunction", "noisy_conjunction": "Noisy conjunction",
              "correlated_distractors": "Correlated distractors", "contradictory_correlates": "Contradictory correlates",
              "xor_misspecified": "XOR: outside the grammar", "no_signal": "No predictive signal"}
    for axis, task in zip(axes.flat, TASKS):
        differences = np.array([[lookup[task, size, seed]["models"]["beam"]["log_loss"]
                                 - lookup[task, size, seed]["models"]["intercept"]["log_loss"]
                                 for seed in SEEDS] for size in SIZES])
        axis.axhline(0, color="#667085", linewidth=.8)
        jitter = np.linspace(-.12, .12, len(SEEDS))
        for index in range(len(SIZES)):
            axis.scatter(index + jitter, differences[index], color="#667085", s=23, alpha=.72,
                         linewidths=0, label="Individual seed" if index == 0 else None)
        axis.plot(np.arange(len(SIZES)), differences.mean(axis=1), "o-", color="#1763A6",
                  linewidth=2, markersize=5, label="Five-seed mean")
        axis.set_title(titles[task], loc="left", fontsize=11, weight="semibold")
        axis.set_xticks(np.arange(len(SIZES)), [str(n) for n in SIZES])
        axis.set_xlim(-.35, 2.35)
        axis.grid(axis="y", color="#EAECF0", linewidth=.6)
        axis.set_axisbelow(True)
    figure.suptitle("Program search can overfit short calibration sets", fontsize=16, weight="semibold", x=.08, ha="left")
    figure.supxlabel("FIT examples and disjoint PROBE examples, each", y=.075, fontsize=11)
    figure.supylabel("Evaluation NLL: program minus intercept\nPositive means program search is worse", x=.012, fontsize=11)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(.985, .97), ncol=1, frameon=False, fontsize=9)
    figure.text(.08, .018, "Each point is one predeclared seed; 5,000 fresh evaluation candidates per case. Supplied synthetic atoms; no recommendation claim.",
                fontsize=8.5, color="#475467")
    figure.subplots_adjust(left=.10, right=.98, top=.83, bottom=.16, hspace=.38, wspace=.16)
    figure.savefig(out.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.out)
