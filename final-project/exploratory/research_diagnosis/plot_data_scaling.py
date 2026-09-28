"""Render the sealed donor study's aggregate scores; no individual data needed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=root / "donor-curve-v1/aggregates.json")
    parser.add_argument("--output", type=Path, default=root / "donor-curve.png")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    seeds = sorted(data["seeds"])
    sizes = sorted(map(int, data["seeds"][seeds[0]]))
    curves = [[data["seeds"][seed][str(size)]["ndcg_at_10"] for size in sizes]
              for seed in seeds]
    means = [sum(values) / len(values) for values in zip(*curves)]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#faf9f6")
    fig.subplots_adjust(left=.12, right=.96, top=.80, bottom=.29)
    ax.set_facecolor("#faf9f6")
    for seed, curve, color in zip(seeds, curves, ["#c98c70", "#7c9bb0", "#92aa83"]):
        ax.plot(sizes, curve, "o--", color=color, alpha=.8, lw=1.3, ms=4, label=seed)
    ax.plot(sizes, means, "o-", color="#163f48", lw=2.8, ms=6, label="Mean")
    for index, (size, mean) in enumerate(zip(sizes, means)):
        ax.annotate(f"{mean:.3f}", (size, mean), xytext=(0, -18 if index == 3 else 10),
                    textcoords="offset points", ha="center", color="#163f48", weight="bold")
    ax.set(xticks=sizes, ylim=(.17, .27),
           xlabel="Other users supplying training histories", ylabel="Development nDCG@10")
    ax.grid(axis="y", color="#dadbd6", lw=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncol=2, fontsize=9, frameon=False)
    fig.text(.12, .94, "More training histories helped in this controlled range",
             fontsize=13, weight="bold", color="#173b43")
    fig.text(.12, .885, "Each evaluated user's history stays fixed; query users never enter fitting.",
             fontsize=10, color="#505c60")
    fig.text(.12, .08,
             "Three reused development splits. Lines show individual splits, not confidence intervals.\n"
             "Reduced-data diagnostic: this is not an improvement over the full-data baseline.",
             fontsize=9, color="#505c60", linespacing=1.6)
    fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
