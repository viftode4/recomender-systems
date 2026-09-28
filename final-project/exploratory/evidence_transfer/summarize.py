"""Summarize the declared pilot decision from aggregate evidence only."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(path):
    data = json.loads(path.read_text())
    if data.get("fresh_confirmation") is not False or data.get("test_read") is not False:
        raise ValueError("Expected exploratory development evidence")
    seeds = sorted(data["seeds"])
    if seeds != ["2026", "2027", "2028"]:
        raise ValueError("Require all three declared seeds")
    roles = ("full_pattern", "no_pattern", "marginal_only", "analytic_donor",
             "full_zero_pattern", "locked_binary")
    metrics = ("ndcg@10", "recall@10", "mrr@10", "coverage@10")
    models = {}
    for role in roles:
        rows = [{**data["seeds"][seed]["models"],
                 **data["seeds"][seed]["references"]}[role] for seed in seeds]
        models[role] = {
            "all_observed_mean": {metric: mean(row["all_observed"][metric] for row in rows)
                                  for metric in metrics},
            "ndcg_by_seed": {seed: row["all_observed"]["ndcg@10"] for seed, row in zip(seeds, rows)},
            "tail_positive_pairs_pooled": sum(row["groups"]["item_groups"]["tail"]["heldout_positive_pairs"] for row in rows),
            "tail_hits_pooled": sum(row["groups"]["item_groups"]["tail"]["hits"] for row in rows),
            "tail_slot_fraction_mean": mean(row["groups"]["item_groups"]["tail"]["exposure"] for row in rows),
        }
    primary = lambda role: models[role]["all_observed_mean"]["ndcg@10"]
    gain = primary("full_pattern") / primary("locked_binary") - 1
    positive_all = all(models["full_pattern"]["ndcg_by_seed"][seed] >
                       models["locked_binary"]["ndcg_by_seed"][seed] for seed in seeds)
    pattern_delta = primary("full_pattern") - primary("no_pattern")
    return {
        "source_aggregates_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stage": "reused_development", "fresh_confirmation": False,
        "models": models,
        "declared_pilot_target": {
            "full_vs_locked_binary_relative_ndcg": gain,
            "positive_in_every_seed": positive_all,
            "full_minus_no_pattern_ndcg": pattern_delta,
            "passes_10percent_and_every_seed_and_pattern_mean": bool(gain >= .1 and positive_all and pattern_delta > 0),
            "interpretation": "A research prioritization target, not a statistical test or novelty criterion.",
        },
        "selected_epochs": {seed: {role: data["seeds"][seed]["selections"][role]["epoch"]
                                    for role in roles[:3]} for seed in seeds},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=root / "results-v1/aggregates.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = ["# Declared pilot comparison", "", "Reused development data; no fresh confirmation.", "",
            "| Model | nDCG@10 | Recall@10 | MRR@10 | Coverage@10 | Tail hits | Tail slots |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        values = model["all_observed_mean"]
        rows.append(f"| {name} | " + " | ".join(f"{values[key]:.6f}" for key in
                    ("ndcg@10", "recall@10", "mrr@10", "coverage@10")) +
                    f" | {model['tail_hits_pooled']}/{model['tail_positive_pairs_pooled']} | " +
                    f"{100 * model['tail_slot_fraction_mean']:.3f}% |")
    decision = report["declared_pilot_target"]
    rows += ["", f"Full-model relative nDCG difference versus locked binary: {100*decision['full_vs_locked_binary_relative_ndcg']:+.3f}%.",
             f"Full-model absolute nDCG difference versus no-pattern: {decision['full_minus_no_pattern_ndcg']:+.6f}.",
             f"Declared substantial-pilot target met: **{decision['passes_10percent_and_every_seed_and_pattern_mean']}**.", "",
             "Tail pairs are pooled over overlapping splits, not independent observations. Tail slots are unweighted recommendation positions. "
             "The full_zero_pattern row changes inputs to the selected full model without retraining. "
             "Full versus no_pattern changes two inputs jointly and does not identify their individual effects. "
             "A fixed training budget does not establish convergence.", ""]
    (args.out / "RESULTS.md").write_text("\n".join(rows))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir())}
    (args.out / "SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["declared_pilot_target"], indent=2))


if __name__ == "__main__":
    main()
