"""Render a checked, task-structured development report from aggregate evidence.

Requires Matplotlib and Times New Roman fonts. Reports default to DRAFT;
--finalize requires a completed frozen test evaluation and actual group details.
The JSON content file can be edited and supplied with --content-json; each of the
three sections contains task, title, paragraphs, tables and discussion fields.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile


FONT_SIZE = 12
LINE_SPACING = 1.15
LEADING = FONT_SIZE * LINE_SPACING
PAGE_WIDTH, PAGE_HEIGHT = 595.276, 841.89  # A4, in points
MARGIN = 48
FAMILIES = ["expert", "static", "user", "item", "disagreement", "context",
            "static-pairwise", "context-pairwise"]
REFERENCES = [
    "[1] Steck (2019). Embarrassingly Shallow Autoencoders for Sparse Data. "
    "WWW. https://arxiv.org/abs/1905.03375",
    "[2] Rendle et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. "
    "UAI. https://arxiv.org/abs/1205.2618",
    "[3] Steck (2018). Calibrated Recommendations. RecSys. "
    "doi:10.1145/3240323.3240372. Our JSD implementation is an adaptation, not an exact reproduction.",
    "[4] Ning and Karypis (2011). SLIM: Sparse Linear Methods for Top-N Recommender Systems. "
    "ICDM. doi:10.1109/ICDM.2011.134.",
    "[5] Kabbur, Ning and Karypis (2013). FISM: Factored Item Similarity Models for Top-N "
    "Recommender Systems. KDD. doi:10.1145/2487575.2487589.",
    "[6] He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for "
    "Recommendation. https://arxiv.org/abs/2002.02126",
    "[7] He et al. (2017). Neural Collaborative Filtering. https://arxiv.org/abs/1708.05031",
    "[8] Wang et al. (2019). Neural Graph Collaborative Filtering. https://arxiv.org/abs/1905.08108",
    "[9] Aiolli (2013). Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets. "
    "RecSys. doi:10.1145/2507157.2507189. Cosine KNN uses the course adapter, not all paper variants.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def word_count(text):
    """Whitespace-separated words; conservative for mathematical expressions."""
    return len(text.split())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_table(caption, entries, value_heading):
    """Display a long model list in two equal column pairs at the same font size."""
    midpoint = (len(entries) + 1) // 2
    rows = []
    for i in range(midpoint):
        rows.append(entries[i] + (entries[i + midpoint] if i + midpoint < len(entries) else ["", ""]))
    return {"caption": caption, "columns": ["Model", value_heading, "Model", value_heading],
            "widths": [0.38, 0.12, 0.38, 0.12], "rows": rows}


def build_content(evidence, group=None, members=()):
    """Construct a report from aggregate, selected-model and source-provenance files."""
    evidence = Path(evidence)
    selected = read_json(evidence / "selected.json")
    aggregates = read_json(evidence / "aggregates.json")
    budgets = read_json(evidence / "candidate-budgets.json")
    policies = read_json(evidence / "group-policies.json")
    provenance = read_json(evidence / "provenance.json")
    seeds = sorted(selected)
    if not seeds or set(seeds) != set(aggregates):
        raise ValueError("Selected and aggregate evidence must have matching, nonempty seeds")

    def avg(family, metric):
        return statistics.mean(selected[s][family][metric] for s in seeds)

    def aggregate_avg(name, metric):
        return statistics.mean(aggregates[s][name(s)]["aggregate"][metric] for s in seeds)

    source_names = {}
    for seed in seeds:
        source_names[seed] = {}
        for name, source in provenance[seed]["manifest"]["sources"].items():
            model = source["manifest"]["model"]
            if name not in aggregates[seed]:
                continue
            if model in source_names[seed]:
                raise ValueError(f"Multiple selected source runs for model {model} in seed {seed}")
            source_names[seed][model] = name
    model_names = sorted(set.union(*(set(v) for v in source_names.values())))
    selected_rows = []
    for model in model_names:
        values = [aggregates[s][source_names[s][model]]["aggregate"]["ndcg@10"]
                  for s in seeds if model in source_names[s]]
        if len(values) != len(seeds):
            raise ValueError(f"Model {model} is missing from one or more seeds; compare matched evidence")
        selected_rows.append([model, f"{statistics.mean(values):.4f}"])
    family_names = [f for f in selected[seeds[0]] if f != "expert"]
    selected_rows += [[f, f"{avg(f, 'ndcg@10'):.4f}"] for f in family_names]
    selected_rows += [[name, f"{aggregate_avg(lambda s, n=name: n, 'ndcg@10'):.4f}"]
                      for name in ["rrf", "group-switch"]]
    metric_rows = []
    for label, key in [("Random", lambda s: source_names[s]["Random"]),
                       ("Exact popularity", lambda s: source_names[s]["ExactPop"]),
                       ("RRF", lambda s: "rrf"),
                       ("Group switching", lambda s: "group-switch")]:
        metric_rows.append([label] + [f"{aggregate_avg(key, m):.4f}" for m in
                                     ["ndcg@10", "recall@10", "novelty@10"]])
    coefficient_rows = []
    for expert in [m for m in model_names if m != "Random"]:
        weights = []
        for seed in seeds:
            coefficients = read_json(evidence / f"{seed}-coefficients.json")
            chosen = coefficients[selected[seed]["static"]["name"]]["weights"]
            source_name = source_names[seed][expert]
            if source_name not in chosen:
                raise ValueError(f"Cannot identify {expert} static coefficient for {seed}")
            weights.append(chosen[source_name])
        coefficient_rows.append([expert, f"{statistics.mean(weights):+.4f}"])

    reranker_rows = []
    has_popularity = all("popularity_jsd" in selected[s]["context"] for s in seeds)
    rerank_metrics = ["ndcg@10", "diversity", "calibration_jsd", "head_exposure"] + (["popularity_jsd"] if has_popularity else [])
    for suffix, label in [("", "Base context"), ("/diversity-0.5", "Diversity"),
                          ("/calibration-0.5", "Calibration"),
                          ("/exposure-0.5", "Item exposure"),
                          ("/popularity_calibration-0.5", "Popularity calibration")]:
        key = lambda s, suffix=suffix: selected[s]["context"]["name"] + suffix
        if all(key(s) in aggregates[s] for s in seeds):
            reranker_rows.append([label] + [f"{aggregate_avg(key, m):.3f}" for m in rerank_metrics])
    ordering_rows = []
    for method, goal in [("diversity", "diversity"), ("calibration", "calibration_jsd"),
                         ("exposure", "head_exposure"), ("popularity_calibration", "popularity_jsd")]:
        before = lambda s, m=method: "rerank-experts-then-rrf/" + m
        after = lambda s, m=method: "rrf-then-rerank/" + m
        if not all(before(s) in aggregates[s] and after(s) in aggregates[s] for s in seeds):
            continue
        ordering_rows.append([method, f"{aggregate_avg(before, 'ndcg@10'):.4f}",
                              f"{aggregate_avg(after, 'ndcg@10'):.4f}",
                              f"{aggregate_avg(before, goal):.3f}", f"{aggregate_avg(after, goal):.3f}"])
    group_rows = []
    for group_id, label in [("0", "Sparse users"), ("1", "Medium users"), ("2", "Dense users")]:
        groups = [aggregates[s][selected[s]["context"]["name"]]["groups"][group_id] for s in seeds]
        group_rows.append([label] + [f"{statistics.mean(g[m] for g in groups):.3f}" for m in
                                    ["ndcg@10", "recall@10", "diversity"]] + ["n/a"])
    for label, metric, exposure in [("Head items", "head_recall", avg("context", "head_exposure")),
                                    ("Tail items", "tail_recall", 1 - avg("context", "head_exposure"))]:
        group_rows.append([label, "n/a", f"{avg('context', metric):.3f}", "n/a", f"{exposure:.3f}"])
    sparse_retention = [policies[s]["development_retention"]["0"] for s in seeds]
    independent_policy = all(policies[s].get("independent_calibration") is not None for s in seeds)
    counts = {key: "/".join(map(str, sorted({provenance[s]["manifest"].get(key, 0) for s in seeds})))
              for key in ["meta_fit_users", "development_users", "calibration_users"]}
    policy_fit = "independent calibration" if independent_policy else "meta-fit"
    policy_description = ("with approximate multiplicity-adjusted lower confidence bounds on retention"
                          if independent_policy else "by empirical mean retention")
    retention_text = (f"Sparse-group development retention was {sparse_retention[0]:.1%}. " if len(seeds) == 1 else
                      f"Sparse-group development retention ranged from {min(sparse_retention):.1%} to {max(sparse_retention):.1%}. ")
    retention_text += ("The fitted constraint failed this development audit. " if min(sparse_retention) < .95 else
                       "The observed audit meets the target but is not a future guarantee. ")
    if all(all(float(a) == 0 for a in policies[s]["strengths"].values()) for s in seeds):
        retention_text += "The conservative policy selected no exposure correction for any group. "
    exposure_note = (f"UPD is per-user popularity-profile JSD (lower is closer). Discounted catalog exposure "
                     f"Gini changes from {avg('context', 'exposure_gini'):.3f} to "
                     f"{aggregate_avg(lambda s: selected[s]['context']['name'] + '/exposure-0.5', 'exposure_gini'):.3f} "
                     "under item-exposure reranking; lower Gini means less concentrated allocation."
                     if has_popularity and all("exposure_gini" in selected[s]["context"] for s in seeds) else "")
    mean_pool = statistics.mean(budgets[s]["mean_pool"] for s in seeds)
    agreement = min(budgets[s]["same_topk_as_full"] for s in seeds)
    sources = {p.name: digest(p) for p in [evidence / "selected.json", evidence / "aggregates.json",
               evidence / "candidate-budgets.json", evidence / "group-policies.json", evidence / "provenance.json"] +
               [evidence / f"{s}-coefficients.json" for s in seeds]}
    return {
        "schema_version": 1, "status": "DRAFT", "group": group, "members": list(members),
        "title": "Hybrid recommendation: usefulness, exceptions and exposure",
        "evidence_label": ("Single-seed development preview; final test evaluation outstanding" if len(seeds) == 1 else
                           "Selected development evidence; final test evaluation outstanding"),
        "source_sha256": sources,
        "references": REFERENCES if "FISMCorrected" in model_names else REFERENCES[:3] + [REFERENCES[8]],
        "cover_notes": [
            "DSAIT4335 Recommender Systems. All reported values are development results, "
            "not final test performance or evidence of state-of-the-art performance.",
            "The assignment specifies one page and at most 200 discussion words per task. "
            "This draft conservatively allocates one page to each major task (1, 2 and 3).",
            "Development evidence is exploratory. Freeze the reviewed model choices before evaluating "
            "held-out tests, and retain the recorded protocol and source hashes with the results.",
            "Group number and actual contributors must be supplied before submission. "
            "AI assistance was used; members must review the implementation and record actual contributions.",
        ],
        "sections": [
            {"task": 1, "title": "Task 1: individual models and hybrids", "paragraphs": [
                f"MovieLens 100K; seeds {', '.join(seeds)}. Every observed rating is an implicit positive. "
                "RecBole uses per-user random 80/10/10 splits and full-catalog ranking. Previously seen items "
                f"are masked. Experts train for fixed budgets. Validation users split into {counts['meta_fit_users']} meta-fit and "
                f"{counts['development_users']} development users; {counts['calibration_users']} users are "
                "reserved for policy calibration. This is not a cold-start experiment.",
                "Meta-fit selects expert settings from predeclared grids. EASE [1] and BPR [2] anchor "
                "the comparison. Standardized expert scores feed ridge regression, with five sampled "
                "unobserved negatives per positive. User, item and disagreement features are ablated. "
                "Hybrid penalties (0.001/0.01/0.1/1) are selected on development users. " +
                ("Explicit-rating experts preserve dislikes; their rating-information controls are included."
                 if "ContrastTransfer" in model_names else "Random and exact popularity provide naive controls."),
                *(["Constrained weights sum to one but can be negative. The unscaled version operates on z-scores; "
                   "the calibrated version first learns per-expert affine response alignment on meta-fit users. "
                   "Effective raw-score coefficients need not sum to one. Outputs remain unbounded ranking scores."]
                  if "calibrated" in family_names else
                  ["The constrained regression hybrid enforces weights summing to one; individual weights "
                   "may remain negative. Unconstrained regression is retained as a control."]
                  if "constrained" in family_names else []),
                *(["Class models follow [4-9]. FISMCorrected fixes the sigmoid/BCEWithLogitsLoss mismatch "
                   "and removes each scored item from its own history and normalization. The supplied "
                   "non-squared norm regularizer is retained: a joint correction, not an exact original-objective reproduction."]
                  if "FISMCorrected" in model_names else []),
            ], "tables": [paired_table("All individual models and selected hybrids: mean development nDCG@10",
                                        selected_rows, "nDCG")],
             "discussion": ("Affine alignment matches expert response scales before enforcing sum-to-one weights. "
                "The alignment uses sampled zero/one labels; it does not produce calibrated probabilities. "
                if "calibrated" in family_names else
                "The sum-to-one constraint ties coefficient scale to standardized expert scores "
                "while regression targets are zero or one. Poorer constrained rankings can reflect this "
                "scale restriction; they do not establish that constrained fusion is generally inferior. "
                if "constrained" in family_names else "") +
                "Contextual coefficients allow expert contributions to vary with history size, "
                "genre entropy and item popularity. Small differences between ablations do not justify "
                "a claim that additional complexity improves generalization. The BPR grid changes dimension "
                "and training budget together, so their effects cannot be isolated. RRF and activity-group "
                "switching provide alternative hybrid controls. Random is excluded from fusion. The best "
                "development settings still require a frozen test evaluation; overlapping data seeds are "
                "sensitivity checks, not independent replications."},
            {"task": 2, "title": "Task 2: effectiveness and interpretation", "paragraphs": [
                "All metrics are implemented independently: precision, recall, nDCG, MRR, hit rate, "
                "catalog coverage, smoothed-frequency novelty, genre Jaccard diversity and genre JSD "
                "calibration. Metrics macro-average users with held-out positives. Training-only history "
                "terciles and item popularity define groups; head items are the most frequent 20% of the catalog.",
            ], "tables": [
                {"caption": "Naive baselines and alternative hybrids (mean development metrics)",
                 "columns": ["Method", "nDCG@10", "Recall@10", "Novelty"], "rows": metric_rows},
                paired_table("Static ridge: mean coefficients on standardized expert scores", coefficient_rows, "Weight"),
                {"caption": "Contextual model: mean user and item group results",
                 "columns": ["Group", "nDCG", "Recall", "Diversity", "Exposure"], "rows": group_rows},
            ], "discussion": "Coefficients are unconstrained regression parameters, not probability weights "
                "or causal contributions. Correlated experts can exchange coefficients without equivalent "
                "changes in ranking; feature removal is therefore a stronger diagnostic than coefficient "
                "magnitude alone. Task 1 differences must be interpreted alongside popularity, "
                "coverage and group utility. Descriptive paired intervals in the aggregate evidence do not "
                "correct for model selection. The proposed exception study must preserve explicit dislikes "
                "and give its controls the same rating information. Its results are not yet included here, "
                "and a behavioural interaction cannot by itself establish emotional understanding or novelty."},
            {"task": 3, "title": "Task 3: societal aspects and reranking", "paragraphs": [
                "Greedy rerankers trade relevance against diversity, calibration [3] or head/tail exposure. "
                "The table fixes strength at 0.5, with a top-100 candidate pool. Lower JSD indicates "
                "better genre calibration; head exposure is a share, not a fairness verdict. Catalog parity "
                "is an explicit diagnostic objective. The ordering comparison uses RRF with constant 60 "
                "and retains full ranking support.",
                *([exposure_note] if exposure_note else []),
            ], "tables": [
                {"caption": "Contextual model: mean development trade-offs",
                 "columns": ["Reranker", "nDCG", "Diversity", "JSD", "Head"] + (["UPD"] if has_popularity else []), "rows": reranker_rows},
                {"caption": "Before/after RRF: nDCG@10 and objective value (diversity, JSD, head share or UPD)",
                 "columns": ["Objective", "Bef. nDCG", "Aft. nDCG", "Bef. goal", "Aft. goal"], "rows": ordering_rows},
            ], "discussion": f"Candidate support limits exposure correction: with H head and T tail candidates, "
                f"a list of k has max(0,k-T)<=heads<=min(k,H). Adaptive expansion used {mean_pool:.1f} "
                f"candidates on average and achieved at least {agreement:.0%} agreement with full-pool "
                "exposure reranking in these runs. This measures candidate work, not end-to-end latency. "
                f"The user-side policy selects exposure strength on {policy_fit} users {policy_description}, "
                "targeting 95% utility retention. " + retention_text +
                "Smaller group gaps can hide harm to all groups, so absolute and worst-group utility must "
                "also be inspected. Calibration and useful preference exceptions may conflict; that hypothesis "
                "needs the separate rating-aware study."},
        ],
    }


def build_final_content(directory, frozen, group=None, members=()):
    """Read a completed evaluation and its matching freeze, never raw test labels.

    All aggregate results accompany the report JSON. Main tables are chosen by
    frozen model role, not by test performance. The report remains DRAFT until
    --finalize is explicitly requested with complete contributor metadata.
    """
    directory, frozen = Path(directory), Path(frozen)
    manifest = read_json(directory / "manifest.json")
    bundle = read_json(frozen / "freeze.json")
    if (manifest.get("status") != "complete" or manifest.get("test_read") is not True or
            manifest.get("test_model_selection") is not False):
        raise ValueError("Expected completed frozen test evaluation without test-based selection")
    if manifest["freeze_sha256"] != digest(frozen / "freeze.json"):
        raise ValueError("Evaluation does not match the supplied freeze")
    if manifest["output_sha256"]["results.json"] != digest(directory / "results.json"):
        raise ValueError("Final results hash mismatch")
    results = read_json(directory / "results.json")
    if set(results) != set(manifest["models"]) or set(results) != set(bundle["models"]):
        raise ValueError("Frozen and evaluated model sets differ")
    k = manifest["k"]
    selection = bundle["selection"]
    context = selection["families"]["context"]
    static = selection["families"]["static"]

    def metric(model, name):
        value = results[model]["aggregate"][name]
        return "n/a" if value is None else f"{value:.3f}"

    def label(name):
        return re.sub(r"^[0-9]+-", "", name).replace("-", " ")

    expert_names = [name for name, spec in bundle["models"].items() if spec["kind"] == "expert"]
    main_names = list(dict.fromkeys(expert_names + list(selection["families"].values()) + ["rrf", "group-switch"]))
    individual_rows = [[label(n), metric(n, f"ndcg@{k}"), metric(n, f"recall@{k}")]
                       for n in main_names if n in results]
    weights = bundle["models"][static]["coefficients"]["weights"]
    coefficient_rows = [[label(name), f"{value:+.3f}"] for name, value in weights.items()]
    group_rows = []
    for group_id, name in [("0", "Sparse users"), ("1", "Medium users"), ("2", "Dense users")]:
        if group_id not in results[context]["groups"]:
            continue
        row = results[context]["groups"][group_id]
        group_rows.append([name] + [f"{row[m]:.3f}" for m in [f"ndcg@{k}", f"recall@{k}", "diversity"]] + ["n/a"])
    for name in ["head", "tail"]:
        exposure = results[context]["aggregate"]["head_exposure"]
        group_rows.append([name.title() + " items", "n/a", metric(context, name + "_recall"), "n/a",
                           f"{exposure if name == 'head' else 1 - exposure:.3f}"])
    reranker_rows = []
    has_popularity = "popularity_jsd" in results[context]["aggregate"]
    rerank_metrics = [f"ndcg@{k}", "diversity", "calibration_jsd", "head_exposure"] + (["popularity_jsd"] if has_popularity else [])
    for suffix, name in [("", "Base context"), ("/diversity-0.5", "Diversity"),
                         ("/calibration-0.5", "Calibration"), ("/exposure-0.5", "Item exposure"),
                         ("/popularity_calibration-0.5", "Popularity calibration")]:
        if context + suffix in results:
            reranker_rows.append([name] + [metric(context + suffix, m) for m in rerank_metrics])
    ordering_rows = []
    for method, goal in [("diversity", "diversity"), ("calibration", "calibration_jsd"),
                         ("exposure", "head_exposure"), ("popularity_calibration", "popularity_jsd")]:
        before, after = "rerank-experts-then-rrf/" + method, "rrf-then-rerank/" + method
        if before in results and after in results:
            ordering_rows.append([method, metric(before, f"ndcg@{k}"), metric(after, f"ndcg@{k}"),
                                  metric(before, goal), metric(after, goal)])
    group_policy = results.get("group-utility-budget-exposure")
    retention = []
    if group_policy:
        for group_id, row in group_policy["groups"].items():
            base = results[context]["groups"][group_id][f"ndcg@{k}"]
            if base > 0:
                retention.append(row[f"ndcg@{k}"] / base)
    audit_text = (f"User-group policy test retention ranged from {min(retention):.1%} to {max(retention):.1%}. "
                  if retention else "No group-policy retention is available for this evaluation. ")
    protocol = manifest["protocol"]
    protocol_text = " ".join(f"{key.replace('_', ' ')}: {protocol[key]}." for key in
                             ["relevance", "history_mask", "score_normalization"] if key in protocol)
    return {
        "schema_version": 1, "status": "DRAFT", "group": group, "members": list(members),
        "title": "Hybrid recommendation: frozen held-out evaluation", "evidence_stage": "held-out test",
        "evidence_label": "Frozen held-out metrics; contributor and interpretation review pending",
        "source_sha256": {}, "references": REFERENCES,
        "final_evaluation": {"manifest_sha256": digest(directory / "manifest.json"),
                             "results_sha256": digest(directory / "results.json"),
                             "freeze_sha256": manifest["freeze_sha256"], "protocol": protocol,
                             "test_model_selection": False},
        "aggregate_evidence": {name: {key: value for key, value in row.items() if key != "per_user"}
                               for name, row in results.items()},
        "cover_notes": [
            "Models, transformations and reranking choices were frozen before the held-out evaluation. "
            "Every frozen model is retained in the aggregate evidence; main tables use model roles, not test rankings.",
            "One page is allocated to each major task, with at most 200 discussion words per task. "
            "The report content JSON contains the complete aggregate metrics without individual histories or recommendations.",
            "Contributor metadata is incomplete; remaining names and the individual peer-feedback workbook "
            "must be completed before submission. AI assistance was used for implementation, analysis and report preparation.",
        ],
        "sections": [
            {"task": 1, "title": "Task 1: individual models and hybrids", "paragraphs": [
                f"MovieLens 100K; full-catalog ranking at k={k}. The frozen evaluation includes "
                f"{manifest['users']} users and {manifest['interactions']} held-out interactions. " + protocol_text,
                "Individual RecBole models include EASE [1] and BPR [2]. Expert hyperparameters are "
                "selected on meta-fit users; regression coefficients use standardized expert scores. "
                "Hybrid families and penalties are selected on development users, then frozen. "
                "No model is refitted and no test-based winner is selected. " +
                ("The constrained hybrid's weights sum to one, but may remain negative. "
                 if "constrained" in selection["families"] else ""),
                *(["The calibrated variant learns per-expert affine response alignment on meta-fit users "
                   "before enforcing sum-to-one weights. Effective raw-z coefficients need not sum to one. "
                   "Scores remain unbounded and are not calibrated probabilities."]
                  if "calibrated" in selection["families"] else []),
            ], "tables": [paired_table(f"Frozen individual models and central hybrids: nDCG@{k}",
                                        [row[:2] for row in individual_rows], "nDCG")],
             "discussion": "A larger model or additional expert does not guarantee better rankings. "
                "The individual and hybrid rows expose whether fusion contributes beyond its constituent "
                "models. Test results answer the frozen comparison; they must not be used to select a new "
                "penalty or reranker and then reported as untouched evaluation. Training budgets, parameter "
                "counts and any corrected upstream implementations belong in the final methodological account."},
            {"task": 2, "title": "Task 2: effectiveness and interpretation", "paragraphs": [
                "Metrics are computed independently of RecBole. Accuracy macro-averages users with held-out "
                "positives. Activity terciles and the top-20% popularity head use training data only. "
                "Genre diversity is pairwise Jaccard distance; item-group recall includes users with positives "
                "in that group. Coefficients below are frozen static regression coefficients.",
            ], "tables": [
                paired_table("Expert contributions to standardized static regression", coefficient_rows, "Weight"),
                {"caption": "Contextual hybrid: user and item group performance",
                 "columns": ["Group", "nDCG", "Recall", "Diversity", "Exposure"], "rows": group_rows},
            ], "discussion": "Coefficients are associations between correlated scores, not causal credit "
                "or probability weights. Matched ablations and user-level comparisons are needed to explain "
                "their contribution. Group results expose whether aggregate utility hides poor performance "
                "for sparse users or tail items. Activity groups do not measure demographic fairness. "
                "A custom exception model must beat controls with equal rating information before attributing "
                "any gain to its proposed mechanism. Rankings alone cannot demonstrate emotional understanding."},
            {"task": 3, "title": "Task 3: societal aspects and reranking", "paragraphs": [
                "The fixed-strength comparison uses 0.5 for diversity, calibration [3] and exposure "
                "rerankers. Genre JSD measures deviation from historical genre shares; lower values mean "
                "better calibration. Head exposure measures allocation to popular items. RRF ordering "
                "comparisons retain full ranking support; objective values accompany accuracy.",
                *([f"UPD is per-user popularity-profile JSD. The base model's discounted catalog exposure "
                   f"Gini is {metric(context, 'exposure_gini')}; lower Gini means less concentrated allocation."]
                  if has_popularity else []),
            ], "tables": [
                {"caption": "Contextual model: accuracy and societal trade-offs",
                 "columns": ["Reranker", "nDCG", "Diversity", "JSD", "Head"] + (["UPD"] if has_popularity else []), "rows": reranker_rows},
                {"caption": "Before/after RRF: nDCG and objective value (diversity, JSD, head share or UPD)",
                 "columns": ["Objective", "Bef. nDCG", "Aft. nDCG", "Bef. goal", "Aft. goal"], "rows": ordering_rows},
            ], "discussion": audit_text + "A group-level fitted constraint is not a held-out guarantee. "
                "Smaller utility gaps can coexist with losses for everyone, so absolute and worst-group "
                "utility must accompany gap metrics. Catalog-proportional item exposure is a declared "
                "objective, not a universal definition of fairness. Candidate support bounds the correction "
                "that any reranker can achieve. Order matters because reranking changes the ranking "
                "information available to fusion. The complete aggregate artifact retains all frozen strengths, "
                "including poor trade-offs; no strength is retuned on test labels."},
        ],
    }


def build_multi_final_content(evaluations, group=None, members=(), expected_seeds=(2026, 2027, 2028)):
    """Aggregate separately verified evaluations by frozen semantic role, never raw run name."""
    seeds = [str(row[0]) for row in evaluations]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(map(str, expected_seeds)):
        raise ValueError("Exactly the declared complete primary seed set is required")
    records, roles, contents = {}, {}, {}
    for seed, directory, frozen in evaluations:
        seed, directory, frozen = str(seed), Path(directory), Path(frozen)
        contents[seed] = build_final_content(directory, frozen, group, members)
        manifest, bundle = read_json(directory / "manifest.json"), read_json(frozen / "freeze.json")
        if str(bundle.get("seed")) != seed or bundle.get("status") != "frozen" or bundle.get("test_read") is not False:
            raise ValueError("Primary evaluation seed differs from its frozen manifest")
        if not bundle.get("code_sha256") or not bundle.get("source_artifacts_sha256"):
            raise ValueError("Primary freeze must preserve code and source-artifact hashes")
        if not bundle.get("expected_test_sha256") or manifest.get("test_sha256") != bundle["expected_test_sha256"]:
            raise ValueError("Primary evaluation test hash differs from its frozen split")
        mapping = {}
        for name, spec in bundle["models"].items():
            if spec["kind"] == "expert":
                model = bundle["sources"][name]["manifest"]["model"]
                role = "expert:" + model
                if role in mapping or spec.get("expert") != name:
                    raise ValueError("Ambiguous frozen expert identity")
                mapping[role] = name
        families = bundle["selection"]["families"]
        if len(set(families.values())) != len(families):
            raise ValueError("Selected family roles must identify distinct frozen models")
        for family, name in families.items():
            spec = bundle["models"][name]
            variant = "static" if family in {"constrained", "calibrated"} else family.removesuffix("-pairwise")
            if (spec.get("kind") != "linear" or spec.get("variant") != variant or
                    not name.startswith(family + "-") or
                    (family == "static" and name.startswith("static-pairwise-")) or
                    (family == "context" and name.startswith("context-pairwise-"))):
                raise ValueError(f"Swapped or incompatible frozen family role: {family}")
            mapping["hybrid:" + family] = name
        context = families["context"]
        for suffix in ["", "/diversity-0.5", "/calibration-0.5", "/exposure-0.5", "/popularity_calibration-0.5"]:
            if context + suffix in bundle["models"]:
                mapping["context" + suffix] = context + suffix
        for name in ["rrf", "group-switch", "group-utility-budget-exposure"] + [
                prefix + method for prefix in ["rerank-experts-then-rrf/", "rrf-then-rerank/"]
                for method in ["diversity", "calibration", "exposure", "popularity_calibration"]]:
            if name in bundle["models"]:
                mapping[name] = name
        records[seed] = {"manifest": manifest, "freeze": bundle, "results": contents[seed]["aggregate_evidence"],
                         "manifest_sha256": digest(directory / "manifest.json"),
                         "results_sha256": digest(directory / "results.json"), "freeze_sha256": digest(frozen / "freeze.json"),
                         "code_sha256": bundle.get("code_sha256", {}),
                         "source_artifacts_sha256": bundle.get("source_artifacts_sha256", {}),
                         "frozen_selection": bundle["selection"]}
        roles[seed] = mapping
    seeds = sorted(records)
    reference = records[seeds[0]]["manifest"]
    if any(set(roles[seed]) != set(roles[seeds[0]]) for seed in seeds):
        raise ValueError("Primary seeds have different canonical model/family coverage")
    if any(records[seed]["manifest"]["k"] != reference["k"] or
           records[seed]["manifest"]["protocol"] != reference["protocol"] for seed in seeds):
        raise ValueError("Primary seeds use different evaluation protocols")
    if any(records[seed]["manifest"]["users"] != reference["users"] for seed in seeds):
        raise ValueError("Primary seeds have different evaluated user counts")
    k = reference["k"]
    def values(role, *keys):
        result = []
        for seed in seeds:
            value = records[seed]["results"][roles[seed][role]]
            for key in keys:
                value = value[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or unavailable primary metric: {role}/{keys}")
            result.append(value)
        return result
    def avg(role, metric):
        return statistics.mean(values(role, "aggregate", metric))
    def fmt(role, metric):
        return f"{avg(role, metric):.3f}"
    content = contents[seeds[0]]
    content.update(title="Hybrid recommendation: three-split frozen evaluation",
                   evidence_label=f"Frozen held-out means across {len(seeds)} overlapping splits; contributors pending",
                   final_evaluation={"seeds": {seed: {key: records[seed][key] for key in
                                      ["manifest", "manifest_sha256", "results_sha256", "freeze_sha256", "code_sha256",
                                       "source_artifacts_sha256", "frozen_selection"]} for seed in seeds},
                                     "aggregation": "Equal-weight arithmetic means of each split's macro-user metrics; shared users and dataset, not independent replications.",
                                     "test_model_selection": False},
                   aggregate_evidence={seed: records[seed]["results"] for seed in seeds}, canonical_roles=roles)
    content["cover_notes"][0] = "Every primary table averages separately frozen evaluations for seeds " + ", ".join(seeds) + ". Model identities and family roles are aligned from frozen metadata; no test-based selection occurs."
    task1, task2, task3 = content["sections"]
    expert_roles = sorted(role for role in roles[seeds[0]] if role.startswith("expert:"))
    family_roles = sorted(role for role in roles[seeds[0]] if role.startswith("hybrid:"))
    task1["tables"] = [paired_table(f"Mean held-out nDCG@{k}; individual models and frozen hybrid families",
        [[role.split(":")[-1], f"{avg(role, f'ndcg@{k}'):.4f}"] for role in expert_roles + family_roles + ["rrf", "group-switch"]], "nDCG")]
    counts = "/".join(str(records[seed]["manifest"]["interactions"]) for seed in seeds)
    task1["paragraphs"][0] = (f"MovieLens 100K, full-catalog ranking at k={k}; {len(seeds)} overlapping random splits "
        f"({', '.join(seeds)}), each {reference['users']} users; held-out interactions by seed: {counts}. "
        "All observed ratings are relevant. Inference uses frozen TRAIN fits and masks TRAIN+validation history. "
        "Tables average per-split macro metrics, not pooled users or independent dataset replications.")
    task1["paragraphs"][1:] = [
        "Meta-fit users select expert settings and fit standardized-score coefficients; development users choose "
        "hybrid penalties before freezing. User fusion adds score-by-activity/genre-entropy interactions; item "
        "fusion adds score-by-popularity interactions; disagreement adds cross-expert score deviation; context "
        "combines these. Pairwise variants fit positive-minus-negative feature differences. RRF sums reciprocal "
        "ranks with offset 60; group-switch selects a meta-fit expert per activity tercile.",
        "Static ridge is unconstrained. The lecture sum-to-one baseline allows negative weights; calibrated ridge "
        "first learns an affine response alignment per expert on meta-fit users. Effective raw-z coefficients "
        "need not sum to one, and scores are unbounded."]
    standalone = []
    for seed in seeds:
        name = records[seed]["freeze"]["selection"]["best_expert"]
        if name not in roles[seed].values() or records[seed]["freeze"]["models"][name].get("kind") != "expert":
            raise ValueError("The frozen standalone reference must identify an individual expert")
        standalone.append(_finite_metric(records[seed]["results"][name]["aggregate"], f"ndcg@{k}"))
    reference_mean = statistics.mean(standalone)
    disagreement_mean = avg("hybrid:disagreement", f"ndcg@{k}")
    relative = f" ({disagreement_mean / reference_mean - 1:+.1%} relative)" if reference_mean > 0 else ""
    task1["discussion"] = (
        f"Disagreement-conditioned fusion scores {disagreement_mean:.4f} versus {reference_mean:.4f} for the "
        f"validation-selected standalone reference{relative}; static fusion scores {avg('hybrid:static', f'ndcg@{k}'):.4f}. "
        "The reference was selected separately in each split; these are descriptive held-out differences. "
        f"Response alignment raises constrained nDCG from {avg('hybrid:constrained', f'ndcg@{k}'):.3f} to "
        f"{avg('hybrid:calibrated', f'ndcg@{k}'):.3f}: raw sum-to-one restrictions also impose a response-scale cost. "
        "Negative weights remain allowed and are not probabilities. "
        "FISMCorrected jointly repairs the supplied logit/loss mismatch and target-history inclusion, retaining its "
        "non-squared regularizer; it is not an exact paper reproduction. LightGCN received a declared longer "
        "training budget following improving validation curves. Test metrics never choose models or budgets.")
    coefficient_rows = []
    for role in expert_roles:
        by_seed = [records[seed]["freeze"]["models"][roles[seed]["hybrid:static"]]["coefficients"]["weights"] for seed in seeds]
        included = [roles[seed][role] in weights for seed, weights in zip(seeds, by_seed)]
        if not any(included):
            continue
        if not all(included):
            raise ValueError("Static hybrids use different expert coverage across primary seeds")
        weights = [weights[roles[seed][role]] for seed, weights in zip(seeds, by_seed)]
        coefficient_rows.append([role.split(":", 1)[1], f"{statistics.mean(weights):+.3f}"])
    group_roles = [role for role in ["expert:EASE", "expert:SLIMElastic", "hybrid:context"] if role in roles[seeds[0]]]
    cross_groups = [[role.split(":")[-1]] + [f"{statistics.mean(values(role, 'groups', group_id, f'ndcg@{k}')):.3f}" for group_id in ["0", "1", "2"]] + [fmt(role, "tail_recall")] for role in group_roles]
    diversity = [f"{statistics.mean(values('context', 'groups', group_id, 'diversity')):.3f}" for group_id in ["0", "1", "2"]]
    task2["tables"] = [paired_table("Mean static-regression coefficients on standardized expert scores", coefficient_rows, "Weight"),
        {"caption": "Cross-model group accuracy: user nDCG and tail-item recall", "columns": ["Model", "Sparse", "Medium", "Dense", "Tail recall"], "rows": cross_groups},
        {"caption": "Context hybrid: beyond-accuracy user and item groups", "columns": ["Measure", "Sparse", "Medium", "Dense", "Head", "Tail"],
         "widths": [.30, .14, .14, .14, .14, .14],
         "rows": [["Diversity"] + diversity + ["n/a", "n/a"], ["Exposure", "n/a", "n/a", "n/a", fmt("context", "head_exposure"), f"{1-avg('context', 'head_exposure'):.3f}"]]}]
    task2["discussion"] = ("Regression weights describe correlated score associations, not causal contributions. "
        "The same model can behave differently for sparse users and tail items; the table compares those groups "
        "across EASE, SLIM and the contextual hybrid without selecting test-specific winners. Genre diversity and "
        "item exposure add information that ranking accuracy alone hides. Activity is not demographic identity. "
        "The separately specified categorical and recorded-event field studies test objective alignment and "
        "recurrent routing using equal-information controls; their complete results belong to the labelled appendices.")
    rerank_metrics = [f"ndcg@{k}", "diversity", "calibration_jsd", "head_exposure", "popularity_jsd"]
    task3["tables"][0]["rows"] = [[label] + [fmt(role, metric) for metric in rerank_metrics]
        for role, label in [("context", "Base context"), ("context/diversity-0.5", "Diversity"),
                            ("context/calibration-0.5", "Calibration"), ("context/exposure-0.5", "Item exposure"),
                            ("context/popularity_calibration-0.5", "Popularity calibration")]]
    task3["tables"][0]["caption"] = "Mean held-out accuracy and societal trade-offs"
    task3["tables"][1]["rows"] = [[method, fmt("rerank-experts-then-rrf/" + method, f"ndcg@{k}"),
        fmt("rrf-then-rerank/" + method, f"ndcg@{k}"), fmt("rerank-experts-then-rrf/" + method, goal),
        fmt("rrf-then-rerank/" + method, goal)] for method, goal in [("diversity", "diversity"),
            ("calibration", "calibration_jsd"), ("exposure", "head_exposure"), ("popularity_calibration", "popularity_jsd")]]
    effects = [[label, f"{statistics.mean(values('context', 'groups', group_id, f'ndcg@{k}')):.3f}",
                f"{statistics.mean(values('group-utility-budget-exposure', 'groups', group_id, f'ndcg@{k}')):.3f}", "nDCG"]
               for group_id, label in [("0", "Sparse users"), ("1", "Medium users"), ("2", "Dense users")]]
    effects += [[label, fmt("context", metric), fmt("context/exposure-0.5", metric), "Recall"]
                for label, metric in [("Head items", "head_recall"), ("Tail items", "tail_recall")]]
    task3["tables"].append({"caption": "Group effects: utility-policy users and fixed-exposure item groups",
                            "columns": ["Group", "Base", "After", "Metric"], "rows": effects})
    task3["paragraphs"][1:] = [f"UPD is per-user popularity-profile JSD. The base contextual hybrid's mean discounted "
        f"catalog exposure Gini is {fmt('context', 'exposure_gini')}; lower values mean less concentrated allocation."]
    retentions = [records[seed]["results"][roles[seed]["group-utility-budget-exposure"]]["groups"][g][f"ndcg@{k}"] /
                  records[seed]["results"][roles[seed]["context"]]["groups"][g][f"ndcg@{k}"] for seed in seeds for g in ["0", "1", "2"]
                  if records[seed]["results"][roles[seed]["context"]]["groups"][g][f"ndcg@{k}"] > 0]
    task3["discussion"] = (
        f"Diversity rises from {fmt('context', 'diversity')} to {fmt('context/diversity-0.5', 'diversity')}, "
        f"while nDCG falls from {fmt('context', f'ndcg@{k}')} to {fmt('context/diversity-0.5', f'ndcg@{k}')}. "
        f"Exposure reranking raises tail recall from {fmt('context', 'tail_recall')} to "
        f"{fmt('context/exposure-0.5', 'tail_recall')}. Across seeds and activity groups, policy utility retention ranged from {min(retentions):.1%} "
        f"to {max(retentions):.1%}. The independent calibration target is not a held-out guarantee. Smaller gaps can "
        "hide losses for every group, so absolute utility accompanies disparity. Catalog-proportional exposure is a "
        "declared objective, not a universal fairness definition. With H head and T tail candidates, feasible head "
        "counts satisfy max(0,k-T)<=heads<=min(k,H). Reranking before fusion changes information supplied to the "
        "fusion rule, so its order can change both accuracy and the target objective. Every frozen strength remains "
        "in aggregate evidence; test outcomes never retune strength.")
    validate_content(content)
    return content


def add_exception_findings(content, directory):
    """Insert a separately labelled, rating-aware development experiment."""
    if content.get("evidence_stage") == "held-out test":
        raise ValueError("Use edited content to place development findings alongside final held-out tables")
    directory = Path(directory)
    path = directory / "summary.json"
    if path.exists():
        summary = read_json(path)
    else:
        path = directory / "aggregates.json"
        aggregate = read_json(path)
        summary = {}
        for model, row in aggregate.items():
            for seed, values in row["per_seed"].items():
                summary.setdefault(seed, {})[model + ":liked_ratings"] = values
    if not summary:
        raise ValueError("Exception summary has no seeds")
    labels = {"contrast_transfer": "Contrast transfer", "signed_channels": "Signed channels",
              "positive_ease": "Positive EASE", "pair_gate": "Pair gate"}
    if all("observed_ease:liked_ratings" in seed for seed in summary.values()):
        labels = {"observed_ease": "Observed EASE", **labels}
    rows = []
    means = {}
    for name, label in labels.items():
        values = [seed[name + ":liked_ratings"] for seed in summary.values()]
        means[name] = statistics.mean(row["like_ndcg"] for row in values)
        rows.append([label, f"{means[name]:.4f}",
                     f"{statistics.mean(row['known_dislike_rate_per_slot'] for row in values):.4f}"])
    section = content["sections"][1]
    section["tables"][0] = {
        "caption": "Separate rating-aware development study: mean nDCG@10 for likes and known-dislike slot rate",
        "columns": ["Model", "Liked nDCG", "Dislike rate"], "rows": rows,
    }
    section["paragraphs"].append(
        f"Exception study ({len(summary)} overlapping seeds): likes >=4, dislikes <=2, and rating 3 neutral. "
        "Seen items are masked. Meta-fit selects settings. Liked-item evaluation excludes users without held-out likes. "
        "Its separate cohort and relevance definition are not directly comparable with Task 1.")
    supported = means["contrast_transfer"] > means["signed_channels"]
    section["discussion"] = (
        "Static coefficients are associations between correlated standardized scores, not causal credit "
        "or probabilities. The group table tests whether aggregate accuracy hides sparse-user or tail-item failures. "
        f"Contrast transfer achieved {means['contrast_transfer']:.4f} liked nDCG versus {means['signed_channels']:.4f} "
        "for separate positive/negative linear channels. " +
        ("This observed difference needs controlled confirmation before a contribution claim. " if supported else
         "The proposed relation mechanism therefore failed this comparison; complexity did not earn its place. ") +
        "The pair-gate revision was motivated by earlier development failures and is exploratory. "
        "Information-matched controls separate rating access from architecture. Low known-dislike rates only "
        "count observed held-out dislikes: unknown ratings cannot be treated as approval. These findings "
        "do not establish emotional understanding, universal superiority or novelty."
    )
    content.setdefault("source_sha256", {})[f"{directory.name}/{path.name}"] = digest(path)
    content["exception_evidence"] = summary
    content["cover_notes"] = [note.replace("complete the rating-aware exception study, ", "")
                              for note in content.get("cover_notes", [])]
    return content


def _supplement(content, directory, kind, filenames, section, evidence):
    """Attach curated, independently labelled evidence without replacing main results."""
    from package_project import check_aggregate_only
    check_aggregate_only(evidence, kind)
    if evidence.get("final_evaluation") and content.get("final_evaluation", {}).get("seeds"):
        if set(evidence["aggregates"]["seeds"]) != set(content["final_evaluation"]["seeds"]):
            raise ValueError("Supplemental held-out seeds must match the complete primary evaluation")
    if (kind in content.get("supplemental_evidence", {}) or
            any(s.get("study") == kind for s in content.get("appendices", []))):
        raise ValueError(f"Duplicate supplemental study: {kind}")
    section.update({"task": 2, "study": kind})
    hashes = {f"{directory.name}/{filename}": digest(directory / filename) for filename in filenames}
    if any(name in content.get("source_sha256", {}) and content["source_sha256"][name] != value
           for name, value in hashes.items()):
        raise ValueError("Supplemental evidence directories need distinct names to preserve every source hash")
    content.setdefault("appendices", []).append(section)
    content.setdefault("supplemental_evidence", {})[kind] = evidence
    content.setdefault("source_sha256", {}).update(hashes)
    return content


def _finite_metric(row, key):
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Supplemental metric must be finite: {key}")
    return value


def _checkpoint_label(protocol):
    checkpoints = protocol.get("checkpoints")
    if (not isinstance(checkpoints, list) or not checkpoints or
            any(type(epoch) is not int or epoch < 1 for epoch in checkpoints) or
            checkpoints != sorted(set(checkpoints))):
        raise ValueError("Field protocol requires ordered positive checkpoint epochs")
    return "/".join(map(str, checkpoints))


def _read_frozen_field(directory, frozen):
    """Verify aggregate final outputs and metadata only; never load prediction or label files."""
    from package_project import check_aggregate_only
    directory, frozen = Path(directory), Path(frozen)
    manifest, seal = read_json(directory / "manifest.json"), read_json(frozen / "manifest.json")
    frozen_hash = digest(frozen / "manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("test_read") is not True or
            manifest.get("test_evaluated") is not True or manifest.get("selection_after_test") is not False):
        raise ValueError("Field final results require completed evaluation without post-test selection")
    if (seal.get("status") != "frozen" or seal.get("test_read") is not False or
            seal.get("test_evaluated") is not False or manifest.get("frozen_manifest_sha256") != frozen_hash or
            (frozen / "manifest.sha256").read_text().strip() != frozen_hash or
            manifest.get("code_sha256") != seal.get("code_sha256")):
        raise ValueError("Field final evaluation does not match its frozen source manifest")
    if manifest.get("output_sha256", {}).get("aggregates.json") != digest(directory / "aggregates.json"):
        raise ValueError("Field final aggregate hash mismatch")
    evidence = read_json(directory / "aggregates.json")
    if evidence.get("stage") != "test" or evidence.get("test_read") is not True:
        raise ValueError("Field final aggregate must declare the held-out test stage")
    seeds = evidence.get("seeds", {})
    if not seeds or set(seeds) != set(map(str, seal.get("seeds", []))):
        raise ValueError("Field final aggregate and frozen seed sets differ")
    protocols, bundles = {}, {}
    for seed, result in seeds.items():
        if not re.fullmatch(r"[0-9]+", seed):
            raise ValueError("Frozen field seed must be a numeric identifier")
        path = frozen / seed / "bundle.json"
        if seal.get("bundle_sha256", {}).get(seed) != digest(path):
            raise ValueError("Frozen field bundle hash mismatch")
        bundle = read_json(path)
        protocol_path = frozen / seed / "protocol.json"
        protocol_hash = digest(protocol_path)
        if (bundle.get("status") != "frozen" or bundle.get("test_read") is not False or
                bundle.get("test_evaluated") is not False or str(bundle.get("seed")) != seed or bundle.get("k") != 10 or
                bundle.get("source_protocol_sha256") != protocol_hash or
                bundle.get("payload_sha256", {}).get("protocol.json") != protocol_hash):
            raise ValueError("Frozen field selection protocol is invalid or its hash differs")
        if (set(result.get("models", {})) != set(bundle.get("models", [])) or
                result.get("selections") != bundle.get("selections")):
            raise ValueError("Field final models or selections differ from the sealed choices")
        protocols[seed], bundles[seed] = read_json(protocol_path), bundle
        protocols[seed].pop("source_run", None)  # Local filesystem paths are unnecessary report metadata.
    comparable = [{key: value for key, value in protocol.items() if key not in {"seeds", "prior_v1_manifest_sha256"}}
                  for protocol in protocols.values()]
    if any(protocol != comparable[0] for protocol in comparable[1:]):
        raise ValueError("Frozen field seeds use different selection protocols")
    protocol = {**next(iter(protocols.values())), "seeds": [int(seed) for seed in sorted(seeds)]}
    verification = {"evaluation_manifest": manifest, "frozen_manifest": seal,
                    "evaluation_manifest_sha256": digest(directory / "manifest.json"),
                    "frozen_manifest_sha256": frozen_hash, "bundles": bundles, "protocols": protocols,
                    "inference": "Frozen TRAIN-only rating context; mask TRAIN+validation items for ranking; no refit or post-test selection."}
    check_aggregate_only({"aggregates": evidence, "verification": verification}, "field-final")
    return evidence, protocol, verification


def add_adaptive_findings(content, directory, frozen=None):
    """Read the categorical model's separate, completed development evidence."""
    directory = Path(directory)
    final = frozen is not None
    if final:
        evidence, protocol, verification = _read_frozen_field(directory, frozen)
    else:
        evidence, protocol = read_json(directory / "aggregates.json"), read_json(directory / "protocol.json")
    from package_project import check_aggregate_only
    check_aggregate_only({"protocol": protocol, "aggregates": evidence}, "adaptive")
    required = {"schema_version": 1, "test_read": False, "stage": "development",
                "training_objective": "categorical_cross_entropy",
                "selection_metric": "meta_fit_macro_cross_entropy",
                "rating_categories": [1, 2, 3, 4, 5], "ranking_adapter": "P(rating>=4)"}
    if any(protocol.get(key) != value for key, value in required.items()) or evidence.get("test_read") is not final:
        raise ValueError("Adaptive evidence requires the declared five-category protocol and verified evaluation stage")
    checkpoints = _checkpoint_label(protocol)
    seeds = evidence.get("seeds", {})
    if not seeds:
        raise ValueError("Adaptive evidence has no completed seeds")
    if "seeds" in protocol and set(map(str, protocol["seeds"])) != set(seeds):
        raise ValueError("Adaptive protocol and aggregate seeds differ")
    labels = {"adaptive": "Adaptive field", "fixed_flow": "Fixed flow", "hard_clamp": "Hard clamp",
              "global_histogram": "Global frequencies", "item_histogram": "Item frequencies",
              "item_user_product": "Item/user product"}
    rows, ce = [], {}
    for name, label in labels.items():
        values = [seed["models"][name] for seed in seeds.values()]
        ce[name] = statistics.mean(_finite_metric(row["categorical"], "macro_cross_entropy") for row in values)
        rows.append([label, f"{ce[name]:.4f}"] + [
            f"{statistics.mean(_finite_metric(row['categorical'], key) for row in values):.4f}"
            for key in ["macro_brier", "macro_expected_rating_rmse"]] + [
            f"{statistics.mean(_finite_metric(row['ranking']['liked_ratings'], 'ndcg@10') for row in values):.4f}"])
    controls = [ce[name] - ce["adaptive"] for name in ["fixed_flow", "hard_clamp"]]
    direction = ("The adaptive field has lower mean categorical loss than both matched controls. "
                 if all(delta > 0 for delta in controls) else
                 "The adaptive field does not improve mean categorical loss over both matched controls. ")
    section = {
        "title": "Appendix: Task 2 categorical evidence experiment",
        "paragraphs": [
            (f"Frozen held-out evaluation, {len(seeds)} overlapping seeds; no model or checkpoint is selected on test. " if final else
             f"Supplemental development experiment, {len(seeds)} overlapping seeds; the project test stays closed. ") +
            "The predictor learns five rating categories from observed training ratings and an observation mask. "
            "Hidden training targets supply no source values. Missing observations are unknown; no hard like/dislike classes train the model.",
            "Adaptive routing and soft source influence are compared with fixed routing/influence and hard-clamped "
            f"observations. Checkpoints at epochs {checkpoints} are selected only by meta-fit macro-user categorical "
            "cross-entropy (CE). Frequency baselines use the same rating information.",
            "Lower CE, Brier score and expected-rating RMSE are better. Ranking uses P(rating>=4); its liked nDCG "
            "excludes users without held-out likes. It is distinct from the assignment's all-observed ranking, "
            "which remains a diagnostic in the aggregate artifact." +
            (" Inference retains TRAIN-only context and masks TRAIN+validation items for ranking." if final else "")],
        "evidence_stage": "held-out test" if final else "development",
        "tables": [{"caption": f"Mean {'held-out test' if final else 'development'} metrics across seeds; macro averages over users",
                    "columns": ["Predictor", "CE", "Brier", "RMSE", "Liked nDCG"],
                    "widths": [.36, .15, .15, .15, .19], "rows": rows}],
        "discussion": direction +
            "This comparison tests predictive performance of revisable evidence influence, not emotional understanding. "
            "Allocated parameter shapes and training budgets are matched; hard clamping deactivates the fidelity gate, "
            "so equal allocated size does not imply equal active capacity. " +
            ("Test splits share users and a dataset, limiting generalization. " if final else
             "Shared users and previously examined validation data limit generalization. ") +
            "Categorical likelihood and liked-item ranking answer different "
            "questions and cannot establish better all-observed recommendations. Learned routing has precedents "
            "in Set Transformer (Lee et al., 2019, PMLR 97:3744-3753); reaction/diffusion networks also predate "
            "this experiment. Custom implementation does not establish novelty or state-of-the-art performance.",
    }
    supplemental = {"protocol": protocol, "aggregates": evidence}
    if final:
        supplemental["final_evaluation"] = verification
    return _supplement(content, directory, "adaptive", ["aggregates.json", "manifest.json" if final else "protocol.json"],
                       section, supplemental)


def add_joint_findings(content, directory, frozen=None, _convergence=False):
    """Add a loss-alignment ablation of the same field; combine field appendices when possible."""
    directory = Path(directory)
    final = frozen is not None
    if final:
        evidence, protocol, verification = _read_frozen_field(directory, frozen)
    else:
        evidence, protocol = read_json(directory / "aggregates.json"), read_json(directory / "protocol.json")
    from package_project import check_aggregate_only
    check_aggregate_only({"protocol": protocol, "aggregates": evidence}, "joint")
    is_convergence = protocol.get("study_kind") == "post_v1_convergence_sensitivity"
    if is_convergence != _convergence:
        raise ValueError("Use the explicit convergence input for post-v1 budget sensitivity evidence")
    required = {"schema_version": 1, "stage": "development", "test_read": False,
                "training_objective": "joint_recorded_item_rating_multinomial",
                "selection_metric": "meta_fit_macro_joint_nll"}
    adapters = protocol.get("ranking_adapters")
    supported_adapters = [{"all_observed": "logsumexp(all5logits)", "liked_record": "logsumexp(rating4,5logits)"},
                          {"all_observed": "logsumexp(all five rating logits)", "liked_record": "logsumexp(rating 4 and 5 logits)"}]
    if (any(protocol.get(key) != value for key, value in required.items()) or evidence.get("test_read") is not final or
            protocol.get("training_categories", protocol.get("rating_categories")) != [1, 2, 3, 4, 5] or
            adapters not in supported_adapters):
        raise ValueError("Joint evidence requires the declared recorded-item/rating protocol and verified evaluation stage")
    checkpoints = _checkpoint_label(protocol)
    seeds = evidence.get("seeds", {})
    if not seeds or set(map(str, protocol.get("seeds", []))) != set(seeds):
        raise ValueError("Joint protocol and completed aggregate seeds must match")
    rows = []
    labels = [("adaptive", "Joint adaptive"), ("fixed_flow", "Joint fixed flow")]
    for name, label in [("event_global_category", "Event/global ratings"), ("event_item_category", "Event/item ratings"),
                        ("event_item_user_tilt", "Event/item/user tilt")]:
        if all(name in seed["models"] for seed in seeds.values()):
            labels.append((name, label))
    for name, label in labels:
        values = [seed["models"][name] for seed in seeds.values()]
        metrics = [statistics.mean(_finite_metric(row["joint"], "macro_joint_nll") for row in values),
                   statistics.mean(_finite_metric(row["categorical"], "macro_cross_entropy") for row in values),
                   statistics.mean(_finite_metric(row["ranking"]["all_observed_adapter"]["all_observed"], "ndcg@10") for row in values),
                   statistics.mean(_finite_metric(row["ranking"]["liked_record_adapter"]["liked_ratings"], "ndcg@10") for row in values)]
        rows.append([label] + [f"{value:.4f}" for value in metrics])
    stage = "held-out test" if final else "development"
    denominator_note = "the liked endpoint excludes users without held-out likes."
    denominators = [seeds[seed]["models"]["adaptive"]["ranking"]["all_observed_adapter"].get("denominators")
                    for seed in sorted(seeds)]
    if final and all(denominators):
        counts = ", ".join(f"{row['all_observed_users']}/{row['liked_ratings_users']}" for row in denominators)
        denominator_note = (f"all-observed/liked users by seed: {counts}. "
                            "Only liked metrics exclude users without held-out likes.")
    joint_table = {"caption": f"Joint objective: {stage} means; NLL and rating CE lower is better",
                   "columns": ["Predictor", "Joint NLL", "Rating CE", "Obs nDCG", "Like nDCG"],
                   "widths": [.36, .16, .16, .16, .16], "rows": rows}
    section = {
        "title": "Appendix: Task 2 field objective alignment",
        "evidence_stage": stage,
        "paragraphs": [
            (f"Frozen held-out evidence ({len(seeds)} overlapping seeds); test never selects models or checkpoints. " if final else
             f"Exploratory development evidence ({len(seeds)} overlapping seeds); the project test stays closed. ") +
            "The same categorical recurrent field is trained with a joint softmax over available movie-rating pairs. "
            "Categorical evidence sends messages through learned global ports; recurrent routing and soft source "
            "influence update item states before a five-logit decoder. All five rating categories remain inputs.",
            "Observed-record ranking uses logsumexp over all five logits; liked-record ranking uses logsumexp "
            "over categories 4 and 5. Conditional rating probabilities normalize the five logits within an item. " +
            ("Inference retains TRAIN-only rating context and masks TRAIN+validation items from ranking; " if final else
             "All training observations are masked from ranking; ") +
            denominator_note,
            f"Joint checkpoints {checkpoints} minimize meta-fit macro-user joint negative log-likelihood (NLL). "
            "Adaptive and fixed-flow variants share the planned budget. Only the original meta-fit results choose a checkpoint."],
        "tables": [joint_table],
        "discussion": "Joint normalization can learn relative recorded-item mass; per-item conditional cross-entropy "
            "alone cannot identify that mass. This is a loss-alignment ablation of the same field, not a new architecture "
            "or an ensemble. Joint NLL and conditional rating CE have different sample spaces and must not be compared "
            "as if they were the same loss. Both ranking adapters must be reported even when one looks stronger. "
            "Observed recording behavior is not satisfaction, and liked-record mass differs from P(rating>=4|item). " +
            ("Test splits share users and a dataset, limiting generalization. " if final else
             "The reused validation cohorts remain exploratory. ") + "The recurrent routing mechanism has prior art "
            "in Set Transformer (Lee et al., 2019, PMLR 97:3744-3753); neither custom code nor objective alignment "
            "establishes novelty or state-of-the-art performance.",
    }
    adaptive = content.get("supplemental_evidence", {}).get("adaptive")
    if adaptive:
        if set(adaptive["aggregates"]["seeds"]) != set(seeds):
            raise ValueError("Combined field comparison requires matching completed seed sets")
        if adaptive["aggregates"].get("test_read") is not final:
            raise ValueError("Combined field comparison requires the same evaluation stage")
        conditional_rows = []
        for name, label in [("adaptive", "Conditional adaptive"), ("fixed_flow", "Conditional fixed flow"),
                            ("hard_clamp", "Conditional hard clamp"), ("global_histogram", "Global frequencies"),
                            ("item_histogram", "Item frequencies"), ("item_user_product", "Item/user product")]:
            values = [seed["models"][name] for seed in adaptive["aggregates"]["seeds"].values()]
            metrics = [statistics.mean(_finite_metric(row["categorical"], "macro_cross_entropy") for row in values),
                       statistics.mean(_finite_metric(row["ranking"]["all_observed"], "ndcg@10") for row in values),
                       statistics.mean(_finite_metric(row["ranking"]["liked_ratings"], "ndcg@10") for row in values)]
            conditional_rows.append([label] + [f"{value:.4f}" for value in metrics])
        conditional_checkpoints = _checkpoint_label(adaptive["protocol"])
        section["paragraphs"].append(
            f"The independent conditional track selects epochs {conditional_checkpoints} by meta-fit rating CE; "
            "its two ranking diagnostics use P(rating>=4|item). Missing observations are unknown in both tracks. "
            "Fixed flow caches initial routes and source gates. Conditional hard-clamp pins observed states "
            "and deactivates gate parameters; equal allocated size is not equal active capacity.")
        section["tables"].insert(0, {"caption": f"Conditional objective: {stage} means; ranking uses its own readout",
                                     "columns": ["Predictor", "Rating CE", "Obs nDCG", "Like nDCG"],
                                     "widths": [.46, .18, .18, .18], "rows": conditional_rows})
    supplemental = {"protocol": protocol, "aggregates": evidence}
    if final:
        supplemental["final_evaluation"] = verification
    _supplement(content, directory, "joint", ["aggregates.json", "manifest.json" if final else "protocol.json"], section, supplemental)
    if adaptive:
        content["appendices"] = [s for s in content["appendices"] if s.get("study") != "adaptive"]
        if final and not _convergence and content.get("sections"):
            joint_rows = [seed["models"]["adaptive"] for seed in seeds.values()]
            fixed_rows = [seed["models"]["fixed_flow"] for seed in seeds.values()]
            conditional_rows = [seed["models"]["adaptive"] for seed in adaptive["aggregates"]["seeds"].values()]
            conditional_ce = statistics.mean(row["categorical"]["macro_cross_entropy"] for row in conditional_rows)
            conditional_rank = statistics.mean(row["ranking"]["all_observed"]["ndcg@10"] for row in conditional_rows)
            joint_rank = statistics.mean(row["ranking"]["all_observed_adapter"]["all_observed"]["ndcg@10"] for row in joint_rows)
            wins = sum(row["ranking"]["all_observed_adapter"]["all_observed"]["ndcg@10"] >
                       fixed["ranking"]["all_observed_adapter"]["all_observed"]["ndcg@10"]
                       for row, fixed in zip(joint_rows, fixed_rows))
            content["sections"][1]["discussion"] = (
                "Coefficients reflect correlated score associations, not causal credit. Cross-model groups show "
                "whether sparse-user or tail-item performance differs from aggregate accuracy; activity is not "
                "demographic identity. The separate frozen field experiments expose objective alignment: "
                f"conditional training gives rating CE {conditional_ce:.3f} but observed nDCG {conditional_rank:.3f}; "
                f"joint recorded-event training gives observed nDCG {joint_rank:.3f}. Adaptive routing beats fixed "
                f"flow in {wins}/{len(seeds)} overlapping splits on that ranking metric. The appendix retains both "
                "objectives, count baselines and relevance definitions. Training budgets do not establish convergence. "
                "These are measured predictive behaviors, not evidence of emotional understanding or novelty.")
    return content


def add_joint_convergence_findings(content, directory, frozen=None):
    """Retain the original joint comparison and label the later budget decision separately."""
    original = content.get("supplemental_evidence", {}).get("joint")
    if not original:
        raise ValueError("Convergence reporting requires the original joint-field evidence first")
    separate = add_joint_findings({}, directory, frozen, _convergence=True)
    extension = separate["supplemental_evidence"]["joint"]
    if (set(original["aggregates"]["seeds"]) != set(extension["aggregates"]["seeds"]) or
            original["aggregates"].get("test_read") != extension["aggregates"].get("test_read")):
        raise ValueError("Original and convergence comparisons require matching seeds and evaluation stages")
    section = separate["appendices"][0]
    section["title"] = "Appendix: Task 2 joint-field budget sensitivity"
    old_budget = max(original["protocol"]["checkpoints"])
    new_budget = max(extension["protocol"]["checkpoints"])
    original_rows = []
    for name, label in [("adaptive", "adaptive"), ("fixed_flow", "fixed flow")]:
        values = [seed["models"][name] for seed in original["aggregates"]["seeds"].values()]
        metrics = [statistics.mean(_finite_metric(row["joint"], "macro_joint_nll") for row in values),
                   statistics.mean(_finite_metric(row["categorical"], "macro_cross_entropy") for row in values),
                   statistics.mean(_finite_metric(row["ranking"]["all_observed_adapter"]["all_observed"], "ndcg@10") for row in values),
                   statistics.mean(_finite_metric(row["ranking"]["liked_record_adapter"]["liked_ratings"], "ndcg@10") for row in values)]
        original_rows.append([f"{old_budget}ep {label}"] + [f"{value:.4f}" for value in metrics])
    table = section["tables"][0]
    table["rows"][0][0], table["rows"][1][0] = f"{new_budget}ep adaptive", f"{new_budget}ep fixed flow"
    table["rows"] = original_rows + table["rows"]
    section["paragraphs"] = [
        f"Separate post-v1 budget sensitivity ({section['evidence_stage']}). The original {old_budget}-epoch "
        f"study is retained. After its meta-fit curves were still improving at the boundary, a single {new_budget}-epoch "
        "cap was declared. The extension reuses the same splits, model, initialization seeds and training episodes; "
        "it is an exploratory follow-up, not independent confirmation.",
        f"Checkpoints {_checkpoint_label(extension['protocol'])} are selected only by meta-fit macro-user joint NLL. "
        "Row labels give maximum training budgets, not selected epochs. Both recurrent and fixed-flow variants "
        "receive the same budget. Count baselines are unchanged controls.",
        "All five rating categories remain evidence. Observed-record ranking sums mass over five categories; "
        "liked-record ranking sums categories 4 and 5. Conditional rating CE normalizes within each item. " +
        ("Frozen inference retains TRAIN-only context and masks TRAIN+validation for ranking." if frozen else
         "Development inference uses TRAIN-only context and masks every training observation.")]
    section["discussion"] = (
        "A larger budget can change the apparent cost or benefit of a mechanism; the original fixed-budget result "
        "must remain visible. This follow-up tests budget sensitivity of the same objective, not a new architecture. "
        "A selected checkpoint at the last allowed epoch does not prove convergence or universal architectural "
        "failure. No budget will be chosen from held-out outcomes. Joint NLL and conditional rating CE have "
        "different denominators. Repeated splits share users and movies, so consistent differences across seeds "
        "would still not establish independent dataset replication, emotional understanding or state-of-the-art performance.")
    filenames = ["aggregates.json", "manifest.json" if frozen else "protocol.json"]
    return _supplement(content, Path(directory), "joint-convergence", filenames, section, extension)


def add_final_field_comparisons(content, directory):
    """Bind strong-reference comparisons to the actual primary and field aggregates."""
    directory = Path(directory)
    aggregate = read_json(directory / "aggregate.json")
    paired = read_json(directory / "per-seed-comparisons.json")
    manifest = read_json(directory / "manifest.json")
    from package_project import check_aggregate_only
    for value in (aggregate, paired, manifest):
        check_aggregate_only(value, "field-comparisons")
    if (manifest.get("status") != "complete" or manifest.get("selection_performed") is not False or
            manifest.get("comparisons_per_seed") != 24):
        raise ValueError("Field comparisons require a complete, fixed 24-contrast analysis")
    for filename in ["aggregate.json", "per-seed-comparisons.json"]:
        if manifest.get("output_sha256", {}).get(filename) != digest(directory / filename):
            raise ValueError("Field comparison artifact hash mismatch")
    seeds = set(content.get("final_evaluation", {}).get("seeds", {}))
    if (not seeds or seeds != set(map(str, aggregate["seeds"])) or
            seeds != set(map(str, manifest["seeds"])) or
            seeds != {str(row["seed"]) for row in paired} or len(paired) != len(seeds) or
            aggregate.get("cross_seed_confidence_interval") is not None):
        raise ValueError("Field comparisons require the same complete primary splits without an independent-seed CI")
    studies = {"conditional": "adaptive", "joint100": "joint", "joint400": "joint-convergence"}
    supplements = content.get("supplemental_evidence", {})
    for track, study in studies.items():
        values = supplements.get(study, {}).get("aggregates", {})
        if values.get("test_read") is not True or set(values.get("seeds", {})) != seeds:
            raise ValueError("Field comparisons require all three frozen held-out field studies")
        verification = supplements[study].get("final_evaluation", {})
        for key in ["frozen_manifest_sha256", "evaluation_manifest_sha256"]:
            if not verification.get(key) or manifest.get("inputs", {}).get(track, {}).get(key) != verification[key]:
                raise ValueError("Field comparison intervals refer to a different frozen evaluation")
    required = {f"{track}/{endpoint}/adaptive-minus-{reference}"
                for track in studies for endpoint in ["all_observed", "liked_ratings"]
                for reference in ["fixed_flow", "EASE", "SLIMElastic", "PositiveEASE"]}
    if set(aggregate["comparisons"]) != required or any(set(row["comparisons"]) != required for row in paired):
        raise ValueError("Field comparisons omit a predeclared track, endpoint or reference")
    def field_value(track, seed, model, endpoint):
        ranking = supplements[studies[track]]["aggregates"]["seeds"][seed]["models"][model]["ranking"]
        if track != "conditional":
            ranking = ranking["all_observed_adapter" if endpoint == "all_observed" else "liked_record_adapter"]
        return _finite_metric(ranking[endpoint], "ndcg@10")
    def equal(actual, expected):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Field comparison metrics disagree with their frozen report evidence")
    for key in sorted(required):
        row = aggregate["comparisons"][key]
        track, endpoint, reference = row["track"], row["endpoint"], row["reference"]
        if key != f"{track}/{endpoint}/adaptive-minus-{reference}" or row["seeds"] != len(seeds):
            raise ValueError("Field comparison semantic role mismatch")
        candidates, references = [], []
        for report in paired:
            seed, pair = str(report["seed"]), report["comparisons"][key]
            candidate = field_value(track, seed, "adaptive", endpoint)
            other = _finite_metric(pair["reference_metrics"], "ndcg@10")
            equal(pair["candidate_metrics"]["ndcg@10"], candidate)
            if reference == "fixed_flow":
                equal(other, field_value(track, seed, "fixed_flow", endpoint))
            elif endpoint == "all_observed" and reference in {"EASE", "SLIMElastic"}:
                name = content["canonical_roles"][seed]["expert:" + reference]
                equal(other, content["aggregate_evidence"][seed][name]["aggregate"]["ndcg@10"])
            equal(pair["mean_ndcg_difference"], candidate - other)
            interval = pair["bonferroni_family_interval_95"]
            if len(interval) != 2 or not all(math.isfinite(x) for x in interval) or interval[0] > interval[1]:
                raise ValueError("Invalid field comparison family interval")
            candidates.append(candidate)
            references.append(other)
        equal(row["mean_candidate_ndcg"], statistics.mean(candidates))
        equal(row["mean_reference_ndcg"], statistics.mean(references))
        equal(row["mean_ndcg_difference"], statistics.mean(candidates) - statistics.mean(references))
    def contrast(track, endpoint, reference):
        return aggregate["comparisons"][f"{track}/{endpoint}/adaptive-minus-{reference}"]
    original = contrast("joint100", "all_observed", "EASE")
    extended = contrast("joint400", "all_observed", "EASE")
    slim = contrast("joint400", "all_observed", "SLIMElastic")
    routing = contrast("joint400", "all_observed", "fixed_flow")
    liked = contrast("joint400", "liked_ratings", "PositiveEASE")
    interval_key = "joint400/all_observed/adaptive-minus-fixed_flow"
    better = sum(row["comparisons"][interval_key]["bonferroni_family_interval_95"][0] > 0 for row in paired)
    worse = sum(row["comparisons"][interval_key]["bonferroni_family_interval_95"][1] < 0 for row in paired)
    conditional = contrast("conditional", "all_observed", "EASE")["mean_candidate_ndcg"]
    content["sections"][1]["discussion"] = (
        "Coefficients reflect correlated score associations, not causal credit. The group tables distinguish "
        "sparse-user and tail-item outcomes; activity is not demographic identity. Conditional rating training "
        f"gives observed nDCG {conditional:.3f}. Joint normalization learns recorded-item mass: adaptive nDCG "
        f"{original['mean_candidate_ndcg']:.3f} at the 100-epoch cap and {extended['mean_candidate_ndcg']:.3f} "
        f"at the 400-epoch cap, versus locked EASE {extended['mean_reference_ndcg']:.3f} and SLIM {slim['mean_reference_ndcg']:.3f}. "
        f"At that cap, adaptive-minus-fixed nDCG is {routing['mean_ndcg_difference']:+.3f}; adjusted paired intervals "
        f"favor adaptive in {better}/{len(seeds)} splits and fixed flow in {worse}/{len(seeds)}. On the separate "
        f"liked-record endpoint, joint400 scores {liked['mean_candidate_ndcg']:.3f} versus liked-selected "
        f"PositiveEASE {liked['mean_reference_ndcg']:.3f}. The appendix retains original controls and the "
        "post-v1 budget extension. Intervals use approximate Bonferroni-adjusted bootstrap bounds for 24 contrasts within each split; "
        "there is no independent-split confidence interval. Objective alignment and custom code do not establish novelty.")
    content["field_comparisons"] = {"aggregate": aggregate, "per_seed": paired, "manifest": manifest}
    for filename in ["aggregate.json", "per-seed-comparisons.json", "manifest.json"]:
        content.setdefault("source_sha256", {})[f"{directory.name}/{filename}"] = digest(directory / filename)
    return content


def add_negative_findings(content, directory, protocol_path=None):
    """Read the fixed-protocol rejection-information audit; retain both query modes."""
    directory = Path(directory)
    evidence = read_json(directory / "aggregate.json")
    from package_project import check_aggregate_only
    check_aggregate_only(evidence, "negative")
    plan, summary = evidence["plan"], evidence["summary"]
    if plan.get("test_read") is not False or plan.get("test_evaluated") is not False:
        raise ValueError("Negative audit must explicitly retain a closed project test")
    if not summary or set(summary) != set(map(str, plan["seeds"])) or set(summary) != set(evidence["seeds"]):
        raise ValueError("Negative audit requires every planned seed to be complete")
    if (plan.get("fixed_penalty") != 250 or plan.get("replicates") != 5 or
            plan.get("penalties") != [50, 250, 1000] or plan.get("selection") != "context meta-fit liked nDCG@10"):
        raise ValueError("Negative audit does not match the declared fixed and equal-grid comparison")
    protocol_path = Path(protocol_path) if protocol_path else Path(__file__).with_name("NEGATIVE_INFORMATION_PROTOCOL.md")
    if digest(protocol_path) != plan.get("protocol_sha256"):
        raise ValueError("Negative audit protocol hash mismatch")
    labels = {"positive": "Positive only", "plain": "Plain dislikes", "surprise": "Surprise weights",
              "weight_permutation": "Permuted weights", "placement_null": "Placement null"}
    rows = []
    means = {}
    for name, label in labels.items():
        branches = [f"{name}_{i}" for i in range(5)] if name in {"weight_permutation", "placement_null"} else [name]
        values = []
        for budget, mode in [("fixed", "context"), ("selected", "context"),
                             ("fixed", "full_train"), ("selected", "full_train")]:
            mean = statistics.mean(statistics.mean(_finite_metric(seed[budget][mode][branch], "liked_ndcg10")
                                                   for branch in branches) for seed in summary.values())
            means[name, budget, mode] = mean
            values.append(f"{mean:.4f}")
        rows.append([label] + values)
    plain_delta = means["plain", "fixed", "context"] - means["positive", "fixed", "context"]
    placement_delta = means["plain", "fixed", "context"] - means["placement_null", "fixed", "context"]
    surprise_delta = means["surprise", "fixed", "context"] - means["weight_permutation", "fixed", "context"]
    intervals = [row["contrasts"]["fixed"]["context"]["plain_minus_placement_null"]
                 ["descriptive_user_bootstrap_interval_95"] for row in evidence["seeds"].values()]
    if any(len(interval) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in interval)
           or interval[0] > interval[1] for interval in intervals):
        raise ValueError("Negative audit intervals must contain two ordered finite bounds")
    interval_note = ("Every plain-minus-placement interval contains zero; a stable placement benefit is unestablished. "
                     if all(lo <= 0 <= hi for lo, hi in intervals) else
                     "Per-seed intervals must be read with the conditional-control caveats below. ")
    liked_counts = "/".join(str(summary[seed]["fixed"]["context"]["positive"]["liked_users"])
                            for seed in sorted(summary))
    section = {
        "title": "Appendix: Task 2 rejection-information audit",
        "paragraphs": [
            f"Supplemental development audit, {len(summary)} overlapping seeds. Likes >=4, dislikes <=2, rating 3 neutral. "
            "Context is 80% of each user's training history; the disjoint remainder supplies decoder targets. "
            "Teacher and decoder inputs use context only. All originally observed training items are masked from ranking.",
            f"Liked-endpoint users by seed: {liked_counts}; users without held-out likes are excluded. "
            "This cohort differs from the main assignment comparison.",
            "Primary queries use context only; full-history queries reuse the fitted decoder and add known history, "
            "changing input length. Fixed ridge is 250. Secondary equal-grid [50,250,1000] selection uses only "
            "context meta-fit liked nDCG; all five replicates share one control-branch setting.",
            "The placement null preserves observed support, positives and user/item dislike counts, including within "
            "the context-defined fitting stratum. Weight permutations preserve each user's dislike identities and "
            "weight multiset. The table averages per-user control metrics, never control scores."],
        "tables": [{"caption": "Liked nDCG@10: mean across seeds and all five control replicates",
                    "columns": ["Branch", "Context fixed", "Context grid", "Full fixed", "Full grid"],
                    "widths": [.36, .16, .16, .16, .16], "rows": rows}],
        "discussion": f"At fixed ridge, plain dislikes change context nDCG by {plain_delta:+.4f} versus positive-only "
            f"and {placement_delta:+.4f} versus the placement null. Surprise weighting changes it by "
            f"{surprise_delta:+.4f} versus permuted weights. These are descriptive differences, not significance claims. "
            + interval_note +
            "The aggregate artifact retains per-seed user-bootstrap intervals, every control replicate, movement "
            "diagnostics and known-dislike slot rates. Structural zeros may disconnect the switch space; movement "
            "does not establish uniform sampling or convergence. Bootstrap intervals condition on five draws. "
            "Full-history results cannot replace the primary context test. Liked relevance differs from the assignment's "
            "all-observed endpoint; low known-dislike rates cannot certify satisfaction. Reused validation users and "
            "existing hard-negative weighting research (LAGCL4Rec, Findings EMNLP 2025, paper 61) preclude "
            "independent-confirmation or novelty claims. Switch-space limitations follow Rapallo and Yoshida "
            "(2010, arXiv:0905.4841).",
    }
    _supplement(content, directory, "negative", ["aggregate.json"], section, evidence)
    content["source_sha256"]["code/NEGATIVE_INFORMATION_PROTOCOL.md"] = digest(protocol_path)
    return content


def validate_content(content):
    if content.get("status") not in {"DRAFT", "FINAL"}:
        raise ValueError("Report status must be DRAFT or FINAL")
    if content["status"] == "FINAL":
        members = content.get("members", [])
        if not content.get("group") or len(members) != 5 or len(set(members)) != 5 or any(not str(m).strip() for m in members):
            raise ValueError("FINAL requires the actual group number and five distinct contributor names")
        if not content.get("final_evaluation") or content.get("evidence_stage") != "held-out test":
            raise ValueError("FINAL requires verified frozen held-out evaluation provenance")
    if content.get("schema_version") != 1:
        raise ValueError("Unsupported content schema")
    if content.get("group") is not None and not re.fullmatch(r"[1-9][0-9]*", str(content["group"])):
        raise ValueError("Group must be a positive integer")
    if [s.get("task") for s in content.get("sections", [])] != [1, 2, 3]:
        raise ValueError("Exactly one section each for tasks 1, 2 and 3 is required")
    appendices = content.get("appendices", [])
    if len(appendices) > 3 or any(s.get("task") != 2 or s.get("study") not in {"adaptive", "joint", "joint-convergence", "negative"} for s in appendices):
        raise ValueError("At most one Task 2 appendix per supplemental study is supported")
    if len({s["study"] for s in appendices}) != len(appendices):
        raise ValueError("Duplicate supplemental study appendix")
    for section in content["sections"] + appendices:
        if word_count(section["discussion"]) > 200:
            raise ValueError(f"Task {section['task']} discussion exceeds 200 words")
        for table in section.get("tables", []):
            if not table["columns"] or any(len(row) != len(table["columns"]) for row in table["rows"]):
                raise ValueError("Table rows must match the number of columns")
            if any(not isinstance(cell, str) for row in [table["columns"], *table["rows"]] for cell in row):
                raise ValueError("Table values must be explicit strings")
            if "widths" in table and (len(table["widths"]) != len(table["columns"]) or
                    any(w <= 0 for w in table["widths"]) or abs(sum(table["widths"]) - 1) > 1e-6):
                raise ValueError("Table widths must be positive shares adding to one")
    return {str(s["task"]): word_count(s["discussion"]) for s in content["sections"]} | {
        "appendix-" + s["study"]: word_count(s["discussion"]) for s in appendices}


def find_fonts(font_dir=None):
    candidates = [Path(font_dir)] if font_dir else [
        Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
        Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts"),
    ]
    for directory in candidates:
        for regular, bold in [("Times New Roman.ttf", "Times New Roman Bold.ttf"),
                              ("times.ttf", "timesbd.ttf"), ("Times_New_Roman.ttf", "Times_New_Roman_Bold.ttf")]:
            if (directory / regular).is_file() and (directory / bold).is_file():
                return directory / regular, directory / bold
    raise FileNotFoundError("Times New Roman regular and bold fonts are required; use --font-dir")


def render_pdf(content, output, font_dir=None):
    """Render a cover, three task pages and optional one-page study appendices."""
    validate_content(content)
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
    width = PAGE_WIDTH - 2 * MARGIN
    page_bottoms = []
    sections = content["sections"] + content.get("appendices", [])
    total_pages = len(sections) + 1

    def wrap(text, limit, strong=False):
        lines = []
        for paragraph in str(text).split("\n"):
            current = ""
            for word in paragraph.split():
                if measure.get_text_width_height_descent(word, fonts[strong], False)[0] > limit:
                    raise ValueError(f"Unbreakable text exceeds column width: {word!r}")
                candidate = f"{current} {word}".strip()
                if current and measure.get_text_width_height_descent(candidate, fonts[strong], False)[0] > limit:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
        return lines

    class Page:
        def __init__(self, number, title, stage=None):
            self.figure = Figure(figsize=(PAGE_WIDTH / 72, PAGE_HEIGHT / 72))
            self.y = PAGE_HEIGHT - MARGIN
            self.number = number
            self.stage = stage or content.get("evidence_stage", "Development evidence")
            self.paragraph(content["status"] + " | " + title, strong=True)
            self.y -= LEADING / 2

        def line(self, text, x=MARGIN, strong=False):
            if self.y < MARGIN + LEADING:
                raise ValueError(f"Page {self.number} overflows at 12pt; shorten the content")
            self.figure.text(x / PAGE_WIDTH, self.y / PAGE_HEIGHT, text,
                             fontproperties=fonts[strong], va="top", parse_math=False)

        def paragraph(self, text, strong=False):
            for line in wrap(text, width, strong):
                self.line(line, strong=strong)
                self.y -= LEADING
            self.y -= LEADING / 2

        def table(self, table):
            self.paragraph(table["caption"], strong=True)
            count = len(table["columns"])
            shares = table.get("widths", [0.44] + [0.56 / (count - 1)] * (count - 1) if count > 1 else [1.0])
            for idx, row in enumerate([table["columns"], *table["rows"]]):
                cells = [wrap(cell, width * share - 8, strong=idx == 0) for cell, share in zip(row, shares)]
                height = max(map(len, cells)) * LEADING + 4
                if self.y - height < MARGIN + LEADING:
                    raise ValueError(f"Table on page {self.number} overflows; shorten the content")
                x, start = MARGIN, self.y
                for lines, share in zip(cells, shares):
                    for i, line in enumerate(lines):
                        self.y = start - i * LEADING
                        self.line(line, x=x, strong=idx == 0)
                    x += width * share
                self.y = start - height
            self.y -= LEADING / 2

        def finish(self, pdf):
            self.figure.text(MARGIN / PAGE_WIDTH, 25 / PAGE_HEIGHT,
                             f"{content['status']} | {self.stage} | {self.number}/{total_pages}",
                             fontproperties=fonts[False])
            pdf.savefig(self.figure)
            page_bottoms.append(round(self.y, 2))

    metadata = {"Title": content["title"] + " (" + content["status"] + ")", "Author": "; ".join(content["members"]),
                "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
                "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc), "Creator": "report.py"}
    with matplotlib.rc_context({"pdf.fonttype": 42, "pdf.compression": 6, "text.usetex": False}):
        with PdfPages(output, metadata=metadata) as pdf:
            page = Page(1, content["title"])
            page.paragraph("DSAIT4335 Recommender Systems")
            page.paragraph("Group: " + (str(content["group"]) if content["group"] is not None else "NOT SUPPLIED"))
            page.paragraph(f"Contributors supplied ({len(content['members'])}/5): " +
                           ("; ".join(content["members"]) or "NOT SUPPLIED"))
            page.paragraph(content["evidence_label"], strong=True)
            for note in content.get("cover_notes", []):
                page.paragraph(note)
            if content.get("references"):
                page.paragraph("Method references", strong=True)
                for reference in content["references"]:
                    page.paragraph(reference)
            page.finish(pdf)
            for number, section in enumerate(sections, start=2):
                page = Page(number, section["title"],
                            "Supplemental " + section.get("evidence_stage", "development") if section.get("study") else None)
                for paragraph in section.get("paragraphs", []):
                    page.paragraph(paragraph)
                for table in section.get("tables", []):
                    page.table(table)
                page.paragraph("Discussion", strong=True)
                page.paragraph(section["discussion"])
                page.finish(pdf)
    return {"pages": total_pages, "main_task_pages": 3, "appendix_pages": len(content.get("appendices", [])),
            "font": "Times New Roman", "font_size": FONT_SIZE,
            "line_spacing": LINE_SPACING, "page_bottoms_pt": page_bottoms,
            "font_sha256": {"regular": digest(regular), "bold": digest(bold)}}


def render_markdown(content):
    lines = ["# " + content["status"] + ": " + content["title"], "", "Group: " + str(content["group"] or "NOT SUPPLIED"), "",
             f"Contributors supplied ({len(content['members'])}/5): " +
             ("; ".join(content["members"]) or "NOT SUPPLIED"), "", content["evidence_label"], ""]
    lines += [note + "\n" for note in content.get("cover_notes", [])]
    if content.get("references"):
        lines += ["**Method references**", ""] + [r + "\n" for r in content["references"]]
    for section in content["sections"] + content.get("appendices", []):
        lines += ["## " + section["title"], ""]
        lines += [p + "\n" for p in section.get("paragraphs", [])]
        for table in section.get("tables", []):
            lines += ["**" + table["caption"] + "**", "", "| " + " | ".join(table["columns"]) + " |",
                      "| " + " | ".join(["---"] * len(table["columns"])) + " |"]
            lines += ["| " + " | ".join(row) + " |" for row in table["rows"]]
            lines.append("")
        lines += ["**Discussion**", "", section["discussion"], ""]
    return "\n".join(lines)


def generate_report(content, out, font_dir=None):
    counts = validate_content(content)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    layout = render_pdf(content, out / "report.pdf", font_dir)
    (out / "report.md").write_text(render_markdown(content), encoding="utf-8")
    (out / "report-content.json").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "status": content["status"], "group": content["group"],
                "contributors_supplied": len(content["members"]) == 5, "layout": layout,
                "discussion_words": counts, "source_sha256": content.get("source_sha256", {}),
                "final_evaluation": content.get("final_evaluation"),
                "generator_sha256": digest(__file__),
                "artifact_sha256": {name: digest(out / name) for name in
                                    ["report.pdf", "report.md", "report-content.json"]}}
    (out / "REPORT-MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence", type=Path, help="aggregate evidence with selected models and source provenance")
    source.add_argument("--content-json", type=Path, help="edited report-content.json with three task sections")
    source.add_argument("--final-results", type=Path, help="completed final_evaluate.py output (aggregate metrics only)")
    source.add_argument("--final-seed", nargs=3, action="append", metavar=("SEED", "RESULTS", "FROZEN"),
                        help="repeat for primary seeds 2026,2027,2028; each real evaluation is verified independently")
    parser.add_argument("--frozen", type=Path, help="matching freeze bundle, required with --final-results")
    parser.add_argument("--finalize", action="store_true", help="mark reviewed held-out report FINAL; requires five names and group")
    parser.add_argument("--exception-evidence", type=Path, help="separate development exception study containing summary.json")
    parser.add_argument("--adaptive-evidence", type=Path, help="curated categorical development study; adds one appendix page")
    parser.add_argument("--adaptive-final-results", type=Path, help="completed categorical held-out aggregate output")
    parser.add_argument("--adaptive-frozen", type=Path, help="matching sealed categorical bundle; metadata only is read")
    parser.add_argument("--joint-evidence", type=Path, help="same-field joint-objective study; combines with the categorical appendix")
    parser.add_argument("--joint-final-results", type=Path, help="completed joint-field held-out aggregate output")
    parser.add_argument("--joint-frozen", type=Path, help="matching sealed joint-field bundle; metadata only is read")
    parser.add_argument("--joint-convergence-evidence", type=Path, help="separate post-v1 development budget sensitivity appendix")
    parser.add_argument("--joint-convergence-final-results", type=Path, help="completed post-v1 held-out aggregate output")
    parser.add_argument("--joint-convergence-frozen", type=Path, help="matching sealed convergence bundle; metadata only is read")
    parser.add_argument("--negative-evidence", type=Path, help="curated rejection-information audit; adds one appendix page")
    parser.add_argument("--field-comparisons", type=Path, help="verified final field comparisons against locked strong references")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group", help="actual positive integer group number; otherwise marked NOT SUPPLIED")
    parser.add_argument("--member", action="append", default=[], help="actual contributor; may be repeated")
    parser.add_argument("--font-dir", type=Path, help="directory containing Times New Roman regular and bold TTFs")
    args = parser.parse_args()
    if args.final_seed:
        if args.frozen:
            parser.error("--final-seed supplies its own frozen directory for each seed")
        content = build_multi_final_content(args.final_seed, args.group, args.member)
    elif args.final_results:
        if not args.frozen:
            parser.error("--final-results requires --frozen")
        content = build_final_content(args.final_results, args.frozen, args.group, args.member)
    else:
        if args.frozen:
            parser.error("--frozen is only used with --final-results")
        content = read_json(args.content_json) if args.content_json else build_content(args.evidence, args.group, args.member)
    if args.content_json and (args.group is not None or args.member):
        parser.error("For --content-json, edit group and members in that file")
    if args.exception_evidence:
        if args.final_results or args.final_seed:
            parser.error("Add development exception findings through edited content to preserve final coefficient tables")
        add_exception_findings(content, args.exception_evidence)
    if args.adaptive_evidence:
        if args.adaptive_final_results:
            parser.error("Choose development or held-out categorical evidence, not both")
        add_adaptive_findings(content, args.adaptive_evidence)
    if bool(args.adaptive_final_results) != bool(args.adaptive_frozen):
        parser.error("--adaptive-final-results and --adaptive-frozen are required together")
    if args.adaptive_final_results:
        add_adaptive_findings(content, args.adaptive_final_results, args.adaptive_frozen)
    if args.joint_evidence:
        if args.joint_final_results:
            parser.error("Choose development or held-out joint evidence, not both")
        add_joint_findings(content, args.joint_evidence)
    if bool(args.joint_final_results) != bool(args.joint_frozen):
        parser.error("--joint-final-results and --joint-frozen are required together")
    if args.joint_final_results:
        add_joint_findings(content, args.joint_final_results, args.joint_frozen)
    if args.joint_convergence_evidence:
        if args.joint_convergence_final_results:
            parser.error("Choose development or held-out convergence evidence, not both")
        add_joint_convergence_findings(content, args.joint_convergence_evidence)
    if bool(args.joint_convergence_final_results) != bool(args.joint_convergence_frozen):
        parser.error("--joint-convergence-final-results and --joint-convergence-frozen are required together")
    if args.joint_convergence_final_results:
        add_joint_convergence_findings(content, args.joint_convergence_final_results, args.joint_convergence_frozen)
    if args.negative_evidence:
        add_negative_findings(content, args.negative_evidence)
    if args.field_comparisons:
        add_final_field_comparisons(content, args.field_comparisons)
    if args.finalize:
        content["status"] = "FINAL"
        content["evidence_label"] = "Frozen held-out evaluation"
    manifest = generate_report(content, args.out, args.font_dir)
    print(json.dumps({"report": str(args.out / "report.pdf"), "status": content["status"],
                      "pages": manifest["layout"]["pages"], "discussion_words": manifest["discussion_words"]}))


if __name__ == "__main__":
    main()
