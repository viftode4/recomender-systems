"""Independent ranking and locked-reference evaluation for conditional evidence.

Metric functions are pure: the runner/data layer must verify its global selection
seal before loading development labels. This module's only file reader loads
hash-pinned TRAIN/meta-selected reference predictions, never outcome files.
"""
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_NAMES = ("EASEexpanded", "categorical", "SLIM")
ACTIVITY_NAMES = ("sparse", "medium", "dense")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(f"{value.dtype}:{value.shape}:".encode())
    result.update(memoryview(value).cast("B"))
    return result.hexdigest()


def identity_digest(users, items):
    return hashlib.sha256(json.dumps({"users": list(users), "items": list(items)},
                                    sort_keys=True).encode()).hexdigest()


def _identities(users, items):
    users, items = list(users), list(items)
    if (not users or len(set(users)) != len(users)
            or any(not isinstance(user, str) for user in users)
            or len(items) < 2 or items[0] != "[PAD]"
            or len(set(items)) != len(items)
            or any(not isinstance(item, str) for item in items)):
        raise ValueError("Require unique string identities and catalog PAD at index zero")
    return users, items


def _observed(train, users, items):
    if isinstance(train, np.ndarray):
        values = np.asarray(train)
        if (values.shape != (len(users), len(items)) or values.dtype.kind not in "biuf"
                or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values > 5) or np.any(values != np.floor(values))):
            raise ValueError("TRAIN must be a finite binary/category matrix in original ID order")
        observed = values > 0
    else:
        observed = np.zeros((len(users), len(items)), dtype=bool)
        user_index, item_index = {v: n for n, v in enumerate(users)}, {v: n for n, v in enumerate(items)}
        pairs = ((user, item) for user, history in train.items() for item in history) if isinstance(train, Mapping) else train
        for user, item in pairs:
            if user not in user_index or item not in item_index or observed[user_index[user], item_index[item]]:
                raise ValueError("Unknown or duplicate TRAIN pair")
            observed[user_index[user], item_index[item]] = True
    if observed[:, 0].any():
        raise ValueError("PAD cannot be observed")
    return observed


def stable_ranking(scores, observed, k=10):
    """Rank every eligible item, with exact ties resolved by catalog position."""
    scores, observed = np.asarray(scores), np.asarray(observed, dtype=bool)
    if (scores.ndim != 1 or observed.shape != scores.shape or not np.isfinite(scores).all()
            or isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError("Invalid scores, mask or ranking cutoff")
    eligible = np.flatnonzero(~observed)
    eligible = eligible[eligible != 0]
    if len(eligible) < k:
        raise ValueError("Not enough eligible candidates for the declared cutoff")
    return eligible[np.lexsort((eligible, -scores[eligible]))]


def _truth_for(truth, chosen):
    # Never inspect another cohort's item values, including malformed values.
    if isinstance(truth, Mapping):
        return {user: set(truth[user]) for user in chosen}
    result = {user: set() for user in chosen}
    for user, item in truth:
        if user in result:
            result[user].add(item)
    return result


def _user_metrics(ranked, relevant, k):
    top = ranked[:k]
    hits = [int(item in relevant) for item in top]
    total = len(relevant)
    ideal = sum(1 / math.log2(position + 2) for position in range(min(k, total)))
    return {f"ndcg@{k}": sum(hit / math.log2(position + 2) for position, hit in enumerate(hits)) / ideal,
            f"recall@{k}": sum(hits) / total, f"precision@{k}": sum(hits) / k,
            f"mrr@{k}": next((1 / (position + 1) for position, hit in enumerate(hits) if hit), 0.),
            "mrr_full_catalog": next((1 / (position + 1) for position, item in enumerate(ranked)
                                      if item in relevant), 0.),
            f"hit@{k}": float(any(hits))}


def _mean_rows(rows):
    if not rows:
        return None
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _groups(rows, rankings, truth, activity, head, items, k):
    users = list(rows)
    activity_groups = {}
    for label in ACTIVITY_NAMES:
        members = [user for user in users if activity[user] == label]
        activity_groups[label] = {"users": len(members), "aggregate": _mean_rows([rows[user] for user in members])}
    item_groups = {}
    for label, members in (("head", head), ("tail", set(items[1:]) - head)):
        positives = hits = slots = 0
        conditional = []
        for user in users:
            relevant = truth[user] & members
            found = len(set(rankings[user][:k]) & relevant)
            positives += len(relevant)
            hits += found
            slots += len(set(rankings[user][:k]) & members)
            if relevant:
                conditional.append(found / len(relevant))
        item_groups[label] = {"catalog_items": len(members), "users_with_positives": len(conditional),
                              "positive_pairs": positives, "hits": hits, "recommendation_slots": slots,
                              f"macro_conditional_recall@{k}": float(np.mean(conditional)) if conditional else None,
                              f"micro_recall@{k}": hits / positives if positives else None,
                              "slot_share": slots / (len(users) * k) if users else None}
    return {"activity": activity_groups, "items": item_groups}


def evaluate_scores(scores, users, items, train, truth, chosen_users, ratings=None, k=10):
    """Evaluate an explicitly supplied cohort; return public aggregates + private rows.

    ``truth`` maps user tokens to item tokens (or is a pair iterable). ``ratings``
    is optional, either a (user,item)->category mapping or triples. Only ratings
    of the chosen cohort's supplied truth are used. All rating categories 1..5
    are relevant to the primary endpoint; >=4 defines the secondary endpoint.
    The runner must store ``per_user`` only in its ignored local run directory.
    """
    users, items = _identities(users, items)
    scores = np.asarray(scores)
    if scores.shape != (len(users), len(items)) or not np.isfinite(scores).all():
        raise ValueError("Require finite full-catalog scores in original ID order")
    requested = list(chosen_users)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(users):
        raise ValueError("Require a nonempty unique evaluation cohort")
    chosen = [user for user in users if user in set(requested)]
    observed = _observed(train, users, items)
    histories = {user: {items[item] for item in np.flatnonzero(observed[row])} for row, user in enumerate(users)}
    selected_truth = _truth_for(truth, chosen)
    if any(not values or values - set(items[1:]) or values & histories[user]
           for user, values in selected_truth.items()):
        raise ValueError("Require nonempty eligible held-out truth for each evaluation user")
    index = {user: row for row, user in enumerate(users)}
    rankings = {user: [items[item] for item in stable_ranking(scores[index[user]], observed[index[user]], k)]
                for user in chosen}
    counts = observed.sum(axis=0)
    thresholds = np.quantile(observed.sum(axis=1), [1/3, 2/3])
    activity = {user: ACTIVITY_NAMES[int(np.searchsorted(thresholds, observed[row].sum(), side="right"))]
                for row, user in enumerate(users)}
    head_count = math.ceil(.2 * (len(items) - 1))
    head = {items[item] for item in sorted(range(1, len(items)), key=lambda item: (-int(counts[item]), items[item]))[:head_count]}
    item_index = {item: row for row, item in enumerate(items)}
    all_rows = {user: _user_metrics(rankings[user], selected_truth[user], k) for user in chosen}
    for user in chosen:
        all_rows[user][f"novelty@{k}"] = float(np.mean([
            -math.log2((counts[item_index[item]] + 1) / (int(counts.sum()) + len(items) - 1))
            for item in rankings[user][:k]]))
    liked_truth, liked_rows = {}, {}
    low_hits = rated_hits = 0
    if ratings is not None:
        allowed = {(user, item) for user in chosen for item in selected_truth[user]}
        if isinstance(ratings, Mapping):
            retained = {pair: ratings[pair] for pair in allowed}
        else:
            retained = {}
            for user, item, rating in ratings:
                if user in selected_truth and item in selected_truth[user]:
                    if (user, item) in retained:
                        raise ValueError("Duplicate held-out category")
                    retained[user, item] = rating
        if set(retained) != allowed:
            raise ValueError("Missing evaluation category")
        if any(not np.isfinite(value) or value not in (1, 2, 3, 4, 5) for value in retained.values()):
            raise ValueError("Evaluation categories must be integer ratings 1..5")
        for user in chosen:
            liked = {item for item in selected_truth[user] if retained[user, item] >= 4}
            if liked:
                liked_truth[user] = liked
                liked_rows[user] = _user_metrics(rankings[user], liked, k)
            for item in rankings[user][:k]:
                if item in selected_truth[user]:
                    rated_hits += 1
                    low_hits += int(retained[user, item] <= 2)
    aggregate = _mean_rows(list(all_rows.values()))
    aggregate[f"coverage@{k}"] = len({item for user in chosen for item in rankings[user][:k]}) / (len(items) - 1)
    public = {"all_observed": aggregate, "liked_ratings": _mean_rows(list(liked_rows.values())),
        "groups": {"all_observed": _groups(all_rows, rankings, selected_truth, activity, head, items, k),
                   "liked_ratings": _groups(liked_rows, rankings, liked_truth, activity, head, items, k) if liked_rows else None},
        "denominators": {"all_observed_users": len(chosen), "all_observed_positive_pairs": sum(map(len, selected_truth.values())),
                         "liked_users": len(liked_rows) if ratings is not None else None,
                         "liked_positive_pairs": sum(map(len, liked_truth.values())) if ratings is not None else None,
                         "catalog_items": len(items) - 1, "recommendation_slots": len(chosen) * k},
        "known_low_rating_hits_per_slot": low_hits / (len(chosen) * k) if ratings is not None else None,
        "known_low_rating_hits_among_rated_hits": low_hits / rated_hits if ratings is not None and rated_hits else None,
        "definitions": {"k": k, "score_ties": "original catalog position",
            "grouping_population_users": len(users), "activity_thresholds": thresholds.tolist(),
            "activity": "TRAIN counts; linear terciles; ties in upper group",
            "head": "top ceil(20% real catalog) TRAIN count, then lexical item token",
            "liked_groups": "same TRAIN definitions; users with at least one liked held-out target",
            "missing_records": "not verified dislikes", "stage": "caller-supplied cohort"}}
    return {"public": public, "per_user": {"all_observed": all_rows, "liked_ratings": liked_rows}}


def meta_ndcg(scores, users, items, train, truth, chosen_users, k=10):
    return evaluate_scores(scores, users, items, train, truth, chosen_users, k=k)["public"]["all_observed"][f"ndcg@{k}"]


def neighbor_scores(train, k=16):
    """Top-k cosine donor baseline; entire query row is ineligible as a donor.

    All TRAIN observations are binary evidence. Ties use original donor order;
    zero-weight donors contribute nothing. Weighted item indicators are divided
    by total selected cosine weight; empty support gives exactly zero.
    """
    values = np.asarray(train)
    if (values.ndim != 2 or values.dtype.kind not in "biuf" or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 5) or np.any(values != np.floor(values))
            or np.any(values[:, 0]) or isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError("Invalid TRAIN matrix or donor budget")
    binary = (values > 0).astype(np.float64)
    norms = np.sqrt(binary.sum(1))
    denominator = norms[:, None] * norms[None, :]
    similarity = np.divide(binary @ binary.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    result = np.zeros_like(binary)
    rows = np.arange(len(binary))
    for query in rows:
        eligible = rows[rows != query]
        donors = eligible[np.lexsort((eligible, -similarity[query, eligible]))[:k]]
        weights = similarity[query, donors]
        if weights.sum() > 0:
            result[query] = weights @ binary[donors] / weights.sum()
    result[:, 0] = 0
    return result


def _safe_path(root, name):
    root, name = Path(root).resolve(), Path(name)
    path = (root / name).resolve()
    if (name.is_absolute() or not path.is_relative_to(root)
            or any(part.lower().startswith(("test", "final")) for part in name.parts)):
        raise ValueError("Forbidden reference artifact path")
    return path


def load_references(seed, inputs, reference_root, users, items):
    """Load only sealed selected arrays and metadata; never read old DEV metrics.

    ``inputs`` is the exact categorical-reconstruction input signature (or the
    new data helper's returned dictionary containing it under ``signature``).
    This verifies the historical selection barrier, copied prediction payloads,
    identities and source hashes without calling the old result evaluator.
    """
    users, items = _identities(users, items)
    signature = inputs.get("signature", inputs)
    root, directory = Path(reference_root), Path(reference_root) / str(seed)
    manifest_path, frozen_path = root / "manifest.json", root / "SELECTIONS-FROZEN.json"
    manifest, frozen = json.loads(manifest_path.read_text()), json.loads(frozen_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("test_read", True)
            or manifest.get("test_evaluated", True) or manifest["input_signatures"].get(str(seed)) != signature
            or manifest["selection_freeze_sha256"] != digest(frozen_path)
            or frozen.get("status") != "all_selections_frozen" or frozen.get("test_read", True)
            or frozen.get("development_evaluated", True) or seed not in frozen["seeds"]):
        raise ValueError("Reference completion, input signature or selection barrier differs")
    if (digest(root / "protocol.json") != frozen["protocol_sha256"]
            or manifest["source_sha256"] != frozen["source_sha256"]
            or manifest["runtime"] != frozen["runtime"]):
        raise ValueError("Reference protocol/source/runtime chain differs")
    for name, expected in frozen["source_sha256"].items():
        if digest(_safe_path(ROOT, name)) != expected:
            raise ValueError("Historical reference source changed")
    selection_path = directory / "selection-manifest.json"
    if (digest(selection_path) != frozen["selection_manifest_sha256"].get(str(seed))
            or manifest["output_sha256"].get(f"{seed}/selection-manifest.json") != digest(selection_path)):
        raise ValueError("Reference selected manifest changed")
    selected_manifest = json.loads(selection_path.read_text())
    if (selected_manifest.get("status") != "selected" or selected_manifest.get("seed") != seed
            or selected_manifest.get("test_read", True) or selected_manifest.get("development_evaluated", True)
            or selected_manifest["input_signature"] != signature
            or selected_manifest["source_sha256"] != frozen["source_sha256"]
            or selected_manifest["runtime"] != frozen["runtime"]
            or signature["ordered_identity_sha256"] != identity_digest(users, items)):
        raise ValueError("Reference selected identity/source signature differs")

    def checked(name):
        path = _safe_path(directory, name)
        expected = selected_manifest["output_sha256"].get(name)
        if expected is None or digest(path) != expected or manifest["output_sha256"].get(f"{seed}/{name}") != expected:
            raise ValueError("Reference payload hash differs")
        return path

    ids_path, cohorts_path = checked("ids.json"), checked("cohorts.json")
    ids, cohorts = json.loads(ids_path.read_text()), json.loads(cohorts_path.read_text())
    if (digest(ids_path) != signature["ids_sha256"] or ids.get("padding_index") != 0 or ids["items"] != items
            or [user for user in ids["users"] if user in set(users)] != users
            or digest(cohorts_path) != signature["cohort_file_sha256"]
            or set(cohorts["meta_fit"]) & set(cohorts["development"])
            or sorted(cohorts["meta_fit"] + cohorts["development"]) != sorted(users)):
        raise ValueError("Reference catalog/cohort identity differs")
    choices = json.loads(checked("selection.json").read_text())
    scores, metadata = {}, {}
    for name, choice_name in (("EASEexpanded", "binary_expanded"), ("categorical", "categorical")):
        choice = choices[choice_name]
        if (choice.get("selection_cohort") != "reused_meta_fit" or choice.get("selection_metric") != "all_observed_ndcg@10"
                or not choice.get("refit_replay_exact") or not choice.get("score_file_replay_exact")):
            raise ValueError("Reference was not selected on declared meta objective")
        path = checked(choice["scores_file"])
        values = np.load(path, allow_pickle=False)
        if digest(path) != choice["scores_file_sha256"] or array_digest(values) != choice["scores_array_sha256"]:
            raise ValueError("Reference score replay digest differs")
        scores[name] = values
        metadata[name] = {"selection": {key: choice[key] for key in ("candidate_id", "candidate_count", "penalty", "category_ratio")},
                          "scores_sha256": digest(path), "scores_array_sha256": array_digest(values)}
    slim = json.loads(checked("locked-slim.json").read_text())
    pinned_slim = {key: value for key, value in slim.items() if key != "source_path"}
    if (pinned_slim != selected_manifest["locked_slim_source"]
            or pinned_slim != manifest["locked_slim_sources"].get(str(seed))
            or slim["ordered_identity_sha256"] != identity_digest(users, items)
            or slim["ids_file_sha256"] != signature["ids_sha256"]
            or slim["selection"]["selection_cohort"] != "meta_fit"
            or slim["selection"]["selection_metric"] != "meta-fit all-observed nDCG@10"):
        raise ValueError("Locked SLIM selection provenance differs")
    slim_path = checked("locked-slim-scores.npy")
    scores["SLIM"] = np.load(slim_path, allow_pickle=False)
    metadata["SLIM"] = {"selection": slim["selection"], "source_manifest_sha256": slim["manifest_sha256"],
                        "source_scores_sha256": slim["score_file_sha256"], "scores_sha256": digest(slim_path),
                        "scores_array_sha256": array_digest(scores["SLIM"])}
    for values in scores.values():
        if values.shape != (len(users), len(items)) or not np.isfinite(values).all():
            raise ValueError("Reference score shape/nonfinite mismatch")
    metadata["provenance"] = {"manifest_sha256": digest(manifest_path), "selection_freeze_sha256": digest(frozen_path),
        "selection_manifest_sha256": digest(selection_path), "input_signature": signature,
        "outcome_files_read": False, "score_scope": "TRAIN-fitted full-catalog predictions, with historical meta selection"}
    return scores, metadata


def _table(per_seed, names):
    if not per_seed or len(set(names)) != len(names):
        raise ValueError("Require seeds and distinct model names")
    table = {}
    for seed, row in per_seed.items():
        values = {name: float(row[name]) for name in names}
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("Require finite nDCG values in [0,1]")
        table[str(seed)] = values
    if len(table) != len(per_seed):
        raise ValueError("Ambiguous seed identities")
    return table, {name: float(np.mean([row[name] for row in table.values()])) for name in names}


def meta_screen(per_seed, raw, controls, references):
    """Strict equal-seed mean screen; strongest reference is chosen on META only."""
    controls, references = tuple(controls), tuple(references)
    if len(controls) != 2 or not references:
        raise ValueError("Declare exactly two mechanism controls and at least one standalone reference")
    table, means = _table(per_seed, (raw, *controls, *references))
    strongest = max(references, key=lambda name: means[name])
    passed = all(means[raw] > means[name] for name in (*controls, strongest))
    return {"stage": "reused_meta_fit", "passed": passed, "raw": raw, "controls": list(controls),
            "references": list(references), "strongest_reference": strongest, "seeds": list(table),
            "mean_ndcg": means, "per_seed_ndcg": table,
            "rule": "raw strict equal-seed mean exceeds both controls and strongest mean meta reference; exact reference ties follow declared order",
            "interpretation": "conservative resource gate, not proof that adaptive filtering cannot help a failing full reader"}


def development_gate(per_seed, frozen_meta_screen):
    """Assess the predeclared 10% target; never choose a reference using DEV."""
    screen = frozen_meta_screen
    if screen.get("stage") != "reused_meta_fit":
        raise ValueError("Require frozen meta screen")
    expected = meta_screen(screen["per_seed_ndcg"], screen["raw"], screen["controls"], screen["references"])
    if expected != screen:
        raise ValueError("Frozen meta screen arithmetic differs")
    raw, controls, reference = screen["raw"], screen["controls"], screen["strongest_reference"]
    table, means = _table(per_seed, (raw, *controls, reference))
    if set(table) != set(screen["seeds"]):
        raise ValueError("Development and meta seed populations differ")
    gain = means[raw] / means[reference] - 1 if means[reference] > 0 else None
    positive = all(row[raw] > row[reference] for row in table.values())
    mechanism = all(means[raw] > means[name] for name in controls)
    return {"stage": "reused_development", "strongest_meta_reference": reference,
            "mean_ndcg": means, "relative_improvement": gain, "target_relative_improvement": .10,
            "positive_in_every_seed": positive, "positive_mechanism_mean": mechanism,
            "substantial_target_met": bool(gain is not None and gain >= .10 - 1e-12 and positive and mechanism),
            "relative_threshold_numerical_tolerance": 1e-12,
            "fresh_confirmation": False, "interpretation": "descriptive target, not significance or novelty"}


def paired_summary(first, second, metric="ndcg@10", resamples=2000, seed=20260929):
    """Descriptive paired user bootstrap; input maps stay private, result is aggregate."""
    if not first or set(first) != set(second) or isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("Require matching nonempty paired users and positive bootstrap count")
    users = sorted(first)
    values = np.asarray([first[user][metric] - second[user][metric] for user in users], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite paired metric")
    rng, samples = np.random.default_rng(seed), []
    for start in range(0, resamples, 128):
        indices = rng.integers(0, len(values), size=(min(128, resamples - start), len(values)))
        samples.extend(values[indices].mean(1).tolist())
    return {"users": len(values), "metric": metric, "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)), "win_fraction": float(np.mean(values > 0)),
            "tie_fraction": float(np.mean(values == 0)), "loss_fraction": float(np.mean(values < 0)),
            "descriptive_bootstrap_95_percentile": np.quantile(samples, [.025, .975]).tolist(),
            "resamples": resamples, "bootstrap_seed": seed,
            "interpretation": "conditional descriptive user bootstrap; shared fitted models and overlapping splits preclude independent-replication or confirmatory claims"}
