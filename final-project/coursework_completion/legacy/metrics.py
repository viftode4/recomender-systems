"""Independent binary ranking metrics; original item IDs, no RecBole imports."""
import math


def ranking_metrics(recommended, relevant, k):
    """Precision denominator is k; NDCG ideal length is min(k, positives)."""
    if k < 1 or len(recommended) != k:
        raise ValueError("Expected exactly k recommendations, k >= 1")
    if len(set(recommended)) != k:
        raise ValueError("Duplicate recommendations")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Evaluation users must have held-out positives")
    hits = [int(item in relevant) for item in recommended]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {
        f"precision@{k}": sum(hits) / k,
        f"recall@{k}": sum(hits) / len(relevant),
        f"ndcg@{k}": dcg / ideal,
        f"mrr@{k}": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
        f"hit@{k}": float(any(hits)),
    }


def evaluate(recommendations, truth, history, catalog, counts, k):
    """Macro-average over all held-out users. Novelty uses training-only counts.

    Novelty = mean -log2((count(item)+1)/(training interactions+catalog size)).
    Coverage denominator is the full real-item catalog, including unseen items.
    """
    catalog = set(catalog)
    if not truth or set(recommendations) != set(truth):
        raise ValueError("Recommendation users must match held-out users exactly")
    if not catalog or set(counts) - catalog or any(c < 0 for c in counts.values()):
        raise ValueError("Invalid training counts or catalog")
    per_user, exposed = {}, set()
    denominator = sum(counts.values()) + len(catalog)
    for user, recs in recommendations.items():
        if set(recs) - catalog or set(truth[user]) - catalog:
            raise ValueError("Unknown item")
        if set(recs) & set(history.get(user, ())):
            raise ValueError("Seen item recommended")
        row = ranking_metrics(recs, truth[user], k)
        row[f"novelty@{k}"] = sum(
            -math.log2((counts.get(item, 0) + 1) / denominator) for item in recs
        ) / k
        per_user[user] = row
        exposed.update(recs)
    aggregate = {key: sum(row[key] for row in per_user.values()) / len(per_user)
                 for key in next(iter(per_user.values()))}
    aggregate[f"coverage@{k}"] = len(exposed) / len(catalog)
    return {"users": len(per_user), "aggregate": aggregate, "per_user": per_user}
