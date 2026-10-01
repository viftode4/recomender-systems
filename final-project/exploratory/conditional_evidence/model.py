"""Small anonymous rating-graph reader and matched evidence summaries.

No file I/O, split construction, optimizer, or relevance labels live here.
The query's complete bank row is excluded, including from population counts.
All candidates must be absent from the supplied, already masked context.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .native_scramble import scramble_history as _scramble_history


FEATURE_NAMES = (
    "log_context_count", "log_donor_support", "donor_support_fraction",
    "log_weighted_support", "weighted_support_fraction",
    "mean_supporter_weight", "log_kish_count", "log_pattern_count",
    "pattern_coverage",
)
VARIANTS = ("raw", "scrambled", "summary")


def _categories(value: Any, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != ndim or array.dtype.kind not in "biuf"
            or not np.isfinite(array).all()
            or not np.isin(array, np.arange(6)).all()):
        raise ValueError(f"{name} must contain integer rating categories 0..5")
    return np.array(array, dtype=np.uint8, copy=True)


@dataclass(frozen=True)
class EvidenceBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    rating: torch.Tensor
    query_idx: torch.Tensor
    candidate_idx: torch.Tensor
    donor_idx: torch.Tensor
    donor_graph_idx: torch.Tensor
    features9: torch.Tensor
    candidate_ids: torch.Tensor
    donor_rows: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any]

    def to(self, device: Any) -> "EvidenceBatch":
        return replace(self, **{
            field.name: getattr(self, field.name).to(device)
            for field in fields(self) if isinstance(getattr(self, field.name), torch.Tensor)
        })

    @property
    def num_graphs(self) -> int:
        return int(self.candidate_ids.numel())


def combine_batches(batches: Sequence[EvidenceBatch]) -> EvidenceBatch:
    """Pack independent query/candidate graphs, preserving autograd inputs."""
    if not batches:
        raise ValueError("at least one batch is required")
    node_offset = graph_offset = 0
    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("x", "edge_index", "rating", "query_idx", "candidate_idx",
                             "donor_idx", "donor_graph_idx", "features9", "candidate_ids")
    }
    donor_rows: list[tuple[int, ...]] = []
    records: list[dict[str, int]] = []
    for batch in batches:
        for name in parts:
            tensor = getattr(batch, name)
            if name in ("edge_index", "query_idx", "candidate_idx", "donor_idx"):
                tensor = tensor + node_offset
            elif name == "donor_graph_idx":
                tensor = tensor + graph_offset
            parts[name].append(tensor)
        node_offset += int(batch.x.shape[0])
        graph_offset += batch.num_graphs
        donor_rows.extend(batch.donor_rows)
        records.extend(batch.metadata["graphs"])
    tensors = {name: torch.cat(values, dim=1 if name == "edge_index" else 0)
               for name, values in parts.items()}
    return EvidenceBatch(**tensors, donor_rows=tuple(donor_rows), metadata={
        **_metadata(records), "materialized_node_count": int(tensors["x"].shape[0]),
        "materialized_edge_count": int(tensors["rating"].numel())})


def _metadata(records: list[dict[str, int]]) -> dict[str, Any]:
    names = ("donor_count", "node_count", "edge_count", "history_edge_count",
             "scramble_attempts", "scramble_successes")
    return {"graphs": records, **{name: sum(row[name] for row in records) for name in names}}


def scramble_history(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    """Exact native/Python same-rating switches, bounded by 20*history edges.

    The portable reference specifies every uint64 random draw and sequential
    proposal. Both backends preserve row/column rating margins. Only the donor
    history panel is supplied, so candidate and query edges cannot be changed.
    """
    return _scramble_history(matrix, seed)


class EvidenceBank:
    """Read-only categorical TRAIN bank; row numbers select data, not parameters."""

    def __init__(self, categories: Any, padding: int = 0,
                 allowed_rows: Any | None = None) -> None:
        array = _categories(categories, 2, "categories")
        if array.shape[1] < 1 or padding != 0 or np.any(array[:, 0]):
            raise ValueError("catalog column zero must be empty PAD")
        if allowed_rows is None:
            allowed = np.ones(array.shape[0], dtype=bool)
        else:
            allowed = np.asarray(allowed_rows)
            if allowed.shape != (array.shape[0],) or allowed.dtype.kind != "b":
                raise ValueError("allowed_rows must be one Boolean per bank row")
            allowed = allowed.copy()
        # Disallowed records disappear from both retrieval and item statistics.
        array[~allowed] = 0
        array.setflags(write=False)
        allowed.setflags(write=False)
        self.categories, self.allowed_rows, self.padding = array, allowed, padding
        self.binary = array > 0
        self.row_counts = self.binary.sum(axis=1, dtype=np.int64)
        self.item_counts = self.binary.sum(axis=0, dtype=np.int64)
        for value in (self.binary, self.row_counts, self.item_counts):
            value.setflags(write=False)

    def build_context_query(self, context: Any, exclude_row: int,
                            donor_rows: Sequence[int]) -> EvidenceBatch:
        """One context-only graph for a utility policy's frozen representations.

        The candidate-role node is isolated with degree zero, contains no item
        identity, and must not be used by the policy. Donor order is preserved.
        """
        context = _categories(context, 1, "context")
        users, items = self.categories.shape
        if context.shape != (items,) or context[0] != 0:
            raise ValueError("context must match the catalog with empty PAD")
        if not isinstance(exclude_row, (int, np.integer)) or not -1 <= exclude_row < users:
            raise ValueError("exclude_row must be -1 or a valid bank row")
        donors = np.asarray(donor_rows)
        if not donors.size:
            donors = np.empty(0, dtype=np.int64)
        if (donors.ndim != 1 or donors.dtype.kind not in "iu"
                or np.any(donors < 0) or np.any(donors >= users)
                or np.unique(donors).size != donors.size
                or np.any(donors == exclude_row) or not self.allowed_rows[donors].all()):
            raise ValueError("context donors must be distinct allowed non-query rows")
        donors = donors.astype(np.int64)
        history = np.flatnonzero(context)
        h, d = history.size, donors.size
        population = self.item_counts.copy()
        if exclude_row >= 0:
            population -= self.binary[exclude_row]
        count, query, candidate = h + d + 2, h, h + 1
        donor_nodes = np.arange(h + 2, count, dtype=np.int64)
        x = np.zeros((count, 5), dtype=np.float32)
        x[:h, 2] = x[query, 0] = x[candidate, 3] = 1
        x[donor_nodes, 1] = 1
        x[:h, 4] = np.log1p(population[history])
        x[query, 4] = np.log1p(h)
        x[donor_nodes, 4] = np.log1p(self.row_counts[donors])
        panel = self.categories[np.ix_(donors, history)]
        dr, hi = np.nonzero(panel)
        src = np.concatenate((np.full(h, query, dtype=np.int64), donor_nodes[dr]))
        dst = np.concatenate((np.arange(h, dtype=np.int64), hi))
        value = np.concatenate((context[history], panel[dr, hi])).astype(np.int64)
        edges = np.stack((np.concatenate((src, dst)), np.concatenate((dst, src))))
        record = {"donor_count": int(d), "node_count": int(count),
                  "edge_count": int(2 * src.size), "history_edge_count": int(dr.size),
                  "scramble_attempts": 0, "scramble_successes": 0}
        return EvidenceBatch(
            x=torch.from_numpy(x), edge_index=torch.from_numpy(edges),
            rating=torch.from_numpy(np.concatenate((value, value))),
            query_idx=torch.tensor([query]), candidate_idx=torch.tensor([candidate]),
            donor_idx=torch.from_numpy(donor_nodes), donor_graph_idx=torch.zeros(d, dtype=torch.long),
            features9=torch.zeros((1, 9)), candidate_ids=torch.tensor([-1]),
            donor_rows=(tuple(donors.tolist()),), metadata=_metadata([record]),
        )

    def build_query(self, context: Any, exclude_row: int, candidate_ids: Any,
                    variant: str = "raw", seed: int = 0,
                    extra_donor: Mapping[int, Sequence[int]] | None = None,
                    materialize_graph: bool = True) -> EvidenceBatch:
        """Pack candidate graphs and summaries from precisely the same donors.

        A summary caller may set ``materialize_graph=False`` to omit unused
        tensors. Semantic graph/evidence counts remain in metadata; explicit
        materialized counts distinguish actual allocation from that evidence.
        """
        context = _categories(context, 1, "context")
        users, items = self.categories.shape
        if context.shape != (items,) or context[0] != 0:
            raise ValueError("context must match the catalog with empty PAD")
        if not isinstance(exclude_row, (int, np.integer)) or not -1 <= exclude_row < users:
            raise ValueError("exclude_row must be -1 or a valid bank row")
        candidates = np.asarray(candidate_ids)
        if candidates.ndim == 1 and not candidates.size:
            candidates = np.empty(0, dtype=np.int64)
        if (candidates.ndim != 1 or candidates.dtype.kind not in "iu"
                or np.any(candidates <= 0) or np.any(candidates >= items)
                or np.unique(candidates).size != candidates.size):
            raise ValueError("candidate_ids must be distinct non-PAD catalog indices")
        if np.any(context[candidates]):
            raise ValueError("candidate records must be hidden from context")
        if variant not in VARIANTS or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("unknown variant or invalid nonnegative integer seed")
        if not isinstance(materialize_graph, bool) or (not materialize_graph and variant != "summary"):
            raise ValueError("only the summary arm can omit graph materialization")
        history = np.flatnonzero(context)
        h = history.size
        available = self.allowed_rows.copy()
        if exclude_row >= 0:
            available[exclude_row] = False
        population = self.item_counts.copy()
        if exclude_row >= 0:
            population -= self.binary[exclude_row]
        overlaps = self.binary[:, history].sum(axis=1, dtype=np.int64)
        denominator = np.sqrt(np.asarray(h * self.row_counts, dtype=np.float64))
        cosine = np.divide(overlaps, denominator, out=np.zeros(users, dtype=np.float64),
                           where=denominator > 0)
        order = np.lexsort((np.arange(users, dtype=np.int64), -cosine))
        order = order[available[order]]
        global_rows = order[:8]
        rest = order[8:]
        contexts = self.categories[:, history]
        node_arrays, sources, targets, ratings = [], [], [], []
        queries, candidate_nodes, donor_nodes, donor_graphs = [], [], [], []
        summaries, selected_rows, records = [], [], []
        offset = 0
        for graph, candidate in enumerate(candidates.tolist()):
            supporting = rest[self.binary[rest, candidate]][:8]
            donors = np.concatenate((global_rows, supporting))
            if extra_donor is not None:
                extras = np.asarray(extra_donor.get(candidate, ()))
                if extras.size:
                    if (extras.ndim != 1 or extras.dtype.kind not in "iu"
                            or np.any(extras < 0) or np.any(extras >= users)
                            or not available[extras].all()):
                        raise ValueError("extra donors must be allowed, non-query bank rows")
                    known = set(donors.tolist())
                    addition = []
                    for donor in extras.tolist():
                        if donor not in known:
                            addition.append(donor)
                            known.add(donor)
                    donors = np.concatenate((donors, np.asarray(addition, dtype=np.int64)))
            panel = contexts[donors].copy()
            candidate_ratings = self.categories[donors, candidate]
            summaries.append(self._summary(panel, candidate_ratings, cosine[donors] ** 2, h))
            scramble = {"attempts": 0, "successes": 0}
            if variant == "scrambled":
                # SeedSequence avoids Python hash randomization; graph order and
                # candidate chunk boundaries cannot alter a candidate's control.
                local_seed = np.random.SeedSequence([int(seed), int(candidate)]).generate_state(1)[0]
                panel, scramble = scramble_history(panel, int(local_seed))
            d = donors.size
            count = h + d + 2
            history_edges = int(np.count_nonzero(panel))
            total_edges = 2 * (h + history_edges + np.count_nonzero(candidate_ratings))
            selected_rows.append(tuple(int(v) for v in donors))
            records.append({"donor_count": int(d), "node_count": int(count),
                            "edge_count": int(total_edges), "history_edge_count": history_edges,
                            "scramble_attempts": scramble["attempts"],
                            "scramble_successes": scramble["successes"]})
            if not materialize_graph:
                continue
            query_node, candidate_node = h, h + 1
            donor_local = np.arange(h + 2, count, dtype=np.int64)
            x = np.zeros((count, 5), dtype=np.float32)
            x[:h, 2] = 1
            x[query_node, 0] = x[candidate_node, 3] = 1
            x[donor_local, 1] = 1
            x[:h, 4] = np.log1p(population[history])
            x[query_node, 4] = np.log1p(h)
            x[candidate_node, 4] = np.log1p(population[candidate])
            x[donor_local, 4] = np.log1p(self.row_counts[donors])
            dr, hi = np.nonzero(panel)
            support = np.flatnonzero(candidate_ratings)
            src = np.concatenate((np.full(h, query_node, dtype=np.int64), donor_local[dr],
                                  donor_local[support]))
            dst = np.concatenate((np.arange(h, dtype=np.int64), hi,
                                  np.full(support.size, candidate_node, dtype=np.int64)))
            value = np.concatenate((context[history], panel[dr, hi], candidate_ratings[support]))
            node_arrays.append(x)
            sources.append(np.concatenate((src, dst)) + offset)
            targets.append(np.concatenate((dst, src)) + offset)
            ratings.append(np.concatenate((value, value)).astype(np.int64))
            queries.append(query_node + offset)
            candidate_nodes.append(candidate_node + offset)
            donor_nodes.append(donor_local + offset)
            donor_graphs.append(np.full(d, graph, dtype=np.int64))
            offset += count
        def cat(values: list[np.ndarray], dtype: Any, shape: tuple[int, ...] = (0,)) -> torch.Tensor:
            return torch.from_numpy(np.concatenate(values) if values else np.empty(shape, dtype=dtype))
        return EvidenceBatch(
            x=cat(node_arrays, np.float32, (0, 5)),
            edge_index=torch.stack((cat(sources, np.int64), cat(targets, np.int64))),
            rating=cat(ratings, np.int64),
            query_idx=torch.tensor(queries, dtype=torch.long),
            candidate_idx=torch.tensor(candidate_nodes, dtype=torch.long),
            donor_idx=cat(donor_nodes, np.int64), donor_graph_idx=cat(donor_graphs, np.int64),
            features9=torch.from_numpy(np.asarray(summaries, dtype=np.float32).reshape(-1, 9)),
            candidate_ids=torch.from_numpy(candidates.astype(np.int64, copy=True)),
            donor_rows=tuple(selected_rows), metadata={**_metadata(records),
                "materialized_node_count": int(offset),
                "materialized_edge_count": int(sum(value.size for value in ratings))},
        )

    @staticmethod
    def _summary(panel: np.ndarray, candidate_ratings: np.ndarray,
                 weights: np.ndarray, h: int) -> np.ndarray:
        supporter = candidate_ratings > 0
        support = int(supporter.sum())
        features = np.zeros(9, dtype=np.float64)
        features[:3] = (np.log1p(h), np.log1p(support), support / max(1, panel.shape[0]))
        if not h or not weights.size or not weights.sum():
            return features
        positive_weights = weights[supporter]
        mass = positive_weights.sum()
        features[3:6] = (np.log1p(mass), mass / weights.sum(), mass / max(1, support))
        if mass:
            features[6] = np.log1p(mass * mass / np.square(positive_weights).sum())
            intersection = panel[supporter] > 0
            overlap = intersection.sum(axis=1)
            factors = np.divide(positive_weights, np.sqrt(overlap),
                                out=np.zeros_like(positive_weights), where=overlap > 0)
            pattern = factors @ intersection
            norm = np.dot(pattern, pattern)
            features[7] = np.log1p(mass * mass / norm) if norm else 0
            features[8] = np.count_nonzero(pattern) / h
        return features


class RatingLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_transform = nn.Linear(width, width)
        self.relations = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(5))

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor,
                rating: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(hidden)
        if rating.numel():
            transformed = torch.stack([relation(hidden) for relation in self.relations])
            message = transformed[rating - 1, edge_index[0]]
            message = message / degree[rating - 1, edge_index[1], None]
            aggregate.index_add_(0, edge_index[1], message)
        return torch.nn.functional.silu(self.self_transform(hidden) + aggregate)


class GraphReader(nn.Module):
    """Three normalized rating layers, shared across anonymous local graphs."""

    def __init__(self, width: int = 16, seed: int = 0) -> None:
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        self.config = {"width": width, "seed": int(seed), "layers": 3, "input_features": 5}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.input_projection = nn.Linear(5, width)
            self.layers = nn.ModuleList(RatingLayer(width) for _ in range(3))
            self.head = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))

    def node_embeddings(self, batch: EvidenceBatch) -> torch.Tensor:
        if batch.query_idx.numel() != batch.num_graphs:
            raise ValueError("GraphReader requires materialized graph tensors")
        hidden = self.input_projection(batch.x)
        nodes = batch.x.shape[0]
        degree = torch.bincount((batch.rating - 1) * nodes + batch.edge_index[1],
                                minlength=5 * nodes).reshape(5, nodes).clamp_min(1)
        degree = degree.to(dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index, batch.rating, degree)
        return hidden

    def encode(self, batch: EvidenceBatch) -> torch.Tensor:
        hidden = self.node_embeddings(batch)
        donor_mean = hidden.new_zeros((batch.num_graphs, self.config["width"]))
        if batch.donor_idx.numel():
            donor_mean.index_add_(0, batch.donor_graph_idx, hidden[batch.donor_idx])
            count = torch.bincount(batch.donor_graph_idx, minlength=batch.num_graphs).clamp_min(1)
            donor_mean = donor_mean / count[:, None]
        return torch.cat((hidden[batch.query_idx], hidden[batch.candidate_idx], donor_mean), dim=1)

    def forward(self, batch: EvidenceBatch) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)


class SummaryReader(nn.Module):
    def __init__(self, width: int = 16, seed: int = 0) -> None:
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        self.config = {"width": width, "seed": int(seed), "input_features": 9}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch: EvidenceBatch) -> torch.Tensor:
        return self.network(batch.features9).squeeze(-1)
