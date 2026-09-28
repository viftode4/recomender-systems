"""One bounded max-product program selected using disjoint inner validation.

No MovieLens, saved recommender scores, attention, or model ensemble is used.
Atoms are supplied relations, not predicates discovered by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class QueryAtoms:
    values: np.ndarray  # candidate x unique evidence identity x supplied atom
    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self):
        x = np.array(self.values, dtype=np.float64, copy=True)
        if x.ndim != 3 or x.shape[:2] != (len(self.candidate_ids), len(self.evidence_ids)):
            raise ValueError("atom axes must match candidate and evidence IDs")
        if x.shape[2] < 1 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("atoms must be finite [0,1] values with at least one atom")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("duplicate candidate identity")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence identity can fabricate a witness")
        if set(self.candidate_ids) & set(self.evidence_ids):
            raise ValueError("candidate cannot occur in its own evidence context")
        x.setflags(write=False)
        object.__setattr__(self, "values", x)

    @property
    def sha256(self):
        h = hashlib.sha256(self.values.tobytes())
        h.update(json.dumps([self.candidate_ids, self.evidence_ids]).encode())
        return h.hexdigest()


def build_atoms(evidence_ids, candidate_ids, n_atoms: int,
                relation: Callable[[str, str, int], float]) -> QueryAtoms:
    """No labels argument. Deduplicate genuine identities before constructing atoms.

    Caller must ensure `relation` uses only this evidence context and learned
    training parameters. A Python callback cannot enforce its closure's provenance.
    """
    evidence = tuple(dict.fromkeys(evidence_ids))
    candidates = tuple(candidate_ids)
    x = np.empty((len(candidates), len(evidence), n_atoms), dtype=float)
    for i, candidate in enumerate(candidates):
        for j, item in enumerate(evidence):
            for k in range(n_atoms):
                x[i, j, k] = relation(candidate, item, k)
    return QueryAtoms(x, candidates, evidence)


@dataclass(frozen=True)
class Tree:
    op: str
    atoms: tuple[int, ...] = ()
    children: tuple["Tree", ...] = ()

    def __post_init__(self):
        if self.op == "atom":
            valid = len(self.atoms) == 1 and not self.children
        elif self.op == "product":
            valid = len(self.atoms) == 2 and not self.children
        elif self.op == "merge":
            valid = len(self.children) == 2 and not self.atoms
        else:
            valid = False
        if not valid or any(type(a) is not int or a < 0 for a in self.atoms):
            raise ValueError("invalid expression")
        if self.op == "product":
            object.__setattr__(self, "atoms", tuple(sorted(self.atoms)))
        if self.op == "merge":
            object.__setattr__(self, "children", tuple(sorted(self.children, key=lambda t: t.key)))

    @property
    def key(self):
        if self.op == "atom":
            return f"a{self.atoms[0]}"
        if self.op == "product":
            return f"distinct(a{self.atoms[0]}*a{self.atoms[1]})"
        return "max(" + ",".join(c.key for c in self.children) + ")"

    @property
    def nodes(self):
        return (3 if self.op == "product" else 1) + sum(c.nodes for c in self.children)

    @property
    def depth(self):
        return 2 if self.op == "product" else 1 + max((c.depth for c in self.children), default=0)

    def to_dict(self):
        return {"op": self.op, "atoms": list(self.atoms),
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, obj):
        return cls(obj["op"], tuple(obj["atoms"]), tuple(cls.from_dict(c) for c in obj["children"]))


def atom(a):
    return Tree("atom", (int(a),))


def product(a, b):
    return Tree("product", (int(a), int(b)))


def merge(a, b):
    return Tree("merge", children=(a, b))


def distinct_product(x: np.ndarray, a: int, b: int):
    """Exact max_{j!=k} x[j,a] x[k,b], linear in number of witnesses."""
    n, evidence, _ = x.shape
    if evidence < 2:
        return np.zeros(n)
    right = x[:, :, b]
    best_index = np.argmax(right, axis=1)
    best = right[np.arange(n), best_index]
    second = np.partition(right, evidence - 2, axis=1)[:, -2]
    compatible = np.broadcast_to(best[:, None], (n, evidence)).copy()
    compatible[np.arange(n), best_index] = second
    return np.max(x[:, :, a] * compatible, axis=1)


def evaluate(tree: Tree, batch: QueryAtoms):
    x = batch.values
    if any(a >= x.shape[2] for a in tree.atoms):
        raise ValueError("unknown atom")
    if tree.op == "atom":
        return np.max(x[:, :, tree.atoms[0]], axis=1) if x.shape[1] else np.zeros(x.shape[0])
    if tree.op == "product":
        return distinct_product(x, *tree.atoms)
    return np.maximum(*(evaluate(c, batch) for c in tree.children))


@dataclass(frozen=True)
class SearchConfig:
    rounds: int = 3
    beam_width: int = 6
    max_nodes: int = 7
    max_depth: int = 3
    max_candidates: int = 320
    size_penalty: float = .003
    calibration_ridge: float = .0001

    def __post_init__(self):
        if min(self.rounds, self.beam_width, self.max_nodes, self.max_depth, self.max_candidates) < 1:
            raise ValueError("positive search bounds required")
        if min(self.size_penalty, self.calibration_ridge) < 0:
            raise ValueError("nonnegative regularization required")


def propose_edits(tree: Tree, n_atoms: int, config: SearchConfig):
    """Pure grammar edits: no data or labels can enter this function."""
    proposed = {tree}
    if tree.op == "atom":
        proposed.update(atom(a) for a in range(n_atoms))
        proposed.update(product(tree.atoms[0], a) for a in range(n_atoms))
    elif tree.op == "product":
        proposed.update(atom(a) for a in tree.atoms)  # delete one premise
        for a in range(n_atoms):
            proposed.add(product(a, tree.atoms[0]))
            proposed.add(product(a, tree.atoms[1]))
    else:
        proposed.update(tree.children)  # actual branch deletion
        for side in range(2):
            for edited in propose_edits(tree.children[side], n_atoms, config):
                children = list(tree.children)
                children[side] = edited
                proposed.add(merge(*children))
    proposed.update(merge(tree, atom(a)) for a in range(n_atoms))
    return tuple(sorted((t for t in proposed if t.nodes <= config.max_nodes and t.depth <= config.max_depth),
                        key=lambda t: (t.nodes, t.key)))


def labels_array(labels, n):
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != (n,) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("finite binary or fractional outcomes required")
    return y


def log_loss(logits, y):
    return float(np.mean(np.logaddexp(0., logits) - y * logits))


@dataclass(frozen=True)
class FittedTree:
    tree: Tree
    bias: float
    scale: float
    fit_loss: float

    def logits(self, batch):
        return self.bias + self.scale * evaluate(self.tree, batch)

    def to_dict(self):
        return {"tree": self.tree.to_dict(), "expression": self.tree.key,
                "bias": self.bias, "scale": self.scale, "fit_loss": self.fit_loss}


def calibrate(tree: Tree, fit: QueryAtoms, labels, ridge=.0001):
    y = labels_array(labels, len(fit.candidate_ids))
    f = evaluate(tree, fit)
    def objective(v):
        logits = v[0] + v[1] * f
        residual = expit(logits) - y
        loss = log_loss(logits, y) + ridge * np.dot(v, v) / 2
        gradient = np.array([residual.mean(), np.mean(residual * f)]) + ridge * v
        return loss, gradient
    result = minimize(objective, np.array([0., 1.]), jac=True, method="L-BFGS-B",
                      bounds=[(-20., 20.), (0., 40.)], options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(f"calibration failed: {result.message}")
    return FittedTree(tree, float(result.x[0]), float(result.x[1]), log_loss(result.x[0] + result.x[1] * f, y))


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[FittedTree, ...]
    config: SearchConfig
    fit_sha256: str
    fit_candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    n_atoms: int
    rounds_completed: int

    @property
    def sha256(self):
        payload = {"candidates": [f.to_dict() for f in self.candidates], "config": asdict(self.config),
                   "fit_sha256": self.fit_sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fit_pool(fit: QueryAtoms, labels, initial: Tree, config=SearchConfig()):
    """Generate and calibrate the entire finite search pool using FIT only."""
    y = labels_array(labels, len(fit.candidate_ids))
    if initial.nodes > config.max_nodes or initial.depth > config.max_depth:
        raise ValueError("initial tree exceeds search bounds")
    cache = {initial: calibrate(initial, fit, y, config.calibration_ridge)}
    beam = [initial]
    completed = 0
    rank = lambda t: (cache[t].fit_loss + config.size_penalty * t.nodes, t.nodes, t.key)
    for _ in range(config.rounds):
        proposals = set(beam)
        # Parents ranked on FIT only; deterministic bounded evaluation order.
        for parent in beam:
            for tree in propose_edits(parent, fit.values.shape[2], config):
                if tree not in cache and len(cache) < config.max_candidates:
                    cache[tree] = calibrate(tree, fit, y, config.calibration_ridge)
                if tree in cache:
                    proposals.add(tree)
        beam = sorted(proposals, key=rank)[:config.beam_width]
        completed += 1
    candidates = tuple(cache[t] for t in sorted(cache, key=lambda t: (t.nodes, t.key)))
    return CandidatePool(candidates, config, fit.sha256, fit.candidate_ids, fit.evidence_ids,
                         fit.values.shape[2], completed)


def select(pool: CandidatePool, probe: QueryAtoms, labels):
    if pool.evidence_ids != probe.evidence_ids or pool.n_atoms != probe.values.shape[2]:
        raise ValueError("FIT and PROBE must share exactly the same evidence and atom definitions")
    if set(pool.fit_candidate_ids) & set(probe.candidate_ids):
        raise ValueError("FIT and PROBE calibration candidates must be disjoint")
    y = labels_array(labels, len(probe.candidate_ids))
    rows = [(f, log_loss(f.logits(probe), y)) for f in pool.candidates]
    rows.sort(key=lambda pair: (pair[1] + pool.config.size_penalty * pair[0].tree.nodes,
                                pair[0].tree.nodes, pair[0].tree.key))
    selected, loss = rows[0]
    return selected, {"probe_loss": loss, "penalized_probe_loss": loss + pool.config.size_penalty * selected.tree.nodes,
                      "candidate_count": len(pool.candidates), "pool_sha256": pool.sha256,
                      "probe_sha256": probe.sha256, "selection_used_probe_labels": True}
