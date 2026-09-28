"""Exact finite Bayesian inference, with no task generator or hidden answers.

The prior is uniform over 16 concrete Boolean tables and, independently for
each task, 18 program descriptors. Transposed tables are grouped only for the
focused information objective; their induced prior/posterior mass is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math

import numpy as np

QUERY_TIE_TOLERANCE = 1e-12

class InconsistentEvidence(ValueError):
    """No library/program configuration explains the noiseless observations."""


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _table(table):
    if not isinstance(table, (int, np.integer)) or not 0 <= int(table) < 16:
        raise ValueError("table must be an integer in 0..15")
    return int(table)


def transpose_table(table: int) -> int:
    table = _table(table)
    return (table & 9) | ((table & 2) << 1) | ((table & 4) >> 1)


def canonical_table(table: int) -> int:
    table = _table(table)
    return min(table, transpose_table(table))


CANONICAL_TABLE = _readonly([canonical_table(table) for table in range(16)], np.int8)
CANONICAL_CLASSES = _readonly(np.unique(CANONICAL_TABLE), np.int8)
_CLASS_INDEX = np.searchsorted(CANONICAL_CLASSES, CANONICAL_TABLE)
INPUTS = _readonly([[index >> 2 & 1, index >> 1 & 1, index & 1] for index in range(8)], np.int8)


@dataclass(frozen=True)
class Program:
    op: str
    indices: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "indices", tuple(self.indices))
        if any(not isinstance(index, (int, np.integer)) for index in self.indices):
            raise ValueError("program indices must be integers")
        valid = (self.op == "direct" and len(self.indices) == 2 and len(set(self.indices)) == 2
                 and set(self.indices).issubset({0, 1, 2}))
        valid |= (self.op in {"left", "right"} and len(self.indices) == 3
                  and set(self.indices) == {0, 1, 2})
        if not valid:
            raise ValueError("invalid program descriptor")

    @property
    def calls(self):
        return 1 if self.op == "direct" else 2

    @property
    def descriptor(self):
        return self.op + "(" + ",".join(str(i) for i in self.indices) + ")"


PROGRAMS = (tuple(Program("direct", pair) for pair in permutations(range(3), 2))
            + tuple(Program("left", triple) for triple in permutations(range(3)))
            + tuple(Program("right", triple) for triple in permutations(range(3))))


def apply_table(table: int, a, b):
    table = _table(table)
    left, right = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    if not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
        raise ValueError("table arguments must be Boolean")
    index = 2 * left.astype(np.int8) + right.astype(np.int8)
    return ((table >> index) & 1).astype(np.int8)


def interpret(table: int, program: Program, inputs=INPUTS):
    x = np.asarray(inputs)
    if x.ndim < 1 or x.shape[-1] != 3 or not np.isin(x, [0, 1]).all():
        raise ValueError("inputs must end in three Boolean coordinates")
    i, j = program.indices[:2]
    if program.op == "direct":
        return apply_table(table, x[..., i], x[..., j])
    k = program.indices[2]
    if program.op == "left":
        return apply_table(table, apply_table(table, x[..., i], x[..., j]), x[..., k])
    return apply_table(table, x[..., i], apply_table(table, x[..., j], x[..., k]))


PREDICTIONS = _readonly([[interpret(table, program) for program in PROGRAMS] for table in range(16)], np.int8)


@dataclass(frozen=True)
class Evidence:
    outcomes: np.ndarray  # task x input; unobserved entries are scrubbed to zero
    observed: np.ndarray

    def __post_init__(self):
        raw = np.asarray(self.outcomes)
        mask_raw = np.asarray(self.observed)
        if raw.ndim != 2 or raw.shape[1] != 8 or mask_raw.shape != raw.shape:
            raise ValueError("outcomes and observed must have shape [tasks,8]")
        if not np.isin(mask_raw, [0, 1]).all():
            raise ValueError("observed mask must be Boolean")
        mask = mask_raw.astype(bool)
        if not np.isin(raw[mask], [0, 1]).all():
            raise ValueError("observed outcomes must be Boolean")
        clean = np.zeros(raw.shape, dtype=np.int8)
        clean[mask] = raw[mask]
        object.__setattr__(self, "outcomes", _readonly(clean))
        object.__setattr__(self, "observed", _readonly(mask))

    @classmethod
    def empty(cls, n_tasks: int):
        if not isinstance(n_tasks, (int, np.integer)) or n_tasks < 0:
            raise ValueError("nonnegative integer task count required")
        return cls(np.zeros((int(n_tasks), 8), dtype=np.int8), np.zeros((int(n_tasks), 8), dtype=bool))

    @property
    def n_tasks(self):
        return self.observed.shape[0]

    def observe(self, task: int, input_index: int, outcome: int):
        _query_indices(self, task, input_index)
        if outcome not in (0, 1):
            raise ValueError("outcome must be Boolean")
        if self.observed[task, input_index] and self.outcomes[task, input_index] != outcome:
            raise InconsistentEvidence("conflicting repeated observation")
        values, mask = self.outcomes.copy(), self.observed.copy()
        values[task, input_index] = outcome
        mask[task, input_index] = True
        return Evidence(values, mask)


def _query_indices(evidence, task, input_index):
    if not isinstance(task, (int, np.integer)) or not 0 <= task < evidence.n_tasks:
        raise ValueError("invalid task index")
    if not isinstance(input_index, (int, np.integer)) or not 0 <= input_index < 8:
        raise ValueError("invalid Boolean input index")


@dataclass(frozen=True)
class Posterior:
    evidence: Evidence
    p_library: np.ndarray  # [16]
    conditional_program: np.ndarray  # [16,tasks,18]
    consistent_counts: np.ndarray  # [16,tasks]
    valid: bool
    log_evidence: float

    def require_valid(self):
        if not self.valid:
            raise InconsistentEvidence("no shared library is consistent with all task observations")


def exact_posterior(evidence: Evidence) -> Posterior:
    # This factorization is equivalent to enumerating 16 * 18**tasks states.
    consistent = np.all((PREDICTIONS[:, None, :, :] == evidence.outcomes[None, :, None, :])
                        | ~evidence.observed[None, :, None, :], axis=-1)
    counts = consistent.sum(axis=-1)
    conditional = np.zeros(consistent.shape, dtype=float)
    np.divide(consistent, counts[:, :, None], out=conditional, where=counts[:, :, None] > 0)
    feasible = np.all(counts > 0, axis=1)
    if not feasible.any():
        return Posterior(evidence, _readonly(np.zeros(16)), _readonly(conditional),
                         _readonly(counts), False, -math.inf)
    log_weights = np.full(16, -math.inf)
    log_weights[feasible] = -math.log(16) + np.log(counts[feasible] / 18.).sum(axis=1)
    maximum = float(log_weights[feasible].max())
    weights = np.zeros(16)
    weights[feasible] = np.exp(log_weights[feasible] - maximum)
    normalizer = float(weights.sum())
    weights /= normalizer
    return Posterior(evidence, _readonly(weights), _readonly(conditional), _readonly(counts),
                     True, maximum + math.log(normalizer))


def binary_entropy(p):
    values = np.asarray(p, dtype=float)
    if not np.isfinite(values).all() or np.any((values < -1e-12) | (values > 1 + 1e-12)):
        raise ValueError("probabilities must be finite and in [0,1]")
    values = np.clip(values, 0., 1.)
    result = np.zeros(values.shape)
    for probability in (values, 1. - values):
        positive = probability > 0
        result[positive] -= probability[positive] * np.log2(probability[positive])
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class QueryStats:
    p_y1_by_library: np.ndarray
    p_y1: float
    output_entropy: float
    library_information: float  # I(Y; canonical table), in bits
    canonical_probabilities: np.ndarray
    p_y1_by_canonical: np.ndarray
    concrete_library_information: float


def information_from_conditional(p_library, p_y1_by_library) -> QueryStats:
    """Mixture-aware canonical MI; aliases need not share outcome probabilities."""
    weights = np.asarray(p_library, dtype=float)
    conditional = np.asarray(p_y1_by_library, dtype=float)
    if (weights.shape != (16,) or conditional.shape != (16,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("normalized 16-table probabilities required")
    binary_entropy(conditional)  # validation even for zero-mass concrete tables
    conditional = np.clip(conditional, 0., 1.)
    p_y1 = float(np.clip(weights @ conditional, 0., 1.))
    entropy = binary_entropy(p_y1)
    group_mass = np.bincount(_CLASS_INDEX, weights=weights, minlength=len(CANONICAL_CLASSES))
    group_positive = np.bincount(_CLASS_INDEX, weights=weights * conditional, minlength=len(CANONICAL_CLASSES))
    group_conditional = np.zeros(len(CANONICAL_CLASSES))
    np.divide(group_positive, group_mass, out=group_conditional, where=group_mass > 0)
    group_conditional = np.clip(group_conditional, 0., 1.)
    focused = entropy - float(group_mass @ binary_entropy(group_conditional))
    concrete = entropy - float(weights @ binary_entropy(conditional))
    return QueryStats(_readonly(conditional), p_y1, entropy, float(np.clip(focused, 0., entropy)),
                      _readonly(group_mass), _readonly(group_conditional), float(np.clip(concrete, 0., entropy)))


def query_stats(posterior: Posterior, task: int, input_index: int) -> QueryStats:
    posterior.require_valid()
    _query_indices(posterior.evidence, task, input_index)
    probabilities = np.clip(np.einsum("lp,lp->l", posterior.conditional_program[:, task, :],
                                       PREDICTIONS[:, :, input_index]), 0., 1.)
    return information_from_conditional(posterior.p_library, probabilities)


@dataclass(frozen=True)
class QueryChoice:
    task: int
    input_index: int
    stats: QueryStats


def choose_query(posterior: Posterior, criterion="library_information"):
    posterior.require_valid()
    if criterion not in {"entropy", "library_information"}:
        raise ValueError("criterion must be entropy or library_information")
    candidates = []
    for task in range(posterior.evidence.n_tasks):
        for input_index in range(8):
            if posterior.evidence.observed[task, input_index]:
                continue
            stats = query_stats(posterior, task, input_index)
            score = stats.output_entropy if criterion == "entropy" else stats.library_information
            candidates.append((QueryChoice(task, input_index, stats), score))
    if not candidates:
        return None
    maximum = max(score for _, score in candidates)
    # Resolve tolerance ties against the GLOBAL maximum, not a streaming best.
    return next(choice for choice, score in candidates if maximum - score <= QUERY_TIE_TOLERANCE)


def select_map_library(posterior: Posterior) -> int:
    """Concrete-table MAP, numeric-minimum ties; never canonical-class MAP."""
    posterior.require_valid()
    return int(np.argmax(posterior.p_library))


@dataclass(frozen=True)
class TransferProgram:
    table: int
    program_index: int
    program: Program
    error_count: int
    predictions: np.ndarray


def select_transfer_program(table: int, support_indices, outcomes) -> TransferProgram:
    """One frozen-table program; target labels outside support are never accepted.

    Ties: support errors, calls, lexicographic model truth vector, descriptor index.
    The truth vector is the MODEL'S prediction, never the task's hidden answers.
    """
    table = _table(table)
    indices_raw, values = np.asarray(support_indices), np.asarray(outcomes)
    if indices_raw.ndim != 1 or values.shape != indices_raw.shape:
        raise ValueError("outcomes must be a support-length vector")
    if not np.isin(indices_raw, range(8)).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("invalid support indices or outcomes")
    indices = indices_raw.astype(int)
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("support indices must be unique")
    predictions = PREDICTIONS[table]
    errors = np.sum(predictions[:, indices] != values[None, :], axis=1)
    chosen = min(range(18), key=lambda p: (int(errors[p]), PROGRAMS[p].calls,
                                          tuple(int(y) for y in predictions[p]), p))
    return TransferProgram(table, chosen, PROGRAMS[chosen], int(errors[chosen]), _readonly(predictions[chosen]))
