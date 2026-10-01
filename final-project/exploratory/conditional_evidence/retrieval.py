"""Gated retrieval learned from out-of-fold masked-TRAIN loss reductions.

This module does not read datasets, select a reader, or open development labels.
The caller supplies the already selected reader configuration and epoch budget.
Each utility-label fold retrains that configuration on the other users only.
At deployment, a single context graph encodes the eight possible donor actions;
one chosen donor is then added to every eligible candidate graph, with deduplication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


GLOBAL_DONORS = 8
ACTION_COUNT = 8
FOLDS = 3
UTILITY_HIDDEN = 16
UTILITY_EPOCHS = 100
UTILITY_LEARNING_RATE = .001


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    seeds: tuple[int, ...]
    mean_raw: float
    control_means: dict[str, float]
    reference_means: dict[str, float]
    threshold: float
    stage: str = "meta_fit"


def meta_gate(raw: Mapping[int, float], controls: Mapping[str, Mapping[int, float]],
              references: Mapping[str, Mapping[int, float]], *, stage: str) -> GateDecision:
    """Strictly exceed both control means and the strongest standalone mean.

    The strongest reference is selected by its equal-seed mean, not the mean of
    per-seed winners. Ties close the gate. Only meta-fit inputs are accepted.
    """
    if stage != "meta_fit" or len(controls) != 2 or not references or not raw:
        raise ValueError("Require meta-fit scores, exactly two controls and standalone references")
    seeds = tuple(sorted(raw))
    if any(not isinstance(seed, (int, np.integer)) for seed in seeds):
        raise ValueError("Seed keys must be integers")
    def mean(values):
        if set(values) != set(seeds):
            raise ValueError("Every gate model must cover the same seeds")
        numbers = [float(values[seed]) for seed in seeds]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in numbers):
            raise ValueError("Meta-fit nDCG values must be finite and in [0, 1]")
        return math.fsum(numbers)/len(numbers)
    raw_mean = mean(raw)
    control_means = {name: mean(values) for name, values in controls.items()}
    reference_means = {name: mean(values) for name, values in references.items()}
    threshold = max(*control_means.values(), *reference_means.values())
    return GateDecision(raw_mean > threshold, seeds, raw_mean, control_means, reference_means, threshold)


@dataclass
class Costs:
    donor_cosine_rows: int = 0
    context_graph_passes: int = 0
    context_nodes: int = 0
    context_edges: int = 0
    reader_forward_passes: int = 0
    candidate_graphs: int = 0
    candidate_nodes: int = 0
    candidate_edges: int = 0
    scored_candidates: int = 0
    donor_insertions: int = 0
    donor_already_present: int = 0
    full_catalog_losses: int = 0
    fold_reader_fits: int = 0
    declared_fold_reader_epochs: int = 0
    fold_training_seconds: float = 0.0
    utility_training_steps: int = 0
    utility_training_examples: int = 0
    utility_prediction_examples: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


def _categories(value):
    array = np.asarray(value)
    if (array.ndim != 2 or array.dtype.kind not in "biuf" or not np.isfinite(array).all()
            or not np.isin(array, np.arange(6)).all()):
        raise ValueError("Categories must be a finite user-by-item matrix in 0..5")
    return np.array(array, dtype=np.uint8, copy=True)


def _context(value, items):
    array = np.asarray(value)
    if (array.shape != (items,) or array.dtype.kind not in "biuf" or not np.isfinite(array).all()
            or not np.isin(array, np.arange(6)).all()):
        raise ValueError("Context must be a finite item vector in 0..5")
    return np.array(array, dtype=np.uint8, copy=True)


@dataclass(frozen=True)
class Episode:
    user_row: int
    context: np.ndarray
    targets: np.ndarray


def eligible_items(context, padding=0):
    context = np.asarray(context)
    result = context == 0
    if not 0 <= padding < len(context):
        raise ValueError("Invalid padding index")
    result[padding] = False
    return result


def _target_indices(targets, items):
    values = np.asarray(targets)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or len(values) == 0
            or len(np.unique(values)) != len(values) or np.any(values < 0) or np.any(values >= items)):
        raise ValueError("Targets must be a nonempty vector of distinct valid item indices")
    return values.astype(np.int64, copy=True)


def full_catalog_loss(logits, targets, eligible, costs: Costs | None = None):
    """Mean NLL over hidden recorded events, normalized over every eligible item."""
    scores = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(eligible)
    if scores.ndim != 1 or mask.shape != scores.shape or mask.dtype.kind != "b" or not mask.any():
        raise ValueError("Require scores and a nonempty Boolean full-catalog eligibility mask")
    targets = _target_indices(targets, len(scores))
    if not mask[targets].all() or not np.isfinite(scores[mask]).all():
        raise ValueError("Targets must be eligible and all eligible scores finite")
    maximum = float(np.max(scores[mask]))
    # Center the target scores too, avoiding cancellation under a large common
    # logit shift. Ineligible entries may be NaN and are never inspected.
    log_partition = math.log(float(np.exp(scores[mask]-maximum).sum()))
    if costs is not None:
        costs.full_catalog_losses += 1
    result = log_partition-float((scores[targets]-maximum).mean())
    if not math.isfinite(result):
        raise ValueError("Full-catalog loss is not representable as a finite value")
    return result


def loss_reduction(base_logits, action_logits, targets, eligible, costs: Costs | None = None):
    """Positive means the action improved the actual multinomial objective."""
    return (full_catalog_loss(base_logits, targets, eligible, costs)
            - full_catalog_loss(action_logits, targets, eligible, costs))


def donor_actions(categories, context, *, exclude_row, allowed_rows=None, costs=None):
    """Next eight cosine donors after the common first eight; stable row ties."""
    history = _categories(categories)
    query = _context(context, history.shape[1]) > 0
    if exclude_row < -1 or exclude_row >= len(history):
        raise ValueError("Invalid query donor exclusion")
    allowed = (np.ones(len(history), dtype=bool) if allowed_rows is None
               else np.asarray(allowed_rows).copy())
    if allowed.shape != (len(history),) or allowed.dtype.kind != "b":
        raise ValueError("Donor permissions must be a Boolean row vector")
    if exclude_row >= 0:
        allowed[exclude_row] = False
    rows = np.flatnonzero(allowed)
    incidence = history[rows] > 0
    overlap = incidence @ query.astype(np.float64)
    denominator = np.sqrt(incidence.sum(axis=1)*float(query.sum()))
    cosine = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
    order = np.lexsort((rows, -cosine))
    if costs is not None:
        costs.donor_cosine_rows += len(rows)
    return tuple(map(int, rows[order][GLOBAL_DONORS:GLOBAL_DONORS+ACTION_COUNT]))


def deduplicated_additions(candidate_ids, donor_rows, action):
    """One global action, deduplicated separately in each candidate graph."""
    if len(candidate_ids) != len(donor_rows):
        raise ValueError("Candidate IDs and donor graph rows differ")
    if action is None:
        return {}
    if not isinstance(action, (int, np.integer)) or action < 0:
        raise ValueError("Action must identify one allowed donor")
    return {int(candidate): (() if action in rows else (int(action),))
            for candidate, rows in zip(candidate_ids, donor_rows)}


def choose_action(actions, predicted_reductions):
    actions = tuple(actions)
    values = np.asarray(predicted_reductions, dtype=np.float64)
    if values.shape != (len(actions),) or not np.isfinite(values).all() or len(set(actions)) != len(actions):
        raise ValueError("Require one finite utility prediction per unique action")
    if not actions or float(values.max()) <= 0:
        return None
    return actions[int(np.argmax(values))]


def control_actions(actions, *, seed, actual_reductions=None):
    """Fixed controls; the oracle is labelled separately and may choose no action."""
    actions = tuple(actions)
    result = {"no_action": None, "next_nearest": actions[0] if actions else None,
              "random": int(np.random.default_rng(seed).choice(actions)) if actions else None}
    if actual_reductions is not None:
        result["oracle"] = choose_action(actions, actual_reductions)
    return result


def _freeze(reader):
    reader.eval()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


def _device(reader):
    return next(reader.parameters(), torch.empty(0)).device


def _bank_parts(bank):
    history = _categories(bank.categories)
    allowed = getattr(bank, "allowed_rows", None)
    if allowed is None:
        allowed = np.ones(len(history), dtype=bool)
    return history, np.asarray(allowed, dtype=bool)


def context_action_features(reader, bank, context, exclude_row, actions, costs):
    """One small context-only graph, with no candidate evidence or target labels.

    Columns are frozen query state, frozen donor state, then raw query-history
    count, donor-history count and intersection count. All action donors are
    visible in this representation pass; this extra work is explicitly charged.
    The candidate node required by the reader is isolated and never read here.
    """
    actions = tuple(actions)
    if not actions:
        return np.empty((0, 35), dtype=np.float64)
    history, allowed = _bank_parts(bank)
    context = _context(context, history.shape[1])
    if any(row == exclude_row or row < 0 or row >= len(history) or not allowed[row] for row in actions):
        raise ValueError("A context probe cannot use an excluded donor")
    batch = bank.build_context_query(context, exclude_row=exclude_row, donor_rows=actions)
    costs.context_graph_passes += 1
    costs.context_nodes += int(batch.x.shape[0])
    costs.context_edges += int(batch.edge_index.shape[1])
    _freeze(reader)
    with torch.inference_mode():
        states = reader.node_embeddings(batch.to(_device(reader))).detach().cpu().numpy()
    query_indices = np.asarray(batch.query_idx.cpu(), dtype=np.int64)
    donor_indices = np.asarray(batch.donor_idx.cpu(), dtype=np.int64)
    rows = tuple(row for group in batch.donor_rows for row in group)
    if len(query_indices) != 1 or len(rows) != len(donor_indices) or set(rows) != set(actions):
        raise ValueError("Context graph does not contain exactly the requested action donors")
    if states.ndim != 2 or states.shape[1] != 16 or not np.isfinite(states).all():
        raise ValueError("Expected finite 16-dimensional frozen reader states")
    query_state = states[query_indices[0]]
    representations = {row: states[node] for row, node in zip(rows, donor_indices)}
    mask = context > 0
    result = []
    for row in actions:
        donor = history[row] > 0
        counts = np.array([mask.sum(), donor.sum(), np.count_nonzero(mask & donor)], dtype=np.float64)
        result.append(np.concatenate([query_state, representations[row], counts]))
    return np.asarray(result, dtype=np.float64)


def score_catalog(reader, bank, context, exclude_row, action, costs, *, chunk_size=128):
    """Score all eligible items after one global donor action; no sampled loss."""
    history, allowed = _bank_parts(bank)
    context = _context(context, history.shape[1])
    if chunk_size < 1:
        raise ValueError("Positive candidate chunk size required")
    if action is not None and (action == exclude_row or action < 0 or action >= len(history) or not allowed[action]):
        raise ValueError("Action donor is excluded from this bank")
    candidates = np.flatnonzero(eligible_items(context, getattr(bank, "padding", 0)))
    scores = np.full(history.shape[1], np.nan, dtype=np.float64)
    _freeze(reader)
    for start in range(0, len(candidates), chunk_size):
        items = candidates[start:start+chunk_size]
        batch = bank.build_query(context, exclude_row=exclude_row, candidate_ids=items, variant="raw")
        costs.donor_cosine_rows += len(history)
        if action is not None:
            additions = deduplicated_additions(items, batch.donor_rows, action)
            costs.donor_insertions += sum(bool(value) for value in additions.values())
            costs.donor_already_present += sum(not value for value in additions.values())
            # Base construction is also real work, even though only the augmented
            # graph receives a reader forward pass.
            costs.candidate_graphs += len(items)
            costs.candidate_nodes += int(batch.x.shape[0])
            costs.candidate_edges += int(batch.edge_index.shape[1])
            batch = bank.build_query(context, exclude_row=exclude_row, candidate_ids=items,
                                     variant="raw", extra_donor=additions)
            costs.donor_cosine_rows += len(history)
        costs.candidate_graphs += len(items)
        costs.candidate_nodes += int(batch.x.shape[0])
        costs.candidate_edges += int(batch.edge_index.shape[1])
        costs.reader_forward_passes += 1
        costs.scored_candidates += len(items)
        with torch.inference_mode():
            values = reader(batch.to(_device(reader))).detach().cpu().numpy()
        if values.shape != (len(items),) or not np.isfinite(values).all():
            raise ValueError("Reader returned invalid candidate scores")
        scores[items] = values
    return scores


@dataclass(frozen=True)
class QueryObservations:
    actions: tuple[int, ...]
    features: np.ndarray
    reductions: np.ndarray
    base_loss: float
    action_losses: np.ndarray


def collect_query(reader, bank, episode, costs, *, chunk_size=128):
    """Counterfactual labels for training only; each action changes the whole catalog."""
    history, allowed = _bank_parts(bank)
    actions = donor_actions(history, episode.context, exclude_row=episode.user_row,
                            allowed_rows=allowed, costs=costs)
    features = context_action_features(reader, bank, episode.context, episode.user_row, actions, costs)
    mask = eligible_items(episode.context, getattr(bank, "padding", 0))
    baseline = score_catalog(reader, bank, episode.context, episode.user_row, None, costs, chunk_size=chunk_size)
    baseline_loss = full_catalog_loss(baseline, episode.targets, mask, costs)
    losses = np.array([full_catalog_loss(
        score_catalog(reader, bank, episode.context, episode.user_row, action, costs, chunk_size=chunk_size),
        episode.targets, mask, costs) for action in actions])
    return QueryObservations(actions, features, baseline_loss-losses, baseline_loss, losses)


@dataclass(frozen=True)
class FoldTrainingRequest:
    fold: int
    categories: np.ndarray
    original_user_rows: np.ndarray
    episodes: tuple[Episode, ...]
    selected_config: dict[str, Any]
    epochs: int
    seed: int  # Fold mask/training RNG; reader initialization stays selected_config['seed'].


@dataclass(frozen=True)
class ReaderFit:
    reader: Any
    completed_epochs: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossfitData:
    gate: GateDecision
    features: np.ndarray
    reductions: np.ndarray
    folds: tuple[dict[str, Any], ...]
    costs: Costs
    stage: str = "masked_train_out_of_fold"


def user_folds(user_count, *, seed):
    if not isinstance(user_count, (int, np.integer)) or user_count < FOLDS:
        raise ValueError("Three nonempty user folds are required")
    result = np.empty(user_count, dtype=np.int64)
    result[np.random.default_rng(seed).permutation(user_count)] = np.arange(user_count) % FOLDS
    return result


def _validated_episodes(categories, episodes, padding):
    result = []
    for episode in episodes:
        row = episode.user_row
        if not isinstance(row, (int, np.integer)) or not 0 <= row < len(categories):
            raise ValueError("Episode user must identify a TRAIN row")
        context = _context(episode.context, categories.shape[1])
        if context[padding] or not np.array_equal(context[context > 0], categories[row][context > 0]):
            raise ValueError("Context must preserve only observed TRAIN categories and exclude padding")
        targets = _target_indices(episode.targets, categories.shape[1])
        hidden = np.flatnonzero((categories[row] > 0) & (context == 0))
        hidden = hidden[hidden != padding]
        if not np.array_equal(np.sort(targets), hidden):
            raise ValueError("Episode targets must be exactly all hidden TRAIN observations")
        context.setflags(write=False)
        targets.setflags(write=False)
        result.append(Episode(int(row), context, targets))
    if not result:
        raise ValueError("Masked TRAIN episodes are required")
    return tuple(result)


def crossfit_labels(categories, episodes, *, gate, selected_config, selected_epochs,
                    seed, fit_reader: Callable[[FoldTrainingRequest], ReaderFit], padding=0,
                    bank_factory=None, query_collector=collect_query):
    """Retrain fixed reader settings three times, with entire user-fold exclusion.

    The callback receives only other-fold categories and remapped training
    episodes. Held users are absent from both that callback and the donor bank.
    Evaluation episodes are external queries (row -1) against that smaller bank.
    No fold chooses an epoch, architecture, learning rate or utility hyperparameter.
    """
    costs = Costs()
    if not isinstance(gate, GateDecision) or gate.stage != "meta_fit":
        raise ValueError("An explicit meta-fit gate is required")
    if not gate.enabled:
        return CrossfitData(gate, np.empty((0, 35)), np.empty(0), (), costs)
    if not isinstance(selected_epochs, (int, np.integer)) or selected_epochs < 0:
        raise ValueError("A fixed nonnegative selected epoch budget is required")
    history = _categories(categories)
    if not 0 <= padding < history.shape[1] or history[:, padding].any():
        raise ValueError("Padding must have no recorded events")
    episodes = _validated_episodes(history, episodes, padding)
    assignments = user_folds(len(history), seed=seed)
    if bank_factory is None:
        from .model import EvidenceBank
        bank_factory = lambda values: EvidenceBank(values, padding=padding)
    started, features, reductions, audits = time.monotonic(), [], [], []
    for fold in range(FOLDS):
        train_rows = np.flatnonzero(assignments != fold)
        held_rows = np.flatnonzero(assignments == fold)
        local_rows = {int(original): local for local, original in enumerate(train_rows)}
        training = tuple(Episode(local_rows[e.user_row], e.context, e.targets)
                         for e in episodes if assignments[e.user_row] != fold)
        held = tuple(e for e in episodes if assignments[e.user_row] == fold)
        if not training or not held:
            raise ValueError("Every fold needs training and held-out masked episodes")
        training_categories = history[train_rows].copy()
        training_categories.setflags(write=False)
        train_rows.setflags(write=False)
        request = FoldTrainingRequest(fold, training_categories, train_rows, training,
            copy.deepcopy(dict(selected_config)), int(selected_epochs), int(seed)+51001+fold)
        before = time.monotonic()
        fitted = fit_reader(request)
        costs.fold_training_seconds += time.monotonic()-before
        costs.fold_reader_fits += 1
        costs.declared_fold_reader_epochs += selected_epochs
        if not isinstance(fitted, ReaderFit) or fitted.completed_epochs != selected_epochs:
            raise ValueError("Fold reader did not follow the frozen epoch budget")
        reader_config = getattr(fitted.reader, "config", {})
        if "seed" in selected_config and "seed" in reader_config and reader_config["seed"] != selected_config["seed"]:
            raise ValueError("Fold reader must use the selected reader's common initialization seed")
        _freeze(fitted.reader)
        bank = bank_factory(training_categories.copy())
        actual_history, actual_allowed = _bank_parts(bank)
        if not np.array_equal(actual_history, training_categories) or not actual_allowed.all():
            raise ValueError("Fold donor bank must contain exactly the other-fold TRAIN rows")
        examples, opportunity = 0, []
        for query_index, episode in enumerate(held):
            observation = query_collector(fitted.reader, bank,
                Episode(-1, episode.context, episode.targets), costs)
            x = np.asarray(observation.features, dtype=np.float64)
            y = np.asarray(observation.reductions, dtype=np.float64)
            if (x.ndim != 2 or x.shape[1] != 35 or y.shape != (len(x),)
                    or len(observation.actions) != len(x) or len(x) > ACTION_COUNT
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError("Invalid crossfit action features or exact loss labels")
            action_losses = np.asarray(observation.action_losses, dtype=np.float64)
            if (not math.isfinite(observation.base_loss) or action_losses.shape != y.shape
                    or not np.isfinite(action_losses).all()
                    or not np.allclose(y, observation.base_loss-action_losses, rtol=0, atol=1e-12)):
                raise ValueError("Action labels must be actual full-catalog loss reductions")
            if any(row < 0 or row >= len(train_rows) for row in observation.actions):
                raise ValueError("Crossfit action is outside the allowed donor bank")
            features.append(x)
            reductions.append(y)
            examples += len(y)
            random_seed = int(np.random.SeedSequence([int(seed), fold, query_index, 53001]).generate_state(1)[0])
            controls = control_actions(observation.actions, seed=random_seed, actual_reductions=y)
            gain = lambda action: 0. if action is None else float(y[observation.actions.index(action)])
            opportunity.append({"base_loss": float(observation.base_loss), "oracle": gain(controls["oracle"]),
                "next_nearest": gain(controls["next_nearest"]), "random": gain(controls["random"])})
        audits.append({"fold": fold, "training_users": len(train_rows), "held_users": len(held_rows),
            "training_episodes": len(training), "held_episodes": len(held), "action_labels": examples,
            "selected_epochs": selected_epochs, "fold_training_seed": request.seed,
            "reader_initialization_seed": selected_config.get("seed"),
            "held_users_excluded_from_training_and_donors": True,
            "training_rows_sha256": hashlib.sha256(train_rows.astype('<i8').tobytes()).hexdigest(),
            "training_opportunity": {"queries": len(opportunity),
                "positive_oracle_queries": sum(value["oracle"] > 0 for value in opportunity),
                **{"mean_"+key: math.fsum(value[key] for value in opportunity)/len(opportunity)
                   for key in ("base_loss", "oracle", "next_nearest", "random")},
                "random_seed_rule": "SeedSequence([seed,fold,query_index,53001])",
                "scope": "Out-of-fold TRAIN full-catalog loss reduction; oracle is privileged and not deployable"},
            "reader_diagnostics": copy.deepcopy(dict(fitted.diagnostics))})
    costs.elapsed_seconds = time.monotonic()-started
    return CrossfitData(gate, np.concatenate(features), np.concatenate(reductions), tuple(audits), costs)


class UtilityNetwork(nn.Module):
    def __init__(self, inputs=35):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, UTILITY_HIDDEN), nn.SiLU(), nn.Linear(UTILITY_HIDDEN, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


@dataclass(frozen=True)
class UtilityModel:
    network: UtilityNetwork
    mean: np.ndarray
    scale: np.ndarray
    seed: int
    examples: int
    training_mse: float

    def predict(self, features, costs=None):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError("Utility features have invalid dimensions or values")
        if costs is not None:
            costs.utility_prediction_examples += len(features)
        values = torch.as_tensor((features-self.mean)/self.scale, dtype=torch.float32)
        with torch.inference_mode():
            return self.network(values).cpu().numpy().astype(np.float64)

    def checkpoint(self):
        """Torch-saveable frozen policy; no episodes, targets or donor identities."""
        return {"config": {"inputs": 35, "hidden": UTILITY_HIDDEN, "activation": "SiLU",
            "optimizer": "Adam", "learning_rate": UTILITY_LEARNING_RATE, "epochs": UTILITY_EPOCHS,
            "loss": "MSE", "batch": "full"}, "seed": self.seed, "examples": self.examples,
            "training_mse": self.training_mse, "mean": torch.from_numpy(self.mean.copy()),
            "scale": torch.from_numpy(self.scale.copy()),
            "state_dict": {name: value.detach().cpu().clone() for name, value in self.network.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        expected = {"inputs": 35, "hidden": UTILITY_HIDDEN, "activation": "SiLU", "optimizer": "Adam",
            "learning_rate": UTILITY_LEARNING_RATE, "epochs": UTILITY_EPOCHS, "loss": "MSE", "batch": "full"}
        mean = np.asarray(checkpoint["mean"], dtype=np.float64)
        scale = np.asarray(checkpoint["scale"], dtype=np.float64)
        if (checkpoint["config"] != expected or mean.shape != (35,) or scale.shape != (35,)
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Frozen utility configuration or normalization differs")
        with torch.random.fork_rng(devices=[]):
            network = UtilityNetwork()
            network.load_state_dict(checkpoint["state_dict"], strict=True)
        if not all(torch.isfinite(value).all() for value in network.state_dict().values()):
            raise ValueError("Non-finite utility checkpoint weights")
        return cls(_freeze(network), mean.copy(), scale.copy(), int(checkpoint["seed"]),
                   int(checkpoint["examples"]), float(checkpoint["training_mse"]))


def fit_utility(data: CrossfitData, *, seed):
    """Fixed full-batch Adam/MSE; normalization uses only out-of-fold TRAIN labels."""
    if not data.gate.enabled:
        return None
    if data.stage != "masked_train_out_of_fold" or len(data.folds) != FOLDS:
        raise ValueError("Utility fitting requires all three masked-TRAIN folds")
    features, labels = np.asarray(data.features, dtype=np.float64), np.asarray(data.reductions, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 35 or not len(features)
            or labels.shape != (len(features),) or not np.isfinite(features).all() or not np.isfinite(labels).all()):
        raise ValueError("Nonempty finite crossfit features and loss-reduction labels are required")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1
    x = torch.as_tensor((features-mean)/scale, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        network = UtilityNetwork()
        optimizer = torch.optim.Adam(network.parameters(), lr=UTILITY_LEARNING_RATE)
        for _ in range(UTILITY_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((network(x)-y)**2)
            if not torch.isfinite(loss):
                raise ValueError("Utility training became non-finite")
            loss.backward()
            optimizer.step()
        network = _freeze(network)
        with torch.inference_mode():
            training_mse = float(torch.mean((network(x)-y)**2))
    data.costs.utility_training_steps += UTILITY_EPOCHS
    data.costs.utility_training_examples += len(features)*UTILITY_EPOCHS
    data.costs.elapsed_seconds += time.monotonic()-started
    return UtilityModel(network, mean, scale, int(seed), len(features), training_mse)


@dataclass(frozen=True)
class ExtensionFit:
    crossfit: CrossfitData
    utility: UtilityModel | None

    def protocol_summary(self):
        return {"gate": asdict(self.crossfit.gate), "stage": self.crossfit.stage,
            "status": "trained" if self.utility is not None else "gate_closed",
            "folds": list(self.crossfit.folds), "costs": self.crossfit.costs.to_dict(),
            "utility_config": self.utility.checkpoint()["config"] if self.utility is not None else None,
            "action_pool": "next eight cosine donors after the first eight, excluding query",
            "utility_features": "query16, donor16, raw query/donor/intersection counts from context-only graph",
            "utility_target": "base minus augmented full-catalog mean recorded-event NLL",
            "deployment": "one global action, or no action when all predicted reductions are nonpositive",
            "representation_limit": "Common initialization does not guarantee aligned learned latent coordinates "
                "between independently trained fold readers and the full reader; donor-bank size also changes",
            "development_used_for_training": False}


def fit_extension(categories, episodes, *, gate, selected_config, selected_epochs, seed,
                  fit_reader, padding=0, bank_factory=None, query_collector=collect_query):
    """Complete gated extension: three fixed reader refits, OOF labels, one utility fit."""
    data = crossfit_labels(categories, episodes, gate=gate, selected_config=selected_config,
        selected_epochs=selected_epochs, seed=seed, fit_reader=fit_reader, padding=padding,
        bank_factory=bank_factory, query_collector=query_collector)
    return ExtensionFit(data, fit_utility(data, seed=int(seed)+52001))


def deploy(reader, bank, context, exclude_row, utility, *, policy="learned", seed=0, chunk_size=128):
    """Label-free deployment. Fixed/random controls pay the same context pass.

    There is no target argument and no utility fitting here. The oracle belongs
    to counterfactual evaluation only, via collect_query/control_actions.
    """
    if policy not in {"learned", "next_nearest", "random", "no_action"}:
        raise ValueError("Unknown deployment policy; oracle is evaluation-only")
    costs, started = Costs(), time.monotonic()
    history, allowed = _bank_parts(bank)
    actions = donor_actions(history, context, exclude_row=exclude_row, allowed_rows=allowed, costs=costs)
    features = context_action_features(reader, bank, context, exclude_row, actions, costs)
    predictions = None
    if policy == "learned":
        if utility is None:
            raise ValueError("A frozen utility model is required")
        predictions = utility.predict(features, costs)
        selected = choose_action(actions, predictions)
    else:
        selected = control_actions(actions, seed=seed)[policy]
    scores = score_catalog(reader, bank, context, exclude_row, selected, costs, chunk_size=chunk_size)
    costs.elapsed_seconds = time.monotonic()-started
    feature_summary = {"count": len(features), "sum": features.sum(axis=0).tolist(),
                       "sum_squared": np.square(features).sum(axis=0).tolist()}
    return {"scores": scores, "action": selected, "actions": actions,
            "predicted_reductions": predictions, "feature_summary": feature_summary, "costs": costs}
