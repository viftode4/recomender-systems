"""Synthetic capacity check for the frozen pattern-evidence scorer.

No real data, model selection, held-out recommendation result, or external source
is read. This tests a deliberately constructed learnable feature relation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exploratory.evidence_transfer.model import (
    FEATURE_NAMES, SharedEvidenceScorer, extract_features, fit_normalizer,
    masked_listwise_loss, prepare_donors,
)

SEEDS = (2026, 2027, 2028)
STEPS = 300
LEARNING_RATE = .01
INSTANCES = 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy()
        result.update(name.encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def instances(seed, count=INSTANCES):
    """Every permutation is paired with one exchanging the two candidate IDs."""
    if count < 2 or count % 2:
        raise ValueError("Require an even positive number of balanced instances")
    donors = np.zeros((4, 7), dtype=np.int8)
    donors[0, [1, 2, 5]] = 1
    donors[1, [3, 4, 5]] = 1
    donors[2, [1, 2, 6]] = 1
    donors[3, [1, 2, 6]] = 1
    context = np.array([[0, 1, 1, 1, 1, 0, 0]], dtype=np.int8)
    rng = np.random.default_rng(seed)
    features, eligible, targets = [], [], []
    positive_has_lower_index = 0
    for _ in range(count // 2):
        order = np.r_[0, rng.permutation(np.arange(1, 7))]
        swapped = order.copy()
        swapped[order == 5], swapped[order == 6] = 6, 5
        for permutation in (order, swapped):
            bank = donors[rng.permutation(4)][:, permutation]
            query = context[:, permutation]
            value = extract_features(prepare_donors(bank), query, np.array([-1]))[0]
            rich, poor = int(np.flatnonzero(permutation == 5)[0]), int(np.flatnonzero(permutation == 6)[0])
            if not np.array_equal(value[rich, :7], value[poor, :7]):
                raise AssertionError("The first seven features must match exactly")
            if not (value[rich, 7] > value[poor, 7] and value[rich, 8] > value[poor, 8]):
                raise AssertionError("The richer-pattern relation is absent")
            mask = np.zeros(7, dtype=bool)
            mask[[rich, poor]] = True
            target = np.zeros(7, dtype=bool)
            target[rich] = True
            if query[0, [rich, poor]].any() or query[0, 0]:
                raise AssertionError("Target/candidate leakage into the context")
            features.append(value)
            eligible.append(mask)
            targets.append(target)
            positive_has_lower_index += int(rich < poor)
    if positive_has_lower_index != count // 2:
        raise AssertionError("Catalog-order shortcut is not exactly balanced")
    return np.stack(features), np.stack(eligible), np.stack(targets)


def measure(model, features, eligible, targets):
    with torch.no_grad():
        scores = model(features)
        loss = float(masked_listwise_loss(scores, targets, eligible))
    rich = scores[targets].cpu().numpy()
    poor = scores[eligible & ~targets].cpu().numpy()
    difference = rich - poor
    # Exact ties are meaningful here: identical masked inputs imply exact scores.
    wins, ties, losses = int(np.sum(difference > 0)), int(np.sum(difference == 0)), int(np.sum(difference < 0))
    index_choice = scores.masked_fill(~eligible, -torch.inf).argmax(dim=1)
    truth_index = targets.long().argmax(dim=1)
    return dict(instances=len(difference), listwise_nll=loss, strict_wins=wins,
                exact_ties=ties, strict_losses=losses,
                tie_aware_accuracy=(wins + .5 * ties) / len(difference),
                catalog_tie_break_accuracy=float((index_choice == truth_index).float().mean()),
                mean_positive_score_margin=float(np.mean(difference)),
                minimum_positive_score_margin=float(np.min(difference)))


def run(output):
    output = output.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite a capacity experiment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_files = (Path(__file__), Path(__file__).with_name("model.py"))
    signatures = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    raw_train, eligible_train, targets_train = instances(50101)
    raw_probe, eligible_probe, targets_probe = instances(50102)
    normalizer = fit_normalizer(raw_train, eligible_train)
    train, probe = (torch.from_numpy(normalizer.transform(value)) for value in (raw_train, raw_probe))
    em_train, tm_train, em_probe, tm_probe = map(torch.from_numpy, (eligible_train, targets_train, eligible_probe, targets_probe))
    protocol = dict(schema_version=1, experiment="synthetic_pattern_capacity", real_data_read=False,
                    test_read=False, model_selection=False, seeds=list(SEEDS), steps=STEPS,
                    learning_rate=LEARNING_RATE, optimizer="Adam", batch="all 64 training instances",
                    variants=["full_pattern", "no_pattern"], width=16,
                    train_instances=INSTANCES, probe_instances=INSTANCES,
                    feature_names=list(FEATURE_NAMES), equal_candidate_channels=list(FEATURE_NAMES[:7]),
                    differing_candidate_channels=list(FEATURE_NAMES[7:]),
                    normalization="Eligible synthetic training candidates only; frozen for probe instances",
                    target="Candidate with two disjoint supporter overlaps rather than repeated identical overlap",
                    synthetic_structure="Four context items; two candidates; four donor rows of length three. Each candidate has two donors with overlap two. Query is external to donors and candidates are hidden from context.",
                    permutations="Independent donor/item permutations; every instance paired with candidate-ID swap, giving exactly balanced catalog tie order in both splits",
                    interpretation="Capacity sanity check of a constructed two-feature relation. Probe cases are permutations of the same structural template, not new relational structures or real recommendation generalization.",
                    source_sha256=signatures,
                    runtime=dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, numerical_threads=1))
    output.mkdir(parents=True)
    save_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    results = {}
    for seed in SEEDS:
        models = {variant: SharedEvidenceScorer(variant=variant, seed=seed) for variant in protocol["variants"]}
        initial = {variant: state_digest(model) for variant, model in models.items()}
        if len(set(initial.values())) != 1:
            raise AssertionError("Variants do not have identical initialized parameters")
        rows = {}
        for variant, model in models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            before = measure(model, probe, em_probe, tm_probe)
            trace = []
            for step in range(1, STEPS + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = masked_listwise_loss(model(train), tm_train, em_train)
                loss.backward()
                if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite synthetic gradient")
                optimizer.step()
                if step in (1, 10, 30, 100, STEPS):
                    trace.append(dict(step=step, training=measure(model, train, em_train, tm_train)))
            after = measure(model, probe, em_probe, tm_probe)
            if variant == "no_pattern" and after["exact_ties"] != INSTANCES:
                raise AssertionError("Masked identical candidate inputs must tie")
            rows[variant] = dict(initial_state_sha256=initial[variant], allocated_parameters=sum(p.numel() for p in model.parameters()),
                                 before=before, after=after, training_curve=trace)
        results[str(seed)] = rows
    if {str(path.relative_to(ROOT)): digest(path) for path in source_files} != signatures:
        raise ValueError("A frozen source changed during the experiment")
    summary = dict(schema_version=1, experiment=protocol["experiment"], real_data_read=False,
                   test_read=False, model_selection=False, sources_unchanged=True,
                   elapsed_seconds=time.perf_counter()-started, protocol_sha256=digest(output / "protocol.json"),
                   training_steps_per_variant=STEPS, seeds=results,
                   theoretical_control="Without pattern channels both eligible candidates have identical input to the same shared network. Their scores necessarily tie for every possible parameter value; optimal balanced two-way NLL is log(2), and tie-aware accuracy is 0.5.")
    save_json(output / "aggregates.json", summary)
    lines = ["# Synthetic pattern-capacity check", "",
             "This constructed case isolates the two pattern channels. Both candidates have exactly equal first seven evidence features; only pattern count and coverage differ. "
             "The richer pattern is labelled positive by construction. Every candidate/item permutation is paired with an ID swap, so catalog tie order is exactly balanced. "
             "The frozen shared model has no ID embeddings.", "",
             "| Seed | Variant | Initial probe NLL | Final probe NLL | Strict wins / 64 | Exact ties / 64 | Tie-aware accuracy |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for seed, rows in results.items():
        for variant, row in rows.items():
            after = row["after"]
            lines.append(f"| {seed} | {variant} | {row['before']['listwise_nll']:.6f} | {after['listwise_nll']:.6f} | {after['strict_wins']} | {after['exact_ties']} | {after['tie_aware_accuracy']:.2%} |")
    lines.extend(["", f"Both variants use identical initialized parameters per seed, {STEPS} full-batch Adam steps at learning rate {LEARNING_RATE}, and the same synthetic examples. "
                  "There is no checkpoint selection, model selection, or data-dependent budget extension. The no-pattern control necessarily ties because the two candidate inputs become identical.", "",
                  "Probe instances are independently permuted copies of the same structural template. This demonstrates learnability of an intentionally supplied pattern relation; it does not demonstrate generalization to new relation types, useful preference signal in real data, a recommendation improvement, or methodological novelty. "
                  "Pattern count and coverage co-vary in this toy, so the experiment does not isolate which of those two channels supports learning. No real data or previous evaluation artifact was read.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    save_json(output / "SHA256.json", {name: digest(output / name) for name in ("protocol.json", "aggregates.json", "RESULTS.md")})
    print(f"Synthetic capacity check completed in {summary['elapsed_seconds']:.3f}s: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
