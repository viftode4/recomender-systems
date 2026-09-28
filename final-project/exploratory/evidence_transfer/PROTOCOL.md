# Shared evidence interpreter: first mechanism test

Written before fitting real-data models. The runner will hash this protocol,
model, runner, and imported project helpers before training. Old studies and
the exposed TEST are preserved. No fresh confirmation is claimed.

## Data and information boundary

- Use the completed categorical reconstruction study's hash-verified TRAIN
  categories, original catalog ordering and cohorts for seeds 2026, 2027, 2028.
- Binarize TRAIN observations for this pilot. This is a modeling choice, not a
  requirement imposed by the project PDF. Keep the existing all-recorded
  nDCG@10 endpoint to compare the new mechanism against existing evidence.
- For each TRAIN user make two deterministic queries, retaining approximately
  80% and 90% of observed items, at least one retained and one hidden. All arms
  use the same query contexts and targets. Users with fewer than two observations
  must be reported and excluded from training episodes, never silently fabricated.
- Use all users' TRAIN records as the donor bank, but entirely exclude the query
  user's donor row from counts, weights, feature normalization denominators and
  pattern summaries for every query. No VALID/TEST target is a training label.
- Purely TRAIN-derived features for original full histories may be precomputed.
  Original 471 meta users' VALID records select checkpoints. The 472 development
  users' VALID outcomes are not evaluated until every seed's selections are
  sealed. These populations have been examined in earlier research.

## Model and controls

Squared cosine donor weights; nine features: log context size, log candidate
support, candidate support fraction, log weighted support mass, weighted support
fraction, mean supporter weight, log Kish count, log effective pattern count,
and pattern coverage fraction. See the derivation and source for exact zero
handling. Candidate zero is padding and never eligible.

The shared scorer has nine inputs, one hidden layer of width 16 with SiLU,
and a scalar linear output. It has no item/user parameters and no expert scores.
Feature scaling uses eligible TRAIN-episode positions only, shared by arms.
The no-pattern arm zeros only the final two standardized channels. The marginal
arm keeps only the first three. Architecture, initialization, batch schedule,
optimizer, training targets and checkpoint opportunities otherwise match.
Effective input information differs by design; equal allocated parameter count
does not mean equal effective capacity.

Three trained arms: full, no-pattern, marginal. Two untrained/reference arms:
weighted donor support fraction and the prior locked expanded binary EASE.
The new scorer and EASE have different fitting objectives and query-row use;
the no-pattern arm is the direct matched test of the pattern-feature hypothesis.
Also report an inference-only intervention that zeros the pattern channels in
the selected full model without retraining or selecting again. It measures
sensitivity under changed inputs, not the benefit of retraining without them.

Features are valid for candidates absent from the supplied query context.
Seen-target features need not remove that target individually because those
positions are excluded from both loss and recommendation. Donor histories may
contain a candidate; that observed supporter relationship is the intended
evidence. The query user's own held-out record never supplies that relationship.

## Training and selection

- CPU, one numerical thread; seeds tied deterministically to each data seed.
- Adam, learning rate 0.001, weight decay 0.0001, batch size 64.
- Full-catalog multinomial loss with TRAIN-context and PAD masked. Normalize
  target mass per episode so each episode contributes equally to mean loss.
- Checkpoints at epochs 0, 10, 30, 60, 100; choose maximum meta all-recorded
  nDCG@10, with earliest checkpoint winning exact ties. One trajectory per arm.
- Do not extend the budget or grid after development outcomes are inspected.
  A TRAIN-only timing/sanity check may run before source sealing.
- Save all checkpoint metrics, training curves, selected state and source/input
  hashes. Serialize and replay selected scores before sealing. Freeze all three
  seeds before computing development outcomes.

## Outcomes and interpretation

Primary: full-catalog all-recorded nDCG@10, same eligible candidates and users
for all arms. Also report MRR@10, recall@10, head/tail recommendation slots and
positive hits, and activity groups with explicit denominators. No new liked
rating or temporal outcome is silently substituted for the primary endpoint.
The existing liked-rating endpoint (rating at least four) and its groups are
secondary diagnostics using the same selected scores, with no separate tuning.

Report all arms and per-seed differences. Full versus no-pattern addresses the
additional pattern information; full versus marginal tests broader collaborative
information; full versus analytic donor support tests learned interpretation.
Comparison with locked EASE asks whether the resulting standalone predictor is
competitive in this pilot. Three overlapping splits are not three independent
datasets; do not advertise significance or fresh generalization from their mean.

For prioritizing further research, a 10% relative primary improvement over the
locked EASE mean, positive differences in every seed, and a positive full versus
no-pattern mean would be a substantial pilot signal worth independent
confirmation. This is a declared research target, not a promised outcome or a
definition of scientific novelty. Smaller gains remain smaller gains. Negative
results narrow this implementation and fixed budget, not all possible mechanisms.

Shareable evidence contains aggregates, source hashes and protocols. Individual
histories, query IDs, features, scores and checkpoints stay under ignored runs.
