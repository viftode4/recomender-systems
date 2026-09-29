# Predictive check of rating-recording groups

This protocol is fixed before model fitting. The question is whether restricting
item-pair evidence to observed rating-recording groups improves ranking compared
with the same item-pair dictionary over unordered user histories, timestamp
shuffles, and tuned EASE. This is an exploratory nested experiment after previous
use of MovieLens100K, not an untouched population test or proof of novelty.

## Information boundary and split

Use the original seed-2026 TRAIN pair IDs only, verified through the categorical
input-signature hash chain to `runs/research-v2/2026-EASE-1/manifest.json`.
Verify original TRAIN, raw interaction, metadata, and source ID-catalog hashes.
Never open original validation/test splits, scores, or metrics.

Before parsing any rating or timestamp value, partition each user's original
TRAIN items by ascending SHA-256 of compact JSON `[20260929,user,item]`, with item
token as an exact-hash tie breaker. For n records, assign the first
`n-2*max(1,floor(.1*n))` to F, the next `max(1,floor(.1*n))` to D, and the remainder
to A. This approximates 80/10/10 with equally sized D/A; every user has all three.
F is fit/context, D chooses hyperparameters, A assesses frozen choices. Save A
identities to an ignored file, then remove them from the fitting/selection path.

Parse timestamp values only for F-member raw rows. Rating values are not model
inputs. F timestamps define same-user exact-integer-timestamp groups. No target
timestamp, target group, validation context, or assessment context is supplied at
prediction. Use the full original source catalog including all 1,682 real items;
exclude PAD and each user's F observations from ranking. D observations remain
eligible during A evaluation. Every model uses F-only context; there is no refit
after selection.

## Models and finite grid

Use one fixed conceptual dictionary of all unordered pairs of real catalog items, with the
dictionary construction, score equation, exclusion constraints and normalization
specified in `model.py`. The same dictionary is
used by grouped, shuffled-group and bag variants. Zero or absent columns may be
implicit; there is no pair feature selection or truncation. All use singleton indicators
plus pair indicators. Grouped pair inputs are one only when both films occur in
the same F recording group; bag inputs are one whenever both are in F. Exclude
the target's own singleton feature and every pair containing that target during
reconstruction. Targets are F observation indicators; missing records are the
reconstruction-zero convention, not confirmed dislikes.

Three shuffle realizations use independent `default_rng(2026092900+r)` for
`r=0,1,2`, processing users in catalog order. Each permutes a user's F item records
among their existing timestamp slots, preserving movies and timestamp group
sizes. They are separately fitted/selected/reported and never score-ensembled.

Each of `true_group`, `shuffle_0`, `shuffle_1`, `shuffle_2`, and `bag` has six
candidates, in lambda-major order:

- lambda = 50, 250, 1000;
- positive beta = .1, 1.

Pair penalty scaling is computed separately for each arm from F-only total pair
feature squared mass divided by singleton squared mass. Record the values;
grouped and shuffled group-size mass agree exactly because the complete dictionary
is used. Shuffled fits use the equal true-group normalizer, checked explicitly.
This is a declared deterministic normalization, not a tuned quantity.

EASE is separately fitted on F with nine lambdas in order:
10, 30, 50, 100, 250, 300, 1000, 3000, 10000.
Total: **39 fits**. No zero-beta fallback in the proposed-model search, no
additional sweep, and no ensemble. Every arm selects the first exact maximum
of D all-recorded macro nDCG@10. All six selections and their full F-context score
arrays are sealed before any A metrics or A rating/timestamp values are parsed.
Freeze source/protocol/runtime/input digests before any fitting and verify them
again at the selection barrier and completion.

## Assessment

Primary: all A recorded movies are relevant. Report macro nDCG, recall, MRR,
precision, hit rate at ten; novelty, catalog coverage, genre diversity/calibration,
F-activity terciles, and F-popularity top-20% head versus tail. Use the existing
independent metric/group helpers with their source hashes frozen. Rank ties by
original catalog index. Report every candidate's D aggregate metrics and timing,
selected specifications, all selected A metrics, and exact split denominators.

Secondary, after the seal only: parse A ratings/timestamps for evaluation. With
unchanged scores/candidates, report subset-truth results for rating >=4, A rows
whose timestamp appears in that same user's F context, and A rows whose timestamp
does not. Exclude users with empty subset truth and state denominators. These are
diagnostics, not a new whole-group holdout design, and never select/refit models.

Paired user bootstrap: 2,000 resamples with replacement over all primary A users,
RNG seed 2026092901; report mean delta and percentile 95% interval for nDCG@10 and
recall@10 comparing true-group against bag, tuned EASE, and the arithmetic mean
of the three separately selected shuffle **per-user metrics**. This last
comparison averages metrics, not scores or ranks. Intervals are descriptive
within this fixed split, not correction for prior exploration or three
independent population samples.

A predeclared promising result requires true-group mean primary nDCG at least
10% relatively above tuned EASE, and greater than both bag and the mean shuffle
metric. Report whether each condition holds, including failures. This threshold
is a research prioritization rule, not a novelty or statistical-significance test.
Separately report whether the paired descriptive nDCG intervals have lower bounds
above zero against both bag and mean-shuffle; this is the declared mechanism
support check. It remains descriptive because of the prior exploration.

## Artifacts and safeguards

Keep identities, F timestamps, split tables, predictions, and per-user outcomes in
ignored `runs/framing-predictive-v1/`. Publish only aggregate JSON/Markdown,
source/protocol hashes, timing and reproducible commands. Use the frozen Python
environment and one numerical thread; no installs. Synthetic tests must pass
before fitting, including value-parsing membership, complete selection barrier,
target-own-feature exclusion, and ranking/metric reference checks. Retain negative
findings and do not change this grid or protocol after viewing A.
