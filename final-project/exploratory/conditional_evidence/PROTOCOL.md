# Conditional evidence: raw relationships and one additional case

Implementation of the user's approved plan, 29 September 2026. This protocol
must be source-sealed before fitting the real study. TRAIN-only timing and
synthetic checks are preparation, not scientific results. MovieLens 100K is
reused development data; its original TEST is already exposed and remains
unopened by this study. Earlier coursework and research artifacts are immutable.

## Claim and endpoint

Test whether retaining the arrangement of collaborative observations improves
ranking beyond the previous summaries and a matched scrambled representation.
The primary endpoint is macro-user, full-catalog, all-recorded nDCG@10. Missing
records are competing outcomes, not confirmed dislikes. The substantial-result
target is at least 10% relative mean improvement over the strongest matched
standalone reference, positive differences in all three splits, and positive
mean differences against both mechanism controls. This is a research target,
not a promised result, significance threshold, or definition of novelty.

This small reader belongs to the established local rating-graph family, including
[IGMC](https://arxiv.org/abs/1904.12058). Accessible records and bounded decisions
are inspiration from RAG and Jev; there are no API calls, pretrained weights,
teacher predictions, text generation, or ensemble scores in the custom model.

## Data boundary

Use the hash-verified original TRAIN categories, ordered identities, and existing
471 meta-selection / 472 development user cohorts for seeds 2026, 2027, 2028.
Only meta users' validation item identities reach model selection. A new loader
filters by user before parsing validation item fields; it does not reuse the
old helper that materializes all validation pairs. Raw rating values are parsed
only after TRAIN membership checks. No TEST file is opened.

The query user's complete donor row is excluded from retrieval, item frequencies,
normalizers, and all evidence. The query node receives its visible history only.
Other users' TRAIN observations of the candidate are permitted evidence. Query
and item population degrees are computed after exclusion, so a hidden query
record cannot affect even an initial node feature. Padding item zero is never
eligible. Histories with fewer than two observations are counted and omitted
from training episodes, without fabricating records.

For each epoch, create one 80% and one 90% retained-history episode per eligible
TRAIN user, with deterministic independently keyed masks. Retain floor(fraction
times history size), clipped to [1, size-1]. Every omitted TRAIN item is a target.
Sample min(128, U) alternatives uniformly without replacement from the U real
catalog items absent from the user's complete TRAIN history. All controls share
the same episodes, sampled alternatives, order, and initialization seeds.

## Evidence and model

For each query and candidate, select eight other users by descending binary
cosine similarity to the visible query history, followed by eight additional
nearest users who recorded that candidate. Deduplicate and use available rows
when fewer exist. Exact ties follow the original stable user ordering. Zero
similarities are eligible; no rating-dependent tiebreaking is allowed. Keep all
visible history columns and all actual donor ratings on those columns and the
candidate. Missing cells are absent edges, distinct from each rating 1 through 5.

Graph nodes have four role indicators (query, donor, history item, candidate)
and log(1 + permitted TRAIN degree); the query degree is visible context size.
Three separate message-passing layers of width 16 apply self transformations
plus mean messages within each rating type, followed by SiLU. Bidirectional
edges share their rating transformation. Empty rating neighborhoods contribute
zero. The output concatenates query, candidate, and mean donor embeddings and
passes them through a 48 -> 16 -> 1 SiLU head. No learned identity embeddings.
Zero-support candidates use this same deterministic model, not random embeddings
or an undeclared fallback expert.

Arms:

1. **raw:** the graph above.
2. **summary:** the old nine-feature interpretation, recomputed from precisely
   these selected donors, with a 9 -> 16 -> 1 SiLU scorer. The nine features are
   log context count, log support, support fraction, log weighted support,
   weighted support fraction, mean supporter weight, log Kish count, log pattern
   count, and pattern coverage. All logarithms use log1p. This control discards
   arrangement and has fewer parameters by design; do not claim equal capacity.
3. **scrambled:** exactly the raw architecture, with rating-preserving 2-switches
   on donor/history edges. Per-donor and per-history-column rating counts remain
   unchanged; query edges, candidate edges, retrieval, and initial degrees are
   unchanged. Make at most 20 proposals per history edge using a deterministic
   per-episode/candidate seed; reject collisions and invalid switches. Record
   attempted and successful swaps, including no-switch graphs. This intervention
   changes higher-order correspondence and does not preserve all nine summaries.

Scrambling is deterministic and independent of candidate batching. All graph
representations retain common-column correspondence. Consistently permuting
rows or columns of a constructed graph preserves its prediction (up to declared
floating-point tolerance); changing identity tie order can change retrieval.

## Optimization and checkpoints

The sampled loss is log(sum_positive exp(score) + U/m * sum_sampled exp(score))
minus mean positive score, calculated stably with logsumexp. When U=0 the
alternative term is absent. m=min(128,U). The denominator estimate is unbiased;
its logarithm is not an unbiased full-softmax loss estimator. Do not claim exact
full-catalog training or calibrated probabilities. Exhaustive sampling must
recover the full-catalog loss exactly. Average equally over episodes; two
episodes per eligible user give equal user weighting.

Adam, learning rates 0.001 and 0.0003, weight decay 0.0001, eight query episodes
per optimizer update. Gradients from candidate microbatches must equal the
declared per-query loss; splitting candidates may not change the denominator.
Use deterministic CPU operations and one numerical thread. Raw and scrambled
start from identical parameters. Summary initialization uses the same seeded
stream but cannot have identical shapes or parameter counts.

Run all three arms, both rates, and three seeds: 18 initial trajectories.
Checkpoints at 0, 25, 50, ..., 300 use exact full-catalog meta nDCG@10. Mask
visible TRAIN observations and PAD; retain existing stable ranking ties. Select
the best checkpoint within each trajectory, then learning rate; exact ties take
the earlier epoch and then the earlier declared learning rate. If any trajectory
selects epoch 300, extend every trajectory, preserving optimizer state, to
checkpoints 325, 350, ..., 600 before development access. Report a selected cap
as an unresolved optimization limit; no further extension is authorized by this
protocol. Checkpoint both latest and best states, optimizer, episode/RNG state,
runtime, source/input digests, and learning curves. Resume rejects mismatches.

The timing pilot measures build, forward/backward, full-catalog inference, memory,
and extrapolated study costs using TRAIN only. Performance changes may alter
batching or caching while preserving the mathematical operation; changing the
model, objective, candidate budget, or schedule requires a new protocol version.

## Comparisons and selection barrier

Compare with expanded-grid EASE, SLIM, categorical reconstruction and a fixed
cosine neighbor reference on the same ordered inputs and candidate rules.
Reuse previously locked scores only after exact provenance checks; disclose
their historical search budgets. The strongest standalone reference is chosen
by mean meta performance before development, not retrospectively by development.

All seeds, arms, learning-rate choices, selected state and replayed catalog scores
must be sealed before development loading. The optional retrieval decision and
its outputs, when triggered, are sealed too. Reading training contexts of
development users is allowed; reading their held-out outcomes for selection is
not. No development labels are fitting labels for the reader or selector.

## Conditional one-case retrieval

Run this extension only when the raw reader's mean meta nDCG is strictly greater
than each control and every matched standalone reference. A failed gate ends
this extension for this protocol, not the broader scientific hypothesis.

For each query the eight actions are the next eight cosine-ranked other users
outside its initial global eight neighbors. Add one selected donor to every
candidate table, deduplicating candidate-specific existing donors. Compare with
no action, the first eligible action, and a deterministic uniform random action
from the identical pool. At most one donor is added; no hidden adaptive loop.

Produce utility supervision with three deterministic user folds inside TRAIN.
For each fold, initialize the selected reader architecture/configuration anew,
fit using only other-fold queries and donors for the selected epoch count, then
freeze it. Entire held-fold rows are excluded from that fold's donor bank. Hidden
TRAIN probes from held-fold users label each action by the reduction in exact
full-catalog multinomial loss, including score changes to competing candidates.
Increasing confidence alone is not a reward. Report the best positive-gain
oracle action as a privileged opportunity diagnostic, never deployable accuracy.

Fit one utility MLP with width 16, SiLU, scalar output, Adam at 0.001 for 100
epochs and squared-error targets, using only out-of-fold TRAIN labels. Inputs
are frozen query/donor representations and overlap counts. At inference use
the full selected frozen reader and select the maximum predicted utility only
when it is strictly positive; stable pool order breaks ties. No actions means
no change. Report the distribution change between crossfit and full donor banks.
Include all fold fitting and utility construction in compute accounting.

## Outcomes, confirmation and reporting

Report every arm's per-seed primary metric, MRR@10, recall@10, liked-rating
nDCG@10, TRAIN activity groups, head/tail recommendation and hit counts, runtime,
memory, and evidence budget. Show paired descriptive user-level differences;
three overlapping ML100K splits are not independent populations. Actual examples
are selected by a fixed rule, not handpicked after looking for a story. Share
only aggregates and synthetic examples; raw user traces stay in ignored runs.

If the substantial-result criterion holds, the fixed model can be confirmed on
MovieLens 1M via a separately sealed 70/10/20 per-user rating-independent hash
split, with all-recorded ranking primary and no refit after validation selection.
Catalog and zero-support outcomes remain explicit, and no cross-release identity
join is permitted. All baseline grids and model choices freeze before the single
reserved evaluation. Data acquisition must use permitted personal tooling.

Keep the existing assignment core unchanged. Research findings enter a separately
versioned supplement only after verification. Names, actual group contributions,
and personal peer feedback remain deferred. No grade or first-ever claim follows
automatically from completion of the implementation.
