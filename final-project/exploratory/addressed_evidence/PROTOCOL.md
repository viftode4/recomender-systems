# Addressed categorical evidence, study version 1

Declared before the first real fit. Post-final-test exploratory research only.
The original MovieLens100K TEST has already been evaluated. This experiment
opens no TEST split or final-test result and makes no fresh-confirmation claim.

## Inputs and question

Use original TRAIN/validation splits for seeds 2026, 2027 and 2028. These overlap;
they are not independent datasets. Use the original 471-user meta-fit cohort
for checkpoint choice and the disjoint 472-user development cohort only after
both choices have been persisted. Both cohorts have been used before.

Question: does a signed distinct-source interaction improve a direct categorical
evidence predictor trained from scratch under the same opportunity to optimize?
The experiment tests this mechanism, not architecture novelty or psychology.

Construct a directed graph from TRAIN presence only: each candidate's top 64
positive cosine co-observation neighbors, exact ties by source index. Exclude
self edges and padding; fill unused slots with padding. Preserve ratings 1–5;
0 means absent. No baseline predictions, item metadata or user-ID embeddings
enter the model. Hash raw inputs, TRAIN categories, graph, source and runtime.

For candidate i and output category c, let s_j be the learned table entry for
source j and its observed rating. An absent source contributes exactly zero.

```
direct(i,c) = sum_j s_j / sqrt(64)
pair(i,c) = ((sum_j s_j)^2 - sum_j s_j^2)
            / (2 * 0.01 * sqrt(64*63/2))
additive_logit(i,c) = bias(i,c) + direct(i,c)
pair_logit(i,c) = additive_logit(i,c) + tanh(gamma(i,c))*pair(i,c)
```

Distinctness refers to source item identity. Every true source occurs once in
the graph. The formula removes self products, including when only one neighbor
is observed. Padding has no effects or trainable effective evidence.

## Initialization and numerical scale

Initialize compatibility tables independently Normal(0,0.01^2); biases and
gamma are zero. Both variants allocate identical tensors with identical initial
values. The additive variant disables its pair branch, so equal allocation does
not imply equal active capacity. Initial logits must be exactly identical.

The fixed pair_vote_scale=0.01 matches initialization units. Under independent
zero-mean source votes, a sqrt(pair-count)-normalized pair sum has variance of
order sigma^4; division by sigma restores the scale of first-order votes.
This argument is about initialization, not learned independence. The TRAIN-only
diagnostic found mean-pair normalization suppresses initial coefficient gradients
below Adam epsilon; fixed scale correction is set before validation fitting.
Keep the diagnostic, including the tested alternatives. No normalization is
chosen by development outcome.

## Fit and convergence policy

Adam, learning rate .001, weight decay 0, batch size 16, deterministic CPU
arithmetic with one numerical thread per process. Mask each user's TRAIN
interactions by identity using the existing seeded 80% context episode generator;
retain at least one probe. All five ratings participate in the joint recorded
item/category likelihood. Normalize over all items outside context and all five
categories, excluding padding. Average probe losses within user and then users.
An unobserved alternative is not asserted to be a disliked item.

Both variants share source graph, initializer seed, epoch masks, batch order,
optimizer settings and the following stopping rule:

- Maximum 2,000 epochs, a numerical backstop, not evidence of convergence.
- Evaluate meta-fit joint NLL at epochs 10, 30, 60, 100 and then every 20 epochs.
- Stop after 200 epochs without a cumulative absolute reduction of at least
  0.0001 relative to the most recent meaningful improvement reference.
- Select the exact minimum finite meta-fit NLL among reached checkpoints;
  exact ties retain the earlier checkpoint. This selection reference is separate
  from the meaningful-improvement reference used for stopping.
- Save every epoch's TRAIN probe loss and every scheduled meta-fit evaluation.
  Persist the selected model/logits and replay them exactly. Keep a latest
  resumable optimizer/model state without retaining hundreds of full checkpoints.
- Match every shared epoch's masks and order between variants. Actual epoch
  counts can differ under the same stopping rule and must be reported.

This is a fixed optimizer comparison. A plateau does not prove a global optimum.
If the backstop is reached while improving, label the fit budget-limited. Any
learning-rate, architecture or budget revision requires a newly declared study
and output directory; never overwrite this run or relabel it as prospective.
An interrupted run may resume only under identical source, protocol, runtime
and input signatures, restoring optimizer state and deterministic epoch indices.

## Evaluation and interpretation

Persist both selected checkpoints before computing development outcomes. Primary
mechanism comparison: pair-minus-additive macro joint NLL (lower is better).
Also report rating NLL/accuracy and all-observed and liked-record nDCG@10/Recall@10,
using the shared evaluator and exact same users and TRAIN-unseen candidate set.
All-observed score is logsumexp of five logits; liked-record score is logsumexp
of categories 4 and 5. Liked-record is not exposure-corrected preference.

Report all seeds, equal-seed descriptive means, per-user paired differences and
how often the pair model improves. Do not select a successful endpoint after
seeing outcomes. Existing EASE/SLIM/PositiveEASE development measurements are
context only and must use the same cohort, split, endpoint and catalog to compare.
No new test statistics on the old final test, no SOTA claim, no population claim
from three overlapping splits, and no substitution into the old frozen report.

The one-epoch benchmark uses TRAIN only and measures feasibility. Its models
are discarded. Real fits start fresh. Record benchmark wall time before making
runtime estimates, and distinguish estimated from observed completion.
