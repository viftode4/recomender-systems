# Personalized rejection information audit

This protocol is written before fitting or evaluating this experiment. It is an
exploratory follow-up to the failed preference-contrast models, using previously
examined validation splits. It does not establish a new method's priority.

**Question.** Do the locations of a user's confirmed dislikes add useful
prediction information beyond their likes, their dislike count, and the
catalog's item-specific dislike counts? Does assigning more weight to dislikes
that a positive-only model would predict help beyond an equal-mass shuffled
weight control?

## Data and separation

- Reuse the original training/validation splits for seeds 2026, 2027 and 2028.
  Only those pair IDs authorize parsing ratings from the atomic file. The test
  split is never opened and its ratings are never parsed or retained.
- Deterministically shuffle each user's training pairs independently of rating
  values, retaining floor(0.8*n) pairs as context and the rest as probe, with at
  least one pair in each set. Likes are actual ratings >=4, dislikes <=2,
  and rating 3 is neutral. Missing observations are unknown.
- Teacher and context representations use context ratings exclusively. Decoder
  targets are binary probe likes. Zero target entries mean absent positive
  labels in a squared reconstruction objective, not confirmed dislike labels.
- The decoder fits users having at least one context like; the minimum number
  of probe likes is zero. Report excluded fitting rows and fitting users with no
  probe likes. The positive teacher fits all context rows.
  Evaluation retains all validation users; liked metrics exclude only users
  without a validation like. Report context/probe support counts, without
  selecting favorable subgroups or dropping unsupported scoring users.
- The primary prediction uses context features only. A predeclared secondary
  prediction uses all original training ratings with the SAME fitted decoder
  and context-fitted teacher. This has more known history and a context-length
  distribution shift. Report both, never choose the better access mode.
- Both modes mask ALL originally observed training items, including neutral and
  disliked items and probe targets. No candidate seen during training is ranked.

## Fixed architecture and five branches

A positive-only EASE teacher fits context likes with ridge 250 and a zero
diagonal. Its scores are converted to within-user midrank percentiles over
non-liked, non-padding items. A confirmed dislike receives raw weight
0.1 + 0.9*percentile; normalize these weights to mean one over that user's
dislikes. Thus surprise weights preserve each user's total dislike feature mass.

Every decoder minimizes squared probe-like reconstruction error plus ridge,
without an intercept. Each item's own positive and, where present, negative
coefficient are fixed to zero. The branches are:

1. Positive context features only.
2. Positive and ordinary dislike channels.
3. Positive and surprise-weighted dislike channels.
4. The same channels after permuting weights among each user's existing
   dislikes, preserving identities, weight multiset and mass; five seeds.
5. Positive and ordinary dislike channels after admissible negative/neutral
   2x2 switches, preserving every observed pair, every positive, every user's
   dislike count, and every item's dislike count; five seeds.

Switch proposals depend only on fixed nonpositive observed support. The budget
is min(50 * number_of_dislikes, 500000) proposals per chain, including rejected
proposals. Restrict switches to user pairs within the same context-defined
decoder-fit eligibility stratum, preserving item dislike counts within the
actual fitted subset as well as globally. Report acceptance, moved-label
fraction, support overlap, and exact
margin/support invariants. Structural zeros may disconnect the switch space;
these are descriptive conditional perturbations, not proven uniform draws,
convergence-certified chains, or conditional-randomization p-values.

For full-history null queries, retain the switched context labels and separately
switch the probe's nonpositive observed labels with another fixed seed before
adding them. This preserves the decoder's context perturbation and the combined
history's user/item dislike margins. Full-history permutation queries use the
same fixed per-replicate random seed on their enlarged dislike sets; the weight
multiset is preserved separately within each inference mode.

## Selection and reporting

- Primary comparison: common fixed ridge 250, without selection.
- Secondary comparison: equal ridge grid [50, 250, 1000]. Select each conceptual
  branch using strict-context liked nDCG@10 on the original meta-fit user half.
  For controls, average meta-fit scores across all five replicates before
  selecting one common ridge; never choose a favorable replicate. Resolve ties
  toward the first grid value. Use the same selection in both inference modes.
- Commit selections before evaluating the development user half. Primary
  outcome is liked nDCG@10. Also report all-observed ranking and the proportion
  of recommendation slots containing a known validation dislike. Missing
  ratings do not establish satisfaction or safety.
- Report plain channels minus positive-only, surprise weights minus plain
  channels, surprise weights minus mean shuffled weights, and plain channels
  minus mean placement-null. Average control per-user metrics, not scores.
  User bootstrap intervals are descriptive and conditional on the five control
  draws, without multiplicity correction. Also show every replicate and its
  range. A benefit over shuffled weights alone would not separate individual
  positive-profile surprise from item-popularity-correlated teacher scores.
  No inferential p-value from
  null chains, no post-selection groups, no claim of independent dataset
  replications, and no SOTA claim.
- Do not increase grids, switch budgets, or model budgets after seeing these
  outcomes. Save code/data/split/protocol hashes, timing and label-access audit.

## Prior art and scope

Harder confirmed negative weighting already appears in [LAGCL4Rec
(2025)](https://aclanthology.org/2025.findings-emnlp.61.pdf). The contribution being
tested here is a carefully controlled information question, not that broad
weighting idea. [Rapallo and Yoshida (2010)](https://arxiv.org/abs/0905.4841)
distinguish full-table switch connectivity from incomplete-table cases; our
observation support is incomplete. Model performance on these reused validation
users is exploratory until a separately frozen, authorized test stage.

## Amendment before decoder fitting or metric evaluation

Independent code review caught that a global label switch would not preserve
item dislike counts within a decoder subset originally chosen by requiring a
probe like. The first launch was interrupted during control preparation, after
the teacher fit but before any decoder fit or validation metric. The corrected
protocol retains zero-probe-like target rows and defines fitting eligibility
from context alone; controls are restricted to that fixed context-defined
stratum and its complement. This prevents probe-dependent feature construction
while holding the fitted matrix's item dislike counts fixed. The interrupted
preparation remains in `runs/negative-information-v1`; the corrected run uses
`runs/negative-information-v2`. No outcome motivated this amendment.
