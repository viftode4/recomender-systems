# Critique of candidate framings

29 September 2026. Conceptual review only. No ratings, TEST records, model fits,
or new evaluation results were accessed. This note distinguishes a useful
question from an established new contribution.

## What kept recurring

Our earlier work repeatedly changed the machinery while keeping the same object:
a user is an unordered bag of recorded movies, and the task is to recover another
record from that bag. Categorical channels changed what a recorded rating means;
pair features changed how records combine; evidence-transfer changed how donor
support is summarized. These are substantive experiments, but none establishes
that the bag is the most informative unit supplied by the data.

Several appealing suggestions are already present in our own project:

- “One observation changes the meaning of another” is the explicit premise of
  [DESIGN.md](../DESIGN.md), including conjunction and exceptions.
- “Learn a small language of explanations” is the existing structural-program
  and operator-library direction.
- “Different parts of the history matter for different candidates” overlaps
  [NAIS](https://arxiv.org/abs/1809.07053), which learns candidate-dependent
  importance of historical items.
- “Learn interactions among the history, rather than adding evidence” overlaps
  [DeepICF](https://arxiv.org/abs/1811.04392), which models higher-order relations.

Restating these in human terms would not answer the user's request for a new
perspective. The exact pair experiment also already tested a much larger family
than a simple conjunction slogan suggests. Its null result does not rule out all
contextual interpretation, but it removes the justification for calling more
interaction capacity an obviously superior concept.

## The strongest alternative question

**What question was the person answering when this record was made?**

A logged rating need not be a spontaneous independent expression of taste.
Someone may be filling a page of recommendations, working through a genre
filter, or recalling a collection of old films. The unobserved page or reporting
occasion determines which missing records are plausible. This changes the
primitive from a lifelong user profile to an observation-producing occasion.

The dataset creators document that early MovieLens showed recommendations and
allowed title, genre and release-date filtering. Before v4, predicted ratings
ordered movie lists; rating an item could alter the next page. Thus shared
logging circumstances are a plausible source of structure, rather than an
invented account of users' emotions. These historical interface facts do not
identify any particular user's page or prove that timestamp bursts recover it.
[Harper and Konstan, pp. 5–7](https://files.grouplens.org/papers/harper-tiis2015.pdf).

The broad insight is established. [Fan et al.](https://arxiv.org/abs/2307.09985)
already analyze MovieLens interaction generation and the influence of the
platform's own recommender. Our candidate contribution would have to be a precise
new inference mechanism or a previously unmeasured consequence, with direct
comparison against that prior work. This review has not established either.

## One falsifiable proposal, before an architecture

**Preserving reporting occasions makes randomly hidden records easier to identify
than the same observed item/rating bag with the occasion assignment destroyed.**

The eventual experiment should expose exactly the same observed records and
candidate catalog in both arms. One arm retains groups defined solely from
observed TRAIN timestamps; the other permutes group membership within the user
while preserving group sizes. Masked target timestamps cannot be supplied to
inference. The proposal predicts a reproducible gain for the true groups beyond
what history length, global time, item frequency and genre alone explain.

All choices defining groups and comparisons must use development data. An
experiment that constructs groups using the complete history before hiding
records would have leaked the hidden targets into its query representation.
A target used to fit a collaborative provider would also invalidate an inner
holdout; such a comparison needs a properly excluded training construction.

The conceptual signature is specific: a bag-invariant predictor cannot react to
changed occasion assignments. A useful occasion-aware method should react to
assignments that carry evidence, and lose that advantage under the matched
permutation. This is a proposed diagnostic, not a claim that sessions identify
psychological intent or that a model has already passed it.

Reject this direction if grouping has no reproducible incremental information,
if only hidden timestamps enable it, or if any gain is explained by a trivial
calendar/popularity control. A negative result should end this hypothesis,
rather than prompt a larger network on the same unsupported premise.

## Strongest objection

Success could reconstruct the behavior of a 1990s website rather than produce
better recommendations for a person. Random held-out rating completion can
reward predicting the old interface's prompts. That is still relevant to
understanding this assignment's benchmark, but it cannot be presented as a
universal preference breakthrough. Rating timestamps are not viewing timestamps,
and close submissions do not prove common exposure. A chronology-based task
would ask a different question and must be reported separately.

This is the strongest untested framing found in this branch because it questions
the data-generating unit and has observable consequences. It is not yet the
novel, winning concept requested by the user. No simple unexamined framing should
be sold as that result merely because it sounds plausible.

## Adversarial review of the exact-timestamp partition proposal

**Verdict: logically distinct input, valid research hypothesis, not yet a novel
concept or evidence of predictive benefit.**

For the same item/rating bag `H`, partitions `{{A,B},{C,D}}` and
`{{A,C},{B,D}}` cannot change a function of `H` alone. A function of `(H, Π)`
can distinguish them. This is a precise representational distinction. It does
not show that the distinction matters to hidden candidate membership in the real
distribution. Histories seldom repeat exactly, so the example is an invariance
argument, not an empirical matched pair or a measured conditional-information
result. Restrict “all previous scorers” to those actually invariant to timestamp
assignment; any earlier model with temporal inputs needs separate inspection.

Five conditions prevent overinterpretation:

1. **Call them equal-timestamp groups.** Identical recorded seconds could reflect
   one form submission, clock precision or ingestion behavior. Without logging
   documentation they do not establish a single viewing or recording event.
   No order is identifiable within a tie. Build the partition from surviving
   known records only, preserving exact stored timestamp equality.
2. **The target group is unknown.** At inference there is no active group query.
   It is legitimate to integrate over observed groups using a rule based only
   on known context, but an entirely hidden singleton group leaves no surviving
   group. A model must account for that case instead of silently assigning every
   target to an observed group. Using a hidden target's time, group size or
   surviving-group identity would change the task and leak information.
3. **A shuffle is not a causal intervention.** Within-user item reassignment to
   timestamp slots preserves the bag, timestamp multiset and group sizes, but
   can destroy item popularity/time and genre coherence together. It is a useful
   predictive ablation, not proof that recording occasions caused the outcome.
   A genre/popularity/time-matched shuffle tests a narrower residual hypothesis;
   report which distinctions it conditions away and whether enough swaps remain.
4. **Match training and inference conditions.** Test-only shuffling can create
   a distribution shift and punish any contextual model. Compare equally trained
   true-group and shuffled-group arms as well as optional test-time sensitivity.
   Every hidden probe must be excluded from learned providers. Choose shuffling,
   grouping and scores before inspecting their assessment outcomes.
5. **Grouped histories are established inputs.** Session-aware recommendation
   already combines history within and across sessions, for example
   [Quadrana et al.](https://arxiv.org/abs/1706.04148). That paper's sequential
   task is not identical to unordered random-record completion, but the broad
   move from one bag to grouped history cannot be claimed as an invention.

A defensible contribution would be a simple mechanism that uses the partition
without the unavailable target-group label, plus evidence that it improves the
same prediction task under matched controls. A discovery of recoverable
interface artifacts would instead be a benchmark finding and should be named as
such. Neither requires pretending the general idea of session or basket context
has not been studied.
