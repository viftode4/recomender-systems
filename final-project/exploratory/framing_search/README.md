# Preserve the circumstances that gave an observation its meaning

Research decision, 29 September 2026. This is a framing investigation, not a
new model, an accuracy result, or an established original contribution.

**Follow-up:** the [predictive experiment](predictive/README.md) is now complete.
All 39 declared fits used a nested split of the original TRAIN data. True-group
nDCG@10 is 0.172555, versus EASE 0.172956 and mean shuffled-group 0.172819.
This implementation did not support the proposed ranking advantage. The text
below preserves the preceding framing decision and its falsifiable hypothesis;
its descriptive coherence finding is not a successful predictive result.

The user's priority is a simple explanatory principle with the potential for a
substantial advance. Model size, number of experiments and new terminology are
not substitutes. The next implementation must follow from a distinct prediction
and a defensible source of information. The required assignment and its existing
review artifacts remain separate from that larger, still unmet research goal.
Framing selection is the current priority; the previously proposed convergence
and richer-input experiments remain unrun.

## The lead

**Ask what produced a record before deciding what it says about a person.**

For this dataset, a concrete version is: preserve which ratings were submitted
together. A person's collection can combine observations from different occasions.
Flattening it into one item vector removes those distinctions before a model
can decide whether they matter.

This is a better-grounded next question than another imagined preference
mechanism. MovieLens records submitted ratings. Its timestamps are not movie
viewing times. In the 100K era, the interface displayed prediction-sorted movie
lists and allowed genre/release-date filtering. These facts make a shared
recording context plausible; they do not let us reconstruct an actual page,
intent, exposure set, or viewing session from a timestamp.
[Dataset creators, sections 2.2 and 3.2](https://files.grouplens.org/papers/harper-tiis2015.pdf).

Operationally, start with **equal-timestamp groups within each user**. This uses
an observed relation without inventing an order between tied records or choosing
a convenient session-gap threshold. A group can reflect form submission, clock
resolution, or another logging mechanism. It is not a verified psychological
episode.

## The different prediction

Consider the same four known movies and ratings, with different groupings:

| Known collection | Submitted grouping |
|---|---|
| A, B, C, D | {A, B}, {C, D} |
| A, B, C, D | {A, C}, {B, D} |

Every scorer using only that collection must return the same predictions for
both rows. This includes an arbitrarily powerful bag model, not only EASE.
A rule using the partition can distinguish them. For example, the constructed
contrast `same_group(A,B) - same_group(A,C)` is +1 in the first row and -1 in the
second. That proves the input distinction, not its usefulness in MovieLens.

Our prior categorical, pair, addressed-evidence and shared-evidence experiments
do not supply this partition to their query scorers. Their negative results do
not test it. Conversely, more information does not guarantee better prediction:
it can be irrelevant, redundant with item identities, or too sparse to learn.

The hypothesis is precise: **does the grouping of the known records help predict
which other record belongs in the collection, when the missing record's group
and timestamp are unavailable?** The score target and candidate set must stay
the same as the existing comparison.

## Why this is not being presented as an invention

The broad insight that MovieLens reflects interaction with its collection
interface is already studied by [Fan et al.](https://arxiv.org/abs/2307.09985).
Their experiments concern a later version of MovieLens; their particular
onboarding stages must not be imported into the 1997/98 data.

[Hidasi and Czapp](https://hidasi.eu/assets/pdf/eval_flaws_recsys23.pdf) show how
timestamp collisions and tie ordering can create spurious sequence patterns.
[Quadrana et al.](https://arxiv.org/abs/1706.04148) already model information
across sessions. Neither grouping histories nor recognizing collection effects
is a first-ever concept.

The potentially useful contribution would have to be more specific: a simple
way to use surviving group structure without being told the target group, a
measured advantage under matched comparisons, or a rigorous discovery about
what the benchmark rewards. This search has not established novelty priority
for such a method, and no method has yet been chosen.

## What was rejected as the next conceptual answer

| Candidate framing | What survives scrutiny |
|---|---|
| Leave unknown values free and complete the known constraints | A valid change of prediction, but observed-subset Gaussian-process inference and local ridge already implement it. [Derivation](CONSTRAINTS.md). |
| Let combinations change meaning, or require a coherent joint explanation | Genuine nonlinear counterexamples exist, but our pair model already represents the two-item example. EASE itself has a Gaussian joint interpretation. [Analysis](COMBINATIONS.md). |
| Model competition between available choices | Coherent choice theory, but the offered alternatives are not observed here; established choice models and relative-rating methods are direct precedents. [Analysis](CHOICE.md). |

Known methods can still be useful controls. Rejecting an invention claim does
not prove an approach would perform poorly.

## Evidence before architecture

The separately declared [TRAIN-only recording audit](recording_audit/PROTOCOL.md)
is complete on the original seed-2026 TRAIN split. It reads no held-out rating
or timestamp values and fits no recommender.

- 56,682 of 80,808 records (70.14%) share an exact timestamp with another retained
  rating by the same person. 942 of 943 users have at least one such group.
- The 66,203 within-group movie pairs have mean genre Jaccard similarity 0.234236,
  versus 0.184155 under 100 within-user shuffles preserving the movies, ratings,
  and timestamp multiplicities.
- Giving each eligible user equal weight also shows greater coherence: 0.231633
  versus a shuffle mean of 0.193739. The effect is not solely a consequence of
  weighting users with more within-group pairs more heavily.

[Reproducible results and denominators](recording_audit/results-v1/RESULTS.md).
An [independent review](recording_audit/REVIEW.md) replayed the counts and all 100
shuffle comparisons using literal genre-set intersections; five tests and all
current provenance hashes passed.
This establishes a property of the TRAIN grouping beyond the unordered
collection under this shuffle comparison. It does not establish accuracy gains,
causal preference, a recoverable screen, or usefulness when the target's group
is hidden. The groups were thinned by the original random split. This is one
descriptive split, not an independent population replication.

Any later prediction study must satisfy these conditions:

1. Form groups from visible context only. Remove hidden records before feature,
   donor and group calculations. Neither target timestamp nor true group identity
   may enter a query. Entirely hidden groups, including singletons, remain possible.
2. Compare matched models trained with true groups, shuffled groups, and no groups.
   Keep observations, ratings, target masks, capacity and optimization opportunity
   matched. Test-only shuffling is a different sensitivity experiment.
3. Preserve the established all-recorded benchmark. Separately assess liked-record
   ranking and whole-group holdout to distinguish completion of an existing
   reporting pattern from prediction in an unseen one. Those extra targets cannot
   substitute for a loss on the primary comparison.
4. State what genre, popularity and calendar-time controls remove. Conditioning
   them away asks whether extra group information remains; it is not the same
   question as whether groups carry useful information at all.
5. Compare with a timestamp-aware established method if a usable mechanism
   emerges. A win against a model denied the new inputs does not establish SOTA.

If true groups supply no predictive advantage under these conditions, stop
promoting this framing as a route to a stronger recommender. If they help only
within partly observed reporting groups, the finding is narrower than better
general preference prediction. Novelty and substantial improvement remain
separate requirements.

Further critique: [observation and assignment audit](OBSERVATIONS.md),
[adversarial framing review](CRITIQUE.md). No new recommender fits or additional
validation/test outcomes are part of this investigation.
