# Research direction: let evidence change the explanation

Status: active research after the original final test. The completed assignment
study remains in `reports/final-review-v1`. This directory contains new diagnosis,
implementations and experiments, not an amendment to the old frozen evidence.

The user explicitly allows substantial training and sustained investigation.
Runtime convenience must not become a scientific conclusion. Local execution
currently has CPU support; CUDA and usable MPS are unavailable. No paid compute
has been provisioned.

## The question beneath the architecture

A rating is a recorded event: somebody encountered, chose and rated something.
It is not a direct measurement of an enduring preference, and a missing record
does not establish dislike. We can learn the distribution of recorded outcomes
from this dataset. We cannot identify the whole exposure process, private mood
or causal satisfaction from ratings alone.

The ambitious hypothesis is that a history should support a revisable predictive
explanation. A new observation might reinforce an existing reason, expose an
exception, or show that two conditions matter together. Those are different
operations; simply adding another weighted vector need not express them well.

An example hypothesis is: two otherwise weak pieces of evidence together support
a candidate, while another observation weakens that conclusion. The model must
earn that interaction on withheld evidence. Calling the observations "mood" or
"conflict" does not make those psychological meanings identified.

"Least resistance" becomes a useful engineering question: which representation
makes accurate generalization easy to learn and cheap to express? We use
predictive loss plus an explicit structural penalty. We do not assert a physical
law, or equate fewest parameters with best predictions. Our current node penalty
is regularization inspired by minimum description length, not a derived code.

## What the existing failures actually tell us

[The field diagnosis](field-diagnosis-v1/FINDINGS.md) finds that a TRAIN-only
rank-16 compression of EASE predictions retains mean development nDCG@10
0.26020 versus 0.26183 uncompressed. The adaptive field reaches 0.22193.
Therefore a compact useful representation exists; this does not prove that
training a rank-16 predictor can find it. The field is nonlinear and has no
simple rank-16 score bound.

Its route weights are not uniformly collapsed, and its decoder remains active.
Route-weight diversity alone does not establish diversity of port values.
All six fixed/adaptive fits still improve from epoch 300 to 400. Representation,
optimization and objective choice remain competing explanations. A bigger model
or longer run is a testable intervention, not a guaranteed repair.

## Three connected experiments

**Preserve evidence until it reaches the candidate.** The new addressed-evidence
model learns candidate/source/rating compatibility tables directly. It uses no
trained expert scores or user embedding. A matched extension adds products of
distinct observed sources. This checks whether explicit evidence identity and
interactions work before introducing program search. Both variants are trained
from scratch under the [prospective protocol](addressed_evidence/PROTOCOL.md).
The additive control is necessary even if it wins.

This pair extension has a substantial restriction: for each candidate/output
category, its off-diagonal interaction coefficients factor through the same
source potentials and one signed scalar. It cannot express arbitrary pair
coefficients or an unrestricted logical theory. It is a focused diagnostic
model in an established higher-order family, not the full research vision.

**Change structure, not only coefficients.** The implemented
[structural-program proof](structural_program/README.md) selects one expression
tree using supplied max/product operators. FIT creates and calibrates a bounded
pool; PROBE selects one tree; a third synthetic sample assesses it after the
selection is saved. It recovers a noisy conjunction, removes an exactly redundant
branch, and fails a deliberately unrepresentable XOR task. An independent review
reproduced its results and passed all eight tests. The clean atoms are supplied;
no usable MovieLens atom learner or learned exception operator exists yet.

**Learn a language of predictive explanations.** The longer-term architecture
would alternate learning shared evidence predicates, searching small personal
programs, and compressing repeatedly useful subprograms into reusable predicates.
Proposed edits include conjunction, merge, deletion and signed exception. Each
edit needs predictive evidence beyond the examples that suggested it. This is
the research destination, not a description of software already implemented.

The central bet is specifically that a structure edit justified by disjoint
history probes generalizes better than coefficient-only adaptation with similar
search resources. That claim could be false, especially with short histories.

The subsequent [90-case synthetic stress test](structural_program/stress-v2/RESULTS.md)
gives this warning concrete form.
With 16 FIT and 16 PROBE examples and no signal, beam search has evaluation NLL
0.8073 versus 0.6968 for an intercept, and loses in all five seeds. For a noisy
conjunction it loses to the intercept at size 16 (0.6719 versus 0.6259) but improves
at size 256 (0.4642 versus 0.5923). These sizes count calibration examples, not
MovieLens history length. More search does not create missing information.
This motivates learning shared predicates and retaining a strong simple default
before personal structure search. The stress test does not establish that either
remedy will work on real ratings.

[The optimization derivation](addressed_evidence/GEOMETRY.md) also establishes
that the additive model's joint loss is convex under fixed features and
parameter-independent masks. The pair extension introduces nonconvexity and
restrictive algebraic ties. A separately declared full-gradient optimization
study can therefore help distinguish an optimization failure from insufficient
representation; it was not part of the completed Adam experiment.

The [completed addressed-evidence experiment](addressed_evidence/results-v1/RESULTS.md)
does not establish a useful new recommender. Across three seeds, the additive
variant reaches development all-observed nDCG@10 0.18544 and the pair variant
0.18983, against matched EASE 0.26183. The primary joint NLL worsens from
7.80988 to 8.64587 with pair interactions. The pair variant selects epoch 10
in every seed and stops at epoch 220; the additive variant selects epoch 100
and stops at epoch 300. More interaction capacity did not produce better
likelihood in this study. The new [research map](deeper_research/RESEARCH_MAP.md)
investigates reusable computation in a setting where the learning question can
be inspected exactly; it is not a claimed fix for these MovieLens results.

## Gates for a real personal-program recommender

1. Establish a competent globally trained evidence representation and inspect
   learning curves. Report score, likelihood and computation; retain failed fits.
2. Extend the synthetic suite with noisy, correlated, irrelevant and contradictory
   atoms. Include equal-budget random edits and coefficient-only controls. A
   larger grammar must justify its search cost and extra selection opportunities.
3. Construct real-data atoms without using a hidden probe's label in their
   training. Use user-fold cross-fitting or an equivalent audited episodic
   exclusion scheme. Hiding a record only at inference is insufficient if the
   proposed inner validation relies on a provider already fitted to that fact.
4. Split a user's known history into support, fitting probes and selection probes.
   Do not label all missing items as dislikes. A real recorded-event/rating loss
   must replace the toy's fully observed binary labels. Define what happens for
   users with too little evidence before evaluating them.
5. Compare fixed structures, coefficient-only adaptation, random edits and the
   strongest matched higher-order control. Count all searched candidates and
   elapsed computation. Save the chosen predictor before assessment.
6. Freeze the design and test on an untouched supplementary dataset. The old
   MovieLens100K test is already exposed; new experiments on its development
   splits are exploratory. See [dataset expansion](DATASET_EXPANSION.md).

With more time, optimize these weak links. Do not spend all available compute
scaling a mechanism before checking whether its objective can identify the
behavior we want. If the new models fail, that informs the next intervention;
it does not justify changing the endpoint or hiding the failure.

## Prior art and the remaining novelty question

[Cheung et al. (2012)](https://arxiv.org/abs/1208.2925) already learn personal
recommendation programs from likes/dislikes. [CF-NADE](https://proceedings.mlr.press/v48/zheng16.html)
already models categorical ratings from partial contexts. [Steck and Liang (2021)](https://recsys.acm.org/recsys21/session-2/)
study signed higher-order recommendation. [LINN](https://arxiv.org/abs/2008.09514)
learns logical operations for recommendation. [DreamCoder](https://arxiv.org/abs/2006.08381)
learns reusable program abstractions. These are direct antecedents, not proof
that our eventual combination has been published or that it is new.

Novelty remains unestablished. The strongest possible contribution is a precise
mechanism, a clear reason it should work, and measurements that survive the
controls. “Nobody thought of this” and a promised grade are not research results.

## Assignment delivery and team of five

The required MovieLens100K model comparisons, hybrid work and societal analysis
remain the deliverable spine. New custom models can strengthen its research
extension; supplementary data cannot replace required experiments. Work divides
into evidence/optimization, structural learning, independent evaluation,
societal effects, and report/reproducibility. These are five responsibility
areas, not invented contributions attributed to the unnamed members.
