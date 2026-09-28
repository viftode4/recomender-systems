# Prior work and limits of the evidence-transfer pilot

This note accompanies [DESIGN.md](DESIGN.md) and [PROTOCOL.md](PROTOCOL.md).
It records the closest verified mechanisms, not a claim of priority. The pilot
implements a shared nine-feature scorer and tests two additional pattern features.
It does not learn arbitrary graph operations, identify preferences independently
of exposure, or establish a new category of recommender.

## Shared interpretation of local evidence is established

**Inductive Graph-based Matrix Completion (IGMC).** Zhang and Chen learn a
shared rating predictor from local user-item subgraphs, using structural roles
instead of global node IDs. Their method removes the queried training edge and
can operate on another graph when interaction neighborhoods are available. It
does not solve completely interaction-free cold start. This is a direct prior
for transferring a learned interpretation of neighborhood evidence across item
identities. Its reported rating-RMSE task does not establish performance on our
full-catalog recorded-item ranking task. [ICLR 2020 paper, §§3.1–3.3](https://arxiv.org/pdf/1904.12058)

**BUDDY.** Chamberlain and colleagues construct link-specific structural
statistics using neighborhood sketches and feed them, with preprocessed node
features, to an MLP. Their experiments also examine an MLP directly on structural
counts. Thus replacing per-ID embeddings or graph message passing with a shared
predictor over structural summaries is not itself new. Our exact donor summaries
and small MovieLens experiment differ, but that difference does not prove
conceptual novelty. [ICLR 2023 paper, §§3 and 5](https://arxiv.org/pdf/2209.15486)

## The pattern-count equation is an existing diversity measure

For positive-weight supporting donors, write their normalized overlap patterns
as unit vectors `a_v`, their normalized weights as `p_v = w_v / sum(w)`, and
their pattern-similarity matrix as `Z_vt = a_v · a_t`. Then our equation is

```text
pattern_count = (sum w)^2 / ||sum_v w_v a_v||^2
              = 1 / (p^T Z p).
```

This is exactly order-2 similarity-sensitive diversity in Leinster and Cobbold's
framework. Identical patterns give one; disjoint patterns recover
`1 / sum(p_v^2)`, the weight-concentration count also used as Kish's effective
sample size. We apply an established diversity statistic to query-relative
support patterns; we do not invent the statistic. The no-support convention is
an implementation boundary, since normalized positive weights then do not
exist. [Leinster and Cobbold, 2012, §§2–3](https://webhomes.maths.ed.ac.uk/~tl/mdiss.pdf)

Its interpretation here is geometric. Similarity of observed histories is not
an estimate of the covariance between donors' response errors. Therefore this
number is not a calibrated independent sample size or a posterior confidence.
Two real people with identical observed patterns may still provide additional
information. Reusing one person's history under multiple masks does not create
additional independent people.

## What can and cannot be inferred

The measured outcome is a recorded interaction. Without exposure observations
or identifying assumptions, a record probability cannot uniquely separate
availability, exposure, choice, liking, and willingness to rate. Non-recording
donors supply a background distribution; they are not confirmed rejections.
Latent exposure modeling is established, but introducing an exposure variable
does not make its true value identifiable from these records alone. [Liang et
al., *Modeling User Exposure in Recommendation*](https://arxiv.org/abs/1510.07025)

The essential leakage boundary is the one in the protocol: the query target is
hidden from its context, and the query user's entire donor row is excluded from
every evidence summary and population normalizer. A different donor's known
TRAIN record of the candidate may legitimately affect that donor's history norm;
this is not the query user's hidden target. Removing the candidate from donor
norms would be a different similarity convention, not a required leakage repair.

There is also no basis for saying EASE treats every estimate as equally certain
because it uses a scalar ridge penalty. That penalty is an isotropic coefficient
prior; data curvature still creates direction-dependent shrinkage. Feature-wise
penalties are a separate established mechanism: Steck derives dropout-induced
diagonal regularization proportional to feature observation counts. This pilot
does not isolate regularization, loss, or uncertainty modeling against EASE.
[Steck, NeurIPS 2020, Eq. 4 and §5](https://proceedings.neurips.cc/paper_files/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf)

Full versus no-pattern is the direct matched comparison. Both already receive
Kish count and the other seven non-pattern features. Removing pattern count and
coverage together tests their incremental usefulness as a pair; it does not
identify either feature's individual role. A win would support this feature
combination under this reused-data protocol. It would establish neither causal
independence of evidence nor broad novelty. Equal summaries necessarily give
equal scores, and items without donor records cannot be distinguished by an
unobserved identity or content signal that the model does not have.

## Stronger future falsifying control, outside this pilot

For a fixed query-candidate pair, retain the donor histories and bucket donors
by exact query-overlap count and donor-history count. Within a bucket all donors
have equal squared-cosine weight. Permute candidate-record indicators only
within these buckets, then recompute the two pattern summaries. With the query
row excluded and the original donor features held fixed, this preserves all
seven non-pattern inputs: candidate count, support mass, fractions, mean weight,
Kish count, and context size. It changes which overlap patterns provide support.

Train and evaluate a separate full scorer under the same declared perturbation
recipe, rather than training on corrupted features and querying real ones. If
the real pattern features cannot outperform this control, the proposed useful
alignment of support patterns has not been demonstrated. Report movable donor
label mass and the fraction of query-candidate feature pairs actually changed;
small or homogeneous buckets can make the control uninformative.

This is a conditional feature perturbation, not a globally consistent randomized
rating graph, a uniform graph-null sampler, or a causal test. It is deliberately
not part of the first pilot and must not be added after inspecting its outcomes
and presented as predeclared confirmation.
