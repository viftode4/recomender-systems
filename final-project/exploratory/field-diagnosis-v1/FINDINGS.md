# What limited the categorical field?

This is a **post-test exploratory diagnosis**, not a new held-out evaluation.
The diagnostic opens only original TRAIN categories, validation pair IDs,
development cohort IDs, saved TRAIN-fitted predictions, checkpoints and training
curves. It does not open the rating dataset, TEST pair files, final result files
or final report. No model is fitted or selected. The old sources and evidence
remain unchanged. Input hashes and the prospective diagnostic settings are in
`protocol.json` and `aggregate.json`; no user IDs or individual histories are
exported here.

## Measurements

All ranking values below use exactly 472 development users per split and the
original TRAIN-unseen catalog. Relevance is all observed validation interactions.
Means equally weight seeds 2026, 2027 and 2028; these are overlapping splits.

| TRAIN-fitted predictor or diagnostic compression | Mean development nDCG@10 |
|---|---:|
| EASE, item/row means only | 0.12370 |
| EASE, 8 interaction components plus item/row means | 0.23513 |
| EASE, 16 interaction components plus item/row means | 0.26020 |
| EASE, 32 interaction components plus item/row means | 0.26136 |
| EASE, original full score matrix | 0.26183 |
| Adaptive field, selected epoch 400 | 0.22193 |
| Fixed-flow field, selected epoch 400 | 0.22589 |

The compression fits an SVD to the double-centered **already trained EASE score
matrix**, using no validation labels. It retains item means and harmless
per-user score offsets. The matrix includes scores at TRAIN-observed positions;
those positions remain excluded from every evaluated recommendation list.
This is evidence that a compact representation of useful predictions exists,
not proof that a rank-16 model trained directly can learn it. It does not turn
EASE into a proposed component of a new model, and ranks were not selected here.

The dim=8 field is not literally a rank-8 score model. Nonlinear routing, tanh
and log-sum-exp can produce a full-rank user/item score matrix. Increasing width
may help, but "EASE wins because its predictions need full rank" is too strong:
the rank-16 score compression retains roughly 99.4% of EASE's nDCG in this
diagnostic. Full EASE uses about 2.83 million off-diagonal item coefficients to
learn these scores, whereas the field has 14,654 parameters. Learning structure
and storing the resulting predictions are different capacity requirements.

Selected checkpoints replayed exactly on 128 uniformly sampled development
profiles per seed, using those users' TRAIN histories. At the fourth update:

| Adaptive field diagnostic | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| Mean cosine similarity between upward port weight vectors | .421 | .286 | .429 |
| Downward route entropy / log(16), unseen candidates | .463 | .436 | .529 |
| Decoder tanh mean derivative, unseen candidates | .581 | .601 | .559 |
| Decoder units with absolute tanh activation > .95 | 13.55% | 13.01% | 16.78% |
| Observed source gates > .95 | 16.06% | 94.05% | 4.06% |

The ports are not all identical or uniformly routed, and the decoder is not
globally inactive through tanh saturation. Seed 2027 does nearly clamp its
observed sources: the corresponding fixed-flow model has 99.91% of source
gates above .95. That limits the usefulness of *changing source gates* in this
split; it does not make all routing constant or establish a causal explanation
of the ranking gap. Candidate state coordinate variation grows over the four
updates rather than collapsing to zero.

Both variants still improve their meta-fit joint NLL from epoch 300 to 400 in
all three splits: adaptive reductions .0222/.0399/.0274, fixed-flow
.0089/.0385/.0235. Their last 20 TRAIN-episode losses also improve relative to
epochs 281–300. There is no convergence certificate. TRAIN losses use newly
masked probes and shorter contexts, so direct TRAIN/validation loss gaps do not
isolate overfitting. This diagnosis does not authorize extending the old budget.

## Structural limitation and interpretation

For an unseen candidate in fixed flow, its downward port weights are independent
of the user because its initial evidence state is zero. Its final 8-vector is
`h_ui = D_i S_u`, where `D_i` has 16 weights and `S_u` is a 16-by-8 accumulated
port summary. The history therefore influences every candidate through 128
summary scalars. On a common set of candidates unseen for every compared user,
the flattened candidate/state matrix factors through those 16 routes. This is
not a rank bound on the nonlinear scores, and it must not be tested by applying
different users' eligibility masks and asserting a common matrix rank.

Adaptive flow can change routing, but all history information still travels
through four 16-by-8 port-value matrices. Unknown candidate states never feed
the upward pool. Item identity must share an 8-vector across routing, source
interaction and decoding; a 16-unit decoder also shares responsibility for
item-event frequency and five-category prediction. There is no direct learned
candidate-to-context-item coefficient table.

The evidence therefore supports a narrower diagnosis: the present pooling and
shared parameterization do not learn enough useful item-specific collaborative
structure within the budget. It does **not** isolate capacity, optimization,
regularization or joint-objective interference as the sole cause. The measured
port diversity specifically argues against diagnosing universal attention
collapse; the score-compression control argues against a simple full-rank
prediction requirement.

## Two implementation ideas, one focused mechanism

**1. Direct categorical evidence operator.** For each candidate `i`, choose 64
context-item neighbors using TRAIN-only item co-observation statistics. Learn
a separate 5-by-5 compatibility table `W[j,i]` on every directed edge, with no
diagonal edges. For context `C` and candidate category `c`, calculate
`a[j,i,c] = W[j,i][rating_j,c]` and `z[i,c] = b[i,c] + sum_j a[j,i,c]`.
Unknown items contribute nothing and are not assigned a dislike category.
This preserves the identity of each movie/rating until it reaches the actual
candidate. With 1,682 real items it uses about 2.69 million edge parameters,
plus category biases, instead of compressing all evidence through 16 ports.
Use one masked-probe joint recorded-event loss, shrink edge tables strongly,
and train from fresh parameters. No existing model score, pretrained expert,
attention layer or ensemble is part of the predictor. This is the necessary
capacity/conditioning control, not a novelty claim.

**2. Candidate-specific evidence dispersion.** Extend that *same* operator with
one directly testable nonlinear mechanism. Let `n` be the number of observed
neighbors and define
`dispersion[i,c] = (n*sum_j a[j,i,c]^2 - (sum_j a[j,i,c])^2)/(n*(n-1))`
for `n>=2`, and zero otherwise. Set
`z[i,c] = b[i,c] + sum_j a[j,i,c] - softplus(gamma[i,c])*dispersion[i,c]`.
The numerator equals the sum of squared differences between distinct evidence
votes, so the model can reduce support when the same candidate/category receives
inconsistent votes. It neither averages those votes into a global user vector
nor computes softmax attention. Setting the dispersion coefficient to zero
gives the exact direct-operator control. Both are single jointly trained models.

Dispersion is a computational property of learned votes, not an identified
psychological conflict: different vote magnitudes may be informative, and
penalizing them can make predictions worse. The decisive experiment is whether
the dispersion model improves over the direct operator on a predeclared
assessment while retaining strict target masking. Compare all users and report
both category likelihood and the ranking endpoints. Do not add a residual from
EASE, choose favorable cohorts after seeing outcomes, or call another experiment
on the already opened assessment independent confirmation. No implementation or
training of these two proposals has been started.

## Prior-art boundary

These proposals can be implemented independently, but their broad mathematical
families already exist. [EASE](https://arxiv.org/abs/1905.03375) establishes the
strength of direct linear item relationships.
[Sparse rating Markov fields](https://arxiv.org/abs/1602.02842) learn interaction
structures rather than routing through a common embedding summary.
[CF-NADE](https://proceedings.mlr.press/v48/zheng16.html) already models categorical
ratings from partial contexts.
[Steck and Liang's 2021 higher-order study](https://recsys.acm.org/recsys21/session-2/)
is a particularly close warning: signed higher-order item interactions are
established prior work. A learned quadratic dispersion term is not evidence that
we invented higher-order recommendation, or that nobody has tried a related rule.
The defensible contribution would be the exact operator, its controls and what
its measured failure or success teaches us about this diagnosed information path.

## Follow-up, 28 September 2026

The proposals above record the design at the time of this diagnosis. A direct
operator has since been implemented in `../addressed_evidence/model.py`, with
the separate study contract in `../addressed_evidence/PROTOCOL.md`. The selected
A/B comparison uses a **signed distinct-source pair product**, not the proposed
nonnegative dispersion penalty. Its pair sum is
`((sum_j s_j)^2 - sum_j s_j^2)/2`, with a learned bounded signed coefficient per
candidate/category and fixed initialization-unit scaling. Both variants start
from exactly the same direct tables and predictions. This preserves the direct
evidence-path question while allowing either sign of quadratic contribution;
it does not identify psychological conflict. Source-pair coefficients remain
tied through the same unary tables, rather than independently learned for each
pair. "Positive neighbors" means positive co-observation similarity, regardless
of whether the recorded ratings are high or low. No prior measurement above
has been changed, and no new final-test access is part of this follow-up.
