# Evaluation decisions before the held-out audit

Recorded September 28, 2026, before opening the final-project test splits.

The course task is full-catalog ranking on MovieLens 100K. Primary relevance is
any observed held-out rating. A separate explicit-rating experiment uses ratings
at least 4 as relevant and reports known dislike hits (ratings at most 2). These
objectives must not be mixed in a single ranking table.

## Fixed comparison set

Use seeds 2026, 2027 and 2028. Standard experts are ExactPop, Random, EASE,
ItemKNN, UserKNN, BPR, SLIMElastic, corrected FISM, genre content, LightGCN,
NeuMF and NGCF. The explicit-rating PositiveEASE, SignedChannelsLinear and
ContrastTransfer models add three research experts. Their configurations are
selected for all-observed ranking when used in the primary hybrid study.
Random is evaluated independently and excluded from every fusion.

Expert configurations use the 471-user meta-fit cohort. Hybrid coefficients
use that same cohort and a common negative sample. Each hybrid family's
regularization is selected on 236 development users. The comparison includes
the lecture's sum-to-one constrained weighted hybrid, its meta-fit response-aligned
control, unconstrained static,
user, item, disagreement and full-context ridge, static/context pairwise
regression, group switching and reciprocal-rank fusion. No family is removed
because its result is unfavorable. The response-aligned control was added before
test access after a meta-fit-only audit diagnosed a score/target scale mismatch;
it does not establish a new function class or probability calibration.

LightGCN's original 20/60-epoch grid was still improving. Before test evaluation,
the grid was extended to 100/200 epochs for every seed, with selection among all
four budgets on meta-fit users. Other neural models retain their declared small
grids. This is a course-scale comparison, not a claim of competitive SOTA tuning.

Diversity, genre calibration, user popularity calibration and item exposure use
the same fixed strengths 0.2, 0.5 and 0.8 and pool size 100. Before/after fusion
order uses strength 0.5 and full ranking support. Candidate expansion checks
both head and tail quota availability. Group-specific exposure strengths use
236 separate calibration users, an approximate multiplicity-adjusted bootstrap
retention bound, and baseline fallback. Historical exploratory studies saw the
broader validation cohort; this is not a pristine preregistered calibration trial.

## Freeze and audit

All three primary bundles and all explicit-rating bundles must be frozen before
any held-out evaluation. Freeze stores source and artifact hashes, original ID
order, training-only features, selected coefficients, score-normalization
parameters, reranker settings and policy strengths. Primary inference must
exactly reproduce saved validation recommendations before test access.

At test time, mask both training and validation interactions. Reuse the frozen
train-fitted scores, profiles and normalizers. Do not retrain, renormalize,
choose another setting, or remove a comparison after seeing test results.
Random has a separately seeded deterministic test stream.

Report every frozen model. Predefined family comparisons use the strongest
expert selected on development data; the policy comparison uses its frozen
contextual base. Paired user-bootstrap intervals use at least 20,000 resamples
and Bonferroni-adjusted two-sided quantiles within each seed. They are approximate
and conditional on this dataset; training users are dependent through shared
models. Seeds overlap in users and interactions. Across-seed averages are
descriptive, not three independent replications or an independent-dataset test.

The explicit-rating experiment retains the unsuccessful original contrast model,
the revised candidate gate, positive-anchor and signed controls, randomized
counterparts, a counterpart-multiset permutation control, and matched-budget
rating-aware linear baselines. The second architecture was motivated by the
first one's development failure, so its development results are exploratory.
Known dislike hit rates do not estimate dissatisfaction with unobserved items.

## Additional research before test access

The [rejection information audit](../NEGATIVE_INFORMATION_PROTOCOL.md) is a
completed exploratory mechanism check. It compares context/probe linear
decoders under equal penalties and information controls. Its five branches,
both declared inference modes, control replicates and unsuccessful results
remain in the evidence. These models are not substituted for the primary
experts or selected as new contenders based on the small secondary differences.
The audit's different training information budget prevents direct performance
claims against the full-history linear models.

The [categorical evidence field](../ADAPTIVE_RESEARCH.md) is a separate
from-scratch research track. Five-category prediction preserves every rating
category in training. Its checkpoint objective is macro-user categorical
cross-entropy on the 471 meta-fit users; development evaluation uses the other
472 users, with explicit denominators for liked ranking. The principal controls
are adaptive updates, fixed routing and fidelity within each prediction, and
hard-clamped categorical sources. They share declared optimization budgets.
Global, smoothed item and smoothed item/user rating histograms provide simple
probabilistic references. Exact settings and runtime decisions must be recorded
before real fitting and metrics, with no development-driven expansion.

For this categorical track, P(rating >=4) is a fixed ranking readout, not the
training label definition. All-observed ranking is reported as an objective
mismatch diagnostic. A better category likelihood is not silently presented as
a better assignment ranking score. Finite changing states do not establish
convergence, recovered feelings, or identified rating reliability.

If included in the held-out batch, all three trained field variants and the
simple references are frozen and reported, regardless of development outcomes.
They use TRAIN ratings only as inference evidence, matching the information
access of the existing frozen experts. Validation items are masked for final
ranking without feeding their rating values into the field. No comparison is
added or removed after test access.

The [joint field experiment](../JOINT_FIELD_PROTOCOL.md) uses the same five
logits to predict a recorded item and its rating together. It was specified
before the conditional experiment's development results were inspected. The
loss is categorical NLL plus item-event NLL with unit coefficients; there is
no added prediction head or ensemble. Checkpoints 10/30/60/100 minimize meta-fit
joint NLL. All-observed ranking sums mass over all five categories; liked-record
ranking sums mass over categories 4 and 5. Missing pairs compete in the
normalizer, so the model still assumes a choice set and learns the dataset's
recording mechanism, not unobserved satisfaction or unbiased exposure.

A separate 400-epoch convergence sensitivity was authorized after inspecting
the original study. For both variants in seeds 2026 and 2027, every meta-fit
checkpoint improved, with the minimum at the 100-epoch boundary. The follow-up
keeps the architecture, initialization, masks, optimizer and both variants
fixed, uses all three seeds, and selects among 10/30/60/100/200/300/400 epochs by
the same meta-fit NLL. It restarts from the original initialization because v1
checkpoints omit Adam state, and requires exact first-100-epoch reproduction.
The cap is 400 with no further expansion. Original and extended results are
both retained. This is an explicitly post-v1 exploratory budget sensitivity,
not a preregistered discovery or a claim that 400 epochs establishes convergence.

The held-out field comparison includes the conditional track, joint 100-epoch
track and joint 400-epoch track separately. Freeze every variant, count reference
and checkpoint before test access. Strong references are the already selected
EASE and SLIMElastic configurations and the liked-selected PositiveEASE model;
no reference is reselected for the field comparison. Evaluate the exact saved
scores with TRAIN+validation masking and both relevance definitions. Paired
contrasts compare adaptive with fixed flow and each of those three references
within each endpoint and track: 24 nDCG contrasts per seed. Report all contrasts,
their cohort sizes and Bonferroni-adjusted uncertainty across these 24 contrasts,
with overlapping-seed means descriptive.

## Limits and interpretation

No psychological state, real-world emotion, causal fairness or research-priority
claim is established. Chronological sensitivity is per-user ordering and does
not by itself fix cross-user global-timeline leakage. The report must expose
negative results, calibration/accuracy tradeoffs and group losses. No grade,
statistically established superiority or production suitability follows from
implementation complexity.
