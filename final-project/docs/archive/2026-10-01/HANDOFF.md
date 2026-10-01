# Start here

The current review deliverables are the
[five-page coursework report](reports/coursework-complete-v1/report.pdf) and
[Group 24 archive](packages/coursework-complete-v1/24.zip), with a separate
[verification receipt](packages/coursework-complete-v1/verification.json).
The completion pass adds the remaining lecture hybrid categories, their tuning
results and a portable reproduction path. Four member names and the group's
actual contributions remain deferred; individual peer feedback is still personal
human work.

Use the [requirements mapping](coursework_completion/COVERAGE.md),
[team review guide](coursework_completion/TEAM_REVIEW.md), and
[portable reproduction instructions](REPRODUCE.md) for the completion handoff.
The lecture explicitly counts diversification, fairness and calibration
rerankers as cascades. The completion additions address mixed and meta-level
hybrids and tuning; another predictive cascade is not a missing requirement.

The additional coursework comparison ran 39 configurations across three seeds.
Mixed and meta-level hybrids scored 0.216840 and 0.198748 mean nDCG@10;
tuned switching and RRF scored 0.274641 and 0.271294. The matched EASE and
existing context references scored 0.273765 and 0.278133. These use the reused
validation-calibration cohort, not the original TEST. The
[completion evidence](coursework_completion/results-v1/aggregates.json) and
[independent numerical audit](coursework_completion/REVIEW.md) retain
every declared setting and selected family. No broad accuracy gain is established.

The separate grouping research test is complete: 39 declared fits, with true timestamp
groups, three shuffled controls, ungrouped pairs, and a newly fitted EASE.
Grouped nDCG@10 is 0.172555 versus EASE 0.172956, a -0.23% relative difference.
Both grouping comparisons have paired descriptive intervals crossing zero.
Observable genre coherence did not produce a measured ranking advantage in
this model. This does not establish a breakthrough or rule out every other
use of timestamps. See [the protocol and findings](exploratory/framing_search/predictive/README.md).
The assessment is nested within previously used TRAIN data; it is explicitly
exploratory and does not replace the original frozen test tables below.

## Read in this order

1. [Current report](reports/coursework-complete-v1/report.pdf) and
   [current review archive](packages/coursework-complete-v1/24.zip).
   The [current requirements mapping](coursework_completion/COVERAGE.md)
   supersedes the older pre-build completion check. The archive also preserves the
   [original frozen report](reports/final-review-v1/report.pdf) and
   [categorical-reconstruction supplement](reports/categorical-reconstruction-v1/research-supplement.pdf).
2. [Primary paired comparisons](evidence/final-comparisons-v3/SUMMARY.md) and
   [custom models against strong references](evidence/final-field-comparisons-v1/RESULTS.md).
3. [Research questions](RESEARCH.md), [categorical model](ADAPTIVE_RESEARCH.md),
   [joint objective](JOINT_FIELD_PROTOCOL.md) and
   [negative-information audit](evidence/negative-information-v2/RESULTS.md).
4. [Synthetic interactive demonstration](reports/demo/categorical-field/index.html),
   [five-person ownership plan](PLAN.md) and [reproduction commands](REPRODUCE.md).

## What the held-out evidence establishes

Every number below averages three overlapping MovieLens 100K splits. All model
choices were frozen before the single final batch. The splits reuse users and
items; they are not independent datasets.

| Model or fixed role | Mean all-observed nDCG@10 |
|---|---:|
| Validation-selected standalone expert | 0.31613 |
| EASE | 0.32155 |
| Static regression hybrid | 0.33564 |
| Disagreement regression hybrid | 0.33701 |
| Conditional categorical field, adaptive | 0.01376 |
| Joint categorical field, adaptive, 100-epoch cap | 0.22882 |
| Joint categorical field, adaptive, 400-epoch cap | 0.26786 |
| Joint categorical field, fixed flow, 400-epoch cap | 0.27201 |

The disagreement hybrid improves 6.6% relative to the standalone expert selected
on validation. That reference is EASE for seed 2026 and SLIMElastic for seeds
2027/2028; **6.6% is not the improvement over EASE**. The predefined paired
comparison has an approximately multiplicity-adjusted positive bootstrap
interval in each split. This supports an improvement in this frozen experiment,
not a universal or state-of-the-art claim. Static regression is close, so extra
contextual complexity is not automatically valuable.

The custom field revealed a target mismatch. Predicting the rating conditional
on an item being recorded does not determine which item will be recorded.
Training the same field on joint item/rating records produces much better
ranking, but it remains below EASE and SLIM. On liked-record ranking, the
400-epoch adaptive model scores 0.24957 versus 0.29439 for liked-selected
PositiveEASE. Every joint adaptive-versus-fixed-flow adjusted interval contains
zero. These results do not establish a benefit from recurrently changing the
routes. The 400-epoch follow-up was declared before test access, preserves the
original 100-epoch results and still selects the budget boundary; it is not a
convergence proof.

The contrast-transfer and candidate-gate models also failed against strong
rating-aware controls. The separate context/probe audit did not establish a
stable benefit from personalized dislike placement or surprise weighting. Its
different training-information budget and development cohort are explicitly
labelled. Failed ideas are preserved, not hidden or renamed as successes.

Reranking exposes real trade-offs. At fixed strength 0.5, the diversity reranker
changes mean nDCG from 0.333 to 0.307 while diversity rises from 0.798 to 0.895.
The item-exposure reranker reduces popular-head share from 0.981 to 0.931 and
raises tail recall from 0.016 to 0.039, with nDCG falling to 0.318. The separate
utility-budget policy retains 99.7–101.0% of baseline utility across split/activity
group combinations, with small exposure changes. It provides no universal
fairness or utility guarantee.

## What was built and checked

- Twelve standard model adapters and three rating-aware research experts;
  fixed tuning grids, shared splits, independently implemented metrics and
  regression, switching and rank-fusion hybrids.
- Mixed-list and learned-profile meta-level hybrids, plus explicit switching
  and RRF tuning. All seven lecture families have concrete implementations;
  feature combination and augmentation overlap within the context model.
- Diversity, genre calibration, popularity calibration and exposure rerankers;
  ordering comparisons, user/item groups and independent policy calibration.
- Custom categorical evidence, conditional and joint objectives, matched
  fixed-flow controls, learning curves and a declared budget sensitivity study.
- Exact validation replay and source/data/runtime seals. All eight original final
  evaluation jobs and five fixed analysis jobs completed in one reserved batch.
  No original model, checkpoint, hyperparameter or comparison was changed
  afterwards. Later studies are separately labelled exploratory.
- Report rendering at Times New Roman 12pt, spacing 1.15. The five-page review
  has a cover, one page per task and a separately scoped grouping appendix.
  Package verification records its layout, tests and exact extracted PDF replay.
  The package excludes raw histories and model checkpoints.
- A functional offline demo with synthetic profiles and six fixed scenarios.
  Its controls and exported predictions were verified; browser visual checking
  was unavailable because Chrome aborted before page loading.

Detailed checks and package verification are recorded with the final evidence.
The isolated environment was verified locally on macOS. No claim is made that
every platform or an entirely new network installation has been tested.
The raw-data rebuild retains all twelve completion selections, but NGCF
retraining changes some recommendations. Six completion methods reproduce all
assessment metrics exactly; the largest mean nDCG difference among the other
three is 0.000281. [Reproduction evidence](REPRODUCE.md) records those differences
separately from the exact replay of the audited report.

## Human handoff

1. Review and understand the methods and results together; use the five proposed
   seats in PLAN.md to assign real ownership without inventing contributions.
2. Add the four remaining names and each person's actual contributions. Each
   student must complete their own real individual peer-feedback assessment.
3. Bring the completed work and methodology questions to lecturer feedback.
   This test is now exposed: further changes cannot reuse it as a fresh
   confirmation set. Label any further exploration accordingly.
4. Review the final submission package after those human inputs. No Brightspace
   submission has been performed.

No experiment identifies human feelings, proves novelty priority, guarantees an
absence of defects or establishes a particular grade.

## Subsequent exploratory research

The original frozen scientific artifacts remain unchanged. New work is indexed
in [exploratory/README.md](exploratory/README.md).

The [requirements correction](exploratory/ASSIGNMENT_AND_RESEARCH.md) distinguishes
the actual assignment constraints from our adopted relevance/split/metric choices.
The latest [shared evidence interpreter](exploratory/evidence_transfer/FINDINGS.md)
learns one rule for query-relative donor evidence across movies. Adding pattern
diversity and coverage improves its matched control by 2.73% in mean development
nDCG, but the complete model remains 6.30% below EASE. Its research target was
not met, and final-checkpoint selections leave optimization unresolved. This
pilot does not complete the user's request for a substantial original advance.

The earlier [categorical reconstruction study](exploratory/categorical_reconstruction/FINDINGS.md)
is complete: an exact custom solver, 243 predeclared candidate fits, shuffled
rating controls, three hybrid arms and user/item group analysis. Mean development
nDCG@10 is 0.263154 versus 0.261688 for expanded-grid binary EASE, a 0.56%
relative gain. Precision and recall improve slightly while MRR declines. The
hybrid gain is negligible and does not consistently beat the shuffled control.
All three seeds' selections were sealed before development evaluation, but
these cohorts were reused after prior TEST exposure. This is exploratory
evidence, not fresh confirmation or an established scientific breakthrough.
The [independent numerical audit](exploratory/categorical_reconstruction/audit-v2/RESULTS.md)
reproduces selected predictions, hybrid calibration/fitting and ranking metrics;
the new review bundle keeps the original frozen study separate from this work.

The addressed-evidence training study has completed all six fits. Its pair
extension fails the primary joint-NLL comparison and both variants remain below
the matched development ranking references. The [postmortem](exploratory/addressed_evidence/POSTMORTEM.md)
preserves those results, verifies the original seals and diagnoses scale growth
using TRAIN-only forward passes. An independent replay reproduced its aggregate
diagnostics exactly.

The [deeper research map](exploratory/deeper_research/RESEARCH_MAP.md) separates
a finite learned-operation/query experiment from a proposed continuous neural
operator architecture. The finite experiment uses synthetic inputs only. It
does not turn the exposed MovieLens test into new confirmatory evidence and
does not establish conceptual novelty or a recommender improvement.

## Local and publishing state

Work is under `final-project/` on branch `project/starter`. Raw runs and
checkpoints are ignored; curated aggregate evidence and the synthetic demo are
available for review. Unrelated Assignment 1 notebook edits are outside this work.

GitHub publishing was rejected by automatic approval review earlier in this
session and has not been retried. No remote push or course submission has been
made. The local review archive is not a submission receipt.
