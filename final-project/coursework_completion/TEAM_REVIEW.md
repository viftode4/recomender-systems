# Five-person review and handoff

These are proposed review responsibilities, not claims about who produced the
existing work. Assign real names only when agreed. Reviewing completed work is
a valid contribution when it is actually done and recorded; do not backdate it
or imply that a reviewer implemented the original system.

## A small concrete review for each person

| Seat | Review responsibility | Evidence to leave |
|---|---|---|
| 1: protocol and integration | Explain TRAIN, meta-fit, development, calibration and original TEST; trace one model through selection and freezing. Verify that completion assessment follows the saved all-seed selection barrier. | A short boundary diagram and any concrete leakage/provenance issue, with file references. |
| 2: expert models | Explain EASE, one neighborhood model and one learned model; inspect their declared tuning grids and matching candidates. Understand disclosed FISM/NGCF adapter changes. | One checked model/config example and a brief explanation of a measured behavior, without causal overclaiming. |
| 3: hybrids | Use `COVERAGE.md` to explain all seven lecture families. Inspect mixed quotas/deduplication and the meta-level profile-to-decoder boundary. Verify selected switching/RRF settings against the declared grid. | A family-to-implementation diagram and one independently checked selection row. Mark unfinished additions as unfinished. |
| 4: metrics and societal effects | Hand-check a short ranked-list example; explain conditional head/tail recall and user-group macro averaging. Interpret one reranking trade-off and before/after-fusion result. | A worked metric example and a check of one reported table row against aggregate evidence. |
| 5: report and reproduction | Read the report without this conversation. Check task coverage, evidence stages, claims and formatting; run packaged synthetic tests and aggregate-only report reproduction when available. | Commands/results and a list of any unclear claim, broken instruction or figure problem. |

Each reviewer should record the date, artifact version/hash, checks actually
performed and unresolved findings. A proposed seat is not a contribution log.
Everyone should understand the complete pipeline, not only their assigned seat.

Start from [COVERAGE.md](COVERAGE.md), the completed study's
[aggregate results](results-v1/aggregates.json) and [protocol](PROTOCOL.md).
The study has run all 39 configurations. Its
[independent audit](audit-v1/audit.json) passed 43,239 numerical checks; the
[review](REVIEW.md) explains scope and limits. This technical verification does
not substitute for each member's actual review.

## Questions everyone should be able to answer

**What are we predicting?** The primary task ranks unobserved movies by whether
a user recorded a rating. All recorded ratings count as relevant under our
declared protocol. A one-star rating is therefore a positive recording event,
not evidence of enjoyment. This binarization is our/course configuration
choice, not an explicit sentence in the assignment PDF.

**Why is a missing interaction not a confirmed dislike?** We do not observe
which movies were offered, seen or rejected. Missing entries provide competing
candidates for the ranking objective; they do not identify a person's feelings.

**What separates training, selection and evaluation?** TRAIN fits expert models
and features. Other declared cohorts fit hybrid coefficients or choose settings.
The original final test was evaluated only after its choices were frozen. Later
studies are exploratory because the dataset and prior outcomes have been used.
New cohort names do not erase previous exposure.

**How is a mixed hybrid different from a weighted hybrid?** Mixed recommendation
places items from several source lists into one list under an explicit allocation
and duplicate rule. A weighted hybrid combines numeric predictions before
ranking. Both can be tuned, but they perform different operations.

**Why do feature combination and augmentation share a model here?** The regression
receives other recommenders' predictions, which is augmentation. It also uses
TRAIN-derived content/context summaries and interactions, which is combination.
These are overlapping mechanisms, not proof of two new independent algorithms.

**What makes the meta-level model meta-level?** A first stage learns a user
representation from content and TRAIN history. A second collaborative stage
uses that frozen representation to predict interaction targets. Merely blending
a content recommendation score with EASE would not establish this mechanism.

**Where is the cascade?** The lecture explicitly uses diversification, fairness
and calibration as examples. Our first-stage rankings followed by these
rerankers meet that description. No separate candidate-shortlist cascade is
needed to justify this mapping.

**Do regression weights identify each expert's causal value?** No. Correlated
experts can substitute for each other. Coefficients depend on normalization,
regularization and other features. Negative weights are permitted in the
declared affine regression; they are not probabilities.

**Does fairness mean maximizing everyone's accuracy?** No. We declare specific
objectives and show absolute utility together with allocation/disparity. Higher
tail exposure or smaller group gaps can cost accuracy. A calibrated retention
target is not a guarantee on unseen users or labels.

**What did the grouping research establish?** Ratings recorded at the same user
timestamp have genre structure beyond the specified shuffled null. The tested
grouped-pair model did not outperform tuned EASE or the controls. Recording
timestamps are not viewing sessions; coherence did not establish predictive
benefit. The nested-TRAIN results are not comparable directly to the main test
table.

**What is the headline improvement?** The frozen disagreement hybrid's reported
6.6% relative nDCG improvement is against the separately validation-selected
standalone reference, not against EASE alone. It is evidence within this
experiment, not a universal best-model claim. Three overlapping splits are not
three independent datasets.

**What counts as bonus work?** The brief specifies no formal bonus points.
Controlled ablations, failed hypotheses, robust evaluation and deeper analysis
go beyond a minimal implementation; they do not guarantee additional marks.

**What did the final family-completion study add?** It implemented and assessed
mixed and meta-level hybrids, and explicitly selected switching groups and RRF
offsets on development data. Mixed and meta-level performed worse than the
matched EASE/context references; tuned switching matched the fixed switch and
RRF changed only slightly. All nine roles are reported. The 236 assessment users
per seed were previously used, so these results establish coursework coverage
and descriptive behavior, not fresh generalization or an accuracy breakthrough.

## Human finalization, after technical review

1. Agree actual names and describe genuine contributions without inventing
   implementation history. Contributor details remain deferred until supplied.
2. Resolve concrete review findings; keep scientific freezes and clearly label
   any subsequent exploratory changes. Do not retune on original TEST.
3. Confirm the final report is self-contained and uses the required typography
   and discussion limits. Review the exact rebuilt archive version together.
4. Each student completes their own honest peer-feedback workbook, based on
   actual contribution, responsibility and communication. The local template is
   `reference/PeerFeedback.xlsx`; it is not part of the group's code ZIP.
5. Before any submission, inspect the exact course/assignment target, files,
   sizes, group effect and current instructions. Submission requires explicit
   approval and a recorded receipt; a local `24.zip` is not a receipt.

No names, retrospective contribution ratings, attendance or lecturer approval
are supplied by this checklist. Dates in downloaded material should not be
presented as freshly verified live deadlines.
