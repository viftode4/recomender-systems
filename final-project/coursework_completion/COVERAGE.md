# Coursework coverage and completion evidence

This maps the assignment to implemented work and the bounded completion study.
It is a **review document, not a submission receipt**. Existing frozen results
remain unchanged. The completion study has run all 39 declared configurations
and published all nine assessed roles across three seeds. Source semantics,
synthetic model tests, public aggregate arithmetic and independent numerical
replay have passed the checks recorded below.

## Sources and corrected interpretation

- [Project brief](../reference/Project-RecSys.pdf), 19 pages: experimental setup
  on page 4; Tasks 1–3 on pages 5–10; deliverables and report rules on pages 12–18.
- [W4S2-Hybrid lecture](https://brightspace.tudelft.nl/d2l/le/content/844485/viewContent/5179119/View),
  all 20 pages, freshly read through Brightspace MCP by the coordinating agent
  on 29 September 2026. The seven families are weighted (p. 6), switching
  (p. 7), mixed (p. 8), feature combination (p. 9), feature augmentation
  (p. 10), cascade (p. 11) and meta-level (p. 12).
- The previously verified deliverable is the
  [five-page framing report](../reports/framing-review-v1/report.pdf), with
  its [package verification](../packages/framing-review-v1/verification.json).
  It preserves the original frozen test results and a separately labelled
  nested-TRAIN grouping experiment.

**Correction:** the lecture explicitly gives diversification, fairness and
calibration as cascade examples. Our implemented rerankers therefore cover
that family. Earlier statements that a separate predictive cascade was absent
did not establish a missing lecture requirement. A second predictive cascade
is unnecessary for this completion pass. Mixed and meta-level implementations,
plus explicit switching/RRF tuning, are now present in the completion study.

## Seven lecture families

| Family and lecture page | Concrete implementation / completion work | Interpretation and status |
|---|---|---|
| Weighted, p. 6 | `study.py` learns expert coefficients by regression; `hybrid_constraints.py` implements sum-to-one ridge and response alignment. | Verified frozen results. Negative coefficients are permitted; effective coefficients after score calibration need not sum to one in the original score units. |
| Switching, p. 7 | `study.py` chooses a meta-fit expert for each TRAIN-activity tercile. The completion study evaluates a 2/3/4-group grid. | Implemented and assessed. Group thresholds use TRAIN activity, not assessment outcomes; development selects the group count. |
| Mixed, p. 8 | `models.py` places EASE, GenreContent and ExactPop recommendations in one list under deterministic quotas, with deduplication. | Implemented and assessed. This merges source lists; it is not another weighted score regression. Quotas and duplicate handling are declared in `PROTOCOL.md`. |
| Feature combination, p. 9 | Context regression combines expert-score features with TRAIN-derived activity, genre entropy and item popularity, including interactions. | Verified. Collaborative signals and content/context summaries become inputs to one predictor. This is a feature-combination mechanism, not an independent additional model beyond the context hybrid. |
| Feature augmentation, p. 10 | Expert predictions become input features to the regression hybrid. | Verified. The same context model also combines side features. These two taxonomy labels overlap; do not advertise them as two independent new algorithms or experiments. |
| Cascade, p. 11 | `study.py` / `societal.py` rerank for diversity, calibration and exposure/user-popularity objectives. Individual-first and fusion-first orderings are compared. | Verified frozen results. This matches the lecture's explicit examples; ordering comparisons retain full candidate support. |
| Meta-level, p. 12 | `models.py` learns genre-based user profiles from TRAIN, freezes them, then uses those profiles as the user representation for an item decoder fitted to collaborative interactions. | Implemented and assessed. The second stage receives only the learned profile, not an appended content score or parallel raw-history representation. |

RRF is an additional rank-fusion baseline, not an eighth lecture family. Its
existing offset is 60; the completion study declares offsets 10/60/100 for
selection. A type label alone does not demonstrate performance or novelty.

The approved completion grid has 13 candidates per seed: mixed allocations
`(6,2,2)`, `(4,4,2)` and `(4,2,4)` in EASE/GenreContent/ExactPop order;
meta-level ridge penalties `0.1, 1, 10, 100`; switching group counts `2, 3, 4`;
and RRF offsets `10, 60, 100`. Mixed scheduling uses deterministic weighted
round-robin with unique eligible items. The implemented meta-level stages are
`P = X G (GᵀG + λI)⁻¹` and `Q = XᵀP (PᵀP + λI)⁻¹`, giving scores `PQᵀ`:
`X` is binary TRAIN interaction membership and `G` is fractional genre content.
Both stages use the same selected penalty; the decoder receives only `P` as its
user representation. All candidates and selected settings are retained in the
[aggregate evidence](results-v1/aggregates.json).

## Every required technical task

| Task; brief page | Evidence and remaining action |
|---|---|
| 1.1 Individual recommenders; p. 5 | Twelve standard adapters in `run.py`, including content, neighborhood, linear, factor and graph/neural models; individual results in main Task 1. The brief itself does not enumerate a fixed model count. |
| 1.2 Individual tuning; p. 5 | Recorded expert grids and budget decisions in `experiment.py` and frozen evidence; the new meta-level grid is explicitly recorded. |
| 1.3 Regression-weighted hybrid; p. 5 | Learned coefficients, constrained/calibrated controls, independent tests and frozen results. |
| 1.4 Other class hybrids; p. 5 | Seven-family mapping above. Mixed and meta-level additions are implemented and assessed; existing cascade coverage is confirmed by the refreshed lecture. |
| 1.5 Hybrid tuning; p. 5 | Regression penalties were tuned. Completion evaluates explicit mixed/meta-level grids, switching group counts and RRF offsets; development selects within each family. |
| 2.1 Independent metrics; p. 6 | `metrics.py` and `societal.py` implement ranking accuracy and beyond-accuracy metrics separately from RecBole. |
| 2.2 Baseline comparisons; p. 6 | Random and ExactPop appear alongside experts and hybrids. Compare methods only within matching cohorts, candidates, relevance and information budgets. |
| 2.3 Coefficient analysis; p. 7 | Main Task 2 reports frozen static coefficients and correlated-expert interpretation; magnitude is not causal importance. |
| 2.4 Insightful analysis; p. 7 | Response-scale control, custom-model objective/control studies, group trade-offs and the controlled timestamp-grouping test. Negative results are retained. |
| 2.5 User/item groups, accuracy and beyond-accuracy; p. 8 | Main Task 2 compares EASE/SLIM/context by activity-group nDCG and genre diversity, plus head/tail recall and exposure. |
| 2.6 Insights and possible improved hybrids; p. 8 | Context features and reranking are motivated by observed behavior; custom and grouping studies test hypotheses with controls. The task asks what the analysis suggests, not for a guaranteed winning invention. |
| 3.1 Societal rerankers; p. 9 | Diversity, genre calibration, item exposure, popularity calibration and independently calibrated user-utility policy. Operational fairness definitions are explicit. |
| 3.2 Accuracy/beyond-accuracy trade-offs; p. 9 | Main Task 3 reports accuracy alongside diversity, calibration, exposure and popularity divergence. |
| 3.3 Before/after fusion; p. 10 | Expert reranking followed by RRF versus RRF followed by reranking, with both utility and objective values. |
| 3.4 User/item impacts; p. 10 | Group utility retention, head/tail recall/exposure and aggregate societal audits. Retention on unseen outcomes is not guaranteed. |

## Evidence boundaries for the completion study

The completed study uses all three existing seeds: meta-fit labels fit permitted
hybrid components, development labels select settings, and the former
calibration cohort assesses the sealed choices. TRAIN determines histories and
features. These cohorts have all been used before, and the original TEST has
already been opened. Calling the last cohort an assessment set does **not**
make it fresh confirmation. New results are post-test exploratory coursework
completion; original TEST must not be read again by this study. Original fits,
frozen scores, reports and packages remain unchanged.

New scores must not be mixed numerically with the original frozen test table.
There are 471 meta-fit, 236 development and 236 reused assessment users per
seed. Assessment records number 2,136 / 2,214 / 2,317 for seeds 2026 / 2027 / 2028.

## Actual completion results and checks

The [protocol](PROTOCOL.md), [run protocol](results-v1/protocol.json),
[aggregate results](results-v1/aggregates.json) and
[provenance](results-v1/provenance.json) record the finished 39-candidate study.
Every family and reference uses the same assessment cohort and candidate rules.
These are equal-weight means of three overlapping splits:

| Role | Assessment nDCG@10 |
|---|---:|
| EASE reference | 0.273765 |
| SLIM reference | 0.265742 |
| Frozen context reference | 0.278133 |
| Fixed three-group switch | 0.274641 |
| Tuned switch | 0.274641 |
| Fixed RRF, offset 60 | 0.270612 |
| Tuned RRF | 0.271294 |
| Mixed | 0.216840 |
| Meta-level | 0.198748 |

Mixed chooses `(6,2,2)` in every seed. Meta-level chooses penalties `1 / 1 / 0.1`;
switching chooses `3 / 3 / 2` groups; RRF chooses offsets `10 / 100 / 100`, in
seed order. Mixed and meta-level are substantially weaker in this experiment.
Switching produces unchanged aggregate accuracy; tuned RRF has only a small
descriptive difference. Completing family coverage is not a superiority claim.

This requirements audit directly inspected `models.py` and the protocol, ran
all 12 independent model tests successfully, verified all three public artifact
hashes, and recalculated every published equal-seed aggregate mean for all nine
roles. It also checked cohort counts and selected settings against the public
candidate records. The separate [independent review](REVIEW.md) and
[audit receipt](audit-v1/audit.json) record a **passed** numerical replay:
43,239 checks, maximum absolute error `7.105427357601002e-15`, all 39 candidate
development metric records, all 27 role/seed assessment tables and per-user
outputs, and every selected/reference ranking equation. The receipt binds the
public artifacts and global selection seal; it reports no original TEST or
`valid_mask` array access. It replays saved candidate metrics and selected model
equations, rather than repeating the hyperparameter search or retraining the
original frozen experts. The pre-fit model and boundary suite contained 21
passing tests. Final report/archive checks will be recorded separately after
these document bytes are bound.

Audit receipt SHA-256:
`7ebf91db0f90fecb7cd32a093f3fd49b51a6b711032fa2fc7b6846da45c6ad0d`.
Aggregate evidence SHA-256:
`592e374ec9216771538333200e277db023d6c506a051b313798107179bf791a5`.

## Academic deliverables

The final version needs a group cover, one page per required task, at most 200
discussion words per task, Times New Roman 12pt and 1.15 spacing. Additional
results belong in a clearly scoped appendix; the main body must stand alone.
Package runnable code, short instructions and the PDF in `24.zip`. Member
details and truthful individual peer-feedback Excel files remain deferred human
inputs; see [TEAM_REVIEW.md](TEAM_REVIEW.md).

The brief defines no formal bonus task or guaranteed extra-credit points.
Controlled research and careful interpretation provide additional depth.
Neither source coverage nor a model win promises a grade. No external submission
has been made by this completion audit.
