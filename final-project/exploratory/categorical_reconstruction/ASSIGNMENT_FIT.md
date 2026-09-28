# Assignment fit: categorical reconstruction recommender

This is a new exploratory extension of the completed project. Its purpose is to
test a standalone recommendation model that uses the rating categories in a
user's observed history to reconstruct the binary record-presence target used
by the project's main ranking evaluation. It is not an ensemble and does not
replace the required hybrid experiments.

## Sources and verification

- Primary source: [Project-RecSys.pdf](../../reference/Project-RecSys.pdf), all
  19 pages independently reread on 2026-09-28. Size: 151,364 bytes. SHA-256:
  `bb56a889d5af915d85027f3f7e5cb3338a41c11100ded8ae19aa18ff76a02c93`.
- The preserved [assignment-source note](../../reference/ASSIGNMENT.md) records
  the earlier successful download from Brightspace course `844485`, final-project
  assignment `173405`, attachment `11701780`.
- The new live MCP `check_auth` and correctly parameterized `get_assignment`
  requests both returned `PERMISSION_DENIED`, HTTP 403. This audit therefore
  verifies the local attachment, not the current live deadline or submission
  state. No submission, login change or other external mutation was performed.

## What the brief requires, and what this extension contributes

| Requirement in the PDF | Role of the new work | Boundary |
|---|---|---|
| Page 4: RecBole, MovieLens 100K, ranking, separately performed evaluation | Consume the same RecBole-exported partitions and ordered IDs; emit catalog scores for the existing independent evaluator. | Preserve the RecBole project setup and full-catalog candidate policy; the new model alone is not the complete project. |
| Page 5, Tasks 1.1–1.2: build and tune individual recommenders | Add an explicitly defined standalone expert and predeclare its fitting and selection protocol. | Retain the class-model coverage and tuning evidence already in the project. |
| Page 5, Tasks 1.3–1.5: regression-weighted hybrid, other class hybrids, hybrid tuning | A later controlled extension can add the custom expert to the existing regression hybrid. | These required tasks cannot be replaced by standalone model results. |
| Page 6, Task 2.1: independently implement accuracy and beyond-accuracy metrics | Reuse the independently implemented metrics, verified on the same candidate and relevance definitions. | Do not present reconstruction loss or rating-category accuracy as a substitute for ranking effectiveness. |
| Page 6, Task 2.2: compare selected models and naïve random/popularity baselines | Compare the selected custom expert with locked standalone references on identical users, truths and candidates. | Report random/popularity in the main project context; strong-reference comparisons are also necessary for a substantive improvement claim. |
| Page 7, Tasks 2.3–2.4: coefficient contributions and insightful analysis | Test whether rating categories carry useful signal beyond binary history, and investigate where that signal helps or hurts. | Standalone model coefficients are not hybrid mixture coefficients; keep those interpretations separate. |
| Page 8, Task 2.5: accuracy and beyond-accuracy across user and item groups | Compare the custom expert and references across TRAIN-defined sparse/medium/dense users and head/tail items. | Include both kinds of groups and both accuracy and beyond-accuracy; do not choose group cutoffs from favorable results. |
| Page 8, Task 2.6: ask whether insights lead to novel and superior hybrid models | If analysis shows complementary errors, test a hybrid with and without the custom expert under the same protocol. | A standalone win alone does not establish a superior hybrid. A null or negative hybrid result remains informative and must be retained. |
| Pages 9–10, Tasks 3.1–3.4: diversification, calibration, user/item fairness, trade-offs, hybrid/reranker order, group impacts | Existing societal analysis remains part of the submission. A selected custom expert may receive the same fixed reranking treatment in a separately identified extension. | A new model does not discharge these tasks; do not silently retune rerankers on reused development outcomes. |

Task 2.6 is an invitation to reason from evidence. It is not a requirement to
claim novelty or superiority when the experiment does not support either.

## Match the actual ranking endpoint

The PDF requires ranking, but does **not** prescribe treating every rating as a
positive interaction. The existing project explicitly adopts that convention
in [run.py](../../run.py): every observed held-out rating is relevant, including
low ratings. Keep that endpoint for direct comparison with the established
baseline results. Retain categorical ratings as inputs, without relabeling the
evaluation to favor the new model.

The proposed predictor outputs one score per candidate item for binary
record-presence reconstruction. Unknown pairs used as zero reconstruction
targets are absent records, not verified dislikes. Improvements would concern
this offline observed-record ranking task, not an established improvement in
user satisfaction or exposure-corrected preference estimation.

The earlier five-category conditional loss estimated the rating distribution
given a user and item. That objective alone did not identify item-event mass.
The later joint item/rating likelihood did model event mass, so it should not
be described as categorically incapable of addressing this ranking endpoint.
The new experiment changes the objective and model assumptions; success or
failure must be measured rather than attributed solely to that earlier loss.

## Standalone test and an optional hybrid extension

The first experiment should answer whether categorical history improves one
direct predictor under the same information, candidates and ranking endpoint.
Use a binary-history control and locked strong references. Inspect parameter
support, self-item exclusion, regularization sensitivity and TRAIN-defined
group outcomes before assigning a mechanism to any score difference.

A hybrid extension is optional new research, not a reason to withhold the
standalone result. If undertaken, compare the original expert set with the
same set plus the custom expert. Fit mixture coefficients only on the declared
coefficient-fitting cohort, use the same tuning budget and selection rule, and
show the custom expert's coefficient, exclusion control and group impacts.
Complementarity must be demonstrated by the hybrid comparison rather than
inferred merely from a different architecture. Preserve the existing required
regression and class-hybrid evidence regardless of the extension's result.

The subsequently declared [protocol](PROTOCOL.md) commits this study to three
such hybrid arms: binary EASE plus locked SLIM, then separately adding the real
or shuffled categorical expert. They are now required to complete this declared
study, while the assignment's broader hybrid requirements remain unchanged.

## Selection and evidence boundaries

1. Freeze source hashes, runtime, model variants, tuning grid, cohort IDs and
   metric definition before real fits. Training targets and score transforms
   must derive from TRAIN only. Remove self-item routes that reveal the very
   record being reconstructed.
2. Select each model using the declared meta-fit cohort and the main
   all-observed ranking metric. Verify that development labels, development
   metrics and final-test artifacts do not enter fitting or selection.
3. Persist every selected choice for all three seeds before opening the new
   development evaluation stage. Hash source data, ordered IDs and exported
   predictions; retain all candidates and failed fits.
4. Evaluate with TRAIN evidence only and the declared candidate mask. Match
   reference cohorts, held-out truths, exclusions, tie rules and metric code.
   A label-aware candidate filter would invalidate the comparison.
5. The original final-test results have already been viewed, and this model is
   motivated by that project's outcomes. Reused development results are
   exploratory evidence. A fresh code freeze or a new split of the same data
   does not retroactively create an untouched confirmation sample. Do not
   claim fresh final-test confirmation or choose the new model using exposed
   final results.
6. Keep the archived final report, original run artifacts and original package
   unchanged. New results and limitations belong in a separately identified
   extension. Any genuinely prospective confirmation needs an independently
   specified, untouched evaluation source and protocol.

This document audits assignment fit and states review gates. It does not
certify a runner that has not yet been implemented and independently checked.

## Report and delivery constraints

Pages 13–16 require one main-body page per task, at most 200 discussion words
per task, a self-contained main body, Times New Roman 12 pt and line spacing
1.15. Additional-results appendices follow the same page/discussion criteria.
Depth of analysis is explicitly valued over adding models or metrics; a higher
nDCG number alone is not an adequate explanation of why a model is better.

The concise main-body contribution should be the tested mechanism, controlled
comparison and the most informative user/item-group result. Full algebra,
tuning curves and rejected hypotheses can accompany the runnable code and
clearly identified additional results. Preserve space for the required hybrid
and societal findings.

Pages 12 and 17 require one group ZIP containing the PDF and runnable code,
named after the group number, with short run instructions. Page 18 requires
each student's separate, truthful peer assessment. The local slides state
October 26 at 23:59; this audit does not newly verify that deadline live.
