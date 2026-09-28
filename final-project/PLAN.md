# Team plan

Source: [official project slides](https://brightspace.tudelft.nl/d2l/le/content/844485/viewContent/5179121/View),
read September 28, 2026. [Latest announcements](https://brightspace.tudelft.nl/d2l/lms/news/main.d2l?ou=844485).

## Dates and deliverables

- September 29: individual-assignment feedback / RecBole help.
- October 7–8: 20-minute group project feedback. Choose a slot together at
  https://queue.tudelft.nl/lab/9749. Building 28, Hardy room 1.W950.
- October 26, 23:59 Amsterdam: hard deadline for group PDF + runnable code ZIP
  named GROUP_NUMBER.zip and each student's peer-evaluation Excel file.
- Older schedule dates conflict; use the newer project slides and announcement.

Report: group details on first page; maximum one page and 200 discussion words
per task; Times New Roman 12, spacing 1.15. Main body must stand alone. Confirm
whether 'task' means major task or numbered subtask. Depth and explanations backed
by evidence matter more than adding many models.

## Proposed work division (unassigned until the group agrees)

| Workstream | Deliverable | Dependencies |
|---|---|---|
| Experiment infrastructure + baselines | Shared split, configuration, score exports, run manifest | This starter |
| Individual models | Small validation-only tuning grid and justified model choices | Frozen evaluation protocol |
| Hybrid models | Regression-weighted hybrid and another hybrid approach; coefficient analysis | Compatible expert scores; separate training labels |
| Evaluation + group analysis | Independent metrics, user/item slices, figures explaining differences | Metadata and agreed metric definitions |
| Reranking | Diversity, calibration, user/item fairness; before/after hybrid comparison | Scores, metrics, group definitions |

Each owner writes their task's findings as experiments proceed. All review the
combined report and understand the full pipeline for the individual exam.

## Next implementation milestones

1. Agree on scope and assign owners. Check the updated instructor repository.
2. Freeze shared splits, candidate masking, top-k, metric definitions and IDs.
   All models must have identical split fingerprints and aligned score columns.
3. Tune two or three complementary experts. Starter EASE, ItemKNN and BPR are
   candidates, not a claim that all satisfy the lecturer's expected coverage.
4. Design leakage-free hybrid training: inner split or out-of-fold scores from
   training only to fit the regression, validation for choices, test once after
   freeze. Avoid fitting coefficients on test or reporting training-fit metrics.
   Define negative sampling, score scaling and coefficient constraints explicitly.
5. Implement regression-weighted fusion and a second class-taught hybrid. Analyse
   coefficient stability, ablations, and user/item groups, not just average scores.
6. Add genre-based diversity and calibration plus defined fairness measures.
   Separate user utility disparity from item exposure fairness. Define groups
   using training data or existing metadata, never test-driven thresholds.
7. Implement rerankers; compare rerank-each-expert-then-hybrid with
   hybrid-then-rerank. Study accuracy/beyond-accuracy trade-offs and group impacts.
8. Freeze final settings and run the held-out evaluation. Include limitations,
   reproducibility instructions and contribution/AI-assistance disclosures.

## Bring to the project feedback session

The slides specify time/location, not mandatory artifacts. Suggested preparation:
- Agreed plan, a baseline run and a small table of validation results.
- Proposed hybrid training protocol and evaluation/group definitions.
- Questions: required model/metric coverage; interpretation of per-task page limits;
  expected fairness objectives; acceptable regression targets/negative sampling;
  and what 'other hybrid models' should cover.

## Not implemented yet

Regression fusion, a second hybrid, hyperparameter search, genre metrics, fairness,
rerankers, group plots, statistical comparisons and a final report. Coverage and
novelty alone do not fulfil all beyond-accuracy requirements.
