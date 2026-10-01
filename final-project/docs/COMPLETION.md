# Assignment completion and follow-up work

Checked against the 19-page assignment brief on 1 October 2026.
All 15 numbered technical subtasks have implementations and recorded results.
The current deliverable is a **complete technical draft for team review**;
contributor details, individual peer feedback and course submission remain open.

## Completed assignment work

| Requirements | Delivered work |
| --- | --- |
| 1.1–1.2 | Individual recommenders, declared tuning grids and recorded comparisons |
| 1.3–1.5 | Regression-weighted hybrid, seven lecture-family mechanisms and hybrid tuning |
| 2.1–2.2 | Independent accuracy/beyond-accuracy metrics, random and popularity baselines |
| 2.3–2.6 | Coefficients, user/item-group analysis, controlled studies and implications for hybrids |
| 3.1–3.4 | Diversity, calibration, user/item objectives, trade-offs, ordering and group impacts |

The [requirement checklist](team-meeting-2026-10-01/ASSIGNMENT_CHECKLIST.md)
links each subtask to its implementation and evidence. The
[coverage explanation](../coursework_completion/COVERAGE.md) gives the hybrid mapping.
The assignment defines no separate formal bonus task.

## Current draft and reproducible code

- [Five-page report](../reports/coursework-complete-v2/report.pdf): cover, one page
  for each main task and one appendix. Version 2 clarifies the existing user-fairness
  policy; recorded experimental results are unchanged.
- [Group 24 code/report archive](../packages/coursework-complete-v2/24.zip),
  with [extraction and report-replay checks](../packages/coursework-complete-v2/verification.json).
- [Reproduction instructions](../REPRODUCE.md), including measured NGCF replay
  differences. Running the small synthetic check is not full model reproduction.

Earlier report and archive versions remain preserved. Original frozen TEST,
later completion on reused validation-calibration users and the nested TRAIN
grouping study remain distinct evidence stages.

## What the team should contribute next

Use the [five-person follow-up plan](team-meeting-2026-10-01/TEAM_PLAN.md).
Each person owns an improvement and its explanation: integration, user-group
analysis, hybrid failure analysis, societal trade-off plots, or report/reproduction.
Review the existing result supporting that improvement and record actual work.

Before submission, supply all five names, agree the report, complete each
member's own peer-feedback workbook, check the live course instructions and
submit the agreed archive. No course submission has been performed.

## Research in progress

The conditional-evidence study resumed on 1 October at 12:23 UTC from its saved
checkpoints with six workers and a monitor. It tests the custom rating-graph
reader against matched controls and fixed references. Selection, any gated
retrieval, assessment and final export are not complete. This dated note is
not a live process status or a performance result.

The newer feedback-revision proposals remain designs only. See the
[research index](RESEARCH_INDEX.md) for scope and operational links. The coursework
draft does not depend on these unfinished experiments.
