# Five-person execution plan

Source: [project brief](https://brightspace.tudelft.nl/d2l/le/content/844485/viewContent/5179121/View)
and [announcements](https://brightspace.tudelft.nl/d2l/lms/news/main.d2l?ou=844485), read September 28, 2026.

The implementation, frozen final evaluation and review report are complete as of
September 28, 2026. The [current review PDF](reports/final-review-v1/report.pdf)
is still a draft pending human review and actual contributor information. The
review archive is at `packages/final-review-v1/24.zip`; this is a local
review artifact, not a submission receipt.

These five seats remain a proposed division for review and ownership. They are
not claims about who implemented the existing work. Names and contributions must
reflect actual work; confirm owners at the group meeting.

| Seat | Owns | Concrete handoff | First review task |
|---|---|---|---|
| 1: integration / research lead | Shared protocol, architecture, experiment orchestration | Reproducible commands, run manifests, completed model freezes | Audit training/validation/test boundaries and final artifact hashes |
| 2: expert models | Course CF, linear, content, graph and neural baselines | Recorded tuning grids, model behavior explanations, timings | Verify exact popularity, per-user Random and disclosed tuning budgets |
| 3: hybrid research | Regression hybrids, switching, contrast controls and categorical field research | Coefficient analysis, feature ablations, matched controls and negative results | Independently check learned weights, routing controls and paired comparisons |
| 4: evaluation / societal aspects | Metrics, calibration, diversity, user/item fairness and reranking | Metric definitions, group analysis, candidate-pool/frontier figures | Check genre encoding, exposure objective and user utility retention |
| 5: report / reproducibility reviewer | Related work, claims, report figures, clean setup and packaging | Self-contained report and a successful clean-clone replay | Challenge novelty/performance claims and verify every reported number |

All five understand the pipeline for the individual exam. Each writes/reviews the
section they actually contributed to. Seat 5 coordinates the report rather than
writing everyone else's analysis alone. Seat 1 can initially build all modules;
that does not establish the other members' contributions.

## Implemented and evaluated

- `run.py`: twelve model adapters; corrected FISM/NGCF paths; independent metrics, score/split exports and timings.
- `experiment.py`: predeclared small expert grids, three data seeds, fixed-budget
  training and selection on the meta-fit cohort, followed by hybrid studies.
- `study.py`: regression hybrids, feature/loss ablations, rank fusion and switching;
  genre metrics, activity groups, head/tail analysis; reranker trade-offs and order
  comparisons; head/tail candidate feasibility; independently calibrated group policies.
- `exception_model.py`: matched contrast-transfer and candidate contrast gates,
  randomized controls, rating-aware linear baselines and explicit-liked metrics.
- `negative_information.py` / `negative_information_experiment.py`: a separate
  context/probe audit with shuffled weights and fixed-margin dislike controls.
- The independent categorical evidence field is implemented and evaluated:
  conditional rating prediction, joint movie/rating-record prediction, fixed-flow
  and hard-clamp controls, and a separately declared 400-epoch convergence study.
  [Final matched comparisons](evidence/final-field-comparisons-v1/RESULTS.md) retain
  the unsuccessful results: the custom fields do not beat the strong references,
  and adaptive routing has no consistent demonstrated advantage.
- `freeze.py` / `final_evaluate.py`: source-sealed saved-model inference with
  exact validation replay, fixed transforms and no test refitting. The controlled
  final evaluation has now run; additional rating-aware and field bundles were
  also frozen before their one-time evaluation, including EASE/SLIM/PositiveEASE
  references under both relevance definitions.
- `report.py` / `package_project.py`: the current PDF includes final results and
  supplemental research evidence; final human review remains.
  `audit_study.py` / `compare_frozen.py`: taste groups and predefined comparisons.
- Tests exercise metric definitions, score masking, regression, divergence,
  ranking, cohort isolation and reranking behavior. Targeted tests and independent
  reviews cover the new models, matched controls, freezes and final evaluators.
- Temporal rating-entry sensitivity and an isolated training-environment check
  are complete. A teammate should still verify the final packaged instructions
  from a clean environment and distinguish reproducibility from new research.

See RESEARCH.md for model equations, hypotheses, definitions and known limitations.
Local generated runs are ignored by Git. Curated aggregate evidence belongs under
`evidence/`, with exact source hashes and reproduction instructions.

The final test results are now known. Preserve the frozen configurations and
reported evaluation; do not use this test set to tune models, select new variants
or relabel a later experiment as an untouched final test. Any experiment inspired
by these results or lecturer feedback is exploratory on the existing data and
needs a new, separately declared assessment before making a fresh confirmatory
claim. Checking a saved result or correcting documented reporting errors is not
permission to rerun model selection on the test outcomes.

## Milestones

| When | Outcome |
|---|---|
| Completed Sep 28 | Implementation, three-seed development studies, model/control audits, frozen final evaluation and review PDF |
| Now | Inspect the review archive; read the final evidence and record review issues |
| Group meeting from Sep 30 | Assign five proposed seats; collect actual names/contributions; agree what each person can explain |
| By Oct 5 | Team review of code, metrics, figures, negative results and report claims; clean-environment package check |
| Oct 7–8 | Bring the completed results and open methodology/report-format questions to lecturer feedback |
| By Oct 16 | Address feedback and document changes; any new experiments remain exploratory unless separately assessed on untouched evidence |
| By Oct 19 | Agree the final narrative, actual contribution statement and submission contents; retain existing scientific freezes |
| Oct 20–23 | Proofread and rebuild the reviewed report/package; verify filenames, contents and reproducibility; complete genuine peer feedback |
| Oct 26 before 23:59 | Submit group ZIP plus individual peer-feedback spreadsheets |

No submission is automated by this plan.

## Official requirements and coverage

| Requirement | Current implementation / remaining review |
|---|---|
| 1.1–1.2 individual recommenders and tuning | Twelve class-related models/baselines; fixed grids, extended LightGCN budgets and model corrections documented |
| 1.3 weighted hybrid learned by regression | Sum-to-one constrained ridge matches lecture equation; unconstrained variants explicit extensions |
| 1.4–1.5 other hybrids and tuning | Lecture-verified switching and cascade; additional RRF/contextual/pairwise variants |
| 2.1 independent accuracy/beyond-accuracy metrics | Precision, Recall, nDCG, MRR, Hit, novelty, coverage, diversity, calibration and group diagnostics |
| 2.2 model comparisons | Controlled final evaluation complete for the frozen methods; paired comparisons and matched strong references reported |
| 2.3–2.4 coefficient and insightful analysis | Feature/loss ablations, paired intervals, harm fractions, candidate feasibility and retained negative results; team interpretation review remains |
| 2.5–2.6 user/item groups and new hybrid ideas | Activity, taste breadth, training contradiction proxy, head/tail items; contextual/switching and custom contrast models |
| 3.1–3.2 rerankers and trade-offs | Diversity, genre calibration, user popularity calibration, item exposure; independently calibrated utility-budget policy |
| 3.3 ordering | Rerank each expert then RRF versus RRF then rerank; full candidate support preserved |
| 3.4 group impacts | User utility retention, taste groups, discounted size-normalized exposure, whole-catalog Gini/entropy and head/tail recall |
| Runnable code/report/peer evaluation | Final review PDF and local archive prepared, isolated environment verified; human package review and four real teammate evaluations remain individual work |

## Feedback-session questions

The final-project session is October 7–8, 20 minutes per group, Building 28,
Hardy 1.W950. Book together: https://queue.tudelft.nl/lab/9749.
The September 29 session concerns individual assignments and RecBole help.

Bring one architecture diagram, a comparison table, an ablation plot and a
fairness frontier. Ask:

1. We use three major-task pages and a cover, 12pt Times New Roman, 1.15 spacing. Is that interpretation acceptable?
2. Is our validation-user meta-fit/development division appropriate for stacking?
3. Are training-activity utility budgets and catalog exposure parity acceptable
   user/item fairness definitions, or are other objectives expected?
4. Does the report distinguish the required model comparisons, finite training
   budgets and unsuccessful architectural experiments clearly enough? The final
   test is already opened; additional feedback-driven experiments would be exploratory.
5. Is an appendix suitable for robustness, candidate feasibility and negative results?

## Report and delivery constraints

Group PDF plus runnable code directory in one ZIP named after the group number;
one group submission. Each student submits the peer-feedback Excel file separately.
Deadline October 26, 23:59 Amsterdam is hard. Older schedule dates conflict: use
newer slides/announcements. Report: group information first, Times New Roman 12,
spacing 1.15, maximum one page and 200 discussion words per task; clarify granularity.
The main body must stand alone. Findings need evidence and explanation, not a
leaderboard alone. Record actual contributions and disclose assistance as required.
