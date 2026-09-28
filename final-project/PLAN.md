# Five-person execution plan

Source: [project brief](https://brightspace.tudelft.nl/d2l/le/content/844485/viewContent/5179121/View)
and [announcements](https://brightspace.tudelft.nl/d2l/lms/news/main.d2l?ou=844485), read September 28, 2026.

The technical build proceeds now. These five seats are a proposed division for
review, extensions and ownership, not a dependency on teammates starting first.
Names/contributions must reflect actual work; confirm owners at the group meeting.

| Seat | Owns | Concrete handoff | First review task |
|---|---|---|---|
| 1: integration / research lead | Shared protocol, architecture, experiment orchestration | Reproducible commands, run manifests, final model freeze | Run the pipeline; audit training/validation/test boundaries |
| 2: expert models | EASE, ItemKNN, BPR and reference baselines | Small tuning grid, model behavior explanations, timings | Verify exact popularity, per-user Random, tuning budget parity |
| 3: hybrid research | Static/contextual regression, pairwise objective, switching and rank fusion | Coefficient analysis, feature ablations, robustness comparisons | Independently check learned weights and paired comparisons |
| 4: evaluation / societal aspects | Metrics, calibration, diversity, user/item fairness and reranking | Metric definitions, group analysis, candidate-pool/frontier figures | Check genre encoding, exposure objective and user utility retention |
| 5: report / reproducibility reviewer | Related work, claims, report figures, clean setup and packaging | Self-contained report and a successful clean-clone replay | Challenge novelty/performance claims and verify every reported number |

All five understand the pipeline for the individual exam. Each writes/reviews the
section they actually contributed to. Seat 5 coordinates the report rather than
writing everyone else's analysis alone. Seat 1 can initially build all modules;
that does not establish the other members' contributions.

## Working now

- `run.py`: five model adapters; independent validation metrics; scores/splits.
- `experiment.py`: predeclared small expert grids, three data seeds, fixed-budget
  training and selection on the meta-fit cohort, followed by hybrid studies.
- `study.py`: regression hybrids, feature/loss ablations, rank fusion and switching;
  genre metrics, activity groups, head/tail analysis; reranker trade-offs and order
  comparisons; candidate feasibility and adaptive expansion; group utility budgets.
- Tests exercise metric definitions, score masking, regression, divergence,
  ranking, cohort isolation and reranking behavior.

See RESEARCH.md for model equations, hypotheses, definitions and known limitations.
Local generated runs are ignored by Git. Curated aggregate evidence belongs under
`evidence/`, with exact source hashes and reproduction instructions.

## Milestones

| When | Outcome |
|---|---|
| Now | Runnable end-to-end development study, explicit research questions, initial evidence |
| Group meeting from Sep 30 | Assign five seats; agree scope, definitions and what the group can explain |
| By Oct 5 | Independent code/metric review; baseline, ablation and fairness figures ready |
| Oct 7–8 | Bring results and open methodology questions to lecturer feedback |
| By Oct 16 | Address feedback; temporal-split sensitivity and clean-environment reproduction; freeze useful ideas |
| By Oct 19 | Freeze final expert/hybrid/reranker configurations and evaluation protocol |
| Oct 20–23 | One controlled test evaluation, final report, runnable package, peer review |
| Oct 26 before 23:59 | Submit group ZIP plus individual peer-feedback spreadsheets |

No submission is automated by this plan.

## Official requirements and coverage

| Requirement | Current implementation / remaining review |
|---|---|
| 1.1–1.2 individual recommenders and tuning | EASE/ItemKNN/BPR grids plus Random/ExactPop; verify expected model coverage |
| 1.3 weighted hybrid learned by regression | Static ridge with coefficients and multiple regularization values |
| 1.4–1.5 other hybrids and tuning | Group switching, reciprocal-rank fusion, contextual/pairwise variants; check lecture alignment |
| 2.1 independent accuracy/beyond-accuracy metrics | Precision, Recall, nDCG, MRR, Hit, novelty, coverage, diversity, calibration and group diagnostics |
| 2.2 model comparisons | All methods compared on identical development users; final test comparison pending freeze |
| 2.3–2.4 coefficient and insightful analysis | Feature/loss ablations, paired intervals, harm fractions, candidate feasibility; interpretation/review pending |
| 2.5–2.6 user/item groups and new hybrid ideas | Activity terciles, head/tail items, contextual and group-switching models |
| 3.1–3.2 rerankers and trade-offs | Diversity, calibration, exposure rerankers; group utility-budget policy and held-out group audit |
| 3.3 ordering | Rerank each expert then RRF versus RRF then rerank; full candidate support preserved |
| 3.4 group impacts | User-group utility retention plus item exposure and head/tail recall |
| Runnable code/report/peer evaluation | Code and development evidence exist; final report and peer evaluation remain |

## Feedback-session questions

The final-project session is October 7–8, 20 minutes per group, Building 28,
Hardy 1.W950. Book together: https://queue.tudelft.nl/lab/9749.
The September 29 session concerns individual assignments and RecBole help.

Bring one architecture diagram, a comparison table, an ablation plot and a
fairness frontier. Ask:

1. Does the one-page / 200-discussion-word limit apply to major tasks or numbered subtasks?
2. Is our validation-user meta-fit/development division appropriate for stacking?
3. Are training-activity utility budgets and catalog exposure parity acceptable
   user/item fairness definitions, or are other objectives expected?
4. Is this model/metric coverage sufficient? Does group switching satisfy the
   requested additional class-taught hybrid?
5. Is an appendix suitable for robustness, candidate feasibility and negative results?

## Report and delivery constraints

Group PDF plus runnable code directory in one ZIP named after the group number;
one group submission. Each student submits the peer-feedback Excel file separately.
Deadline October 26, 23:59 Amsterdam is hard. Older schedule dates conflict: use
newer slides/announcements. Report: group information first, Times New Roman 12,
spacing 1.15, maximum one page and 200 discussion words per task; clarify granularity.
The main body must stand alone. Findings need evidence and explanation, not a
leaderboard alone. Record actual contributions and disclose assistance as required.
