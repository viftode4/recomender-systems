# Five-person plan for today's meeting

The assignment's technical core is already built. Our next job is to understand,
check and improve it together. These are proposed seats, **not past contribution
claims**. Vlad is suggested for integration; agree all five owners today.

Start with the [project map](../PROJECT_MAP.md) and
[current report draft](../../reports/coursework-complete-v1/report.pdf).
Training is paused. New revisable-state ideas are design documents only.
The [three proposals](../PROPOSALS.md) recommend trust across recommenders and
societal trade-offs as the main story, with feedback revision as an optional extension.

## Pick a seat in two minutes

Each first task should fit a 45–90 minute work block. Save notes in
`meeting/reviews/` in the shared pack, or
`docs/team-meeting-2026-10-01/reviews/` in the checkout; these notes do not exist yet.
Code paths below start at `coursework/code/` in the pack or `final-project/`
in the checkout. Use the [review template](REVIEW_TEMPLATE.md).

| Seat / suggested owner | First concrete action | Finished output | Reviewer |
|---|---|---|---|
| **1. Integration / Vlad** | Trace one method through `experiment.py`, `freeze.py` and `final_evaluate.py`; compare with the completion protocol. | `01-protocol.md`: one diagram showing which data fits, selects and evaluates, plus any mismatch. | Seat 5 |
| **2. Individual models / open** | Choose one neighborhood and one learned model in `run.py`; locate their settings in `experiment.py` and their report rows. | `02-models.md`: two short explanations, exact settings/budgets, linked results and one limitation each. | Seat 3 |
| **3. Hybrids / open** | Read `coursework_completion/COVERAGE.md`; inspect `models.py` in that folder and check one selected setting against [the aggregate evidence](../../coursework_completion/results-v1/aggregates.json). | `03-hybrids.md`: seven-family map and one checked selection, explaining overlap between the taxonomy labels. | Seat 2 |
| **4. Metrics and societal effects / open** | Hand-calculate one small nDCG example using `metrics.py`; trace one report reranking row to its aggregate evidence. | `04-evaluation.md`: worked example and checked accuracy/diversity or exposure trade-off, including who loses accuracy. | Seat 1 |
| **5. Report and reproduction / open** | Read the PDF as a new reader; run the exact command in the [quickstart](QUICKSTART.md) for your checkout/pack layout and inspect the package verification receipt. | `05-reproduction.md`: exact commands/results and page-specific report issues; distinguish passing synthetic tests from reproducing trained results. | Seat 4 |

Record author, date, checked artifact version/hash and unresolved issues in each
note. Each person explains and edits their own section; Seat 5 coordinates the
report. A setup failure is a finding to record, not a reason to invent a pass.

## One optional improvement after each review

| Seat | Bounded follow-up, chosen from an actual finding |
|---|---|
| 1 | Add one end-to-end example to the project map, with exact input/output file names. |
| 2 | Improve one model's explanation or document a tuning-budget limitation with evidence. |
| 3 | Add a tiny worked example showing the difference between mixed lists and weighted scores. |
| 4 | Add one clearly labelled trade-off plot from existing aggregates, with both accuracy and the societal objective. |
| 5 | Fix one confusing setup instruction or figure caption, then have its reviewer retry it. |

These can run in parallel without training. Preserve frozen scientific files
and original results; make agreed explanation/report changes in a new version.

## Twenty-minute meeting

| Minutes | Outcome |
|---|---|
| 0–3 | Open the report and project map; establish what already exists. |
| 3–7 | Explain one result and one accuracy/societal trade-off. |
| 7–12 | Choose five owners, review partners and first tasks. |
| 12–16 | Agree report scope and whether to pursue one optional research question. |
| 16–20 | Set the next check-in and name the artifact each person will bring. |

## Four decisions before leaving

1. **Owners:** fill all five names, first tasks and a mutually workable check-in.
2. **Report:** keep Tasks 1–3 self-contained; agree which exploratory findings,
   if any, belong in the appendix and what needs lecturer clarification.
3. **Evidence:** review one shared artifact version. Keep original TEST results
   fixed; label reused-data studies exploratory and retain reproduction differences.
4. **Optional research:** pursue or defer one question: does revealing one real
   rating improve predictions on other concealed items with equal feedback?
   If pursued, assign an owner and a small executable protocol before training.

Over the next 48 hours, adjusted to availability: write notes in parallel,
exchange them with review partners, then each present one checked finding at the
next meeting. This is a proposed cadence, not a course deadline. Training stays
paused during these tasks; no extra sweep is required to finish the coursework.

The [research designs](../PROJECT_MAP.md#optional-research) are proposals, not
results. No breakthrough is established. Use the fuller
[review guide](../../coursework_completion/TEAM_REVIEW.md) for explanations and
questions everyone should understand. Record genuine contributions; each member
later completes their own peer feedback from actual work.
