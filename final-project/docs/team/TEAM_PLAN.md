# Five-person plan for today's meeting

The assignment's technical core is already built. Our next job is to understand,
check and improve it together. These are proposed seats, **not past contribution
claims**. Vlad is suggested for integration; agree all five owners today.

Start with the [project map](../PROJECT_MAP.md) and
[current report draft](../../reports/coursework-complete-v2/report.pdf).
The [three proposals](../PROPOSALS.md) recommend trust across recommenders and
societal trade-offs as the main story, with feedback revision as an optional extension.

## Pick a seat in two minutes

Each first task should fit a 45–90 minute work block. Save notes in
`meeting/reviews/` in the shared pack, or
`final-project/docs/team/reviews/` from the checkout root.
Code paths below start at `coursework/code/` in the pack or `final-project/`
in the checkout. Use the [review template](REVIEW_TEMPLATE.md).

| Seat / suggested owner | First concrete action | Finished output | Reviewer |
|---|---|---|---|
| **1. Integration / Vlad** | Trace fitting, selection and evaluation through `experiment.py`, `freeze.py` and `final_evaluate.py`; reconcile the completion audit and each teammate's changes. | `01-protocol.md`: a data-partition diagram, linked audit findings and an integration checklist. | Seat 5 |
| **2. User-group analysis / open** | Read the existing user-group results; compare hybrid gains for sparse versus active histories and identify a weak group. Keep comparisons within one evaluation table. | `02-user-groups.md`: a group table with user counts, gain/loss, exact evidence paths and a limitation or follow-up hypothesis. | Seat 3 |
| **3. Hybrid failure explanations / open** | Trace two contrasting examples through component scores and the chosen hybrid; explain when combination helps or lets a poor expert dominate. Start from existing predictions where available. | `03-hybrid-failures.md`: two anonymized worked cases, component/combined scores and an explanation checked against the code. | Seat 2 |
| **4. Societal trade-offs / open** | Turn existing accuracy/diversity, calibration or exposure evidence into a labelled plot; investigate which groups pay the accuracy cost. | `04-tradeoffs.md` and a figure: axes, setting, cohort, evidence source and a short interpretation. Use only measured settings. | Seat 1 |
| **5. Report and reproduction / open** | Run the [quickstart](../QUICKSTART.md) on your machine, record the package receipt, and revise one report section or figure caption using the reviewed findings. | `05-reproduction.md`: exact commands/results, platform and corrected report text; distinguish the small check from trained-model reproduction. | Seat 4 |

Record author, date, checked artifact version/hash and unresolved issues in each
note. Each person explains and edits their own section; Seat 5 coordinates the
report. A setup failure is a finding to record, not a reason to invent a pass.

## Deeper follow-up after the first artifact

| Seat | Bounded follow-up, chosen from an actual finding |
|---|---|
| 1 | Integrate reviewed changes and give the team one reproducible command for the final draft. |
| 2 | Check whether a group's apparent gain survives a different sensible grouping; report sample sizes. |
| 3 | Test a proposed failure explanation on more cases without using original TEST to tune a new model. |
| 4 | Add a second measured objective or group breakdown; state where data is insufficient. |
| 5 | Have a reviewer retry corrected setup instructions and check the PDF against the assignment limits. |

These contributions extend the existing technical core and can start in parallel
without new training. If a diagnostic needs local predictions, use the recorded
reproduction path; do not commit personal rating histories. Preserve frozen scientific files
and original results; make agreed explanation/report changes in a new version.

Use the [meeting brief](MEETING_BRIEF.md) for the twenty-minute agenda.

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
next meeting. This is a proposed cadence, not a course deadline. Optional research has its
own [status and protocols](../RESEARCH_INDEX.md).

The [research designs](../PROJECT_MAP.md#optional-research) are proposals, not
results. No breakthrough is established. Use the fuller
[review guide](../../coursework_completion/TEAM_REVIEW.md) for explanations and
questions everyone should understand. Record genuine contributions; each member
later completes their own peer feedback from actual work.
