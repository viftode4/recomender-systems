# Group 24: a working project to build on

Team meeting · 1 October 2026 · review draft, not submitted

**Today’s goal:** agree who owns each part, get everyone running a small check,
and decide which findings belong in the report. There is already working code,
a five-page report draft, and saved experimental evidence.

## A 30-second opening

“I’ve prepared a working base for all three assignment tasks: individual models
and hybrids, independent evaluation, and societal reranking. There is a report
draft and a reproducible code package. Some experiments worked and others did
not; the findings are separated clearly. Today I’d like us to pick ownership,
check the work ourselves, and agree on one focused extension if we want one.”

The implementation and report were prepared with AI assistance. Proposed
ownership is future work, not a claim that teammates wrote or reviewed it already.

## What we can show now

| Deliverable | What is ready | What the team should do next |
| --- | --- | --- |
| Coursework implementation | Course-model adapters, seven lecture hybrid families, independent metrics, user/item groups and societal rerankers | Trace each requirement to its code and evidence |
| Report | Five-page draft: cover, one page for each task, grouping appendix | Review explanations, choose emphasis, supply actual contributor details |
| Reproduction | Pinned setup, public aggregate evidence and packaged tests | Run the short check on another member’s machine; record any setup problem |
| Demonstration | Offline interactive example: change one fictional rating and inspect six forecasts | Use it to explain evidence-sensitive prediction, not as a performance result |
| Research | Custom models, ablations, negative findings and a resumed conditional-evidence study | Review its final assessment when ready; agree any separate follow-up |

## Three findings worth discussing

**1. Combining predictions improved ranking in the frozen coursework study.**
Mean nDCG@10 was **0.33701** for disagreement regression and **0.31613** for
the standalone model selected on validation: **+6.6% relative**. Static
regression was already close at **0.33564**, so the added contextual complexity
needs justification. These are three overlapping splits of MovieLens 100K,
not independent datasets or a state-of-the-art result.
See the [frozen comparisons](../../evidence/final-comparisons-v3/SUMMARY.md).

**2. Accuracy and societal objectives require explicit trade-offs.**
In the report’s fixed-strength comparison, diversity reranking raises genre
diversity from about **0.798 to 0.895**, while nDCG@10 falls from about
**0.333 to 0.307**. The useful discussion is what objective we chose, who benefits,
and what accuracy cost we measured. These operational metrics do not prove
causal fairness. See [Task 3 in the report](../../reports/coursework-complete-v2/report.pdf).

**3. A better idea must earn its place through a test.**
The custom models have not established a breakthrough. The conditional-evidence
study **resumed on 1 October at 12:23 UTC** with six workers and monitoring.
It remains incomplete, with selection/assessment results pending. The newer revisable-state
and useful-evidence models are **design documents only**. The interactive demo
illustrates an earlier model’s response on a fictional profile; it does not
validate those newer designs.

## Five-minute walkthrough

1. Open the report and point to the three task pages.
2. Show the frozen comparison and one accuracy/diversity trade-off.
3. Open the [synthetic demo](../../reports/demo/categorical-field/index.html),
   change the GoldenEye rating, and explain why predictions move. No live
   training or network connection is needed.
4. Run the [quick check](QUICKSTART.md). It checks software and saved evidence;
   it does not retrain models or reproduce the full study.
5. Assign the five seats in the [team plan](TEAM_PLAN.md).

## Decisions to leave with

- A named owner and reviewer for each of the five work areas.
- A first review artifact from each member within the next 48 hours, adjusted
  to the group’s actual availability.
- Agreement to keep the required coursework as the main report and clearly
  label exploratory research.
- One optional research question, with a test and a stopping condition, or an
  explicit decision to spend the time on reproduction and explanation instead.

A suitable optional question is: **does revealing one real additional rating
improve predictions on other hidden ratings?** Compare a learned update with a
fixed-state control under the same observations and compute budget. Train and
select on permitted TRAIN-derived partitions; freeze the rule before assessment.
The previously exposed data makes this an exploratory test. This question is
proposed, not implemented, and has no measured result yet.

## What remains before submission

The [assignment checklist](ASSIGNMENT_CHECKLIST.md) distinguishes implementation
coverage from team review and submission. Four contributor names are still
missing from the draft. Each person must complete their own honest peer feedback.
The PDF specifies no formal bonus-point task; deeper analysis can strengthen
the work but cannot guarantee extra marks.

The saved assignment material lists feedback on **7/8 October** and the final
deadline on **26 October at 23:59**. Verify the current dates and feedback signup
in Brightspace: its connector returned HTTP 403 during this preparation, so
these dates were not freshly confirmed. The local code/report package is not a
submission receipt.

## Message you can paste in the group chat

“Am pregătit o bază pentru proiect: cod pentru cele trei task-uri, un draft de
raport, rezultate salvate și un demo mic. Vă trimit pachetul ca să avem ceva
concret de discutat azi. Aș vrea să împărțim cele cinci zone de lucru, să verificăm
fiecare câte o parte și să alegem împreună dacă mai facem un experiment. E un
draft făcut cu ajutor AI, deci trebuie să-l înțelegem și să-l verificăm noi înainte
să-l considerăm final.”
