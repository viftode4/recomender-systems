# How this project fits together

We recommend movies from MovieLens 100K and study how combining recommenders
and changing their final rankings affects accuracy and societal objectives.
The assignment's technical core is implemented; the team is reviewing it.

The main ranking task predicts **which movies receive a recorded rating**.
All recorded ratings count as relevant under this setup, including low ratings.
That is different from predicting enjoyment. An unrated movie is not a known dislike.

## Three stages

```text
1. Fit individual recommenders on TRAIN; compare settings using validation
                         ↓
2. Combine their scores/lists; apply diversity, calibration or exposure reranking
                         ↓
3. Freeze choices; evaluate accuracy and societal effects; build the report
```

The pipeline is already implemented. Do not start by running every script or
retraining every model. Begin with the [report](../reports/coursework-complete-v2/report.pdf),
then trace one result to its evidence and code.

## Where the required work lives

Paths below are relative to `final-project/` in the repository, or
`coursework/code/` in the shared pack.

| Need | Start here | What it does |
|---|---|---|
| Understand assignment coverage | `coursework_completion/COVERAGE.md` | Maps Tasks 1–3 and the seven lecture hybrid families to implementations. |
| Individual recommenders and tuning | `run.py`, `experiment.py` | Adapts models, records configurations, creates comparable predictions. |
| Regression and other hybrids | `study.py`, `hybrid_constraints.py` | Learns combinations, selects settings, compares feature/loss variants. |
| Completed hybrid-family additions | `coursework_completion/models.py`, `coursework_completion/run.py` | Implements mixed/meta-level methods and explicit switching/rank-fusion tuning. |
| Accuracy and societal evaluation | `metrics.py`, `societal.py` | Implements independent ranking metrics, reranking and group/exposure analysis. |
| Locked final evaluation | `freeze.py`, `final_evaluate.py` | Saves selected models and verifies them before final-test inference. |
| Current report and archive | `coursework_completion/build_review.py` | Builds the coursework-complete review deliverables from recorded evidence. |
| Repeat the work | `REPRODUCE.md` | Documents input reconstruction, model retraining and observed reproduction differences. |

The [team plan](team-meeting-2026-10-01/TEAM_PLAN.md) turns these areas into five
small first tasks. The [review guide](../coursework_completion/TEAM_REVIEW.md)
explains the questions everyone should be able to answer.

## Training, validation and test in plain words

- **TRAIN:** the examples used to learn individual recommenders and construct
  history/content features.
- **Validation:** examples used during development. We divide validation users
  into groups: one fits hybrid combinations, another chooses settings, and a
  third calibrates the original societal policy. These roles must stay separate.
- **Original TEST:** held-out outcomes opened after the original choices were
  frozen. Those results are already known. They must not be used to choose a new
  model and then presented as an untouched final test.
- **Later completion/research studies:** separate, labelled experiments. The
  hybrid-family completion study reuses the former calibration users for its
  assessment. This is exploratory evidence, not a new independent test.

Only compare rows with the same users, candidates, relevance definition and
available information. The original final-test table and the later completion
table are different evaluations; do not merge their numbers into one leaderboard.
Three overlapping data splits are not three independent datasets.

## Optional research

The [research index](RESEARCH_INDEX.md) links implemented studies and their
findings, including unsuccessful ideas. They add depth but do not establish a
broad performance breakthrough. Conditional-evidence training resumed on
1 October at 12:23 UTC; its selection and assessment are still incomplete.

Two newer directions are **design only, not implemented or trained**:

- [Revisable predictive state](plans/2026-09-29-revisable-predictive-state-design.md):
  learn how a real new observation should update predictions.
- [Useful evidence](plans/2026-09-29-useful-evidence-design.md): anticipate which
  feedback might improve a decision, without treating imagined answers as data.

These proposals are optional team decisions. They are not missing coursework
requirements and do not need a new training sweep before the team can review
the existing deliverables.
