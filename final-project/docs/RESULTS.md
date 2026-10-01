# Results to start from

The existing implementation covers the assignment's main technical work.
The team still needs to review the findings and agree the final explanation.
The [five-page report](../reports/coursework-complete-v1/report.pdf) contains
the task tables; this page gives the few facts worth starting with.

## Ranking: the original frozen coursework evaluation

Mean nDCG@10 across three overlapping MovieLens 100K splits:

| Method | nDCG@10 |
| --- | ---: |
| Standalone model selected on validation | 0.31613 |
| Static regression hybrid | 0.33564 |
| Disagreement regression hybrid | 0.33701 |

The disagreement hybrid improves 6.6% relative to the validation-selected
standalone reference. Static regression is already close; the team should
explain whether the extra complexity adds enough value. These splits share
users and items. This is not evidence of universal superiority or a new best
method across datasets. See the [frozen comparison evidence](../evidence/final-comparisons-v3/SUMMARY.md).

## Societal objectives: show both benefit and cost

In the report's fixed-strength reranking comparison:

| Method | nDCG@10 | Genre diversity |
| --- | ---: | ---: |
| Base context hybrid | 0.333 | 0.798 |
| Diversity reranking | 0.307 | 0.895 |

More genre diversity comes with lower measured ranking accuracy in this setting.
Calibration and exposure have their own trade-offs; Task 3 reports them.
These are declared operational metrics, not proof of causal fairness.

## What remains exploratory

The later hybrid-family completion study uses previously exposed validation
users. Its scores belong in a separate table. Custom-model and grouping
experiments have not established a breakthrough. The conditional-evidence run
is paused and incomplete; the newer feedback-revision ideas are design only.

Choose the next task from the [three proposals](PROPOSALS.md). Read the
[research index](RESEARCH_INDEX.md) only when taking on an optional experiment.
