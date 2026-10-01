# What we found

The [report draft](../reports/coursework-complete-v2/report.pdf) covers all three
assignment tasks. These are its main findings.

## Combining models improves ranking

Mean nDCG@10 on the original frozen test, across three overlapping MovieLens
100K splits. Higher is better.

| Method | nDCG@10 |
| --- | ---: |
| Standalone model selected on validation | 0.31613 |
| Static regression hybrid | 0.33564 |
| Context hybrid, using user/item features | 0.33262 |
| Disagreement hybrid | 0.33701 |

Disagreement fusion gains 6.6% over the selected standalone reference. Static
regression is already close. Context scores below static, so adaptive weights
do not improve the average here. See the [comparisons and uncertainty](../evidence/final-comparisons-v3/SUMMARY.md).
These splits share users and observations; they are evidence from one dataset.

## Reranking has a measurable benefit and cost

The same context hybrid, with reranking strength 0.5:

| Method | nDCG@10 ↑ | Genre diversity ↑ | Genre JSD ↓ | Head exposure ↓ |
| --- | ---: | ---: | ---: | ---: |
| Base context | 0.333 | 0.798 | 0.141 | 0.981 |
| Diversity | 0.307 | 0.895 | 0.131 | 0.967 |
| Genre calibration | 0.329 | 0.806 | 0.125 | 0.980 |
| Item exposure | 0.318 | 0.793 | 0.141 | 0.931 |

JSD measures deviation from the user's historical genre proportions. Head
exposure is the share of slots given to the most popular 20% of catalog items;
lowering it is our chosen exposure objective. Diversity increases at an accuracy
cost, while exposure reranking gives more slots to tail items.

The [saved aggregates](../evidence/final-primary-v3/aggregates.json) contain all
measured strengths and user/item-group breakdowns. These support the next task:
explain **which groups benefit and which lose accuracy**.

## Keep later studies separate

The hybrid-completion study reuses validation users. The recording-group study
uses a nested TRAIN split and found no ranking advantage. Their evaluation data
differ from the original test, so their scores belong in separate tables.

Choose a review task from the [team plan](team/TEAM_PLAN.md).
Optional research has its own [index and dated status](RESEARCH_INDEX.md).
