# Three project proposals

**Recommendation:** make proposals 1 and 3 the main assignment story. Keep
proposal 2 as one optional, bounded experiment after the team reviews the base.
These are questions to investigate, not promises of improvement or novelty.

## 1. When should we trust each recommender?

**Question:** do users with short and long rating histories benefit from different
models, and can a simple switching rule use that difference?
**Start:** inspect `fit_switch` and `switch_scores` in
[models.py](../coursework_completion/models.py). From the existing
[aggregates](../coursework_completion/results-v1/aggregates.json), make one table
of accuracy by activity group for individual models and the switching hybrid.

**First deliverable:** a short explanation of the rule, the group table and one
worked user example. Trace each number to a seed, cohort and saved setting.
Accept the review when another teammate can reproduce the table from saved data.

**Known versus open:** switching is implemented; the completed tuning study left
assessment rankings unchanged. Whether another justified rule helps is open.
Any new rule must be selected using development data, with reused assessment
clearly labelled exploratory. This supports the models and evaluation tasks.

## 2. Can one new rating usefully change the model's mind?

**Question:** after seeing one real rating, can our own learned update improve
predictions for other, still concealed movies?

**Start:** turn the [feedback design](plans/2026-09-29-useful-evidence-design.md)
into one small episode: visible history, one revealed rating and distinct target
ratings. Specify `predict`, `observe` and an evidence ledger before implementation.

**First deliverable:** an executable toy example where feedback changes predictions
and replaying the same observation does not count it twice. Before training,
declare disjoint TRAIN-derived fit/development pools and fixed episodes. Compare
loss before and after the same reveal, including a no-update control; keep shared
weights fixed during assessment and exclude the revealed item from target scores.

**Acceptance:** correct mechanics justify a small pilot; useful transfer requires
lower paired target loss, reported uncertainty and the fraction of harmful updates.
**Status:** design only, not implemented or trained. This is an optional extension;
its rating-prediction endpoint differs from the main recorded-interaction ranking task.

## 3. What accuracy cost buys more useful recommendation lists?

**Question:** how do diversity, genre calibration and item exposure change when
we rerank recommendations, and which user groups lose accuracy?
**Start:** trace one result through [societal.py](../societal.py) and the saved
[societal audit](../evidence/final-societal-2026/societal-audit.json).
Use the [report](../reports/coursework-complete-v2/report.pdf) to locate its context.

**First deliverable:** one small trade-off table or plot with accuracy and one
societal metric, plus the worst affected user group. State the metric direction,
cohort and reference method; compare only matching evaluations.

**Acceptance:** another teammate can trace the plotted values to saved evidence
and explain both the benefit and cost. Rerankers and audits already exist;
the best choice depends on the stated objective. Exposure is an operational
measure, not proof of real-world fairness. This supports evaluation and societal tasks.
