# What the completed experiment establishes

The categorical model produced a small standalone improvement on the reused
MovieLens 100K development cohorts. It did not establish a substantial hybrid
gain or a field-level breakthrough. All 243 predeclared search fits completed;
the independent audit reproduced the selected predictions, hybrid fitting and
ranking metrics. Read the [complete results](results-v1/RESULTS.md),
[numerical audit](audit-v2/RESULTS.md) and [prior-art comparison](RELATED_WORK.md).

## Standalone result

Equal-seed means; each split has 472 development users. These are overlapping,
previously used development cohorts after the original project's TEST exposure.

| Metric | Expanded binary EASE | Categorical | Shuffled-category control |
|---|---:|---:|---:|
| All-observed nDCG@10 | 0.261688 | 0.263154 | 0.261388 |
| All-observed Precision@10 | 0.174788 | 0.176554 | 0.174718 |
| All-observed Recall@10 | 0.241746 | 0.243831 | 0.240583 |
| All-observed MRR@10 | 0.431761 | 0.430306 | 0.431823 |
| Liked-rating nDCG@10 | 0.245301 | 0.246720 | 0.245218 |
| Liked-rating Recall@10 | 0.284725 | 0.287879 | 0.284000 |
| Liked-rating MRR@10 | 0.371663 | 0.370481 | 0.372050 |

The primary nDCG gain is about **0.56% relative**, or **0.001466 absolute**.
Precision and recall improve while MRR declines: there is no all-metric win.
The original three-penalty EASE grid scores 0.261827, slightly above expanded
binary. More tuning choices do not guarantee better development performance.

Real categories select categorical coefficients in seeds 2026 and 2027, with
category penalties ten times the binary penalty. Seed 2028 selects the binary
fallback. Their per-seed nDCG changes versus expanded binary are +0.001822,
+0.002576 and exactly zero. Only 13.35%, 22.25% and 0% of development users,
respectively, improve strictly; average gains are not improvements for everyone.

The real-category family exceeds the shuffled family in the first two splits
and ties it in the third. This is limited evidence that which user supplies a
rating category matters within this model. It is not a causal explanation of
preferences. Each categorical family has 45 selectable candidates, versus nine
for binary; both categorical families contain the binary fallback. The strongest
liked-focused reference, PositiveEASE, still has higher liked nDCG (0.257489),
under its different selection objective and information budget.

## Hybrid and group result

The binary-plus-SLIM hybrid scores 0.265593. Adding real categories yields
0.265666, an absolute change of only 0.000073. Adding shuffled categories yields
0.265644. Per-seed real-category hybrid changes are +0.000682, +0.000934 and
-0.001398. This does **not** demonstrate meaningful additional hybrid utility.
When the categorical model equals binary, the third expert is a duplicate;
retaining it changes ridge regularization geometry. That is not new information.

Categorical versus binary mean activity-group nDCG rises from 0.240050 to
0.241975 for sparse users, 0.218514 to 0.218904 for medium users, and 0.331838
to 0.334079 for dense users. These small mean changes do not establish per-user
fairness. TRAIN activity defines the groups before evaluation.

Tail conditional recall rises from 0.008720 to 0.009667, with tail exposure from
1.045% to 1.095%. The small absolute base makes a large relative percentage
misleading. About 98.9% of recommendations still go to the popular head. The
categorical hybrid slightly reduces tail recall versus the baseline hybrid.
The original assignment's dedicated reranking study addresses larger explicit
accuracy/exposure trade-offs; this reconstruction study does not replace it.

## Engineering and evidence

We built the model and its exact solver, rather than importing its predictions.
It handles 10,098 binary/category features using a 943-user primary linear
system and per-target six-dimensional corrections. All target-derived channels
are excluded. Every grid fit used the ordinary numerical path; the largest
relative solve residual was below 1.8e-13. A single TRAIN timing check measured
0.414 seconds fitting plus 0.202 seconds scoring for one categorical setting,
on this machine only. See [the recorded timing](TRAIN_TIMING.json).

The independent checker replayed 15 distinct selected/original-grid
configurations across the three seeds and rebuilt hybrid fitting with a separate
KKT implementation. Maximum aggregate ranking-metric discrepancy was 3.61e-16.
Old EASE exports agree numerically within 2.7e-6. Eight of nine reference
configurations have identical top-ten lists for all 943 users; one near-tie in
the ninth changes one user's list, with unchanged meta nDCG. New selected-score
replays are exact. All candidate counts and maximum/tie selection rules passed.

The defensible contribution is this implemented model, exact derivation,
controlled comparison and measured trade-offs. A broader scientific claim would
require fresh data, stronger gains, and a demonstrated distinction from prior
methods. Existing TEST data cannot become an untouched confirmation set again.
