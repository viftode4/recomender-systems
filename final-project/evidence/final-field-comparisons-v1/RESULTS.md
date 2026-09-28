# Frozen field models versus fixed flow and strong references

Every comparison was fixed before test access: conditional field, joint field with a 100-epoch budget, and its separate 400-epoch convergence study, each against fixed flow, EASE, SLIMElastic and liked-selected PositiveEASE. All-observed relevance counts every test-rated item. Liked-record relevance counts only ratings at least four; users without a test like are excluded only from that endpoint.

Conditional models use P(rating >= 4) for both endpoints. Joint models use total event mass for all-observed ranking and categories 4/5 event mass for liked-record ranking. References use their original frozen ranking. All rankings exclude the same TRAIN and validation items and retain frozen TRAIN-only model context.

Descriptive paired user-bootstrap percentile intervals conditional on the frozen models and this MovieLens dataset. Bonferroni adjustment covers all 24 predeclared nDCG comparisons within each seed, not across seeds; bootstrap coverage is approximate. Repeated splits share users and items. Cross-seed means have no pooled confidence interval. Objectives and tuning budgets differ across model families, so reference comparisons do not isolate architecture. No direct joint400-minus-joint100 contrast is included; comparing their reference intervals does not test a budget effect. No model is selected using these test outcomes.

| Track | Endpoint | Reference | Mean adaptive nDCG | Mean reference nDCG | Mean difference |
|---|---|---|---:|---:|---:|
| conditional | all_observed | fixed_flow | 0.01376 | 0.01446 | -0.00069 |
| conditional | all_observed | EASE | 0.01376 | 0.32155 | -0.30779 |
| conditional | all_observed | SLIMElastic | 0.01376 | 0.31341 | -0.29964 |
| conditional | all_observed | PositiveEASE | 0.01376 | 0.28468 | -0.27092 |
| joint100 | all_observed | fixed_flow | 0.22882 | 0.22937 | -0.00055 |
| joint100 | all_observed | EASE | 0.22882 | 0.32155 | -0.09273 |
| joint100 | all_observed | SLIMElastic | 0.22882 | 0.31341 | -0.08459 |
| joint100 | all_observed | PositiveEASE | 0.22882 | 0.28468 | -0.05587 |
| joint400 | all_observed | fixed_flow | 0.26786 | 0.27201 | -0.00415 |
| joint400 | all_observed | EASE | 0.26786 | 0.32155 | -0.05369 |
| joint400 | all_observed | SLIMElastic | 0.26786 | 0.31341 | -0.04555 |
| joint400 | all_observed | PositiveEASE | 0.26786 | 0.28468 | -0.01682 |
| conditional | liked_ratings | fixed_flow | 0.01446 | 0.01529 | -0.00083 |
| conditional | liked_ratings | EASE | 0.01446 | 0.28678 | -0.27232 |
| conditional | liked_ratings | SLIMElastic | 0.01446 | 0.28019 | -0.26573 |
| conditional | liked_ratings | PositiveEASE | 0.01446 | 0.29439 | -0.27993 |
| joint100 | liked_ratings | fixed_flow | 0.20774 | 0.20685 | +0.00089 |
| joint100 | liked_ratings | EASE | 0.20774 | 0.28678 | -0.07904 |
| joint100 | liked_ratings | SLIMElastic | 0.20774 | 0.28019 | -0.07245 |
| joint100 | liked_ratings | PositiveEASE | 0.20774 | 0.29439 | -0.08665 |
| joint400 | liked_ratings | fixed_flow | 0.24957 | 0.25079 | -0.00121 |
| joint400 | liked_ratings | EASE | 0.24957 | 0.28678 | -0.03721 |
| joint400 | liked_ratings | SLIMElastic | 0.24957 | 0.28019 | -0.03062 |
| joint400 | liked_ratings | PositiveEASE | 0.24957 | 0.29439 | -0.04482 |

The means above weight the overlapping split seeds equally. No confidence interval is formed by treating these seeds as independent.

| Seed | Track | Endpoint | Reference | Paired users | Difference | Adjusted 95% family interval |
|---|---|---|---|---:|---:|---|
| 2026 | conditional | all_observed | fixed_flow | 943 | +0.00050 | [-0.00055, +0.00158] |
| 2026 | conditional | all_observed | EASE | 943 | -0.31102 | [-0.33437, -0.28804] |
| 2026 | conditional | all_observed | SLIMElastic | 943 | -0.30285 | [-0.32690, -0.27863] |
| 2026 | conditional | all_observed | PositiveEASE | 943 | -0.27581 | [-0.29725, -0.25267] |
| 2026 | joint100 | all_observed | fixed_flow | 943 | +0.00942 | [-0.00260, +0.02143] |
| 2026 | joint100 | all_observed | EASE | 943 | -0.08142 | [-0.09860, -0.06334] |
| 2026 | joint100 | all_observed | SLIMElastic | 943 | -0.07325 | [-0.09381, -0.05294] |
| 2026 | joint100 | all_observed | PositiveEASE | 943 | -0.04621 | [-0.06568, -0.02614] |
| 2026 | joint400 | all_observed | fixed_flow | 943 | +0.00391 | [-0.00864, +0.01717] |
| 2026 | joint400 | all_observed | EASE | 943 | -0.04465 | [-0.06099, -0.02815] |
| 2026 | joint400 | all_observed | SLIMElastic | 943 | -0.03647 | [-0.05456, -0.01795] |
| 2026 | joint400 | all_observed | PositiveEASE | 943 | -0.00944 | [-0.02810, +0.01022] |
| 2026 | conditional | liked_ratings | fixed_flow | 882 | +0.00026 | [-0.00111, +0.00147] |
| 2026 | conditional | liked_ratings | EASE | 882 | -0.27768 | [-0.30266, -0.25265] |
| 2026 | conditional | liked_ratings | SLIMElastic | 882 | -0.27452 | [-0.30114, -0.24952] |
| 2026 | conditional | liked_ratings | PositiveEASE | 882 | -0.28669 | [-0.31417, -0.26053] |
| 2026 | joint100 | liked_ratings | fixed_flow | 882 | +0.00860 | [-0.00359, +0.02085] |
| 2026 | joint100 | liked_ratings | EASE | 882 | -0.07449 | [-0.09588, -0.05335] |
| 2026 | joint100 | liked_ratings | SLIMElastic | 882 | -0.07133 | [-0.09680, -0.04697] |
| 2026 | joint100 | liked_ratings | PositiveEASE | 882 | -0.08351 | [-0.10518, -0.06200] |
| 2026 | joint400 | liked_ratings | fixed_flow | 882 | -0.00007 | [-0.01444, +0.01476] |
| 2026 | joint400 | liked_ratings | EASE | 882 | -0.03396 | [-0.05276, -0.01516] |
| 2026 | joint400 | liked_ratings | SLIMElastic | 882 | -0.03080 | [-0.05245, -0.00884] |
| 2026 | joint400 | liked_ratings | PositiveEASE | 882 | -0.04297 | [-0.06379, -0.02294] |
| 2027 | conditional | all_observed | fixed_flow | 943 | -0.00026 | [-0.00135, +0.00077] |
| 2027 | conditional | all_observed | EASE | 943 | -0.31399 | [-0.33931, -0.28881] |
| 2027 | conditional | all_observed | SLIMElastic | 943 | -0.30569 | [-0.33006, -0.28139] |
| 2027 | conditional | all_observed | PositiveEASE | 943 | -0.27379 | [-0.29726, -0.24983] |
| 2027 | joint100 | all_observed | fixed_flow | 943 | -0.00785 | [-0.01809, +0.00224] |
| 2027 | joint100 | all_observed | EASE | 943 | -0.09465 | [-0.11324, -0.07691] |
| 2027 | joint100 | all_observed | SLIMElastic | 943 | -0.08635 | [-0.10572, -0.06637] |
| 2027 | joint100 | all_observed | PositiveEASE | 943 | -0.05445 | [-0.07288, -0.03526] |
| 2027 | joint400 | all_observed | fixed_flow | 943 | -0.01174 | [-0.02482, +0.00172] |
| 2027 | joint400 | all_observed | EASE | 943 | -0.05534 | [-0.07211, -0.03900] |
| 2027 | joint400 | all_observed | SLIMElastic | 943 | -0.04704 | [-0.06543, -0.02862] |
| 2027 | joint400 | all_observed | PositiveEASE | 943 | -0.01514 | [-0.03430, +0.00309] |
| 2027 | conditional | liked_ratings | fixed_flow | 882 | -0.00072 | [-0.00221, +0.00044] |
| 2027 | conditional | liked_ratings | EASE | 882 | -0.27411 | [-0.30064, -0.24798] |
| 2027 | conditional | liked_ratings | SLIMElastic | 882 | -0.27052 | [-0.29668, -0.24331] |
| 2027 | conditional | liked_ratings | PositiveEASE | 882 | -0.28113 | [-0.30818, -0.25459] |
| 2027 | joint100 | liked_ratings | fixed_flow | 882 | -0.00335 | [-0.01332, +0.00696] |
| 2027 | joint100 | liked_ratings | EASE | 882 | -0.07887 | [-0.09938, -0.05771] |
| 2027 | joint100 | liked_ratings | SLIMElastic | 882 | -0.07527 | [-0.09987, -0.05108] |
| 2027 | joint100 | liked_ratings | PositiveEASE | 882 | -0.08589 | [-0.10741, -0.06411] |
| 2027 | joint400 | liked_ratings | fixed_flow | 882 | -0.00243 | [-0.01669, +0.01207] |
| 2027 | joint400 | liked_ratings | EASE | 882 | -0.03295 | [-0.05260, -0.01346] |
| 2027 | joint400 | liked_ratings | SLIMElastic | 882 | -0.02936 | [-0.05181, -0.00684] |
| 2027 | joint400 | liked_ratings | PositiveEASE | 882 | -0.03997 | [-0.05996, -0.02052] |
| 2028 | conditional | all_observed | fixed_flow | 943 | -0.00232 | [-0.00420, -0.00076] |
| 2028 | conditional | all_observed | EASE | 943 | -0.29834 | [-0.32381, -0.27350] |
| 2028 | conditional | all_observed | SLIMElastic | 943 | -0.29039 | [-0.31484, -0.26518] |
| 2028 | conditional | all_observed | PositiveEASE | 943 | -0.26315 | [-0.28585, -0.24068] |
| 2028 | joint100 | all_observed | fixed_flow | 943 | -0.00323 | [-0.00825, +0.00187] |
| 2028 | joint100 | all_observed | EASE | 943 | -0.10213 | [-0.12048, -0.08385] |
| 2028 | joint100 | all_observed | SLIMElastic | 943 | -0.09418 | [-0.11407, -0.07407] |
| 2028 | joint100 | all_observed | PositiveEASE | 943 | -0.06694 | [-0.08501, -0.04846] |
| 2028 | joint400 | all_observed | fixed_flow | 943 | -0.00463 | [-0.01473, +0.00524] |
| 2028 | joint400 | all_observed | EASE | 943 | -0.06108 | [-0.07791, -0.04406] |
| 2028 | joint400 | all_observed | SLIMElastic | 943 | -0.05313 | [-0.07136, -0.03574] |
| 2028 | joint400 | all_observed | PositiveEASE | 943 | -0.02589 | [-0.04388, -0.00766] |
| 2028 | conditional | liked_ratings | fixed_flow | 884 | -0.00201 | [-0.00395, -0.00024] |
| 2028 | conditional | liked_ratings | EASE | 884 | -0.26517 | [-0.29313, -0.23667] |
| 2028 | conditional | liked_ratings | SLIMElastic | 884 | -0.25216 | [-0.27884, -0.22479] |
| 2028 | conditional | liked_ratings | PositiveEASE | 884 | -0.27196 | [-0.29818, -0.24554] |
| 2028 | joint100 | liked_ratings | fixed_flow | 884 | -0.00257 | [-0.00723, +0.00172] |
| 2028 | joint100 | liked_ratings | EASE | 884 | -0.08376 | [-0.10437, -0.06359] |
| 2028 | joint100 | liked_ratings | SLIMElastic | 884 | -0.07074 | [-0.09370, -0.04952] |
| 2028 | joint100 | liked_ratings | PositiveEASE | 884 | -0.09054 | [-0.11092, -0.06994] |
| 2028 | joint400 | liked_ratings | fixed_flow | 884 | -0.00114 | [-0.01226, +0.01021] |
| 2028 | joint400 | liked_ratings | EASE | 884 | -0.04472 | [-0.06415, -0.02538] |
| 2028 | joint400 | liked_ratings | SLIMElastic | 884 | -0.03171 | [-0.05222, -0.01179] |
| 2028 | joint400 | liked_ratings | PositiveEASE | 884 | -0.05151 | [-0.07208, -0.03096] |

Unadjusted intervals, recall means, model choices, exact cohort/label denominators and input hashes are retained in JSON. No individual user IDs, recommendation lists, ratings or predictions are exported. This report reads completed metrics and sealed metadata only; it performs no test-label access, inference, refitting or selection.
