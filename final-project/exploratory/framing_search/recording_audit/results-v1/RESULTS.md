# TRAIN recording-bundle audit

Descriptive evidence only; no model fit, ranking evaluation, or held-out value access.

Original seed-2026 TRAIN contains 80,808 records from 943 users, forming 44,316 exact-time groups.

| Minimum group size | Users / all users | Records / all TRAIN records | Groups / all groups |
|---|---:|---:|---:|
| 2 | 942/943 (99.89%) | 56,682/80,808 (70.14%) | 20,190/44,316 (45.56%) |
| 5 | 593/943 (62.88%) | 10,249/80,808 (12.68%) | 1,735/44,316 (3.92%) |
| 10 | 20/943 (2.12%) | 230/80,808 (0.28%) | 23/44,316 (0.05%) |

| Quantile | All group sizes | Tied group sizes | Largest group / user's TRAIN records |
|---|---:|---:|---:|
| 0 | 1 | 2 | 0.91% |
| 0.25 | 1 | 2 | 4.64% |
| 0.5 | 1 | 2 | 9.26% |
| 0.75 | 2 | 3 | 16.00% |
| 0.9 | 3 | 4 | 22.73% |
| 0.95 | 4 | 5 | 29.11% |
| 0.99 | 6 | 8 | 43.38% |
| 1 | 10 | 10 | 56.25% |

| Maximum adjacent gap | Including tied rows | Positive gaps between distinct times |
|---|---:|---:|
| 1s | 39,596/79,865 (49.58%) | 3,104/43,373 (7.16%) |
| 60s | 65,990/79,865 (82.63%) | 29,498/43,373 (68.01%) |
| 300s | 76,653/79,865 (95.98%) | 40,161/43,373 (92.59%) |
| 1800s | 78,134/79,865 (97.83%) | 41,642/43,373 (96.01%) |

Exactly zero: 36,492/79,865 adjacent event gaps (45.69%).

Genre coherence uses 66,203 unordered same-user tied-time pairs across 942/943 users. The comparison preserves each user's movies/ratings and exact timestamp multiplicities in 100 deterministic shuffles.

| Genre Jaccard statistic | Observed | Shuffle mean | Difference | Shuffle 2.5–97.5 percentiles |
|---|---:|---:|---:|---:|
| pair_weighted | 0.234236 | 0.184155 | +0.050081 | 0.182131–0.185937 |
| macro_user | 0.231633 | 0.193739 | +0.037894 | 0.191140–0.197230 |

These are descriptive shuffle comparisons, not p-values. Timestamp ties do not identify a screen, consumption time, or causal interface effect. The random TRAIN split thins the original records. Any observed genre coherence is compatible with multiple recording processes and is not evidence of improved recommendation accuracy or a novel method.
