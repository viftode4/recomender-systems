# Independent reconstruction results audit

All checks passed on the completed, globally sealed study. No TEST split or final-result artifact was opened.

243 distinct candidate rows, all selection maxima/ties, selected predictions, original-grid EASE compatibility and three hybrid arms were verified.
Development nDCG@10, Recall@10 and Precision@10 were recomputed with independent ranking/metric code, including per-user agreement.

| Seed | Selected configurations plus EASE grid replayed | Maximum metric error |
|---|---:|---:|
| 2026 | 6 | 3.61e-16 |
| 2027 | 5 | 2.78e-16 |
| 2028 | 4 | 2.78e-16 |

Hybrid verification independently reconstructs sampled cells, user-wise score normalization, monotone affine calibration, sum-one ridge KKT systems, penalty selection and final score assembly.

Full score/ranking tolerances, cohort denominators and hashes are in audit.json. This verifies exploratory reused-development results; it does not create a new assessment sample.
