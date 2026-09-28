# Per-user chronology sensitivity

Validation only. Both protocols use the same four experts, tuning grids, user cohorts and hybrid code.
Chronological ordering is within each user; it does not prevent global-timeline leakage. Held-out labels change, so absolute differences measure split sensitivity.

MovieLens timestamps record rating entries, which may be submitted long after viewing or backfilled in batches. This experiment orders rating activity and cannot establish viewing order or changing taste. [Harper and Konstan (2015), section 3.2](https://files.grouplens.org/papers/harper-tiis2015.pdf).

| Model family | Random NDCG@10 | Chronological NDCG@10 |
|---|---:|---:|
| Random | 0.0069 | 0.0069 |
| ExactPop | 0.1292 | 0.0527 |
| EASE | 0.2559 | 0.1199 |
| ItemKNN | 0.2344 | 0.1142 |
| best_expert | 0.2559 | 0.1199 |
| static | 0.2602 | 0.1245 |
| constrained | 0.2348 | 0.1077 |
| item | 0.2602 | 0.1254 |
| context | 0.2616 | 0.1278 |
| user | 0.2619 | 0.1286 |
| rrf | 0.2447 | 0.1179 |
| group_switch | 0.2559 | 0.1185 |

Same cohort counts: 471 meta-fit, 236 development selection, 236 independent policy calibration. Test remains reserved.
EASE selects regularization 250 with random ordering and 50 with chronological ordering; ItemKNN selects 200 neighbors in both.
Both controls exclude BPR, so hybrid values need not match the full primary study.
These are selected development results on one seed. They do not establish final-test gains, causality, or deployment realism.
The temporal training runs used the fresh offline environment documented in ../environment-check. The random control reused its original train-fitted scores.
The first random-control fit was interrupted under concurrent numerical-library contention. Its unchanged retry finished in 16.05 seconds with one numerical thread; partial artifacts remain in ignored runs/.

User-conditioned fusion improves selected development NDCG by 2.33% relative to EASE under random ordering and 7.25% under chronological ordering. Both descriptive paired 95% intervals include zero (random [-0.00046, 0.01264]; chronological [-0.00170, 0.01909]); this does not establish a reliable improvement.

Chronological validation targets have mean training frequency 106.35 versus 134.42 under random ordering. The fraction absent from training rises from 0.11% to 0.34%. These describe changes in task difficulty, not a causal explanation of the metric gap.
