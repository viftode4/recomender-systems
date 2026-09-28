# Addressed evidence: post-final-test exploratory study

Standalone operators share TRAIN data, allocated parameters, initialization and a common maximum/checkpoint/stopping policy. Actual training lengths can differ.
Original meta-fit users select checkpoints; original disjoint development users describe results. Both sets are reused.
The dataset TEST was evaluated before this research direction. This runner never opens TEST or final-result artifacts. These are not fresh held-out results.

All nDCG values below use the reused development cohort: all-observed uses the five-category item-event score; liked uses the liked-record score and only users with at least one observed liked validation rating.

| Seed | Variant | Joint NLL ↓ | All nDCG@10 | Liked nDCG@10 | Selected epoch | Trained epochs | Stop reason |
|---|---|---:|---:|---:|---:|---:|---|
| 2026 | additive | 7.834011 | 0.184999 | 0.189084 | 100 | 300 | predeclared_patience |
| 2026 | pair | 8.674828 | 0.185861 | 0.184004 | 10 | 220 | predeclared_patience |
| 2027 | additive | 7.768986 | 0.184797 | 0.189263 | 100 | 300 | predeclared_patience |
| 2027 | pair | 8.610124 | 0.195020 | 0.198191 | 10 | 220 | predeclared_patience |
| 2028 | additive | 7.826633 | 0.186527 | 0.201658 | 100 | 300 | predeclared_patience |
| 2028 | pair | 8.652648 | 0.188609 | 0.198824 | 10 | 220 | predeclared_patience |

Descriptive equal-seed means follow. Overlapping splits are not independent datasets.

| Variant | Mean joint NLL ↓ | Mean all nDCG@10 | Mean liked nDCG@10 |
|---|---:|---:|---:|
| additive | 7.809877 | 0.185441 | 0.193335 |
| pair | 8.645867 | 0.189830 | 0.193673 |

Primary mechanism comparison: pair minus additive macro joint NLL; negative favors pair.

| Seed | Paired NLL difference | Fraction of users with lower pair NLL |
|---|---:|---:|
| 2026 | +0.840818 | 0.072034 |
| 2027 | +0.841137 | 0.065678 |
| 2028 | +0.826015 | 0.052966 |

Descriptive equal-seed mean paired NLL difference: +0.835990.
Reaching the maximum is budget-limited; patience stopping does not prove a global optimum. Learned contribution diagnostics are descriptive, not evidence of predictive benefit.
