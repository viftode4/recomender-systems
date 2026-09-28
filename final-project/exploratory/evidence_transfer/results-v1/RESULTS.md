# Shared evidence interpreter: reused-development exploration

The original MovieLens100K TEST was previously examined. No TEST is opened here. Every seed was sealed before new development metrics. This is not fresh confirmation.

| Seed | Model | All nDCG@10 | Liked nDCG@10 | Selected epoch |
|---|---|---:|---:|---:|
| 2026 | analytic_donor | 0.212247 | 0.199376 | fixed |
| 2026 | full_pattern | 0.249838 | 0.233438 | 100 |
| 2026 | full_zero_pattern | 0.225849 | 0.214850 | 100 |
| 2026 | marginal_only | 0.118400 | 0.107452 | 10 |
| 2026 | no_pattern | 0.238054 | 0.224486 | 100 |
| 2026 | locked_binary | 0.259004 | 0.239702 | fixed |
| 2027 | analytic_donor | 0.216529 | 0.197272 | fixed |
| 2027 | full_pattern | 0.246021 | 0.224913 | 100 |
| 2027 | full_zero_pattern | 0.200830 | 0.188224 | 100 |
| 2027 | marginal_only | 0.121113 | 0.112453 | 10 |
| 2027 | no_pattern | 0.243901 | 0.224837 | 100 |
| 2027 | locked_binary | 0.262180 | 0.244211 | fixed |
| 2028 | analytic_donor | 0.217101 | 0.198943 | fixed |
| 2028 | full_pattern | 0.239709 | 0.225220 | 100 |
| 2028 | full_zero_pattern | 0.202493 | 0.193194 | 100 |
| 2028 | marginal_only | 0.125443 | 0.117989 | 0 |
| 2028 | no_pattern | 0.234081 | 0.219844 | 100 |
| 2028 | locked_binary | 0.263879 | 0.251989 | fixed |

Equal-seed means are descriptive; overlapping reused splits are not independent datasets.

| Model | Mean all nDCG@10 | Mean liked nDCG@10 |
|---|---:|---:|
| full_pattern | 0.245189 | 0.227857 |
| no_pattern | 0.238678 | 0.223056 |
| marginal_only | 0.121652 | 0.112631 |
| analytic_donor | 0.215292 | 0.198530 |
| full_zero_pattern | 0.209724 | 0.198756 |
| locked_binary | 0.261688 | 0.245301 |

All three trained controls have equal allocated capacity, initialization, episodes, updates and checkpoint opportunities. The full_zero_pattern row is an inference feature intervention using the selected full_pattern weights, with no retraining; it changes the input distribution and is distinct from no_pattern. The analytic donor control is unfitted. No model receives query-user donor evidence. Aggregate JSON includes explicit cohorts, TRAIN-defined user/item groups and paired descriptive differences. No population significance or novelty-priority claim follows from this experiment.
