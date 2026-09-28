# Context interactions: exploratory reused-development results

MovieLens100K TEST was already examined before this study. No TEST is opened here. All seed selections were sealed before any new development metric. This is not fresh confirmation.

Binary selects from 9 penalties. Pair-only selects from 36 pair models. Nested selects from their 45-candidate union and may choose binary. Search budgets differ. Old binary and categorical predictions are frozen references. All inputs to new models are binary TRAIN histories.

| Seed | Model | All nDCG@10 | Liked nDCG@10 | Lambda | Pair weight |
|---|---|---:|---:|---:|---:|
| 2026 | binary | 0.259004 | 0.239702 | 300 | 0.0 |
| 2026 | nested | 0.259305 | 0.240589 | 300 | 0.1 |
| 2026 | pair_only | 0.259305 | 0.240589 | 300 | 0.1 |
| 2026 | locked_binary | 0.259004 | 0.239702 | 300 | reference |
| 2026 | locked_categorical | 0.260826 | 0.242203 | 300 | reference |
| 2027 | binary | 0.262180 | 0.244211 | 300 | 0.0 |
| 2027 | nested | 0.262020 | 0.244121 | 300 | 0.01 |
| 2027 | pair_only | 0.262020 | 0.244121 | 300 | 0.01 |
| 2027 | locked_binary | 0.262180 | 0.244211 | 300 | reference |
| 2027 | locked_categorical | 0.264756 | 0.245968 | 250 | reference |
| 2028 | binary | 0.263879 | 0.251989 | 300 | 0.0 |
| 2028 | nested | 0.263879 | 0.251989 | 300 | 0.0 |
| 2028 | pair_only | 0.263656 | 0.251710 | 300 | 0.01 |
| 2028 | locked_binary | 0.263879 | 0.251989 | 300 | reference |
| 2028 | locked_categorical | 0.263879 | 0.251989 | 300 | reference |

Equal-seed means are descriptive; overlapping splits are not independent datasets.

| Model | Mean all nDCG@10 | Mean liked nDCG@10 |
|---|---:|---:|
| binary | 0.261688 | 0.245301 |
| pair_only | 0.261660 | 0.245474 |
| nested | 0.261735 | 0.245567 |
| locked_binary | 0.261688 | 0.245301 |
| locked_categorical | 0.263154 | 0.246720 |

aggregates.json includes both relevance endpoints, their explicit denominators, TRAIN-defined activity/popularity groups, exposure, diversity, and paired descriptive nDCG differences. Group diagnostics do not select models. Scores are unbounded ranking values, not probabilities. Higher-order linear recommendation has existing prior art; this experiment makes no architectural priority claim.
