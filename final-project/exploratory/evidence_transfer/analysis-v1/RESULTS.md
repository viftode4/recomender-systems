# Declared pilot comparison

Reused development data; no fresh confirmation.

| Model | nDCG@10 | Recall@10 | MRR@10 | Coverage@10 | Tail hits | Tail slots |
|---|---:|---:|---:|---:|---:|---:|
| full_pattern | 0.245189 | 0.226542 | 0.420586 | 0.189655 | 41/4734 | 2.394% |
| no_pattern | 0.238678 | 0.221985 | 0.409444 | 0.156956 | 30/4734 | 2.027% |
| marginal_only | 0.121652 | 0.116221 | 0.227502 | 0.028141 | 0/4734 | 0.000% |
| analytic_donor | 0.215292 | 0.199671 | 0.373497 | 0.067776 | 0/4734 | 0.000% |
| full_zero_pattern | 0.209724 | 0.202310 | 0.355393 | 0.268133 | 41/4734 | 13.602% |
| locked_binary | 0.261688 | 0.241746 | 0.431761 | 0.183512 | 25/4734 | 1.045% |

Full-model relative nDCG difference versus locked binary: -6.305%.
Full-model absolute nDCG difference versus no-pattern: +0.006511.
Declared substantial-pilot target met: **False**.

Tail pairs are pooled over overlapping splits, not independent observations. Tail slots are unweighted recommendation positions. The full_zero_pattern row changes inputs to the selected full model without retraining. Full versus no_pattern changes two inputs jointly and does not identify their individual effects. A fixed training budget does not establish convergence.
