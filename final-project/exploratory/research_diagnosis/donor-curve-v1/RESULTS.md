# Other-user donor-data sensitivity

Reused-development exploratory result, following the previously evaluated original TEST. No TEST access or fresh confirmation. Every query retains its complete original TRAIN history; none of the 118 inner or 472 development users contributes a fitting row. Lambda was selected separately per donor prefix on the fixed inner query users, and all twelve choices were globally sealed before development evaluation.

| Seed | Donors | Lambda | Inner nDCG@10 | DEV nDCG@10 | DEV recall@10 | DEV recall@100 | Donor-observed items | Unsupported DEV positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 88 | 50 | 0.21024 | 0.19390 | 0.1808 | 0.6138 | 1168 | 153/4655 |
| 2026 | 176 | 100 | 0.23513 | 0.21464 | 0.2041 | 0.6533 | 1391 | 44/4655 |
| 2026 | 264 | 300 | 0.24979 | 0.23483 | 0.2208 | 0.6797 | 1441 | 29/4655 |
| 2026 | 353 | 50 | 0.25734 | 0.22538 | 0.2061 | 0.6567 | 1471 | 24/4655 |
| 2027 | 88 | 50 | 0.21745 | 0.20893 | 0.1881 | 0.6136 | 1265 | 100/4584 |
| 2027 | 176 | 100 | 0.23828 | 0.22636 | 0.2076 | 0.6681 | 1391 | 41/4584 |
| 2027 | 264 | 100 | 0.25204 | 0.23590 | 0.2174 | 0.6810 | 1541 | 17/4584 |
| 2027 | 353 | 100 | 0.25431 | 0.24295 | 0.2235 | 0.6819 | 1558 | 14/4584 |
| 2028 | 88 | 300 | 0.20125 | 0.19726 | 0.1873 | 0.6275 | 1191 | 114/4620 |
| 2028 | 176 | 100 | 0.20806 | 0.22435 | 0.2146 | 0.6605 | 1315 | 53/4620 |
| 2028 | 264 | 100 | 0.21713 | 0.23637 | 0.2272 | 0.6749 | 1404 | 33/4620 |
| 2028 | 353 | 100 | 0.22705 | 0.24510 | 0.2366 | 0.6738 | 1490 | 24/4620 |

| Donors | Equal-seed mean DEV nDCG@10 | Equal-seed mean DEV recall@10 |
|---|---:|---:|
| 88 | 0.200030 | 0.185411 |
| 176 | 0.221780 | 0.208755 |
| 264 | 0.235699 | 0.221795 |
| 353 | 0.237810 | 0.222035 |

More donor rows change both estimation and item coverage, so this experiment cannot separate those effects. The observed range ends at 353 other-user donors; its absolute metrics are not comparable to the prior 943-row full-system study. No extrapolation, new-data guarantee, metadata conclusion, or claim of an information-theoretic ceiling follows. The three splits reuse the same dataset and are not independent replications. All activity/popularity definitions are fixed from the complete original TRAIN population.
