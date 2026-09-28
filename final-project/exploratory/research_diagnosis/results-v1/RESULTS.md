# Reconstruction error diagnosis

Post-final-test exploratory diagnosis on the already reused development cohort (472 users per seed). No fitting, retuning, TEST access, or fresh confirmation occurred. Positives are every recorded validation item, regardless of rating category.

Full-catalog ranking excludes padding and each user's original TRAIN items. Ties follow catalog column order. All means across seeds below give each split equal weight; the three splits reuse the same dataset and are not independent replications.

| Model | nDCG@10 | Recall@10 (macro) | Recall@100 (macro) | Positive mean rank | Head slots |
|---|---:|---:|---:|---:|---:|
| binary_original_grid | 0.261827 | 0.2408 | 0.7144 | 141.2 | 98.83% |
| binary_expanded | 0.261688 | 0.2417 | 0.7167 | 138.6 | 98.95% |
| categorical | 0.263154 | 0.2438 | 0.7162 | 139.0 | 98.91% |
| shuffled_categories | 0.261388 | 0.2406 | 0.7168 | 138.5 | 98.95% |
| slim | 0.257954 | 0.2402 | 0.6870 | 145.8 | 96.19% |
| hybrid_baseline2 | 0.265593 | 0.2465 | 0.7194 | 134.4 | 98.69% |
| hybrid_real3 | 0.265666 | 0.2462 | 0.7195 | 134.6 | 98.69% |
| hybrid_shuffled3 | 0.265644 | 0.2459 | 0.7197 | 134.5 | 98.70% |

| First → second | Eligible score Pearson | Top-10 overlap | Positive mean rank improvement | Oracle whole-list gain |
|---|---:|---:|---:|---:|
| binary_expanded → categorical | 0.99932 | 97.81% | -0.42 | 0.00263 |
| binary_expanded → shuffled_categories | 0.99981 | 99.11% | 0.14 | 0.00117 |
| binary_expanded → slim | 0.87255 | 66.05% | -7.17 | 0.04070 |
| categorical → slim | 0.87157 | 65.90% | -6.75 | 0.03990 |
| hybrid_baseline2 → hybrid_real3 | 0.99981 | 98.37% | -0.16 | 0.00243 |

Pearson is computed separately over each user's eligible scores after subtracting their mean; it is invariant to positive score scaling. Eligible Spearman uses average ranks for score ties and is stored separately in aggregates.json. Oracle whole-list selection uses development truth and cannot be deployed as measured; its gain does not show that a learned selector can recover it. The union-of-top-10 statistic in JSON has up to 20 slots and must not be compared with fixed-budget recall@10.

| Seed | Model | Head positive pairs | Head recall@10 (pooled) | Tail positive pairs | Tail recall@10 (pooled) | Tail share of misses |
|---|---|---:|---:|---:|---:|---:|
| 2026 | binary_expanded | 3035 | 0.2629 | 1620 | 0.0037 | 41.91% |
| 2026 | categorical | 3035 | 0.2682 | 1620 | 0.0037 | 42.09% |
| 2026 | slim | 3035 | 0.2590 | 1620 | 0.0099 | 41.63% |
| 2027 | binary_expanded | 3059 | 0.2681 | 1525 | 0.0052 | 40.39% |
| 2027 | categorical | 3059 | 0.2703 | 1525 | 0.0066 | 40.43% |
| 2027 | slim | 3059 | 0.2589 | 1525 | 0.0118 | 39.93% |
| 2028 | binary_expanded | 3031 | 0.2745 | 1589 | 0.0069 | 41.78% |
| 2028 | categorical | 3031 | 0.2745 | 1589 | 0.0069 | 41.78% |
| 2028 | slim | 3031 | 0.2580 | 1589 | 0.0201 | 40.91% |

Head is the top ceil(20% of the catalog) by TRAIN interaction count, with lexicographic item-token ties. Activity groups use TRAIN history terciles over all 943 users, with ties assigned upward. Group statistics include positive-pair and user denominators; conditional macro recall excludes users with no positives in that item group. A high fraction of all errors is not evidence of disproportionate error unless compared with that group's fraction of all positive pairs.

The JSON also contains activity × popularity strata, per-model positive rank percentiles, MRR, top-10 hit intersections, and per-user nDCG win/loss counts. Different metrics need not move together: first-hit MRR can worsen while multi-positive nDCG improves. These random per-user splits do not establish changing taste, chronological generalization, or a causal benefit from an architectural feature.
