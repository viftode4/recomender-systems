# Locked-reference comparison for the field models

Every row uses the field study's same 472 development users, original catalog, and full observed-TRAIN candidate mask. Liked metrics exclude only users without a held-out rating >=4; denominators are reported below. These are exploratory comparisons on reused validation users, with previously locked parameters and checkpoints. No new tuning or test access.

EASE and SLIM were selected on meta-fit all-observed nDCG@10. PositiveEASE uses its recorded selection objective below. Conditional fields select categorical cross-entropy and rank by P(rating>=4). Joint fields select joint recorded-event likelihood; their all-observed endpoint ranks logsumexp of five categories, and their liked endpoint ranks logsumexp of categories4/5. Training objectives and tuning budgets differ, so this is a practical locked-reference comparison, not an isolated architecture ablation.

## all_observed nDCG@10

| Model | 2026 | 2027 | 2028 | Mean |
|---|---:|---:|---:|---:|
| EASE | 0.2596 | 0.2620 | 0.2639 | 0.2618 |
| SLIMElastic | 0.2546 | 0.2607 | 0.2585 | 0.2580 |
| PositiveEASE | 0.2302 | 0.2427 | 0.2380 | 0.2370 |
| Conditional-adaptive | 0.0116 | 0.0068 | 0.0209 | 0.0131 |
| Conditional-fixed_flow | 0.0110 | 0.0070 | 0.0231 | 0.0137 |
| Conditional-hard_clamp | 0.0127 | 0.0064 | 0.0182 | 0.0124 |
| Joint-adaptive | 0.2114 | 0.1946 | 0.1861 | 0.1974 |
| Joint-fixed_flow | 0.1952 | 0.1949 | 0.1894 | 0.1932 |

## all_observed recommendation frequency

Every frequency row uses all 472 development users and 4720 recommendation slots, including users without a held-out like.

| Model | Seed | Mean TRAIN count | Median TRAIN count | Zero-count slots | Head slots |
|---|---|---:|---:|---:|---:|
| EASE | 2026 | 244.51 | 232.0 | 0.00% | 98.77% |
| EASE | 2027 | 248.13 | 239.0 | 0.00% | 99.05% |
| EASE | 2028 | 247.57 | 236.0 | 0.00% | 98.69% |
| SLIMElastic | 2026 | 228.89 | 220.0 | 0.00% | 96.55% |
| SLIMElastic | 2027 | 232.18 | 220.0 | 0.00% | 96.63% |
| SLIMElastic | 2028 | 230.13 | 214.0 | 0.00% | 95.40% |
| PositiveEASE | 2026 | 245.58 | 234.0 | 0.00% | 99.07% |
| PositiveEASE | 2027 | 247.69 | 238.0 | 0.00% | 99.13% |
| PositiveEASE | 2028 | 244.49 | 232.0 | 0.00% | 99.30% |
| Conditional-adaptive | 2026 | 72.87 | 48.0 | 0.00% | 39.89% |
| Conditional-adaptive | 2027 | 37.17 | 10.0 | 0.00% | 13.69% |
| Conditional-adaptive | 2028 | 94.20 | 93.0 | 0.00% | 58.73% |
| Conditional-fixed_flow | 2026 | 68.90 | 38.0 | 0.00% | 38.03% |
| Conditional-fixed_flow | 2027 | 37.03 | 10.0 | 0.00% | 13.16% |
| Conditional-fixed_flow | 2028 | 104.34 | 98.0 | 0.00% | 63.07% |
| Conditional-hard_clamp | 2026 | 75.69 | 59.0 | 0.00% | 42.35% |
| Conditional-hard_clamp | 2027 | 45.10 | 17.0 | 0.00% | 18.56% |
| Conditional-hard_clamp | 2028 | 93.47 | 98.0 | 0.00% | 61.00% |
| Joint-adaptive | 2026 | 274.27 | 258.0 | 0.00% | 99.19% |
| Joint-adaptive | 2027 | 282.89 | 269.0 | 0.00% | 99.64% |
| Joint-adaptive | 2028 | 286.62 | 278.0 | 0.00% | 99.94% |
| Joint-fixed_flow | 2026 | 281.00 | 267.0 | 0.00% | 99.58% |
| Joint-fixed_flow | 2027 | 285.61 | 280.0 | 0.00% | 99.83% |
| Joint-fixed_flow | 2028 | 288.10 | 278.0 | 0.00% | 99.96% |

## liked_ratings nDCG@10

| Model | 2026 | 2027 | 2028 | Mean |
|---|---:|---:|---:|---:|
| EASE | 0.2404 | 0.2421 | 0.2525 | 0.2450 |
| SLIMElastic | 0.2358 | 0.2416 | 0.2508 | 0.2427 |
| PositiveEASE | 0.2482 | 0.2678 | 0.2564 | 0.2575 |
| Conditional-adaptive | 0.0117 | 0.0067 | 0.0224 | 0.0136 |
| Conditional-fixed_flow | 0.0114 | 0.0069 | 0.0240 | 0.0141 |
| Conditional-hard_clamp | 0.0122 | 0.0069 | 0.0199 | 0.0130 |
| Joint-adaptive | 0.2010 | 0.1781 | 0.1742 | 0.1844 |
| Joint-fixed_flow | 0.1863 | 0.1800 | 0.1742 | 0.1801 |

## liked_ratings recommendation frequency

Every frequency row uses all 472 development users and 4720 recommendation slots, including users without a held-out like.

| Model | Seed | Mean TRAIN count | Median TRAIN count | Zero-count slots | Head slots |
|---|---|---:|---:|---:|---:|
| EASE | 2026 | 244.51 | 232.0 | 0.00% | 98.77% |
| EASE | 2027 | 248.13 | 239.0 | 0.00% | 99.05% |
| EASE | 2028 | 247.57 | 236.0 | 0.00% | 98.69% |
| SLIMElastic | 2026 | 228.89 | 220.0 | 0.00% | 96.55% |
| SLIMElastic | 2027 | 232.18 | 220.0 | 0.00% | 96.63% |
| SLIMElastic | 2028 | 230.13 | 214.0 | 0.00% | 95.40% |
| PositiveEASE | 2026 | 245.58 | 234.0 | 0.00% | 99.07% |
| PositiveEASE | 2027 | 247.69 | 238.0 | 0.00% | 99.13% |
| PositiveEASE | 2028 | 244.49 | 232.0 | 0.00% | 99.30% |
| Conditional-adaptive | 2026 | 72.87 | 48.0 | 0.00% | 39.89% |
| Conditional-adaptive | 2027 | 37.17 | 10.0 | 0.00% | 13.69% |
| Conditional-adaptive | 2028 | 94.20 | 93.0 | 0.00% | 58.73% |
| Conditional-fixed_flow | 2026 | 68.90 | 38.0 | 0.00% | 38.03% |
| Conditional-fixed_flow | 2027 | 37.03 | 10.0 | 0.00% | 13.16% |
| Conditional-fixed_flow | 2028 | 104.34 | 98.0 | 0.00% | 63.07% |
| Conditional-hard_clamp | 2026 | 75.69 | 59.0 | 0.00% | 42.35% |
| Conditional-hard_clamp | 2027 | 45.10 | 17.0 | 0.00% | 18.56% |
| Conditional-hard_clamp | 2028 | 93.47 | 98.0 | 0.00% | 61.00% |
| Joint-adaptive | 2026 | 277.11 | 258.0 | 0.00% | 99.39% |
| Joint-adaptive | 2027 | 281.00 | 266.0 | 0.00% | 99.68% |
| Joint-adaptive | 2028 | 282.07 | 267.0 | 0.00% | 99.96% |
| Joint-fixed_flow | 2026 | 279.01 | 258.0 | 0.00% | 99.85% |
| Joint-fixed_flow | 2027 | 281.22 | 266.0 | 0.00% | 99.87% |
| Joint-fixed_flow | 2028 | 283.37 | 268.0 | 0.00% | 99.98% |

## Denominators and selection

| Seed | Development users | Users with validation likes | Validation observations | Validation likes | PositiveEASE selection |
|---|---:|---:|---:|---:|---|
| 2026 | 472 | 444 | 4655 | 2607 | liked_ratings, ridge 250.0 |
| 2027 | 472 | 434 | 4584 | 2626 | liked_ratings, ridge 250.0 |
| 2028 | 472 | 433 | 4620 | 2582 | liked_ratings, ridge 250.0 |

TRAIN count includes every observed training pair, irrespective of rating. The head is the top ceil(20% of the nonpadding catalog) by that count, with original item-token lexical tie breaking. Zero-count means no original training observation for that item. These are descriptive exposure diagnostics; they do not identify why a model ranks an item or establish a causal failure mechanism.

All dataset, train/validation pair, internal-ID file, ordered score-ID and field-output hashes were checked. Older EASE/SLIM manifests did not pin prediction-file hashes originally; this comparison records their current score-file hashes and verifies their locked selection records and common data/ID mappings. Newer exports also verify their pre-existing score hashes. Raw per-user metrics and recommendation lists remain in ignored runs/; this directory contains aggregate-only evidence.

Full Recall, Precision, MRR, HitRate, coverage, novelty and known-dislike measures are retained in `aggregates.json`, alongside explicit readout descriptions, selection settings and provenance. No results from the smaller 236-user main-study cohort are reused.
