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
| Joint-adaptive | 0.2398 | 0.2076 | 0.2183 | 0.2219 |
| Joint-fixed_flow | 0.2331 | 0.2209 | 0.2237 | 0.2259 |

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
| Joint-adaptive | 2026 | 238.19 | 234.0 | 0.00% | 95.17% |
| Joint-adaptive | 2027 | 257.42 | 247.0 | 0.00% | 97.80% |
| Joint-adaptive | 2028 | 249.67 | 241.0 | 0.00% | 96.10% |
| Joint-fixed_flow | 2026 | 247.30 | 238.0 | 0.00% | 95.87% |
| Joint-fixed_flow | 2027 | 254.81 | 245.0 | 0.00% | 97.46% |
| Joint-fixed_flow | 2028 | 248.53 | 241.0 | 0.00% | 96.14% |

## liked_ratings nDCG@10

| Model | 2026 | 2027 | 2028 | Mean |
|---|---:|---:|---:|---:|
| EASE | 0.2404 | 0.2421 | 0.2525 | 0.2450 |
| SLIMElastic | 0.2358 | 0.2416 | 0.2508 | 0.2427 |
| PositiveEASE | 0.2482 | 0.2678 | 0.2564 | 0.2575 |
| Conditional-adaptive | 0.0117 | 0.0067 | 0.0224 | 0.0136 |
| Conditional-fixed_flow | 0.0114 | 0.0069 | 0.0240 | 0.0141 |
| Conditional-hard_clamp | 0.0122 | 0.0069 | 0.0199 | 0.0130 |
| Joint-adaptive | 0.2420 | 0.2037 | 0.2122 | 0.2193 |
| Joint-fixed_flow | 0.2365 | 0.2138 | 0.2114 | 0.2206 |

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
| Joint-adaptive | 2026 | 241.44 | 236.0 | 0.00% | 96.14% |
| Joint-adaptive | 2027 | 258.26 | 247.0 | 0.00% | 97.73% |
| Joint-adaptive | 2028 | 250.91 | 241.0 | 0.00% | 96.86% |
| Joint-fixed_flow | 2026 | 248.40 | 238.0 | 0.00% | 96.17% |
| Joint-fixed_flow | 2027 | 255.23 | 245.5 | 0.00% | 97.61% |
| Joint-fixed_flow | 2028 | 250.10 | 241.0 | 0.00% | 96.74% |

## Denominators and selection

| Seed | Development users | Users with validation likes | Validation observations | Validation likes | PositiveEASE selection |
|---|---:|---:|---:|---:|---|
| 2026 | 472 | 444 | 4655 | 2607 | liked_ratings, ridge 250.0 |
| 2027 | 472 | 434 | 4584 | 2626 | liked_ratings, ridge 250.0 |
| 2028 | 472 | 433 | 4620 | 2582 | liked_ratings, ridge 250.0 |

TRAIN count includes every observed training pair, irrespective of rating. The head is the top ceil(20% of the nonpadding catalog) by that count, with original item-token lexical tie breaking. Zero-count means no original training observation for that item. These are descriptive exposure diagnostics; they do not identify why a model ranks an item or establish a causal failure mechanism.

All dataset, train/validation pair, internal-ID file, ordered score-ID and field-output hashes were checked. Older EASE/SLIM manifests did not pin prediction-file hashes originally; this comparison records their current score-file hashes and verifies their locked selection records and common data/ID mappings. Newer exports also verify their pre-existing score hashes. Raw per-user metrics and recommendation lists remain in ignored runs/; this directory contains aggregate-only evidence.

Full Recall, Precision, MRR, HitRate, coverage, novelty and known-dislike measures are retained in `aggregates.json`, alongside explicit readout descriptions, selection settings and provenance. No results from the smaller 236-user main-study cohort are reused.
