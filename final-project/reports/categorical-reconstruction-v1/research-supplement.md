# REVIEW DRAFT: Task 2 supplement: categorical reconstruction

Post-test exploratory research; reused development only

Group 24; supplied names: Vlad George Iftode

One ridge model reconstructs binary record presence from binary history and five TRAIN-centered rating channels; all six target-item channels are excluded. Binary-only is EASE. Hybrids use calibrated sum-to-one ridge; weights may be negative.

Equal means of three overlapping splits; 471 meta-fit / 472 development users each. All ratings are relevant for All; rating >=4 defines Liked (444/434/433 users). Full-catalog top-10 excludes TRAIN. References retain their original tuning; no score probabilities are claimed.

## Task 2.4/2.6: standalone and hybrid controls

| Model | All nDCG | Liked nDCG | Coverage % |
| --- | --- | --- | --- |
| Binary: original grid | 0.2618 | 0.2450 | 18.9 |
| Binary: expanded grid | 0.2617 | 0.2453 | 18.4 |
| Categorical | 0.2632 | 0.2467 | 18.5 |
| Shuffled categories | 0.2614 | 0.2452 | 18.3 |
| Hybrid: binary + SLIM | 0.2656 | 0.2502 | 19.1 |
| Hybrid + categorical | 0.2657 | 0.2505 | 19.1 |
| Hybrid + shuffled | 0.2656 | 0.2503 | 19.1 |
| Locked EASE | 0.2618 | 0.2450 | 18.9 |
| Locked SLIM | 0.2580 | 0.2427 | 24.1 |
| Locked PositiveEASE | 0.2370 | 0.2575 | 15.9 |

## Task 2.5: TRAIN activity / item popularity groups

| Group / metric | Binary | Category | Hybrid 2 | Hybrid 3 |
| --- | --- | --- | --- | --- |
| Sparse nD/div | 0.240/0.798 | 0.242/0.797 | 0.247/0.798 | 0.247/0.798 |
| Medium nD/div | 0.219/0.807 | 0.219/0.806 | 0.221/0.806 | 0.221/0.805 |
| Dense nD/div | 0.332/0.817 | 0.334/0.818 | 0.333/0.819 | 0.334/0.820 |
| Head R/exp | 0.329/0.990 | 0.332/0.989 | 0.334/0.987 | 0.334/0.987 |
| Tail R/exp | 0.009/0.010 | 0.010/0.011 | 0.012/0.013 | 0.012/0.013 |

nD/div: nDCG and genre-Jaccard diversity. R/exp: conditional recall and exposure share. TRAIN activity terciles; head = top 20% by frequency. User counts (seed order): sparse 169/155/158; medium 155/166/167; dense 148/151/147. Item-group positive users: head 462/462/467; tail 335/352/343.

## Discussion

The standalone nDCG gain over expanded binary is small (+0.0015); real minus shuffled categories is +0.0018. MRR declines from 0.4318 to 0.4303. The real-category hybrid gain is negligible (+0.00007); shuffled augmentation gives +0.00005. Real categories do not consistently beat the shuffled hybrid across splits. Real categories select binary fallback in 1/3 splits; an added duplicate expert can change regularization. The 45-choice categorical searches have more tuning opportunities than the nine-choice binary search. Self-item exclusion prevents copying the target; itemwise shuffling preserves item rating histograms. Group results describe accuracy and exposure, not causal fairness. All choices were sealed before this evaluation, but validation reuse after earlier TEST exposure precludes fresh confirmation.

Independent implementation; EASE/FEASE and categorical-feedback prior art are documented in RELATED_WORK.md. Audit and source fingerprints accompany the report.

## Per-seed details and group denominators

### Seed 2026

| Model | Features | Penalty | Ratio | All nDCG | Liked nDCG |
|---|---|---:|---:|---:|---:|
| Binary: original grid | binary | 250.0 | None | 0.2596 | 0.2404 |
| Binary: expanded grid | binary | 300.0 | None | 0.2590 | 0.2397 |
| Categorical | categorical | 300.0 | 10.0 | 0.2608 | 0.2422 |
| Shuffled categories | shuffled_categories | 300.0 | 10.0 | 0.2581 | 0.2395 |
| Hybrid: binary + SLIM | calibrated_weighted_hybrid | 0.001 | None | 0.2614 | 0.2442 |
| Hybrid + categorical | calibrated_weighted_hybrid | 0.01 | None | 0.2621 | 0.2443 |
| Hybrid + shuffled | calibrated_weighted_hybrid | 1.0 | None | 0.2627 | 0.2445 |

| Model / group | Users / positive users | nDCG or recall | Diversity or exposure |
|---|---:|---:|---:|
| Binary: original grid / dense | 148 | 0.3142 | 0.8121 |
| Binary: original grid / medium | 155 | 0.2163 | 0.8087 |
| Binary: original grid / sparse | 169 | 0.2514 | 0.7969 |
| Binary: original grid / head | 462 | 0.3340 | 0.9877 |
| Binary: original grid / tail | 335 | 0.0078 | 0.0123 |
| Binary: expanded grid / dense | 148 | 0.3151 | 0.8123 |
| Binary: expanded grid / medium | 155 | 0.2151 | 0.8092 |
| Binary: expanded grid / sparse | 169 | 0.2502 | 0.7972 |
| Binary: expanded grid / head | 462 | 0.3330 | 0.9892 |
| Binary: expanded grid / tail | 335 | 0.0077 | 0.0108 |
| Categorical / dense | 148 | 0.3194 | 0.8119 |
| Categorical / medium | 155 | 0.2165 | 0.8094 |
| Categorical / sparse | 169 | 0.2501 | 0.7959 |
| Categorical / head | 462 | 0.3371 | 0.9890 |
| Categorical / tail | 335 | 0.0077 | 0.0110 |
| Shuffled categories / dense | 148 | 0.3152 | 0.8122 |
| Shuffled categories / medium | 155 | 0.2177 | 0.8118 |
| Shuffled categories / sparse | 169 | 0.2452 | 0.7978 |
| Shuffled categories / head | 462 | 0.3296 | 0.9890 |
| Shuffled categories / tail | 335 | 0.0077 | 0.0110 |
| Hybrid: binary + SLIM / dense | 148 | 0.3194 | 0.8118 |
| Hybrid: binary + SLIM / medium | 155 | 0.2169 | 0.8095 |
| Hybrid: binary + SLIM / sparse | 169 | 0.2516 | 0.7937 |
| Hybrid: binary + SLIM / head | 462 | 0.3347 | 0.9873 |
| Hybrid: binary + SLIM / tail | 335 | 0.0077 | 0.0127 |
| Hybrid + categorical / dense | 148 | 0.3205 | 0.8116 |
| Hybrid + categorical / medium | 155 | 0.2167 | 0.8095 |
| Hybrid + categorical / sparse | 169 | 0.2526 | 0.7935 |
| Hybrid + categorical / head | 462 | 0.3349 | 0.9873 |
| Hybrid + categorical / tail | 335 | 0.0077 | 0.0127 |
| Hybrid + shuffled / dense | 148 | 0.3222 | 0.8123 |
| Hybrid + shuffled / medium | 155 | 0.2164 | 0.8104 |
| Hybrid + shuffled / sparse | 169 | 0.2531 | 0.7941 |
| Hybrid + shuffled / head | 462 | 0.3341 | 0.9869 |
| Hybrid + shuffled / tail | 335 | 0.0082 | 0.0131 |

### Seed 2027

| Model | Features | Penalty | Ratio | All nDCG | Liked nDCG |
|---|---|---:|---:|---:|---:|
| Binary: original grid | binary | 250.0 | None | 0.2620 | 0.2421 |
| Binary: expanded grid | binary | 300.0 | None | 0.2622 | 0.2442 |
| Categorical | categorical | 250.0 | 10.0 | 0.2648 | 0.2460 |
| Shuffled categories | binary | 300.0 | None | 0.2622 | 0.2442 |
| Hybrid: binary + SLIM | calibrated_weighted_hybrid | 0.001 | None | 0.2658 | 0.2466 |
| Hybrid + categorical | calibrated_weighted_hybrid | 0.1 | None | 0.2667 | 0.2488 |
| Hybrid + shuffled | calibrated_weighted_hybrid | 1.0 | None | 0.2660 | 0.2478 |

| Model / group | Users / positive users | nDCG or recall | Diversity or exposure |
|---|---:|---:|---:|
| Binary: original grid / dense | 151 | 0.3413 | 0.8225 |
| Binary: original grid / medium | 166 | 0.2262 | 0.8091 |
| Binary: original grid / sparse | 155 | 0.2232 | 0.7999 |
| Binary: original grid / head | 462 | 0.3214 | 0.9905 |
| Binary: original grid / tail | 352 | 0.0076 | 0.0095 |
| Binary: expanded grid / dense | 151 | 0.3406 | 0.8218 |
| Binary: expanded grid / medium | 166 | 0.2269 | 0.8091 |
| Binary: expanded grid / sparse | 155 | 0.2236 | 0.7989 |
| Binary: expanded grid / head | 462 | 0.3246 | 0.9922 |
| Binary: expanded grid / tail | 352 | 0.0076 | 0.0078 |
| Categorical / dense | 151 | 0.3430 | 0.8223 |
| Categorical / medium | 166 | 0.2266 | 0.8075 |
| Categorical / sparse | 155 | 0.2294 | 0.7987 |
| Categorical / head | 462 | 0.3284 | 0.9909 |
| Categorical / tail | 352 | 0.0105 | 0.0091 |
| Shuffled categories / dense | 151 | 0.3406 | 0.8218 |
| Shuffled categories / medium | 166 | 0.2269 | 0.8091 |
| Shuffled categories / sparse | 155 | 0.2236 | 0.7989 |
| Shuffled categories / head | 462 | 0.3246 | 0.9922 |
| Shuffled categories / tail | 352 | 0.0076 | 0.0078 |
| Hybrid: binary + SLIM / dense | 151 | 0.3446 | 0.8232 |
| Hybrid: binary + SLIM / medium | 166 | 0.2271 | 0.8068 |
| Hybrid: binary + SLIM / sparse | 155 | 0.2304 | 0.7993 |
| Hybrid: binary + SLIM / head | 462 | 0.3327 | 0.9892 |
| Hybrid: binary + SLIM / tail | 352 | 0.0112 | 0.0108 |
| Hybrid + categorical / dense | 151 | 0.3422 | 0.8243 |
| Hybrid + categorical / medium | 166 | 0.2290 | 0.8066 |
| Hybrid + categorical / sparse | 155 | 0.2335 | 0.7982 |
| Hybrid + categorical / head | 462 | 0.3354 | 0.9883 |
| Hybrid + categorical / tail | 352 | 0.0112 | 0.0117 |
| Hybrid + shuffled / dense | 151 | 0.3436 | 0.8244 |
| Hybrid + shuffled / medium | 166 | 0.2266 | 0.8057 |
| Hybrid + shuffled / sparse | 155 | 0.2327 | 0.7989 |
| Hybrid + shuffled / head | 462 | 0.3327 | 0.9890 |
| Hybrid + shuffled / tail | 352 | 0.0112 | 0.0110 |

### Seed 2028

| Model | Features | Penalty | Ratio | All nDCG | Liked nDCG |
|---|---|---:|---:|---:|---:|
| Binary: original grid | binary | 250.0 | None | 0.2639 | 0.2525 |
| Binary: expanded grid | binary | 300.0 | None | 0.2639 | 0.2520 |
| Categorical | binary | 300.0 | None | 0.2639 | 0.2520 |
| Shuffled categories | binary | 300.0 | None | 0.2639 | 0.2520 |
| Hybrid: binary + SLIM | calibrated_weighted_hybrid | 0.001 | None | 0.2696 | 0.2596 |
| Hybrid + categorical | calibrated_weighted_hybrid | 0.001 | None | 0.2682 | 0.2586 |
| Hybrid + shuffled | calibrated_weighted_hybrid | 0.001 | None | 0.2682 | 0.2586 |

| Model / group | Users / positive users | nDCG or recall | Diversity or exposure |
|---|---:|---:|---:|
| Binary: original grid / dense | 147 | 0.3405 | 0.8185 |
| Binary: original grid / medium | 167 | 0.2144 | 0.8019 |
| Binary: original grid / sparse | 158 | 0.2449 | 0.7978 |
| Binary: original grid / head | 467 | 0.3286 | 0.9869 |
| Binary: original grid / tail | 343 | 0.0108 | 0.0131 |
| Binary: expanded grid / dense | 147 | 0.3398 | 0.8183 |
| Binary: expanded grid / medium | 167 | 0.2135 | 0.8026 |
| Binary: expanded grid / sparse | 158 | 0.2464 | 0.7979 |
| Binary: expanded grid / head | 467 | 0.3292 | 0.9873 |
| Binary: expanded grid / tail | 343 | 0.0108 | 0.0127 |
| Categorical / dense | 147 | 0.3398 | 0.8183 |
| Categorical / medium | 167 | 0.2135 | 0.8026 |
| Categorical / sparse | 158 | 0.2464 | 0.7979 |
| Categorical / head | 467 | 0.3292 | 0.9873 |
| Categorical / tail | 343 | 0.0108 | 0.0127 |
| Shuffled categories / dense | 147 | 0.3398 | 0.8183 |
| Shuffled categories / medium | 167 | 0.2135 | 0.8026 |
| Shuffled categories / sparse | 158 | 0.2464 | 0.7979 |
| Shuffled categories / head | 467 | 0.3292 | 0.9873 |
| Shuffled categories / tail | 343 | 0.0108 | 0.0127 |
| Hybrid: binary + SLIM / dense | 147 | 0.3354 | 0.8229 |
| Hybrid: binary + SLIM / medium | 167 | 0.2203 | 0.8002 |
| Hybrid: binary + SLIM / sparse | 158 | 0.2604 | 0.8009 |
| Hybrid: binary + SLIM / head | 467 | 0.3335 | 0.9843 |
| Hybrid: binary + SLIM / tail | 343 | 0.0182 | 0.0157 |
| Hybrid + categorical / dense | 147 | 0.3382 | 0.8228 |
| Hybrid + categorical / medium | 167 | 0.2185 | 0.7998 |
| Hybrid + categorical / sparse | 158 | 0.2556 | 0.8015 |
| Hybrid + categorical / head | 467 | 0.3319 | 0.9852 |
| Hybrid + categorical / tail | 343 | 0.0170 | 0.0148 |
| Hybrid + shuffled / dense | 147 | 0.3382 | 0.8228 |
| Hybrid + shuffled / medium | 167 | 0.2185 | 0.7998 |
| Hybrid + shuffled / sparse | 158 | 0.2556 | 0.8015 |
| Hybrid + shuffled / head | 467 | 0.3319 | 0.9852 |
| Hybrid + shuffled / tail | 343 | 0.0170 | 0.0148 |

Full selection diagnostics, calibrated coefficients, all metric definitions and paired descriptive differences remain in the hash-verified aggregates.json; no user-level records are included.
