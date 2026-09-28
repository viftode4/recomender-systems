# Categorical reconstruction: reused-development exploratory results

The original dataset TEST was examined before this study. These results are not fresh held-out confirmation. Every declared seed selection was sealed before any development metric was computed. All model inputs and categorical centering use TRAIN only.

The expanded binary model has 9 selection candidates. Each real/shuffled categorical family has 36 categorical candidates plus the same 9 cached binary candidates. The original-grid binary row selects from 3 of those binary fits. Thus tuning opportunities differ; nested fallback can select the binary model. The shuffle preserves per-item TRAIN rating histograms and observation identities.

| Seed | Model | All nDCG@10 | Liked nDCG@10 | Selected features | Lambda | Category ratio |
|---|---|---:|---:|---|---:|---:|
| 2026 | binary_original_grid | 0.259563 | 0.240439 | binary | 250 | — |
| 2026 | binary_expanded | 0.259004 | 0.239702 | binary | 300 | — |
| 2026 | categorical | 0.260826 | 0.242203 | categorical | 300 | 10.0 |
| 2026 | shuffled_categories | 0.258105 | 0.239454 | shuffled_categories | 300 | 10.0 |
| 2026 | hybrid_baseline2 | 0.261448 | 0.244176 | calibrated_weighted_hybrid | 0.001 | — |
| 2026 | hybrid_real3 | 0.262130 | 0.244304 | calibrated_weighted_hybrid | 0.01 | — |
| 2026 | hybrid_shuffled3 | 0.262726 | 0.244545 | calibrated_weighted_hybrid | 1 | — |
| 2027 | binary_original_grid | 0.262041 | 0.242079 | binary | 250 | — |
| 2027 | binary_expanded | 0.262180 | 0.244211 | binary | 300 | — |
| 2027 | categorical | 0.264756 | 0.245968 | categorical | 250 | 10.0 |
| 2027 | shuffled_categories | 0.262180 | 0.244211 | binary | 300 | — |
| 2027 | hybrid_baseline2 | 0.265758 | 0.246643 | calibrated_weighted_hybrid | 0.001 | — |
| 2027 | hybrid_real3 | 0.266692 | 0.248770 | calibrated_weighted_hybrid | 0.1 | — |
| 2027 | hybrid_shuffled3 | 0.266031 | 0.247807 | calibrated_weighted_hybrid | 1 | — |
| 2028 | binary_original_grid | 0.263878 | 0.252465 | binary | 250 | — |
| 2028 | binary_expanded | 0.263879 | 0.251989 | binary | 300 | — |
| 2028 | categorical | 0.263879 | 0.251989 | binary | 300 | — |
| 2028 | shuffled_categories | 0.263879 | 0.251989 | binary | 300 | — |
| 2028 | hybrid_baseline2 | 0.269574 | 0.259645 | calibrated_weighted_hybrid | 0.001 | — |
| 2028 | hybrid_real3 | 0.268176 | 0.258568 | calibrated_weighted_hybrid | 0.001 | — |
| 2028 | hybrid_shuffled3 | 0.268176 | 0.258568 | calibrated_weighted_hybrid | 0.001 | — |

Equal-seed means are descriptive; overlapping splits are not independent datasets.

| Model | Mean all nDCG@10 | Mean liked nDCG@10 |
|---|---:|---:|
| binary_original_grid | 0.261827 | 0.244994 |
| binary_expanded | 0.261688 | 0.245301 |
| categorical | 0.263154 | 0.246720 |
| shuffled_categories | 0.261388 | 0.245218 |
| hybrid_baseline2 | 0.265593 | 0.250154 |
| hybrid_real3 | 0.265666 | 0.250547 |
| hybrid_shuffled3 | 0.265644 | 0.250307 |

Existing locked EASE/SLIM/PositiveEASE development references, full ranking metrics, explicit denominators, frequency diagnostics and paired descriptive differences are retained in aggregates.json. Reference models have different information/selection budgets. No probabilities, likelihood equivalence, population confidence claim or general architectural novelty is inferred.
