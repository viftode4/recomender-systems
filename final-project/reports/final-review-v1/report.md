# DRAFT: Hybrid recommendation: three-split frozen evaluation

Group: 24

Contributors supplied (1/5): Vlad George Iftode

Frozen held-out means across 3 overlapping splits; contributors pending

Every primary table averages separately frozen evaluations for seeds 2026, 2027, 2028. Model identities and family roles are aligned from frozen metadata; no test-based selection occurs.

One page is allocated to each major task, with at most 200 discussion words per task. The report content JSON contains the complete aggregate metrics without individual histories or recommendations.

Contributor metadata is incomplete; remaining names and the individual peer-feedback workbook must be completed before submission. AI assistance was used for implementation, analysis and report preparation.

**Method references**

[1] Steck (2019). Embarrassingly Shallow Autoencoders for Sparse Data. WWW. https://arxiv.org/abs/1905.03375

[2] Rendle et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI. https://arxiv.org/abs/1205.2618

[3] Steck (2018). Calibrated Recommendations. RecSys. doi:10.1145/3240323.3240372. Our JSD implementation is an adaptation, not an exact reproduction.

[4] Ning and Karypis (2011). SLIM: Sparse Linear Methods for Top-N Recommender Systems. ICDM. doi:10.1109/ICDM.2011.134.

[5] Kabbur, Ning and Karypis (2013). FISM: Factored Item Similarity Models for Top-N Recommender Systems. KDD. doi:10.1145/2487575.2487589.

[6] He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. https://arxiv.org/abs/2002.02126

[7] He et al. (2017). Neural Collaborative Filtering. https://arxiv.org/abs/1708.05031

[8] Wang et al. (2019). Neural Graph Collaborative Filtering. https://arxiv.org/abs/1905.08108

[9] Aiolli (2013). Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets. RecSys. doi:10.1145/2507157.2507189. Cosine KNN uses the course adapter, not all paper variants.

## Task 1: individual models and hybrids

MovieLens 100K, full-catalog ranking at k=10; 3 overlapping random splits (2026, 2027, 2028), each 943 users; held-out interactions by seed: 9596/9596/9596. All observed ratings are relevant. Inference uses frozen TRAIN fits and masks TRAIN+validation history. Tables average per-split macro metrics, not pooled users or independent dataset replications.

Meta-fit users select expert settings and fit standardized-score coefficients; development users choose hybrid penalties before freezing. User fusion adds score-by-activity/genre-entropy interactions; item fusion adds score-by-popularity interactions; disagreement adds cross-expert score deviation; context combines these. Pairwise variants fit positive-minus-negative feature differences. RRF sums reciprocal ranks with offset 60; group-switch selects a meta-fit expert per activity tercile.

Static ridge is unconstrained. The lecture sum-to-one baseline allows negative weights; calibrated ridge first learns an affine response alignment per expert on meta-fit users. Effective raw-z coefficients need not sum to one, and scores are unbounded.

**Mean held-out nDCG@10; individual models and frozen hybrid families**

| Model | nDCG | Model | nDCG |
| --- | --- | --- | --- |
| BPR | 0.2847 | SignedChannelsLinear | 0.2870 |
| ContrastTransfer | 0.1599 | UserKNN | 0.2805 |
| EASE | 0.3215 | calibrated | 0.3339 |
| ExactPop | 0.1412 | constrained | 0.2677 |
| FISMCorrected | 0.2717 | context | 0.3326 |
| GenreContent | 0.0232 | context-pairwise | 0.3100 |
| ItemKNN | 0.2809 | disagreement | 0.3370 |
| LightGCN | 0.3003 | item | 0.3345 |
| NGCF | 0.2863 | static | 0.3356 |
| NeuMF | 0.2678 | static-pairwise | 0.3290 |
| PositiveEASE | 0.2847 | user | 0.3340 |
| Random | 0.0089 | rrf | 0.3205 |
| SLIMElastic | 0.3134 | group-switch | 0.3211 |

**Discussion**

Disagreement-conditioned fusion scores 0.3370 versus 0.3161 for the validation-selected standalone reference (+6.6% relative); static fusion scores 0.3356. The reference was selected separately in each split; these are descriptive held-out differences. Response alignment raises constrained nDCG from 0.268 to 0.334: raw sum-to-one restrictions also impose a response-scale cost. Negative weights remain allowed and are not probabilities. FISMCorrected jointly repairs the supplied logit/loss mismatch and target-history inclusion, retaining its non-squared regularizer; it is not an exact paper reproduction. LightGCN received a declared longer training budget following improving validation curves. Test metrics never choose models or budgets.

## Task 2: effectiveness and interpretation

Metrics are computed independently of RecBole. Accuracy macro-averages users with held-out positives. Activity terciles and the top-20% popularity head use training data only. Genre diversity is pairwise Jaccard distance; item-group recall includes users with positives in that group. Coefficients below are frozen static regression coefficients.

**Mean static-regression coefficients on standardized expert scores**

| Model | Weight | Model | Weight |
| --- | --- | --- | --- |
| BPR | +0.004 | LightGCN | +0.001 |
| ContrastTransfer | +0.006 | NGCF | +0.008 |
| EASE | +0.033 | NeuMF | +0.050 |
| ExactPop | +0.001 | PositiveEASE | +0.018 |
| FISMCorrected | +0.011 | SLIMElastic | +0.029 |
| GenreContent | +0.006 | SignedChannelsLinear | +0.002 |
| ItemKNN | +0.015 | UserKNN | -0.004 |

**Cross-model group accuracy: user nDCG and tail-item recall**

| Model | Sparse | Medium | Dense | Tail recall |
| --- | --- | --- | --- | --- |
| EASE | 0.254 | 0.260 | 0.450 | 0.010 |
| SLIMElastic | 0.251 | 0.254 | 0.434 | 0.029 |
| context | 0.259 | 0.273 | 0.466 | 0.016 |

**Context hybrid: beyond-accuracy user and item groups**

| Measure | Sparse | Medium | Dense | Head | Tail |
| --- | --- | --- | --- | --- | --- |
| Diversity | 0.789 | 0.794 | 0.810 | n/a | n/a |
| Exposure | n/a | n/a | n/a | 0.981 | 0.019 |

**Discussion**

Coefficients reflect correlated score associations, not causal credit. The group tables distinguish sparse-user and tail-item outcomes; activity is not demographic identity. Conditional rating training gives observed nDCG 0.014. Joint normalization learns recorded-item mass: adaptive nDCG 0.229 at the 100-epoch cap and 0.268 at the 400-epoch cap, versus locked EASE 0.322 and SLIM 0.313. At that cap, adaptive-minus-fixed nDCG is -0.004; adjusted paired intervals favor adaptive in 0/3 splits and fixed flow in 0/3. On the separate liked-record endpoint, joint400 scores 0.250 versus liked-selected PositiveEASE 0.294. The appendix retains original controls and the post-v1 budget extension. Intervals use approximate Bonferroni-adjusted bootstrap bounds for 24 contrasts within each split; there is no independent-split confidence interval. Objective alignment and custom code do not establish novelty.

## Task 3: societal aspects and reranking

The fixed-strength comparison uses 0.5 for diversity, calibration [3] and exposure rerankers. Genre JSD measures deviation from historical genre shares; lower values mean better calibration. Head exposure measures allocation to popular items. RRF ordering comparisons retain full ranking support; objective values accompany accuracy.

UPD is per-user popularity-profile JSD. The base contextual hybrid's mean discounted catalog exposure Gini is 0.927; lower values mean less concentrated allocation.

**Mean held-out accuracy and societal trade-offs**

| Reranker | nDCG | Diversity | JSD | Head | UPD |
| --- | --- | --- | --- | --- | --- |
| Base context | 0.333 | 0.798 | 0.141 | 0.981 | 0.149 |
| Diversity | 0.307 | 0.895 | 0.131 | 0.967 | 0.133 |
| Calibration | 0.329 | 0.806 | 0.125 | 0.980 | 0.148 |
| Item exposure | 0.318 | 0.793 | 0.141 | 0.931 | 0.097 |
| Popularity calibration | 0.333 | 0.795 | 0.141 | 0.961 | 0.124 |

**Before/after RRF: nDCG and objective value (diversity, JSD, head share or UPD)**

| Objective | Bef. nDCG | Aft. nDCG | Bef. goal | Aft. goal |
| --- | --- | --- | --- | --- |
| diversity | 0.315 | 0.298 | 0.832 | 0.887 |
| calibration | 0.321 | 0.313 | 0.142 | 0.132 |
| exposure | 0.321 | 0.304 | 0.979 | 0.946 |
| popularity_calibration | 0.320 | 0.320 | 0.154 | 0.142 |

**Group effects: utility-policy users and fixed-exposure item groups**

| Group | Base | After | Metric |
| --- | --- | --- | --- |
| Sparse users | 0.259 | 0.260 | nDCG |
| Medium users | 0.273 | 0.273 | nDCG |
| Dense users | 0.466 | 0.465 | nDCG |
| Head items | 0.387 | 0.376 | Recall |
| Tail items | 0.016 | 0.039 | Recall |

**Discussion**

Diversity rises from 0.798 to 0.895, while nDCG falls from 0.333 to 0.307. Exposure reranking raises tail recall from 0.016 to 0.039. Across seeds and activity groups, policy utility retention ranged from 99.7% to 101.0%. The independent calibration target is not a held-out guarantee. Smaller gaps can hide losses for every group, so absolute utility accompanies disparity. Catalog-proportional exposure is a declared objective, not a universal fairness definition. With H head and T tail candidates, feasible head counts satisfy max(0,k-T)<=heads<=min(k,H). Reranking before fusion changes information supplied to the fusion rule, so its order can change both accuracy and the target objective. Every frozen strength remains in aggregate evidence; test outcomes never retune strength.

## Appendix: Task 2 field objective alignment

Frozen held-out evidence (3 overlapping seeds); test never selects models or checkpoints. The same categorical recurrent field is trained with a joint softmax over available movie-rating pairs. Categorical evidence sends messages through learned global ports; recurrent routing and soft source influence update item states before a five-logit decoder. All five rating categories remain inputs.

Observed-record ranking uses logsumexp over all five logits; liked-record ranking uses logsumexp over categories 4 and 5. Conditional rating probabilities normalize the five logits within an item. Inference retains TRAIN-only rating context and masks TRAIN+validation items from ranking; all-observed/liked users by seed: 943/882, 943/882, 943/884. Only liked metrics exclude users without held-out likes.

Joint checkpoints 10/30/60/100 minimize meta-fit macro-user joint negative log-likelihood (NLL). Adaptive and fixed-flow variants share the planned budget. Only the original meta-fit results choose a checkpoint.

The independent conditional track selects epochs 10/30/60/100 by meta-fit rating CE; its two ranking diagnostics use P(rating>=4|item). Missing observations are unknown in both tracks. Fixed flow caches initial routes and source gates. Conditional hard-clamp pins observed states and deactivates gate parameters; equal allocated size is not equal active capacity.

**Conditional objective: held-out test means; ranking uses its own readout**

| Predictor | Rating CE | Obs nDCG | Like nDCG |
| --- | --- | --- | --- |
| Conditional adaptive | 1.2848 | 0.0138 | 0.0145 |
| Conditional fixed flow | 1.2840 | 0.0145 | 0.0153 |
| Conditional hard clamp | 1.2847 | 0.0140 | 0.0142 |
| Global frequencies | 1.4631 | 0.0273 | 0.0234 |
| Item frequencies | 1.3773 | 0.0496 | 0.0520 |
| Item/user product | 1.2960 | 0.0469 | 0.0486 |

**Joint objective: held-out test means; NLL and rating CE lower is better**

| Predictor | Joint NLL | Rating CE | Obs nDCG | Like nDCG |
| --- | --- | --- | --- | --- |
| Joint adaptive | 6.9612 | 1.3092 | 0.2288 | 0.2077 |
| Joint fixed flow | 6.9627 | 1.2979 | 0.2294 | 0.2069 |
| Event/global ratings | 7.8749 | 1.4631 | 0.1412 | 0.1289 |
| Event/item ratings | 7.7892 | 1.3773 | 0.1412 | 0.1285 |
| Event/item/user tilt | 7.6952 | 1.2960 | 0.1407 | 0.1274 |

**Discussion**

Joint normalization can learn relative recorded-item mass; per-item conditional cross-entropy alone cannot identify that mass. This is a loss-alignment ablation of the same field, not a new architecture or an ensemble. Joint NLL and conditional rating CE have different sample spaces and must not be compared as if they were the same loss. Both ranking adapters must be reported even when one looks stronger. Observed recording behavior is not satisfaction, and liked-record mass differs from P(rating>=4|item). Test splits share users and a dataset, limiting generalization. The recurrent routing mechanism has prior art in Set Transformer (Lee et al., 2019, PMLR 97:3744-3753); neither custom code nor objective alignment establishes novelty or state-of-the-art performance.

## Appendix: Task 2 joint-field budget sensitivity

Separate post-v1 budget sensitivity (held-out test). The original 100-epoch study is retained. After its meta-fit curves were still improving at the boundary, a single 400-epoch cap was declared. The extension reuses the same splits, model, initialization seeds and training episodes; it is an exploratory follow-up, not independent confirmation.

Checkpoints 10/30/60/100/200/300/400 are selected only by meta-fit macro-user joint NLL. Row labels give maximum training budgets, not selected epochs. Both recurrent and fixed-flow variants receive the same budget. Count baselines are unchanged controls.

All five rating categories remain evidence. Observed-record ranking sums mass over five categories; liked-record ranking sums categories 4 and 5. Conditional rating CE normalizes within each item. Frozen inference retains TRAIN-only context and masks TRAIN+validation for ranking.

**Joint objective: held-out test means; NLL and rating CE lower is better**

| Predictor | Joint NLL | Rating CE | Obs nDCG | Like nDCG |
| --- | --- | --- | --- | --- |
| 100ep adaptive | 6.9612 | 1.3092 | 0.2288 | 0.2077 |
| 100ep fixed flow | 6.9627 | 1.2979 | 0.2294 | 0.2069 |
| 400ep adaptive | 6.7485 | 1.2862 | 0.2679 | 0.2496 |
| 400ep fixed flow | 6.7376 | 1.2811 | 0.2720 | 0.2508 |
| Event/global ratings | 7.8749 | 1.4631 | 0.1412 | 0.1289 |
| Event/item ratings | 7.7892 | 1.3773 | 0.1412 | 0.1285 |
| Event/item/user tilt | 7.6952 | 1.2960 | 0.1407 | 0.1274 |

**Discussion**

A larger budget can change the apparent cost or benefit of a mechanism; the original fixed-budget result must remain visible. This follow-up tests budget sensitivity of the same objective, not a new architecture. A selected checkpoint at the last allowed epoch does not prove convergence or universal architectural failure. No budget will be chosen from held-out outcomes. Joint NLL and conditional rating CE have different denominators. Repeated splits share users and movies, so consistent differences across seeds would still not establish independent dataset replication, emotional understanding or state-of-the-art performance.

## Appendix: Task 2 rejection-information audit

Supplemental development audit, 3 overlapping seeds. Likes >=4, dislikes <=2, rating 3 neutral. Context is 80% of each user's training history; the disjoint remainder supplies decoder targets. Teacher and decoder inputs use context only. All originally observed training items are masked from ranking.

Liked-endpoint users by seed: 444/434/433; users without held-out likes are excluded. This cohort differs from the main assignment comparison.

Primary queries use context only; full-history queries reuse the fitted decoder and add known history, changing input length. Fixed ridge is 250. Secondary equal-grid [50,250,1000] selection uses only context meta-fit liked nDCG; all five replicates share one control-branch setting.

The placement null preserves observed support, positives and user/item dislike counts, including within the context-defined fitting stratum. Weight permutations preserve each user's dislike identities and weight multiset. The table averages per-user control metrics, never control scores.

**Liked nDCG@10: mean across seeds and all five control replicates**

| Branch | Context fixed | Context grid | Full fixed | Full grid |
| --- | --- | --- | --- | --- |
| Positive only | 0.1618 | 0.1813 | 0.1734 | 0.1860 |
| Plain dislikes | 0.1609 | 0.1797 | 0.1704 | 0.1876 |
| Surprise weights | 0.1598 | 0.1797 | 0.1693 | 0.1870 |
| Permuted weights | 0.1609 | 0.1800 | 0.1697 | 0.1867 |
| Placement null | 0.1618 | 0.1795 | 0.1726 | 0.1865 |

**Discussion**

At fixed ridge, plain dislikes change context nDCG by -0.0009 versus positive-only and -0.0009 versus the placement null. Surprise weighting changes it by -0.0010 versus permuted weights. These are descriptive differences, not significance claims. Every plain-minus-placement interval contains zero; a stable placement benefit is unestablished. The aggregate artifact retains per-seed user-bootstrap intervals, every control replicate, movement diagnostics and known-dislike slot rates. Structural zeros may disconnect the switch space; movement does not establish uniform sampling or convergence. Bootstrap intervals condition on five draws. Full-history results cannot replace the primary context test. Liked relevance differs from the assignment's all-observed endpoint; low known-dislike rates cannot certify satisfaction. Reused validation users and existing hard-negative weighting research (LAGCL4Rec, Findings EMNLP 2025, paper 61) preclude independent-confirmation or novelty claims. Switch-space limitations follow Rapallo and Yoshida (2010, arXiv:0905.4841).
