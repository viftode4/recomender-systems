# DRAFT: Hybrid recommendation: usefulness, exceptions and exposure

Group: 24

Contributors supplied (1/5): Vlad George Iftode

Selected development evidence; final test evaluation outstanding

DSAIT4335 Recommender Systems. All reported values are development results, not final test performance or evidence of state-of-the-art performance.

The assignment specifies one page and at most 200 discussion words per task. This draft conservatively allocates one page to each major task (1, 2 and 3).

Development evidence is exploratory. Freeze the reviewed model choices before evaluating held-out tests, and retain the recorded protocol and source hashes with the results.

Group number and actual contributors must be supplied before submission. AI assistance was used; members must review the implementation and record actual contributions.

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

MovieLens 100K; seeds 2026, 2027, 2028. Every observed rating is an implicit positive. RecBole uses per-user random 80/10/10 splits and full-catalog ranking. Previously seen items are masked. Experts train for fixed budgets. Validation users split into 471 meta-fit and 236 development users; 236 users are reserved for policy calibration. This is not a cold-start experiment.

Meta-fit selects expert settings from predeclared grids. EASE [1] and BPR [2] anchor the comparison. Standardized expert scores feed ridge regression, with five sampled unobserved negatives per positive. User, item and disagreement features are ablated. Hybrid penalties (0.001/0.01/0.1/1) are selected on development users. Explicit-rating experts preserve dislikes; their rating-information controls are included.

Constrained weights sum to one but can be negative. The unscaled version operates on z-scores; the calibrated version first learns per-expert affine response alignment on meta-fit users. Effective raw-score coefficients need not sum to one. Outputs remain unbounded ranking scores.

Class models follow [4-9]. FISMCorrected fixes the sigmoid/BCEWithLogitsLoss mismatch and removes each scored item from its own history and normalization. The supplied non-squared norm regularizer is retained: a joint correction, not an exact original-objective reproduction.

**All individual models and selected hybrids: mean development nDCG@10**

| Model | nDCG | Model | nDCG |
| --- | --- | --- | --- |
| BPR | 0.2204 | SignedChannelsLinear | 0.2312 |
| ContrastTransfer | 0.1396 | UserKNN | 0.2264 |
| EASE | 0.2499 | calibrated | 0.2612 |
| ExactPop | 0.1232 | constrained | 0.2189 |
| FISMCorrected | 0.2220 | context | 0.2579 |
| GenreContent | 0.0244 | context-pairwise | 0.2376 |
| ItemKNN | 0.2192 | disagreement | 0.2622 |
| LightGCN | 0.2263 | item | 0.2613 |
| NGCF | 0.2231 | static | 0.2620 |
| NeuMF | 0.2104 | static-pairwise | 0.2551 |
| PositiveEASE | 0.2294 | user | 0.2582 |
| Random | 0.0071 | rrf | 0.2519 |
| SLIMElastic | 0.2502 | group-switch | 0.2507 |

**Discussion**

Affine alignment matches expert response scales before enforcing sum-to-one weights. The alignment uses sampled zero/one labels; it does not produce calibrated probabilities. Contextual coefficients allow expert contributions to vary with history size, genre entropy and item popularity. Small differences between ablations do not justify a claim that additional complexity improves generalization. The BPR grid changes dimension and training budget together, so their effects cannot be isolated. RRF and activity-group switching provide alternative hybrid controls. Random is excluded from fusion. The best development settings still require a frozen test evaluation; overlapping data seeds are sensitivity checks, not independent replications.

## Task 2: effectiveness and interpretation

All metrics are implemented independently: precision, recall, nDCG, MRR, hit rate, catalog coverage, smoothed-frequency novelty, genre Jaccard diversity and genre JSD calibration. Metrics macro-average users with held-out positives. Training-only history terciles and item popularity define groups; head items are the most frequent 20% of the catalog.

Exception study (3 overlapping seeds): likes >=4, dislikes <=2, and rating 3 neutral. Seen items are masked. Meta-fit selects settings. Liked-item evaluation excludes users without held-out likes. Its separate cohort and relevance definition are not directly comparable with Task 1.

**Separate rating-aware development study: mean nDCG@10 for likes and known-dislike slot rate**

| Model | Liked nDCG | Dislike rate |
| --- | --- | --- |
| Observed EASE | 0.2450 | 0.0162 |
| Contrast transfer | 0.1359 | 0.0060 |
| Signed channels | 0.2579 | 0.0090 |
| Positive EASE | 0.2575 | 0.0085 |
| Pair gate | 0.0908 | 0.0014 |

**Static ridge: mean coefficients on standardized expert scores**

| Model | Weight | Model | Weight |
| --- | --- | --- | --- |
| BPR | +0.0043 | LightGCN | +0.0007 |
| ContrastTransfer | +0.0057 | NGCF | +0.0080 |
| EASE | +0.0329 | NeuMF | +0.0504 |
| ExactPop | +0.0009 | PositiveEASE | +0.0178 |
| FISMCorrected | +0.0109 | SLIMElastic | +0.0292 |
| GenreContent | +0.0059 | SignedChannelsLinear | +0.0021 |
| ItemKNN | +0.0151 | UserKNN | -0.0040 |

**Contextual model: mean user and item group results**

| Group | nDCG | Recall | Diversity | Exposure |
| --- | --- | --- | --- | --- |
| Sparse users | 0.230 | 0.329 | 0.784 | n/a |
| Medium users | 0.220 | 0.236 | 0.796 | n/a |
| Dense users | 0.325 | 0.152 | 0.803 | n/a |
| Head items | n/a | 0.317 | n/a | 0.985 |
| Tail items | n/a | 0.012 | n/a | 0.015 |

**Discussion**

Static coefficients are associations between correlated standardized scores, not causal credit or probabilities. The group table tests whether aggregate accuracy hides sparse-user or tail-item failures. Contrast transfer achieved 0.1359 liked nDCG versus 0.2579 for separate positive/negative linear channels. The proposed relation mechanism therefore failed this comparison; complexity did not earn its place. The pair-gate revision was motivated by earlier development failures and is exploratory. Information-matched controls separate rating access from architecture. Low known-dislike rates only count observed held-out dislikes: unknown ratings cannot be treated as approval. These findings do not establish emotional understanding, universal superiority or novelty.

## Task 3: societal aspects and reranking

Greedy rerankers trade relevance against diversity, calibration [3] or head/tail exposure. The table fixes strength at 0.5, with a top-100 candidate pool. Lower JSD indicates better genre calibration; head exposure is a share, not a fairness verdict. Catalog parity is an explicit diagnostic objective. The ordering comparison uses RRF with constant 60 and retains full ranking support.

UPD is per-user popularity-profile JSD (lower is closer). Discounted catalog exposure Gini changes from 0.939 to 0.927 under item-exposure reranking; lower Gini means less concentrated allocation.

**Contextual model: mean development trade-offs**

| Reranker | nDCG | Diversity | JSD | Head | UPD |
| --- | --- | --- | --- | --- | --- |
| Base context | 0.258 | 0.794 | 0.139 | 0.985 | 0.152 |
| Diversity | 0.240 | 0.894 | 0.125 | 0.975 | 0.140 |
| Calibration | 0.260 | 0.804 | 0.121 | 0.985 | 0.152 |
| Item exposure | 0.252 | 0.790 | 0.140 | 0.944 | 0.106 |
| Popularity calibration | 0.259 | 0.792 | 0.140 | 0.968 | 0.130 |

**Before/after RRF: nDCG@10 and objective value (diversity, JSD, head share or UPD)**

| Objective | Bef. nDCG | Aft. nDCG | Bef. goal | Aft. goal |
| --- | --- | --- | --- | --- |
| diversity | 0.2576 | 0.2382 | 0.828 | 0.887 |
| calibration | 0.2555 | 0.2529 | 0.138 | 0.128 |
| exposure | 0.2520 | 0.2432 | 0.983 | 0.957 |
| popularity_calibration | 0.2524 | 0.2524 | 0.155 | 0.145 |

**Discussion**

Candidate support limits exposure correction: with H head and T tail candidates, a list of k has max(0,k-T)<=heads<=min(k,H). Adaptive expansion used 109.4 candidates on average and achieved at least 100% agreement with full-pool exposure reranking in these runs. This measures candidate work, not end-to-end latency. The user-side policy selects exposure strength on independent calibration users with approximate multiplicity-adjusted lower confidence bounds on retention, targeting 95% utility retention. Sparse-group development retention ranged from 99.7% to 100.2%. The observed audit meets the target but is not a future guarantee. Smaller group gaps can hide harm to all groups, so absolute and worst-group utility must also be inspected. Calibration and useful preference exceptions may conflict; that hypothesis needs the separate rating-aware study.
