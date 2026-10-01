# DRAFT: Hybrid recommendation: three-split frozen evaluation

Group: 24

Contributors supplied (1/5): Vlad George Iftode

Frozen held-out means across 3 overlapping splits; contributors pending

The original tables average frozen test evaluations for seeds 2026, 2027 and 2028. Task 1 separately labels later hybrid completion on reused validation-calibration users. The grouping appendix uses a different nested TRAIN assessment. These stages are not one leaderboard.

One page is allocated to each major task, with at most 200 discussion words per task. The report content JSON contains the complete aggregate metrics without individual histories or recommendations.

Contributor metadata is incomplete; remaining names and the individual peer-feedback workbook must be completed before submission. AI assistance was used for implementation, analysis and report preparation.

The lecture's seven hybrid families are mapped to explicit implementations. Earlier reports and negative research evidence are preserved separately; added coverage does not establish novelty or guarantee improvement.

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

[10] Harper and Konstan (2015). The MovieLens Datasets: History and Context. ACM TiiS. Rating timestamps are recording times, not consumption times. https://files.grouplens.org/papers/harper-tiis2015.pdf

[11] Course W4S2-Hybrid lecture, pp. 6–12: weighted, switching, mixed, feature combination, feature augmentation, cascade and meta-level hybrids.

## Task 1: individual models and hybrids

MovieLens 100K; k=10; all recorded ratings relevant. Frozen test: three overlapping 943-user splits, each with 9,596 held-out records; TRAIN+validation items masked. Seeds reuse people and observations.

Lecture families [11]: weighted regression; activity switching; mixed lists; score interactions with activity, genre entropy and popularity (feature combination and augmentation); societal reranking (cascade); genre profiles feeding a collaborative decoder (meta-level). Labels can overlap.

Ridge uses standardized expert scores: static is unconstrained; constrained weights sum to one; calibrated applies that constraint after response alignment. Pairwise fits score differences; RRF sums reciprocal ranks. Original ridge penalties: .001/.01/.1/1.

Completion: 471 meta-fit/236 selection/236 previously exposed calibration users per split; TRAIN-only masking. Grids: mixed quotas 6/2/2, 4/4/2, 4/2/4; meta penalties .1/1/10/100; switch groups 2/3/4; RRF offsets 10/60/100. All choices sealed before assessment.

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

**Separate exploratory completion: mean assessment nDCG@10**

| Model | nDCG | Matched model | nDCG |
| --- | --- | --- | --- |
| Mixed lists | 0.2168 | Meta-level | 0.1987 |
| Tuned switch | 0.2746 | Fixed switch | 0.2746 |
| Tuned RRF | 0.2713 | Fixed RRF | 0.2706 |
| EASE | 0.2738 | Context regression | 0.2781 |

**Discussion**

Frozen disagreement fusion scores 0.3370 versus 0.3161 for the validation-selected standalone reference (+6.6% relative); static fusion scores 0.3356. Response alignment raises constrained nDCG from 0.268 to 0.334, exposing a score-scale cost; negative coefficients are allowed, not probabilities. FISMCorrected repairs supplied loss/history errors, not the paper's exact objective. LightGCN's budget extension followed improving validation curves. Mixed and meta-level score below matched EASE; switching is unchanged and RRF tuning changes utility little. Reused assessment cannot establish fresh generalization or novelty.

## Task 2: effectiveness and interpretation

Metrics are independent of RecBole. Accuracy macro-averages users with held-out positives. Activity terciles and the top-20% popularity head use TRAIN only. Genre diversity is pairwise Jaccard distance. Head/tail recall conditions on users with positives in that group; exposure is recommendation-slot share. All rows are equal-weight means of three frozen splits. Coefficients are frozen static-regression weights.

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

**User-group nDCG; conditional head/tail item recall**

| Model | Sparse | Medium | Dense | Head | Tail |
| --- | --- | --- | --- | --- | --- |
| EASE | 0.254 | 0.260 | 0.450 | 0.383 | 0.010 |
| SLIMElastic | 0.251 | 0.254 | 0.434 | 0.361 | 0.029 |
| context | 0.259 | 0.273 | 0.466 | 0.387 | 0.016 |

**User-group genre diversity; head/tail item exposure**

| Model | Sparse | Medium | Dense | Head | Tail |
| --- | --- | --- | --- | --- | --- |
| EASE | 0.801 | 0.807 | 0.823 | 0.985 | 0.015 |
| SLIMElastic | 0.802 | 0.816 | 0.828 | 0.956 | 0.044 |
| context | 0.789 | 0.794 | 0.810 | 0.981 | 0.019 |

**Discussion**

Coefficients reflect correlated score associations, not causal credit; NeuMF's largest weight does not make it the strongest individual model. SLIM allocates more slots to the tail and has higher tail recall than EASE, despite lower overall nDCG. Context fusion improves user-group accuracy while reducing genre diversity in every group. These differences motivate explicit accuracy/diversity and head/tail trade-offs, rather than declaring one model universally better. Our earlier custom field rose from nDCG 0.014 under conditional-rating training to 0.268 under joint-record training at a 400-epoch cap, still below EASE; adaptive routing had no consistent demonstrated advantage. Its full controls remain in the archived report. A separate TRAIN-only audit finds 70.14% of records in same-user timestamp ties and pair-weighted genre Jaccard 0.234 versus 0.184 after within-user shuffling. This establishes recording-group structure, not predictive benefit or viewing sessions [10]. The separately declared grouping study kept target timestamps hidden and found no ranking advantage (appendix). Reused development evidence is exploratory; no new method or superior hybrid is inferred from coherence alone.

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


# Appendix: do rating-recording groups improve ranking?

REVIEW DRAFT | Exploratory nested TRAIN split. Original VALID/TEST not accessed in this experiment.

Motivation: 70.14% of original TRAIN ratings share a user's timestamp. Within-group genre Jaccard is 0.234 versus 0.184 across 100 within-user shuffles. This is descriptive evidence.

Known constrained ridge, singleton + pair features; own-target features excluded. F/D/A = 65,518/7,645/7,645 records; 943 users. All 39 candidates use F context; D selects, A assesses after sealing. All 1,682 items eligible except F observations; no refit or target timestamps.

## All-recorded assessment at ten (every selected arm)

| Model | lambda / beta | nDCG | Recall |
| --- | --- | --- | --- |
| Tuned EASE | 300 / 0 | 0.1730 | 0.2043 |
| Unordered pairs | 250 / 0.1 | 0.1727 | 0.2027 |
| Recording groups | 250 / 0.1 | 0.1726 | 0.2035 |
| Shuffle 1 | 250 / 0.1 | 0.1728 | 0.2031 |
| Shuffle 2 | 250 / 0.1 | 0.1730 | 0.2039 |
| Shuffle 3 | 250 / 0.1 | 0.1726 | 0.2035 |

## Recording groups minus control: paired nDCG

| Control | Mean change | 95% interval |
| --- | --- | --- |
| vs EASE | -0.0004 | [-0.0020, +0.0013] |
| vs unordered | -0.0002 | [-0.0014, +0.0012] |
| vs shuffle mean | -0.0003 | [-0.0012, +0.0006] |

Intervals: 2,000 paired-user resamples, descriptive and not multiplicity adjusted. Shuffle comparison averages metrics, never scores. Full subgroup/secondary metrics remain in aggregate evidence.

## Discussion

Recording groups did not meet the declared research-priority rule: at least 10% relative nDCG gain over tuned EASE and higher means than unordered pairs and shuffled groups. The measured EASE-relative change was -0.23%. Coherent recording groups alone do not establish predictive value. The unordered control tests the grouping restriction; timestamp shuffles retain each user's movies and group sizes. They do not identify exposure, interface design, or watching sessions. Pairwise linear recommendation is established; the contribution here is a controlled input-structure test. This single nested split follows earlier dataset exploration. Its assessment is separated from model selection, but is not fresh population confirmation or directly comparable with the preceding test tables.

Prior: Steck & Liang (2021), Negative Interactions for Improved Collaborative Filtering: Don’t go Deeper, go Higher. doi:10.1145/3460231.3474273. Protocol, source hashes and all candidates accompany this report.
