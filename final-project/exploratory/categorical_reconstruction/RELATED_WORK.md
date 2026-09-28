# What this study contributes, and what already exists

Primary sources checked on 28 September 2026. This is a focused prior-art check,
not an exhaustive novelty search or a state-of-the-art leaderboard. Reported
scores from other papers cannot be compared numerically without matching data,
splits, relevance definitions, candidate sets and tuning budgets.

| Primary work | Established idea | Relation to this implementation |
|---|---|---|
| [Steck, EASE, WWW 2019](https://arxiv.org/abs/1905.03375) | Full-rank linear reconstruction with a zero diagonal and a closed-form solution. | Our binary-only mode is exactly this objective. The underlying ridge formulation is established. |
| [Cui, Zhang and Lee, FEASE, 2025 preprint, section 3.1](https://arxiv.org/html/2504.02288v4#S3.SS1) | Extend EASE inputs with user and item side features. | Feature expansion of EASE is established. We use observed-only rating residuals and exclude every feature derived from the target item. This does not establish priority for that variant. |
| [Frolov and Oseledets, Fifty Shades of Ratings, RecSys 2016](https://arxiv.org/abs/1607.04228) | Treat feedback as categorical and model users, items and rating values jointly, including negative-feedback scenarios. | Preserving the full rating spectrum instead of binarizing it is established. Our prediction target and constrained linear solver differ from their tensor approach. |
| [Steck and Liang, Negative Interactions for Improved Collaborative Filtering, RecSys 2021](https://doi.org/10.1145/3460231.3474273) | Extend full-rank linear recommendation with explicit higher-order input features and permit negative higher-order coefficients. | An input-expanded linear model is not automatically a new architecture. Our six channels represent individual historical ratings; they are not learned higher-order item conjunctions. |

The fourth paper's author-hosted PDF was available in the search index; its live
PDF URL returned 404 during verification. The indexed abstract supports the
description above. No unseen implementation details from that paper are assumed.

Our concrete deliverable is an independently implemented categorical
reconstruction model with a derived exact solver, checked against a separate
constrained primal solution. It can score arbitrary histories using frozen TRAIN
centering, excludes all target-derived input channels, and nests the binary
baseline. The user-space solve is an application of standard ridge duality and
the Woodbury identity, not a newly discovered matrix identity.

The empirical contribution is a controlled comparison: a wider binary tuning
grid, the same categorical search on within-item shuffled ratings, identical
ranking candidates and cohorts, and a matched hybrid ablation. The shuffle
preserves item rating histograms and observation identities while disrupting
user/category associations. A gain over binary alone is weaker evidence than a
gain that also survives this control. Category arms have more selectable
candidates than the binary arm; this must remain visible in interpretation.

Any observed gain applies to reused MovieLens 100K development cohorts. The
original project already exposed its test results before this study was designed.
Neither a positive result nor an exact implementation establishes a field-level
breakthrough, universal improvement, or first-ever method. A negative result
still answers the stated question for this model and protocol; it should not be
generalized into a claim that rating information is useless.
