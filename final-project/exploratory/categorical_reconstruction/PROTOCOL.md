# Categorical reconstruction: frozen exploratory protocol

Declared before real-data candidate fitting. Dataset TEST was already evaluated
in the original coursework. This study uses the existing, repeatedly used
validation cohorts and is **post-test exploratory research**, not fresh test
confirmation. The new runner must never open TEST splits or final-test results.
The original report and submission archive remain unchanged.

## Question and contribution

Does retaining the categories of historical ratings improve prediction of an
unseen recorded interaction, beyond a more thoroughly tuned binary EASE model?
Does it add useful information to the EASE/SLIM hybrid required by Task 2.6?

We implement a constrained categorical ridge reconstruction model and a
user-space exact solver. This is an independently implemented and tested model,
not a demonstrated new scientific principle. EASE (Steck, 2019), FEASE (2025)
and categorical rating methods such as Fifty Shades of Ratings (2016) are close
prior art. Any novelty statement must distinguish implementation, controlled
evaluation and a genuinely unestablished methodological contribution.

## Model and information boundary

For TRAIN matrix R, define X_ui = 1[R_ui > 0]. For categories r=1,...,5,
Z_uir = 1[R_ui=r] - X_ui p_ir. Here p_ir is the item/category TRAIN count,
smoothed with 20 times the global TRAIN category frequency, divided by the
item's TRAIN count plus 20. All six features are zero for absent interactions.
An entirely empty TRAIN matrix uses a uniform global prior. No validation
ratings, item metadata or pretrained model scores enter this standalone model.

Fit B to minimize ||X - F B||_F^2 + sum_j D_j ||B_j,:||_2^2, where
F = [X, Z_1, ..., Z_5]. For every target item i, constrain all its own six
feature coefficients B[I_i,i] to zero. Binary-only F=X is the exact EASE
special case. D equals lambda_binary on X and lambda_binary*category_ratio on
the five categorical channels. Scores are ranking scores, not probabilities.
Zeros in the reconstruction target are the implicit-feedback convention, not
confirmed dislikes. At prediction time use original TRAIN histories only.

An exact user-space Cholesky solve plus per-target Schur correction avoids
inverting a feature-square matrix. Reduced-design SVD is an exceptional
numerical fallback. Check residuals and independently compare small examples
to constrained primal ridge and binary EASE. Full-query prediction must remove
all own-feature contributions; the faster kernel scores are valid only for
target-absent candidates. Exclude PAD and all TRAIN interactions from ranking.

## Fixed standalone search

- Seeds: 2026, 2027, 2028, preserving original identities, TRAIN/VALID splits,
  and user cohorts: 471 meta-fit users and 472 development users per seed.
- Binary penalties in declaration order:
  [10, 30, 50, 100, 250, 300, 1000, 3000, 10000].
- Categorical/binary penalty ratios in declaration order: [0.1, 1, 10, 100].
- Smoothing: 20, fixed. No gradient checkpoints or warm-start selection.
- Binary expanded arm: all 9 binary candidates. Original-grid reference:
  the subset [50, 250, 1000], selected independently on the same meta-fit users.
- Real categorical arm: 36 categorical candidates plus those 9 cached binary
  candidates as a nested fallback (45 selectable candidates).
- Shuffled categorical arm: the same 36 categorical candidates plus the same
  9 cached binary candidates (45 selectable candidates).
- Total: 81 distinct search fits per seed, 243 across all three seeds.
  Selected-candidate deterministic replay fits are additional verification.
- Shuffle observed TRAIN categories within each item using RNG seed+17011.
  Preserve observation identities and every item's exact category histogram;
  use the shuffled histories for both fitting and querying this control.
- Select exact maximum meta-fit all-observed nDCG@10. Exact ties retain the
  earlier declared candidate: binary precedes categorical; penalties and ratios
  follow the lists above. No numerical-tolerance-based post-hoc tie changes.

Log every candidate's configuration, meta-fit metric, timing and numerical
diagnostics. A numerical failure aborts the study for inspection; do not silently
discard difficult candidates. Refit selected candidates and require identical
scores. Record unequal candidate budgets in every comparison narrative.

## Fixed hybrid extension for Task 2.6

Fit three arms: expanded binary EASE + locked SLIM; those two + selected real
categorical model; those two + selected shuffled categorical control. These are
hybrid ablations, not a claim that ensembling itself is new. Locked SLIM expert
selection must have used the same meta-fit users and unchanged TRAIN/VALID data.

For each expert, standardize each user's scores over eligible non-TRAIN,
non-PAD items. Use only meta-fit VALID observation identities as positive
examples. For each meta user sample without replacement min(5 times positive
count, available count) negatives excluding TRAIN and all that user's meta-fit
VALID positives, using RNG seed+24011. Share samples across arms and penalties;
record any shortages. Unobserved examples are sampled training negatives, not
known dislikes.

Sort meta user IDs, shuffle with RNG seed+24012, and split floor(n/2) versus the
remaining users: 235 coefficient-fit and 236 penalty-selection users. Fit
nonnegative-slope affine score calibration on the coefficient-fit examples,
then calibrated ridge weights constrained to sum to one, with an unpenalized
intercept. Negative weights are permitted. Select penalties [0.001, 0.01, 0.1,
1] by inner penalty-selection all-observed nDCG@10; exact ties prefer the larger
penalty. Refit calibration and weights on all 471 meta users at the selected
penalty. Expert selection itself used all meta users, so this is not an unbiased
nested estimate of expert performance; the development users remain disjoint.
If category selection returns the binary fallback, report duplicate-expert and
regularization effects explicitly rather than attributing a gain to categories.

## Global selection barrier and evaluation

Seal all standalone choices and hybrid choices/coefficients for all three seeds
in one hash-verified selection barrier **before computing any development
metric**. Seal executable sources and transitive local helpers, runtime,
protocol, source data, split, ordered identity, cohort and selected-score hashes.
Store raw user-level data only under ignored runs/. Publish aggregate evidence.

Primary descriptive contrast: selected real categorical minus expanded binary
on development all-observed nDCG@10. The shuffled arm tests whether any apparent
benefit comes from meaningful user/rating association. The three hybrid arms
test assignment usefulness. Also retain the original-grid binary, locked EASE,
SLIM and PositiveEASE references with matched candidates/cohorts and disclosed
information and tuning budgets.

Secondary endpoints: Precision/Recall/MRR@10, liked-rating nDCG@10 (ratings>=4),
known-dislike recommendation rate (ratings<=2), coverage and novelty. Use one
selected score vector for both relevance definitions; do not tune on likes.
Report endpoint denominators, per-seed values, equal-seed descriptive means and
fractions of users improved. No new population significance claim; repeated
splits overlap and development was used in earlier project work.

Predeclared group analysis, without any further tuning:

- User activity terciles from TRAIN counts only; thresholds at 1/3 and 2/3
  quantiles, searchsorted(side='right'). Per-group macro ranking metrics,
  pairwise genre-Jaccard list diversity, and genre/popularity Jensen-Shannon
  divergence where defined.
- Item head: top ceil(20% of non-PAD catalogue) by TRAIN frequency, breaking
  ties by original item token. Tail: remaining items. Conditional recall for
  users with VALID positives in each group, with explicit user denominators;
  ordinary and discounted recommendation exposure shares, item exposure Gini
  and entropy where available.

Recheck sealed files after evaluation. Independently recompute primary metrics,
selection maxima and prediction replay, and verify the original-grid EASE
special case against locked original scores. Any mismatch must be resolved
before report rendering. Negative or mixed results remain reportable results.

## Deliverables and claim gate

Deliver solver/derivation, tests, runnable experiment, complete aggregate
evidence, assignment-fit analysis, and a separate measured research supplement
plus review archive preserving the original report. Do not modify original
frozen study files. Missing real member names/contributions remain deferred.

A development improvement supports a narrowly worded empirical finding on
these cohorts. A field-level leap, universal improvement, SOTA, first-ever
method, and guaranteed grade are not supported by this protocol. Fresh external
data and stronger independent confirmation would be required for broader claims.
