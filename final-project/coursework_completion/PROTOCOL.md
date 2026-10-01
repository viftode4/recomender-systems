# Bounded lecture-family completion

Fixed before fitting. This adds explicit mixed and meta-level hybrids, and finite
tuning of switching and reciprocal-rank fusion. It makes no novelty or required
improvement claim. These are established hybrid families applied to the existing
MovieLens100K coursework predictors.

## Existing information and cohort reuse

Use the hash-verified `runs/frozen-v3/{2026,2027,2028}` predictors, exact catalog,
TRAIN matrices and frozen source files. Do not read original TEST interactions or
final TEST results. These frozen choices arose before that earlier TEST audit;
the dataset and validation cohorts have already been extensively explored.
The existing 471 meta-fit users learn switches; the existing 236 development users
choose a setting within each new family; the existing 236 policy-calibration users
receive the final exploratory assessment. These calibration labels were already
used for prior policy fitting and later exploratory analyses. They are reused
assessment data, not a fresh holdout. Seeds share people and are not independent
populations. All comparisons below use the same calibration cohort and candidates.

Before selecting all three seeds, hash complete frozen artifacts as opaque bytes,
but lazily load only named prediction/TRAIN arrays, never `valid_mask` from NPZ.
Read each VALID row's user first; parse item fields only for meta-fit/development
users. Original calibration item labels are interpreted only after the global
all-three-seed barrier seals choices and prediction arrays. No refit follows.

## Families and ordered candidate grids

All ranking ties use original catalog index. Exclude PAD and each user's TRAIN
observations, leaving VALID observations eligible. Cutoff is ten. Inputs are frozen
TRAIN-fitted expert scores; the active 14 experts exclude Random and follow the
original frozen `active_experts` order.

1. **Mixed:** EASE, GenreContent, ExactPop in that order. Quotas `(6,2,2)`, `(4,4,2)`,
   `(4,2,4)`. At zero-based output position t, choose the expert maximizing
   `(t+1)*quota - 10*accepted_count`, among experts with remaining quota and a next
   eligible unique item. Ties use expert order. Skipping seen/PAD/duplicate items
   does not consume a quota. If an expert exhausts, continue other quotas; any
   remaining slots use EASE's remaining ranking. Fail if ten unique eligible items
   cannot be supplied. This is list interleaving, not score averaging.
2. **Meta-level:** Let X be the full binary TRAIN matrix on real items, and G the
   fixed catalog's fractional genre matrix (each row sums to one). Learn
   `P = X G (G'G + lambda I)^-1`, then `Q = X' P (P'P + lambda I)^-1`.
   Predict `P Q'`; the second stage receives only learned content user profiles P.
   Both stages minimize sum squared reconstruction error plus lambda times the
   squared parameter norm, without intercepts. Shared lambda candidates are
   `.1, 1, 10, 100`. No VALID labels fit either stage, and G contains no padding.
3. **Tuned switch:** group-count candidates `2,3,4`. Thresholds use linear quantiles
   of all users' TRAIN activity at `1/G,...,(G-1)/G`; ties go to the upper group.
   Each group chooses the active expert with maximal mean meta-fit nDCG@10 in
   that group, with frozen expert-order ties. A group with no meta-fit users uses
   the global meta-fit winner. Development selects group count.
4. **Tuned RRF:** offsets `10,60,100` applied to the same frozen active experts.
   Sum `1/(offset+rank)` with ranks beginning at one over the eligible catalog.

Total: thirteen configurations per seed, thirty-nine across three seeds. Each
family selects the first exact maximum development macro nDCG@10 in the ordered
grid above. No cross-family winner is selected, no additional grid is introduced,
and poor performance is retained.

## Matched references and assessment

Retain locked EASE, SLIMElastic, original selected context hybrid, original
three-group fixed switch, and original offset-60 RRF. Reapply their frozen score
functions on the TRAIN-only candidate mask. The original context score is replayed
through unchanged `freeze.score_models(..., 'valid')` in user chunks; no TEST phase
is called. Original reference coefficients and expert choices are never tuned.

After all seeds' selections/predictions are sealed, assess all nine roles on each
seed's calibration users, treating all recorded VALID items as relevant. Use the
existing independently implemented metrics and group helper: nDCG, recall, MRR,
precision, hit rate at ten, novelty, catalog coverage, genre diversity/calibration,
TRAIN-activity terciles, and TRAIN-popularity head/tail metrics and discounted
exposure. Group definitions use the full TRAIN population. Publish every candidate
development metric, selected parameters, all assessment aggregates and counts.
Report equal-weight seed means as descriptive summaries, with no significance or
independent-population interpretation. Retain per-user outputs/predictions only in
ignored run storage; public files contain aggregate values and cryptographic hashes.

## Execution and sealing

The standalone runner accepts source-root, item metadata, ignored run output and
public evidence output. Existing output paths are rejected. Verify source/data/
catalog hashes and freeze all implementation/helper/protocol hashes plus runtime
before fitting. Check them at each seed boundary, at the global selection barrier,
and completion. Require all seeds, four selected families and five references in
every private prediction file before assessment access. Synthetic independent tests
must pass before launch. The independent verifier later replays selection/metrics
and selected model equations without opening TEST. Original sealed source files,
reports and packages remain unchanged.
