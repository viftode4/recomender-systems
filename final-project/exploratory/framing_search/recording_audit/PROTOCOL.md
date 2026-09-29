# TRAIN-only recording-bundle audit

This protocol is written before this audit reads timestamp values. It asks whether
the observed rating records contain substantial same-user recording bundles, and
whether films recorded at exactly the same second share genres more than the
same user's films assigned randomly to their existing timestamp slots. It does
not fit a recommender or inspect ranking outcomes.

## Inputs and boundary

- Use only original seed-2026 TRAIN pairs (80,808 expected) from
  `runs/categorical-reconstruction-v1/2026/train.tsv`.
- Verify their SHA-256 against the categorical input signature and original
  `runs/research-v2/2026-EASE-1/manifest.json`; verify that manifest's own digest
  against the signature. Verify raw interaction and item-metadata digests against
  both records before parsing records.
- Read user/item identifiers from each raw interaction only to test membership.
  Parse rating and timestamp values **only after membership in TRAIN is true**.
  All other rows are skipped without interpreting either value. Never open
  validation or test split files, prediction files, or metric files. The original
  source manifest attests to the split; this audit does not re-read held-out IDs.
- Read static item genre metadata for TRAIN items. Do not use user demographics,
  held-out interactions, title-derived information, or external user information.
- Write aggregate statistics and source hashes only; no user/item IDs, ratings,
  timestamp values, per-user histories, or pair-level results in output evidence.

## Predeclared descriptive statistics

For a user, an exact-time group is all retained TRAIN records with the same
integer timestamp. Report the total retained rows, users, groups, and adjacent
within-user event gaps; all ratios state their denominator.

- For exact-time group size thresholds 2, 5, and 10: users with a qualifying
  group, number of qualifying groups, and records in qualifying groups.
- Quantiles (0, .25, .50, .75, .90, .95, .99, 1) of all group sizes and of group
  sizes at least two, each weighting every group equally.
- The same quantiles of each user's largest exact-time group divided by their
  number of retained TRAIN records, weighting users equally.
- Counts and fractions of adjacent retained event gaps at most 1, 60, 300, and
  1,800 seconds, including zero gaps. Also report zero gaps and repeat those
  thresholds for positive gaps between distinct recorded timestamps, so ties
  cannot be mistaken for many independent close-together recording events.

## Genre check and deterministic null

For every unordered pair of TRAIN films within a user's exact-time group of size
at least two, calculate genre-set Jaccard intersection/union. An empty genre
union has similarity zero. Static `(unknown)` is retained as supplied metadata.
Report (a) the mean over all such pairs (pair weighted), and (b) the equal-user
mean of each eligible user's pair mean (macro user). Users with no tied pair are
excluded from both observed and shuffled genre summaries; count them explicitly.

Create 100 independently seeded shuffles using NumPy `default_rng(2026092900+r)`
for replicate `r=0,...,99`. Process users in lexicographically sorted identifier
order, and process each user's records in lexicographically sorted item-ID order.
For each replicate/user, permute that user's complete item/rating records among
their fixed timestamp slots. Thus items, ratings, timestamp multiplicities and
user-specific genre preferences remain fixed, while association between film
identity and recording slot is removed. Ratings travel with their items but are
not used in the genre statistic.

For both statistics report all 100 aggregate null values, null mean, standard
deviation (ddof=0), 2.5/50/97.5 percentiles, and observed-minus-null-mean. These are
descriptive randomization comparisons, **not hypothesis-test p-values**. Do not
select a statistic, time window, subset, or model from the observed result.

## Interpretation limits and checks

MovieLens timestamps record ratings, not watching. Same-second timestamps do not
identify a screen, browser action, causal influence, or a consumption session.
Genre coherence could reflect the recording interface, the user's organization
of memory, taste, or other causes. A random TRAIN split thins recording groups;
counts describe retained TRAIN records, not complete original submissions. This
audit alone cannot establish a recommendation improvement, a new method, or a
causal explanation.

Use the frozen environment with numerical threads set to one. Check the optimized
genre calculation against literal pair enumeration on synthetic histories;
check that poisoning rating/timestamp text outside TRAIN has no effect; and
check the input hash failure path. Save reproducible code, aggregate JSON,
human-readable findings, and SHA-256 digests. No new model or ranking run follows
automatically from this audit.
