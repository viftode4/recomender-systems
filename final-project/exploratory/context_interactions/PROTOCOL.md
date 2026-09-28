# Does independent interaction capacity improve recommendation?

This is a prospective specification for one additional experiment on the
project's already reused MovieLens 100K development cohorts. The original TEST
has been exposed. These results cannot establish fresh confirmation, a universal
advantage, or conceptual novelty. Previous sealed studies stay unchanged.

## Question and mechanism

The latest categorical reconstruction model sums fixed contributions from
individual source items. The hypothesis here is that independently weighted
conjunctions of distinct history items can improve observed-record ranking.
The previous addressed-evidence model tied pair coefficients through its unary
parameters and generalized poorly as pair-logit magnitude grew. That failure
does not test independently regularized pair coefficients under reconstruction.

Let X be the TRAIN observation matrix, including its empty PAD column, and
G = X X^T. All distinct binary pair features have Gram matrix
P = (G elementwise-squared - G) / 2. Set
nu = sum_u choose(history_size_u, 2) / sum_u history_size_u;
use nu=1 when this ratio is zero or there are no observations. The kernel is
K = (G + beta P/nu)/lambda. This is one jointly fitted ridge predictor with
independent linear and pair coefficients, not a mixture of fitted model scores.
The scale nu is one fixed TRAIN statistic, never tuned on labels.

For target i remove its unary feature and every pair involving i from both
reference and query features. The model solves ridge reconstruction of X[:,i]
with that reduced feature map. A shared user-space factorization and support-
sized corrections make the complete pair dictionary implicit. beta=0 must
reproduce EASE; a direct expanded-feature primal solve on small data must
independently verify both fitted coefficients/scores and target exclusion.

The formulation is an application of known polynomial kernels and constrained
ridge regression. The experiment tests this precise mechanism and solver;
it does not claim to invent higher-order recommendation.

## Inputs, grid and selection

- Original TRAIN/VALID partitions and catalog ordering for seeds 2026, 2027,
  2028 from `runs/research-v2/{seed}-EASE-1`.
- Original 471 meta-fit / 472 development user cohorts from `runs/adaptive-v1`.
  Verify source, split, cohort and ordered-identity hashes.
- Predictor inputs are binary TRAIN identities only. Rating values and item
  genres support secondary evaluation, not training or model selection.
- Lambda grid, in order: 10, 30, 50, 100, 250, 300, 1000, 3000, 10000.
- Pair strengths, in order: 0.01, 0.1, 1, 10. Fit the nine beta=0 models first,
  followed by all 36 positive-strength configurations in the declared runner
  order. The complete ordered candidate list is serialized before fitting.
- 45 distinct fits per seed; 135 total. Cached binary fits are reused.
- Select three model roles: `binary` from nine candidates, `pair_only` from
  36 candidates, and `nested` from all 45. Exact maximum meta-fit all-observed
  nDCG@10 wins; exact ties retain the first declared candidate.
- Report all three roles and tuning budgets. A binary fallback must remain
  visible; a larger selection family alone does not isolate representation.
- Save all candidates and numerical failures. A failed candidate receives no
  replacement parameters. A failure that invalidates selection stops the run.
- Save predictions, selections, source/runtime and input hashes for every seed.
  Seal all three seeds together before any new development metric is computed.
  Source changes after sealing are rejected. Fresh output directories only.

TRAIN-only runtime benchmarking and synthetic correctness tests may precede
the run. They cannot use validation metrics to alter this grid. Changes prompted
by implementation or runtime constraints must be recorded before actual fits.

## Evaluation and comparisons

Use full-catalog ranking, k=10, exclude original TRAIN items and PAD, preserve
the established deterministic tie rule. Scores use original TRAIN histories;
no VALID records are added as inference context. Every held-out VALID rating
is relevant for the primary all-observed endpoint. Ratings >=4 define the
separate liked-record endpoint; disclose excluded users without a held-out like.

Report nDCG, precision, recall, reciprocal rank, hit rate, coverage, novelty,
known-dislike rate, and the existing TRAIN-defined activity and head/tail item
group analyses. Group definitions and metrics do not select models. Compare
pair_only and nested with binary, and replay the already locked categorical
model as a contextual reference after verifying identical cohorts and inputs.
Replaying that reference is not a new selection or an equal-budget experiment.

Preserve per-seed results and paired descriptive user differences. Overlapping
splits and reused development do not provide independent confirmation. Do not
pool them into a new significance claim. There is no additional hybrid search
in this experiment: it tests a standalone information-processing mechanism.

Independent verification must recompute rankings and aggregate metrics from
the saved selected predictions, check candidate completeness/selection ties,
validate the global barrier, and check the exact binary control against the
previous study. Full selected-model replay is required where practical.

## Delivery and interpretation

Raw histories, predictions and detailed records remain in ignored
`runs/context-interactions-v1`. Curated source, protocol, aggregate results and
audits belong under this new exploratory directory. No old report or ZIP is
replaced and no course submission is performed by this experiment.

A gain supports this mechanism on this dataset and protocol. A null result
only rejects these configurations as a route to substantial improvement; it
does not establish an information-theoretic ceiling. Missing exposure, intent
or satisfaction measurements remain separate hypotheses. No computation on
these logs can manufacture observations of those missing variables.
