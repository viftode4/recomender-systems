# Categorical reconstruction recommender

One exact ridge model reconstructs observed items from binary history and
TRAIN-centered categorical rating features. It removes every target's complete
six-feature block, so its own rating cannot copy into its prediction. Binary-only
mode is exactly EASE. The [derivation](DERIVATION.md) gives the equations,
numerical safeguards, and limits of the claim.

This study reuses the original development cohorts after the project's earlier
TEST evaluation. Its results are exploratory, not fresh held-out confirmation.
The [fixed protocol](PROTOCOL.md) separates the standalone model from the
assignment's secondary hybrid comparison.

## Environment and inputs

Run commands from the `final-project` repository root. The solver and experiment
runner use Python, NumPy, SciPy, and local project modules; they do not train a
RecBole model or require a GPU. The verified development environment is Python
3.11.15, NumPy 1.26.4, and SciPy 1.17.1 on macOS arm64. To create a small separate
environment:

```sh
python3.11 -m venv .venv-categorical-reconstruction
.venv-categorical-reconstruction/bin/python -m pip install numpy==1.26.4 scipy==1.17.1
```

The complete project's [training dependency snapshot](../../requirements.lock.txt)
is also available. Exact score replay is checked within the recorded runtime;
another operating system, numerical library, or Python version is not promised
to produce identical bytes.

The experiment deliberately builds on existing, hash-checked project artifacts:

- `runs/research-v2/{seed}-EASE-1/`: original TRAIN/VALID pairs, ordered IDs,
  saved score catalog and source manifest for seeds 2026, 2027, and 2028.
- `runs/adaptive-v1/{seed}/`: original meta-fit/development user-cohort records.
- `runs/field-reference-v1/` and `evidence/field-reference-v1/`: the locked
  development comparison, including EASE, SLIMElastic and PositiveEASE.
- `runs/coverage-v1/` and `runs/coverage-v1-more/`: the locked SLIM predictions
  for the secondary assignment hybrid (first seed and remaining seeds).
- The instructor dataset's `ml-100k.inter` and `ml-100k.item`, whose hashes must
  match the saved source manifests. Item genres support evaluation diagnostics;
  the standalone reconstruction model uses no item metadata.

Raw datasets and user-level runs are excluded from the shareable review archive.
A source-only checkout therefore needs these original local artifacts before the
full reproduction command can succeed. Do not replace missing source runs with
new random splits: that would be a different comparison. The runner's source,
split, identity, cohort and reference checks reject mismatched artifacts.

## Direct model API

```python
from exploratory.categorical_reconstruction.model import prepare_features, fit, load

# ratings: users × catalog, integer1..5 for TRAIN records, zero for absence.
prepared = prepare_features(ratings, smoothing=20.0)
model = fit(prepared, lambda_binary=100.0, category_ratio=1.0)
scores = model.predict()  # TRAIN histories; no seen-item or padding mask.
query_scores = model.predict(query_ratings)  # Uses fixed TRAIN centering.
model.save("selected-model.npz")
replayed_scores = load("selected-model.npz").predict(query_ratings)

binary = fit(prepared, lambda_binary=100.0, category_ratio=None)
```

Feature preparation caches the binary and categorical user-space kernels for
grid reuse. Normal fitting stores the sparse reference features and user-by-item
dual coefficients, rather than a feature-square inverse. `coefficient_matrix()`
is available for small independent mathematical checks. `predict_unadjusted()`
is valid only at target-absent positions; use `predict()` for general histories.

The saved source catalogs have 943 user rows and 1,683 item columns including
padding, hence 10,098 categorical feature columns. The primary dense factorization
is 943 by 943 rather than 10,098 by 10,098: about 115 times fewer matrix entries
(6.8 MiB versus 778 MiB in float64). These figures describe the matrix alone,
not total process memory or a measured speedup. Sparse features, dual coefficients,
score arrays, and concurrent worker processes require additional memory.

The runner owns data joins, source hashes, fixed model selection, candidate
masks, and metrics. The model has no access to labels outside its supplied TRAIN
matrix. No fit quality or superiority is asserted by the existence of this
implementation.

## Checks and bounded search

Run the independent algebra tests in an environment with NumPy and SciPy:

```sh
.venv-categorical-reconstruction/bin/python -m unittest \
  exploratory.categorical_reconstruction.test_model -v
```

The declared grid has nine binary penalties and four categorical/binary penalty
ratios, with smoothing fixed at 20. Each seed runs 9 binary, 36 true-category, and
36 within-item shuffled-category fits: 81 distinct fits per seed, 243 overall.
Each categorical selection family also includes the same nine cached binary
candidates as a fallback. Thus a categorical family has 45 selection options
versus nine for the expanded binary control; report that unequal tuning budget.
Selected-fit replay and the three fixed hybrid comparisons are additional work.

The runner permits one, two, or three worker processes and fixes numerical
library threads to one per process. Three workers parallelize seeds; one worker
uses less concurrent memory. Use a fresh output directory for every run. The
runner refuses to overwrite an existing result tree. Runtime guidance will use
recorded timings rather than matrix dimensions alone.

The initial TRAIN-only timing used seed 2026, one numerical thread, the macOS
arm64 environment above, and `lambda_binary=250`. Feature preparation took
0.134 seconds. The measured model operations were:

| Model | Fit | Predict all TRAIN histories | SVD fallbacks |
|---|---:|---:|---:|
| Binary-only | 0.193 s | 0.115 s | 0 |
| Categorical, ratio 1 | 0.414 s | 0.202 s | 0 |

The maximum checked reduced-system residual was approximately `5.1e-15`.
These are individual measurements recorded in
`runs/categorical-reconstruction-train-timing-v1.json`, with no validation
evaluation. They exclude full-grid selection, file I/O, selected-model replay,
hybrid fitting, development evaluation, and worker contention. They are not a
hardware-independent latency claim or a measured end-to-end study duration.

## Reproduce the study

Set `CR_DATA` to your instructor checkout's MovieLens directory. The example
uses the checkout location from the repository's setup instructions; an existing
instructor checkout elsewhere is equally valid if the source hashes match.

```sh
CR_DATA=vendor/RecBole_DSAIT4335/dataset/ml-100k

.venv-categorical-reconstruction/bin/python -m exploratory.categorical_reconstruction.run_experiment run \
  --source-root runs/research-v2 \
  --cohort-root runs/adaptive-v1 \
  --reference-run runs/field-reference-v1 \
  --reference-evidence evidence/field-reference-v1 \
  --slim-root runs/coverage-v1 \
  --slim-more-root runs/coverage-v1-more \
  --ratings "$CR_DATA/ml-100k.inter" \
  --item-metadata "$CR_DATA/ml-100k.item" \
  --seeds 2026 2027 2028 --workers 3 \
  --out runs/categorical-reconstruction-reproduction \
  --evidence exploratory/categorical_reconstruction/results-reproduction
```

This performs the declared search, verifies deterministic selected-model replay,
seals all choices across all three seeds, evaluates the reused development users,
and writes aggregate review evidence. It never opens TEST files. The shuffled
control uses shuffled TRAIN ratings for both fitting and querying; only original
rating values define the later evaluation endpoints.

A separate optional TRAIN-only timing command prepares features and fits one
binary and one categorical model at `lambda=250`, categorical ratio 1. It does
not compute validation metrics or alter the declared search:

```sh
.venv-categorical-reconstruction/bin/python -m exploratory.categorical_reconstruction.run_experiment benchmark \
  --source runs/research-v2/2026-EASE-1 \
  --ratings "$CR_DATA/ml-100k.inter" \
  --out runs/categorical-reconstruction-benchmark-reproduction.json
```

To regenerate only the aggregate bundle from an already completed run, use a
new destination. This checks the recorded source/runtime and output hashes and
does not repeat fitting or development evaluation:

```sh
.venv-categorical-reconstruction/bin/python -m exploratory.categorical_reconstruction.run_experiment curate \
  --research runs/categorical-reconstruction-reproduction \
  --out exploratory/categorical_reconstruction/results-reproduction-copy
```

Independently verify a completed run and bind the exact curated files used by the
report. The verifier refuses to read development labels before the global
selection seal and complete run manifest exist. It checks all 243 grid rows and
selection ties, refits the selected configurations and original binary grid,
reconstructs the hybrid calibration and scores independently, and recomputes
development ranking metrics. It does not refit every grid candidate.

For the completed local study, using its verified numerical environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
runs/environment-check/.venv/bin/python exploratory/categorical_reconstruction/verify_results.py \
  --research runs/categorical-reconstruction-v1 \
  --data-path "$CR_DATA" \
  --evidence exploratory/categorical_reconstruction/results-v1 \
  --out exploratory/categorical_reconstruction/audit-reproduction
```

The `--evidence` argument is essential for report verification: it binds
`aggregates.json`, `protocol.json`, and `provenance.json` directly in the audit.
Use a new audit destination. The completed evidence-bound audit is
[`audit-v2/audit.json`](audit-v2/audit.json); `audit-v1` is the earlier raw-run
verification without curated-file bindings. Both preserve the distinction
between numerical tolerance and exact ranking agreement with older float32
EASE exports. These checks verify reused-development evidence, not a new test.

## Output and sharing boundary

The original local result root is `runs/categorical-reconstruction-v1/`; curated
review evidence is `exploratory/categorical_reconstruction/results-v1/`.
Detailed runs retain source identities, TRAIN categories, selected score arrays,
per-user development metrics, solver diagnostics, and selection manifests.
Keep these under ignored `runs/`; do not include them in a shareable report or ZIP.

The run records `protocol.json` before fitting and seals every seed's standalone
and hybrid choices in `SELECTIONS-FROZEN.json` before any development evaluation.
`DEVELOPMENT-OPENED.json` marks that later evaluation stage. Completed results
contain `aggregates.json` and a hash manifest. These boundaries prevent this run's
development metrics from selecting its models; they cannot undo the project's
earlier reuse of the same dataset.

The curated directory contains `RESULTS.md`, `aggregates.json`, `protocol.json`,
`provenance.json`, and `SHA256.json`. It includes cohort sizes, aggregate metrics,
configuration and timing summaries, and provenance hashes. It excludes user IDs,
individual histories, recommendations, raw score matrices, and item-ID lists.
Source hashes identify the implementation used; they are not evidence of a new
scientific principle or of stronger ranking performance.
