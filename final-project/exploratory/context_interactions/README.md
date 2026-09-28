# Context interactions through exact kernel reconstruction

This standalone predictor adds independently weighted pairs of history items
to a linear binary reconstruction model. Every target is excluded from both
singleton and pair sources. It uses no other recommender's predictions.
The [derivation](DERIVATION.md) explains the exact kernel and efficient solver;
the [protocol](PROTOCOL.md) defines the prospective exploratory comparison.

The intended comparison asks whether interactions improve over a strong linear
control with the same inputs and training objective. Pair strength zero is
exactly EASE. Higher-order recommendation and polynomial kernels are established
methods; this is an independent implementation and controlled experiment, not
a claim of first-ever architecture.

## Model API

Run from the repository root in Python 3.11 with NumPy and SciPy. The measured
environment uses Python 3.11.15, NumPy 1.26.4 and SciPy 1.17.1 on macOS arm64.

```python
from exploratory.context_interactions.model import prepare_features, fit

# TRAIN observation identities, users × catalog; entries must be zero or one.
prepared = prepare_features(training_ratings != 0)
model = fit(prepared, lambda_binary=250.0, pair_weight=1.0)
scores = model.predict()  # All TRAIN histories, complete unmasked catalog.
query_scores = model.predict(query_binary_histories)
linear = fit(prepared, lambda_binary=250.0, pair_weight=0.0)
```

The TRAIN normalizer balances total singleton and pair feature energy. Query
histories never update it. Rating values, genres, validation outcomes and other
model scores are not predictor inputs. Scores are reconstruction values, not
probabilities. Candidate eligibility and tie-breaking belong to the runner.

## Correctness and runtime

```sh
runs/environment-check/.venv/bin/python -m unittest \
  exploratory.context_interactions.test_model -v
```

The core tests compare against independent explicit-feature ridge solutions,
including arbitrary query histories and removal of every pair involving the
candidate. They also verify EASE equivalence, cold/empty cases, deterministic
replay, and elementary conjunction/XOR representation capacity. The latter
is a correctness check, not a recommendation benchmark.

The initial TRAIN-only timing is recorded under
`runs/context-interactions-train-timing-v1.json`. It used seed 2026, one numerical
thread, 943 users and 1,683 catalog columns. Feature preparation took 0.041 s.

| Binary penalty | Pair strength | Fit | Full-history prediction | Fallbacks |
|---:|---:|---:|---:|---:|
| 250 | 0 | 0.292 s | 0.175 s | 0 |
| 250 | 1 | 0.792 s | 0.172 s | 0 |
| 10 | 10 | 0.787 s | 0.172 s | 0 |

The largest checked relative residual was `2.16e-14`. These individual timings
exclude model selection, data joins, score persistence, independent replay,
development evaluation, and concurrent-worker contention. They do not measure
the complete study's elapsed time or establish hardware-independent latency.
The implementation represents 1,415,403 possible pair columns implicitly,
including identically zero PAD-derived columns. That is dictionary size, not
the number of independently estimable parameters.

## Experiment boundary

The declared search has nine penalties and four positive pair strengths plus
the zero-strength linear control: 45 distinct fits per seed, 135 for three seeds.
The runner reports the nine-candidate linear family, 36-candidate pair-only
family, and 45-candidate nested family separately. It freezes all selections
before development evaluation; unequal selection budgets remain a limitation.

Detailed outputs belong under ignored `runs/context-interactions-v1/`. Curated
outputs contain aggregate metrics and provenance, not user IDs, histories,
recommendations or raw prediction arrays. The original MovieLens TEST was
already examined before this exploratory study was designed. No new result on
these reused development cohorts constitutes fresh confirmation, and the
experiment does not read TEST data or change earlier sealed results.
