# Shared evidence interpreter

This custom standalone predictor tests whether the structure of supporting
histories adds ranking information beyond support amount and popularity. One
177-parameter scoring function is shared across every movie and user; no expert
scores or learned identity embeddings enter it. It is a small explicit test of
this mechanism, not an arbitrary learned reasoning system or an established
breakthrough.

- [Design and competing explanations](DESIGN.md).
- [Declared protocol](PROTOCOL.md) and [feature/model equations](DERIVATION.md).
- [Closest prior work](RELATED_WORK.md), including the existing diversity measure.
- [Measured results](results-v1/RESULTS.md) and [declared pilot comparison](analysis-v1/RESULTS.md).
- [Interpretation and remaining gap](FINDINGS.md) and [optimization diagnosis](OPTIMIZATION.md).
- [Independent replay](audit-v1/RESULTS.md).
- [Synthetic capacity check](synthetic-capacity-v1/RESULTS.md): a constructed
  pattern distinction is learnable, which alone says nothing about its usefulness
  in real recommendation data.

The user requested original, substantial advances. This study is one attempt to
identify a useful mechanism toward that goal. The goal is not satisfied merely
by implementing a model, passing tests, or meeting coursework requirements.

## Reproduce

From the final-project root, using the existing frozen numerical environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 runs/environment-check/.venv/bin/python -m unittest discover -s exploratory/evidence_transfer -p 'test_*.py' -q
```

The recorded pilot command is:

```sh
runs/environment-check/.venv/bin/python -m exploratory.evidence_transfer.run_experiment run \
  --ratings /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --item-metadata /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --source-root runs/research-v2 --cohort-root runs/adaptive-v1 \
  --reference-run runs/categorical-reconstruction-v1 --seeds 2026 2027 2028 \
  --out runs/evidence-transfer-v1 --evidence exploratory/evidence_transfer/results-v1
```

For a replay, supply new output directories. The runner refuses existing study
directories and checks the source/data/runtime signatures. It requires the
recorded earlier experiment artifacts; this is not a one-command fresh-data
installation. Reproduction instructions for the original pipeline are in the
project README. No new dependency download is needed in the current environment.

All individual records, episode masks, model states and predictions stay in
ignored `runs/`. Shareable directories contain aggregates and hash provenance.
Checkpoint selection is performed for every data split before new development
outcomes are calculated. The development data was examined in earlier studies;
the original TEST is already exposed. This experiment does not restore fresh
confirmation by creating new masks or hashes.

The controlled full/no-pattern comparison removes both pattern count and
coverage together. The inference-only zero-pattern intervention changes inputs
without retraining and has a different interpretation. The comparison with EASE
also changes representation, objective and fitting procedure, so a performance
difference there cannot alone isolate why it occurs.
