# Learning an operation and choosing examples that reveal it

Our own exact finite implementation learns an anonymous Boolean operation and
task-specific programs from input/output examples. The supplied interpreter and
grammar remain explicit assumptions. No pretrained model or MovieLens data is
used. This is a small mechanism study, not a general intelligence architecture.

Read the [completed results](results-v1/RESULTS.md),
[standalone figure](results-v1/query-transfer.pdf), and
[prospective protocol](PROTOCOL.md). The information target has established
prior art, including [Sloman et al., UAI 2024](https://arxiv.org/abs/2310.14968v3).
Novelty of the objective is not claimed.

## What was learned

There are 16 possible four-bit operations and 18 possible programs per task.
The learner infers the operation's contents and the programs' input bindings
and composition. Exact Bayesian inference retains all supported hypotheses
during acquisition. Deployment freezes one concrete MAP operation and selects
one program from a new task's support examples; its output is not an ensemble.

Three query policies acquire information about the whole task, about the shared
operation, or randomly. After acquisition, the evaluator tests all unused
Boolean functions expressible by the declared grammar, excluding exact semantic
duplicates of training tasks. Every two-example and four-example support subset
is evaluated on its complement. Local support-only search and an oracle-table
reference use identical support sets.

All 160 shared training worlds and ten no-sharing training worlds completed.
The no-sharing worlds are each evaluated against all 16 transfer tables, giving
320 world/table cells and 4,800 arm/budget records. Those 160 no-sharing cells
still contain only ten independently generated training worlds.

Focused queries improve intermediate-budget transfer in the shared fixture;
ordinary information-seeking catches up at the largest budget. When tasks do
not share an operation, focused and random acquisition eventually falsify the
shared model in all ten training worlds. The learner then abstains, receiving
the declared loss of one. This is earlier detection of an incorrect assumption,
not 100% confidently incorrect predictions. The prototype has no repair or
adaptive-sharing mechanism. The result identifies that missing capability;
it does not demonstrate one.

## Verification and evidence

Seventeen core and runner tests pass. Independent checks compared the factored
posterior against all 93,312 joint states on twelve synthetic evidence sets,
checked all 2,304 interpreter outputs, and matched the vectorized transfer
evaluator to a literal calculation on 16,660 episodes. After the run, independent
reaggregation verified all 1,260 control-metric summaries and twenty paired
contrasts against the 4,800 records. Root verification also checked every query's
label, uniqueness and exact budget, and all source/output hashes.

The [evidence archive](results-v1/evidence.zip) contains every original synthetic
world, query trace, result record, summary and failure event, with the original
manifest and the sealed source/protocol copies. These are synthetic truth
tables, not user histories. [The archive index](results-v1/evidence-manifest.json)
records entry hashes and the archive digest. The figures retain all prescribed
budgets and support sizes; no confidence intervals or matched-compute claim are
made.

## Reproduce

From the final-project directory, using the existing Python environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python -m unittest \
  exploratory.library_queries.test_core exploratory.library_queries.test_runner

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python \
  -m exploratory.library_queries.run_experiment \
  --out runs/library-queries-replay

MPLCONFIGDIR=/private/tmp/library-query-matplotlib \
  python3 -m exploratory.library_queries.summarize \
  --run runs/library-queries-replay --out runs/library-queries-replay-report
```

Both output directories must be new. The runner hashes its source and protocol
before trials and checks they are unchanged on completion. Wall-clock fields
and run timestamps will differ on replay; predictions and numerical summaries
are deterministic for the recorded NumPy version. Plotting requires Matplotlib;
the training environment is NumPy 1.26.4 on Python 3.11.15. Neither command
installs dependencies or accesses real-data evaluation splits.
