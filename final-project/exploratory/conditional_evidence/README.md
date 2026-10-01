# Conditional evidence study

The implemented question is simple: **does the arrangement of other people's
observations contain useful evidence that our previous summaries discarded?**
The custom reader sees a small rating graph for each candidate. If that reader
beats both controls and the strongest reference on selection data, a second
model learns which single additional person's history would actually improve its
prediction. This is an experiment, with no performance or novelty claim yet.

The [protocol](PROTOCOL.md) fixes the hypothesis, data boundaries, architecture,
budgets, comparisons and decision rules. The [independent review](REVIEW.md)
records the numerical and leakage checks. Earlier coursework reports and packages
are preserved; this folder is a separate research supplement.

The full study was launched on 29 September 2026 with six workers. Initial
checkpoints and completed 1,886-episode training epochs were verified. All 91
tests passed, and 341 protected historical files were unchanged at launch.
See [implementation verification](IMPLEMENTATION-VERIFIED.json) and the live
progress files described below. A launch is not a completed scientific result.

## What runs

| Part | Implementation |
|---|---|
| Raw relationship reader | 5,553 learned parameters; three rating-specific graph layers; no identity embeddings |
| Matched structural control | Same reader, with rating margins preserved while donor/history correspondences are scrambled |
| Summary control | Nine old evidence summaries, computed from the same donors; 177 parameters |
| Selection | Three splits, two learning rates, all three arms: 18 trajectories |
| Training | 300 epochs, automatically extended to 600 for all trajectories if any best checkpoint reaches 300 |
| References | Previously locked, provenance-verified EASE, categorical reconstruction, SLIM; fixed cosine neighbor model |
| Conditional acquisition | Three-fold refits; utility labels are actual full-catalog TRAIN loss reductions; at most one added donor |
| Assessment | All choices and optional acquisition predictions sealed before development outcomes are read |

RAG contributes the idea of accessible evidence at prediction time; Jev contributes
the bounded decision interface. This implementation uses our own local model,
without external model APIs or pretrained weights. Local rating-graph prediction
has prior art, including IGMC; calling the implementation a first-ever architecture
would be unsupported. The [research notes](../jev_rag/RESEARCH.md) give context.

## Run and recover

Commands below run from `final-project/`, using the existing personal environment.
It contains Python 3.11.15, NumPy 1.26.4 and Torch 2.14.0. Each process uses one
numerical thread; six independent trajectories can run together.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
runs/environment-check/.venv/bin/python -u -m exploratory.conditional_evidence.workflow \
  --out runs/conditional-evidence-v1 --workers 6
```

After an interruption, use the identical command with `--resume`. The OS lock
rejects a second workflow for the same output. Checkpoints preserve Adam, RNG
state and the completed epoch; at most one current epoch per worker is repeated.
Source, input, compiler/library and runtime mismatches reject a resume. Keep this
source and environment unchanged while a sealed study is running. Machine sleep
pauses progress; reboot or process termination requires the resume command.

```sh
cat runs/conditional-evidence-v1-progress.json
tail -n 4 runs/conditional-evidence-v1.log
```

The progress JSON records the current pipeline stage. The log records completed
epochs and selection checkpoints. `runs/conditional-evidence-v1-launch.json`
records the actual launch command and PID when the background study is started.
Private checkpoints, arrays and per-user diagnostics remain under ignored `runs/`.

## Cost measured before fitting

The TRAIN-only pilot used 16 episodes and two full-catalog users for each arm.
These are extrapolations from a small timing sample, not guaranteed completion
times or recommendation results.

| Arm | Estimated seconds/epoch | Estimated hours/300-epoch trajectory | Pilot peak process memory |
|---|---:|---:|---:|
| Raw | 149.3 | 12.44 | 506 MiB |
| Scrambled | 288.9 | 24.08 | 509 MiB |
| Summary | 27.8 | 2.32 | 354 MiB |

Including selection checks, all 18 base trajectories total roughly 241 process
hours. Six workers suggest approximately two days of elapsed time, with variation
from history sizes and contention. The 600-epoch extension roughly doubles the
training. Gated retrieval adds nine fold fits and full-catalog utility-label
construction. Detailed pilot records are under `runs/conditional-benchmark-v1-*`.
The deterministic C scrambling kernel is checked against the Python reference;
the fallback has identical output but is slower.

## Outputs and interpretation

The workflow automatically performs training, global selection, gated retrieval,
assessment and report export. On completion, `results-v1/` contains:

- `RESULTS.md` and `aggregates.json`: every arm and seed, primary nDCG@10,
  recall, MRR, liked-rating and activity/head/tail diagnostics, paired differences.
- `compute-and-curves.json`: all training/selection curves, resource counts,
  process memory and historical reference search budgets.
- `protocol.json`, `provenance.json`, `SHA256.json`, `RESULTS-SHA256.json`:
  experiment and artifact bindings.
- If acquisition triggers: policy results, TRAIN oracle opportunity, compute
  totals and aggregate fold-to-full-reader feature-distribution changes.

MovieLens 100K development data was used in earlier research. Its three splits
overlap; a result here is exploratory. This study never opens the original TEST.
The substantial-result target is at least 10% relative mean nDCG improvement over
the reference chosen on selection data, positive in all three splits and above
both controls on average. Passing it triggers a separate reserved MovieLens 1M
confirmation protocol; that dataset acquisition and independent study are not
implemented or claimed complete here. `confirmation-status.json` makes this
remaining condition explicit.

Raw versus scrambled is the matched architecture comparison. Raw versus summary
also changes available information and parameter count. Utility representations
can differ between independently trained fold readers and the full reader even
with common initialization. Those limitations remain when tests pass.

## Verify

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
runs/environment-check/.venv/bin/python -m unittest discover \
  -s exploratory/conditional_evidence -t . -p 'test_*.py' -v
```

Tests cover literal graph calculations, masked-row exclusion, exact scrambling
parity, sampled-loss derivatives, full-catalog ranking, crossfit isolation,
interrupted/resumed optimizer replay, source/selection barriers, retrieval
deployment and complete output recovery. Synthetic results and timing pilots are
never mixed with scientific outcome tables.
