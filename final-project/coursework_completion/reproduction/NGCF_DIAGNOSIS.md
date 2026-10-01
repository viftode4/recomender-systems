# NGCF reproduction diagnosis

The fresh rebuild is not numerically identical to the historical evidence. NGCF
changes the top-ten item sets for 148, 101 and 146 of 943 users in seeds
2026/2027/2028, respectively. Its maximum score differences are 0.2162, 0.2769
and 0.3870. The other 42 rebuilt expert runs retain the historical top-ten order.
The [score comparison](source-scores-replay-v1.json) records the exact scope.

The historical and rebuilt NGCF checkpoints have the same training settings,
package versions, model source and epoch 59. Their saved effective configurations
differ only in the output checkpoint directory. Learned weights differ, so this
is not merely a change in inference or score-file serialization. Node dropout
is zero. Their first logged training losses are nearly equal before diverging.

We ran a bounded four-epoch diagnostic with seed 2026, identical TRAIN data and
identical initialized parameter hashes. No settings were selected using relevance:

| Thread environment | PyTorch threads | Epoch 0 loss | Epoch 3 loss |
|---|---:|---:|---:|
| Thread variables unset, first run | 12 | 18.320919007 | 11.507607847 |
| Same environment, independent repeat | 12 | 18.320947975 | 11.508308649 |
| All five thread limits set to one | 1 | 18.320917398 | 11.507430941 |
| Only OMP limit set to one | 1 | 18.320917398 | 11.507430941 |
| Portable probe, all limits one | 1 | 18.320917398 | 11.507430941 |

All runs used 16 inter-op threads. The three one-thread probes match in every
recorded loss and every final parameter hash. Their float32 TensorBoard losses
also match the first four epochs of the fresh 60-epoch rebuild. The two
12-thread runs differ despite identical seeds, settings and initial parameters.
This demonstrates nonrepeatable multithreaded CPU training in the current NGCF
execution. It does not isolate the particular numerical kernel responsible.

Historical thread settings were not recorded. Parallel numerical sensitivity is
a supported explanation for the drift, but its exact historical cause cannot be
certified. Four repeatable epochs do not establish full 60-epoch determinism.
We have not recovered, replaced or altered the historical score arrays. There was
no search for a thread configuration that improves accuracy or matches a desired
historical result. The rebuilt downstream comparison remains a measured numerical
reproduction with disclosed differences, not an exact replay of every score.

The aggregate [diagnostic receipt](NGCF_DIAGNOSIS.json) binds the original/rebuilt
manifests and checkpoints, probe source, private probe results and score audit.
No user identifiers, histories, predictions or parameter values are included.

## Repeating the bounded probe

From the extracted archive's `code/` directory, with the environment and pinned
data described in [REPRODUCE.md](../../REPRODUCE.md):

```sh
env -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_NUM_THREADS \
  -u VECLIB_MAXIMUM_THREADS -u NUMEXPR_NUM_THREADS \
  .venv/bin/python coursework_completion/reproduction/ngcf_probe.py \
  --instructor-checkout vendor/RecBole_DSAIT4335 \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --out runs/ngcf-default-a --epochs 4

env -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_NUM_THREADS \
  -u VECLIB_MAXIMUM_THREADS -u NUMEXPR_NUM_THREADS \
  .venv/bin/python coursework_completion/reproduction/ngcf_probe.py \
  --instructor-checkout vendor/RecBole_DSAIT4335 \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --out runs/ngcf-default-b --epochs 4

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python coursework_completion/reproduction/ngcf_probe.py \
  --instructor-checkout vendor/RecBole_DSAIT4335 \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --out runs/ngcf-single-a --epochs 4
```

Repeat the last command with `runs/ngcf-single-b` to compare independent one-thread
processes. Every output path must be new. Compare `result.json` loss arrays and
initial/final parameter hashes, not wall-clock timings or checkpoint-file bytes.
Unset-thread defaults depend on the machine; inspect the recorded thread counts.
The probe reproduces the diagnostic procedure, not a guaranteed numerical result
on other hardware or libraries.

The standard splitter processes the raw dataset to build its partition. The probe
verifies the exact historical TRAIN digest and consumes only TRAIN after that
partitioning; it never iterates or evaluates validation/test loaders. Private
checkpoints and embedding files produced under `runs/` must not be redistributed.
