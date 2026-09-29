# Recording-bundle diagnostic

This TRAIN-only descriptive check tests whether MovieLens rating records have
same-user temporal grouping and whether films recorded at exactly the same time
are genre-coherent beyond a within-user shuffle. It does not establish why that
grouping occurs, a new algorithm, or a recommendation gain. Timestamp semantics
are rating recording, not watching.

The [protocol](PROTOCOL.md) was written before timestamp values were inspected.
All inputs must match the original frozen split/hash chain. Only TRAIN values
are parsed; no validation/test split, prediction, or metric file is opened.

From `final-project`, run tests with the existing environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python -m unittest exploratory.framing_search.recording_audit.test_audit -v
```

Replay the audit into a **new** output directory (change the final path if the
recorded output exists):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python -m exploratory.framing_search.recording_audit.audit \
  --source runs/research-v2/2026-EASE-1 \
  --signature runs/categorical-reconstruction-v1/2026/input-signature.json \
  --train runs/categorical-reconstruction-v1/2026/train.tsv \
  --ratings /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --metadata /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --output exploratory/framing_search/recording_audit/results-v1
```

`aggregates.json` contains the aggregate recording counts and all 100 aggregate
shuffle results. `provenance.json` records source/protocol/input digests and the
runtime. `SHA256.json` seals the output files. The raw dataset and original frozen
split artifacts are existing local prerequisites; they are not redistributed.

The first launch used `2026-EASE-0` and stopped at the source-manifest hash check
before any timestamp access. Hashing the original seed-2026 source manifests
identified `2026-EASE-1` as the source referenced by the frozen categorical input
signature. The source path was corrected before the completed launch; descriptive
statistics, shuffle design, and all analysis choices were unchanged.

The completed `results-v1` code and protocol are frozen. Input digests were
verified before parsing; the code/protocol digests were recorded after the
calculation. This is not a protected before/after snapshot of every input and
source file during execution. Independent current-hash verification checks the
saved artifacts against their recorded versions. The synthetic tests and actual
data include tied pairs and positive time gaps; the report renderer is not
generalized to an entirely untied or zero-positive-gap dataset. Neither edge
case occurs in this run (66,203 tied pairs and 43,373 positive time gaps).
