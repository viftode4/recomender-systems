# Prediction from recording groups

This finite exploratory experiment compares singleton-plus-recording-group pair
reconstruction against the same pair dictionary over unordered histories, three
independently shuffled recording partitions, and independently tuned EASE. The
[protocol](PROTOCOL.md) defines all 39 fits, the nested original-TRAIN split, the
selection barrier, and the assessment endpoints before fitting.

The completed [aggregate results](results-v1/aggregates.json) are negative for the
proposed mechanism: primary assessment nDCG@10 is 0.172555 for true recording
groups, 0.172721 for bag pairs, 0.172819 for the mean shuffled-model metric, and
0.172956 for tuned EASE. True grouping is 0.232% relatively below EASE; both the
predeclared improvement target and descriptive mechanism-support check fail.
The dataset's recording coherence does not by itself establish a useful
predictive contribution from this pair mechanism. All 39 fits completed in
38.56 seconds, with no additional sweep after assessment.

The single conceptual change is that two visible ratings with equal timestamps
can contribute a pair feature, while ratings with different timestamps do not. The
underlying method is constrained ridge reconstruction with singleton and pair
features, not a new claim about pair kernels. Timestamps mark rating submission;
equal values do not establish a shared consumption context or exact screen.

Only original TRAIN pair IDs are partitioned into F/D/A. Before the all-arm seal,
only F timestamps are parsed; D item identities choose hyperparameters. All
models are scored with the same F context. After the seal, A ratings/timestamps
are evaluator-only diagnostics. The full original 1,682-item catalog is ranked,
with PAD and F observations excluded. No original VALID/TEST file is read.

From `final-project`, run the tests:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python -m unittest discover \
  -s exploratory/framing_search/predictive -p 'test_*.py' -v
```

Then run the declared experiment into paths that do not already exist:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python -m exploratory.framing_search.predictive.run \
  --source runs/research-v2/2026-EASE-1 \
  --signature runs/categorical-reconstruction-v1/2026/input-signature.json \
  --train runs/categorical-reconstruction-v1/2026/train.tsv \
  --ratings /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --metadata /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --run-dir runs/framing-predictive-v1 \
  --evidence exploratory/framing_search/predictive/results-v1
```

Source, protocol, runtime, and input hashes are recorded before fitting. Every
selected score array and every choice is sealed before assessment; the guard
requires all 39 candidates and verifies the first exact development maximum for
every role. The original dataset and ignored previous split artifacts are local
replay prerequisites. Public results contain aggregates and cryptographic digests;
identities, split rows, timestamps, score arrays, and per-user metrics remain in
ignored `runs/`.

The three shuffled models are never prediction-ensembled. The control comparison
averages their per-user evaluation metrics. Paired bootstrap intervals describe
this explored dataset; the nested assessment is not a fresh population test.

The initial launch stopped at input dimensions before splitting, timestamp parsing
or fitting because the original user catalog includes one `[PAD]` entry. The
loader was corrected to remove that explicit padding user, preserving all 943
real users in original order and the complete item catalog. Tests were rerun and
the corrected source was frozen before the completed launch. No experimental
setting or candidate was changed in response to a ranking result.
