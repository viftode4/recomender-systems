# Reproducing the supplied evidence

Run these commands from the extracted archive's `code/` directory. All paths
below are relative to that directory or supplied explicitly. The source archive
contains no ratings, personal histories, predictions, model checkpoints, or
author-specific absolute paths required by the command line.

## Environment and instructor data

Use Python 3.11 with `requirements.lock.txt` and the instructor checkout pinned
to commit `081c3f6edf8e466d3ed5e163631a1afb6fe892bf`. An existing offline checkout
is sufficient; the bootstrap performs no network access, package installation,
or Git operation. Its default split-only stage performs no model training; the
explicit completion stage below retrains the declared experts. The [setup guide](docs/SETUP.md) documents environment
installation. The lock records the tested macOS environment; portability to
another operating system is not asserted merely because versions are pinned.

In the commands below, `vendor/RecBole_DSAIT4335` is the local instructor checkout,
and `.venv/bin/python` is an environment containing the recorded dependencies.
Both may be replaced with explicit local paths. Set numerical thread limits:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The bootstrap verifies all three original MovieLens atomic-file digests and the
259-file RecBole Python/YAML source tree before loading it. It refuses a changed
dataset or source tree. It also refuses every existing output directory.

## Exact original TRAIN prerequisite, without old run artifacts

```sh
.venv/bin/python -m unittest coursework_completion.test_bootstrap -v
.venv/bin/python -m coursework_completion.bootstrap \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --instructor-checkout vendor/RecBole_DSAIT4335 \
  --output-root runs/rebuilt-inputs
```

This invokes the pinned RecBole split with the original seed-2026 configuration.
RecBole processes the complete raw dataset to build its partition. The bootstrap
exports only the 80,808 TRAIN pair identities and the known catalog; it does not
iterate, export or evaluate the original validation/test loaders. No historical
model training is needed for this stage.

It requires the regenerated TRAIN SHA-256 to equal
`f4792fba583e3c01a26482f5fe0df62e02738cd8814672eaba32cb136c2fb7ec`
and the ordered catalog JSON SHA-256 to equal
`780252f4f0511dea952a45628c94dc1f6f8d357fa1064ed66ed1044d960a0cb6`.
A mismatch stops the run rather than substituting a different split.

The output includes an honestly labelled split-only source manifest and a
compatible input signature. These satisfy the frozen grouping runner's input
checks; they do not impersonate a historical EASE fit or categorical-model run.
Their provenance hashes differ from the original historical artifacts, while
the scientific TRAIN identities and catalog match exactly.

## Recording-group study: all 39 fits

```sh
.venv/bin/python -m unittest discover \
  -s exploratory/framing_search/predictive -p 'test_*.py' -v
.venv/bin/python -m exploratory.framing_search.predictive.run \
  --source runs/rebuilt-inputs/source-seed2026 \
  --signature runs/rebuilt-inputs/input-signature.json \
  --train runs/rebuilt-inputs/source-seed2026/train.tsv \
  --ratings vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --metadata vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --run-dir runs/replayed-grouping \
  --evidence runs/replayed-grouping-evidence
```

The scientific source and protocol are unchanged. The runner partitions original
TRAIN into F65,518/D7,645/A7,645 before parsing timestamp values, fits the declared
39 candidates, seals all six selections, and then evaluates assessment outcomes.
Original validation/test records are never model-selection or assessment labels
in this study. Output timing, paths, bootstrap provenance and associated hashes
will naturally differ; the selected candidates and scientific metric values are
the appropriate numerical replay comparison.

The original result was negative: true-group nDCG@10 0.172555, bag 0.172721,
mean shuffled-model metric 0.172819, and tuned EASE 0.172956. A reproduction must
retain that finding; an improved number from a changed split is not reproduction.

## Coursework-completion evidence: train the fixed experts from data

```sh
.venv/bin/python -m coursework_completion.bootstrap \
  --stage completion --workers 3 \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --instructor-checkout vendor/RecBole_DSAIT4335 \
  --output-root runs/rebuilt-completion
```

This explicitly trains the 12 already selected standard configurations and three
research configurations for each of seeds 2026, 2027 and 2028. It consumes the
public `coursework_completion/rebuild_recipe.json`, not saved predictions. It
preserves each seed's expert order and verifies original TRAIN/catalog hashes.
Use `--workers 1` to run seed pipelines sequentially; every worker limits numeric
libraries to one thread. The model budgets are fixed, so this stage does not
repeat historical expert hyperparameter searches.

The five original standard experts use the bundled historical adapter whose
source bytes match their recorded hashes. The seven later standard experts use
the current adapter, likewise verified against recorded hashes. The research
helper uses the declared historical settings and current mathematical functions;
the recipe explicitly retains their different historical source hashes. A
separate numerical comparison establishes observed score agreement, rather than
asserting that different source files are identical.

The stage then runs the existing validation-only hybrid study and freeze tools.
Those historical study builders use original validation labels to reconstruct
their old hybrid selections; this is separate from the completion study's new
selection barrier. Original TEST is never evaluated. The standard model adapter
does create split-ID files, including the original TEST IDs, as part of the
deterministic dataset split; no TEST outcome metric is computed. The research
helper uses TRAIN ratings only.

The output `runs/rebuilt-completion/frozen/` contains the newly trained and frozen
three-seed source required by the completion analysis. `REBUILD-VERIFIED.json`
records every fixed configuration, exact adapter checks, score-file hash
comparisons, source hashes, commands and elapsed time. Changed numerical results
must be reported; successful execution alone is not proof of historical agreement.

Run the completion analysis against these rebuilt inputs:

```sh
.venv/bin/python -m coursework_completion.run \
  --source-root runs/rebuilt-completion/frozen \
  --items vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/replayed-completion \
  --evidence runs/replayed-completion-evidence
```

This stage has its own fixed protocol and freezes every seed's choices before
assessment. The source rebuild and the completion study have different scopes;
the nested assessment remains exploratory after previous use of this dataset.

## Measured NGCF difference

The completed raw-data reproduction found material NGCF score differences; the
other 42 expert runs retained their historical top-ten order. A bounded diagnostic
repeated four epochs with identical initial parameters: two 12-thread runs
diverged, whereas three one-thread probes matched exactly and matched the fresh
rebuild's loss prefix. Historical thread settings were not recorded, so this
supports numerical sensitivity without certifying the exact historical cause.
It is not a claim of full 60-epoch determinism or exact historical reproduction.
See the [diagnosis and repeatable probe](coursework_completion/reproduction/NGCF_DIAGNOSIS.md)
and its aggregate receipt for the measured limits. Historical evidence is unchanged.

## Observed replay, including differences

These commands were exercised from a temporary ZIP extraction with the pinned
offline instructor checkout and the recorded environment. No old run artifacts
or saved predictions were supplied to the builders. The original artifacts were
opened afterward for comparison. The extracted scientific files matched all
recorded source hashes.

The split bootstrap reproduced the exact original TRAIN and catalog hashes.
All 39 grouping candidates, all six selections and score arrays, and all
assessment metrics and bootstrap intervals reproduced exactly. That replay took
75.50 seconds while the independent expert rebuild ran concurrently. See the
[grouping replay receipt](coursework_completion/reproduction/grouping-replay-v1.json).

The full 45-expert rebuild completed in 543.98 seconds. Its 36 standard adapters
matched their historical source bytes. Of the 45 expert score files, 37 were
byte-identical and 42 preserved every user's top-10 order. All three NGCF runs
changed materially: 148, 101 and 146 users respectively received different
top-10 item sets. The largest score difference was 0.386966. The cause was not
established by the comparison; this is **not an exact numerical reproduction
of every historical expert**. The independent
[source-score audit](coursework_completion/reproduction/source-scores-replay-v1.json)
separates file, array, full-ranking and top-10 agreement, without reading relevance
labels. The other floating-point differences did not change top-10 orders.

The completion analysis then ran against the newly rebuilt frozen bundles in
37.58 seconds. All 12 selected candidate IDs and decision settings were retained.
The switching models' fitted group-mean diagnostics changed, but their selected
experts did not. EASE, SLIM, both switching variants, mixed and meta retained all
recommendations and assessment metrics. The three methods that combine the
broader expert pool changed as follows:

| Mean assessment nDCG@10 | Original | Rebuilt | Rebuilt minus original |
| --- | ---: | ---: | ---: |
| Context model | 0.2781326172 | 0.2781667012 | +0.0000340840 |
| Fixed RRF | 0.2706118275 | 0.2703309475 | -0.0002808800 |
| Tuned RRF | 0.2712939097 | 0.2714641241 | +0.0001702143 |

The [completion comparison](coursework_completion/reproduction/completion-replay-v1.json)
retains all changed development, settings and assessment fields, all 171 mean
metric comparisons, and aggregate top-10 differences for each seed and method.
The original scientific evidence is preserved; changed results were neither
substituted nor hidden. These runs demonstrate an executable reconstruction with
bounded observed differences, not universal bitwise determinism or a fresh
population test. Timing and provenance hashes naturally differ between runs.

The standalone comparison can be rerun when both completed run directories are
available, using explicit paths:

```sh
.venv/bin/python coursework_completion/reproduction/compare_completion.py \
  --reference path/to/original-completion-run \
  --replay runs/replayed-completion \
  --rebuild-receipt runs/rebuilt-completion/REBUILD-VERIFIED.json \
  --source-archive path/to/extracted-source.zip \
  --output runs/completion-comparison.json
```

Original private predictions are deliberately absent from the supplied source
archive, so the standalone historical comparison requires those separately.
Their absence does not prevent rebuilding the models or running either study
from the pinned instructor data.
