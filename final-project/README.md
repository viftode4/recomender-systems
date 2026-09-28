# Recommender Systems: hybrid research project

A runnable DSAIT4335 project base with an original implementation of contextual
regression fusion, controlled ablations, reranking and empirical analysis.
This is a development research checkpoint, not a finished graded submission.

Start with [the five-person plan](PLAN.md), [research design](RESEARCH.md), and
[results with figures](evidence/research-v2/RESULTS.md).

## What works

- RecBole training for EASE, ItemKNN and BPR; independent Random and exact
  training-frequency popularity baselines.
- Predeclared expert tuning grids; shared splits; three data seeds; no test
  evaluation in the development pipeline.
- Regression-weighted hybrids with static, user, item, disagreement and full
  contextual features; static/contextual pairwise regression variants.
- Reciprocal-rank fusion and user-activity-group expert switching.
- Independent accuracy, novelty, coverage, genre diversity, calibration,
  activity-group utility and head/tail exposure/recall metrics.
- Diversity, calibration and exposure rerankers; rerank-before/after-fusion
  comparisons; a group utility-budget policy with development-cohort audits.
- Candidate-pool feasibility bounds and selective expansion experiments.
- Score/split/config exports, input hashes, coefficient artifacts, paired
  descriptive intervals, figures and aggregate evidence suitable for review.

## Setup

Use Python 3.11 and the instructor's RecBole fork. From this directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install 'numpy==1.26.4'
mkdir -p vendor
git clone https://github.com/masoudmansoury/RecBole_DSAIT4335.git vendor/RecBole_DSAIT4335
git -C vendor/RecBole_DSAIT4335 checkout 081c3f6edf8e466d3ed5e163631a1afb6fe892bf
.venv/bin/python -m pip install -e vendor/RecBole_DSAIT4335
```

This is the instructor checkout used locally, not a claim that it is the latest
version announced September 25. Review upstream changes and repeat checks before
changing the shared dependency. A fresh installation has not been tested; these
commands are not a verified cross-platform lockfile. Existing course environments
can run these scripts directly with their Python and local dataset directory.

## One-command experiment suite

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiment.py \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --out runs/research-v2 \
  --tune-experts
```

This runs seeds 2026, 2027 and 2028. Use `--seeds 2026` for a shorter first check.
The output directory must not already exist. The suite stops on failure and keeps
logs/artifacts; it does not overwrite, resume or silently reuse previous results.

Expert grids: EASE regularization 50/250/1000, ItemKNN neighbors 50/100/200,
BPR dimension/epoch pairs (32,20), (64,60), (128,100). BPR configurations vary two
factors together; they are not an isolated embedding-dimension ablation.

Training uses fixed budgets. The validation users are deterministically divided
into 471 meta-fit and 472 development users. Expert variants and regression
coefficients/group policies use meta-fit users. Hybrid settings are selected on
development users, so their reported development metrics are selection-biased.
No test labels are read by the hybrid study. See RESEARCH.md for the full protocol.

## Smaller runs and studies

```sh
.venv/bin/python run.py --model EASE \
  --data-path vendor/RecBole_DSAIT4335/dataset --out runs/ease --fixed-epochs
.venv/bin/python run.py --model ItemKNN \
  --data-path vendor/RecBole_DSAIT4335/dataset --out runs/itemknn --fixed-epochs
.venv/bin/python study.py --runs runs/ease runs/itemknn \
  --items vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item --out runs/study
```

`--data-path` is the parent of `ml-100k/`. `run.py --config custom.json` accepts
another shared RecBole configuration. Without `--fixed-epochs`, the runner uses
validation early stopping, recorded in its manifest. This mode is supported for
exploration but excluded from primary multi-seed evidence by `summarize.py`.

The individual-model runner supports deliberate `--test` evaluation after final
configuration freeze. The hybrid development script does not have a test mode.
A frozen hybrid inference/export path remains to build before final evaluation;
do not retrain it on test labels. Test split IDs are exported for reproducibility,
so workflow discipline remains necessary even though test metrics are off by default.

## Generate shareable evidence

The plotting Python needs Matplotlib (locally verified with 3.9.4), independently
of the training environment:

```sh
python3 summarize.py --runs runs/research-v2 --out evidence/research-v2
```

Use a new output path if the checked-in evidence folder already exists. This
creates Markdown, PNG/PDF figures, aggregate/group metrics, coefficient and
selection records, and manifests. Individual user histories and recommendations
stay under ignored `runs/`. Do not compare runs without matching split/data hashes
and score ID order; `study.py` checks these and rejects incomplete/test-evaluated inputs.

## Data and metric contract

All observed ratings are implicit positives; no rating threshold. Ranking uses
all real items, excluding padding and previously seen interactions. Validation
history is training only; the individual-model test path masks training and
validation. Scores are exported unmasked for hybrid feature construction.
`*-scores.npz` contains `scores`, original-ID `users` and `items`; padding is column
zero. Load with NumPy's default `allow_pickle=False` and apply history masks.

Metrics macro-average users with held-out positives. Novelty uses training counts
with Laplace smoothing; diversity is pairwise genre Jaccard distance; calibration
is genre JSD. Activity groups and head/tail definitions use training interactions.
See RESEARCH.md for equations, normalization, fairness choices and limitations.

ExactPop and Random deliberately differ from the supplied RecBole implementations:
exact interaction counts and independent per-user scores, respectively. This is
labelled in the outputs; they are not exact replays of supplied Pop/Random behavior.

## Verification and remaining work

The existing local course environment has RecBole 1.2.1, NumPy 1.26.4, SciPy 1.17.1,
PyTorch 2.14.0 and pandas 3.0.6. Upstream loading emits pandas chained-assignment
and non-writable-array warnings. This starter does not patch upstream code.
The source checkout may have coursework edits; its commit alone is not an
installation lock. Inspect evidence provenance before treating a replay as identical.

Tests include hand-computed metrics, duplicate/seen-item rejection, score ID/split
mismatch rejection, cohort isolation, regression recovery, divergence and reranking
limits. The multi-seed development suite exercises the complete pipeline.
Remaining: independent review, clean setup verification, temporal sensitivity,
lecturer feedback, frozen final test evaluation, task-formatted report and peer
feedback. More complexity is retained only when the evidence supports its value.

AI assistance was used to implement this base. Review and understand the work,
record actual contributions, and follow the course's assistance-disclosure rules.
