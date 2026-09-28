# Recommender Systems final-project starter

A working base for DSAIT4335's MovieLens 100K hybrid recommendation project.
The group still needs to agree on model choices, ownership, and experiment scope.
This is infrastructure and initial baselines, not a completed project submission.

## Included

- One shared RecBole split/configuration; CPU execution and seed 2026.
- ExactPop (exact training frequency), Random (independent per-user scores),
  and training adapters for EASE, ItemKNN and BPR.
- Independent Precision, Recall, nDCG, MRR and Hit metrics; catalog coverage
  and training-frequency novelty. Per-user results support later group analysis.
- Original-ID split exports, split/data/source fingerprints, environment versions,
  full score matrices for future hybrids, recommendations and metrics.
- Validation-only evaluation by default; explicit `--test` opt-in.
- Refusal to overwrite a run directory; completion status in each run manifest.

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

The commit is the instructor checkout used for the local smoke checks. It is not
claimed to be the latest version announced September 25. Review any upstream
updates as a team and repeat the checks before changing the shared dependency.
A fresh installation has not yet been tested; this is not a dependency lockfile.
Data, vendor code, model checkpoints and experiment outputs are ignored by Git.

If the course environment is already installed, use its Python directly and pass
the existing `dataset/` path. No copies of assignment reports or instructor PDFs
are needed in this repository.

## Run

```sh
python3 -m unittest discover -s tests -p 'test_metrics.py' -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py --model ExactPop --data-path vendor/RecBole_DSAIT4335/dataset --out runs/exact-pop
.venv/bin/python run.py --model Random --data-path vendor/RecBole_DSAIT4335/dataset --out runs/random
.venv/bin/python run.py --model EASE --data-path vendor/RecBole_DSAIT4335/dataset --out runs/ease
```

`--data-path` is the parent of `ml-100k/`, not the dataset folder itself. Model
choices are ExactPop, Random, EASE, ItemKNN, BPR. Copy `config.json` for a tuning
variant and pass `--config path.json`. Keep split, seed and data processing the
same across candidates. Compare all three split fingerprints before combining
runs; also check identical user and item IDs in the score archives.

Run directories contain `effective-config.txt` with resolved model defaults.
The initial EASE/ItemKNN/BPR defaults are not a hyperparameter search. Preserve
configs for each candidate; select using validation MRR@10. The starter does
not automatically choose or freeze the winning model.

After documenting frozen settings, a separate output directory plus `--test`
trains that configuration and evaluates test as well. This flag is a deliberate
workflow safeguard, not an access-control mechanism: test split IDs are exported
for reproducibility. Do not tune after viewing test results. Seed 2026 differs
from Assignment 3's 2020 but reuses MovieLens, so this is not a new independent
population or untouched confirmatory benchmark.

## Evaluation contract

All observed ratings are implicit positives (no threshold). Use full-catalog
ranking with k=10, excluding padding and training interactions during validation;
test also excludes validation interactions. Each evaluated user needs at least k
unseen candidates and at least one positive. Macro averages include every user
in that held-out split. Ties use stable internal item order; IDs are exported.

Novelty is the average `-log2((training_count + 1) / (training_interactions +
catalog_size))`. Coverage is distinct recommended items divided by the complete
real-item catalog. These are initial beyond-accuracy measures; diversity,
calibration and fairness remain to implement.

`*-scores.npz` contains unmasked full-catalog scores, original-ID `users` and
`items`. Column zero is padding; exclude it and each user's history before
ranking or fitting hybrid features. Load with NumPy's default `allow_pickle=False`.
Do not fit a regression hybrid and report its performance on the same labels:
agree on an inner split or out-of-fold training protocol first (see PLAN.md).

ExactPop and Random are independent reference implementations using the RecBole
split. They are deliberately labelled: the instructor fork's Pop training uses
batch count updates, and its Random full-sort path shares scores within a batch.
Our baselines are not exact replays of those implementations.

## Verification and limitations

Local verification uses the existing course environment, not a fresh install.
Observed versions: RecBole 1.2.1, NumPy 1.26.4, SciPy 1.17.1, PyTorch 2.14.0,
pandas 3.0.6. The upstream data loader emits pandas chained-assignment and
non-writable-array warnings on this environment. No upstream code is patched by
this starter. The local source may have coursework edits; the commit alone is
not a complete environment lock.

See PLAN.md for the task mapping, feedback questions and report constraints.
AI assistance was used to scaffold this base; every team member should review
and understand their contributions and follow the course's disclosure rules.

Verified locally on September 28, 2026: 8 unit tests passed. ExactPop, Random and
EASE completed full-catalog validation runs for all 943 users with matching split
fingerprints and 943 × 1683 raw score matrices (including padding column). No
MovieLens test metrics were produced. These are smoke checks, not tuned results.
ItemKNN and BPR adapters have not yet received end-to-end checks.
