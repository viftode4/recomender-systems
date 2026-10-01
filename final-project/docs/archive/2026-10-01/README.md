# Recommender Systems: hybrid research project

A runnable DSAIT4335 research project with audited course-model adapters,
regression hybrids, explicit-rating contrast experiments, societal reranking,
frozen inference and a task-formatted report builder. Group 24 has five seats;
actual member names and contributions must be supplied before final submission.

Start with [the handoff and findings](HANDOFF.md), the
[requirements and lecture coverage](coursework_completion/COVERAGE.md), and
[portable reproduction instructions](REPRODUCE.md). The completion pass adds
mixed and meta-level hybrids and explicit switching/RRF tuning in a separate
exploratory study. The original frozen study and grouping experiment remain
unchanged. Real contributor information and individual peer feedback remain
deferred human inputs; no course submission has been made.

The current review deliverables are the
[five-page coursework report](reports/coursework-complete-v1/report.pdf) and
[Group 24 ZIP](packages/coursework-complete-v1/24.zip). The
[package verification](packages/coursework-complete-v1/verification.json) records
the archive checks, bundled tests and report replay. The
[team review guide](coursework_completion/TEAM_REVIEW.md) explains the methods
and the checks for five real contributors.

The completion study ran all 39 declared configurations across three seeds.
On its reused validation-calibration users, mean nDCG@10 is 0.216840 for mixed,
0.198748 for meta-level, 0.274641 for tuned switching and 0.271294 for tuned RRF,
against matched EASE at 0.273765 and the existing context hybrid at 0.278133.
These results complete the lecture-family comparison without establishing a new
accuracy advance. See the [aggregate results](coursework_completion/results-v1/aggregates.json)
and [independent audit](coursework_completion/REVIEW.md).

The [grouping experiment](exploratory/framing_search/predictive/README.md) tested
whether equal-timestamp rating groups improve prediction without revealing a
missing target's timestamp. All 39 declared fits completed. On its nested
TRAIN-derived assessment, the grouped model scores 0.172555 nDCG@10, versus
0.172956 for newly fitted EASE (-0.23% relative), 0.172721 for the ungrouped pair
control, and 0.172819 averaged over three shuffled-group controls. The experiment
does not support the proposed predictive advantage or the requested substantial
advance. These are different data partitions from the original results below;
they must not be compared as a performance trend.

The disagreement hybrid's mean held-out nDCG@10 is 0.33701, versus 0.31613 for
the standalone expert selected on validation (+6.6% relative); EASE separately
scores 0.32155. The custom joint categorical field reaches 0.26786 at its
400-epoch cap and does not establish an adaptive-routing advantage. These are
three overlapping splits of one dataset, not independent replications or a
state-of-the-art claim. See [paired hybrid comparisons](evidence/final-comparisons-v3/SUMMARY.md)
and [matched research comparisons](evidence/final-field-comparisons-v1/RESULTS.md).

## What works

- RecBole training for EASE, ItemKNN, UserKNN, BPR, SLIMElastic, corrected FISM,
  LightGCN, NGCF and NeuMF; genre-content, Random and exact popularity baselines.
- Predeclared expert tuning grids; shared splits; three data seeds; no test
  evaluation in the development pipeline.
- Sum-to-one constrained regression plus static, user, item, disagreement and full
  contextual features; static/contextual pairwise regression variants.
- Reciprocal-rank fusion and user-activity-group expert switching.
- The separate [coursework completion study](coursework_completion/README.md)
  covers mixed-list and meta-level hybridization and tunes switching/RRF.
  Its assessment reuses previously exposed validation cohorts and is explicitly
  exploratory; it does not replace the original held-out model comparison.
- Independent accuracy, novelty, coverage, genre diversity, calibration,
  activity-group utility and head/tail exposure/recall metrics.
- Diversity, genre calibration, user popularity calibration and exposure
  rerankers; before/after-fusion comparisons; independently calibrated group
  policies with explicit uncertainty and holdout audits.
- Two custom liked/disliked contrast architectures and matched ablations, plus
  positive-only and signed linear controls. Failed hypotheses are retained.
- A categorical evidence field learned from scratch: five-category evidence,
  recurrent attention and learned source weights; conditional-rating and joint
  recorded-item/rating objectives, with fixed-flow and hard-clamp controls.
- A negative-information audit with context/probe separation, shuffled weights
  and controls preserving each user's and item's dislike counts.
- Training-only taste groups, whole-catalog exposure Gini/entropy, discounted
  group exposure, and liked-versus-all-history calibration audits.
- Frozen model bundles with exact validation replay and source hashes; final
  inference cannot retune or refit. Report rendering and group ZIP packaging.
- Candidate-pool feasibility bounds and selective expansion experiments.
- Score/split/config exports, input hashes, coefficient artifacts, paired
  descriptive intervals, figures and aggregate evidence suitable for review.

## Setup

Use Python 3.11 and the instructor's RecBole fork. From this directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
mkdir -p vendor
git clone https://github.com/masoudmansoury/RecBole_DSAIT4335.git vendor/RecBole_DSAIT4335
git -C vendor/RecBole_DSAIT4335 checkout 081c3f6edf8e466d3ed5e163631a1afb6fe892bf
.venv/bin/python -m pip install -e vendor/RecBole_DSAIT4335
```

This pins the instructor source used for the experiments. The September 25
upstream revision 439c5a899296cbaad9842751dd4355393f22d679 adds neural model
configs and an NGCF sparse-matrix compatibility correction. Our run adapter
provides explicit configs and the compatibility correction locally. It also
corrects FISM target-history inclusion and double-sigmoid loss; both have
independent mathematical tests. The instructor checkout remains unchanged.

The requirements file records the observed Python 3.11 macOS environment. It is
not a universal cross-platform dependency lock. `verify_environment.py --out
evidence/environment.json` records imported source hashes and package versions.
See the environment reproduction evidence for exactly what has been verified.

## One-command experiment suite

Avoid numerical-library oversubscription when running jobs concurrently:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiment.py \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --out runs/core \
  --tune-experts --extended
```

This runs seeds 2026, 2027 and 2028. Use `--seeds 2026` for a shorter first check.
The output directory must not already exist. The suite stops on failure and keeps
logs/artifacts; it does not overwrite, resume or silently reuse previous results.

Expert grids: EASE regularization 50/250/1000, ItemKNN neighbors 50/100/200,
BPR dimension/epoch pairs (32,20), (64,60), (128,100). BPR configurations vary two
factors together; they are not an isolated embedding-dimension ablation.
Extended grids add UserKNN neighbors, SLIM penalty/mixing, FISM history exponent,
genre-IDF power, and neural epoch budgets. `--thorough-lightgcn-budget` includes
20/60/100/200 epochs for LightGCN; ordinary extended mode uses 20/60. These are
finite budgets, not proofs of convergence or fully tuned SOTA comparisons.

Training uses fixed budgets. Current validation cohorts are 471 meta-fit users,
236 development/selection users and 236 policy-calibration users. Expert variants
and regression coefficients use meta-fit users; hybrid settings use selection
users; group-policy strengths use calibration users. Development scores remain
selection-biased. Historical research-v2 used the older 471/472 two-cohort design;
`study.py --calibration-fraction 0` reproduces that exploratory setting.
No test labels are read by the hybrid study. Use the thread limits below when running several studies concurrently. See RESEARCH.md for the full protocol.

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
`freeze.py` now saves the hybrid parameters and transforms, verifies exact
validation ranking replay, and seals inference source hashes. `final_evaluate.py`
uses that bundle without refitting or test-based selection. Test split IDs are exported for reproducibility,
so workflow discipline remains necessary even though test metrics are off by default.

## Generate shareable evidence

The plotting Python needs Matplotlib (locally verified with 3.9.4), independently
of the training environment:

```sh
python3 summarize.py --runs runs/research-v3 --out evidence/research-v3
```

Use a new output path if the checked-in evidence folder already exists. This
creates Markdown, PNG/PDF figures, aggregate/group metrics, coefficient and
selection records, and manifests. Individual user histories and recommendations
stay under ignored `runs/`. Do not compare runs without matching split/data hashes
and score ID order; `study.py` checks these and rejects incomplete/test-evaluated inputs.

## What the new model is learning

A rating answers a conditional question: how was an item judged once it became
part of someone's recorded history? The assignment also asks which items will
appear in held-out records. These are different targets. Optimizing the first
does not automatically solve the second, especially when an item is rarely
observed. The conditional field's poor ranking performance makes that mismatch
visible rather than hiding it behind a combined score.

The field learns how rating categories interact with item identities. It routes
the observed evidence through learned intermediate vectors, updates the context
representation, and revises its routes and source weights within a prediction.
Unknown candidate predictions never become evidence for other candidates. The
fixed-flow control keeps the first routes and weights throughout those updates;
the hard-clamp control keeps supplied categorical sources fixed. Neither control
isolates every causal component, and changing internal weights do not identify
human feelings or trust.

The joint experiment changes only the likelihood applied to the same five
logits. Total item mass supplies the assignment ranking; mass for rating 4 or 5
supplies a separate liked-record ranking. A same-budget control tests whether
recurrent adaptation helps. The 400-epoch follow-up checks the original
100-epoch boundary under a declared cap; original results remain visible.
All comparisons include strong fixed baselines as well as simple count models.
See [the model rationale](ADAPTIVE_RESEARCH.md),
[joint objective](JOINT_FIELD_PROTOCOL.md), and
[pre-test decisions](evidence/EVALUATION_PROTOCOL.md).

For example, after `runs/core` has exported the shared EASE splits:

```sh
.venv/bin/python categorical_experiment.py --source-root runs/core \
  --ratings vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --item-metadata vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/categorical --evidence evidence/categorical-reproduction
.venv/bin/python joint_field_experiment.py --source-root runs/core \
  --ratings vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --item-metadata vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/joint --evidence evidence/joint-reproduction
.venv/bin/python categorical_evaluation.py freeze \
  --research-root runs/categorical --out runs/frozen-categorical
.venv/bin/python joint_evaluation.py freeze \
  --research-root runs/joint --out runs/frozen-joint
.venv/bin/python joint_field_convergence.py --source-root runs/core \
  --prior-root runs/joint \
  --ratings vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --item-metadata vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/joint-convergence --evidence evidence/joint-convergence-reproduction
.venv/bin/python joint_convergence_evaluation.py freeze \
  --research-root runs/joint-convergence --out runs/frozen-joint-convergence
```

These commands stop before test evaluation. Use new output directories, preserve
all protocol/source files while fitting, and apply the numerical thread limits
above. The separate convergence runner requires an intact v1 run for exact
first-100-epoch verification; its protocol records that dependency.

The offline [synthetic-profile demonstration](reports/demo/categorical-field/index.html)
shows the selected conditional model's responses to six fixed rating scenarios.
It contains no real user history. The displayed probabilities demonstrate
sensitivity, not recommendation accuracy or calibration. The packaged copy is
at `demo/index.html` beside the report.

## Data and metric contract

In the primary course study, all observed ratings are implicit positives; no
rating threshold. The separate research tracks preserve five-category ratings
and label their all-observed and rating-at-least-four endpoints explicitly.
Ranking uses
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

## Verification and human handoff

The existing local course environment has RecBole 1.2.1, NumPy 1.26.4, SciPy 1.17.1,
PyTorch 2.14.0 and pandas 3.0.6. Upstream loading emits pandas chained-assignment
and non-writable-array warnings. This starter does not patch upstream code.
The tracked instructor source was checked clean on September 28; unrelated
assignment scripts are untracked. Imported source hashes are recorded because a
commit alone does not establish which Python package was actually imported.

Tests include hand-computed metrics, duplicate/seen-item rejection, score ID/split
mismatch rejection, cohort isolation, regression recovery, divergence and reranking
limits. The multi-seed development suite exercises the complete pipeline.
Remaining human work includes verifying the five actual contributors, explaining
the methods, lecturer feedback and individual peer evaluation. Random splits do
not establish temporal or online effectiveness. The generated report remains a
draft until its required names and reviewed content are complete. More complexity
is retained only when evidence supports its value.

The held-out batch completed all eight evaluation jobs and five analysis jobs
with the frozen choices unchanged. This test is now exposed. Any future model
changes require explicitly exploratory reporting or a fresh assessment;
re-running altered models on these splits cannot supply untouched confirmation.

AI assistance was used to implement this base. Review and understand the work,
record actual contributions, and follow the course's assistance-disclosure rules.

## Freeze, audit, report and package

To reproduce the combined research study, run `experiment.py --extended
--thorough-lightgcn-budget --tune-experts --skip-study` into a fresh core directory,
then run the contrast experiment and assemble each seed. For example:

```sh
.venv/bin/python exception_experiment.py --revision 2 --source-root runs/core \
  --ratings vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --items vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/contrasts
.venv/bin/python assemble_study.py --seed 2026 \
  --selections runs/core/expert-selection-2026.json \
  --research-run runs/contrasts/2026 \
  --items vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/combined/study-2026
```

Repeat assembly for 2027 and 2028. `assembly.json` preserves exact source-run
order for freezing. Multiple `--selections` files permit a declared extended
budget to replace the same model from an earlier grid; no new metric selection
occurs in the assembler. The extra contrast model is included despite poor
development performance so its negative result remains visible.

Keep selected expert paths in the exact order passed to `study.py`. Do not open
test labels while choosing models. After the full comparison set is fixed:

```sh
.venv/bin/python freeze.py --study runs/study --runs runs/ease runs/itemknn \
  --items vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/frozen
.venv/bin/python final_evaluate.py --frozen runs/frozen \
  --test runs/ease/test.tsv --out runs/final
python3 report.py --final-results runs/final --frozen runs/frozen \
  --out reports/review --group 24 --member 'Vlad George Iftode'
python3 package_project.py --group 24 --report-dir reports/review \
  --out packages/24.zip --allow-draft
```

`--allow-draft` makes a review archive, not a submission-ready claim. A finalized
report requires the five actual names and reviewed evidence. The package
whitelists source/tests/report artifacts and excludes datasets, model checkpoints
and individual user histories. No command uploads to Brightspace.

`audit_study.py --study STUDY --source-run EXPERT --data DATASET_DIRECTORY
--out AUDIT` joins raw ratings only for training pairs to define taste groups.
`compare_frozen.py` generates predefined paired comparisons across frozen runs;
it does not select the best test model. Run each command with `--help` for the
complete arguments.

Report rendering uses `requirements-report.txt` and installed Times New Roman
regular/bold fonts (`report.py --font-dir` can locate them). Training and
rendering may use separate Python environments. The font files are not bundled.

`run_final_batch.py` orchestrates the exact retained experiment trees in this
workspace. Its `prepare` command verifies all six frozen families and writes a
reviewable plan; `run` reserves one batch before any held-out label access.
Its paths intentionally identify the recorded study versions rather than acting
as a generic training command. On a new reproduction, use the individual
freeze/evaluate CLIs with the new run paths. Never copy a completed batch marker
into an unrelated study or delete one to rerun a changed configuration.

For a three-seed final report, pass each actual pair separately:

```sh
python3 report.py \
  --final-seed 2026 runs/final-v3/2026 runs/frozen-v3/2026 \
  --final-seed 2027 runs/final-v3/2027 runs/frozen-v3/2027 \
  --final-seed 2028 runs/final-v3/2028 runs/frozen-v3/2028 \
  --adaptive-final-results runs/final-adaptive-v1 --adaptive-frozen runs/frozen-adaptive-v1 \
  --joint-final-results runs/final-joint-field-v1 --joint-frozen runs/frozen-joint-field-v1 \
  --joint-convergence-final-results runs/final-joint-field-convergence-v1 \
  --joint-convergence-frozen runs/frozen-joint-field-convergence-v1 \
  --negative-evidence evidence/negative-information-v2 \
  --field-comparisons evidence/final-field-comparisons-v1 \
  --out reports/review --group 24 --member 'Vlad George Iftode'
```

The builder verifies each evaluation independently and averages by frozen model
role. Overlapping splits remain sensitivity checks rather than independent
datasets. Optional categorical, joint and convergence appendices use their own
matched frozen bundles; `report.py --help` lists the arguments. All task and
appendix pages retain the same font, spacing and discussion-word constraints.
