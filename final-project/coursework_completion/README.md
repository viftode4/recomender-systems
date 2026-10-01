# Explicit hybrid-family completion

This bounded extension covers mixed and meta-level hybrids and tunes the existing
switching and rank-fusion families. It is coursework coverage, not a claim of a new
algorithm. The [protocol](PROTOCOL.md) defines all choices before fitting.

For a clean reproduction, follow [REPRODUCE.md](../REPRODUCE.md) to prepare the
pinned instructor checkout and Python environment. Run from the project root, or
the extracted archive's `code/` directory. The commands use relative paths and
rebuild the required fixed experts from data; historical private run directories
are not required.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover \
  -s coursework_completion -p 'test_*.py' -v

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m coursework_completion.bootstrap \
  --stage completion --workers 3 \
  --data-path vendor/RecBole_DSAIT4335/dataset \
  --instructor-checkout vendor/RecBole_DSAIT4335 \
  --output-root runs/rebuilt-completion

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m coursework_completion.run \
  --source-root runs/rebuilt-completion/frozen \
  --items vendor/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/replayed-completion \
  --evidence runs/replayed-completion-evidence
```

Use new output paths for every execution. Raw data and predictions are not
redistributed. The bootstrap trains the declared configurations and reconstructs
the earlier validation-based hybrid choices; its checks and scope are documented
in [REPRODUCE.md](../REPRODUCE.md). The completion runner then applies its own
selection barrier. `models.py` supplies pure deterministic model functions;
`run.py` enforces the reused-cohort boundaries. The recorded results below used
the original hash-verified `runs/frozen-v3` source. Rebuilt inputs must be compared
numerically with that evidence; finishing the rebuild alone does not prove agreement.
The report's older TEST figures and this extension's validation-calibration figures
have different endpoints/cohorts and must not be presented as one matched table.

The original 471/236/236 meta-fit/development/calibration split is reused. All
calibration predictions are frozen across three seeds before those labels are
parsed by this runner, but those labels have already been exposed elsewhere in
the project. This is exploratory assessment, not a newly independent holdout.
No original TEST data or final evaluation result is read. Contributor names and
honest peer evaluations remain human responsibilities.

## Completed bounded study

All 39 configurations completed in 37.22 seconds. The twelve family choices and
all matched reference predictions were sealed across seeds 2026/2027/2028 before
this runner parsed any calibration item labels. The selected quotas were `(6,2,2)`
for every seed; meta-level penalties were `1,1,0.1`; switch group counts `3,3,2`;
RRF offsets `10,100,100`. There was no additional sweep after assessment.

Equal-weight means on the same 236 reused-calibration users per seed:

| Role | nDCG@10 | Recall@10 | Genre diversity |
|---|---:|---:|---:|
| Locked EASE | 0.273765 | 0.256034 | 0.805366 |
| Locked SLIM | 0.265742 | 0.253693 | 0.808858 |
| Locked context hybrid | 0.278133 | 0.262760 | 0.791994 |
| Mixed lists | 0.216840 | 0.207936 | 0.772672 |
| Meta-level | 0.198748 | 0.187839 | 0.772992 |
| Fixed switch | 0.274641 | 0.258270 | 0.805446 |
| Tuned switch | 0.274641 | 0.258270 | 0.805446 |
| Fixed RRF | 0.270612 | 0.257176 | 0.798073 |
| Tuned RRF | 0.271294 | 0.256481 | 0.796721 |

Mixed and meta-level methods cover additional lecture mechanisms but do not improve
ranking accuracy in this bounded comparison. Tuning the activity switch leaves the
assessment rankings unchanged; tuning RRF produces only a small descriptive mean
change. Mixed lists reduce mean head exposure (0.8701 versus EASE's 0.9874) but also
reduce measured genre diversity; adding a content source does not automatically
make a list more diverse. These are operational metrics, not causal fairness claims.

Public [aggregates](results-v1/aggregates.json) retain all candidate metrics, every
selected setting, nine matched methods and full activity/item-group results.
`results-v1/SHA256.json` binds aggregates, runtime/source protocol and provenance.
The independent [review](REVIEW.md) and [audit receipt](audit-v1/audit.json) confirm
43,239 numerical checks, with maximum absolute difference `7.11e-15`. The auditor
replayed all 39 candidate development metrics, the selected equations for all nine
roles across three seeds, and all assessment/group/exposure tables. It checked
the source, input, output and selection seals; it did not repeat source-expert
training or the hyperparameter search. No raw predictions or user identity
collections belong in the public evidence directory.
