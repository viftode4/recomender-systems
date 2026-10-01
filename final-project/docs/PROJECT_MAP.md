# Code guide

Start with the [results](RESULTS.md), then follow one result through the code.
Paths below start at `final-project/` in the checkout or `coursework/code/`
in the meeting ZIP.

## Follow the pipeline

```text
MovieLens ratings + genres
    → experiment.py calls run.py: fit and compare individual models
    → study.py: fit hybrids and compare reranking choices
    → freeze.py: save the selected settings and model state
    → final_evaluate.py: measure frozen choices on TEST
    → coursework_completion/build_review.py: assemble the report and ZIP
```

For a first run, use the [quickstart](QUICKSTART.md).
The experiment commands and dependencies are in [REPRODUCE.md](../REPRODUCE.md).
Running `study.py` fits new hybrid weights; it does not just display saved results.

## Read these files first

| File | Purpose | Functions to start with |
| --- | --- | --- |
| [metrics.py](../metrics.py) | Accuracy, novelty and catalog coverage | `ranking_metrics()`, `evaluate()` |
| [run.py](../run.py) | Individual models, genre content and score export | `genre_content_features()`, `top_k()`, then `main()` |
| [study.py](../study.py) | Score normalization, hybrid fitting and reranking | `normalize_scores()`, `build_features()`, `fit_ridge()`, `rerank()` |
| [societal.py](../societal.py) | User groups, calibration, exposure and group policies | `training_taste_groups()`, `discounted_exposure_metrics()`, `fit_group_policy()` |
| [coursework_completion/models.py](../coursework_completion/models.py) | Mixed lists, meta-level profiles, switching and rank fusion | `mix_lists()`, `fit_meta_level()`, `fit_switch()`, `rrf_scores()` |

Read these small functions before the larger experiment `main()` functions.
Worked examples are in [metric tests](../tests/test_metrics.py),
[regression tests](../tests/test_hybrid_constraints.py),
[societal tests](../tests/test_societal.py) and
[hybrid-family tests](../coursework_completion/test_models.py).

## Translate the result names

| Name | Meaning |
| --- | --- |
| `ExactPop` / `GenreContent` | Popularity baseline / our genre-only content model |
| `static` | One learned weight per recommender; Daniel's H1 |
| `group-switch` | Select an expert by TRAIN activity group; Daniel's H2 |
| `context` | Weight interactions with history size, genre entropy and item popularity, plus disagreement; FWLS-style, as in Daniel's H3 |
| `user` / `item` | Restrict context interactions to user features / item popularity |
| `disagreement` | Static score features plus the spread between experts' standardized scores |
| `constrained` / `calibrated` | Sum-to-one weights before / after aligning scores to the regression target |
| `rrf` | Reciprocal-rank fusion: combine item ranks |
| `mixed` / `meta-level` | Combine list slots / use genre profiles as input to a collaborative decoder |

## Trace a reported number

For example, the static hybrid's mean nDCG@10 is **0.33564**:

1. Read its row in the [frozen comparisons](../evidence/final-comparisons-v3/SUMMARY.md).
2. Find each seed's selected static model in [the frozen aggregates](../evidence/final-primary-v3/aggregates.json).
3. Read `build_features(..., "static")` and `fit_ridge()` in [study.py](../study.py).
4. Check `ranking_metrics()` in [metrics.py](../metrics.py) for the nDCG calculation.

The [report content](../reports/coursework-complete-v2/report-content.json)
contains the report's tables and source evidence. `summarize.py` produces
**development** summaries; its plots are a different evaluation from the frozen test.

## Data rules to understand

- Each model exports raw user/item scores and their ID ordering. Consumers mask
  observed items for each evaluation phase before ranking.
- The main experiment uses a random per-user 80/10/10 split. All recorded ratings
  count as relevant, including low ratings: we predict recorded interactions.
- TRAIN supplies histories/features. Separate validation-user cohorts fit hybrid
  weights, select settings and calibrate the group policy. TEST was opened after
  freezing choices and is already known.
- [Later hybrid completion](../coursework_completion/results-v1/aggregates.json)
  reuses validation-calibration users. Keep its results in their own table.

## Where everything else lives

| Location | Contents |
| --- | --- |
| [tests/](../tests/README.md) | Core algorithm and experiment checks |
| `coursework_completion/` | Additional hybrids, their tests and the current report builder |
| [evidence/](../evidence/README.md), [reports/](../reports/README.md), [packages/](../packages/README.md) | Saved results, reports and shareable snapshots |
| `operations/` | Starter check, meeting-pack builder and research monitoring |
| `runs/`, `vendor/` | Local outputs and instructor code/data; excluded from Git |

The root model scripts retain their paths because frozen runs and reproduction
recipes refer to them. Use this guide to find the relevant functions; the
[documentation index](README.md) separates setup, teamwork and research.

## Optional research

The [research index](RESEARCH_INDEX.md) explains `exploratory/` and the field,
negative-information and other research scripts. It separates measured results
from proposed experiments. Follow it when working on a specific research question.
