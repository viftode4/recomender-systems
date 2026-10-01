# Recommender Systems project

**Question:** which recommendation signals work well together, and what happens
when we rerank their results for diversity, calibration or exposure?

We have implemented the three assignment tasks and saved their results.
The report is a draft awaiting team review and contributor details.

## Read in this order

1. [Results](docs/RESULTS.md): the main findings, with links to evidence.
2. [Five-page report](reports/coursework-complete-v2/report.pdf): the current deliverable.
3. [Code guide](docs/PROJECT_MAP.md): the pipeline, model names and files to read.
4. [Quickstart](docs/QUICKSTART.md): run 45 small checks without training.

## Where to work

| Assignment part | Main code | Team question |
| --- | --- | --- |
| 1. Models and hybrids | [run.py](run.py), [study.py](study.py), [models.py](coursework_completion/models.py) | Which signals should we combine? |
| 2. Evaluation and analysis | [metrics.py](metrics.py), [final_evaluate.py](final_evaluate.py) | Which users and items benefit? |
| 3. Societal reranking | [study.py](study.py), [societal.py](societal.py) | Who gains variety or exposure, and who loses accuracy? |

Use the [team plan](docs/team/TEAM_PLAN.md) to choose a task.
The [completion checklist](docs/COMPLETION.md) records what remains before submission.

## When you need more detail

- **Full experiments:** [setup](docs/SETUP.md) and [reproduction commands](REPRODUCE.md).
- **Requirement-by-requirement evidence:** [coverage](coursework_completion/COVERAGE.md).
- **Code/report snapshot:** [24.zip](packages/coursework-complete-v2/24.zip) and its [verification receipt](packages/coursework-complete-v2/verification.json).
- **Optional experiments and their dated status:** [research index](docs/RESEARCH_INDEX.md).

The ZIP preserves the verified coursework snapshot; current onboarding docs
live in this checkout. To create an offline copy of the updated guides, install
`requirements-handoff.txt` and run `make handoff` from the repository root.
AI assistance was used in the code and report. Nothing has been submitted.
