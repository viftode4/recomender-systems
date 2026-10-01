# Group 24 · Recommender Systems project

A working DSAIT4335 coursework base: individual recommenders and hybrids,
independent evaluation, and societal reranking. The code and five-page report
are ready for team review. Contributor details and final submission remain open.

## Start here

1. **Pick the direction:** [three proposals](docs/PROPOSALS.md),
   [short results summary](docs/RESULTS.md), and [code map](docs/PROJECT_MAP.md).
2. **Prepare for today's meeting:** [meeting brief](docs/team-meeting-2026-10-01/MEETING_BRIEF.md)
   and [five-person work proposal](docs/team-meeting-2026-10-01/TEAM_PLAN.md).
3. **Run something small:** [quickstart](docs/team-meeting-2026-10-01/QUICKSTART.md).
4. **Read the output:** [report draft](reports/coursework-complete-v1/report.pdf)
   and [assignment checklist](docs/team-meeting-2026-10-01/ASSIGNMENT_CHECKLIST.md).

The quick check runs 45 existing synthetic tests and checks saved evidence.
It needs no dataset and starts no training. From this directory, with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-smoke.txt
.venv/bin/python operations/team_smoke_check.py
```

## Three parts of the assignment

| Part | Main entry points | Question |
| --- | --- | --- |
| Models and hybrids | `run.py`, `study.py`, `coursework_completion/models.py` | How should several recommendation signals be combined? |
| Evaluation and analysis | `metrics.py`, `summarize.py` | Which results improve, for which users and items? |
| Societal reranking | `societal.py` | What accuracy cost accompanies diversity, calibration or exposure goals? |

For full installation use [setup](docs/SETUP.md). For exact data partitions,
replay commands and known numerical differences use [REPRODUCE.md](REPRODUCE.md).
The [full requirement mapping](coursework_completion/COVERAGE.md) connects the
assignment to the implementation and evidence.

## Research and current state

The custom-model experiments are collected in the [research index](docs/RESEARCH_INDEX.md).
Some are complete with negative results; the conditional-evidence training is
paused. The newer feedback-revision models are design documents only.

The main report separates original frozen evaluation from later exploratory
studies. The work does not establish a breakthrough or a universal best model.
AI assistance was used; each member should understand and verify their part.

This is a review draft, not a submitted assignment. The verified code/report
snapshot is tracked at `packages/coursework-complete-v1/24.zip`. To rebuild the
meeting pack, install `requirements-handoff.txt` in the same environment and run
`make handoff` from the repository root. Generated previews and meeting ZIPs
stay outside Git. The [contribution guide](../CONTRIBUTING.md) explains the team workflow.
