# Recommender Systems · Group 24

The group project lives in **[final-project/](final-project/README.md)**.
Start with these three pages:

1. [Project proposals](final-project/docs/PROPOSALS.md): two main questions and one optional research extension.
2. [Five-person plan](final-project/docs/team-meeting-2026-10-01/TEAM_PLAN.md): choose an area, first task and reviewer.
3. [Results summary](final-project/docs/RESULTS.md): what the existing experiments establish.

## Run the first check

From this repository's root, using Python 3.11:

```sh
python3.11 -m venv final-project/.venv
final-project/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r final-project/requirements-smoke.txt
final-project/.venv/bin/python final-project/operations/team_smoke_check.py
```

This runs 45 synthetic tests and checks saved report/evidence files. No dataset
download or training is required. With `make` installed, the last command is
also available as `make check`. On Windows use `final-project\.venv\Scripts\python`
for the environment's interpreter.

Use the [code map](final-project/docs/PROJECT_MAP.md) to find your part,
[contribution guide](CONTRIBUTING.md) for the team workflow, and
[full setup](final-project/docs/SETUP.md) only when you need model training.

**Progress as of 1 October 2026:** the coursework implementation, saved results,
five-page report draft and 45-test starter check are complete. Team review,
contributor details, final report edits and submission remain open. See the
[current handoff](final-project/HANDOFF.md) for artifact status and the
[work plan](final-project/PLAN.md) for proposed next tasks.

The existing report is a draft prepared with AI assistance. Everyone should
understand, verify and improve their own part. Contributor details and submission
remain open; optional research is separate from the required coursework.
The experimental training is stopped; feedback-revision models are designs only.

`assignment1/` contains separate individual coursework. Leave it outside group-project changes.
