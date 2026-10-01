# Run the first check

Use Python 3.11. This check runs 45 small tests and verifies saved report/evidence
files. It needs NumPy, with no dataset download or model training.

## From the repository

Run from the repository root, the directory containing `final-project/`:

```sh
python3.11 -m venv final-project/.venv
final-project/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r final-project/requirements-smoke.txt
final-project/.venv/bin/python final-project/operations/check_docs.py
final-project/.venv/bin/python final-project/operations/team_smoke_check.py
```

After setup, `make check` runs both the documentation and smoke checks.
Use `make check-docs` to check links only. To use an existing environment:

```sh
make check PYTHON=/path/to/python3.11
```

On Windows, use your Python 3.11 interpreter to create the environment, then
replace `final-project/.venv/bin/python` with `final-project\.venv\Scripts\python`.
`make` is optional.

## From the shared meeting ZIP

Extract the ZIP and open `START_HERE.html`. Run these commands from the
extracted folder containing `coursework/` and `tools/`:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple numpy==1.26.4
.venv/bin/python tools/team_smoke_check.py --project-root coursework/code
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python`.

## Expected result

The final output includes:

```json
{
  "status": "passed",
  "synthetic_tests": {"tests": 45, "failures": 0, "errors": 0, "skipped": 0},
  "training_started": false
}
```

This verifies small metric/hybrid/reranking examples, evidence hashes and the
report hash. It does not retrain the models or reproduce the full experiments.

If Python cannot import NumPy, run the install command with the same interpreter
as the check. If an artifact is missing or its hash differs, keep the error and
check that you are using one complete checkout or extracted package.

## What to do next

Read the [code guide](PROJECT_MAP.md), choose a task in the
[team plan](team/TEAM_PLAN.md), and save your command and result in a
[review note](team/REVIEW_TEMPLATE.md).

Full experiments need [the full setup](SETUP.md) and
[reproduction commands](../REPRODUCE.md). In the meeting ZIP, the detailed
replay guide is `coursework/code/REPRODUCE.md`.
