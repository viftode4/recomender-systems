# Recommender Systems · Group 24

We use MovieLens 100K to study **which recommenders work well together** and
**what accuracy we trade for diversity, calibration and fairer exposure**.

The implementation, saved experiments, five-page report draft and code archive
are ready for team review. Contributor details and submission are still open.

## Start here

1. [Results](final-project/docs/RESULTS.md): what we tested and what we found.
2. [Report draft](final-project/reports/coursework-complete-v2/report.pdf): the three assignment tasks.
3. [Code guide](final-project/docs/PROJECT_MAP.md): which files to read and how they connect.

## Run the first check

From this repository's root, with Python 3.11:

```sh
python3.11 -m venv final-project/.venv
final-project/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r final-project/requirements-smoke.txt
final-project/.venv/bin/python final-project/operations/check_docs.py
final-project/.venv/bin/python final-project/operations/team_smoke_check.py
```

Expect `"status": "passed"` and 45 passing tests. This checks documentation,
small examples and saved artifacts without downloading data or training models.
Once installed, `make check` runs both checks. For Windows and the shared ZIP, see the
[quickstart](final-project/docs/QUICKSTART.md).

## Find what you need

| I want to… | Open |
| --- | --- |
| Choose a task and reviewer | [Team plan](final-project/docs/team/TEAM_PLAN.md) |
| Make a contribution | [Contribution guide](CONTRIBUTING.md) |
| Check assignment coverage and remaining work | [Completion checklist](final-project/docs/COMPLETION.md) |
| Train models or reproduce experiments | [Setup](final-project/docs/SETUP.md), then [reproduction guide](final-project/REPRODUCE.md) |
| Explore optional research | [Research index](final-project/docs/RESEARCH_INDEX.md) |

The [documentation folder](final-project/docs/README.md) groups the guides by purpose.

All group work is in `final-project/`. `assignment1/` is separate individual
coursework. AI assistance was used; each teammate should understand and verify
the part they contribute.
