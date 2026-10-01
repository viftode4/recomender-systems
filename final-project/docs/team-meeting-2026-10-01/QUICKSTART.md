# A teammate's first working session

From a Git checkout, start with the short check below. In the shared meeting
pack, open `START_HERE.html`. The pack contains
the existing coursework draft and source under `coursework/`, meeting notes under
`meeting/`, and a small smoke check under `tools/`. The smoke check uses synthetic
examples and public aggregate evidence; it needs no MovieLens download or model
training.

## From a Git checkout

Run from the repository root with Python 3.11:

```sh
python3.11 -m venv final-project/.venv
final-project/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r final-project/requirements-smoke.txt
final-project/.venv/bin/python final-project/operations/team_smoke_check.py
```

If `make` is installed, `make check` is equivalent to the last command. On
Windows use `final-project\.venv\Scripts\python`. These commands use the
tracked code, public evidence and report; ignored historical runs are unnecessary.

## From the shared meeting pack

Use Python 3.11 with NumPy. For a fresh environment, run these commands from the
extracted meeting pack's root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple numpy==1.26.4
.venv/bin/python tools/team_smoke_check.py --project-root coursework/code
```

Installing Python or downloading NumPy may take longer than a minute. Only the
check itself is intended to be quick. An existing Python 3.11 environment with
NumPy can run the last command directly. On Windows, use `.venv\Scripts\python`
instead of `.venv/bin/python`.

On Vlad's existing checkout, the environment is already prepared. From the
project root:

```sh
runs/environment-check/.venv/bin/python operations/team_smoke_check.py
```

The check runs 45 existing tests for hand-computed ranking metrics, regression
constraints, deterministic list construction, mixed and meta-level hybrids,
switching, rank fusion, and societal diagnostics. It also verifies three public
completion-evidence files against their SHA-256 manifest and verifies the report
hash and PDF header. The five-page count comes from the report manifest; this
command does not independently render or parse the PDF.

Expected final output includes:

```json
{
  "status": "passed",
  "synthetic_tests": {"tests": 45, "failures": 0, "errors": 0, "skipped": 0},
  "training_started": false
}
```

The actual output includes additional provenance fields. On 1 October 2026, the
repository check passed in **0.573 seconds**; an independent temporary extraction
of the existing coursework archive passed in **0.426 seconds**. Both used Python
3.11.15 and NumPy 1.26.4 on the existing local environment. Those measurements do
not establish performance or dependency compatibility on another machine.

Passing this check confirms the small mathematical examples and artifact
integrity. It does not reproduce the full experiments or certify every model's
accuracy. The completion results use a previously exposed validation-calibration
cohort, as the output explicitly records.

## Find the part you can own

Choose a question from the [three proposals](../PROPOSALS.md). Start with
the [results summary](../RESULTS.md) before exploring older research.

Paths below are relative to `coursework/code/` in the pack, or `final-project/`
in a repository checkout.

| Starting point | What to inspect or improve first | Existing check |
| --- | --- | --- |
| `metrics.py` | Work one nDCG@10 example by hand; check exclusion of seen items | `tests/test_metrics.py` |
| `hybrid_constraints.py` and `study.py` | Explain what the sum-to-one constraint permits and how coefficients are selected | `tests/test_hybrid_constraints.py` |
| `coursework_completion/models.py` | Trace mixed lists, meta-level fitting, switching and reciprocal-rank fusion on toy inputs | `coursework_completion/test_models.py` |
| `societal.py` | Explain a diversity/exposure tradeoff and the assumptions behind group definitions | `tests/test_societal.py` |
| `coursework_completion/COVERAGE.md` and `coursework_completion/TEAM_REVIEW.md` | Map assignment requirements to implemented evidence and identify unclear report claims | Five-page report and the coverage checklist |

All five people should first run the smoke check and read the report. Claim a
specific review or improvement with an observable output, such as a checked
equation, a corrected paragraph, or a new diagnostic with a stated hypothesis.
Record actual work when it happens; proposed ownership is not a contribution
claim. Keep experiment changes and their outputs in a new location so that the
existing evidence remains reproducible.

## Full experiments are a separate next step

The small smoke environment intentionally contains only NumPy. Full model runs
need the project's recorded dependencies and instructor RecBole checkout.
Follow [setup](../SETUP.md) for installation and
[REPRODUCE.md](../../REPRODUCE.md) for data and replay commands. Use the recorded
instructor commit `081c3f6edf8e466d3ed5e163631a1afb6fe892bf`; the dependency file
records a Python 3.11 macOS environment and does not guarantee every platform.

In the meeting pack, those same guides are `coursework/code/README.md` and
`coursework/code/REPRODUCE.md`. Raw ratings and saved user predictions are not
included. The report and public aggregate evidence are available without them.

The separate conditional-evidence study resumed on 1 October at 12:23 UTC with
six workers and monitoring; final results remain pending. This quickstart does
not start or alter it. Choose any next research experiment
after agreeing on the hypothesis, evaluation split, compute budget and owner.
