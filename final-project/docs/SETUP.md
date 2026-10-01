# Set up the full coursework environment

For the first team session, use the [quickstart](team-meeting-2026-10-01/QUICKSTART.md).
It needs only Python 3.11 and NumPy and starts no training. These instructions
are for running the full project later, from the `final-project/` directory or
the extracted coursework archive's `code/` directory.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.lock.txt
mkdir -p vendor
git clone https://github.com/masoudmansoury/RecBole_DSAIT4335.git vendor/RecBole_DSAIT4335
git -C vendor/RecBole_DSAIT4335 checkout 081c3f6edf8e466d3ed5e163631a1afb6fe892bf
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e vendor/RecBole_DSAIT4335
```

Use your personal GitHub context. The tested dependency versions describe a
Python 3.11 macOS environment; they do not establish cross-platform compatibility.
For the original instructor source and data checks, follow
[REPRODUCE.md](../REPRODUCE.md). The local FISM and NGCF adapter changes and the
documented NGCF reproduction discrepancy must remain disclosed.

Limit numerical-library threads before running concurrent experiments:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s coursework_completion -p 'test_*.py' -v
```

Use a new output directory for every new experiment. Do not overwrite the
frozen study or tune models against its already observed test results. Start
with the fixed-data split reproduction in [REPRODUCE.md](../REPRODUCE.md), then
agree on a bounded training command and compute budget with the team.

The paused conditional-evidence study is a separate research workflow. Reading
these instructions or running the quick check does not restart it.
