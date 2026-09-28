# Fresh training-environment verification

A new Python 3.11.15 virtual environment installed all 58 pinned packages plus
RecBole 1.2.1 from the personal coursework cache, with the public PyPI index
explicitly selected and network access disabled. `uv pip check` found all 59
installed distributions compatible. User-site packages are disabled.

The complete suite passed: 80 tests, with two PDF-rendering tests skipped because
Matplotlib and pypdf are absent. An isolated 1.25 MB copy of project source, tests
and aggregate fixtures produced the same result. Source hashes are recorded.

This verifies a fresh dependency installation on macOS arm64. The 7.13 MB RecBole
source snapshot was copied from the actual working instructor checkout, and all
38 inspected imported-source hashes match it. This does not verify a pristine
upstream checkout, Linux, or report/plot dependency installation. An offline
attempt to install plotting extras stopped because Matplotlib was not cached.
No scientific test labels were inspected.

Reproduce with a public package cache or normal access to public PyPI:

```sh
uv --no-config venv --python python3.11 .venv
uv --no-config pip install --python .venv/bin/python --default-index https://pypi.org/simple -r requirements.lock.txt
uv --no-config pip install --python .venv/bin/python --default-index https://pypi.org/simple --no-build-isolation -e vendor/RecBole_DSAIT4335
.venv/bin/python verify_environment.py --requirements requirements.lock.txt --out environment.json
.venv/bin/python -m unittest discover -s tests -v
```

The instructor source must first be obtained and pinned as described in the
project README. Add `--offline` and the appropriate `UV_CACHE_DIR` to reproduce
this cache-only installation. Global index overrides must not redirect these
commands to non-public services.
