# Tests

For the first check, run `make check` from the repository root after the
[quickstart setup](../docs/QUICKSTART.md). It checks current documentation and
runs 45 small algorithm tests plus saved-artifact integrity checks.

## Find the relevant tests

| Area | Tests |
| --- | --- |
| Accuracy, novelty and coverage | [test_metrics.py](test_metrics.py) |
| Hybrid regression constraints | [test_hybrid_constraints.py](test_hybrid_constraints.py) |
| User groups, exposure and calibration | [test_societal.py](test_societal.py) |
| Mixed, meta-level, switching and rank fusion | [test_models.py](../coursework_completion/test_models.py) |
| Frozen evaluation and data boundaries | [test_freeze.py](test_freeze.py), [test_study.py](test_study.py) |
| Report assembly and package navigation | [test_report.py](test_report.py), [test_build_review.py](../coursework_completion/test_build_review.py) |

To run one file, use the matching environment from `final-project/`:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_metrics.py' -v
```

The full suite needs the [full environment](../docs/SETUP.md); the small NumPy-only
setup covers the starter check. Tests beside an exploratory model belong to that
study. Running tests does not establish that every historical training run reproduced.
