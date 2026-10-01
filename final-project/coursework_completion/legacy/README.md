# Historical baseline runner

These unmodified `run.py` and `metrics.py` files are extracted from personal
repository commit `3bc481f` (`final-project/`), which produced the first five
frozen baseline exports. Their bytes match those recorded source hashes.
They are retained for reproduction of those fixed configurations only.

The remaining standard models use the current top-level runner. The completion
source rebuild selects the runner from the recorded source hash, never from
which version gives a better metric. It passes an explicit configuration and
uses `--fixed-epochs`; it does not request TEST evaluation.

Both runners use the separately pinned instructor RecBole checkout and external
MovieLens data. This directory contains no third-party package or dataset.
