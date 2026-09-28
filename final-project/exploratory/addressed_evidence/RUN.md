# Run and recover the long study

The study trains both variants from scratch for each of seeds 2026, 2027 and
2028, with three CPU workers. Each variant can use up to 2,000 epochs under the
stopping rule in [PROTOCOL.md](PROTOCOL.md). The benchmark measured approximately
3.2 seconds per additive epoch and 4.1 seconds per pair epoch in isolation.
Those are timing observations, not a promised completion time under contention.

From the final-project directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
runs/environment-check/.venv/bin/python -u \
  -m exploratory.addressed_evidence.run_experiment run \
  --source-root runs/research-v2 \
  --ratings /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.inter \
  --item-metadata /Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out runs/addressed-evidence-v1 \
  --evidence exploratory/addressed_evidence/results-v1 \
  --workers 3 --seeds 2026 2027 2028 \
  --max-epochs 2000 --checkpoints 10 30 60 100 \
  --evaluation-interval 20 --patience 200 --min-improvement 0.0001
```

The output directory must be new. The runner writes its protocol and source
signatures before fitting. Once started, keep the files listed in the protocol's
`source_sha256` unchanged. Changes require a separately declared run.

Saved progress can be inspected without loading models or rating records:

```sh
python3 -m exploratory.addressed_evidence.status --run runs/addressed-evidence-v1
```

This reports the last durable checkpoint, not process liveness. Checkpoints
arrive at the scheduled evaluation epochs, so an unchanged file between those
epochs is normal. The actual launch also redirects console output to
`runs/addressed-evidence-v1.log`.

After an interruption, rerun the training command with `--resume` appended,
after ensuring the original process has stopped. Resume restores the optimizer,
model, deterministic epoch index and selection state; it rejects changed source,
runtime, input or protocol signatures. It recovers from the latest complete
checkpoint, so work since that checkpoint may repeat. Do not launch two writers
against the same run directory.

When all seeds finish, the runner verifies artifact hashes and writes aggregate
evidence and a readable results table to `exploratory/addressed_evidence/results-v1`.
Until those completion artifacts exist, there is no completed model comparison.
The results remain exploratory on reused development data. They cannot replace
the original final-test evidence or establish fresh confirmation.
