# Project tools

For getting started, use `team_smoke_check.py`. From `final-project/` with the
environment in [the setup instructions](../README.md):

```sh
.venv/bin/python operations/team_smoke_check.py
```

It runs the fast synthetic examples and checks the saved report/evidence. It
does not train a model. The root Makefile provides the equivalent `make check`.

To regenerate the offline meeting pack, install `requirements-handoff.txt` and
run `make handoff` from the repository root. The builder reads the tracked,
verified coursework ZIP; it keeps every original scientific artifact unchanged
and adds the current proposal/starting documents. Generated ZIPs/previews stay ignored.

## Optional local study monitoring

`conditional_evidence_monitor.py` is outside the frozen scientific source. It
checks the existing study every five minutes without modifying training,
selection, checkpoints, or the experimental protocol.

It checks the workflow lock, source hashes, log errors, finite losses and each
active trajectory's checkpoint age. A checkpoint is flagged after the greater of
30 minutes or six times its recent epoch plus validation duration. Completed and
queued runs are excluded. The monitor records warnings; it never kills or
restarts a worker based on age alone. Retrieval label construction lacks a
per-query heartbeat, so its duration is displayed without a false stall claim.

Local files, relative to `final-project/`:

- `runs/conditional-evidence-v1-monitor.json`: latest health check and trajectory progress.
- `runs/conditional-evidence-v1-monitor-events.jsonl`: status changes and alerts.
- `runs/conditional-evidence-v1-monitor-launch.json`: monitor PID and exact launch command.

The monitor exits when the workflow fails, stops, or completes. At completion it
checks public result hashes. This process does not send chat notifications and
does not survive reboot. After explicitly resuming the study, restart it with:

```sh
runs/environment-check/.venv/bin/python -u -m operations.conditional_evidence_monitor \
  --study runs/conditional-evidence-v1 --interval 300
```

Add `--once` for a single snapshot when another monitor is not already running.
The monitor's own lock prevents duplicates. Eight synthetic tests cover healthy,
stale, queued, extended, failed-integrity and completed states, partial log lines,
non-finite values and stopped processes:

```sh
runs/environment-check/.venv/bin/python -m unittest operations.test_conditional_evidence_monitor -v
```
