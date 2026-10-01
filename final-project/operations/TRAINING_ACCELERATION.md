# Conditional-evidence training acceleration

Activated on 1 October 2026 at 12:23 UTC after the user requested resumption.
The study continued from its durable checkpoints with six workers; subsequent
completed epochs and new checkpoints were observed. The original experiment's
scientific source files, model, data, optimizer, episode schedule, candidate
chunks and evaluation rules are unchanged. This is a dated execution record;
the local progress and monitor files show current process state.

## What changes

The original backward routine builds every candidate graph and executes its
forward pass twice: once to construct the complete query loss, then again to
backpropagate each chunk while limiting activation memory.

The optional accelerator builds and forwards each chunk once, retains the outputs
for one query, computes the same loss derivative on detached logits, and
backpropagates the saved outputs in the original order. It releases each chunk's
autograd graph after backward. Eight queries still form one Adam update, with
64-candidate chunks and the same examples, negatives, seeds and learning rates.

This trades additional memory for reduced computation. Physical graph-build
counters halve; they count actual work rather than distinct examples. The number
of training examples and optimizer updates does not change.

## Measurements

Copies of actual seed-2026 checkpoints were continued for 32 TRAIN episodes per
method. Three repeat pairs alternated execution order. Each pair began with
identical model parameters, Adam state and RNG state. The table reports median
time; the existing six-worker study was running concurrently.

| Arm | Original | Accelerated | Observed speedup |
|---|---:|---:|---:|
| Raw graph reader | 2.766 s | 1.670 s | 1.66x |
| Scrambled graph control | 5.542 s | 3.063 s | 1.81x |
| Summary control | 0.340 s | 0.193 s | 1.76x |

Losses, final gradients, model tensors, Adam tensors and RNG states matched
exactly for every pair. These are bounded throughput measurements, not a measured
speedup of the whole remaining study. Selection evaluations and other work are
unchanged. An independent 16-episode pilot from initialization observed similar
speedups. No validation labels or development outcomes were used by these pilots.
The copied checkpoint payloads include historical selection metadata, which the
continuation verifier does not consult.

A separate-process stress check used the largest seed-2026 TRAIN history (591
records), producing contexts of 472 and 531 records and four/three candidate
chunks. Peak process memory rose from 651 MB to 966 MB; the two-query run was
1.49x faster with identical model, Adam and RNG tensors. These process peaks
include imports and TRAIN loading and are not bounds for every epoch or for six
concurrent workers. The six-worker count is therefore unchanged.

Evidence:

- `runs/conditional-evidence-speedup-resume-verification.json`: trained-checkpoint
  continuation, per-repeat timing, checkpoint hashes and source bindings.
- `runs/conditional-evidence-acceleration-v1/parity.json`: five synthetic tests
  covering all arms, ordered gradients, multiple Adam steps and checkpoint resume.
- `runs/conditional-evidence-acceleration-v1/timing/benchmark.json`: independent
  TRAIN-only timing and parity pilot.
- `runs/conditional-evidence-speedup-builder.json`: separate graph-builder profile.
  Skipping unused summaries is a measured possible follow-up, not part of the
  activated acceleration implementation.
- `runs/conditional-evidence-acceleration-v1/largest-history/parity-memory.json`:
  separate-process memory check on the largest available TRAIN history.

The acceleration helper is
[conditional_evidence_acceleration.py](conditional_evidence_acceleration.py).
The independent continuation verifier is
[verify_training_acceleration.py](verify_training_acceleration.py).

## Scientific and operational limits

This optimization relies on the reader's existing deterministic, stateless
forward contract. It is not automatically valid for dropout, mutable batch
normalization or a changed numerical backend. Parity must be tested again if
those assumptions change.

The original source guards must continue to pass. A separate execution receipt
must identify the accelerator, launcher and verification evidence; the original
source inventory alone does not describe the overridden computation. New
checkpoints must atomically include that extra provenance while preserving their
original model/configuration guard. Timing, counters and provenance differ, so
complete checkpoint files are not expected to have identical hashes.

The environment reports 16 logical CPUs, MPS compiled in but unavailable, and
CUDA unavailable. No GPU speedup has been benchmarked. This change retains the
original CPU backend, one numerical thread per worker and six-worker schedule.

## Resume from a personal terminal

The launcher preflights the original inputs, source hashes, runtime and tested
accelerator before any interruption. It verifies the recorded workflow command,
PID and start time, sends SIGINT only to that parent, and waits at most 60 seconds
for it and its owned workers to exit. It refuses unverified or forced takeover.
Completed epochs are checkpointed; at most one in-progress epoch per worker is
repeated. Original latest checkpoints are copied before continuation.

From the project directory:

```sh
cd /Users/vliftode/personal/recomender-systems/final-project
runs/environment-check/.venv/bin/python -u -m operations.accelerated_conditional_evidence --background --resume --workers 6
```

Run this only after the study has stopped; the study lock prevents duplicate
workers. `--takeover` is only for replacing the original, non-accelerated workflow
whose identity matches its recorded launch. Check the returned startup status
and `runs/conditional-evidence-v1.log`.
`runs/conditional-evidence-v1-acceleration-active.json` records installation;
`runs/conditional-evidence-v1-acceleration-launch.json` records the observed
background startup status. A process merely starting is not proof of training.

After an accelerated run stops, resume with this launcher and `--resume`, omitting
`--takeover`. Do not use the original workflow command: it does not enforce or
preserve the execution-overlay provenance. Failure progress records now provide
the accelerated resume command. Original checkpoint guards are retained, while
new checkpoints atomically include the extra execution receipt. Spawned workers
inherit and verify the same manifest. Final exports receive an independently
hashed `execution-overlay/` supplement with the effective source files.

The launcher does not signal or replace the separate read-only monitor. If that
monitor exits during interruption, it can be restarted independently using the
existing monitor command. It refuses to duplicate a monitor that still holds its
lock. Workflow progress and training logs remain available regardless.

Validation: 40 related numerical, runner, model, checkpoint/provenance and
process-safety tests passed, plus the actual-checkpoint continuation and largest-
history memory pilots above. Process signals in launcher tests were mocked;
real takeover could not be exercised in the restricted sandbox.
