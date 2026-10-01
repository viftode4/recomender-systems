# Working together

1. Pick a task and reviewer in the [team plan](final-project/docs/team/TEAM_PLAN.md).
2. Run the [first check](README.md#run-the-first-check) and read the [results](final-project/docs/RESULTS.md).
3. Make one focused change on a branch, such as `team/hybrid-example`.
4. Record the result or correction using the [review template](final-project/docs/team/REVIEW_TEMPLATE.md).
5. Run `make check`. If you changed scientific code, also run its relevant tests.
6. Open a pull request explaining what changed, why, how you checked it and
   whether it affects reported results. Ask your reviewer to check it.

Save review notes in `final-project/docs/team/reviews/`.
Each person writes and explains their own contribution; the report owner
coordinates the final document.

## Experiments and saved results

Use a new output directory for each experiment. Preserve the frozen source and
result snapshots. The original test results are already known; new model choices
need a declared evaluation protocol and reused-data studies must be labelled exploratory.

Keep raw ratings, per-user predictions, checkpoints, environments and credentials
out of Git. The tracked `24.zip` is a verified code/report snapshot. Build new
previews with `make handoff`; publishing or submitting them is a separate step.

Optional research has its own [status and protocols](final-project/docs/RESEARCH_INDEX.md).
Agree the hypothesis, owner and compute budget before starting a new experiment.
Each member completes their own peer feedback based on actual contributions.
