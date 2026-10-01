# Working together

1. Agree one of the [five seats](final-project/docs/team-meeting-2026-10-01/TEAM_PLAN.md)
   and a review partner. Names in that table are proposals, not completed contributions.
2. Run the [first check](README.md#run-the-first-check), then read the
   [results summary](final-project/docs/RESULTS.md) and report.
3. Take one bounded task: a checked result, a clearer explanation, a bug fix,
   or an agreed experiment. Work on a short branch such as `team/hybrid-example`.
4. Record what you actually checked using the
   [review template](final-project/docs/team-meeting-2026-10-01/REVIEW_TEMPLATE.md).
   Put notes in `final-project/docs/team-meeting-2026-10-01/reviews/`.
5. Run `make check` from the repo root (or its Python equivalent in the README).
   For scientific code changes, also run the relevant model/evaluation tests.
6. Open a focused pull request with the change, its reason, the check performed,
   and any effect on reported results. Ask the agreed partner to review it.

Everyone writes and reviews their own section; the report owner coordinates
the document. Preserve existing source/result snapshots and give new experiments
new output directories. Historical TEST results are already exposed, so new work
needs a declared evaluation protocol and honest exploratory labels.

Keep ratings, per-user predictions, checkpoints, local environments and credentials
out of Git. `runs/`, `vendor/`, environments and generated package previews are
ignored. The single tracked coursework ZIP is an intentionally preserved, verified
public code/report snapshot, not a place to add new data.

Do not restart the paused research as part of onboarding. Agree the hypothesis,
owner and compute budget first. Each person later completes their own peer
feedback based on actual contributions.
