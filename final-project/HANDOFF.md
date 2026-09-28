# Start here

The project has moved beyond a scaffold. The local implementation includes expert
training/tuning, several hybrid architectures and loss ablations, independent
metrics, rerankers, fairness diagnostics and a three-split experiment suite.

## Read in this order

1. [Results and four-panel research figure](evidence/research-v2/RESULTS.md).
2. [Research questions, architecture and methodological limitations](RESEARCH.md).
3. [Five-person roles and official requirement mapping](PLAN.md).
4. [Commands and environment notes](README.md).

## What the experiments currently support

- Expert grids: 33 model runs across three data seeds. Selected experts feed
  hybrid/ablation/reranking studies with 64 evaluated variants per seed.
- Best mean selected development nDCG comes from the item-conditioned ridge
  variant (about 0.2648), versus 0.2618 for the best standalone expert. Full
  context is similar (about 0.2647), so extra user/disagreement complexity has
  not established additional value. Pairwise regression was worse here.
- Selective candidate expansion matched full-pool exposure-reranking top-10
  lists for every development user in these runs, using mean pools around
  109–111 items. This is a reranking-stage candidate result, not a measured
  total pipeline speedup or a universal guarantee.
- A group utility constraint fitted on meta-fit users did not protect sparse
  history users sufficiently on development data. The group-retention plot
  exposes this failure; do not describe the policy as proven fair or safe.

These results were selected on development users. They are not test performance,
statistically established superiority, a novel-research priority claim or a grade
prediction. Negative findings are retained in the evidence.

## Minimal immediate input from you / the group

Confirm the five role owners and read the architecture/results before presenting.
Use the lecturer feedback session to resolve the report page limit and expected
fairness definitions. The technical build does not need to wait for that meeting.

## Next research work with the highest value

1. Independently audit the metric/ranking implementation and reproduce from a
   clean personal environment. Verify the updated instructor code and pin it.
2. Investigate why item-conditioned weights capture most of the gain. Inspect
   coefficient stability and per-user winners, not only aggregate nDCG.
3. Replace the optimistic group-utility budget with a policy fitted on a separate
   calibration cohort or conservatively bounded group losses; evaluate sparse
   users explicitly. Avoid selecting a fairness claim by its prettiest average.
4. Add a chronological split sensitivity experiment. Course-compatible random
   splitting alone does not establish realistic deployment performance.
5. Freeze a design, implement its saved-parameter test inference, then evaluate
   test once. Convert evidence into the required task-by-task report and package.

## Local and publishing state

Work is under `final-project/` on branch `project/starter`. Raw runs/checkpoints
are ignored. Aggregate evidence and figures are included for review. The unrelated
Assignment 1 notebook edits are outside the project changes.

GitHub publishing remains blocked by this session's approval policy. No remote
push or course submission has been made. Once authenticated GitHub writes are
available, publish the branch using the normal personal Git workflow.
