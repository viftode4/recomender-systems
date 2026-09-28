# What the fixed-budget pilot resolves

The substantial-advance goal remains unmet. On reused development data, the
full model reaches nDCG@10 **0.245189**, versus **0.238678** without the two
pattern channels and **0.261688** for locked EASE. Full beats no-pattern in
every seed and loses to EASE in every seed. This supports the narrower value of
the supplied pattern features under this budget; it does not establish a better
overall recommender. The pooled tail result is 41 hits versus 30 without patterns
and 25 for EASE, over 4,734 tail positives in overlapping splits. These are small
counts, not independent replications.

This note reads existing checkpoint metrics and online training traces. It does
not refit a model, open TEST, change checkpoints, or supersede the frozen pilot.

## Improvement continues, but slows

Mean meta-selection nDCG across the three seeds:

| Arm | Epoch 10 | Epoch 30 | Epoch 60 | Epoch 100 |
|---|---:|---:|---:|---:|
| Full pattern | 0.228544 | 0.240566 | 0.245693 | 0.249682 |
| No pattern | 0.224621 | 0.236896 | 0.237864 | 0.241553 |

For the full model, mean improvement per ten epochs falls from **0.006011**
over epochs 10–30 to **0.001709** over 30–60 and **0.000997** over 60–100.
Late improvement is uneven:

| Seed | Full, 60→100 | No pattern, 60→100 |
|---|---:|---:|
| 2026 | +0.004398 | +0.005635 |
| 2027 | +0.007456 | +0.004378 |
| 2028 | +0.000114 | +0.001053 |

All six full/no-pattern fits select the epoch-100 budget boundary. The third
full-model seed is nearly flat late in training. The no-pattern trajectory is
not uniformly smooth: seed 2026 drops from epoch 30 to 60 before recovering.
Five widely spaced checkpoints cannot establish a stable asymptote.

The mean online TRAIN loss still decreases: full drops by **0.015380** from
epoch 60 to 100 and **0.002826** from 90 to 100; no-pattern drops by **0.013212**
and **0.002496**, respectively. These are online minibatch losses, not an exact
fixed-state objective evaluated after every epoch. Both loss and checkpoint
ranking improvement leave optimization unresolved, while their slowing gives
no basis for promising that more epochs will close the EASE gap. Meta-selection
and development are different cohorts; their numerical gaps cannot be directly
extrapolated into a forecast.

The full-minus-no-pattern meta gap grows from **0.003670** at epoch 30 to
**0.007829** at 60 and **0.008130** at 100. The evidence does not suggest that
the pattern channels were simply ignored. The marginal-only control has an
unchanged ranking from epoch 10 onward even as its loss slightly improves,
which also illustrates that lower likelihood loss need not improve ranking.

## Sensitivity to removing pattern inputs

The predeclared inference-only removal of both pattern channels gives 0.209724
development nDCG and 13.602% tail slots, versus 0.245189 and 2.394% for full.
Both have 41 pooled tail hits. Thus this changed-input intervention greatly
increases tail exposure without additional tail hits. It measures sensitivity
to disrupting jointly learned inputs, not the effect of retraining without them.

## Next decision

First resolve the optimization question with one separately declared, bounded
convergence study of the **same fixed features and same full/no-pattern arms**.
Keep architecture, initialization, episodes, optimizer and selection metric
unchanged, retain the original checkpoints as stopping options, and preserve
this pilot. A fresh deterministic restart is needed for exact continuation
because the stored checkpoints contain network weights but no Adam state;
verify the first 100 epochs against the original traces and saved states.
Any new budget and stopping rule must be fixed before that study runs. No
extension has been launched.

Learning from richer relational inputs is a separate structural question.
The nine summaries discard which particular context relations produced the
same aggregate evidence. More layers cannot recover information already lost
in those summaries. If convergence fails to close the gap, a later declared
experiment should ask whether richer relational input improves predictions
under matched optimization and controls. The current result neither proves
that this information loss caused the EASE gap nor justifies treating another
larger scorer as a solution.

Source evidence: the frozen `runs/evidence-transfer-v1/{2026,2027,2028}/`
`checkpoints/{full_pattern,no_pattern,marginal_only}/checkpoint-metrics.json`
and `epoch-metrics.json`; aggregate comparisons are in
[analysis-v1/RESULTS.md](analysis-v1/RESULTS.md). All findings are exploratory
on reused data, not fresh confirmation.
