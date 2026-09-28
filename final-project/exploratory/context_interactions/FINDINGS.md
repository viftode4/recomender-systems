# Independent interactions did not produce the needed improvement

The exact singleton-plus-pair reconstruction model completed all 135 declared
configurations across three reused MovieLens 100K splits, plus selected-model
replay. Independent verification reproduced selected predictions exactly and
ranking/group metrics within 8e-15. This was a test of added interaction
capacity under the same reconstruction target, not a breakthrough claim.

| Selected role | Mean all-observed nDCG@10 | Mean liked nDCG@10 |
|---|---:|---:|
| Binary control | 0.261688 | 0.245301 |
| Pair-only family | 0.261660 | 0.245474 |
| Binary/pair nested family | 0.261735 | 0.245567 |
| Previously locked categorical model | 0.263154 | 0.246720 |

The pair-only family changes primary nDCG by approximately -0.0000274.
Allowing fallback to binary changes it by +0.0000469, with mean recall, MRR
and coverage slightly lower than binary. These are descriptive means on
overlapping, repeatedly used development splits. No fresh TEST was opened.

Pair-only selection uses weights 0.1, 0.01 and 0.01 for seeds 2026, 2027 and
2028, with lambda 300 in all three. The nested family selects binary in 2028.
The useful pair contribution is therefore strongly restricted by meta-fit
selection; the larger weights were not selected. Binary has nine candidates,
pair-only 36, and the nested family 45. Unequal search opportunities remain
part of the comparison.

## What this rules out, and what it does not

The new model can assign independent signed coefficients to every distinct
source-item pair and every candidate. Small, explicitly enumerated problems
verify this capability and exact removal of every target-derived feature.
The solver reaches the regularized reconstruction solution without iterative
neural training. Insufficient SGD training and the old addressed model's
products of unary weights therefore do not explain this particular result.

The result does not establish that all nonlinear models fail, that all pair
interactions are uninformative, or that more training data cannot help. It does
not isolate overfitting as the cause. Kernel scaling, regularization, the
reconstruction objective, and the available sample remain possible limits.
No information-theoretic ceiling has been measured.

The practical implication is to stop treating interaction capacity alone as a
demonstrated route to a large improvement. The separate
[error diagnosis](../research_diagnosis/results-v1/RESULTS.md),
[tail support analysis](../research_diagnosis/TAIL_SUPPORT.md), and controlled
donor-data experiment investigate different bottlenecks. Any richer signal or
new objective needs its own comparison; changing the task is not an accuracy
gain on the original task.

## Evidence

- [Declared protocol](PROTOCOL.md) and [exact derivation](DERIVATION.md).
- [Full measured results](results-v1/RESULTS.md), with aggregate/group details
  in the accompanying JSON.
- [Independent audit](audit-v1/RESULTS.md): all 135 grid rows, selected fits,
  independent direct target solves, 2,580 group values and 30 comparisons.
- Seventeen model and runner tests passed, including independently constructed
  primal/kernel references, forced numerical fallback, and evaluation boundaries.

The original report and review bundle remain unchanged. This negative result
is retained because it changes the research direction, not because it satisfies
the project's ambition for a large improvement.
