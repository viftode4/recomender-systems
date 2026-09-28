# Exact shared-operator query experiment, version 1

Declared 28 September 2026 before trials. This is an exploratory finite
mechanism experiment, not a new general architecture or a recommender result.
No MovieLens inputs, checkpoints or final-test results enter this experiment.
Code, runtime, this protocol and fixture encodings must be hashed before trials.
Synthetic correctness checks and runtime pilots must be labelled separately.

The information target is established prior art. [Sloman, Bharti, Martinelli
and Kaski, Bayesian Active Learning in the Presence of Nuisance Parameters,
UAI 2024](https://arxiv.org/abs/2310.14968) studies the tension between target
information and nuisance uncertainty, including transfer learning. The earlier
version used the title The Fundamental Dilemma of Bayesian Active Meta-learning.
We implement our own exact small investigation of that tension, not a novel
acquisition objective.

## Fixture and exact inference

Inputs are all eight Boolean triples in lexicographic order. Each of the 16
binary Boolean truth tables is a possible shared operation f. Record the exact
bit-to-table encoding in the manifest. Grammar descriptors, in fixed order, are:

1. `f(x_i,x_j)` for each ordered distinct pair i,j: six programs.
2. `f(f(x_i,x_j),x_k)` for each permutation of 0,1,2: six programs.
3. `f(x_i,f(x_j,x_k))` for each permutation of 0,1,2: six programs.

Conditional on a shared table, three training-task descriptors are IID uniform
over these 18 choices, **with replacement**. Duplicate syntax and duplicate
semantics are allowed. Do not force coverage or resample worlds with poor
transfer coverage. Those changes would alter the prior and can invalidate the
factorized posterior. The learner sees task IDs, inputs and queried outputs,
not true tables, descriptors, intermediate states or semantic vectors.

Learning an anonymous four-bit operation here is exhaustive finite hypothesis
selection, not continuous parameter training. The interpreter, 16-table search
space and 18-program grammar are supplied by us. This is a small prototype for
an information-acquisition question, not a general engine that invents its own
computational language.

The prior is uniform over 16 concrete tables and independently uniform over
18 descriptors for each task. For observed data D_t, let n_t(f) count the
consistent descriptors. The exact table posterior is proportional to
`(1/16) * product_t [n_t(f)/18]`; conditional on f, descriptors consistent with
D_t are uniform. No learned guide, beam or Monte Carlo approximation is used.

Define Gamma(f) as the smaller encoded table of f and its argument transpose.
Its posterior is the sum of those concrete-table posterior masses. Symmetric
classes have one table and asymmetric classes two; there is **not** a uniform
prior over these classes. After acquisition, select one **concrete-table MAP**,
with ties by ascending table index. Do not substitute class-MAP selection.

## Worlds, acquisition and budgets

- Shared regime: seeds 4100 through 4109 crossed with all 16 true tables,
  yielding 160 training worlds.
- No-sharing regime: ten training worlds, one per seed. Draw the three true
  task tables independently and uniformly from all 16, along with IID task
  descriptors. Evaluate all 16 possible true transfer tables for each world.
  This yields 160 world/transfer-table cells without inventing a shared table.
- In both regimes, supply one uniformly drawn input/output observation for each
  of the three training tasks initially. All arms share the same world and the
  same three initial observations.
- Additional acquisition budgets are 0, 2, 4, 8 and 12. Run each policy once to
  12 queries and snapshot these prefixes. Total training oracle calls are
  exactly `3 + additional_budget`, not the additional budget alone.
- Admissible queries are unobserved `(task,input)` pairs from the 24-pair grid.
  There are initially 21 available pairs. Never query a pair twice.

Compare these policies on the same hypothesis space and inference procedure:

```text
entropy: H(Y_q | D)
focused: I(Y_q ; Gamma(f) | D)
         = H(Y_q | D) - sum_g P(g|D)*H(Y_q | Gamma(f)=g,D)
random:  choose a uniformly randomized remaining query
```

Use base-two entropy and 0 log 0 = 0. For noiseless deterministic hypotheses,
entropy is information about the full table/program hypothesis. For every query
`0 <= focused <= entropy`; rankings can differ. Posteriors differ after policies
acquire different observations. Equal information access means the same initial
observations, oracle and budget, not identical subsequent queried labels.

For deterministic policies, scores within absolute 1e-12 of the maximum tie;
choose the lexicographically first `(task,input)` among them. Random queries use
one fixed uniform permutation of the 24 pairs and take the next unseen pair.
If every information score is zero, deterministic policies still choose the
first unseen pair so budgets remain exact.

Random streams are independent of iteration order and policy execution:
`default_rng(SeedSequence([seed, regime, table_key, stream]))`, where regime
is 0 shared or 1 no-sharing, table_key is the true shared table or zero for
no-sharing, and stream is 0 task descriptors, 1 initial inputs, 2 random-query
permutation, or 3 no-sharing tables. Save the generated world and acquisition
traces so the encoding can be independently replayed.

## Misspecification and failures

The no-sharing regime deliberately violates the shared-table assumption. If
all shared hypotheses become inconsistent, record the first failure step.
The failed learner abstains, with declared bounded prediction loss 1, at that
and all later snapshots. Remaining queries follow lexicographic unseen-pair
order until the budget is spent. Do not reset, soften the likelihood, select
a convenient fallback table, or resample the world.
This tests the current model's inability to reject or replace its sharing
assumption; it does not implement a mechanism that learns when to share.

Shared-regime zero consistency is an implementation error because the generating
hypothesis is in the prior support. Stop and investigate it before reporting
results. A consistent but wrong or nonidentifiable MAP table is not an error to
discard. Report posterior class mass/entropy and failure separately from risk.

## Transfer universe and prediction

For each true transfer table, enumerate the 18 programs' complete eight-bit
output vectors and deduplicate exact semantic aliases. Remove every vector
equal to any of the three actual training-task vectors. Those training vectors
are visible only to the evaluator for defining the target distribution, never
to acquisition, posterior updating, MAP selection or program fitting.

This evaluates **unseen Boolean functions within the depth-at-most-two grammar**.
It does not evaluate greater depth or broad cross-domain reasoning. The filtered
transfer distribution is conditional on training tasks and is not IID uniform
syntax. Report full and remaining semantic-universe sizes for every cell.
Empty universes are explicit N/A with coverage counts, never zero error, never
silent exclusion and never grounds to regenerate a world. In particular constant
shared tables have no unseen semantic transfer function.

For every remaining vector, evaluate all 28 support subsets of size two and all
70 subsets of size four from the eight inputs. Each subset is a separate transfer
episode; reset task-specific state. Freeze the acquired MAP table. Fit one of
its 18 programs using only the support labels, minimizing support error count.
Break ties by: fewer operation calls, lexicographically smaller **model-predicted**
eight-bit vector, then descriptor index. Predicted vectors can be computed
without true evaluation labels; this makes transpose aliases insensitive to
arbitrary syntax order. Do not refit the table, query additional labels, or
choose the program based on errors at the remaining inputs.

Score mean Boolean 0/1 error on the remaining six or four inputs. Nonzero support
error is retained and reported; it is not a reason to reject an episode. The
support allowance is separate from acquisition: label access is `3+b` training
oracle outputs plus `s` support outputs per transfer episode. Evaluation labels
are never used by the learner. Exhaustive support enumeration supplies many
correlated evaluation episodes, not additional independent experimental worlds.

Two references use exactly the same support sets and tie rules:

- Local search: choose from all 16 tables times 18 programs using support only;
  ties are support error, calls, model-predicted vector, table index, then
  descriptor index. It has no shared
  training information but a larger transfer-time search space.
- Oracle-table reference: fit one program with the true transfer table supplied.
  This is a privileged-information reference, not a fair competing learner.

The controls remain evaluable when the shared learner fails. A failed learner's
loss is 1 on every otherwise available transfer episode. Also show conditional
accuracy among nonfailed cases and the failure rate, so this penalty is visible.

## Aggregation, cost and falsification

Average error within a support subset over its unseen inputs, then equally over
all support subsets, then equally over unique transfer semantic vectors. In
each regime, average available seed cells within each true transfer table, then
average the available transfer tables equally. The no-sharing cells still reuse
only ten training worlds and are not 160 independent training replications.
Publish denominators, N/A counts and per-table results.
The same availability set applies to all methods; include failure loss 1 there.
Acquisition failure rates also cover worlds lacking transfer targets.

Report the full prespecified budget by support-size grid of paired
focused-minus-entropy errors and each policy versus random/local/oracle. Lower
error is better. Do not select a favorable budget or support size as the result.
If uncertainty summaries are used, block by world/seed, not by individual support
subsets or query bits. These are finite constructed tasks, not a population of
real recommendation problems.

All methods may share a precomputed table of `16*18*8=2,304` program outputs.
Computing that table requires `16*(6+12*2)*8=3,840` primitive truth-table calls.
Record this common one-time cost and its amortization. Separately count
acquisition candidate scoring, cached hypothesis-output lookups, posterior
updates, entropy computations, transfer candidate fits, and observed wall time.
Queries obtained through any future counterexample finder would count as oracle
calls too; none is allowed in this version.

The primary comparison matches oracle budgets. It does **not** match total
compute: focused information has extra conditional-entropy work, and the local
reference searches more transfer candidates. Do not claim compute efficiency
from fewer labels or from the common output-cache size. Report measured costs
alongside errors rather than imply equality.

A focused-policy win would show useful acquisition choices for this finite
shared-operator setting. It would not establish a new objective or architecture.
A loss, equivalence, or benefit erased by additional compute is a substantive
negative result. No-sharing failures measure fragility under a wrong sharing
assumption, not universal failure of transfer. This pilot cannot establish
human preference mechanisms, SOTA recommendation, or universal learner rankings.
