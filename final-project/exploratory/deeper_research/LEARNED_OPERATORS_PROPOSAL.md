# Learn the operations as well as the program

Status: proposal only. No implementation or experiment is authorized by this document. Existing models, stress results, and sealed sources remain unchanged.

The next useful claim is narrow: **can a small shared library of learned functions make unfamiliar tasks easier to solve by composing those functions?** This requires learning the content of an operation, not merely choosing among our supplied max/product rules. It does not establish a system for every task type.

## One executable model

Start with four anonymous, trainable functions of two real arguments. They receive no labels such as addition, conjunction, similarity, or exception:

\[
g_k(a,b)=\alpha_k a+\beta_k b+c_k+
\sum_{h=1}^{16}v_{kh}\tanh(w_{kh1}a+w_{kh2}b+d_{kh}),
\qquad k\in\{1,2,3,4\}.
\]

There are 67 parameters per operation and 268 shared parameters altogether. All parameters start from seeded random initializations. The direct affine terms provide a modest smooth-function bias; they do not encode multiplication, max, or named task operations.

A task program is an ordered expression tree:

\[
T ::= x_j\;|\;0\;|\;1\;|\;g_k(T_1,T_2).
\]

Use four raw input coordinates and at most three operation calls. Argument order matters. A task has just two continuous calibration parameters:

\[
f_t(x)=b_t+s_tT_t(x).
\]

The same operation is the same learned function everywhere it is called, across tasks and across positions within a program. Prediction executes **one selected tree**. It contains no attention mechanism, weighted mixture of trees, pretrained predictor, or input-dependent soft choice among operation outputs.

The quantities learned are (1) shared operation contents, (2) task-specific composition and argument bindings, and (3) the small output calibration. We still supply real-valued inputs, binary arity, an activation family, bounded search, and a loss. Learning without any representational assumptions is not a coherent claim.

## Learning without operator names or program supervision

A training task supplies only input/output examples. Split its examples into construction set A and learning set B. Neither a generator expression nor an operation ID enters the learner.

1. Hold the current operation library fixed. Generate a bounded pool of programs using A only. Fit each program's two output coefficients on A. Keep one task program using A loss plus its declared complexity penalty.
2. Execute that program on B. Update the shared operation parameters through its actual function calls using B predictive loss. The discrete program choice and fitted output coefficients are held fixed for this update.
3. Periodically rebuild the task programs with the changed library. Repeat across many training tasks.

This is an explicit alternating optimization algorithm, not a claim to differentiate through discrete search or obtain a global optimum. In equations:

\[
(T_t,b_t,s_t)\leftarrow
\arg\min_{T,b,s}\left[\bar\ell_A(b+sT_\theta(x),y)
 +\frac{\beta |T|}{|A|}+\eta(b^2+s^2)\right],
\]

\[
\theta\leftarrow\theta-\gamma\nabla_\theta
\left[\frac1{|\mathcal T_{\rm batch}|}\sum_t
\bar\ell_{B_t}(b_t+s_tT_{t,\theta}(x),y)+\lambda\|\theta\|_2^2\right].
\]

Here |T| counts operation calls. B is training data for the shared library, so it must never be called an untouched evaluation. A separate set of meta-development tasks selects the population checkpoint and the fixed regularization settings.

At an entirely new task, freeze the library. FIT alone constructs programs and calibrates their outputs. Disjoint PROBE examples select one program, including zero-operation constant/coordinate candidates. Serialize that selected predictor before accessing final evaluation examples. Use a complexity cost proportional to |T|/|PROBE|, with its coefficient chosen only on meta-development tasks. This is a declared sample-size-sensitive penalty, not a proven generalization bound or a guarantee that short histories are safe.

The prior 90-case stress result motivates this separation: with 16 FIT and 16 PROBE examples, search overfit even the no-signal task. This proposal therefore does not allow unrestricted task-specific training of the operation library at evaluation time.

## A benchmark that does not secretly provide the answer

Keep the data generator in an independently specified module that does not import the model, operation class, or search grammar. The learner receives arrays and task boundaries only. No expression strings, task-family labels, primitive IDs, intermediate values, or execution traces are model inputs.

Use two benchmark tracks. The first establishes compositional transfer under a deliberately favorable structural assumption; the second exposes that assumption.

**Held-out compositions.** Define analytic function families independently of the neural operation parameterization, using bounded arithmetic, smooth nonlinearities, and input interactions. For example, separate analytic definitions of normalized sums, products, squares and sinusoidal responses can generate tasks whose intermediate ranges remain controlled. The model does not receive these operations; its only operations are the anonymous learned functions above.

Training tasks contain multiple composite families and randomized input-coordinate permutations. Avoid a curriculum of named primitive demonstrations. Reserve specific parent/child operation relationships and complete compositions for evaluation while ensuring that their constituent analytic behavior occurs elsewhere in training. Keep numerical input and intermediate ranges overlapping so compositional transfer is not confounded with unannounced numerical extrapolation. Vary output scale/offset independently within a fixed range.

This track still favors compositional models because its generator is compositional. State that explicitly. It does not demonstrate general-purpose operator invention. The exact function roster, held-out relationships, seeds, and ranges must be frozen by the benchmark owner before any fitting.

**Different and misspecified families.** Include functions sampled by a mechanism that does not use our program library: smooth random Fourier expansions or Gaussian-process draws, piecewise functions with unrelated boundaries, and pure-noise outcomes. Use random orthogonal input rotations for an additional diagnostic, because axis-aligned leaves can otherwise make our decomposition artificially easy. These are transfer/failure controls, not cases to discard if the dense reference wins.

Prevent accidental split leakage in two ways: generator-graph separation and numerical function fingerprints on an independent fixed grid. Near-identical functions produced by argument swapping, algebraic equivalence or output rescaling must not appear on both sides under different names. Fingerprint checks are approximate; they do not prove symbolic nonequivalence. Generator provenance and split information remain available to the evaluator but inaccessible to the learner.

Task-level splits must be made before generating training examples: proposed pilot 96 training tasks, 24 meta-development tasks, and 48 evaluation tasks per declared track. At evaluation, cross FIT/PROBE sizes 16, 64 and 256 with noise levels 0, 0.05 and 0.15, using five task-generation seeds. Include all failures. These counts are a proposed budget, not a completed preregistration.

Use continuous regression first and binary outcomes as a second output schema. The former uses squared error or a declared Gaussian likelihood; the latter uses Bernoulli likelihood, with examples sampled from a bounded latent response through a logistic link. The link/loss follows the observed output schema; it is not an operator-identity hint. This covers two supervised prediction settings, not text, images, control, or universal intelligence.

## Controls that could defeat the proposal

**Strong dense reference.** A plain neural function of x plus an eight-dimensional fast task code, with widths 12→12→8→1, has 269 shared parameters, almost exactly the library's 268. Learn its shared weights on the identical training tasks. On a new task, optimize the code and output calibration on FIT only; PROBE selects among its predeclared adaptation checkpoints. It also returns one predictor. Supply the same task examples and population-training opportunities. This is a substantive alternative explanation: perhaps useful shared features plus coefficient adaptation suffice.

Equal parameter counts do not imply equal compute. Before seeing predictive metrics, benchmark both implementations on dummy arrays and fix adaptation budgets with similar measured CPU time and the same maximum PROBE selection opportunities. Report total task-specific forward/backward work, candidate counts, wall time, and learning curves for both. Permit the dense reference multiple seeded FIT restarts within that budget. Do not compare hundreds of program trials against one unadapted dense pass.

**Frozen random operation bank.** Freeze an identically sized random bank and use exactly the same program search/calibration/selection protocol. This isolates whether learning operation contents contributes beyond random nonlinear features and composition search. Its random initialization is fixed prospectively, not selected after results.

**No composition.** Use the trained operation bank but allow at most one operation call, with the same search budget ceiling and output calibration. This tests whether repeated composition adds anything. Allocated parameter capacity remains the same; executable depth does not.

**No shared learning.** Fit task-specific operations from scratch within the same adaptation budget, where practical. This asks whether the population library transfers useful knowledge. It may overfit short contexts; that failure is informative and must remain visible.

The minimum first pilot needs the full model, dense reference, frozen-random bank and no-composition ablation. The task-specific-bank control can follow only if the fixed pilot is computationally feasible; it must not be added selectively after outcomes.

## What would count as evidence

The primary endpoint is held-out task predictive loss, summarized over task seeds, not over thousands of correlated evaluation points as if they were independent model trials. Retain per-task differences and compute cost. Report short/noisy/no-signal cases alongside the attractive compositions.

Required outcomes for a positive mechanism claim:

1. Learning the operation bank improves held-out tasks over the frozen-random bank under matched search.
2. Composition improves held-out compositions over the one-call ablation, without requiring more per-task examples or hidden labels.
3. The full method improves over the strong dense adaptation reference at a comparable compute budget; otherwise program structure is not established as necessary.
4. The system can remain simple or fail transparently on short/no-signal tasks, rather than interpreting a low PROBE loss as proof of a discovered explanation.

Inspect each learned operation on a fixed two-dimensional grid and count its reuse across unrelated tasks. Measure operator collapse and whether both arguments actually matter. Do not name an operation "multiplication" merely because one contour plot looks familiar. Equivalent parameterizations and overloaded operations are possible; predictive transfer does not identify human meanings.

Before global training, run only a tensor-throughput benchmark and deterministic tiny tests: operation gradients change function values; swapping operation-bank labels and program references preserves predictions; held-out labels cannot change proposed programs or FIT calibration; serialization exactly replays hard execution; and a task cannot pass an example's target into its own input. A first bounded run should use one numerical CPU thread and three model initializations. Runtime is unmeasured; the 268-parameter functions are small, but repeated search can dominate. Set the actual wall-time ceiling after the premetric throughput check, not after a poor score.

## Direct predecessors and claim boundary

- [Neural Language Interpreter, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9cde817d04811ba28e44071bd9f76a5-Abstract-Conference.html) already learns a vocabulary of neural primitives and discrete compositions, with a differentiable executor and test-time program refinement. It is a direct antecedent. Our proposed hard search and short-data stress protocol do not make the broad idea new.
- [DreamCoder](https://arxiv.org/abs/2006.08381) learns reusable symbolic abstractions and program-search knowledge across tasks. Its library learning directly precedes the shared-vocabulary ambition.
- [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279) learn compositional program execution, using execution-trace supervision in the original work. Our proposed absence of such supervision is a methodological distinction, not a novelty claim.
- [Neural Programmer](https://research.google/pubs/neural-programmer-inducing-latent-programs-with-gradient-descent/) induces programs from final answers using built-in operations. Learning operation contents would go beyond that specific supplied-operation setup, but not beyond the later learned-language literature.

The defensible goal is an independently implemented, understandable experiment about when learned operations and hard composition transfer. A universal architecture or a first-ever idea is not established by this proposal.
