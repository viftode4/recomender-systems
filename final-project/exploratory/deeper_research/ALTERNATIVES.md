# Learning reusable computation: alternatives and one decisive experiment

Design note, 28 September 2026. No implementation, model fitting or project
assessment access was performed for this note. Existing experiment sources,
weights and reports remain unchanged. This proposal is separate from the running
addressed-evidence experiment and from the completed assignment assessment.

The strongest fit to the requested ambition is a system that **learns executable
operations, stores them, and synthesizes new compositions**. A precise first
question is whether its learning process acquires operations that help future
tasks, instead of spending its observations resolving incidental details of
individual training tasks. That is narrower than inventing a universal learning
algorithm, but it is directly measurable. No universal superiority or priority
claim follows from this design.

## Three routes, and what each actually learns

These routes concern different parts of a system. An operator library is a
representation of reusable computation; energy relaxation is an execution and
learning method; information-seeking is a way to obtain evidence. They can be
combined, but simply combining their names supplies no new mechanism.

| Route | Learned object and execution | Information advantage, if its assumptions hold | Compute and transfer limitations |
|---|---|---|---|
| Operators and program library | Learn functions and reusable subprograms; synthesize a typed discrete program that invokes them. Execute one selected program. | A correctly learned operation can support many compositions from few new examples. Explicit typing removes invalid compositions. | Search grows rapidly with program depth and grammar size. A wrong primitive vocabulary or unsupported true program cannot be repaired by more confidence. Transfer requires reusable structure, compatible types and role binding. |
| Constraint or energy relaxation | Learn compatibility functions or constraints; repeatedly change an output state to reduce a scalar energy. | Known constraints can exclude large numbers of invalid outputs from relatively little supervision. | Each prediction needs solver iterations; local minima, relaxation and rounding can fail. Fixed variable-indexed constraints do not automatically transfer to new graph sizes or roles. |
| Information-seeking experiments | Learn which input or intervention to query next by the uncertainty it resolves. The predictor still needs one of the above representations. | A distinguishing example can eliminate many explanations at once when an accurate oracle is available. | Query scoring and posterior updating can cost more than the observations saved. Agreement among incomplete hypotheses can be confidently wrong. Offline observations do not supply an intervention oracle. |

For libraries, [DreamCoder](https://arxiv.org/abs/2006.08381) already learns
symbolic abstractions and a search guide. [Stitch](https://arxiv.org/abs/2211.16605)
learns library functions by compressing program corpora, including evaluation
on held-out programs. [HOUDINI](https://arxiv.org/abs/1804.00218) already combines
typed program search with learning neural functions and transfer across tasks.
Thus neither learning a library, composing learned functions, nor measuring
held-out library reuse is itself a new contribution.

The closer recent boundary is [Neural Language Interpreter, ICLR
2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9cde817d04811ba28e44071bd9f76a5-Abstract-Conference.html):
it learns a vocabulary of primitive operations, executes variable-length
programs through a differentiable interpreter, and adapts programs using
gradients. A learned operation bank plus a composer is therefore not a new
architectural category. The finite exact substrate below is chosen to isolate
a learning question, not to claim that category as an invention.

For relaxation, [SPEN](https://proceedings.mlr.press/v48/belanger16.html) learns
an energy `E_theta(x,y)` and predicts by approximately minimizing it over relaxed
outputs. Its original structured-margin objective can be written as
`max_y [Delta(y,y*) - E_theta(x,y) + E_theta(x,y*)]`, using approximate
loss-augmented inference. The execution changes the candidate answer, rather
than just computing a weighted mixture of representations.

[SATNet](https://proceedings.mlr.press/v97/wang19e.html) learns continuous
MAXSAT-relaxation coefficients and differentiates through a coordinate-descent
solver. Relaxed solutions and discrete rounding are distinct; the learned
coefficients are not automatically a human-readable logical theory. This is a
substantive different computation, but not a learned library by itself.
[Equilibrium propagation](https://arxiv.org/abs/1602.05179) uses free and weakly
target-nudged equilibria of `E_theta(x,s)+beta*C(s,y)` to obtain a gradient in
the small-beta limit. Its guarantee depends on equilibrium regularity and
settling assumptions; it does not by itself discover new discrete operations.

For acquisition, [Jha et al.](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-15.html)
already synthesize from distinguishing oracle queries, while
[Tiwari et al.](https://arxiv.org/abs/2006.12638) use information gain over
candidate programs. [SmartLabel](https://arxiv.org/abs/2508.15750) addresses active
neurosymbolic synthesis when neural components can mispredict. Therefore active
queries plus a library plus counterexamples is also established territory.

## Proposed substrate: finite learned operators with executable composition

Start with an exact finite environment where the claimed computation can be
audited. It deliberately avoids perceptual encoders, pretrained language models
and any fitted recommender input.

- A cell holds a Boolean value. An elementary learned binary operation has a
  four-entry truth table `f:{0,1}^2 -> {0,1}`. The entries are inferred; the
  learner is not supplied named AND, XOR or other correct operator semantics.
- A typed program contains input reads, constants, calls to learned operations,
  and explicitly supplied generic composition/fold constructs. Wiring and call
  choices are learned by discrete search. The interpreter and these constructs
  are human-specified inductive bias, not something discovered from nothing.
- A library stores learned tables and compact reusable program bodies. A
  subprogram can become a callable macro when it improves the fixed description
  length objective. Exact evaluation makes its meaning inspectable.
- Each task's deployed predictor is one selected compiled program. Multiple
  hypotheses are maintained during learning to decide what information is
  missing; their predictions are not ensembled at deployment.

This first substrate learns the contents of operations as well as their use,
but Boolean tables are intentionally a small controlled case. It is related to
circuit/program synthesis, not a claim to a new fundamental computational model.
Larger alphabets and continuous functions should be attempted only after the
finite experiment identifies what helps. Increasing alphabet size makes the
truth-table search space grow very quickly.

Let `L` be the learned library, `p_t` the composition for task t, and `D_t` its
observed input/output examples. Use a declared prefix-code description length
`DL`, including all table bits, macro bodies, wiring and constants:

```text
J(L, {p_t}) = sum_t sum_(x,y in D_t) -log P_epsilon(y | execute(L,p_t,x))
             + ln(2) * [DL(L) + sum_t DL(p_t | L)]
```

For the initial noiseless fixture use exact consistency, equivalent to taking
the likelihood penalty to infinity for an incorrect output. A later noisy
fixture must declare its noise likelihood beforehand. Search first minimizes
inconsistency, then total code length, with deterministic ties. A posterior over
the bounded exact hypothesis space is proportional to `exp(-J)`; if a beam is
later necessary, report that this is an approximation and test coverage against
the exact small case. A failure to find any consistent program is an explicit
model/search failure, not an observation to discard.

## One narrow mechanism: spend queries on reusable behavior

**Hypothesis:** at the same total computation and observation budget, queries
chosen to reduce uncertainty about reusable operator behavior improve prediction
on unseen compositions more than queries chosen to resolve whole training-task
programs. This information target is established targeted Bayesian design, not
a proposed novel objective. The concrete exact pilot described in
[the separate protocol](../library_queries/PROTOCOL.md) matches oracle budgets
and reports computation separately; it cannot claim the equal-compute version
of this hypothesis without another comparison that actually matches work.

The direct predecessor is [Sloman, Bharti, Martinelli and Kaski, Bayesian Active
Learning in the Presence of Nuisance Parameters, UAI
2024](https://arxiv.org/abs/2310.14968). Its current title replaces the earlier
The Fundamental Dilemma of Bayesian Active Meta-learning. It already studies
target information versus nuisance uncertainty and warns that learning nuisance
parameters can be necessary for accurate target estimation.
[Bartuska, Espath and Tempone](https://link.springer.com/article/10.1007/s11222-024-10544-z)
explicitly formulate information gain for parameters of interest after
marginalizing nuisance uncertainty. Our defensible contribution would be an
independent exact implementation and what its controlled comparison reveals,
not conceptual priority for either objective.

Write a hypothesis as `h=(L,{p_t})`. A query `q=(t,x)` asks the known fixture
oracle for the output of training task t at x. Define `Gamma(L)` as a canonical
behavioral signature of the library on a fixed finite basis of admissible
typed calls/contexts. Canonicalize operator renamings and syntactic aliases;
otherwise entropy can count different spellings as scientific uncertainty.
The signature basis is fixed from the training grammar, not from final transfer
answers. In the smallest truth-table fixture equivalence can be checked exactly.

Compare two acquisition scores with the same candidate hypotheses, candidate
hypothesis space, query pool, search algorithm, code-length objective and library
admission rule:

```text
ordinary program information: U_A(q) = I(Y_q ; h | D)
reusable behavior information: U_B(q) = I(Y_q ; Gamma(L) | D)

I(Y;Z|D) = H(Y|D) - sum_z P(z|D)*H(Y|Z=z,D)
```

The conditional term in `U_B` matters: it subtracts output uncertainty not
explained by the chosen library signature. A finite or coarse signature can
leave library behavior and role bindings unresolved as well as task composition;
noisy fixtures additionally leave observation noise. Only a complete, noiseless,
query-sufficient library signature would isolate composition uncertainty. For a
deterministic full hypothesis, `U_A` reduces to query-output entropy. Since
`Gamma(L)` is a deterministic function of h, data processing gives
`0 <= U_B(q) <= U_A(q)` for every fixed query. A lower information value per query
can nevertheless produce a different ranking of candidate experiments.
It can also fail: irrelevant library uncertainty, an inappropriate prior, or
a signature that misses the useful distinctions can waste every query.

The algorithm is concrete:

1. Infer consistent library/program hypotheses from the same seed examples.
2. Execute each candidate on a fixed admissible query pool; compute A or B.
3. Choose the maximum, resolve ties deterministically, and request one oracle
   answer. Keep both successful and failing examples.
4. Update the version space and resynthesize. Compress repeated program bodies
   using the same fixed rule for both arms. Preserve an executable expansion of
   every admitted macro and check it on the same admission witnesses.
5. At declared budgets, freeze the library, synthesize compositions for new
   tasks with the same adaptation allowance, and execute one selected program.

Finite witnesses detect failures; they do not prove correctness outside the
declared domain. A full equivalence checker could give a bounded-domain proof,
but access to its counterexamples must be equally charged to both arms. Do not
give the proposed arm free oracle labels through a so-called verifier.

Library admission and macro growth are held fixed in the primary experiment.
A later study could test admission by transfer benefit rather than compression,
but changing both admission and acquisition now would obscure the mechanism.
Admitting a primitive because it lowers held-out task loss is model selection;
calling that criterion semantic portability does not make it a new algorithm.
The algorithmic difference under investigation is the random variable targeted
by information gain and therefore which observations are acquired. The direct
prior art above establishes this distinction. No novelty claim is warranted.

An optional portability diagnostic can avoid conflating this with whole-model
refitting. Before results, choose typed operator slots in frozen program contexts
and their argument bindings. Replace only the operator at those slots with the
candidate or a declared comparison operator. Hold all other operations, encoders,
constants and program wiring fixed; permit no refit. For each context report
`mean_x[loss(C[o](x),y)-loss(C[o_comparison](x),y)]` on its unused inputs. Match
call count and report execution work; include all declared slots rather than
selecting the ones where replacement helps. Separate contexts on which the
operator was learned from new shapes/bindings. This measures the effect of a
specific executable replacement in the fixture, conditional on that frozen
context. It is an empirical portability test, not a proof of universal semantics,
a causal statement about people, or an independently novel learning architecture.

## Fixtures, information access and transfer tests

The following are design choices to turn into a separately reviewed protocol;
no fixture has been generated or scored yet.
The matched finite generator is a mechanism-recovery fixture. It favors the
assumed compositional family by construction, so it cannot support a claim of
general superiority. A broader study must include independently authored tasks,
unrelated smooth functions and no-signal cases, and match competitors' supervision
and tuning opportunity rather than give this system privileged generator code.

| Fixture | Training problems | Transfer that would count | What it cannot establish |
|---|---|---|---|
| Small Boolean circuits | Several tasks composed from a hidden shared library, with nuisance input permutations and different compositions. Only task I/O is visible. | New circuit shapes and source bindings; disjoint compositions, not merely new inputs to the same circuit. Evaluate all Boolean inputs when the domain is small. | General intelligence or discovery of unique internal semantics. |
| Sequence and graph composition | Finite Boolean sequences folded through learned operations; small graph relations composed through the same typed library. | Longer folds and held-out operation ordering, then new graph bindings using frozen operators. Generic iteration/join structure is disclosed human bias. | A claim of independent domains merely because the same Boolean rule is given a different story. |
| Synthetic categorical-evidence fixture | Generated item histories whose outcomes follow explicit compositional rules, including main effects, pair effects and distractors. | New item bindings and rule compositions, with known generator held away from the learner. | Any human preference or MovieLens causal claim. The generator defines the answer. |

Use disjoint task-generator seeds and disjoint program structures for library
learning, admission/model choices, and final synthetic transfer. Oracle queries
must be restricted to training tasks; adaptation on a transfer task uses only
its prespecified support-query budget, never its scored evaluation labels. Report zero-shot
and fixed few-shot transfer separately. Vary shared structure deliberately:
include a **no-sharing negative control** in which every task uses an independent
library. This should remove the proposed mechanism's advantage and can reveal
that it is only exploiting accidental data overlap.

A single task's factorization need not be identifiable: inserting an invertible
mapping h between A and B leaves `B(A(x))` unchanged after replacing them by
`h(A(x))` and `B(h^-1(x))`. Even several tasks may leave equivalent libraries.
Measure behavioral transfer and executable reuse, not recovery of the supposedly
true meaning of each hidden component. Report equivalence-class recovery only
where the finite oracle and task family actually identify it.

## Comparisons and a result that would falsify the idea

The required primary control is **ordinary program-information acquisition**
with exactly the same library learner. Add random-query acquisition to measure
whether either information objective repays its cost. A no-shared-library learner
measures the separate value of reuse; it is not the mechanism-isolating control.
An ordinary predictor with comparable fitted-parameter and total-compute ranges
is useful practical context, but does not replace the matched synthesis control.
Give all methods the same initial observations, oracle access and observation
budget; later acquired labels may differ by policy. Disclose primitives.
Before any claim about a superior learned-computation architecture, also include
an NLI-style learned-language reference, a frozen random operator bank with the
same search, and a learned bank without composition. The narrower acquisition
experiment cannot stand in for those architectural comparisons.

The primary endpoint is mean exact-output accuracy on unseen compositions at
fixed total candidate-execution budgets, with task-family-balanced averaging.
Also report oracle calls to a fixed accuracy threshold, wall time, search work,
memory, library size and inference cost. Every invalid/timeout answer counts as
an error. Use paired generator seeds and show family-specific outcomes; a pooled
average cannot support a claim of winning across every task family.

Information gain is not free. For example, scoring 256 candidate queries against
64 hypotheses takes 16,384 candidate executions before one oracle answer, apart
from hypothesis search and updating. Report both equal-oracle-call curves and
equal-total-compute curves. A mechanism that saves two cheap simulator calls by
performing thousands of additional executions is not a compute improvement.

The proposed mechanism is unsupported if its unseen-composition advantage
vanishes under equal total work, exists only when supplied the correct operator
library, fails under input renaming, or depends on final-context information in
its signature. If it wins on transfer while losing runtime, the honest result is
sample efficiency with a compute cost. If it also helps on the no-sharing fixture,
investigate an unintended difference in search or information access before
crediting reusable structure. A clean negative result is a reason to reject this
acquisition target, not to rename the same system and keep claiming progress.

## Connection to the recommender project

The current addressed model learns candidate/source/rating tables and a fixed
quadratic interaction form. It does not learn its operations or their composition.
A future library system could learn whether a candidate uses a count, threshold,
exception, conjunction or another compact evidence computation, then reuse those
operations under new item bindings. That would change model structure, rather
than combine scores from existing recommenders.

However, MovieLens contains observed ratings, not an oracle that answers what
the same person would rate after an arbitrary edited history. Manufactured
history edits cannot provide valid labels for the proposed active experiment.
An offline adaptation could query only masked ratings already present in a
training pool, record that availability restriction, and learn from these
observational examples. The active-choice effect would then be confounded with
the pool's exposure process unless the estimand explicitly conditions on it.

The sensible sequence is: test the reusable-information mechanism on a finite
oracle fixture; if it survives, specify a single categorical program family and
offline masking protocol; then compare it to direct additive/pair evidence and
strong references on a new, untouched assessment. Existing ML100K development
can debug the implementation but cannot supply new independent confirmation.
No new model has been selected and no old final-test result is used in this note.
