# Learn the computation, and test what deserves to transfer

28 September 2026. Research extension, separate from the completed assignment
study. The ambition is our own implemented learning system that acquires useful
operations and uses them on unfamiliar problems. Universal superiority and
conceptual novelty are not established by this ambition.

## The deeper question

The earlier structural learner was given its vocabulary: max, products and
clean evidence atoms. It learned which expression to choose. A deeper learner
would also infer what its operations do. A further step is to choose observations
that distinguish reusable structure from details confined to one task.

These are three distinct objects to learn:

1. **Operation contents.** A function that transforms inputs, learned from
   examples rather than supplied as a named AND, similarity or exception rule.
2. **Programs.** Which operations act on which values, including intermediate
   results. Deployment executes one selected program.
3. **Experiments.** Which permitted observation would clarify a reusable
   operation, and when identifying the present task matters more.

That is a concrete sense in which we can build our own system. We still supply
a representation, an interpreter, a loss, a search budget and an environment.
Removing every assumption leaves no basis for choosing between explanations
that agree with all observations. Numerical libraries are infrastructure; using
them does not make an architecture or its experiment somebody else's model.

## What is implemented and tested

The [finite library-query experiment](../library_queries/PROTOCOL.md) is an
exact microscope for this question. A shared operation has four unknown Boolean
outputs. Each task has an unknown program that calls the operation once or
twice with different input bindings. The learner sees only the input/output
pairs it is allowed to request, not the hidden operation or generating program.

We can enumerate every candidate operation and program. Thus a failure cannot
be excused by an approximate posterior or an optimizer that might not have
converged. The experiment compares information about the shared operation,
information about the complete task explanation, and random acquisition.
After learning, it freezes one operation and constructs one program for each
new function using only that function's support examples.

The information target is already an established idea. Our work here is a new
implementation and a controlled investigation, not a new information-theoretic
principle. The experiment includes cases without shared operations and records
when the shared-library hypothesis becomes inconsistent. It excludes training
functions from transfer by their complete finite behavior, not just their
program spelling. Degenerate environments with no unused function are reported.

The [completed study](../library_queries/README.md) finds earlier transfer gains
from focused queries when tasks share an operation, with ordinary acquisition
catching up at the largest budget. With independent task operations, focused
queries expose the shared-model contradiction earlier; the current learner
abstains and receives its declared penalty. This is a distinction between
detecting a wrong assumption and repairing it. Learning which tasks can share
an operation is therefore a concrete next capability, not one already present.

This small domain has serious limits: four unknown bits, shallow composition,
a generator matching the learner's grammar, and an exact synthetic oracle.
It cannot establish deep reasoning, cross-domain intelligence, causal human
preference, or superiority to neural recommenders. It tests whether a learning
mechanism behaves as its explanation predicts.

## The scalable architecture remains a proposal

The [continuous-operator proposal](LEARNED_OPERATORS_PROPOSAL.md) replaces those
four bits with four small anonymous bivariate neural functions. Their parameters
are shared across tasks, while task-specific hard programs compose them.
Disjoint examples within training tasks separate proposing a program from
updating its operations. On held-out tasks the operation library is frozen.

The [alternatives analysis](ALTERNATIVES.md) also considers learned constraint
systems and active experiment design. These are different choices about what
the machine represents, how it executes and how it gets evidence. Combining
them is not, by itself, a contribution.

Before increasing scope, require evidence at each transition:

| Transition | What must be demonstrated | What could defeat it |
|---|---|---|
| Finite operations to continuous operations | Frozen learned functions help unfamiliar compositions relative to random operations, no composition and dense adaptation controls. | The apparent reuse comes from generator matching or extra search compute. |
| Familiar depth to deeper programs | Accuracy survives additional operation calls at a reported inference cost. | Local approximation errors compound, or search becomes impractical. |
| One task family to independent families | Transfer survives independently defined smooth, piecewise and no-signal controls, then an external benchmark. | The representation only fits the original family; sharing causes negative transfer. |
| Synthetic tasks to recommendations | A real-data operation learner improves recorded-event/rating prediction using only permitted histories. | Exposure confounding, sparse histories or weak evidence prevents identifying useful operations. |

The recommender bridge needs special care. MovieLens cannot answer arbitrary
questions about altered user histories. We may hide and reveal already recorded
ratings under a declared offline protocol; we cannot treat that as an oracle
for how a person would respond to an intervention. A learned recommendation
program still needs a competent representation of actual items and ratings.

## Prior art that changes the claims

[DreamCoder](https://arxiv.org/abs/2006.08381) learns reusable symbolic
abstractions. [Neural Language Interpreter, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9cde817d04811ba28e44071bd9f76a5-Abstract-Conference.html)
already learns a neural operation vocabulary and composes programs. A learned
operator bank and a composer are therefore not a new architectural category.

[Sloman et al., UAI 2024](https://arxiv.org/abs/2310.14968v3) studies active
learning with target and nuisance parameters, including transferable and
task-specific quantities. It identifies a danger in pursuing only target
information while misunderstanding nuisance variables. Our focused acquisition
objective belongs to that established framework; this source also supplies
a reason to test failures rather than assume transfer-focused queries win.

The empirical contribution must be precise: which mechanism helps, on which
tasks, with how many observations and how much computation. A large collection
of task names is not evidence for generality if every task uses the same hidden
rule. A negative result that identifies this boundary is useful; it does not
become a breakthrough by changing the language around it.

## Assignment fit

The required MovieLens100K comparisons, hybrids and societal analysis remain
in the existing report. These experiments are supplementary research material.
The original assessment split has already been opened, so subsequent MovieLens
development results cannot establish a fresh confirmatory advantage. Broad
claims require both an untouched supplementary dataset and stronger external
benchmarks than this controlled fixture.
