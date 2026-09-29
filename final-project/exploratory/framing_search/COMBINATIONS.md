# Can recommendation learn compatibility of combinations?

Research note, 2026-09-29. No new fits, code, or TEST access. The answer is
qualified: non-additive compatibility is a real modeling distinction, but the
broad idea is established and our exact pair experiment already tested an
important version of it. This review does not establish a new algorithm or a
reason to expect a large gain from another capacity increase.

## A counterexample that actually defeats additive scoring

Fix a common history C. Let a and b be two additional source items, and let t
and r be two eligible candidates, distinct from all context items. Suppose the
desired ranking is:

| a present | b present | Desired ranking |
|---:|---:|---|
| 0 | 0 | r above t |
| 1 | 0 | t above r |
| 0 | 1 | t above r |
| 1 | 1 | r above t |

This is a mathematical example, not a discovered MovieLens pattern. For EASE,
the score difference has the form `d(a,b) = c + alpha*a + beta*b`, where c
includes C and even an optional item bias. Therefore
`d(0,0) + d(1,1) = d(1,0) + d(0,1)`. Strict rankings in the table require the
left side negative and the right side positive, a contradiction.

The additional term `a*b` resolves this: `d = -1 + 2*a + 2*b - 4*a*b`.
That term is already in our complete pair-feature model. In contrast, a simple
rule saying "recommend t only when both a and b occur" is not a sufficient
counterexample: the additive difference `a+b-1.5` represents that ranking.

Three-source odd parity requires degree three: any degree-at-most-two score
has zero third alternating difference, whereas strict parity signs make all
eight signed contributions have the same sign. This identifies a capacity
boundary beyond our pair model. It supplies no evidence that such a pattern is
frequent, estimable, or valuable in this dataset. These two representational
arguments were independently checked by the assignment-audit agent.

## Do not confuse joint modeling with a missing property of EASE

EASE is not merely an unprincipled collection of unrelated votes. For
`P = (X^T X + lambda*I)^(-1)`, its coefficients are `B[j,i] = -P[j,i]/P[i,i]`.
They are the conditional-mean coefficients of a positive Gaussian joint model
with precision P and conditional variance `1/P[i,i]`. Asymmetric B can arise
from different conditional variances; asymmetry alone does not prove
incompatible predictions. Steck derives the same dense solution through
Gaussian Markov random fields. This is a mathematical interpretation, not an
assertion that binary records are Gaussian. [Steck, *Markov Random Fields for
Collaborative Filtering*, §2.2.1, Eq. 4](https://arxiv.org/pdf/1910.09645)

A discrete joint model does impose useful consistency constraints. For binary
full-conditional log odds `ell_i(x_-i)`, compatibility with one positive joint
distribution implies `Delta_j ell_i = Delta_i ell_j`. Both sides are the same
four-cell log-probability contrast. However, this is established graphical-model
theory, and it does not imply that converting EASE's real scores into arbitrary
logits preserves its Gaussian interpretation. Dependency networks already
distinguish consistent joint conditionals from separately learned conditional
models and apply them to collaborative filtering. [Heckerman et al., JMLR 2000,
definitions](https://www.jmlr.org/papers/volume1/heckerman00a/html/node3.html)
and [recommendation application](https://www.jmlr.org/papers/volume1/heckerman00a/html/node11.html)

## What the candidate framings reduce to

| Framing | Actual mechanism and nearest primary precedent | What remains unproved here |
|---|---|---|
| Explain-away | Conditional dependence changes when another explanation is observed; Bayesian/dependency networks already use conditional probability models. [Heckerman et al.](https://www.jmlr.org/papers/volume1/heckerman00a/html/) | A lower score after adding b can already be a negative linear coefficient. Context-dependent reversal, not the phrase "explaining away", is the relevant nonlinearity. Observational dependence does not identify psychological causes. |
| Higher-order or hypergraph compatibility | Explicit conjunctions can affect a candidate jointly. Steck and Liang add signed higher-order input features; HCCF instead learns hypergraph-enhanced representations. [RecSys 2021](https://recsys.acm.org/recsys21/session-2/), [HCCF](https://arxiv.org/abs/2204.12200) | A hyperedge is a representation, not proof of an irreducible conjunction. Linear propagation on a fixed hypergraph remains linear in its input. Our complete pair dictionary already expresses all two-source conjunction coefficients independently. |
| Closure or set completion | An itemset implication says that a context predicts an extension. Frequent-pattern rules and conditional tables provide this behavior. | Observational near-closure is not a logical necessity. Adding unrelated history can invalidate a conditional rule; a monotone completion operator cannot express every veto. Sparse support can create accidental perfect rules. |
| Compression or MDL | Select a pattern dictionary by total model-plus-data description length. KRIMP explicitly does this; its research program also includes missing-data completion. [KRIMP paper](https://www.patternsthatmatter.org/publications/2011/krimp_mining_itemsets_that_compress-vreeken,vanleeuwen,siebes.pdf), [author project](https://vreeken.eu/prj/krimp/) | Compression is a concrete complexity criterion, not a new objective invented here. Short codes for common histories need not optimize personalized nDCG. A code for the exact set C plus i is not automatically the probability of i given a partially observed C. |
| Exact categorical joint distribution | Autoregressive factorization represents a joint distribution through conditional categorical models. CF-NADE models a user's rating vector this way. [CF-NADE, Eqs. 1–5](https://proceedings.mlr.press/v48/zheng16.pdf) | CF-NADE's cited likelihood concerns ratings of the specified rated items; it does not by itself model which items were recorded. A six-state variable can model recorded absence plus ratings, but absence must not be reinterpreted as known dislike. |
| Context-specific independence | A dependency matters only in specified branches; other branches share a probability table. Tree-structured conditional tables explicitly represent this. [Boutilier et al., UAI 1996](https://ai.stanford.edu/~nir/Papers/BFGK1.pdf) | Sparse branching can reduce estimation burden, but the method is known. Greedy one-feature search can miss pure XOR because neither feature has marginal predictive gain. A proposed learner must state its interaction-search procedure. |

"Compatibility of the user's history with one candidate" is also different from
"compatibility among the ten recommended candidates." The latter concerns a
slate's combined utility. Our item-relevance nDCG endpoint does not measure
whether a person enjoyed a collection as a package.

## What our existing studies did and did not test

| Local study | Tested mechanism | Boundary relevant to this proposal |
|---|---|---|
| [Categorical reconstruction](../categorical_reconstruction/DERIVATION.md) | Linear contributions from binary presence and observed rating-category residuals, with all target channels excluded. | A richer value for each separate source is still additive across sources. Mean primary nDCG was 0.263154 versus binary 0.261688, a small gain. |
| [Exact context interactions](../context_interactions/DERIVATION.md) | Every distinct source pair with independent signed candidate-specific coefficients, jointly fitted by exact ridge reconstruction. | It already solves the two-bit counterexample in its hypothesis space. It did not implement sparse conditional-table sharing or a general discrete joint distribution. Mean pair-only nDCG 0.261660 versus binary 0.261688 gives no useful gain. [Results](../context_interactions/FINDINGS.md) |
| [Categorical field](../../categorical_field.py) | Nonlinear recurrent state updates followed by five category logits per item. | It was not restricted to additive input votes, but its particular learned state and optimization did not establish successful recovery of higher-order relations. |
| [Joint field objective](../../JOINT_FIELD_PROTOCOL.md) | One normalized recorded-item/rating event conditional on context; losses average across masked probes. | "Joint" here joins item identity and rating for one event. It is not a normalized joint probability over the entire user's item set, nor a guarantee of compatible full conditionals across all masks. |

Changing only terminology to "combinations", "closure", or "energy" would
therefore recycle existing capacity or established modeling ideas. A different
statistical constraint or learning objective must be explicit and controlled.

## One simple candidate worth specifying, not yet launching

The clearest alternative is a **sparse conditional rule table**: a candidate's
effect changes only in a small number of explicitly learned contexts; elsewhere
the table shares its default probability. A depth-two table can encode the
two-bit example, and a depth-three table can encode three-source parity. The
model does not add votes from independently trained experts.

For TRAIN-generated masked histories C and original TRAIN record indicators
`Y[u,i]`, a concrete objective is

```text
sum over eligible (u,C,i) BCE(Y[u,i], q_i(C; T_i))
    + lambda * description_length({T_i}).
```

Here `T_i` is a conditional decision table with tied/default leaves. The zero
label means no record in that TRAIN snapshot, not an observed negative opinion.
The recipe would need explicit mask generation, candidate sampling or full
catalog weighting, smoothing, and exclusion of every target-derived feature.
Two- or three-step lookahead is necessary if the search claims to discover
interactions with no one-variable gain. An unconstrained list of separately
fitted tables is a dependency model, not automatically one compatible joint.

The distinct local research question would be whether **selective contexts and
parameter sharing estimate useful nonlinear relations more efficiently than
the complete pair dictionary under a global ridge penalty**. This is a question
about estimation and inductive bias, not a newly discovered form of reasoning.
Its nearest predecessors are conditional decision trees, context-specific
independence, dependency networks, and MDL pattern selection.

The strongest first controls would use the same TRAIN episodes and record
loss: additive logits, degree-two logits, and the conditional-table model with
its sharing removed or restricted. Compare search budgets and effective model
size explicitly. A win only against squared-error EASE cannot separate changed
loss from changed conditional structure. A gain that disappears against matched
degree-two logits is not evidence for a beyond-pair mechanism. Synthetic XOR
and parity establish implementation capacity, not MovieLens relevance.

Before another fit, a useful prerequisite is a predeclared TRAIN-only audit of
whether candidate-context reversals recur across disjoint user folds with enough
support. Mining and assessing reversals on the same users would manufacture
apparent evidence. Even a successful audit would justify testing, not promise
large gains. If no reliable contexts survive, the proposed rule learner lacks
an empirical foundation on this dataset.

## Information and computational limits

An unrestricted binary joint over m items has `2^m` cells; adding recorded
rating categories produces `6^m`. These are distributions, not estimable tables
from 943 profiles without substantial assumptions. Multiple masked histories
from one user do not create new independent users. Conditional independence,
small cliques, latent variables, or autoregressive sharing supply assumptions;
none is free information.

Evaluating a full joint density can be easy while conditioning on an arbitrary
partially observed subset is expensive, because unobserved variables require
marginalization. A shortest single completion is a mode, not that marginal.
Similarly, treating every unlisted item as a known negative state changes the
statistical question. Neither exact normalization nor an elegant rule language
recovers unknown exposure, current intent, or counterfactual satisfaction from
these records alone.

No candidate reviewed here earns a first-ever novelty claim. The defensible
contribution would be a precise implementation, a demonstrated estimation
advantage over matched controls, and a reproducible result on an untouched
assessment. At present the argument establishes a research question and rejects
several misleading reformulations; it does not establish the requested large
performance improvement.
