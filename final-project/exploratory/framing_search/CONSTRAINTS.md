# Complete the unknown entries, or condition on their recorded zeros?

Research note, 2026-09-29. No implementation, model fit, data evaluation or TEST
access was performed. The examples below are algebraic constructions, not
experimental recommendation results.

**Finding:** conditioning only on the observed subset changes EASE's predictions,
but the resulting rule is standard Gaussian-process conditioning. It also equals
query-local ridge regression and a particular quadratic constrained completion.
The surviving question is empirical: which conditioning assumption better serves
this recorded-item ranking task? There is no new model principle here yet.

## The minimal distinction

Let `X` be a binary TRAIN user–item matrix, `K = X^T X`, and `lambda > 0`.
For one query, let `H` be its observed item indices, `U` the other items, and
`y_H = 1`. Use the same fixed `K` and regularization for both rules.

```text
A = K + lambda I
P = A^(-1)

EASE:       s_i = -P[i,H] y_H / P[i,i],                 i in U
Observed:   s_U = K[U,H] (K[H,H] + lambda I)^(-1) y_H.
```

For a zero-mean Gaussian with covariance `A`, EASE's first line is the
conditional mean of `z_i` given `z_H = y_H` **and** `z[U\{i}] = 0`.
The second line conditions on `H` and integrates out the other coordinates.
Steck's original paper already explains the full-conditional Gaussian
interpretation, in its unregularized special case. Using `A` supplies the
corresponding regularized algebra. This is an interpretation of an approximate
continuous model, not proof that EASE labels every unrecorded movie a dislike.
[Steck, 2019, §3.2](https://arxiv.org/pdf/1905.03375)

There is a direct recommender precedent, not merely a resemblance to statistics
in another field. Lawrence and Urtasun explicitly marginalize missing ratings
and predict an unseen item from the user's observed subset using
`k_iH (K_HH + noise I)^(-1) y_H`. Their kernel is learned through a GP latent
variable model, and their task is rating prediction. Replacing that kernel with
the binary TRAIN Gram changes the prior and task, not the conditional-prediction
principle. [Lawrence and Urtasun, ICML 2009, §§2.1 and 3.1, Eq. 3](https://people.csail.mit.edu/rurtasun/publications/lawrence_urtasun_icml09.pdf)

The same expression is the standard GP posterior mean with observation-noise
variance `lambda`. On unseen coordinates, independent diagonal noise does not
alter the cross-covariance, explaining the equality to conditioning under `A`.
[Rasmussen and Williams, Chapter 2, Eq. 2.23](https://gaussianprocess.org/gpml/chapters/RW2.pdf)

## Why a recurrent completion system does not escape this equivalence

Consider one joint completion with observed coordinates fixed:

```text
minimize_z    (1/2) z^T P z
subject to    z_H = y_H.
```

Since `P` is positive definite, the unique minimizing unknown vector obeys

```text
P[U,U] z_U + P[U,H] y_H = 0
z_U = -P[U,U]^(-1) P[U,H] y_H
    = A[U,H] A[H,H]^(-1) y_H
    = K[U,H] (K[H,H] + lambda I)^(-1) y_H.
```

This is a block-inverse identity. Initialize all unknown entries at zero and
update each one once in parallel using its Gaussian conditional: that first
update is EASE. Continue solving those conditionals while keeping `H` fixed,
and any converged solution is the observed-subset rule. Sequential coordinate
minimization converges for this positive-definite quadratic; unrestricted
synchronous updates need not. A neural or fluid analogy adds no new inference
mechanism to this calculation.

There is also an exact local-regression identity. For each candidate, solve

```text
b_i = argmin_b ||X[:,i] - X[:,H] b||_2^2 + lambda ||b||_2^2
    = (K[H,H] + lambda I)^(-1) K[H,i].
```

Then `y_H^T b_i` is the observed-subset score. Equivalently, fit EASE restricted
to columns `H union {i}` and inspect its candidate column. Restricting the
already fitted global EASE matrix to those columns is **not** equivalent.
Here “local EASE” describes this algebraic identity; this search did not establish
a published method with that exact name. The direct GP recommender precedent
already rules out presenting the inference principle as new.

## A realizable ranking reversal

Consider three items `A, B, C` with TRAIN Gram and `lambda = 1`:

```text
K = [[9,  6, 5],
     [6, 19, 8],
     [5,  8, 9]].
```

This is a valid binary Gram: take history rows `100, 010, 110, 101, 011, 111`
with multiplicities `2, 9, 2, 1, 4, 4`, respectively. These 22 rows give the
matrix exactly. The query records only `A`.

| Rule | Score B | Score C | Ranking |
|---|---:|---:|---|
| Observed-subset conditioning | `6/10 = 0.600000` | `5/10 = 0.500000` | B above C |
| Full EASE conditioning | `(6*10 - 8*5)/(10*10 - 5^2) = 0.266667` | `(5*20 - 8*6)/(10*20 - 6^2) = 0.317073` | C above B |

The difference is not a rescaling or numerical solver effect. EASE's B
prediction conditions on C's zero and vice versa. The observed-subset rule
does not turn either unknown candidate into an observation.

With only catalog items A and B, EASE's B score is `0.6`. Add unrecorded C while
preserving the A/B block, and it becomes `0.266667`; observed-subset B stays
`0.6`. **C is correlated, not statistically unrelated.** Adding a completely
independent catalog coordinate changes neither rule. Calling any unrecorded
item “irrelevant” would assume the very missingness interpretation at issue.

The observed-subset rule also has a general invariance: adding unobserved items
while preserving the kernel blocks on existing query/candidate coordinates
cannot change those candidates' scores. Global EASE need not have this property.
This is an observable algebraic distinction, not evidence that the invariant
rule is more accurate.

For two observed items with unit diagonal, mutual kernel value `rho`, and equal
candidate cross-kernel value `c`, subset prediction is
`2c/(1 + lambda + rho)`. Correlated observations contribute less than two
independent observations with the same candidate similarities. This familiar
kernel correction should not be advertised as newly discovered redundancy
reasoning. With one observed item, the rule ranks candidates by raw co-occurrence;
with a diagonal observed block it reduces to fixed, item-normalized voting.
For large `lambda`, it approaches scaled summed co-occurrence.

## Harmonic fields, associative memory, and higher-order constraints

Gaussian-field harmonic extension is another established boundary-value
formulation. With graph Laplacian `L`, unknown values solve
`L_UU z_U = -L_UH y_H`. Our precision `P` generally is not a graph Laplacian:
its off-diagonal signs and row sums need not permit a positive-conductance
random-walk interpretation. The shared block solve does not make every inverse
Gram model ordinary label propagation. [Zhu, Ghahramani and Lafferty, ICML 2003,
§2, Eq. 5](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf)

A useful failure case follows immediately: an ungrounded connected Laplacian
with only boundary labels equal to one extends the constant value one to every
reachable unknown. It cannot rank those items. Grounding, shrinkage or other
boundary values changes the problem by adding assumptions. “Let preferences
flow until consistent” is incomplete without that specification.

Energy descent and content-addressable completion are established associative
memory ideas. Our strictly convex quadratic has one completion and no collection
of attractor basins; calling its solver a Hopfield network would not add the
multi-attractor mechanism of associative recall. [Hopfield, 1984](https://authors.library.caltech.edu/records/rhj8z-8dj90)

Higher-order energies can express constraints absent from pairwise covariance,
but that route is also established. Dense associative memory explicitly extends
quadratic interactions and discusses an XOR example. Merely introducing a cubic
term or a rule that completions must agree is not a new principle. It would need
a precise learnable energy, an inference rule, and evidence that the relevant
relations exist in sparse movie records. [Krotov and Hopfield, NeurIPS 2016,
§§2–3](https://arxiv.org/pdf/1606.01164)

## What remains worth asking

The narrowly defensible question is: **for the same TRAIN kernel and comparable
regularization selection, does conditioning only on recorded context predict
held-out recorded items better than full zero conditioning?** This separates
one inference assumption without adding a richer feature dictionary, pretrained
model, ensemble, or new neural architecture. It could be investigated as an
existing-method comparison, not marketed as an original invention.

The objections are substantial:

- `X^T X` still comes from zero-filled observations. Changing query inference
  does not repair an unidentified exposure process or make all training
  missingness ignorable.
- Positive observations are a selected subset. Integrating everything else out
  discards potentially predictive information in non-recording; it is not
  automatically the right likelihood for the recorded-item endpoint.
- Gaussian scores may be negative or exceed one. This is continuous completion,
  not a calibrated distribution over feasible binary recommendation sets.
- The kernel still contains only pairwise statistics. Joint inference introduces
  context-dependent coefficients, not newly learned higher-order constraints.
- Different rankings and catalog-extension invariance establish a distinction,
  not improvement. Matching score masks, data, kernel scaling and tuning budgets
  is necessary for any later comparison; reused data cannot supply fresh
  confirmation.

**Decision:** retain this as a clear, falsifiable framing comparison. Reject the
claims that observed-subset conditioning, recurrent Gaussian completion, local
ridge/EASE, or generic higher-order associative memory constitute a new concept.
No implementation or experiment is authorized or performed by this note.
