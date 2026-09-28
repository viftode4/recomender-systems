# Complete pair reconstruction without materializing the pair dictionary

The question is whether independent, target-specific interactions between two
observed history items improve the assignment's recorded-item ranking. This
model changes representation while retaining the binary reconstruction target,
ridge regularization, full histories, and complete target exclusion. It does
not combine fitted recommender scores or assume that an unrecorded item was
disliked. The [protocol](PROTOCOL.md) fixes the actual selection and evaluation.

## One jointly fitted scoring function

Let `X` be the binary TRAIN observation matrix with `n` users and `m` catalog
columns. Zero means no recorded TRAIN interaction. For target `i`, define

\[
s_i(x)=\sum_{j\ne i} a_{ji}x_j
 +\sum_{j<k;\,j,k\ne i} b_{jki}x_jx_k.
\]

Fit all coefficients by minimizing

\[
\sum_i\|X_{:i}-s_i(X)\|_2^2
 +\lambda\sum_{j,i}a_{ji}^2
 +\frac{\lambda\nu}{\beta}\sum_{j<k,i}b_{jki}^2,
\qquad \lambda>0,\ \beta>0.
\]

Every target's singleton and every pair containing that target are excluded.
The two coefficient types are fitted jointly and their signs are unrestricted.
There is no constraint tying a pair coefficient to products of singleton
coefficients. At `beta=0`, omit the pair features entirely: the model becomes
exactly EASE's binary constrained ridge problem.

The fixed TRAIN scale is

\[
h_u=\sum_jX_{uj},\qquad
\nu=\frac{\sum_u {h_u\choose2}}{\sum_u h_u}.
\]

Set `nu=1` if the numerator or denominator is zero. This balances aggregate
singleton and pair feature energy. It does not normalize every user's history
length and is not tuned on evaluation outcomes.

The equivalent feature map contains singleton features `x_j` and all distinct
pair features `sqrt(beta/nu) x_j x_k`, with the same ridge penalty `lambda` on
their scaled coefficients. This is linear regression in an expanded feature
space and nonlinear prediction in the original history indicators.

## Exact kernel

For two binary histories, let `g=x^T z` be the number of shared observed items.
There are exactly `choose(g,2)` unordered pairs shared by both histories. Thus

\[
G=XX^T,\qquad P=\frac{G\odot G-G}{2},\qquad
K=\frac{G+\gamma P}{\lambda},\quad \gamma=\beta/\nu.
\]

`P` is the Gram matrix of the complete distinct-pair dictionary. Repeated-item
terms such as `x_j^2` are excluded; for binary inputs they would merely duplicate
the singleton. Positive `beta` keeps this kernel positive semidefinite.

For target `i`, write `v=X[:,i]`, `S={u:v_u=1}`, and let `E_S` embed vectors
indexed by `S` into all user rows. Deleting the target source gives

\[
G_i=G-vv^T,\qquad
K_i=\frac{G_i+\gamma(G_i\odot G_i-G_i)/2}{\lambda}.
\]

Because `v` is binary, the difference is confined to the observed-user block:

\[
K-K_i=E_S\Delta_iE_S^T,\qquad
\Delta_i=\frac{\mathbf1\mathbf1^T+
 \gamma(G_{SS}-\mathbf1\mathbf1^T)}{\lambda}.
\]

The first term removes the target singleton. The second removes all pairs
formed from the target and another observed item. It is positive semidefinite:
`G_SS - 11^T` is the Gram matrix after deleting the target's all-one column
within this block. This remains true when `gamma>1`; a negative coefficient on
`11^T` in the expanded algebra is not a negative kernel mixture.

## One shared factorization, then small support solves

The target-specific dual solution satisfies

\[
(I+K_i)c_i=X_{:i}.
\]

Factor the shared user-space system `A=I+K` once, and solve for `H=A^-1` using
its Cholesky factor. For each nonempty target support, factor `H_SS=L L^T`
and define

\[
M_i=I-L^T\Delta_iL.
\]

Then

\[
\boxed{c_i=H_{:S}L^{-T}M_i^{-1}L^T\mathbf1.}
\]

To see this, eliminate the unobserved-user block of `A`. Its Schur complement
on `S` is `H_SS^-1`; the target removal changes that complement to
`H_SS^-1 - Delta_i = L^-T M_i L^-1`. The right-hand side is one on `S` and
zero elsewhere, giving the boxed solution. Both Schur complements are positive
definite in exact arithmetic.

For `beta=0`, the removal is rank one and simplifies to

\[
c_i=\frac{Hv}{1-v^THv/\lambda}.
\]

This avoids a support-sized factorization for the binary control. Empty target
columns have `c_i=0` and always receive zero score, including the empty PAD
column. No model-side masking gives PAD a special role.

The main cost is a shared `n`-square factorization plus support-sized operations
scaling with `sum_i |S_i|^3`, and dense score/residual multiplication. It is not
one full `n`-square factorization per target. The seed-2026 TRAIN benchmark has
943 users, 1,683 catalog columns, and a largest target support of 471 users.
The implicit pair dictionary has 1,415,403 columns including zero PAD-derived
columns, but is never stored. This count is not an effective parameter count:
many pairs are absent, dependencies exist, and each target kernel has rank at
most `n`.

## Arbitrary-query scoring also excludes the target

For binary query histories `Q`, let `G_q=QX^T`. Raw kernel scores are

\[
S_{\rm raw}=\frac{G_q+\gamma(G_q\odot G_q-G_q)/2}{\lambda}C.
\]

If query target `i` is absent, no correction is needed. Otherwise the target
singleton and its pairs must also be removed from the cross-kernel. Set
`R=X odot C`. The complete vectorized correction is

\[
\boxed{S=S_{\rm raw}-\frac{Q\odot
[(1-\gamma)\mathbf1(R^T\mathbf1)^T+\gamma G_qR]}{\lambda}.}
\]

Consequently changing only `Q[:,i]` cannot change the prediction for `i`.
`predict()` always applies this correction. `predict_unadjusted()` is valid
only at positions where the candidate is absent. Neither method masks seen
items, PAD, or any other candidates; ranking masks belong to the runner.

## Numerical and information checks

The solver checks `(I+K_i)c_i-X[:,i]` for every regular target. Failed support
factorizations or relative residuals above `2e-8` trigger direct construction
and Cholesky solution of that target's reduced user-space system. A fallback
residual above `2e-7` stops with an error rather than silently accepting it.
Positive finite `lambda` and nonnegative finite `beta` are required. No
numerical fallback changes the objective or tunes a parameter.

Independent tests explicitly enumerate all pairs on small matrices, physically
remove every feature containing a target, and solve augmented primal least
squares. They compare both implied coefficients and arbitrary-query scores.
Other checks cover the EASE limit, cold targets, empty histories, query
target-invariance, sparse/dense replay, and no-evaluation-data normalization.
An XOR example uses `x_1+x_2-2x_1x_2`; it demonstrates representation capacity
that the linear control lacks. It is not evidence that MovieLens requires this
function or that the larger class will generalize better.

## What this experiment can establish

The earlier addressed pair model multiplied its own learned singleton votes,
used no weight decay, and optimized joint item/rating likelihood under masked
histories. Its observed scale growth and poor generalization do not establish
that independently regularized interactions are useless. This kernel study
removes those particular restrictions: independent coefficients, an exact
convex solve, the same reconstruction objective as its linear control, and
full target-excluded histories in training and inference.

A gain over the matched linear control would support the tested interaction
features under this protocol. It could also be described as emphasizing users
who share more history items: `choose(overlap,2)` is the same mechanism in
kernel space. It would not prove discovered human rules or identified exposure.
A null result would constrain this model family and grid, not establish a
ceiling on the information in the dataset.

Higher-order itemset reconstruction is established in
[HOSLIM](https://conservancy.umn.edu/bitstreams/021a3338-53fa-4f4b-9bc5-ffdec7d1f149/download)
and signed higher-order extensions in
[Steck and Liang, 2021](https://doi.org/10.1145/3460231.3474273).
The model also directly nests [EASE](https://arxiv.org/abs/1905.03375).
This implementation claims neither invention of polynomial kernels nor a new
theory of recommendation. Our contribution is the exact implementation and
the declared comparison. Exposure modeling is a separate assumption-bearing
approach, as in [Liang et al., 2016](https://arxiv.org/abs/1510.07025);
these MovieLens records do not directly tell us whether an unrecorded movie
was unseen, rejected, forgotten, or simply unrated.
