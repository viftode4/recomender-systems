# Categorical reconstruction with complete item-block exclusion

This is one jointly fitted linear recommender. It reconstructs the observed
user–item matrix from binary and categorical source features. Its binary-only
submodel is exactly the EASE objective. It does not combine predictions from
separately fitted recommenders. Adding features to a shallow autoencoder is
established prior art; this implementation makes no architectural novelty claim.

## Observed-only features

Let `R[u,i]` be a TRAIN rating in `{1,2,3,4,5}`, with zero denoting absence, and
let `X[u,i] = 1[R[u,i] != 0]`. Define TRAIN counts and probabilities

\[
n_{ir}=\sum_u1[R_{ui}=r],\qquad
p_r=\frac{\sum_i n_{ir}}{\sum_{i,r'}n_{ir'}},\qquad
p_{ir}=\frac{n_{ir}+\alpha p_r}{n_i+\alpha}.
\]

The frozen experiment uses `alpha=20`. An entirely empty TRAIN matrix uses a
uniform global prior. An empty item with zero smoothing uses the global prior.
These conventions keep the transform defined; they supply no observed evidence.

For each rating category, form

\[
Z_{r,ui}=1[R_{ui}=r]-X_{ui}p_{ir},\qquad
F=[X\mid Z_1\mid Z_2\mid Z_3\mid Z_4\mid Z_5].
\]

Only observed entries are centered. An absent item has six zero features,
which is essential both for the information boundary and for fast scoring of
unseen candidates. The five residual channels sum to zero. Their resulting
linear dependence is harmless under strictly positive ridge penalties.

Ratings are categorical sources: no liked/disliked threshold, numerical rating
distance, ordinal monotonicity, or emotion interpretation is assumed. A global
permutation of the five category names leaves the fitted scores unchanged.

Query histories use the fixed TRAIN probabilities. The model never updates
centers from a query, validation labels, or other users' query histories. Each
item owns the channel-major feature indices
`I_i = i + n_items * arange(6)`. Binary mode instead has `I_i={i}`.

## Objective and exact exclusion

Set the reconstruction target to `Y=X`. With diagonal penalties

\[
D=\operatorname{diag}(\lambda_b\mathbf 1_m,
\lambda_c\mathbf 1_{5m}),\qquad \lambda_c=\rho\lambda_b>0,
\]

solve the strictly convex problem

\[
\min_B\ \|Y-FB\|_F^2+\operatorname{tr}(B^TDB)
\quad\text{subject to}\quad B_{I_i,i}=0\ \text{for every item }i.
\]

All six features of a target are excluded, rather than only its binary feature.
Thus a target's own observed rating cannot directly reconstruct that target.
Other TRAIN observations still fit shared coefficients and centering statistics,
as in ordinary supervised parameter estimation. The objective penalizes error at
unobserved entries with target zero; that is the declared reconstruction loss,
not an assertion that an unrecorded interaction is an explicit dislike.

Let `F_i=F[:,I_i]` and `D_i=D[I_i,I_i]`. Removing that block gives

\[
A=I_n+FD^{-1}F^T,\qquad H=A^{-1},\qquad
A_{-i}=A-F_iD_i^{-1}F_i^T.
\]

Both `A` and `A_-i` are positive definite. The constrained column is

\[
c_i=A_{-i}^{-1}y_i,\qquad
b_{-I_i,i}=D_{-I_i}^{-1}F_{-I_i}^Tc_i,\qquad b_{I_i,i}=0.
\]

Woodbury's downdate identity yields the implemented small-block correction

\[
S_i=D_i-F_i^THF_i,\qquad
c_i=Hy_i+HF_iS_i^{-1}F_i^THy_i.
\]

`S_i` is positive definite in exact arithmetic. The ordinary path computes a
Cholesky factorization of the user-sized `A`, and Cholesky solves for each
six-dimensional `S_i`; it never forms a feature-by-feature Gram matrix.

## Full query scores and the absent-target fast path

For query features `F_q`, define the kernel

\[
K_q=F_qD^{-1}F^T.
\]

The full constrained score is

\[
s_{qi}=(K_qC)_{qi}-F_{q,I_i}D_i^{-1}F_i^Tc_i.
\]

For an absent target, `F_q[:,I_i]=0`, so the second term vanishes. For an
arbitrary query it must be subtracted. `predict()` always applies this correction.
`predict_unadjusted()` is explicitly restricted to positions where the target is
absent; it is not a general replacement for the full predictor. The exceptional
SVD path below already stores exactly constrained coefficient columns.

No score masking occurs inside the model. Padding, seen items, candidate
eligibility, and ranking metrics belong to the runner. Any entirely unobserved
target, including an empty padding column, has a zero reconstruction target and
therefore zero fitted coefficients.

## Exact binary EASE nesting

With categorical channels omitted, the objective becomes

\[
\min_B\|X-XB\|_F^2+\lambda_b\|B\|_F^2,
\quad \operatorname{diag}(B)=0.
\]

If `P=(X^T X + lambda_b I)^-1`, the solution is
`B[:,i]=-P[:,i]/P[i,i]` with the diagonal set to zero. The independent tests
compare this item-space formula with the user-space solver. The binary option
is explicit (`category_ratio=None`); infinity is not used as a numerical ridge
value. The categorical model also contains the binary coefficient solution as a
feasible point, but this does **not** guarantee better held-out ranking.

## Computation and numerical safeguards

`prepare_features` caches sparse `F` and the two dense user-space kernels

\[
K_b=XX^T,\qquad K_c=\sum_{r=1}^5Z_rZ_r^T,
\quad A=I+K_b/\lambda_b+K_c/\lambda_c.
\]

Each grid fit reuses those kernels. Working memory is sparse feature storage,
user-square matrices, user-by-item `C`, and blocks of 128 items. Normal fitting
does not materialize the feature-by-item coefficient matrix. One all-target
residual check verifies

\[
Ac_i-F_iD_i^{-1}F_i^Tc_i-y_i=0.
\]

All penalties must be finite and strictly positive. The normal path falls back
for failed factorizations, a normalized Schur eigenvalue below `1e-9`, or a
relative reduced-system residual above `2e-8`. A weighted kernel diagonal above
`1e8` proactively selects the fallback because dual cancellation can destroy
accurate coefficient reconstruction even when factorization succeeds.

The fallback physically removes the target block and computes a reduced SVD
of `Z=F_-i D_-i^(-1/2)=U diag(s) V^T`. It returns

\[
b_{-I_i,i}=D_{-I_i}^{-1/2}
V\operatorname{diag}\!\left(\frac{s}{1+s^2}\right)U^Ty_i.
\]

It stores these exceptional coefficient columns directly and checks relative
ridge stationarity at tolerance `2e-7`. This can cost more for extremely small
penalties, but does not construct a feature-square inverse and preserves the
same objective. Diagnostics report fallback counts, the minimum normalized
Schur eigenvalue, and maximum residuals. The declared experimental grid starts
at `lambda_b=10`, with `rho>=0.1`; the fallback is principally a robustness guard,
not an alternative tuned model.

Saved NPZ models contain numeric TRAIN ratings, fitted centering statistics,
ridge values, `C`, own-block corrections, exceptional coefficients, and solver
diagnostics. Reloading validates centering statistics against the saved TRAIN
matrix and never fits from evaluation data. Serialized arrays use no pickle.

## Scientific scope and primary prior art

[EASE (Steck, 2019)](https://arxiv.org/abs/1905.03375) supplies the binary
constrained linear reconstruction objective and its closed-form solution.
[FEASE (Cui, Zhang, and Lee, 2025)](https://arxiv.org/abs/2504.02288) already
augments EASE with side information. Our experiment evaluates categorical
TRAIN-rating sources with complete own-item exclusion, a nested binary control,
and a within-item rating permutation control. It does not establish that
feature-augmented EASE or the matrix identities are new.

The within-item permutation preserves each item's rating counts and binary
history while disrupting which users supplied which categories. The comparison
tests whether that association helps under the specified split, model, and
regularization grid. It does not isolate a psychological mechanism or prove
causality. The surrounding project has already inspected other MovieLens
results; this is exploratory work, not an untouched confirmation experiment.
