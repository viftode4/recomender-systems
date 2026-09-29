# Grouped-pair constrained reconstruction

This describes the frozen [implementation](model.py). Data boundaries, candidate
grids and selection rules are in [PROTOCOL.md](PROTOCOL.md). It makes no accuracy
or novelty claim.

## Visible features and objective

Let `X[u,j]` indicate that user `u` has a visible FIT record for item `j`. Let
`g[u,j]` be that record's exact timestamp, interpreted only as a row-local group
label. No timestamp or group label of a hidden target is needed.

For every unordered pair of distinct catalog items, define

\[
\Phi_{u,\{j,k\}}=X_{uj}X_{uk}\,\mathbf1[g_{uj}=g_{uk}],\qquad j<k.
\]

The bag arm replaces the equality indicator with one. True groups, shuffled
groups and bags therefore use the same conceptual pair dictionary. Grouped
storage omits columns that are zero throughout FIT: ridge assigns those columns
exactly zero coefficients, including when such a pair occurs in a new query.
The all-zero padding item introduces no active feature.

With `beta >= 0`, `lambda > 0`, and normalizer `nu > 0`, set

\[
\gamma=\beta/\nu,\qquad Z=[X,\sqrt\gamma\,\Phi].
\]

For target `i`, let `J_i` contain its singleton column and **every** pair column
containing `i`. Fit

\[
\min_{w_i}\|X_{:i}-Zw_i\|_2^2+\lambda\|w_i\|_2^2,
\qquad (w_i)_{J_i}=0.
\]

Thus the effective penalty on unscaled pair coefficients is
`lambda * nu / beta` when beta is positive. Coefficients may have either sign.
Unrecorded entries are reconstruction zeros; they are not confirmed dislikes.

## Exact kernel and target exclusion

Write `G = X X^T` and `P = Phi Phi^T`. Grouped pair inner products also satisfy

\[
P_{uv}=\sum_{a\in\Pi_u}\sum_{b\in\Pi_v}
 { |a\cap b| \choose 2},
\]

where `Pi_u` partitions only user `u`'s visible items. For bags,
`P = G * (G - 1) / 2`, with elementwise multiplication.

Define `A = I + (G + gamma P)/lambda` and `H = A^{-1}`. If `S_i` is the set of
FIT users observing target `i`, and `E_i` selects those user rows, the prohibited
features contribute only inside that support block. With
`R_i = Phi[S_i, pairs containing i]`,

\[
\Delta_i=(\mathbf1\mathbf1^T+\gamma R_iR_i^T)/\lambda,\qquad
A_i=A-E_i\Delta_i E_i^T.
\]

The desired dual coefficients are `alpha_i = A_i^{-1} X[:,i]`. The implementation
shares the factorization of `A` and solves each support downdate. For
`H[S_i,S_i] = L L^T`, it uses

\[
\alpha_i=H_{:S_i}L^{-T}
 (I-L^T\Delta_i L)^{-1}L^T\mathbf1.
\]

Residual checks guard the update. A failed factorization or excessive residual
triggers a direct Cholesky solve after physically excluding the target features.
A target with no FIT observations has zero coefficients and zero scores.

For an arbitrary visible query `q`, the complete score is

\[
s(q,i)=\lambda^{-1}Z_{q,-J_i}Z_{FIT,-J_i}^T\alpha_i.
\]

The code first evaluates the shared full kernel, then subtracts the target's
singleton and all its pair contributions. If `q_i=0`, those contributions are
already zero; otherwise the correction remains necessary. Scores are unmasked;
the runner excludes visible items and padding from ranking.

## EASE limit and normalization

At beta zero, pair features disappear and only the diagonal singleton constraint
remains. This is exactly binary EASE at the same lambda: with
`Q = (X^T X + lambda I)^{-1}`, `B[:,i] = -Q[:,i]/Q[i,i]`, followed by `B[i,i]=0`.

The default scale is the FIT-only trace ratio

\[
\nu=\frac{\operatorname{tr}(P)}{\operatorname{tr}(G)}
 =\frac{\sum_u\sum_{a\in\Pi_u}{|a|\choose2}}
 {\sum_u|H_u|},
\]

with fallback one if either mass is zero. Size-preserving timestamp shuffles
have exactly the true-group pair mass; the runner checks equality and passes
the true-group normalizer explicitly. The bag arm uses its own declared trace
ratio, whose numerator is `sum_u choose(|H_u|,2)`. This changes feature-family
scale deterministically, not by selection or hidden group size.

## Why a query partition survives

Suppose visible items are `{A,B,C,D}` and the missing candidate is `E`.
Partition `{{A,B},{C,D}}` activates pair terms `c_AB,E + c_CD,E`; partition
`{{A,C},{B,D}}` activates `c_AC,E + c_BD,E`. The singleton score is identical,
but the pair score can differ. For example, effective coefficients
`c_AB,E=1` and the other three zero yield a difference of one. This is a
representational example, not a claim that training learns those coefficients.
Neither partition requires knowing anything about the missing `E` record.

## Relationship to prior work and reproduction

Explicit higher-order features and constrained linear reconstruction already
appear in Steck and Liang, *Negative Interactions for Improved Collaborative
Filtering: Don't go Deeper, go Higher* (RecSys 2021),
[DOI 10.1145/3460231.3474273](https://doi.org/10.1145/3460231.3474273).
The local hypothesis here concerns **which visible item pairs are activated by
recording groups**, not a new ridge solver or a new discovery of higher-order
recommendation.

From `final-project`, run the independent explicit-primal checks with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  runs/environment-check/.venv/bin/python -m unittest \
  exploratory.framing_search.predictive.test_model -q
```

Those tests enumerate the complete pair dictionary and physically remove each
target-owned feature block before solving an independent augmented least-squares
problem. They also check the EASE limit, partition sensitivity, relabeling and
permutation invariance, arbitrary-query correction, and unseen pair columns.
