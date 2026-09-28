# Optimization geometry of addressed categorical evidence

This is a derivation from `model.py` and the declared joint recorded-item/rating loss, not a new experiment or a change to `PROTOCOL.md`. No fitted results are needed. The current Adam study remains unchanged. Any optimizer, regularizer, initialization or mask-bank follow-up below needs a separate declared protocol and output directory. This is exploratory work after the earlier project test was examined, not independent confirmation or a novelty claim.

## 1. The additive training problem is convex

Fix the TRAIN-derived directed neighbor graph, context masks, probe labels and candidate eligibility. Let `x_ujr` indicate that source item `j` is visible with category `r` in episode `u`; absent evidence has all five indicators zero. With `K` allocated neighbor slots, the additive logit for candidate `i`, category `c` is

\[
z_{uic}=b_{ic}+K^{-1/2}\sum_{j\in N_i}\sum_{r=1}^{5}W_{ijrc}x_{ujr}.
\]

Stack the tables and biases into `θ`. On the eligible outcomes `A_u = {(i,c): i is real and absent from context, c=1,…,5}`, this is `z_u=X_uθ` with a fixed design matrix. Let `q_u` place equal mass on that user's held-out probe records. The implemented mean-probe loss is

\[
\ell_u(\theta)=\log\sum_{a\in A_u}e^{(X_u\theta)_a}-q_u^TX_u\theta.
\]

For positive episode weights `ω_u`, writing `p_u=softmax(X_uθ)` gives

\[
\nabla^2 F=\sum_u\omega_u X_u^T[\operatorname{diag}(p_u)-p_up_u^T]X_u,
\qquad
v^T\nabla^2Fv=\sum_u\omega_u\operatorname{Var}_{a\sim p_u}[(X_uv)_a]\geq0.
\]

Thus the additive loss is convex in the compatibility tables and biases. Masking excluded outcomes to minus infinity simply restricts the rows; fixed square-root scaling does not change the conclusion. This is the standard affine log-sum-exp geometry, derived here for the actual loss. [Boyd and Vandenberghe, convex-functions slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf).

The expectation over **parameter-independent** fresh masks is convex too. Fresh masks add sampling noise, not nonconvexity. This proof does not cover learned graph selection, parameter-dependent candidate mining, the discrete hyperparameter search, or nDCG/Recall. Convexity of training loss does not order validation or ranking performance.

## 2. Convex does not mean identifiable or finitely attained

For a fixed bank, an additive perturbation is an exact joint-probability gauge precisely when

\[
X_u\,\delta\theta=t_u\mathbf 1 \quad\text{for every episode }u.
\]

The same condition characterizes the Hessian null space at finite parameters: all eligible outcomes have positive softmax probability, so each variance above must vanish. Consequences:

* Adding one common constant to all real candidate/category biases is a joint gauge. Adding a separate constant to each candidate's five categories preserves **conditional rating** probabilities but generally changes joint item mass. Pure candidate offsets must agree whenever the candidates are co-eligible; disconnected components of that co-eligibility graph can admit additional offsets.
* A category-dependent offset shared across items is generally not a gauge: it changes category odds. Only a common offset across all eligible outcomes is automatically invisible to the joint loss.
* Adding `d_ijr` to every output-category entry of one source-category table shifts candidate `i` by `g_ui=K^(-1/2) Σ_jr d_ijr x_ujr`. This preserves its conditional rating probabilities, but preserves the joint distribution only if the resulting shifts, including any bias compensation, agree across eligible candidates.
* In particular, a source-specific shift `d_ijr=a_jr` is **not automatically** a joint gauge: different candidates have different top-64 neighbors. With a complete graph excluding self, it would be a gauge, because every eligible candidate is absent from context and therefore sees the same observed sources. The actual sparse graph needs the explicit condition above.
* Padding parameters, padded source slots, `pair_raw` in the additive variant, and source-category features never active while their candidate is eligible are unidentifiable. Finite mask banks can introduce further feature dependencies. A feature constant over its relevant episodes can alias a bias; changing masks can break that alias.

Therefore centering every table across output categories is not automatically a harmless gauge fix: it can remove identifiable item-mass information. Table signs and pair signs have no identified psychological meaning.

The protocol has zero weight decay. Separation or quasi-separation can make the unregularized infimum occur only as parameter norms diverge; this is a possibility, not a claim about these data. Convexity alone guarantees neither a unique solution nor a finite minimizer. Positive quadratic regularization on all free coordinates makes this finite-dimensional objective strongly convex and coercive; penalizing only tables does not generally guarantee finite bias estimates. Such regularization changes the statistical objective. An exact zero gradient of the convex objective certifies global optimality if attained; a small gradient alone is not a general numerical objective-gap certificate without additional bounds.

## 3. What the additive representation excludes

For any two eligible item/category outcomes, their log-odds are affine in visible source-category indicators. There are no explicit source-pair terms in those log-odds. A candidate's raw logits only depend on its fixed neighbors, although changes to other candidates can still change the shared joint normalizer. Candidates with no visible neighbors fall back to their biases.

This does **not** mean probabilities or item ranking scores are additive: softmax normalization and the item score `logsumexp` over five categories are nonlinear. Apparent probability interactions alone do not establish a missing explicit pair mechanism. Fixed graph coverage, limited observations per table entry, and the difference between 80%-masked training contexts and full-TRAIN prediction contexts remain statistical limitations even with excellent optimization.

## 4. The tied pair model is nonconvex

For one candidate/output category, write `s_j=Σ_r W_jr x_jr`, `B=pair_vote_scale × sqrt(max(K(K−1)/2,1))`, and `α=pair_bound × tanh(γ)`. The actual pair logit is

\[
z=b+\frac{\sum_j s_j}{\sqrt K}+\frac{\alpha}{B}\sum_{j<k}s_js_k.
\]

Here is an explicit negative-curvature slice. Take `K=2`, two visible source items, one eligible candidate with five categories, zero biases and a probe in category 2. Give its wrong category 1 source votes `t` and `−t`; set other votes to zero. Fix `α=0.5`, `pair_vote_scale=0.01`. The direct term cancels and the wrong-category logit is `−50t²`, so

\[
f(t)=\log(4+e^{-50t^2}),\qquad f''(0)=-20<0.
\]

This is an affine line in table parameters with the pair coefficient fixed. Nonconvexity therefore comes from the tied products, not merely the `tanh` parameterization. A synthetic float64 check using the actual model and `joint_probe_loss` reproduced `f''(0)=-20`, `f(0)=1.6094379124`, and `f(±0.1)=1.5274750053`, with no optimizer steps or dataset reads.

At fixed tables, logits are affine in biases and **α**, making that subproblem convex with box constraints on α. It need not be convex in raw γ. The original finite γ parameterization reaches only the open coefficient interval; a closed-box solve includes limiting endpoints. Bounded α does not bound logits because the tables are unbounded, and saturated `tanh` can suppress coefficient gradients.

At initialization `γ=0`, the two variants have identical predictions and identical table/bias gradients. The pair-coefficient gradient can still be nonzero. Equal initialization does not imply equal later geometry or effective capacity.

## 5. The pair coefficients have restrictive algebraic ties

For distinct sources, the category-pair coefficient is

\[
\Theta_{(j,r),(k,t)}=\frac{\alpha}{B}W_{jr}W_{kt}.
\]

Every distinct-source 5×5 block has rank at most one; blocks share the same source vectors and one coefficient for this candidate/output category. These are also the vectors used in the direct term. The underlying outer product is rank one, but the matrix after deleting within-source blocks can have higher rank. Calling that whole masked matrix rank one would be incorrect.

For fixed source categories, let `v_j=W_jr_j` and `a_jk=(α/B)v_jv_k`. Every nonzero triangle obeys

\[
a_{jk}a_{k\ell}a_{j\ell}=(\alpha/B)^3v_j^2v_k^2v_\ell^2.
\]

Hence all nonzero triangle products for one candidate/category have the sign of α. A coefficient pattern with mixed triangle-product signs cannot be represented. Four distinct sources additionally satisfy `a_jk a_ℓm = a_jℓ a_km = a_jm a_kℓ`. Zeros are degenerate cases, not violations. These are testable restrictions on computational interactions, not evidence about human agreement or conflict.

Rescaling all votes and inversely rescaling α can preserve the quadratic branch, but generally changes the direct branch; it is not a gauge of the full model. Adding independent factors can break the single-factor restrictions, but belongs to established polynomial/factorization-machine designs. It would need a predeclared complexity criterion and a new study. [Blondel et al., Polynomial Networks and Factorization Machines](https://arxiv.org/abs/1607.08810); [Higher-Order Factorization Machines](https://arxiv.org/abs/1607.07195).

## 6. A concrete follow-up that spends effort on optimization first

The strongest immediate follow-up is a well-optimized additive reference, not an assumption that a larger model is needed:

1. Declare a fixed mask bank, its seeds, graph, objective, regularization choice and stopping criteria before further development comparisons. Regenerating masks deterministically from stored seeds is sufficient. Use the same bank for both variants.
2. Stream the exact bank objective and full gradient across users. For `Q` equally weighted masks and `N` users, a batch's already user-mean loss contributes `batch_size/(NQ)`. Accumulate gradients across all batches before any step. Every line-search closure must evaluate the same bank at the supplied parameter iterate, without intermediate optimizer updates.
3. Try limited-memory BFGS for the additive problem, or a Hessian-vector method using the covariance formula above. Benchmark closure time and peak memory first; repeated full passes and curvature history can dominate costs. PyTorch requires a closure that reevaluates the model and returns the loss, and documents substantial history memory. [Official L-BFGS documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html).
4. Record the objective evaluated at a single iterate, gradient norms, parameter norms, function evaluations and stopping reason. The present epoch training trace averages pre-update minibatch losses at different iterates and uses new masks each epoch; it is not a fixed-objective convergence certificate. A meta-fit plateau is also not a training stationarity certificate.
5. In a separately declared comparison, initialize the pair model from the optimized additive tables with α=0. Optionally first solve only the fixed-table convex coefficient subproblem, then investigate full nonconvex refinement. This separates a benefit of explicit pair features from a failure to optimize the additive control.

Changing masks inside ordinary L-BFGS closure reevaluations gives different functions to its line search and curvature updates. A stochastic quasi-Newton method requires a different design. Conversely, minimizing a finite fixed bank does not automatically minimize the full expected fresh-mask risk.

The pair function class contains the additive class at α=0, so its **global infimum on the same unregularized bank** cannot be worse. A reached Adam solution, early-stopped model, validation loss or ranking score has no such guarantee. Unrestricted coefficients on fixed pair features would make training convex again, but create a much larger model; learned low-rank factors trade that convexity for parameter sharing. None of these arguments establishes a performance gain.

## Scope and source fingerprints

The analytical claims assume the exact fixed features and loss described above. Only a synthetic curvature calculation was executed for this note; no fit or held-out evaluation was performed. The current training run and its stopping rule are unchanged.

| Inspected source | SHA-256 |
|---|---|
| `model.py` | `7677a451e9eeb2b197ab9312b2fa744ae3b211131dbff2329c88402e01214a1d` |
| `PROTOCOL.md` | `5c869a9b6772c4816006e80a05687155bab0d978da7749f8b7f4fc98cf23618c` |
| `run_experiment.py` | `8abeb6fb5cde0048f250b0f83b2f8446cd2e2f35edad488550c7f14a1645c6f1` |
| `../../joint_field_experiment.py` | `e2228dd51c4ac909c587eb98cff795872d23232dd68e3790ce97957c40e28c0f` |
| `../../categorical_experiment.py` | `2d836a4acdab2bf09fdb54140ee98f2eeb3cf2be98607c21fe8911c59130ad1d` |
