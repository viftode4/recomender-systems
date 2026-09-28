# Shared interpretation of donor evidence

This model learns one scoring function across query–candidate pairs. Its inputs
describe how other TRAIN users support a candidate in the context of the query's
known records. It has no learned user or item parameters, rating categories,
metadata, or scores from another recommender. The local hypothesis is that
query-relative pattern diversity and coverage can add predictive information
beyond support mass and weight concentration. That is an empirical hypothesis,
not an established explanation of earlier models' errors.

The fixed experiment and information boundaries are in [PROTOCOL.md](PROTOCOL.md).
Closest prior mechanisms and limitations are in [RELATED_WORK.md](RELATED_WORK.md).
This is a small shared predictor over supplied relational summaries, not a system
that learns arbitrary graph operators or discovers a new diversity statistic.

## Input and complete query-row exclusion

Let `X[v,j]` be binary TRAIN incidence for `N` donors and `M` items. A query is
one supplied binary context `q[j]`. Let `V` contain every donor except the
query user's entire row. For an external query there is no excluded donor.
Only candidates with `q[i] = 0` are valid for ranking or supervision. Padding
and all context items are excluded by the runner, not inside the scorer.

Define, for each remaining donor:

```text
h   = sum_j q[j]
d_v = sum_j X[v,j]
c_v = sum_j q[j] X[v,j]
w_v = c_v^2 / (h d_v)                    if h d_v > 0, else 0
a_v[j] = q[j] X[v,j] / sqrt(c_v)         if c_v > 0, else 0
```

`w_v` is squared binary cosine similarity. A nonzero `a_v` is the unit-length
pattern of query items that this donor shares. The candidate may be in a
different donor's history and its history-length denominator; this is legitimate
TRAIN evidence. No hidden target from the query user's donor row contributes.

All statistics below sum over `V`. In particular, the candidate-support
denominator is `|V|`, including empty donor rows. The implementation subtracts
the excluded row from marginal counts and gives it weight zero before computing
every personalized quantity. Arbitrarily replacing that row while keeping the
supplied query context fixed leaves every feature unchanged.

The common context is not individually recomputed for seen candidates. Their
features may contain self-context information and must never be supervised or
ranked. This target-hidden input contract is checked by the runner's masks.

## Nine features

For candidate `i`, define:

```text
n_i = sum_v X[v,i]
M_i = sum_v X[v,i] w_v
T   = sum_v w_v
S_i = sum_v X[v,i] w_v^2
b_i = sum_v X[v,i] w_v a_v
K_i = M_i^2 / S_i
D_i = M_i^2 / ||b_i||_2^2
C_i = count_j(b_i[j] > 0) / h
```

Every division whose denominator is zero returns zero. The code uses explicit
conditional division, not an additive epsilon: even a single tiny nonzero
supporter gives `K_i = D_i = 1`. `C_i` is union coverage of the context, so
repeating a pattern does not increase its covered item count.

The exact feature order, before TRAIN standardization, is:

| Index | Name | Value |
|---|---|---|
| 0 | `log_context_count` | `log(1+h)` |
| 1 | `log_donor_support` | `log(1+n_i)` |
| 2 | `donor_support_fraction` | `n_i / |V|` |
| 3 | `log_weighted_support` | `log(1+M_i)` |
| 4 | `weighted_support_fraction` | `M_i / T` |
| 5 | `mean_supporter_weight` | `M_i / n_i` |
| 6 | `log_kish_count` | `log(1+K_i)` |
| 7 | `log_pattern_count` | `log(1+D_i)` |
| 8 | `pattern_coverage` | `C_i` |

`K_i` measures weight concentration. `D_i` additionally accounts for overlap
between supporting patterns. For positive supporting mass, put
`p_v = X[v,i] w_v / M_i` and `Z_vt = a_v dot a_t`. Then

```text
D_i = 1 / (p^T Z p).
```

This is exactly the order-2 similarity-sensitive diversity of
[Leinster and Cobbold, Eq. 1](https://webhomes.maths.ed.ac.uk/~tl/mdiss.pdf).
It is an existing geometric diversity measure applied to these query-relative
patterns. Since `0 <= Z_vt <= 1` and nonzero patterns have unit norm,
`1 <= D_i <= K_i` whenever the supporting mass is positive. Identical patterns
give one; orthogonal patterns give `D_i = K_i`.

Replicating the whole donor pool preserves `D_i` and coverage. Replicating just
one pattern in a mixed pool changes its relative weight and can increase or
decrease `D_i`. The statistic is neither a literal count of distinct patterns
nor an estimate of independent people, error covariance, trust, or confidence.
An empty query has zero personalized features; it still has candidate marginals.
Candidates without donor records share the same zero-support features and cannot
be distinguished by unavailable identities or content.

## Shared scorer and controls

Let `z` be a nine-feature vector standardized using population mean and standard
deviation over eligible TRAIN-episode positions only. Constant channels use
scale one. The same fitted scaler is used for every arm and later query.

```text
s_theta(q,i) = W2 SiLU(W1 (z(q,i) * mask) + b1) + b2
W1: 16 x 9; b1: 16; W2: 1 x 16; b2: 1
```

There are 177 allocated trainable parameters. No parameter is indexed by a user
or item. Donor and item permutations preserve features and scores after the
corresponding query/candidate reindexing, up to floating-point summation error.

- `full_pattern`: keep all nine standardized channels.
- `no_pattern`: zero channels 7 and 8 after standardization.
- `marginal_only`: keep channels 0, 1 and 2 only.

Initialization, allocated width and training budget match. Effective input
capacity differs because masked input columns are inactive. The direct ablation
jointly removes diversity and coverage; it does not isolate redundancy alone or
either feature individually. An additional inference intervention loads the full
model's selected weights into `no_pattern`; its changed-input performance is not
the same question as retraining without those channels. The analytic donor
reference is unstandardized feature 4 and has no fitted network.

## Recorded-item objective

For an episode, `E_q` is the catalog excluding padding and all context items.
`P_q` contains the held-out TRAIN records, with `P_q` nonempty and a subset of
`E_q`. Every hidden positive remains jointly eligible in the common context.

```text
p_theta(i | q, eligible) = exp(s_theta(q,i)) / sum_{j in E_q} exp(s_theta(q,j))
loss(q) = logsumexp_{j in E_q} s_theta(q,j)
          - (1 / |P_q|) sum_{i in P_q} s_theta(q,i)
loss(batch) = mean_q loss(q)
```

Each episode contributes equally regardless of its number of hidden records.
Missing eligible items compete as alternative outcomes in a multinomial
normalizer; they are not declared explicit dislikes. The task predicts recorded
items and cannot separately identify exposure, liking or willingness to rate.
Context and padding logits are masked before the normalizer and have zero loss
gradient. A common shift of all eligible scores leaves the objective unchanged.

This fitting objective differs from EASE's reconstruction loss. Full versus
no-pattern is the matched mechanism comparison; comparison with a locked EASE
predictor evaluates overall competitiveness and does not isolate one cause.
Shared interpretation of local graph evidence has direct precedents in
[IGMC](https://arxiv.org/abs/1904.12058) and structural-summary predictors such as
[BUDDY](https://arxiv.org/abs/2209.15486). This implementation does not claim that
using relational summaries with an MLP is novel.

## Implementation and replay

`prepare_donors` validates binary dense/CSR incidence. Sparse duplicate
coordinates and nonbinary original entries are rejected rather than coalesced
into plausible observations. Per query, let `J` be the observed context columns.
The implementation forms

```text
B = X^T [ diag(w / sqrt(c)) X[:,J] ].
```

Each row is `b_i` restricted to its possible nonzero context coordinates. This
avoids materializing donor-by-donor similarities or candidate-by-donor-by-item
tensors. Intermediates use float64; the cache is float32 `[queries, items, 9]`.
Counts and all matrix products exclude the query donor as described above.

The pure API is `prepare_donors`, `extract_features`, `fit_normalizer`,
`FeatureNormalizer.transform`, `SharedEvidenceScorer`, and
`masked_listwise_loss`. A replay needs the source-verified donor matrix,
query contexts/exclusion rows, normalizer `mean` and `scale`, scorer `config`,
and PyTorch `state_dict`. The variant mask is deliberately nonpersistent so
loading full-model weights into the no-pattern intervention cannot restore the
removed channels. Network construction uses a local RNG scope and does not
advance the caller's global PyTorch random stream.

Synthetic tests independently enumerate donor contributions, test complete-row
exclusion, geometry limits, weak nonzero weights, permutations, empty inputs,
TRAIN-only scaling, matched initialization, excluded-logit gradients and exact
serialized prediction replay. Passing them establishes implementation properties;
it supplies no evidence of improved recommendation quality. Real-data timing,
selection and results belong to the separately sealed runner artifacts.
