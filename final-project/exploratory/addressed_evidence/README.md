# Direct categorical evidence experiment

This independently implemented predictor learns from movie/rating identities.
It consumes no fitted expert predictions. The experiment is **post-final-test
exploratory research**: old development cohorts are reused, the original final
test remains outside this workflow, and novelty has not been established.
The training and selection contract is in [PROTOCOL.md](PROTOCOL.md).

## Model and control

`build_neighbors(training_categories, k=64)` constructs one directed source
list per candidate from TRAIN observation-presence cosine. "Positive" means
strictly positive co-observation similarity, not a high rating. Exact ties use
source index; self edges and item-zero padding are excluded. Missing neighbors
are padded with zero. Rating values do not otherwise affect graph construction.

For candidate `i`, output rating `c`, and source item `j` at its unique neighbor
slot, let `s_j = W[i,j,r_j,c]` when that source is observed, and zero otherwise.
The five-by-five table keeps source identity and source rating distinct. Zero
means absent evidence and is not a sixth rating or a dislike.

```text
L = max(K*(K-1)/2, 1)
D = sum_j s_j / sqrt(K)
Q = ((sum_j s_j)^2 - sum_j s_j^2) / (2 * pair_vote_scale * sqrt(L))

additive: z[i,c] = bias[i,c] + D
pair:     z[i,c] = bias[i,c] + D + pair_bound*tanh(pair_raw[i,c])*Q
```

Defaults are `K=64`, `pair_vote_scale=0.01`, and `pair_bound=1`. Fixed scaling
preserves information about the amount of observed evidence. The squared-sum
identity includes each unordered pair of distinct source items exactly once.
It gives zero with zero or one observed source, including `K=1`.

Both variants allocate identical tensors and initialize their compatibility
tables from the same local seed with standard deviation 0.01. Biases and pair
coefficients start at zero. They therefore have exactly identical initial
predictions and direct-table gradients; the additive control leaves its pair
coefficients inactive. Equal allocation does not mean equal active capacity.

The interaction is deliberately restricted: within each candidate and output
category its pair coefficients have the tied form `gamma*W[j,r,c]*W[k,t,c]`,
an outer product before same-source terms are removed. It uses the same tables
as the direct term. It is not a freely learned table for every pair of sources,
and it does not identify psychological conflict or redundant evidence. A bounded
`gamma` does not bound total pair logits because the compatibility tables can
grow. Signed higher-order interactions are established prior work, including
[Steck and Liang's 2021 study](https://recsys.acm.org/recsys21/session-2/).

## Initialization units and measured gradients

If all K initial source votes are independent, zero-mean, with variance sigma²,
`Var(sum(s)/sqrt(K)) = sigma²`. Distinct pair products have variance sigma⁴;
their cross-covariances vanish under these assumptions. Thus
`Var(sum(j<k, s_j*s_k)/sqrt(L)) = sigma⁴`. Dividing by the fixed scale sigma
aligns initial units with the direct branch. When fewer neighbors are observed,
the two branches depend differently on evidence count. Independence is an
initialization argument, not an assumption that remains true after learning.
The fixed `pair_vote_scale` is never recomputed from learned weights or labels.

[initialization-diagnostic-v2.json](initialization-diagnostic-v2.json) records
the current model/source/TRAIN hashes and a no-optimizer diagnostic on 64 TRAIN
profiles, four batches of 16. The same weights, masked probes and objective are
used in every row; initial predictions, losses and direct-table gradients are
exactly identical because pair coefficients are zero.

| Pair feature denominator | Direct feature RMS | Pair feature RMS | Pair coefficient gradient RMS |
|---|---:|---:|---:|
| Number of possible pairs | 0.002315 | 0.0000002344 | 0.0000000001998 |
| Square root of possible pairs | 0.002315 | 0.00001052 | 0.000000008971 |
| 0.01 × square root of possible pairs | 0.002315 | 0.001052 | 0.0000008971 |

The last row is the implemented setting, decided before validation fitting.
The direct-table gradient RMS is 0.00001089 in every row. Direct/pair feature
correlation is about 0.00419 at initialization. Adam rescales gradients, so raw
gradient size alone does not prove that a branch cannot learn. Its epsilon and
the bounded branch coefficient also matter. The artifact includes covariance,
gradient norms, zero fractions and the fractions below Adam epsilon.

The earlier mean-normalized source is retained as
`model-mean-before-correction.py.txt` for the hash in diagnostic v1. It is an
archived design, not the imported model. Neither diagnostic trains a model.

## Capacity accounting

For the seed-2026 TRAIN graph, both variants allocate 2,709,630 parameters:
`M*(25*K+10)` with `M=1683`, including padding and inactive slots. There are
105,954 nonpadding directed edges, or 2,648,850 edge-table entries. Counting
only source categories occurring somewhere in TRAIN leaves 408,778 supported
edge/category combinations, or 2,043,890 potentially addressable table entries.
This is support accounting, not a count of confirmed nonzero gradients.

There are 8,410 real candidate/category biases. In the pair variant, 8,295
candidate/category coefficients have two source items simultaneously observed
in at least one full TRAIN profile, so could carry a pair feature; actual masked
episodes and gradient cancellation can reduce activity. The additive variant
does not use any pair coefficient. The graph and category-support counts depend
on the split; these numbers do not substitute for per-run diagnostics.

## Python interface

```python
from exploratory.addressed_evidence.model import AddressedEvidenceModel, build_neighbors

neighbors = build_neighbors(training_categories, k=64)
model = AddressedEvidenceModel(
    n_items=training_categories.shape[1], neighbors=neighbors,
    k=64, variant="pair", seed=2026,
)
out = model(context_long)  # [batch, items], categories 0..5; padding column zero
logits = out["rating_logits"]  # [batch, items, 5]
```

`model.config` contains JSON-compatible constructor settings. Save it with the
state dict; the graph is a persistent buffer. Restore with
`AddressedEvidenceModel(**config)` followed by strict `load_state_dict`.
`return_diagnostics=True` additionally returns the direct contribution,
normalized pair feature, unscaled mean pair product, pair coefficients and
observed-neighbor counts. Rating probabilities are the five-logit softmax.
The recorded-item ranking score is logsumexp over all five logits; the liked
record score is logsumexp over categories four and five. These are different
quantities from conditional rating probabilities.

From the project root, using the verified training environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python \
  -m unittest exploratory.addressed_evidence.test_model -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  runs/environment-check/.venv/bin/python \
  exploratory/addressed_evidence/diagnose_initialization.py --out /tmp/addressed-initialization-replay.json
```

The diagnostic refuses to overwrite an existing output. Eleven model tests
cover graph construction, source/rating identity, absence and self masking,
the distinct-pair identity, slot permutation, exact initialization, category
gradients, single-slot behavior, scale validation and strict checkpoint replay.
The separate runner implements masking, objective, selection and stopping;
its benchmark uses TRAIN only and discards those fitted benchmark weights.
