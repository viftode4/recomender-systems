# A bounded structural program proof

This is an exploratory synthetic prototype, separate from all sealed recommender experiments. It reads no MovieLens data or old final-test labels. It selects **one expression tree**, with one affine probability calibration. The implementation does not combine trained recommenders.

For supplied relational atoms `X[candidate, evidence identity, atom]` in `[0,1]`:

```
Atom(a)(i)       = max_e X[i,e,a]
Product(a,b)(i)  = max_{e != f} X[i,e,a] * X[i,f,b]
Merge(P,Q)(i)    = max(P(i), Q(i))
p(y=1 | i)      = sigmoid(b + w * Program(i)), w >= 0
```

Evidence identities are deduplicated before constructing atoms; direct duplicate identities are rejected. Product requires two genuine witnesses. This requirement is supplied by the grammar, not learned. Candidate IDs must be disjoint from evidence IDs, including calibration candidates.

The search changes actual program structure: replace a predicate, introduce a distinct-witness product, add a merge branch, or delete a branch/premise. A bounded beam is ordered by FIT logistic loss plus a node-count penalty. Every evaluated tree is counted toward a hard candidate limit; ties prefer fewer nodes and then deterministic syntax. Each tree's `b,w` are calibrated on FIT only. The entire pool is fixed before PROBE labels choose the final tree. This is inner model selection, not an untouched evaluation. The synthetic evaluation sample is constructed only after the selected program and references have been serialized.

Run from the project root:

```
python -m unittest exploratory.structural_program.test_core
python -m exploratory.structural_program.synthetic_proof --out runs/structural-program-synthetic-v1
```

The proof includes conjunction recovery amid label noise, deletion of a functionally redundant branch, a fixed-structure calibration-only control, and signed logistic regression over the same unary and pair atoms. The XOR task deliberately exceeds this monotone tree grammar and must be reported even if it fails. The signed linear/pair reference can represent XOR. Recovery means selection of a supplied operator composition, not discovery of logic. The synthetic features provide unusually clean latent concepts; no claim is made that an effective MovieLens atom builder exists yet. No recommendation-quality or novelty claim follows from these tests.

Relevant established work: [higher-order factorization machines](https://arxiv.org/abs/1607.07195), [neural collaborative reasoning](https://arxiv.org/abs/2005.08129), [differentiable inductive logic programming](https://arxiv.org/abs/1711.04574), and [MeLU user-specific meta-adaptation](https://arxiv.org/abs/1908.00413). These already cover important parts of higher-order interaction modeling, logical recommendation, learned rules, and local adaptation. This small prototype does not establish a new research family.
