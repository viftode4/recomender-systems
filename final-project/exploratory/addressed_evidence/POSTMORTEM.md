# What the addressed-evidence study actually established

The distinct-source pair operator failed its declared primary comparison. It fitted TRAIN probes more strongly, but assigned much worse probabilities to held-out validation records than its additive control. Its small top-10 ranking gain did not close the gap to existing EASE or SLIM references. The extra term was active, grew sharply during training, and amplified full-history inputs more than the direct term. These observations narrow the failure; they do not establish that another architectural layer is needed.

This is a **post-hoc diagnosis of post-final-test exploratory work**. The 471 meta-fit and 472 development users per seed are reused, although disjoint within each split. This diagnosis opened neither TEST labels nor earlier final-test results, performed no fitting, changed no checkpoint selection, and preserved every sealed source and study artifact. All numbers below describe the existing study or new TRAIN-only forward passes.

## Matched development results

The comparison verifies identical raw-data hashes, TRAIN/validation split hashes, ordered user/item identities, exact cohort-file hashes, and metric denominators against the existing [development reference comparison](../../evidence/field-reference-v1/RESULTS.md). Every row uses 472 development users for all-observed ranking. Liked-ranking users number 444, 434 and 433 for seeds 2026, 2027 and 2028 respectively. Candidates exclude original TRAIN observations. The item/category joint model's all-observed score is `logsumexp` of all five rating logits; its liked-record score uses categories 4 and 5.

| Model | All nDCG@10, 2026 | 2027 | 2028 | Equal-seed mean | Liked nDCG@10 mean |
|---|---:|---:|---:|---:|---:|
| EASE | 0.259563 | 0.262041 | 0.263878 | 0.261827 | 0.244994 |
| SLIMElastic | 0.254628 | 0.260722 | 0.258511 | 0.257954 | 0.242724 |
| PositiveEASE | 0.230180 | 0.242725 | 0.237987 | 0.236964 | 0.257489 |
| Addressed additive | 0.184999 | 0.184797 | 0.186527 | 0.185441 | 0.193335 |
| Addressed pair | 0.185861 | 0.195020 | 0.188609 | 0.189830 | 0.193673 |

Pair improves all-observed nDCG in each seed, by 0.000862, 0.010223 and 0.002082. Liked nDCG changes by −0.005080, +0.008928 and −0.002834; its mean gain is only 0.000338. Both addressed variants trail both EASE and SLIM on both endpoints in every seed. The pair model's mean all-observed nDCG remains 0.071997 below EASE.

These are matched users and candidates, not matched objectives or tuning budgets. EASE and SLIM were selected using meta-fit all-observed nDCG, PositiveEASE using meta-fit liked nDCG, and the addressed models using meta-fit joint NLL. The addressed models also use categorical ratings, whereas the standard EASE/SLIM references use observation identities. The ranking table neither isolates architecture nor implies that EASE's arbitrary scores are joint probabilities. No EASE/SLIM NLL is fabricated. The three overlapping splits are not independent datasets.

| Variant | Development joint NLL ↓ | Item-event NLL ↓ | Conditional rating NLL ↓ |
|---|---:|---:|---:|
| Additive | 7.809877 | 6.362480 | 1.447396 |
| Pair | 8.645867 | 7.096600 | 1.549267 |
| Pair minus additive | +0.835990 | +0.734120 | +0.101870 |

These are equal-seed means of macro-user losses in natural-log units. About 88% of the pair model's excess joint NLL comes from item-event probability, with the remainder from conditional rating prediction. Pair improves individual development-user joint NLL for only 7.20%, 6.57% and 5.30% of users across the three seeds. A small gain in ordering a few top items can coexist with worse probability assignment over all recorded outcomes; the objectives measure different properties.

## More training was already tried, and selection rejected it

Both variants share the same initialization, TRAIN episode masks, Adam settings, maximum of 2,000 epochs, checkpoint opportunities and 200-epoch patience rule. They do not have equal realized training lengths. In all three seeds, additive selects epoch 100 and stops at 300; pair selects the first assessed checkpoint, epoch 10, and stops at 220. Neither reaches the maximum.

| Epoch | Additive TRAIN probe NLL | Additive meta-fit NLL | Pair TRAIN probe NLL | Pair meta-fit NLL |
|---|---:|---:|---:|---:|
| 10 | 8.6132 | 8.6584 | 8.4382 | 8.7356 |
| 30 | 7.9867 | 8.2056 | 7.3339 | 9.1270 |
| 60 | 7.4735 | 7.9514 | 6.5416 | 10.9791 |
| 100 | 7.1122 | 7.9045 | 6.0220 | 16.1542 |
| 200 | 6.5673 | 8.0694 | 5.4208 | 36.9138 |
| 220 | 6.4775 | 8.1156 | 5.3378 | 40.7408 |
| 300 | 6.1542 | 8.3073 | — | — |

The table gives equal-seed means from the immutable traces. The pair model's later checkpoints are much worse on the already-declared selection objective despite lower training loss. This rules out “the run simply stopped at its computational cap” as an account of this run. It does not establish optimal optimization or rule out a differently regularized model. Epochs before 10 were not assessed, so no claim is made about their validation performance.

TRAIN losses average pre-update minibatch losses at different parameter iterates, under a fresh identity-only mask each epoch. They are not fixed-objective convergence certificates. TRAIN probes also differ from validation records in target identities and context construction. The widening gap is evidence of poor generalization under this protocol; its numerical size is not a clean estimate of one isolated cause. See the separate [optimization derivation](GEOMETRY.md) for the additive objective's geometry and its limits.

## The pair term is active and its scale grows

The original uncentered diagnostic gave a roughly 13× pair/direct RMS ratio at the selected pair checkpoint. A shared shift of every eligible logit leaves joint probabilities unchanged, so this postmortem removes each user's mean over the same eligible item/category outcomes before measuring magnitude. It also separates within-item category contrasts from across-item mean contrasts and records covariances with direct and bias terms. These are logit decompositions; nonlinear `logsumexp` ranking scores do not decompose into the same additive contributions. In particular, an item's mean logit is not its item-event score: within-item category dispersion can also change `logsumexp` and ranking.

New inference uses all 943 users' TRAIN histories and only original TRAIN-unseen, nonpadding candidates. No held-out outcome values enter these diagnostics. `best` is the original selection; `latest` is inspected solely to diagnose later training, not to select another model.

| Variant/checkpoint | Centered direct RMS | Centered applied-pair RMS | Centered bias RMS | Centered total-logit RMS |
|---|---:|---:|---:|---:|
| Additive, selected epoch 100 | 0.8029 | 0 | 1.4345 | 1.5745 |
| Additive, latest epoch 300 | 1.7528 | 0 | 3.2698 | 3.4054 |
| Pair, selected epoch 10 | 0.1506 | 2.2440 | 0.2365 | 2.3724 |
| Pair, latest epoch 220 | 0.6704 | 101.3915 | 2.4169 | 101.6528 |

Values are equal-seed means of pooled outcome-weighted RMS. Centering does not remove the pair model's large term: it is about 15× the direct term at selection, and approximately 45× larger at epoch 220 than at epoch 10. Its selected full-history RMS includes both substantial within-item category contrast (1.717–1.727) and across-item mean contrast (1.391–1.467), so it is neither just a joint-probability-invariant common offset nor solely a conditional-rating-invariant item shift. Direct/pair centered covariance is positive in every inspected seed/checkpoint; the much smaller bias term does not cancel the pair scale overall. Large magnitude alone does not prove that the term causes the predictive deficit.

Mean absolute pair coefficient is 0.195 at selection; **zero** coefficients exceed absolute value 0.95. The smallest `tanh` derivative is 0.572–0.583, so early failure cannot be attributed to widespread coefficient saturation. By epoch 220, mean absolute coefficient is 0.634 and 8.38–8.79% exceed 0.95. Allocated compatibility-table RMS grows from about 0.138 to 0.785. The coefficient bound limits `tanh(pair_raw)`, not the pair logit: the latter contains products of learned table values and can grow without bound. Allocated-table RMS includes inactive entries and is a scale diagnostic, not an effective-capacity count.

## Context-count amplification is real, but not a complete explanation

For each fixed selected model, this diagnosis nests 40%, 60% and 80% identity-only subsets of each TRAIN history using the existing epoch-1 masking rule, then restores the full history. It always measures the **same original TRAIN-unseen candidate set**. Thus changing candidate eligibility cannot explain these differences. These are forward-pass interventions only; no loss or ranking endpoint is evaluated under the new masks.

| Fraction of original TRAIN history kept | Selected pair model: centered direct RMS | Centered applied-pair RMS | Centered total-logit RMS |
|---|---:|---:|---:|
| 40% | 0.0649 | 0.4009 | 0.5115 |
| 60% | 0.0934 | 0.8486 | 0.9529 |
| 80% | 0.1219 | 1.4644 | 1.5779 |
| 100% | 0.1506 | 2.2440 | 2.3724 |

The 80%→full change multiplies direct RMS by 1.234–1.236 and pair RMS by 1.530–1.534. Mean visible neighbors per fixed candidate increase from about 3.316 to 4.167, while mean distinct visible-neighbor pairs increase from about 21.28 to 33.50. The number of allocated slots stays 64 throughout.

This agrees qualitatively with the operator's different polynomial orders. With fixed learned source votes, a uniform `m`-of-`H` subsample of the original history gives

`E[direct_mask] = (m/H) direct_full`

`E[pair_mask] = m(m−1)/(H(H−1)) pair_full`.

The exact-count mask uses sampling without replacement, so the second multiplier is not exactly `(m/H)²`. These are expectations of signed terms; they do not specify RMS ratios. For training targets conditioned on being withheld, the population also differs, so the same formula cannot be transferred without adjusting the conditioning.

Two limits matter. First, this observed context change increases the pair/direct RMS ratio by about 1.24×, not 15×. It cannot on its own account for the learned branch imbalance. Second, original meta-fit selection already used **full TRAIN histories**, just like development inference. There was no unnoticed switch from masked to full histories after checkpoint selection. The change is between the training-episode distribution and full-history evaluation, and may interact with memorization, table growth and missing-record competition. This study does not causally apportion their contributions.

TRAIN activity quartiles further illustrate the scale dependence. At the selected pair checkpoint, the lowest-history group (230 users, mean 20.46 TRAIN entries) has centered pair RMS 0.398–0.423; the highest group (237 users, mean 200.46 entries) has 4.162–4.266. These are descriptive magnitudes, not subgroup accuracy gaps: the groups also differ in item and category composition.

## What remains defensible

The defensible result is a negative primary comparison with a small secondary ranking improvement, plus a reproducible diagnosis of scale growth and context sensitivity. It is not evidence that interaction terms never help, that additive models have reached their best performance, or that a larger architecture will solve this problem. This implementation ties each pair interaction to products of the same source vectors used by the direct branch; it does not test arbitrary distinct-source interactions. Both variants allocate 2,709,630 parameters, but additive's pair parameters are inactive and graph/category support limits both variants' active capacity.

This is also one fixed five-category movie catalog and one recorded-event objective. Learned tables are addressed by candidate/source identities. Nothing here establishes transfer to a new catalog, language, continuous targets, or other task types. The measured behavior suggests specific quantities to test in a separately declared study: held-out performance under matched context-size distributions, sensitivity to regularization and logit scale, and whether any pair benefit survives a stronger additive reference. Those are hypotheses, not changes made by this postmortem or promises of a successful next model.

## Reproduction and provenance

The new [diagnostic script](postmortem_diagnostics.py) verifies existing payloads before using them, verifies exact selected-logit replay, checks each latest checkpoint's nested integrity digest, and refuses to overwrite its new output. It never constructs or calls an optimizer. Its full aggregate output, including per-seed values, activity groups, coefficient derivative quantiles, covariances and checkpoint hashes, is [postmortem-diagnostics-v1.json](postmortem-diagnostics-v1.json). No user IDs, per-user metrics, histories or recommendations are exported there. The aggregate privacy guard passed. A synthetic common-shift invariance check, full centered-variance reconstruction including covariance terms, and exact repeated diagnostic values also passed. An independent reviewer checked the formulas, masks and interpretation.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
runs/environment-check/.venv/bin/python \
  -m exploratory.addressed_evidence.postmortem_diagnostics \
  --out runs/addressed-postmortem-replay.json
```

Run from the final-project directory. The new output path must not already exist. Runtime is the sealed Python 3.11.15 / NumPy 1.26.4 / SciPy 1.17.1 / Torch 2.14.0 environment. No dependency installation is needed.

| Artifact | SHA-256 |
|---|---|
| Diagnostic script | `c57116f1691052d63debc9bbed87fbad16c1220fb656dcd424ce745fe7781b0e` |
| Diagnostic aggregate | `68478284baff9f44e0acba4531e40f683177fcba789073fd392b2ecc8848c5e6` |
| Original addressed development aggregates | `f73f1efc32e10c1036a6b2dd440944a3e15a45405913109607f7f949702eac72` |
| Existing matched development reference aggregates | `18498956cd71d25f1af901b0cd8463739bbf21c9bdd57ad3f34dd93d324a1f46` |
| Addressed 2026 run manifest | `4b5b8c0272a741b1be93c46a9516662e08c04a3c519a3ae509ca84f8f2add779` |
| Addressed 2027 run manifest | `f2431100940f8d2a4c03f8a468163207a69188ca68a76d5e387687e4ca877b15` |
| Addressed 2028 run manifest | `2a16983f096b9ba44df8afeb49efc43c838235bb7617340b8067e440056b76a3` |

All original source/runtime signatures and per-payload hashes are retained in the diagnostic JSON and the unchanged [results provenance](results-v1/provenance.json). TEST hashes may appear as existing provenance metadata; no corresponding TEST payload was opened. All development comparison values were read from previously completed aggregate evidence, not recomputed under a new selection rule.
