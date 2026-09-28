# Why the earlier 0.32–0.34 scores and newest 0.26 scores differ

The new categorical model has **not been measured on the earlier TEST endpoint**.
Its 0.263154 development nDCG therefore cannot be subtracted from the earlier
EASE TEST result, 0.321550, to measure model deterioration. On the same development
users, target pairs and candidate catalog, the original EASE scores 0.261827.
The new model is slightly better on that comparison, rather than 0.058 worse.
Its primary comparison against the expanded binary grid is 0.263154 versus
0.261688, a small 0.001466 absolute gain.

This diagnosis is post-test exploratory. It reads existing published TEST
**aggregates**, but no TEST interaction file, rating label or recommendation
payload. The new computations use only frozen TRAIN-fitted scores, TRAIN/VALID
masks, coefficients and user-cohort metadata. No model is fitted or selected.
All original scientific sources and evidence remain unchanged.

## The actual assessment contracts

| Property | Original frozen final evaluation | New categorical reconstruction development |
|---|---|---|
| Relevant pairs | Original TEST observations | Original VALID observations |
| All-observed users per seed | 943 | 472 |
| Observation endpoint | Every recorded interaction is relevant, independent of rating | Same relevance rule, different pairs |
| Liked endpoint | Separate ratings >=4 endpoint, with its own denominator | Separate ratings >=4 endpoint, with its own denominator |
| Candidate exclusions | PAD, TRAIN and VALID observations | PAD and TRAIN observations |
| Input history for the standalone score | TRAIN only | TRAIN only |
| Refitting with VALID before final scoring | None | None |
| Catalog | 1,682 real items | Same 1,682 items |
| TRAIN / VALID observations per seed | 80,808 / 9,596 | Identical pairs and identities |
| Primary metric | Macro-user binary nDCG@10, ideal length min(10, relevant count) | Same formula |
| Assessment status | Previously opened final assessment, choices frozen beforehand | Repeatedly used development data, after that final assessment |

The final primary evaluator also evaluates all **943** users. Its **236** users
refer to the earlier development cohort used to choose hybrid hyperparameters,
not the final metric denominator. The final field-reference EASE comparison
also uses 943 all-observed users. These counts are explicit in
[final primary aggregates](../../evidence/final-primary-v3/aggregates.json) and
[final reference aggregates](../../evidence/final-field-references-v1/aggregates.json).

The mask change is an explicit protocol choice, not extra training: the final
score function excludes validation items from recommendations while retaining
the original TRAIN-fitted predictions. See [freeze.py](../../freeze.py),
`score_models`, and [field_reference_evaluation.py](../../field_reference_evaluation.py),
`freeze_seed` / `evaluate_frozen`. The new contract is in
[the reconstruction protocol](../categorical_reconstruction/PROTOCOL.md).

## Quantifying cohort and candidate changes without TEST labels

The following uses the **same locked EASE predictor**, ridge 250 in each seed.
All fresh metrics in this table use VALID labels and TRAIN-only masking.

| Cohort | Users per seed | Mean VALID nDCG@10 |
|---|---:|---:|
| New development cohort | 472 | 0.261827 |
| Every original user | 943 | 0.265326 |
| Original hybrid-choice subset | 236 | 0.249890 |
| Original policy-calibration subset | 236 | 0.273765 |
| Original meta-fit subset | 471 | 0.268831 |

The all-user cohort shift is +0.003498 under the same VALID assessment. The
published EASE TEST mean is 0.321550, leaving +0.056224 between the two all-user
assessments. That residual combines different relevance pairs and a different
candidate mask. These aggregates **do not isolate their separate contributions**.
In particular, it would be wrong to attribute the whole residual to masking.
The meta-fit number is a descriptive replay on selection-used labels, not a
new unbiased performance estimate.

The candidate change is small in catalog count but concentrated near the top:

| Diagnostic on the 472 new development users | EASE | Original disagreement hybrid |
|---|---:|---:|
| Mean TRAIN-only eligible items | 1,599.40 | 1,599.40 |
| Mean eligible items after additionally excluding VALID | 1,589.62 | 1,589.62 |
| VALID items removed from the existing top ten per user | 1.751 | 1.818 |
| Fraction of top-ten lists changed by that additional mask | 78.11% | 79.45% |

Only about 9.787 candidates per user are additionally removed, but they include
many highly ranked items. This establishes a substantial change in what the
ranking procedure can recommend. It is **not** a new accuracy evaluation under
that mask: validation relevance would itself be excluded, and TEST labels were
not opened. The number of removed top-ten VALID items is also exactly ten times
VALID Precision@10; it is not an independent success metric.

The old EASE numerical compatibility check independently reproduced all nine
seed/penalty settings within 2.7e-6. Eight matched every top-ten list; one changed
one user's near-tied list. The new selected predictions replay exactly. Thus
there is no evidence that the 0.32 versus 0.26 headline difference comes from an
incorrect EASE implementation. See [audit-v2](../categorical_reconstruction/audit-v2/audit.json).

## The two hybrids are also different models

The original 0.337012 final model is a regression over **14 active nonrandom
expert scores plus an expert-disagreement feature**. Fifteen score exports were
available; Random is excluded from the hybrid's input features. Its experts
include multiple collaborative families and earlier custom rating models.
Coefficients are fitted on 471 meta-fit users; a four-penalty family choice is
made using 236 other users. The remaining 236 users calibrate a separate
reranking policy. The frozen expert ordering, coefficient maps and selections
are in `runs/frozen-v3/{seed}/freeze.json`; the selection procedure is explicit
in [study.py](../../study.py).

The new 0.265666 model has only three inputs: expanded binary EASE, locked SLIM
and the selected categorical reconstruction model. It uses monotone affine
calibration followed by a sum-one ridge combination. Its inner coefficient-fit
and penalty-selection cohorts both come from the 471 meta-fit users; the final
coefficients are then refitted on all 471. Its 472 development labels do not
choose any of its models or coefficients within this study. This is a narrower
test of whether the categorical prediction adds useful information to those two
strong references, not a replacement experiment for the old 14-expert hybrid.

For transparency, the **unchanged old hybrid** scores 0.272703 when replayed on
the same 472 VALID users and TRAIN-only candidates as the new study. However,
236 of these users' VALID labels selected its hyperparameters. There is zero
overlap with its coefficient-fit cohort, but the selection overlap remains.
The 0.272703 versus 0.265666 difference is therefore descriptive and cannot
establish superior generalization under matched selection access. A clean
comparison would require a new predeclared assessment, not relabeling this
replayed score as held out.

The original hybrid's published +6.60% figure uses its **preselected expert
role**, whose final mean is 0.316133, as its reference. That role selects EASE
for one seed and SLIM for two. It is not the same reference as the always-EASE
mean 0.321550. This distinction matters even within the original final results;
see [the declared comparisons](../../evidence/final-comparisons-v3/SUMMARY.md).

## What the successive models actually tested

These earlier all-observed development rows have the same 472-user cohort and
TRAIN-only candidate mask. Training objectives and selection budgets differ,
so this is a practical comparison, not an isolated architecture ablation.

| Predictor | Mean development nDCG@10 | Measured limitation |
|---|---:|---|
| Original EASE | 0.261827 | Strong direct collaborative reference |
| Conditional adaptive field | about 0.0131 | Rating likelihood was a poor objective for predicting which item is recorded |
| Joint adaptive field, up to 400 epochs | 0.22193 | Better objective alignment, still below EASE; selection loss was still improving at the cap |
| Joint fixed-flow field, up to 400 epochs | 0.22589 | Adaptive routing did not consistently improve the simpler control |
| Addressed additive operator | 0.185441 | Direct item/rating tables alone did not ensure strong generalization |
| Addressed distinct-source pair operator | 0.189830 | Small ranking gain, but much worse held-out joint likelihood |
| Expanded binary reconstruction | 0.261688 | More penalty choices slightly reduced development performance |
| Categorical reconstruction | 0.263154 | Small gain; binary fallback selected in one of three seeds |

The field diagnosis does **not** prove that too few dimensions caused failure.
A TRAIN-only compression of already-fitted EASE predictions to 16 interaction
components, retaining item means, still scored 0.26020. This shows that compact
useful predictions exist, not that a directly trained low-dimensional model can
learn them. The ports were not universally collapsed, and the decoder was not
globally saturated. See [the measured field diagnosis](../field-diagnosis-v1/FINDINGS.md).

The addressed pair study was allowed up to 2,000 epochs with a fixed patience
rule; it did not merely stop at a small compute cap. All pair models selected
epoch 10 and stopped at 220 as meta-fit likelihood deteriorated despite falling
TRAIN loss. The selected pair branch's centered logit RMS was about 15 times
the direct branch; it grew further, and full histories amplified it more than
linear evidence. Its bounded coefficient did not bound the quadratic logit.
These observations identify testable scaling/generalization concerns, not proof
that every interaction architecture fails. See [the postmortem](../addressed_evidence/POSTMORTEM.md).

The explicit-negative studies answer a different question. Under liked-rating
relevance, PositiveEASE reached 0.2575, versus about 0.2450 for ordinary EASE;
adding dislike channels did not consistently improve the positive-only control.
The later context/probe rejection audit found no stable benefit from personalized
dislike placement under its estimator. Those weaker context/probe scores cannot
be mixed into the full-history all-observed table. See
[the rating study](../../evidence/exception-v2/RESULTS.md) and
[the information audit](../../evidence/negative-information-v2/RESULTS.md).

## Limits of the newest design, rather than a claim that research is finished

The categorical reconstruction solver reaches the exact optimum of its declared
strictly convex ridge objective; all 243 grid fits had relative residual below
1.8e-13. Additional epochs are not a missing opportunity for this solver. More
work would mean changing the objective, representation, regularization or
assessment, with a new experiment, rather than training this same solution longer.

Its score is still an additive sum of candidate-specific contributions from
individual observed source items and their categories. A source's contribution
cannot switch because a second source is present. It has no learned cross-source
interaction, temporal context, item text, user-specific ordinal scale or new-item
representation. Missing target entries are reconstructed toward zero, and the
loss treats all matrix cells rather than optimizing top-ten utility directly.
These are structural limits, not demonstrated explanations for the exact size
of this gain. EASE shares some of these limits and remains strong.

The nine binary penalties, four category ratios and fixed smoothing value are a
declared finite search, not proof of optimal regularization. The categorical
selection family has 45 choices including binary fallback versus nine binary
choices; this unequal budget is disclosed. Actual category penalties selected
in two seeds are ten times their binary penalties; the third seed rejects the
additional categories. MRR falls slightly, and adding real categories to the
two-expert hybrid changes mean nDCG by only 0.000073. These facts support a narrow
incremental result. They do not support either a breakthrough claim or the claim
that further research is pointless.

The most defensible next question is whether a specified new mechanism improves
over the strong **same-protocol** binary/additive control, with equally careful
tuning and an eventual new assessment. Existing TEST values cannot become an
untouched confirmation set again. Benchmark numbers from another study also need
matching splits, rating thresholds, sampled versus full catalogs, masks, metric
formulas and selection rules before they can answer why it scored higher.

For a concrete example, the original EASE paper reports **0.420 nDCG@100 on
MovieLens 20M** using disjoint training/validation/test users. Its **0.6258
nDCG@10** result instead uses MovieLens 10M, with at most 30% of each user's
history for training and the remainder for testing. Neither is our MovieLens
100K assessment. The different amount of relevant held-out material changes the
ranking task; it does not by itself quantify the expected score increase.
Likewise, increasing the cutoff does not necessarily raise normalized DCG,
because its ideal-ranking denominator also changes.
[Primary source: EASE, Section 5.1 and Tables 1–2](https://arxiv.org/pdf/1905.03375).

## Reproduction

The aggregate-only measurements and input hashes are in
[comparability-v1.json](comparability-v1.json). The new script verifies frozen
payloads, independently reconstructs the old disagreement score, checks its
original validation metric, and verifies the same-cohort EASE replay. It never
calls a TEST evaluator or constructs an optimizer.

```sh
runs/environment-check/.venv/bin/python \
  exploratory/research_diagnosis/compare_protocols.py \
  --out exploratory/research_diagnosis/comparability-reproduction.json
```

Run from the project root with its existing artifacts. The script fixes
numerical library threads to one and refuses to overwrite output. Its output
contains no user identifiers, individual histories, item lists or recommendations.
