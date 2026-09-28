# Prospective structural-program stress test, version 1

This protocol is fixed before the stress results are generated. It extends the synthetic proof without changing `core.py`, the original proof, its artifacts, or any sealed recommender source. It is exploratory work after the original MovieLens final test, and reads no real data.

## Question and bounded design

When does choosing a small program overfit a short calibration history or unreliable supplied atoms? This tests selection within an existing max/distinct-product grammar, not discovery of operators, predicate learning, or recommendation quality.

Run all **six tasks × three FIT sizes × five seeds = 90 cases**. Seeds are `3101,3102,3103,3104,3105`; FIT sizes are `16,64,256`. Each case has a disjoint PROBE sample of the same size and an evaluation sample of **5,000**. FIT/PROBE sizes refer to labeled calibration candidates, not the number of evidence witnesses. There are four unique synthetic witnesses and seven supplied atoms in every case. All methods see exactly the same FIT and PROBE arrays. Candidate IDs are disjoint from support identities.

Each case uses an independent seed stream identified by base seed, task index, sample size, and partition. Evaluation arrays and outcomes are generated only after every selected model/reference has been serialized into that case's `selection.json`. No budget, task, seed, penalty, or reference changes are allowed after inspecting results. All 90 rows are retained, including failures.

## Data-generating tasks

Two independent latent bits A,B have probability 0.5. The observed signal atoms occupy different genuine witnesses. Atom 2 is an independent nuisance bit; atoms 5 and 6 are independent Bernoulli(0.15) values at each witness. Noisy signals independently flip A and B with probability 0.15. Labels are independent Bernoulli draws given the latent target, with probability 0.94 for target 1 and 0.06 otherwise.

1. **clean_conjunction:** observed signals are A,B; target A AND B. Atoms 3,4 are independent witness noise.
2. **noisy_conjunction:** observed signals are noisy A,B; target still latent A AND B. Atoms 3,4 are independent witness noise.
3. **correlated_distractors:** same noisy conjunction, but atoms 3,4 copy the respective *observed* signals with an additional 0.05 flip probability, at other witnesses. These proxies add no information about latent truth conditional on the supplied signals, although finite samples can make them attractive.
4. **contradictory_correlates:** same as task 3, with proxies of the complemented observed signals. Again they add no conditional information; this is a test of redundant contradictory-looking measurements, not identified psychological conflict.
5. **xor_misspecified:** clean signal atoms, target A XOR B, independent distractors. The monotone grammar lacks the mixed signs needed to express this target using those signal atoms.
6. **no_signal:** all supplied atoms follow the clean-signal/nuisance generator, but outcomes are independent Bernoulli(0.5). A useful structure cannot exist in the population.

Independent distractor atoms 3,4 use Bernoulli(0.15) values at all four witnesses. Correlated versions instead occupy one witness each. Random outcomes are generated only after atom generation and do not enter atoms. Atom-column names do not enter search scores.

## Search and controls

The unchanged structural core uses three rounds, beam width 6, maximum 7 nodes, maximum depth 3, and at most **320 evaluated unique syntax trees**. FIT logistic loss plus **0.003 per node** orders the beam. Each tree has only an affine output calibration, with nonnegative scale, fitted on FIT by the original solver with ridge **0.0001**. PROBE logistic loss plus the same node penalty chooses one tree. Ties prefer smaller trees, then canonical syntax. The initial tree is the predeclared irrelevant `a5` for every case.

Controls:

- **fixed_structure:** `a5`, with FIT calibration only. It is a deliberately weak coefficient-only control, not the main performance reference.
- **selected_unary:** calibrate all seven unary atoms on FIT; select one using the same penalized PROBE objective. This is a stronger coefficient/feature-selection comparison.
- **random_grammar:** enumerate the entire canonical bounded syntax space, then sample unique trees uniformly without replacement, including the common initial `a5`. Calibrate them on FIT and select on PROBE identically. The number is **exactly the actual number of trees evaluated by the beam** in that case. Its random draw sees no labels; the count is allowed to depend on the beam's FIT-only work. Uniform syntax sampling is not uniform functional sampling. Different syntax can express the same function, for both methods.
- **linear_and_pairs:** the existing signed logistic reference on every supplied unary and pair feature, fixed ridge **0.001**, no PROBE tuning. This is a fixed-grammar reference with greater coefficient flexibility. Its different capacity and regularization prevent interpreting tiny score differences as a controlled superiority claim.
- **intercept:** a FIT-only Bernoulli rate with a Beta(0.5,0.5) prior, no feature use.

Every method yields one final predictor; no predictions are ensembled. Total tree evaluations, pool hashes, selected expressions, fit/probe loss, candidate count, and elapsed computation are recorded. Search bounds and runtime/source signatures are written before fitting. Uniform syntax sampling and local beam edits have different size, depth, and functional priors; a beam advantage would not isolate an effect of adaptive reasoning.

## Endpoints and interpretation

Primary endpoint: evaluation binary negative log-likelihood. Also report accuracy, MSE to the known latent outcome probability, selected node count, and evaluation NLL minus PROBE NLL (selection optimism). Report every seed, and means/sample standard deviations for each task/size/method. Paired beam-minus-reference NLL differences are reported for all five seeds; no significance or population-generalization claim is based on five repeats.

The planned diagnosis is whether beam selection remains useful as FIT/PROBE shrink, and whether low PROBE loss exaggerates evaluation quality. The no-signal and XOR tasks must remain visible even when they fail. The atoms and fully labeled binary outcomes remain synthetic; a real recommender requires a leakage-audited atom learner and a recorded-event objective. More search candidates are extra selection opportunities, not evidence of more reasoning.

Run on one numerical CPU thread. No MovieLens fitting, final-test access, larger grammar, or follow-up tuning is authorized by this protocol.

## Technical amendment after an incomplete first execution

The first execution stopped after 77 completed cases when the unchanged core's L-BFGS-B affine calibration reported `ABNORMAL` for `no_signal`, FIT/PROBE 16, seed 3103. No numerical result tables were inspected before this correction. The incomplete output, error, and exact source snapshot are retained in `runs/structural-program-stress-v1`.

A stress-only wrapper now calls the original calibration first. Only if it raises a numerical optimization error does the wrapper solve the **identical bounded, ridge-regularized two-parameter objective** with SLSQP and require projected-gradient residual below 1e-5. Every fallback is recorded. The original core and v1 proof remain unchanged. No task, data, seed, search bound, penalty, reference, or endpoint changed.

The complete run is repeated into `runs/structural-program-stress-v2`. Every previously completed case's chosen predictors, candidate pools, data hashes, and metrics (excluding runtime) must match the original execution exactly. This checks that the correction did not alter the earlier successful work. This is a numerical repair, not an outcome-driven follow-up experiment.
