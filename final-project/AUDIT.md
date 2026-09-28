# Independent assignment and implementation audit

Audit date: 2026-09-28. This records the development checkpoint before the new
exception-learning study. Source line numbers describe the inspected checkpoint;
they may move as fixes are integrated. No test labels or test scores were read.

## Follow-up checkpoint after implementation

The findings below are the original audit trail, not a list of unresolved
defects. A subsequent source review verified these corrections:

- `study.py` includes and tunes the equality-constrained ridge baseline, with
  unpenalized intercept and explicitly permitted negative weights.
- Its default user cohorts are 471 coefficient/expert fitting users, 236
  development/model-selection users and 236 policy-calibration users. The
  policy boundary rejects overlap; low-support groups fall back to the baseline.
  Repeated development and historical exploration still prevent treating this
  calibration split as pristine confirmatory research.
- All four reranking objectives are present: diversity, genre calibration,
  personalized popularity calibration and item exposure. Position-discounted
  exposure includes unexposed catalog items, group-size normalization, Gini and
  entropy. Raw exposure remains separately labelled for quota interpretation.
- Candidate feasibility and adaptive expansion check both head and tail
  support. The comparison with full-pool reranking remains an empirical check,
  not an end-to-end retrieval-speed claim.
- Local FISM training and full-catalog scoring consistently remove target
  history contributions, and training uses logits exactly once. UserKNN aliases,
  NGCF sparse compatibility, content-only training profiles and model-specific
  hyperparameters were independently inspected. Focused adapter tests passed.
- `freeze.py` seals transformations, source hashes and expert order, and requires
  exact validation-ranking replay. `final_evaluate.py` checks the held-out split
  hash and reports every frozen comparison without test-based selection or
  refitting. These paths were inspected without opening real test interactions.
- Rating-aware contrast experiments now have tuned positive-only, signed and
  dual-channel linear controls and equal-kernel pairing ablations. Their
  operational pair definition does not establish psychological exceptions,
  architectural novelty or causal preference discovery.
- The report builder, package builder and artifact-driven summary exist.
  Expanded evidence and report rendering still need their own successful runs;
  source inspection alone does not establish a finished deliverable.

Remaining grading-critical completion steps are the reviewed task-formatted
report, verified runnable package, actual member/contribution details and the
deliberately frozen final evaluation. The independent chronological sensitivity
study is useful methodological evidence, but its within-user timestamp ordering
does not prevent all global-time leakage. No result supports promising a grade.

## Sources actually checked

- The assignment attachment, [Project-RecSys.pdf](reference/Project-RecSys.pdf),
  all 19 pages, plus [the assignment-entry record](reference/ASSIGNMENT.md).
- Live Brightspace MCP read of [W4S2-Hybrid](https://brightspace.tudelft.nl/d2l/le/content/844485/viewContent/5179119/View),
  20 pages, retrieved 2026-09-28 at 11:25 UTC.
- Live Brightspace MCP read of [W3S2-Fairness](https://brightspace.tudelft.nl/d2l/le/content/844485/viewContent/4732270/View),
  all 54 pages, retrieved 2026-09-28 at 11:26-11:27 UTC.
- `run.py`, `experiment.py`, `study.py`, `metrics.py`, tests, README, PLAN,
  HANDOFF and the checked-in development evidence.
- Instructor checkout at
  `/Users/vliftode/personal/recommender-systems/assignment-3/RecBole_DSAIT4335`:
  supplied model configurations and actual EASE, ItemKNN, FISM, SLIMElastic,
  Random and Pop source. The installed environment resolves RecBole to this
  checkout, verified by Python imports.

## Highest-priority findings

### 1. Weighted hybrid did not implement the lecture's sum-to-one equation

The assignment requires regression-learned weights (project page 5). The hybrid
lecture page 6 additionally shows `sum(w_k) = 1`. `study.py:127-135` fits
unconstrained ridge; `study.py:267-272` uses it as the static weighted hybrid.
Its weights need not satisfy the lecture equation. Context-dependent weights
are explicitly permitted by the lecture, but that does not remove the sum
constraint from the displayed formulation.

Fix implemented in this audit: [hybrid_constraints.py](hybrid_constraints.py)
provides `fit_sum_to_one_ridge(features, target, penalty) -> (weights, intercept)`.
It minimizes mean squared error plus L2 regularization with an unpenalized
intercept and the equality constraint. Five tests pass, including an analytical
solution, identical experts, a single expert and intercept shift invariance.
Integrate it as a clearly named baseline and tune its penalty; retain
unconstrained models as research variants. The lecture does not visibly impose
nonnegative weights, so this implementation is an affine weighted combination,
not a convex mixture.

### 2. No frozen hybrid inference or final report existed at the audited checkpoint

`study.py:44-80` only accepts validation exports; fitting and selection happen
inside its main program. `run.py:60` has individual-model test evaluation, but
there is no complete saved-hybrid inference contract. Coefficient JSON alone
does not contain every transformation and decision needed for replay: expert
ordering, score normalization rule, contextual standardization parameters,
activity thresholds, selected reranker/pool settings and history policy matter.

The current development results are suitable for choosing a method, not for
confirming that it is superior. `study.py:343-347` selects winners on the same
development cohort reported in the comparison, and `study.py:438-449` correctly
labels its intervals descriptive.

Before final evaluation: save a complete fitted pipeline with hashes and exact
choices, verify prediction round trips without training, freeze the comparison,
then run one controlled test evaluation. Do not recompute normalization or
selection rules from test outcomes. A final task-formatted report and runnable
submission package remain separate required deliverables (project pages 12-17).

### 3. User-side fairness was only a tentative policy, not an established result

`study.py:410-436` chooses exposure strength per activity group subject to a
95% fitted nDCG retention threshold. It fits that policy using the same meta-fit
labels used for the hybrid. Its base hybrid is itself selected on development
labels. The development evidence already shows retention below 95% for sparse
users. The implementation and HANDOFF acknowledge this correctly.

This can be reported as a failed protection hypothesis, but it cannot support
an accuracy guarantee or a claim that user-side fairness improved. Fit any
revised policy on a separate calibration cohort or through cross-fitting, then
report held-out utility gaps, worst-group quality and retention. The policy's
selection procedure must be fixed before its confirmatory evaluation.

The fairness lecture offers a direct complementary formulation: page 38 compares
group recommendation quality, and page 39 defines User Popularity Deviation as
distance between the popularity distributions in a user's history and list.
An individualized popularity-calibration reranker would explicitly address
different users' popularity preferences. A universal 20% head target instead
encodes an item-side catalog-parity choice.

### 4. Item-side exposure metric loses position and within-group concentration

`study.py:300,314` measures raw head-item share and distance from catalog share.
This is a defensible labelled diagnostic, but different rankings of the same
head/tail list receive identical exposure values. The fairness lecture page 43
weights positions by `1/log(1+position)` and divides group exposure by group size.
Page 42 also discusses Gini or entropy across item exposure counts.

Add position-discounted head/tail exposure and one whole-catalog concentration
metric. Keep the raw share to explain candidate quotas. Equal head/tail shares
can still concentrate nearly all exposure on a few items inside each group.

### 5. Do not import supplied FISM uncritically

The installed instructor `recbole/model/general_recommender/fism.py:75` uses
`BCEWithLogitsLoss`, but `inter_forward` at lines 137-139 already applies sigmoid
before lines 183-184 feed its result into that loss. Training therefore uses a
double-sigmoid objective. The positive target remains in the summed history at
lines 121-139 despite the class's stated exclusion of self-estimation.

If FISM is included, implement and identify a verified local adapter correcting
both issues, with a test showing that a positive target cannot contribute its
own source embedding. Do not silently modify the instructor checkout or claim
that its uncorrected results represent the intended FISM algorithm. The existing
project does not use FISM, so this does not invalidate the current three experts.

## Coverage that the sources support

| Requirement | Audited state and needed action |
|---|---|
| Task 1.1-1.2 individual models and tuning | EASE, ItemKNN and BPR are present with grids; Random/ExactPop provide naive baselines. The PDF does not enumerate an obligatory fixed model count and page 16 prioritizes depth. Add models for a hypothesis, not merely quantity. |
| Task 1.3 regression-weighted hybrid | Unconstrained ridge exists; add and evaluate the sum-to-one helper above for literal lecture alignment. |
| Task 1.4 other class hybrids | Switching is explicitly lecture page 7; the current activity-group switch qualifies. Cascade/reranking is explicitly lecture page 11. RRF is an extra rank-fusion technique, not needed as the sole evidence of lecture coverage. |
| Task 1.5 hybrid tuning | Ridge penalties are tuned; switching group design and RRF constant are fixed. State which choices are fixed and why. A small switching/cascade design comparison is more useful than silently calling every method tuned. |
| Task 2.1 independent metrics | Binary accuracy, novelty, coverage, diversity, genre calibration and group diagnostics are implemented outside RecBole. Add fairness measures above and preserve precise definitions. |
| Task 2.2 experiments versus baselines | Shared split/ID checks and exact popularity/random baselines are present. Current scores are selected development results; final frozen comparison remains. |
| Task 2.3 coefficients | Coefficients are saved. Interpret interactions and correlated-expert substitution; coefficient magnitude alone is not causal contribution. Compare stability and removal of an expert. |
| Task 2.4 insightful analysis | Feature/loss ablations, candidate feasibility and failed retention audit are useful. Exception-learning experiments should add matched-information controls and explicit falsification tests. |
| Task 2.5-2.6 groups and new hybrids | Training-activity groups and popularity item groups exist. Add training-only exception-history groups only after defining them independently of evaluation outcomes. |
| Task 3.1-3.2 rerankers and tradeoffs | Diversity, genre calibration and item exposure rerankers exist. User-side policy requires the careful treatment above. |
| Task 3.3 reranking order | Current comparison uses RRF, with full candidate support retained after promoting each expert's reranked top-k. Explain this exact operation; it is not equivalent to retraining a regression hybrid after score changes. |
| Task 3.4 group effects | Activity-group utility and conditional head/tail recall are reported. Include uncertainty and group sample counts; do not infer demographic fairness from activity groups. |
| Deliverables | Actual-contribution cover page, final report, runnable code ZIP named by group, and separate personal peer feedback are still required. |

## Protocol checks and research controls

1. **Current binarization matches instructor defaults.** The supplied configs use
   per-user random 80/10/10 splits, full-catalog ranking and no rating threshold.
   EASE and ItemKNN call `inter_matrix` without a rating field. Effective
   MovieLens Config loads user ID, item ID, rating and timestamp, but threshold is
   `None`. Merely loading a rating does not make these models rating-aware.
2. **UserKNN is an ItemKNN mode.** The instructor UserKNN configuration sets
   `model: ItemKNN` and `knn_method: user`; there is no need to invent a UserKNN
   class. Preserve the conceptual label in model manifests if adding it.
3. **Rating-aware experiments need matched information.** Preserve the original
   split assignment, then join ratings. Do not filter all ratings before splitting
   and compare against old runs. Compare exception learning with a simpler signed
   or rating-aware predictor given the same observed values. Define dislikes,
   neutral ratings and positives explicitly. A held-out low rating must not be
   counted as a liked exception, and unobserved items are not known dislikes.
4. **Exceptions must be defined out of sample.** A flexible taste model can fit a
   rating and then call its own residual unsurprising. Identify training
   exceptions with leave-one-out/cross-fitted predictions, or clearly justify a
   fixed structural definition independent of the candidate model. Do not use
   held-out ratings to decide a user's model input or group membership.
5. **BPR's grid confounds dimension and training budget.** `experiment.py:43`
   changes `(dimension, epochs)` together. Valid for configuration selection,
   invalid as a dimension effect explanation. Factorial or fixed-budget checks
   are needed before claiming one factor caused the performance difference.
6. **Adaptive candidate matching is empirical.** `study.py:393-396` ensures
   enough tail candidates but not enough head candidates. In a tail-dominated
   prefix, the full pool might permit a two-head/eight-tail list that the prefix
   cannot. Either require both quotas or keep the observed matching claim
   dataset-specific. All full-catalog scores have already been computed; smaller
   reranking pools do not establish end-to-end speedup.
7. **Random splitting is not a temporal deployment simulation.** The effective
   MovieLens configuration already loads timestamps. A per-user chronological
   sensitivity run is possible, but it still does not ensure a global time
   boundary. Label the actual split semantics accurately.
8. **Coefficient fitting and expert selection share meta-fit labels.** This is
   allowed as development, but can make stacked training features optimistic.
   Cross-fitting experts would be a stronger control if claiming a small gain.
   The independent final test is essential regardless.
9. **Clean installation remains unverified.** README acknowledges that the
   instructor checkout may contain coursework edits and there is no full
   lockfile. Source and environment hashes are valuable but not a replacement
   for a fresh installation and runnable-code check.

## Verification performed

The existing 18 unit tests passed using the actual course environment. The five
new constrained-regression tests passed independently. This audit used no test
labels, test scores, held-out model evaluation, Git mutation, external write or
course submission. The inspected metric equations and split/score safeguards
are internally consistent for their stated binary-ranking protocol; this is
not a statistical validation of any novelty or superiority claim.

## Categorical evidence-field review (2026-09-28)

The separate `categorical_field.py` and `categorical_experiment.py` pilot was
reviewed before real fitting. No mathematical or masking blocker was found.
Five rating categories enter as learned categorical representations; absent
ratings contribute neither source values nor upward attention. Only observed
context items send messages through the learned ports. The convex updates keep
states within [-1, 1], without establishing convergence or a physical fluid
interpretation. Unknown candidate states cannot become evidence for other
candidates.

The runner removes every probe category from its shared context, minimizes
cross-entropy only on masked TRAIN observations, gives each user equal loss
weight, and reuses the same initialization and identity-based masking schedule
across controls. It selects checkpoints with meta-fit validation users, saves
all three selections, then evaluates development users. Full TRAIN evidence is
the sole inference input. Rating-completion metrics and the fixed P(rating>=4)
ranking adapter measure different targets and must remain separately labeled.

The controls have limits that matter when explaining results:

- Adaptive versus fixed-flow jointly changes repeated routing and source-gate
  updates. It does not identify either component's individual effect.
- Hard clamping fixes observed states. Because only observed states enter the
  upward pool, upward attention and port values then remain constant, even
  though the code recomputes them. Candidate downward queries can still change.
  The source-gate parameters are inactive for predictions in this control.
- Equal allocated parameter counts do not imply equal active capacity. Gates
  and attention values do not identify trust, feelings, or causal explanations.
- The user halves for this pilot are 471 meta-fit / 472 development on the
  current 943-user source, whereas the main policy study separately reserves
  calibration users. Direct table comparisons must name their actual cohorts.

All 20 model and runner unit tests passed. Additional independent synthetic
checks verified joint catalog/embedding permutation equivariance, finite
backward gradients with empty contexts, exactly constant hard-clamp upward
attention and zero source-gate gradients, and exact checkpoint/probability
replay through two complete miniature training runs. Poisoning every
development rating label with NaN left training, meta-fit selection and saved
checkpoint/probability hashes unchanged. Selected-model aggregate dynamics
and comparisons with empty-context predictions were also reviewed; their
batch aggregation uses the appropriate observation/user denominators.

These checks used synthetic data and inspected source code only; no real test
split or test labels were opened. The earlier sections describe the project's
initial audit state and should not be read as a current completion checklist.

The separate `categorical_evaluation.py` helper also passed independent source
review and all five synthetic integration tests. Its freeze stage verifies
checkpoint minima, source and payload hashes, original split hashes, and exact
TRAIN-query probability replay. Final evaluation retains TRAIN-only evidence,
excludes TRAIN plus validation items from ranking, and creates its one-time
opening marker before reading labels. Bitwise replay assumes the same numerical
environment; carrying model state does not by itself guarantee that property
across dependency versions.

## Joint recorded-event objective review (2026-09-28)

The new `joint_field_experiment.py` uses the unchanged field architecture with
one softmax over eligible item/category outcomes. Padding and all context items
are excluded; all simultaneously masked probes remain eligible. Both training
and validation first average recorded-event losses within each user, then
average users. Validation uses the full TRAIN context only. The loss equals
conditional category NLL plus item-event NLL, each with coefficient one.
All-observed ranking marginalizes all five category logits; liked-record
ranking marginalizes only categories four and five. The latter describes a
liked recorded event, not merely a high rating conditional on a particular item.

Conditional category loss is unchanged by an arbitrary common shift to one
item's five logits. Joint normalization supplies a gradient for relative shifts
among eligible items while retaining invariance to a shared query-wide shift.
This output-level loss property does not make neural parameters, exposure, or
latent psychological meaning identifiable. Missing outcomes compete in the
normalizer and receive gradients without being explicitly labeled dislikes.
The observed data's recording and exposure biases remain.

The fixed count references use the same eligible set. The item/user category
tilt is deliberately not normalized within each item before joint normalization,
so it changes item-event mass as well as conditional category probabilities.
This distinction is documented in `JOINT_FIELD_PROTOCOL.md`.

All nine new unit tests passed independently before real joint fitting. Extra
synthetic checks verified unequal-probe macro averaging, exact NLL decomposition,
zero gradients for excluded context/padding logits, global-shift invariance,
the relative-item shift effect, and identical checkpoint/raw-logit hashes from
two complete miniature training runs with poisoned development labels. No
implementation blocker was found. No real test data was accessed.

The subsequent `joint_evaluation.py` freeze/final helper was independently
reviewed against the conditional evaluator. Six initial synthetic integration
tests passed, including an independently calculated count-model likelihood over
the TRAIN-plus-validation-excluded catalog. Model inference itself still sees
only TRAIN categories; a separate binary exclusion matrix controls evaluation
normalization and ranking. Both marginal ranking adapters use the frozen raw
logits, and conditional category probabilities remain per-item softmaxes.

Additional execution guards verified that all frozen inference finishes before
the test-opening marker, that the marker precedes synthetic label access, and
that code/runtime signature changes fail before either marker or output creation.
The audit identified a missing SciPy version in the runtime signature; the owner
added it to both evaluators and added direct signature-rejection tests. Raw-logit
replay alone would not have detected a changed SciPy metric postprocessor. No
running model source or real test split was modified or opened during this review.
After the runtime-signature fix, all 14 joint and conditional evaluator tests
passed together in an independent rerun.

## Bounded convergence follow-up review (2026-09-28)

The inspected v1 meta-fit curves attain their minima at the 100-epoch budget
boundary in both variants and both completed seeds. Every recorded checkpoint
improves over the previous one:

| Seed | Variant | Meta-fit joint NLL at 60 | At 100 | Change |
|---|---|---:|---:|---:|
| 2026 | adaptive | 7.117571 | 6.949402 | -0.168170 |
| 2026 | fixed_flow | 7.166711 | 6.982403 | -0.184309 |
| 2027 | adaptive | 7.101535 | 7.004070 | -0.097465 |
| 2027 | fixed_flow | 7.128961 | 6.981577 | -0.147384 |

Training probe losses also continue decreasing. This supports one bounded
convergence-sensitivity study; it neither establishes that 400 epochs is optimal
nor guarantees that the model has converged at that point. The approved follow-up
uses both variants, all three seeds, the same architecture and optimization
settings, and checkpoints 10/30/60/100/200/300/400, with no further budget extension.
It must remain labeled as an exploratory follow-up to the preserved v1 results.
The recommendation used training traces and meta-fit values, not development
outcomes or TEST.

Existing snapshots omit Adam optimizer state, so loading epoch-100 model weights
and starting a fresh Adam optimizer would change the trajectory. A faithful
longer-budget comparison should restart deterministically and verify the first
100 epochs against v1: initial state, episode hashes, training losses, and saved
checkpoint state/raw logits. New runner/protocol/output and evaluation provenance
must be explicit; changing imported global constants or bypassing source-set
checks is not an acceptable substitute for sealing the extension's source.

The separate `field_reference_comparison.py` review found correct endpoint
pairing, full observed-TRAIN exclusion, matching field cohorts, and matching fixed
label denominators. EASE/SLIM use their already selected all-observed scores;
PositiveEASE uses its recorded selection; conditional fields rank P(rating>=4),
while the joint field uses the appropriate five-category or liked-category
event marginal. Its description correctly limits this to a practical locked
comparison under unequal objectives and tuning budgets. The owner was asked to
add explicit meta-fit-selection validation and path-containment checks for field
manifest artifacts before sealing the final comparison source.

The implemented `joint_field_convergence.py` passed all five focused tests in
an independent rerun before launch. The tests execute the original 100-epoch
training loop on a synthetic catalog and reproduce its complete trajectory in
the new loop before extending to epoch 102; poisoned development labels leave
selection unchanged. Changed initial seeds, prior losses and prior checkpoint
states are rejected. The runner records the prior manifest before fitting and
revalidates both it and its artifacts at completion. Original module globals,
source files, protocol and output directories remain unchanged.

The separate convergence evaluator now verifies the capped protocol and runtime,
loads the preserved v1 selections, matches cohorts and original TRAIN categories,
compares prefix traces and meta-fit grids, and independently recomputes both old
and new checkpoint states/raw logits at epochs 10/30/60/100. Its portable bundle
contains the resulting verification records. This duplication keeps the
previously sealed v1 inference code intact while retaining the same safeguards.

The final `joint_convergence_evaluation.py` independently passed all four
synthetic integration tests, using an actual original-100-epoch run followed by
a fresh 400-epoch run. The tests reject resealed false prefix flags, missing
checkpoints, wrong origin hashes, changed traces/state digests and incompatible
runtimes. Frozen inference needs no original research directory; its selected
logits replay exactly, and the synthetic final stage excludes both TRAIN and
validation observations and refuses a second opening.

The separate `field_reference_evaluation.py` passed all seven tests in the same
independent run (11 evaluator tests total). It locks the first recorded meta-fit
maximum for EASE/SLIM and the liked-selected PositiveEASE choice; all three
producers must share the expected TEST hash as well as TRAIN/validation and data
signatures. Its saved score payloads remain usable after deleting synthetic
producer trees. Tests cover selection/family/seed/cohort errors, source/runtime/
payload tampering before the opening marker, TRAIN-plus-validation exclusion,
separate liked-user denominators, and marker persistence after failed label
access. No real TEST file was opened and no running source was changed.

## Fixed final comparison review (2026-09-28)

`compare_fields_final.py` passed all seven synthetic tests independently. The
comparison checks frozen/result/marker hash chains, aligns per-user outcomes
within each relevance endpoint, and exports aggregate results only. It uses the
declared conditional or joint marginal ranking readout and all 24 predeclared
contrasts per seed: three field tracks, four locked comparators, and two relevance
endpoints. Paired user bootstrap intervals share sampled users within an
endpoint, and the simultaneous intervals apply the full 24-comparison Bonferroni
correction. Seed averages and sample standard deviations are descriptive;
overlapping split seeds are not treated as independent population samples.

There is no direct joint-400 versus joint-100 contrast among those 24 tests.
Differences between their separate reference intervals cannot establish a
significant training-budget benefit. Before any held-out batch, the final batch
plan should pin this comparison source and `evidence/EVALUATION_PROTOCOL.md`;
recording the comparator's hash only after loading results would not by itself
prove that the analysis was fixed before TEST. This is an interpretation and
provenance requirement, not a requested expansion of the contrast family.

## Final batch launcher review (2026-09-28)

The independent review identified that a plan could originally name a different
Python interpreter from the one whose runtime had been checked. That could let
the primary evaluator open TEST before another family rejected an incompatible
runtime. The owner fixed this by requiring the requested interpreter's absolute
path to match `sys.executable`, without collapsing virtual-environment symlinks.
The launcher also checks raw dataset bytes against their frozen checksum before
reservation; this hashes bytes without parsing held-out labels.

All eight final launcher tests passed independently after these changes. They
cover the real cross-family signature collector with synthetic data, rejecting
ordered-ID, split, dataset and shared-source mismatches; exact eight-evaluator
and five-analysis job lists; at most three concurrent subprocesses; an exclusive
global reservation before every subprocess; all evaluation successes before
comparison execution; command/source/runtime changes before reservation; and
failed batches remaining reserved without retries. Existing outputs are refused.
The three primary evaluators rely on this global reservation for the batch's
single-use guarantee. No real plan was prepared or launched during the audit.

The final analysis list contains two fixed comparisons followed by one
`audit_study.py --aggregate-only` call for each primary seed. These commands use
the matching final/frozen directories and the atomic data directory, retain the
validation-selected reference, and derive taste groups from TRAIN ratings only.
The script and raw interaction/item metadata checksums are sealed before launch.
Its outputs summarize existing recommendations and metrics without a new fit,
test-based selection, or another TEST interaction read. The eight launcher tests
passed again after this bounded addition.

## Multi-seed final report review (2026-09-28)

The final report builder aligns experts by the model type in frozen source
metadata and hybrid families by their frozen roles, rejecting swapped or
incompatible roles. Tables use equal-weight means of each split's macro-user
metrics. Static coefficients are aligned by expert identity; an expert excluded
from fusion in every seed is omitted, while inconsistent inclusion is rejected.
All original split results and their manifest, result, freeze, code and source
artifact hashes remain in the report content. No model is chosen by TEST values.

The review identified a missing main-body presentation of Task 3.4: aggregate
reranker trade-offs and a policy-retention range did not show impacts on groups.
The owner added fixed context-to-policy nDCG comparisons for sparse/medium/dense
users and context-to-exposure-reranker recall for head/tail items. The report now
rejects unequal evaluated user counts, missing frozen source provenance, and
evaluation TEST hashes that differ from their freeze. Retention is a ratio of
group macro nDCGs within each split, with the range across split/group pairs;
it is not a ratio of overall split-averaged means.

All 21 report/package tests passed independently under the report environment,
including the full-size four-page layout, Times New Roman 12-point font, 1.15 line
spacing, per-task discussion limits, deterministic PDF output and isolated
supplemental pages. Additional synthetic arithmetic with deliberately unequal
split values verified canonical nDCG means, coefficient means, all group-effect
means, retention ratios and the new metadata-rejection paths. This clears the
aggregation implementation; interpretation of actual final outcomes remains a
separate review after the controlled evaluation. No real TEST data was accessed.

The subsequent strong-reference summary reads all 24 fixed contrasts per seed,
matches candidate means to the corresponding embedded field objective/readout,
and checks EASE/SLIM all-observed means against canonical primary model roles.
The review requested two refinements, both implemented: exact freeze/evaluation
manifest hashes now bind the paired intervals to the embedded field studies,
and the discussion describes approximate Bonferroni-adjusted bootstrap bounds
rather than guaranteed family coverage. The 100/400 labels explicitly describe
epoch caps, since checkpoint selection may choose an earlier epoch. Both focused
synthetic tests passed independently, including the complete seven-page layout.

## Completed held-out evidence review (2026-09-28)

After the root agent completed the single controlled evaluation batch, the
auditor checked every numeric cell in the three main task tables against all
three original completed result files, all canonical static coefficient means,
the copied aggregate evidence, and report artifact hashes. These checks passed.
The policy-retention range is 0.996524 to 1.009590, correctly displayed as
99.7% to 101.0%. The field appendices contain exact copies of the corresponding
completed aggregate data. All-observed comparisons include 943 users per split;
liked comparisons include 882/882/884 users with held-out likes.

The actual field results do not support superiority over the strong references.
Mean all-observed nDCG is 0.228817 under the 100-epoch cap and 0.267861 under the
400-epoch cap, versus EASE 0.321550 and SLIM 0.313408. Under the 400-epoch cap,
adaptive minus fixed-flow nDCG is -0.004151; every adjusted per-split interval
contains zero. Liked-record nDCG is 0.249571 versus liked-selected PositiveEASE
0.294391. This supports reporting objective/budget sensitivity without claiming
an established adaptive-routing gain. The direct 100-versus-400 contrast was not
among the fixed interval tests.

Required hybrid results remain separate: calibrated sum-to-one nDCG is 0.333901,
unconstrained static 0.335637, and the disagreement family 0.337012. The original
unscaled sum-to-one control is 0.267679. These are frozen-family comparisons,
not authority to choose a new TEST winner. The report accurately preserves the
custom model's negative comparisons and overlapping-split limitations. Visual
review of actual task pages and the combined field appendix found no clipping,
overlaps or unreadable tables at the required font and spacing. Final editorial
feedback requested explicit hybrid definitions and concrete scale/reranking
trade-offs in the discussion, rather than generic statements alone.

The separate `curate_final_evidence.py` passed all three focused tests and an
independent check of the six actual evidence directories: 23 indexed payloads
plus six hash-index files, with aggregate-only schema checks and byte-identical
copies of all ten original aggregate/evaluation-manifest files. The curator
verifies the complete 13-job batch, exports the primary fixed aggregate allowlist,
and performs no fit or selection. The auditor opened completed derived metrics
and metadata only, never a raw TEST split or held-out ratings source.

The final editorial revision now defines the user/item/disagreement/context
features, pairwise fitting, reciprocal-rank fusion and group switching. Its
reported +6.6% difference uses the separately validation-selected standalone
reference (mean 0.316133), rather than implying a comparison against fixed EASE
or a newly selected TEST winner. The response-alignment and societal trade-offs
match the completed evidence, and the field appendix states both endpoint
denominators explicitly. The revised task discussions contain 94/122/113 words;
all three appendices also remain below 200 discussion words. Independent visual
checks of the changed pages found no clipping or overlap. All report artifact
hashes match the manifest. The reviewed PDF SHA-256 is
`08888a2ce58b6fe9ae8200c785e185bdaa0acba247c26f8e719123a8759d2fe0`.
No numerical, interpretation or layout blocker remains in this revision. The
report appropriately remains DRAFT until the real contributor metadata and
required individual peer-feedback work are complete; this review does not
constitute an external submission.
