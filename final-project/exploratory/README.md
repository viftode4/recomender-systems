# Research after the completed assignment study

Start with [the research design](DESIGN.md). This work investigates new models
and structural learning after the original final test was already opened.
MovieLens100K development results here are exploratory. The original report,
code freeze and review archive remain unchanged.

- [Completed grouping prediction study](framing_search/predictive/README.md):
  a 39-fit nested TRAIN experiment tests the input distinction identified by the
  [framing investigation](framing_search/README.md). Preserving timestamp groups
  gives nDCG 0.172555 versus 0.172956 for newly fitted EASE and 0.172819 averaged
  over three shuffled controls. The descriptive grouping coherence does not
  establish a predictive advantage; the measured mechanism and substantial-gain
  criteria failed. The current review report includes this negative finding.
- [Assignment requirements versus our choices](ASSIGNMENT_AND_RESEARCH.md):
  the actual brief leaves more protocol and model freedom than previously stated.
- [Shared evidence interpreter](evidence_transfer/FINDINGS.md): a custom
  177-parameter predictor transfers a common rule across items. Pattern features
  improve nDCG by 2.73% over its matched control, but the full model remains
  6.30% below locked EASE. Nine trajectories completed; the substantial-gain
  research target was not met. The useful feature effect, optimization limits
  and established prior art are kept distinct.
- [Why the gains remain small](research_diagnosis/README.md): prediction overlap,
  error locations, item support, comparison with published evaluation protocols,
  and a controlled data-volume experiment. With fixed query histories, increasing
  donor users from 88 to 353 raised mean nDCG from 0.200030 to 0.237810. This
  reduced-data sensitivity is not a gain over the full-data baseline.
- [Completed independent pair-interaction study](context_interactions/FINDINGS.md):
  exact target-excluded unary and pair coefficients, 135 declared configurations,
  and an independent audit. Added interaction capacity under the same
  reconstruction objective produced no meaningful ranking improvement.
- [Completed categorical reconstruction study](categorical_reconstruction/FINDINGS.md):
  our exact constrained model, 243 candidate fits, shuffled-rating controls and
  matched hybrid/group analyses. Mean development nDCG rises from 0.261688 to
  0.263154 versus expanded binary EASE; MRR declines and hybrid gains are
  negligible. [Independent numerical audit](categorical_reconstruction/audit-v2/RESULTS.md)
  verifies predictions, hybrid fitting and ranking metrics.
- [Diagnosis of the earlier field](field-diagnosis-v1/FINDINGS.md): compact
  useful predictions exist, but the previous field had not learned them or
  reached convergence. The measurements separate evidence from hypotheses.
- [Direct categorical evidence model](addressed_evidence/README.md): independently
  implemented additive and distinct-source interaction variants, trained from
  scratch. [Protocol](addressed_evidence/PROTOCOL.md),
  [launch receipt](addressed_evidence/LAUNCH.json), and
  [run, progress and recovery instructions](addressed_evidence/RUN.md).
  [Completed results](addressed_evidence/results-v1/RESULTS.md): both variants
  remain substantially below the matched EASE reference; the pair extension
  worsens the primary joint likelihood. All six fits completed.
- [Optimization geometry](addressed_evidence/GEOMETRY.md): convex additive
  training, nonconvex pair interactions and exact representational restrictions,
  with a concrete future optimization experiment.
- [Structural program proof](structural_program/RESULTS.md): one selected
  expression, successful conjunction recovery and redundancy removal, plus
  an explicit XOR failure. These supplied synthetic atoms do not establish
  a real recommendation result.
- [Completed stress test](structural_program/stress-v2/RESULTS.md):
  all 90 declared task/size/seed cases, including small-sample overfitting,
  misleading atoms and misspecified grammar, with equal-count random search and
  signed linear references. [Standalone figure](structural_program/stress-v2/stress_nll_difference.pdf).
- [Supplementary dataset plan](DATASET_EXPANSION.md): MovieLens1M first, with
  a genuinely unopened assessment and a later distinct-domain option. The
  official archive has not been acquired in this restricted environment.
- [Deeper research map](deeper_research/RESEARCH_MAP.md): learning operation
  contents, their composition and which evidence to acquire; distinguishes
  implemented experiments from proposed extensions and known prior art.
- [Completed learned-operation query study](library_queries/README.md): exact
  inference, 170 training worlds, unseen-function transfer and a no-sharing
  control. Focused acquisition helps earlier in the shared fixture but exposes
  an assumption the current learner cannot repair in the no-sharing fixture.

These studies establish implementations, diagnostics and narrowly scoped
empirical findings. The categorical reconstruction gain is small and exploratory;
conceptual novelty and a broad real-data advantage remain unestablished.
