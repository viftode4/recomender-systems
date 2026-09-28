# Research after the completed assignment study

Start with [the research design](DESIGN.md). This work investigates new models
and structural learning after the original final test was already opened.
MovieLens100K development results here are exploratory. The original report,
code freeze and review archive remain unchanged.

- [Diagnosis of the earlier field](field-diagnosis-v1/FINDINGS.md): compact
  useful predictions exist, but the previous field had not learned them or
  reached convergence. The measurements separate evidence from hypotheses.
- [Direct categorical evidence model](addressed_evidence/README.md): independently
  implemented additive and distinct-source interaction variants, trained from
  scratch. [Protocol](addressed_evidence/PROTOCOL.md),
  [launch receipt](addressed_evidence/LAUNCH.json), and
  [run, progress and recovery instructions](addressed_evidence/RUN.md).
  The receipt records a launch, not a completed result. The runner writes its
  final comparison automatically after every seed finishes.
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

The current files establish implementations, diagnostics and a running research
process. Novelty and a real-data advantage remain questions to answer.
