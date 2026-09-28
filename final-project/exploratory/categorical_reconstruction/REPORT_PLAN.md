# Categorical reconstruction: report and reproduction plan

Status: planning only, before the new study's results. This file specifies the
deliverable; it contains no performance claims. Read the eventual study protocol
for the model, comparisons and selection rule. The report must describe what
was actually run, including unsuccessful variants and numerical failures.

## Scope and assignment fit

The original MovieLens test has already been inspected. This supplement is
**post-test exploratory research using development evidence**, not a new
confirmatory test or an amendment to the frozen experiment. Do not open TEST,
select another test winner, or mix development and held-out numbers in one
comparison. Reused splits are not independent datasets.

The assignment brief, pages 7–8, asks for the following:

| Requirement | Evidence the supplement should explain |
|---|---|
| Task 2.4: insightful analysis of a hybrid or its individual models | State the reconstruction objective and ranking score; compare the protocol's matched controls; connect an observed effect to the changed representation, constraints or objective. Include the numerical audit that supports the implementation. A higher ranking score alone does not explain why. |
| Task 2.5: accuracy and beyond-accuracy for user and item groups | Compare the new model with matched references within TRAIN-defined activity groups and head/tail item groups. Give denominators and definitions. Include a user-group beyond-accuracy measure and item exposure alongside group accuracy, if supplied by the declared evaluation. Never substitute one model's group table for a cross-model comparison. |
| Task 2.6: insights that could lead to a superior hybrid | State whether the measured differences suggest useful complementary behavior. A standalone improvement is not evidence that adding it to a hybrid helps. Report a hybrid improvement only if a separately declared, matched hybrid experiment actually measures it; otherwise mark the hybrid implication as a hypothesis. |

This supplement supports Task 2. It does not replace the required individual
models, regression hybrid, own metrics, naïve baselines or Task 3 reranker
analysis already covered by the main project. Existing linear reconstruction
and categorical-feedback work must receive attribution; an independent
implementation does not establish conceptual novelty.

## One-page research appendix

Produce a separate `appendix.pdf`, `appendix.md` and machine-readable content
file in a **new**, versioned report directory. Include group 24 and only supplied
contributor names. Missing names do not prevent a technical review draft.

The page should contain:

1. **Scope and method:** a short statement of the model, objective, candidate
   masking, selected hyperparameters, actual development cohort and endpoint.
   Identify the protocol and source/evidence hashes through a concise manifest
   reference. Mark the page “EXPLORATORY DEVELOPMENT — REVIEW DRAFT.”
2. **Task 2.4 result table:** the new model, declared ablations and matched strong
   references; primary all-observed nDCG@10 and one complementary metric.
   Show liked-only ranking separately when available because its relevance
   labels and eligible-user denominator differ. Retain negative results.
3. **Task 2.5 group comparison:** compact baseline/new-model contrasts for
   sparse/medium/dense users and head/tail items. Use explicit metric labels:
   user-group ranking accuracy and the supplied diversity/novelty measure;
   item-group recall and recommendation exposure. Put exhaustive group tables
   and every seed in the accompanying Markdown/evidence.
4. **Discussion, at most 200 words:** the measured mechanism finding, group
   trade-off, uncertainty or alternative explanation, and bounded Task 2.6
   implication. State plainly when the new model loses or the controls do not
   isolate the proposed explanation. Do not claim causal identification,
   superior hybrids, novelty priority, SOTA or a guaranteed grade.

Use Times New Roman 12pt, spacing 1.15, one content page and overflow refusal.
If the compact tables do not fit, reduce content rather than font size; retain
the essential comparison and group finding. Additional analysis belongs in the
Markdown companion or a separately labelled page subject to the same limits.
Any later proposal to integrate this into the assignment main body must keep
that body self-contained and preserve one page per major task.

`report.py` currently requires three main tasks and recognizes only its existing
appendix types. Do not fabricate task sections or change its validation to
render this standalone study. A new local renderer may reuse its verified font,
spacing, deterministic PDF metadata and overflow-checking approach. Record the
renderer and font hashes, page count and discussion word count. The renderer
should consume audited aggregate evidence, never fit a model or read raw ratings.

## Evidence required before writing results

The runner owner has specified the following planned handoff, not completed
results: `results-v1/{protocol.json,aggregates.json,provenance.json,RESULTS.md,SHA256.json}`.
Per-seed model roles are `binary_original_grid`, `binary_expanded`, `categorical`
and `shuffled_categories`, with selections, references, comparisons and cohorts.
Each model reports all-observed and liked-rating aggregates from the same
selected score, denominators, known-dislike rate and training-frequency
diagnostics. The scope fields are `stage: reused_development`, `test_read: false`
and `fresh_confirmation: false`.

The declared cohorts are 471 meta-fit and 472 development users per split.
EASE, SLIMElastic and PositiveEASE references come from the matched
`evidence/field-reference-v1` artifacts, verified against their original run
manifest. All three seeds' selections must be sealed before any new development
evaluation. Cite the actual protocol for grids, rather than treating unequal
search spaces as an isolated representation effect.

Before fitting, the project owner approved a pure group-analysis helper for
TRAIN-defined user/item groups and a controlled hybrid extension. The latter
compares expanded binary EASE plus SLIM against the same experts plus actual
categorical scores or plus shuffled-category scores. All use the same
affine-calibrated, sum-to-one ridge fitting procedure: select among four declared
penalties on an inner 235/236 split of meta-fit users, then refit on all 471.
Every seed's selection must be saved before development evaluation. Read the
final protocol and implementation audit for exact penalty values and tie rules.

This supplies a direct Task 2.6 experiment rather than relying on standalone
accuracy as evidence of complementarity. In the one-page result table, retain
the binary baseline, real/shuffled categorical comparison and three hybrid
conditions; keep exhaustive individual-model rows in the Markdown companion.
The group helper and hybrid evidence are approved but not yet measured. Do not
claim implementation clearance, subgroup findings or hybrid gains before their
actual audits/results. No values may be borrowed from old test results.

Do not infer missing numbers from earlier experiments. Accept the completed
artifacts only after independent review verifies:

- Source/protocol/runtime hashes, declared seed coverage, split fingerprints,
  actual cohort counts, label thresholds, masking and selection timing. The
  record must distinguish this study's lack of TEST access from the project's
  earlier TEST exposure.
- Independent implementation checks of reconstruction constraints and scoring,
  plus all declared model and reference rows. Any fit failure or fallback is
  retained with its effect on comparison coverage.
- Identical users, candidates, histories, relevance and metric definitions for
  each claimed comparison. Existing baseline scores are reusable only when
  those conditions and their provenance match. A 943-user test result cannot
  serve as a baseline for a smaller development cohort.
- Per-seed aggregates and group denominators; exact arithmetic of reported
  means. Paired intervals, if supplied, must use matched users and explain
  their scope. No fresh significance claim from reused validation or treating
  overlapping splits as independent samples.
- A source for every numerical statement. Store aggregate tables in the report
  content file and their verified artifact hashes in its manifest. Inspect all
  rendered pages for clipping, readability and visible scope labels.

The independent audit has confirmed the assignment mapping above. Scientific
clearance remains pending the actual model, run and evidence audits.

## Reproduction bundle and delivery

Create a new `24-research-review-v2.zip` after the measured appendix is ready,
in a new versioned package directory. It is a **review bundle**, not a
submission-ready replacement for the original `24.zip`. Include the
byte-identical original PDF as `report.pdf` and the new one-page appendix as
`research-supplement.pdf`. A short root README should identify the first as the
completed frozen study and the second as later exploratory development work.
The original self-contained main body remains intact. Do not merge new numbers
into its held-out tables or imply that the two studies share one evaluation.

The bundle also includes:

- The supplement's readable Markdown and aggregate content/manifest.
- The study protocol, actual model/runner/evaluator sources, focused tests and
  their required local imports, plus dependency versions or locks.
- A README with exact fitting, aggregate-verification and report commands,
  expected input layout and lawful dataset setup instructions. Identify any
  required RecBole adapter and explain how the custom experiment connects to
  the course pipeline. Record actual runtime/environment limits.
- Curated aggregate evidence, comparison/group tables, source fingerprints and
  verification receipt. Include a learning/diagnostic plot only when it answers
  a concrete question and comes from recorded evidence.

Use an explicit file whitelist. Exclude raw ratings, user histories, per-user
predictions, recommendation lists, checkpoints, credentials and the copyrighted
assignment PDF. Preserve hashes of excluded source artifacts for provenance
without packaging their contents. Verify ZIP paths, CRCs, declared hashes and
the extracted source imports/tests. Reproduction must not depend silently on
unpackaged files or absolute paths from this checkout.

Deliver the new appendix and its runnable research bundle as review artifacts.
Do not submit to Brightspace or replace the original group archive. The
original artifacts were verified unchanged while preparing this plan:

| Protected artifact | SHA-256 |
|---|---|
| `reports/final-review-v1/report.pdf` | `08888a2ce58b6fe9ae8200c785e185bdaa0acba247c26f8e719123a8759d2fe0` |
| `packages/final-review-v1/24.zip` | `7e1d86708471b0d70a4be375ff7211f69a12505802d626e951f2a0fdbc4fd0bb` |

Next handoff: model/runner owners provide completed aggregate paths; the
independent reviewer checks them; the report owner renders the actual findings
and verifies the new bundle. Until then, this plan is the only report artifact
for the new study.
