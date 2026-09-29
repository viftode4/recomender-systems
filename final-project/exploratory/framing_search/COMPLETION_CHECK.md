# Assignment completion check

Checked on 29 September 2026 against the actual downloaded
[19-page brief](../../reference/Project-RecSys.pdf), the existing
[seven-page report](../../reports/final-review-v1/report.pdf), its content and
manifest, and `packages/final-review-v1/24.zip`. This check performed no new
model fits or held-out evaluation. It inspected already published aggregate
report content, not raw held-out records.

The scientific assignment already has substantial coverage. The remaining
work is to finish a coherent deliverable, strengthen the group comparison,
and substantiate the new grouping claim at the level actually supported by
its evidence. Missing contributor information remains explicitly deferred;
it is not a reason to stop this work or invent contributions.

## Exact requirements and coverage

| Brief requirement | PDF pages | Existing coverage / completion action |
|---|---:|---|
| RecBole, MovieLens 100K, ranking; evaluate exported results separately | 4 | Implemented in `run.py`, `study.py`, independent metrics and frozen evaluation. |
| Individual course models and tuning | 5, Tasks 1.1–1.2 | Model table, recorded grids and adapter corrections are present. The PDF itself does not enumerate the full course model list; do not infer an additional mandatory algorithm from it. |
| Regression-weighted hybrid; other class hybrids; tune hybrids | 5, Tasks 1.3–1.5 | Constrained/calibrated/static/context regression, switching and RRF are present. Regression penalties are tuned; activity bins and the RRF offset are declared fixed choices. No cascade implementation was found. |
| Independently implement accuracy and beyond-accuracy metrics; compare best models with Random and Pop | 6, Tasks 2.1–2.2 | Independent metric code and report comparisons present; ExactPop/Random adaptations are disclosed. |
| Coefficients and insightful behavioral analysis | 7, Tasks 2.3–2.4 | Coefficients, response-alignment comparison, failed custom-model controls and group analysis present. The new TRAIN recording audit adds a concrete data-generating-process observation. |
| Accuracy and beyond-accuracy across user and item groups; implications for better hybrids | 8, Tasks 2.5–2.6 | Existing main table compared several models' accuracy but only context's group diversity/exposure. The new main content fixes this using existing aggregates. Task 2.6 asks what the analysis suggests; it does not require a guaranteed new best model. |
| Diversity, calibration and user/item fairness rerankers; trade-offs | 9, Tasks 3.1–3.2 | Implemented and reported with declared operational fairness definitions, independent utility policy and retained trade-offs. |
| Rerank individual experts before combining versus rerank the combination; user/item impacts | 10, Tasks 3.3–3.4 | RRF-order comparison and user utility/head-tail effects are present. |
| Group PDF plus runnable code in a ZIP named for the group; individual peer Excel | 12 | Existing `24.zip` is a local review archive. Individual Excel and actual member details remain deferred human inputs, not fabricated deliverables. |
| Cover, at most one page and 200 discussion words per task; appendix follows same criteria; self-contained main body | 13–14 | New plan: cover + three required pages + one grouping appendix. Preserve the older research report separately. |
| Times New Roman 12pt, line spacing 1.15; explain choices; depth over model count | 14–16 | Existing report and new main render satisfy recorded typography. Preserve matched comparisons and qualify grouping interpretation. |
| Clean runnable code and short instructions | 17 | Archive contains source, tests, dependency pins and reproduction instructions; improve archive-relative links and add exact grouping replay instructions. |
| Genuine individual peer assessment of contribution/responsibility/communication | 18 | Deferred; cannot be generated from fictional team activity. |

The PDF does **not** explicitly mandate all-ratings-positive relevance, a
rating threshold, random 80/10/10 splitting, cutoff 10, full-catalog candidates,
three seeds, or sum-to-one coefficients. Some are course defaults or lecture
choices; others are our protocol. Retain these choices for comparable results,
and distinguish them from the PDF's fixed constraints. A new endpoint is not an
improvement over the old one.

The PDF lists October 26, 23:59 as the submission deadline (p. 12) and October
7–8 project feedback (p. 2). These are the downloaded brief's dates, not a fresh
Brightspace verification; the parent reports the current connector returned 403.

## Artifact verification performed now

- The existing PDF has seven A4 pages and embedded Times New Roman regular/bold.
  Its manifest records 12pt/1.15. Discussion counts are 94/122/113 words for
  Tasks 1–3 and 112/89/131 for the three research appendices.
- The ZIP has 225 entries and 2,913,268 bytes. CRC validation passed, and all
  224 manifest-indexed entry hashes matched. The archived report and report
  manifest exactly match the files under `reports/final-review-v1`.
- Every archived Python source/test matches the corresponding current file.
  No raw split, score array, model checkpoint or pickle payload is included.
- The earlier extracted-package test receipt records 216 discovered tests:
  211 passed and five rendering checks were separately covered by 23 passing
  report tests. This audit verified the receipt and archive integrity; it did
  not rerun that entire historical suite or claim a fresh network installation.

```text
Original report PDF SHA-256
08888a2ce58b6fe9ae8200c785e185bdaa0acba247c26f8e719123a8759d2fe0

Original 24.zip SHA-256
7e1d86708471b0d70a4be375ff7211f69a12505802d626e951f2a0fdbc4fd0bb
```

Seven pages are not prohibited by an explicit total-page limit. However, the
brief's appendix wording does not explicitly authorize multiple pages for the
same task. The earlier allocation of three separate Task 2 research appendices
is an interpretation, not a verified lecturer ruling. A five-page current
report with one Task 2 appendix is a conservative consolidation. Keeping the
older report in the archive preserves negative results without requiring all
of them to occupy the current main narrative.

## Concrete deliverable revision

1. Use the new
   [main-report-content.json](../../reports/framing-review-v1/main-report-content.json).
   It retains frozen main results and the coefficient table, and gives EASE,
   SLIM and context the same user-group nDCG/diversity and head/tail
   recall/exposure comparisons. Values are equal-weight means of the existing
   three per-seed aggregate records, aligned by their canonical model roles.
   Exact values and source hashes are retained in `editorial_revision`.
2. Render these four pages, then append one separately rendered grouping page.
   The original renderer does not support a grouping appendix schema; the new
   builder must not disguise it as a joint-field study or change sealed code.
   A temporary render of the new main source passed: discussion counts
   94/164/113, four pages, TNR 12pt/1.15, lowest content baseline 129.59pt.
3. The grouping appendix can already report the audited observation: 70.14% of
   seed-2026 TRAIN records share a timestamp with another record by that user;
   genre Jaccard is 0.234236 versus 0.184155 under the declared within-user null.
   The [independent review](recording_audit/REVIEW.md) reproduced all descriptive
   values and 100 null draws. This supports group structure, not improved ranking.
4. The separately declared [predictive study](predictive/PROTOCOL.md) is now
   complete: 39 candidate fits, with 65,518 fitting records, 7,645 development
   records and 7,645 assessment records nested entirely inside original TRAIN.
   Six arms use the same 943 users and full 1,682-item catalog. Its aggregate
   evidence reports original validation/test unread and no fresh population test.
   True-group nDCG is **0.172555**, versus **0.172956** for tuned EASE,
   **0.172721** for the bag-pair model and **0.172819** for the arithmetic mean
   of three separately fitted shuffle metrics. Every reported descriptive
   nDCG difference interval includes zero. The declared 10% improvement and
   mechanism-support checks both fail. This tested use of the partition does
   not improve ranking; it does not invalidate the measured genre coherence
   or prove every possible grouping model useless. The appendix must report
   the negative result and its distinct nested-TRAIN evaluation scope.
5. Build a **new** `24.zip` preserving the original report, its provenance and
   negative evidence. Include the new grouping code/protocol/tests, aggregate
   evidence, independent review and exact commands. Map packaged paths explicitly:
   executable code is under `code/`; report/evidence are at the archive root.
   Existing checkout-relative links in `code/README.md` are not all valid inside
   the old ZIP. The new builder is preparing the revised artifact; final archive,
   layout, numerical-table and extraction verification remains pending at this
   document's binding point.

The completed predictive aggregate SHA-256 is
`89f1d14696cc3d524eef60299de96a6ef753df7cc3c1a041897892bdfef9b556`;
its protocol JSON is
`eceadbacbfd16256fbc45dbcd13f65202724426399717486c486876890125d71`.
These bind the newly reported result; they do not represent a new evaluation
performed by this completion audit. Once the package binds this document, record
final verification in a separate package receipt rather than altering its bytes.

The unsupported phrase “switching and cascade” in `PLAN.md` was corrected to
the implemented switching/RRF/contextual/pairwise families and their actual
tuning scope. No model was added to make the old wording appear true.

Useful verification commands, from `final-project`:

```sh
pdftotext -layout reference/Project-RecSys.pdf -
pdfinfo reports/final-review-v1/report.pdf
pdffonts reports/final-review-v1/report.pdf
runs/environment-check/.venv/bin/python -m unittest \
  exploratory.framing_search.recording_audit.test_audit -v
```

Final packaging/submission checks remain separate from scientific claims.
Contributor metadata and genuine peer work stay deferred as requested. They do
not prevent completing, testing and reviewing the revised code/report now.
No external submission, connector mutation or model evaluation was performed
by this audit.
