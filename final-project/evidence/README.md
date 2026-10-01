# Saved evidence

Start with the [results summary](../docs/RESULTS.md). These files contain public
aggregate results and verification records; local per-user predictions live in
ignored run directories.

| Evidence | What it supports |
| --- | --- |
| [final-primary-v3](final-primary-v3/RESULTS.md) | Original frozen-test model, group and reranking results; [all aggregates](final-primary-v3/aggregates.json) |
| [final-comparisons-v3](final-comparisons-v3/SUMMARY.md) | Paired hybrid comparisons and uncertainty |
| [Societal audit, seed 2026](final-societal-2026/SUMMARY.md) | Additional TRAIN-defined taste groups; matching audits exist for 2027 and 2028 |
| [Coursework completion](../coursework_completion/results-v1/aggregates.json) | Later mixed/meta-level/switching assessment on reused validation users |
| [Grouping study](../exploratory/framing_search/predictive/README.md) | Separate nested-TRAIN experiment and controls |

Keep comparisons within the same evaluation: users, candidate items, relevance
and available information must match. The completion and grouping studies are
separate from the original test table. `research-v3/` contains development-stage
summaries; its plots are not the frozen-test figures.

Where supplied, `SHA256.json` binds a study's saved files. Preserve those files;
write a new experiment or analysis to a new directory. Use the
[research index](../docs/RESEARCH_INDEX.md) for other evidence folders.
