# What the assignment fixes, and what we can design

Checked against all 19 pages of the downloaded
[Project-RecSys.pdf](../reference/Project-RecSys.pdf), September 28, 2026.
This corrects earlier statements that confused our adopted defaults with the
assignment's requirements. Existing results keep their original definitions.

| Required deliverable | Source pages | Current work |
|---|---|---|
| RecBole, MovieLens 100K, ranking, separately evaluated exported outputs | 4 | Existing pipeline and independent evaluation |
| Tune course recommenders, implement the missing content model | 5 | Existing individual models and recorded tuning |
| Regression-weighted hybrid and other class hybrids, with tuning | 5 | Existing regression, switching and fusion studies |
| Own accuracy/beyond-accuracy metrics; random/popularity comparisons | 6 | Existing metrics and baselines |
| Coefficient analysis, insightful behavior analysis, user/item groups, implications for better hybrids | 7–8 | Existing analyses plus new mechanism studies |
| Diversity, calibration, user/item fairness rerankers; trade-offs, ordering and group impacts | 9–10 | Existing reranking studies |
| PDF plus runnable code and instructions in a group-number ZIP; individual peer feedback | 12–18 | Local review draft; names/contributions and human review remain |

The PDF does not dictate an all-ratings-positive interpretation, rating threshold,
80/10/10 split, random order, cutoff 10, full-catalog candidates, three seeds,
our meta/development cohorts, or sum-to-one regression weights. Some are course
code defaults and some are our experimental choices. Any changed protocol needs
an explicit rationale and matched baselines; it is not evidence of improvement
merely because its scores are numerically larger.

The report rules on pages 13–15 require a cover with group/member information,
at most one page and 200 discussion words per task, Times New Roman 12pt and
spacing 1.15; appendices follow the stated criteria. Essential results belong in
the main body. Page 16 prioritizes depth of analysis over model or metric counts.
The interpretation of "task" as each of the three main tasks remains an item
for lecturer clarification; it must not be presented as an independently
confirmed formatting ruling.

The five seats in [PLAN.md](../PLAN.md) are proposed ownership: protocol and
integration; individual models; hybrids and research; metrics and reranking;
report and reproducibility. They are not assertions about actual contributions.

## The new research decision

The course's methods are comparison points. A standalone custom model can be an
additional expert and scientific contribution without replacing required hybrid
or societal analyses. There is no requirement to claim an unobserved win.

The immediate custom experiment is [shared interpretation of donor evidence](evidence_transfer/DESIGN.md).
It tests whether support-pattern structure transfers useful ranking information
across items. The existing random all-recorded evaluation is retained for the
first controlled comparison. Rating-aware and chronological evaluations remain
valid separately declared extensions; we have not proved their outcomes by
running this experiment.

The existing report is a coursework review draft, not fulfillment of the user's
broader request for a substantial original advance. That research aim remains
open until a mechanism earns it through strong comparisons and untouched-data
confirmation. The four missing names and retrospective peer assessments remain
deferred as requested; none will be invented.
