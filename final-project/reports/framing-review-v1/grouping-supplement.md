# Appendix: do rating-recording groups improve ranking?

REVIEW DRAFT | Exploratory nested TRAIN split. Original VALID/TEST not accessed in this experiment.

Motivation: 70.14% of original TRAIN ratings share a user's timestamp. Within-group genre Jaccard is 0.234 versus 0.184 across 100 within-user shuffles. This is descriptive evidence.

Known constrained ridge, singleton + pair features; own-target features excluded. F/D/A = 65,518/7,645/7,645 records; 943 users. All 39 candidates use F context; D selects, A assesses after sealing. All 1,682 items eligible except F observations; no refit or target timestamps.

## All-recorded assessment at ten (every selected arm)

| Model | lambda / beta | nDCG | Recall |
| --- | --- | --- | --- |
| Tuned EASE | 300 / 0 | 0.1730 | 0.2043 |
| Unordered pairs | 250 / 0.1 | 0.1727 | 0.2027 |
| Recording groups | 250 / 0.1 | 0.1726 | 0.2035 |
| Shuffle 1 | 250 / 0.1 | 0.1728 | 0.2031 |
| Shuffle 2 | 250 / 0.1 | 0.1730 | 0.2039 |
| Shuffle 3 | 250 / 0.1 | 0.1726 | 0.2035 |

## Recording groups minus control: paired nDCG

| Control | Mean change | 95% interval |
| --- | --- | --- |
| vs EASE | -0.0004 | [-0.0020, +0.0013] |
| vs unordered | -0.0002 | [-0.0014, +0.0012] |
| vs shuffle mean | -0.0003 | [-0.0012, +0.0006] |

Intervals: 2,000 paired-user resamples, descriptive and not multiplicity adjusted. Shuffle comparison averages metrics, never scores. Full subgroup/secondary metrics remain in aggregate evidence.

## Discussion

Recording groups did not meet the declared research-priority rule: at least 10% relative nDCG gain over tuned EASE and higher means than unordered pairs and shuffled groups. The measured EASE-relative change was -0.23%. Coherent recording groups alone do not establish predictive value. The unordered control tests the grouping restriction; timestamp shuffles retain each user's movies and group sizes. They do not identify exposure, interface design, or watching sessions. Pairwise linear recommendation is established; the contribution here is a controlled input-structure test. This single nested split follows earlier dataset exploration. Its assessment is separated from model selection, but is not fresh population confirmation or directly comparable with the preceding test tables.

Prior: Steck & Liang (2021), Negative Interactions for Improved Collaborative Filtering: Don’t go Deeper, go Higher. doi:10.1145/3460231.3474273. Protocol, source hashes and all candidates accompany this report.
