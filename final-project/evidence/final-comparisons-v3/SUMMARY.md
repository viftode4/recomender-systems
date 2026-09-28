# Frozen paired comparisons

Descriptive paired percentile-bootstrap intervals conditional on the frozen models and this dataset; approximate simultaneous coverage is adjusted within each seed, not across seeds. Repeated splits share users and items, and one MovieLens dataset does not establish population-level or cross-dataset gains. No model or parameter is selected using these evaluation outcomes.

Candidate and reference models were specified in freeze manifests. Rows are not ordered by evaluation performance.

Primary metric: ndcg@10. Splits: 2026, 2027, 2028.

| Predeclared comparison | Mean candidate | Mean reference | Mean difference | Relative change |
|---|---:|---:|---:|---:|
| hybrid:calibrated | 0.33390 | 0.31613 | +0.01777 | +5.62% |
| hybrid:constrained | 0.26768 | 0.31613 | -0.04845 | -15.33% |
| hybrid:context | 0.33262 | 0.31613 | +0.01649 | +5.22% |
| hybrid:context-pairwise | 0.30998 | 0.31613 | -0.00615 | -1.95% |
| hybrid:disagreement | 0.33701 | 0.31613 | +0.02088 | +6.60% |
| hybrid:item | 0.33451 | 0.31613 | +0.01837 | +5.81% |
| hybrid:static | 0.33564 | 0.31613 | +0.01950 | +6.17% |
| hybrid:static-pairwise | 0.32900 | 0.31613 | +0.01286 | +4.07% |
| hybrid:user | 0.33405 | 0.31613 | +0.01791 | +5.67% |
| hybrid:rrf | 0.32049 | 0.31613 | +0.00436 | +1.38% |
| hybrid:group-switch | 0.32112 | 0.31613 | +0.00498 | +1.58% |
| policy:group-utility-budget-exposure | 0.33307 | 0.33262 | +0.00045 | +0.14% |

Per-seed paired intervals use confidence 95.0%, with Bonferroni adjustment across all declared comparisons within each seed.

| Split | Comparison | Candidate | Reference | Mean difference | Adjusted interval |
|---|---|---|---|---:|---|
| 2026 | hybrid:calibrated | calibrated-ridge-0.1 | 2026-EASE-1 | +0.01012 | [+0.00099, +0.01955] |
| 2026 | hybrid:constrained | constrained-ridge-1 | 2026-EASE-1 | -0.05139 | [-0.06793, -0.03517] |
| 2026 | hybrid:context | context-ridge-1 | 2026-EASE-1 | +0.01109 | [+0.00295, +0.01914] |
| 2026 | hybrid:context-pairwise | context-pairwise-1 | 2026-EASE-1 | -0.00391 | [-0.01699, +0.00958] |
| 2026 | hybrid:disagreement | disagreement-ridge-1 | 2026-EASE-1 | +0.01444 | [+0.00606, +0.02301] |
| 2026 | hybrid:item | item-ridge-1 | 2026-EASE-1 | +0.01331 | [+0.00551, +0.02111] |
| 2026 | hybrid:static | static-ridge-1 | 2026-EASE-1 | +0.01459 | [+0.00617, +0.02316] |
| 2026 | hybrid:static-pairwise | static-pairwise-1 | 2026-EASE-1 | +0.00849 | [-0.00143, +0.01806] |
| 2026 | hybrid:user | user-ridge-1 | 2026-EASE-1 | +0.01299 | [+0.00403, +0.02166] |
| 2026 | hybrid:rrf | rrf | 2026-EASE-1 | -0.00096 | [-0.01227, +0.01022] |
| 2026 | hybrid:group-switch | group-switch | 2026-EASE-1 | +0.00169 | [-0.00654, +0.01004] |
| 2026 | policy:group-utility-budget-exposure | group-utility-budget-exposure | context-ridge-1 | +0.00022 | [-0.00029, +0.00091] |
| 2027 | hybrid:calibrated | calibrated-ridge-0.01 | 2027-SLIMElastic-1 | +0.02186 | [+0.01098, +0.03264] |
| 2027 | hybrid:constrained | constrained-ridge-1 | 2027-SLIMElastic-1 | -0.04608 | [-0.06364, -0.02846] |
| 2027 | hybrid:context | context-ridge-0.001 | 2027-SLIMElastic-1 | +0.01807 | [+0.00665, +0.02915] |
| 2027 | hybrid:context-pairwise | context-pairwise-1 | 2027-SLIMElastic-1 | -0.01314 | [-0.02921, +0.00268] |
| 2027 | hybrid:disagreement | disagreement-ridge-0.1 | 2027-SLIMElastic-1 | +0.02274 | [+0.01325, +0.03279] |
| 2027 | hybrid:item | item-ridge-1 | 2027-SLIMElastic-1 | +0.01922 | [+0.00844, +0.03032] |
| 2027 | hybrid:static | static-ridge-1 | 2027-SLIMElastic-1 | +0.02030 | [+0.00888, +0.03186] |
| 2027 | hybrid:static-pairwise | static-pairwise-1 | 2027-SLIMElastic-1 | +0.01325 | [+0.00012, +0.02603] |
| 2027 | hybrid:user | user-ridge-0.1 | 2027-SLIMElastic-1 | +0.01893 | [+0.00812, +0.02968] |
| 2027 | hybrid:rrf | rrf | 2027-SLIMElastic-1 | +0.00649 | [-0.00819, +0.02116] |
| 2027 | hybrid:group-switch | group-switch | 2027-SLIMElastic-1 | +0.00531 | [-0.00380, +0.01441] |
| 2027 | policy:group-utility-budget-exposure | group-utility-budget-exposure | context-ridge-0.001 | +0.00005 | [-0.00163, +0.00189] |
| 2028 | hybrid:calibrated | calibrated-ridge-0.001 | 2028-SLIMElastic-1 | +0.02133 | [+0.01193, +0.03068] |
| 2028 | hybrid:constrained | constrained-ridge-1 | 2028-SLIMElastic-1 | -0.04789 | [-0.06593, -0.02996] |
| 2028 | hybrid:context | context-ridge-0.1 | 2028-SLIMElastic-1 | +0.02031 | [+0.00967, +0.03062] |
| 2028 | hybrid:context-pairwise | context-pairwise-1 | 2028-SLIMElastic-1 | -0.00140 | [-0.01691, +0.01390] |
| 2028 | hybrid:disagreement | disagreement-ridge-0.1 | 2028-SLIMElastic-1 | +0.02546 | [+0.01591, +0.03489] |
| 2028 | hybrid:item | item-ridge-0.01 | 2028-SLIMElastic-1 | +0.02258 | [+0.01266, +0.03209] |
| 2028 | hybrid:static | static-ridge-0.01 | 2028-SLIMElastic-1 | +0.02363 | [+0.01443, +0.03258] |
| 2028 | hybrid:static-pairwise | static-pairwise-1 | 2028-SLIMElastic-1 | +0.01685 | [+0.00466, +0.02876] |
| 2028 | hybrid:user | user-ridge-0.1 | 2028-SLIMElastic-1 | +0.02182 | [+0.01076, +0.03244] |
| 2028 | hybrid:rrf | rrf | 2028-SLIMElastic-1 | +0.00755 | [-0.00663, +0.02187] |
| 2028 | hybrid:group-switch | group-switch | 2028-SLIMElastic-1 | +0.00795 | [-0.00506, +0.02123] |
| 2028 | policy:group-utility-budget-exposure | group-utility-budget-exposure | context-ridge-0.1 | +0.00109 | [-0.00025, +0.00286] |
