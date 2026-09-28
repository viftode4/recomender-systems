# Training-defined societal audit

Descriptive audits of existing predictions; groups and calibration targets use training ratings only. Genre contradiction is a proxy, not emotional understanding.

All metrics below reuse the existing evaluation cohort. No test interactions or held-out rating values were opened.

| Training group definition | Membership counts across all training users |
|---|---|
| activity | active: 316, medium: 314, sparse: 313 |
| taste_breadth | broad: 314, insufficient-evidence: 3, medium: 313, narrow: 313 |
| contradiction | has-contradictory-likes: 284, insufficient-evidence: 188, no-contradictory-likes: 471 |

| Model | Worst activity nDCG | Worst contradiction-group nDCG | All-history JSD | Liked-history JSD |
|---|---:|---:|---:|---:|
| 2027-BPR-1 | 0.2136 | 0.1874 | 0.1635 | 0.1909 |
| 2027-EASE-1 | 0.2542 | 0.2342 | 0.1439 | 0.1700 |
| 2027-ExactPop-0 | 0.1037 | 0.0916 | 0.1988 | 0.2267 |
| 2027-FISMCorrected-0 | 0.2241 | 0.1890 | 0.1532 | 0.1776 |
| 2027-GenreContent-1 | 0.0112 | 0.0096 | 0.2515 | 0.2551 |
| 2027-ItemKNN-2 | 0.2182 | 0.1943 | 0.1632 | 0.1892 |
| 2027-LightGCN-200 | 0.2331 | 0.1979 | 0.1577 | 0.1814 |
| 2027-NGCF-1 | 0.2223 | 0.1904 | 0.1647 | 0.1909 |
| 2027-NeuMF-0 | 0.1989 | 0.1766 | 0.1652 | 0.1886 |
| 2027-Random-0 | 0.0039 | 0.0040 | 0.2447 | 0.2654 |
| 2027-SLIMElastic-1 | 0.2500 | 0.2302 | 0.1399 | 0.1667 |
| 2027-SLIMElastic-1/adaptive-pool-exposure | 0.1180 | 0.1132 | 0.1593 | 0.1833 |
| 2027-SLIMElastic-1/calibration-0.2 | 0.2497 | 0.2315 | 0.1352 | 0.1622 |
| 2027-SLIMElastic-1/calibration-0.5 | 0.2503 | 0.2276 | 0.1248 | 0.1524 |
| 2027-SLIMElastic-1/calibration-0.8 | 0.2285 | 0.2116 | 0.0943 | 0.1229 |
| 2027-SLIMElastic-1/diversity-0.2 | 0.2485 | 0.2302 | 0.1291 | 0.1584 |
| 2027-SLIMElastic-1/diversity-0.5 | 0.2301 | 0.2127 | 0.1450 | 0.1797 |
| 2027-SLIMElastic-1/diversity-0.8 | 0.1951 | 0.1849 | 0.1825 | 0.2199 |
| 2027-SLIMElastic-1/exposure-0.2 | 0.2514 | 0.2300 | 0.1400 | 0.1665 |
| 2027-SLIMElastic-1/exposure-0.5 | 0.2347 | 0.2112 | 0.1406 | 0.1667 |
| 2027-SLIMElastic-1/exposure-0.8 | 0.1655 | 0.1544 | 0.1485 | 0.1734 |
| 2027-SLIMElastic-1/full-pool-exposure | 0.1180 | 0.1132 | 0.1593 | 0.1833 |
| 2027-SLIMElastic-1/popularity_calibration-0.2 | 0.2499 | 0.2300 | 0.1402 | 0.1667 |
| 2027-SLIMElastic-1/popularity_calibration-0.5 | 0.2502 | 0.2275 | 0.1398 | 0.1662 |
| 2027-SLIMElastic-1/popularity_calibration-0.8 | 0.2422 | 0.2212 | 0.1402 | 0.1658 |
| 2027-UserKNN-0 | 0.2275 | 0.1966 | 0.1702 | 0.1963 |
| calibrated-ridge-0.01 | 0.2639 | 0.2434 | 0.1470 | 0.1687 |
| constrained-ridge-1 | 0.2050 | 0.1806 | 0.1355 | 0.1485 |
| context-pairwise-1 | 0.2153 | 0.1956 | 0.1412 | 0.1631 |
| context-ridge-0.001 | 0.2627 | 0.2262 | 0.1400 | 0.1641 |
| context-ridge-0.001/adaptive-pool-exposure | 0.1247 | 0.1216 | 0.1638 | 0.1811 |
| context-ridge-0.001/calibration-0.2 | 0.2606 | 0.2287 | 0.1345 | 0.1591 |
| context-ridge-0.001/calibration-0.5 | 0.2543 | 0.2214 | 0.1222 | 0.1469 |
| context-ridge-0.001/calibration-0.8 | 0.2349 | 0.2061 | 0.0915 | 0.1176 |
| context-ridge-0.001/diversity-0.2 | 0.2598 | 0.2247 | 0.1241 | 0.1512 |
| context-ridge-0.001/diversity-0.5 | 0.2418 | 0.2132 | 0.1292 | 0.1613 |
| context-ridge-0.001/diversity-0.8 | 0.1954 | 0.1790 | 0.1681 | 0.2038 |
| context-ridge-0.001/exposure-0.2 | 0.2631 | 0.2250 | 0.1399 | 0.1641 |
| context-ridge-0.001/exposure-0.5 | 0.2282 | 0.2012 | 0.1394 | 0.1626 |
| context-ridge-0.001/exposure-0.8 | 0.1808 | 0.1635 | 0.1471 | 0.1669 |
| context-ridge-0.001/full-pool-exposure | 0.1247 | 0.1216 | 0.1638 | 0.1811 |
| context-ridge-0.001/popularity_calibration-0.2 | 0.2619 | 0.2253 | 0.1398 | 0.1639 |
| context-ridge-0.001/popularity_calibration-0.5 | 0.2621 | 0.2259 | 0.1402 | 0.1644 |
| context-ridge-0.001/popularity_calibration-0.8 | 0.2541 | 0.2225 | 0.1397 | 0.1627 |
| contrast-all_observed | 0.0746 | 0.0961 | 0.1966 | 0.2148 |
| disagreement-ridge-0.1 | 0.2660 | 0.2423 | 0.1390 | 0.1607 |
| group-switch | 0.2542 | 0.2245 | 0.1414 | 0.1672 |
| group-utility-budget-exposure | 0.2631 | 0.2250 | 0.1399 | 0.1641 |
| item-ridge-1 | 0.2608 | 0.2366 | 0.1431 | 0.1659 |
| positive_ease-all_observed | 0.2175 | 0.1991 | 0.1554 | 0.1603 |
| rerank-experts-then-rrf/calibration | 0.2470 | 0.2193 | 0.1412 | 0.1627 |
| rerank-experts-then-rrf/diversity | 0.2413 | 0.2173 | 0.1427 | 0.1668 |
| rerank-experts-then-rrf/exposure | 0.2473 | 0.2174 | 0.1474 | 0.1683 |
| rerank-experts-then-rrf/popularity_calibration | 0.2462 | 0.2195 | 0.1478 | 0.1684 |
| rrf | 0.2460 | 0.2185 | 0.1479 | 0.1686 |
| rrf-then-rerank/calibration | 0.2415 | 0.2150 | 0.1307 | 0.1531 |
| rrf-then-rerank/diversity | 0.2342 | 0.2143 | 0.1325 | 0.1622 |
| rrf-then-rerank/exposure | 0.2278 | 0.1939 | 0.1471 | 0.1668 |
| rrf-then-rerank/popularity_calibration | 0.2463 | 0.2189 | 0.1473 | 0.1679 |
| signed_channels-all_observed | 0.2208 | 0.1988 | 0.1526 | 0.1598 |
| static-pairwise-1 | 0.2574 | 0.2290 | 0.1413 | 0.1652 |
| static-ridge-1 | 0.2623 | 0.2379 | 0.1438 | 0.1643 |
| user-ridge-0.1 | 0.2621 | 0.2300 | 0.1408 | 0.1638 |
