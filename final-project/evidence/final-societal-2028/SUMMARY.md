# Training-defined societal audit

Descriptive audits of existing predictions; groups and calibration targets use training ratings only. Genre contradiction is a proxy, not emotional understanding.

All metrics below reuse the existing evaluation cohort. No test interactions or held-out rating values were opened.

| Training group definition | Membership counts across all training users |
|---|---|
| activity | active: 316, medium: 314, sparse: 313 |
| taste_breadth | broad: 314, insufficient-evidence: 3, medium: 313, narrow: 313 |
| contradiction | has-contradictory-likes: 279, insufficient-evidence: 207, no-contradictory-likes: 457 |

| Model | Worst activity nDCG | Worst contradiction-group nDCG | All-history JSD | Liked-history JSD |
|---|---:|---:|---:|---:|
| 2028-BPR-2 | 0.2215 | 0.2243 | 0.1526 | 0.1788 |
| 2028-EASE-1 | 0.2401 | 0.2312 | 0.1448 | 0.1714 |
| 2028-ExactPop-0 | 0.0880 | 0.0941 | 0.2045 | 0.2322 |
| 2028-FISMCorrected-0 | 0.2036 | 0.1966 | 0.1515 | 0.1786 |
| 2028-GenreContent-1 | 0.0131 | 0.0083 | 0.2502 | 0.2535 |
| 2028-ItemKNN-2 | 0.1978 | 0.2036 | 0.1613 | 0.1886 |
| 2028-LightGCN-200 | 0.2198 | 0.2141 | 0.1597 | 0.1844 |
| 2028-NGCF-1 | 0.1955 | 0.1880 | 0.1687 | 0.1945 |
| 2028-NeuMF-0 | 0.1871 | 0.1879 | 0.1692 | 0.1945 |
| 2028-Random-0 | 0.0047 | 0.0062 | 0.2407 | 0.2630 |
| 2028-SLIMElastic-1 | 0.2372 | 0.2213 | 0.1416 | 0.1692 |
| 2028-SLIMElastic-1/adaptive-pool-exposure | 0.1247 | 0.1112 | 0.1599 | 0.1835 |
| 2028-SLIMElastic-1/calibration-0.2 | 0.2382 | 0.2209 | 0.1372 | 0.1650 |
| 2028-SLIMElastic-1/calibration-0.5 | 0.2407 | 0.2214 | 0.1249 | 0.1528 |
| 2028-SLIMElastic-1/calibration-0.8 | 0.2169 | 0.1958 | 0.0928 | 0.1212 |
| 2028-SLIMElastic-1/diversity-0.2 | 0.2366 | 0.2180 | 0.1291 | 0.1591 |
| 2028-SLIMElastic-1/diversity-0.5 | 0.2196 | 0.1981 | 0.1453 | 0.1794 |
| 2028-SLIMElastic-1/diversity-0.8 | 0.1835 | 0.1718 | 0.1848 | 0.2222 |
| 2028-SLIMElastic-1/exposure-0.2 | 0.2386 | 0.2210 | 0.1414 | 0.1688 |
| 2028-SLIMElastic-1/exposure-0.5 | 0.2250 | 0.2040 | 0.1413 | 0.1683 |
| 2028-SLIMElastic-1/exposure-0.8 | 0.1646 | 0.1519 | 0.1487 | 0.1727 |
| 2028-SLIMElastic-1/full-pool-exposure | 0.1247 | 0.1112 | 0.1599 | 0.1835 |
| 2028-SLIMElastic-1/popularity_calibration-0.2 | 0.2361 | 0.2197 | 0.1414 | 0.1691 |
| 2028-SLIMElastic-1/popularity_calibration-0.5 | 0.2366 | 0.2219 | 0.1415 | 0.1690 |
| 2028-SLIMElastic-1/popularity_calibration-0.8 | 0.2316 | 0.2138 | 0.1413 | 0.1687 |
| 2028-UserKNN-1 | 0.2051 | 0.1865 | 0.1748 | 0.2015 |
| calibrated-ridge-0.001 | 0.2508 | 0.2391 | 0.1425 | 0.1639 |
| constrained-ridge-1 | 0.2003 | 0.1923 | 0.1355 | 0.1500 |
| context-pairwise-1 | 0.2188 | 0.2216 | 0.1447 | 0.1685 |
| context-ridge-0.1 | 0.2516 | 0.2458 | 0.1396 | 0.1619 |
| context-ridge-0.1/adaptive-pool-exposure | 0.1395 | 0.1415 | 0.1599 | 0.1776 |
| context-ridge-0.1/calibration-0.2 | 0.2502 | 0.2433 | 0.1344 | 0.1567 |
| context-ridge-0.1/calibration-0.5 | 0.2438 | 0.2318 | 0.1234 | 0.1468 |
| context-ridge-0.1/calibration-0.8 | 0.2184 | 0.2053 | 0.0919 | 0.1179 |
| context-ridge-0.1/diversity-0.2 | 0.2485 | 0.2394 | 0.1244 | 0.1505 |
| context-ridge-0.1/diversity-0.5 | 0.2391 | 0.2220 | 0.1288 | 0.1601 |
| context-ridge-0.1/diversity-0.8 | 0.2081 | 0.1886 | 0.1682 | 0.2045 |
| context-ridge-0.1/exposure-0.2 | 0.2541 | 0.2470 | 0.1397 | 0.1619 |
| context-ridge-0.1/exposure-0.5 | 0.2511 | 0.2403 | 0.1396 | 0.1610 |
| context-ridge-0.1/exposure-0.8 | 0.1998 | 0.1983 | 0.1437 | 0.1636 |
| context-ridge-0.1/full-pool-exposure | 0.1395 | 0.1415 | 0.1599 | 0.1776 |
| context-ridge-0.1/popularity_calibration-0.2 | 0.2528 | 0.2457 | 0.1396 | 0.1618 |
| context-ridge-0.1/popularity_calibration-0.5 | 0.2564 | 0.2484 | 0.1401 | 0.1623 |
| context-ridge-0.1/popularity_calibration-0.8 | 0.2582 | 0.2489 | 0.1386 | 0.1603 |
| contrast-all_observed | 0.0733 | 0.1012 | 0.1990 | 0.2203 |
| disagreement-ridge-0.1 | 0.2540 | 0.2442 | 0.1385 | 0.1593 |
| group-switch | 0.2401 | 0.2312 | 0.1448 | 0.1714 |
| group-utility-budget-exposure | 0.2541 | 0.2470 | 0.1397 | 0.1619 |
| item-ridge-0.01 | 0.2567 | 0.2404 | 0.1382 | 0.1597 |
| positive_ease-all_observed | 0.2169 | 0.2214 | 0.1584 | 0.1635 |
| rerank-experts-then-rrf/calibration | 0.2333 | 0.2279 | 0.1425 | 0.1636 |
| rerank-experts-then-rrf/diversity | 0.2310 | 0.2199 | 0.1419 | 0.1660 |
| rerank-experts-then-rrf/exposure | 0.2386 | 0.2328 | 0.1478 | 0.1685 |
| rerank-experts-then-rrf/popularity_calibration | 0.2352 | 0.2297 | 0.1475 | 0.1684 |
| rrf | 0.2341 | 0.2299 | 0.1479 | 0.1689 |
| rrf-then-rerank/calibration | 0.2252 | 0.2209 | 0.1322 | 0.1540 |
| rrf-then-rerank/diversity | 0.2246 | 0.2113 | 0.1325 | 0.1628 |
| rrf-then-rerank/exposure | 0.2312 | 0.2281 | 0.1473 | 0.1676 |
| rrf-then-rerank/popularity_calibration | 0.2357 | 0.2302 | 0.1478 | 0.1685 |
| signed_channels-all_observed | 0.2231 | 0.2215 | 0.1593 | 0.1664 |
| static-pairwise-1 | 0.2435 | 0.2373 | 0.1454 | 0.1709 |
| static-ridge-0.01 | 0.2578 | 0.2443 | 0.1387 | 0.1595 |
| user-ridge-0.1 | 0.2506 | 0.2435 | 0.1407 | 0.1618 |
