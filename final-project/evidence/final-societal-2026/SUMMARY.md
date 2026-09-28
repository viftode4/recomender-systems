# Training-defined societal audit

Descriptive audits of existing predictions; groups and calibration targets use training ratings only. Genre contradiction is a proxy, not emotional understanding.

All metrics below reuse the existing evaluation cohort. No test interactions or held-out rating values were opened.

| Training group definition | Membership counts across all training users |
|---|---|
| activity | active: 316, medium: 314, sparse: 313 |
| taste_breadth | broad: 314, insufficient-evidence: 2, medium: 313, narrow: 314 |
| contradiction | has-contradictory-likes: 278, insufficient-evidence: 201, no-contradictory-likes: 464 |

| Model | Worst activity nDCG | Worst contradiction-group nDCG | All-history JSD | Liked-history JSD |
|---|---:|---:|---:|---:|
| 2026-BPR-1 | 0.2263 | 0.2204 | 0.1640 | 0.1899 |
| 2026-EASE-1 | 0.2564 | 0.2415 | 0.1463 | 0.1727 |
| 2026-EASE-1/adaptive-pool-exposure | 0.1077 | 0.1097 | 0.1559 | 0.1754 |
| 2026-EASE-1/calibration-0.2 | 0.2584 | 0.2428 | 0.1426 | 0.1694 |
| 2026-EASE-1/calibration-0.5 | 0.2564 | 0.2428 | 0.1311 | 0.1577 |
| 2026-EASE-1/calibration-0.8 | 0.2377 | 0.2267 | 0.0965 | 0.1241 |
| 2026-EASE-1/diversity-0.2 | 0.2540 | 0.2389 | 0.1329 | 0.1616 |
| 2026-EASE-1/diversity-0.5 | 0.2279 | 0.2079 | 0.1417 | 0.1746 |
| 2026-EASE-1/diversity-0.8 | 0.1883 | 0.1730 | 0.1776 | 0.2135 |
| 2026-EASE-1/exposure-0.2 | 0.2559 | 0.2410 | 0.1462 | 0.1727 |
| 2026-EASE-1/exposure-0.5 | 0.2439 | 0.2302 | 0.1468 | 0.1725 |
| 2026-EASE-1/exposure-0.8 | 0.1604 | 0.1555 | 0.1465 | 0.1690 |
| 2026-EASE-1/full-pool-exposure | 0.1077 | 0.1097 | 0.1559 | 0.1754 |
| 2026-EASE-1/popularity_calibration-0.2 | 0.2561 | 0.2413 | 0.1464 | 0.1729 |
| 2026-EASE-1/popularity_calibration-0.5 | 0.2533 | 0.2386 | 0.1466 | 0.1728 |
| 2026-EASE-1/popularity_calibration-0.8 | 0.2425 | 0.2290 | 0.1461 | 0.1716 |
| 2026-ExactPop-0 | 0.1076 | 0.0941 | 0.2062 | 0.2338 |
| 2026-FISMCorrected-0 | 0.2184 | 0.2061 | 0.1533 | 0.1790 |
| 2026-GenreContent-1 | 0.0185 | 0.0182 | 0.2522 | 0.2567 |
| 2026-ItemKNN-2 | 0.2150 | 0.2054 | 0.1629 | 0.1885 |
| 2026-LightGCN-200 | 0.2392 | 0.2274 | 0.1610 | 0.1856 |
| 2026-NGCF-1 | 0.2283 | 0.2131 | 0.1650 | 0.1892 |
| 2026-NeuMF-0 | 0.2098 | 0.2070 | 0.1698 | 0.1951 |
| 2026-Random-0 | 0.0029 | 0.0006 | 0.2472 | 0.2669 |
| 2026-SLIMElastic-2 | 0.2466 | 0.2299 | 0.1435 | 0.1689 |
| 2026-UserKNN-0 | 0.2179 | 0.2147 | 0.1714 | 0.1974 |
| calibrated-ridge-0.1 | 0.2639 | 0.2474 | 0.1495 | 0.1701 |
| constrained-ridge-1 | 0.2024 | 0.2129 | 0.1400 | 0.1544 |
| context-pairwise-1 | 0.2544 | 0.2487 | 0.1427 | 0.1622 |
| context-ridge-1 | 0.2615 | 0.2446 | 0.1443 | 0.1667 |
| context-ridge-1/adaptive-pool-exposure | 0.1192 | 0.1215 | 0.1554 | 0.1711 |
| context-ridge-1/calibration-0.2 | 0.2605 | 0.2456 | 0.1389 | 0.1616 |
| context-ridge-1/calibration-0.5 | 0.2600 | 0.2427 | 0.1280 | 0.1513 |
| context-ridge-1/calibration-0.8 | 0.2436 | 0.2340 | 0.0951 | 0.1203 |
| context-ridge-1/diversity-0.2 | 0.2648 | 0.2466 | 0.1306 | 0.1567 |
| context-ridge-1/diversity-0.5 | 0.2437 | 0.2284 | 0.1339 | 0.1647 |
| context-ridge-1/diversity-0.8 | 0.1996 | 0.1885 | 0.1708 | 0.2055 |
| context-ridge-1/exposure-0.2 | 0.2623 | 0.2459 | 0.1440 | 0.1665 |
| context-ridge-1/exposure-0.5 | 0.2549 | 0.2435 | 0.1439 | 0.1658 |
| context-ridge-1/exposure-0.8 | 0.1835 | 0.1829 | 0.1417 | 0.1600 |
| context-ridge-1/full-pool-exposure | 0.1192 | 0.1215 | 0.1554 | 0.1711 |
| context-ridge-1/popularity_calibration-0.2 | 0.2619 | 0.2453 | 0.1440 | 0.1665 |
| context-ridge-1/popularity_calibration-0.5 | 0.2606 | 0.2441 | 0.1438 | 0.1663 |
| context-ridge-1/popularity_calibration-0.8 | 0.2546 | 0.2410 | 0.1438 | 0.1654 |
| contrast-all_observed | 0.0937 | 0.1261 | 0.1901 | 0.2078 |
| disagreement-ridge-1 | 0.2694 | 0.2512 | 0.1459 | 0.1661 |
| group-switch | 0.2593 | 0.2294 | 0.1448 | 0.1700 |
| group-utility-budget-exposure | 0.2623 | 0.2459 | 0.1441 | 0.1665 |
| item-ridge-1 | 0.2686 | 0.2506 | 0.1431 | 0.1652 |
| positive_ease-all_observed | 0.2282 | 0.2342 | 0.1586 | 0.1630 |
| rerank-experts-then-rrf/calibration | 0.2542 | 0.2361 | 0.1412 | 0.1622 |
| rerank-experts-then-rrf/diversity | 0.2470 | 0.2382 | 0.1441 | 0.1676 |
| rerank-experts-then-rrf/exposure | 0.2499 | 0.2376 | 0.1478 | 0.1679 |
| rerank-experts-then-rrf/popularity_calibration | 0.2522 | 0.2376 | 0.1477 | 0.1682 |
| rrf | 0.2529 | 0.2383 | 0.1483 | 0.1689 |
| rrf-then-rerank/calibration | 0.2437 | 0.2334 | 0.1317 | 0.1534 |
| rrf-then-rerank/diversity | 0.2265 | 0.2218 | 0.1353 | 0.1656 |
| rrf-then-rerank/exposure | 0.2224 | 0.2177 | 0.1466 | 0.1662 |
| rrf-then-rerank/popularity_calibration | 0.2508 | 0.2379 | 0.1475 | 0.1678 |
| signed_channels-all_observed | 0.2276 | 0.2268 | 0.1591 | 0.1659 |
| static-pairwise-1 | 0.2633 | 0.2444 | 0.1442 | 0.1685 |
| static-ridge-1 | 0.2696 | 0.2512 | 0.1457 | 0.1658 |
| user-ridge-1 | 0.2613 | 0.2513 | 0.1455 | 0.1650 |
