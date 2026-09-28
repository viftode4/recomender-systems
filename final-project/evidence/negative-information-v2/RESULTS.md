# Personalized rejection information audit

The primary comparison gives mean liked nDCG@10 0.1618 for positive-only, 0.1609 for plain dislike channels, and 0.1598 for surprise weights. The matched placement controls score 0.1618, and shuffled weights score 0.1609. Every plain-minus-placement-control interval includes zero, across both penalty choices and both inference modes. This audit does not demonstrate a stable benefit from the personalized placement of rejection labels under this estimator; it does not establish that dislikes lack useful information for every model.

Validation-only exploratory evidence under a protocol fixed before decoder fitting. The strict context-only comparison is primary; full-history prediction is a predeclared secondary check. The same decoder is used in both modes. No test labels or test metrics were opened.

The decoder predicts disjoint probe likes from context ratings. The positive teacher also fits context only. All original training items, including probe observations, are masked from validation recommendations. Likes mean actual ratings >=4; dislikes mean ratings <=2; rating 3 is neutral. Missing ratings are unknown.

The common-penalty comparison uses ridge 250. Equal-grid selection uses [50,250,1000] and only strict-context meta-fit liked nDCG@10. Each control branch selects one common penalty from the average of all five control replicates; no replicate is selected. Both inference modes reuse that choice.

## context: fixed penalty

| Branch | 2026 | 2027 | 2028 | Mean nDCG@10 | Known dislikes/slot |
|---|---:|---:|---:|---:|---:|
| Positive only | 0.1655 | 0.1789 | 0.1411 | 0.1618 | 0.57% |
| Plain dislike channels | 0.1703 | 0.1691 | 0.1433 | 0.1609 | 0.59% |
| Surprise-weighted dislikes | 0.1666 | 0.1698 | 0.1432 | 0.1598 | 0.63% |
| Permuted weights (mean of 5) | 0.1693 | 0.1699 | 0.1434 | 0.1609 | 0.61% |
| Placement null (mean of 5) | 0.1687 | 0.1729 | 0.1437 | 0.1618 | 0.59% |

Control replicate ranges (not confidence intervals):

| Control | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| Permuted weights (mean of 5) | 0.1670–0.1730 | 0.1691–0.1722 | 0.1421–0.1450 |
| Placement null (mean of 5) | 0.1662–0.1728 | 0.1701–0.1763 | 0.1419–0.1463 |

Paired liked nDCG@10 differences with descriptive user-bootstrap 95% intervals:

| Contrast | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| plain minus positive | +0.0048 [-0.0031, +0.0131] | -0.0098 [-0.0174, -0.0017] | +0.0022 [-0.0053, +0.0104] |
| surprise minus plain | -0.0038 [-0.0093, +0.0010] | +0.0007 [-0.0034, +0.0050] | -0.0001 [-0.0039, +0.0036] |
| surprise minus weight permutation | -0.0028 [-0.0074, +0.0016] | -0.0002 [-0.0045, +0.0042] | -0.0002 [-0.0037, +0.0033] |
| plain minus placement null | +0.0017 [-0.0039, +0.0074] | -0.0038 [-0.0094, +0.0017] | -0.0004 [-0.0062, +0.0057] |

## full_train: fixed penalty

| Branch | 2026 | 2027 | 2028 | Mean nDCG@10 | Known dislikes/slot |
|---|---:|---:|---:|---:|---:|
| Positive only | 0.1711 | 0.1869 | 0.1623 | 0.1734 | 0.59% |
| Plain dislike channels | 0.1712 | 0.1793 | 0.1607 | 0.1704 | 0.62% |
| Surprise-weighted dislikes | 0.1671 | 0.1810 | 0.1599 | 0.1693 | 0.65% |
| Permuted weights (mean of 5) | 0.1689 | 0.1796 | 0.1605 | 0.1697 | 0.64% |
| Placement null (mean of 5) | 0.1719 | 0.1842 | 0.1617 | 0.1726 | 0.64% |

Control replicate ranges (not confidence intervals):

| Control | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| Permuted weights (mean of 5) | 0.1663–0.1730 | 0.1789–0.1819 | 0.1595–0.1616 |
| Placement null (mean of 5) | 0.1686–0.1744 | 0.1810–0.1871 | 0.1589–0.1642 |

Paired liked nDCG@10 differences with descriptive user-bootstrap 95% intervals:

| Contrast | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| plain minus positive | +0.0000 [-0.0072, +0.0078] | -0.0076 [-0.0152, +0.0004] | -0.0016 [-0.0099, +0.0067] |
| surprise minus plain | -0.0041 [-0.0088, +0.0003] | +0.0017 [-0.0027, +0.0061] | -0.0008 [-0.0049, +0.0032] |
| surprise minus weight permutation | -0.0018 [-0.0061, +0.0024] | +0.0014 [-0.0025, +0.0052] | -0.0006 [-0.0044, +0.0030] |
| plain minus placement null | -0.0007 [-0.0062, +0.0050] | -0.0049 [-0.0109, +0.0014] | -0.0009 [-0.0078, +0.0063] |

## context: selected penalty

| Branch | 2026 | 2027 | 2028 | Mean nDCG@10 | Known dislikes/slot |
|---|---:|---:|---:|---:|---:|
| Positive only | 0.1828 | 0.1964 | 0.1646 | 0.1813 | 0.61% |
| Plain dislike channels | 0.1811 | 0.1895 | 0.1685 | 0.1797 | 0.65% |
| Surprise-weighted dislikes | 0.1804 | 0.1908 | 0.1681 | 0.1797 | 0.67% |
| Permuted weights (mean of 5) | 0.1813 | 0.1908 | 0.1679 | 0.1800 | 0.67% |
| Placement null (mean of 5) | 0.1809 | 0.1920 | 0.1656 | 0.1795 | 0.64% |

Control replicate ranges (not confidence intervals):

| Control | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| Permuted weights (mean of 5) | 0.1791–0.1844 | 0.1897–0.1922 | 0.1620–0.1721 |
| Placement null (mean of 5) | 0.1757–0.1836 | 0.1898–0.1936 | 0.1639–0.1675 |

Paired liked nDCG@10 differences with descriptive user-bootstrap 95% intervals:

| Contrast | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| plain minus positive | -0.0016 [-0.0074, +0.0041] | -0.0069 [-0.0132, -0.0009] | +0.0039 [-0.0030, +0.0107] |
| surprise minus plain | -0.0008 [-0.0043, +0.0028] | +0.0012 [-0.0030, +0.0054] | -0.0004 [-0.0046, +0.0035] |
| surprise minus weight permutation | -0.0009 [-0.0042, +0.0025] | -0.0001 [-0.0036, +0.0033] | +0.0002 [-0.0036, +0.0043] |
| plain minus placement null | +0.0003 [-0.0038, +0.0042] | -0.0025 [-0.0074, +0.0022] | +0.0029 [-0.0022, +0.0081] |

## full_train: selected penalty

| Branch | 2026 | 2027 | 2028 | Mean nDCG@10 | Known dislikes/slot |
|---|---:|---:|---:|---:|---:|
| Positive only | 0.1808 | 0.2002 | 0.1770 | 0.1860 | 0.67% |
| Plain dislike channels | 0.1844 | 0.1968 | 0.1817 | 0.1876 | 0.73% |
| Surprise-weighted dislikes | 0.1835 | 0.1977 | 0.1799 | 0.1870 | 0.76% |
| Permuted weights (mean of 5) | 0.1818 | 0.1974 | 0.1809 | 0.1867 | 0.74% |
| Placement null (mean of 5) | 0.1825 | 0.1982 | 0.1788 | 0.1865 | 0.73% |

Control replicate ranges (not confidence intervals):

| Control | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| Permuted weights (mean of 5) | 0.1803–0.1832 | 0.1966–0.1981 | 0.1793–0.1834 |
| Placement null (mean of 5) | 0.1803–0.1844 | 0.1966–0.2000 | 0.1776–0.1799 |

Paired liked nDCG@10 differences with descriptive user-bootstrap 95% intervals:

| Contrast | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| plain minus positive | +0.0035 [-0.0016, +0.0093] | -0.0035 [-0.0092, +0.0025] | +0.0047 [-0.0017, +0.0111] |
| surprise minus plain | -0.0008 [-0.0037, +0.0021] | +0.0009 [-0.0030, +0.0047] | -0.0018 [-0.0051, +0.0012] |
| surprise minus weight permutation | +0.0017 [-0.0013, +0.0050] | +0.0002 [-0.0026, +0.0030] | -0.0010 [-0.0043, +0.0025] |
| plain minus placement null | +0.0019 [-0.0023, +0.0064] | -0.0014 [-0.0057, +0.0035] | +0.0029 [-0.0020, +0.0081] |

## Control and label-access diagnostics

| Seed | Decoder fit / excluded users | Fitting users without probe likes | Context / probe observations | Context moved dislikes | Context acceptance |
|---|---:|---:|---:|---:|---:|
| 2026 | 942 / 1 | 13 | 64266 / 16542 | 43.6%–44.0% | 7.8%–7.9% |
| 2027 | 942 / 1 | 14 | 64266 / 16542 | 43.3%–44.2% | 7.9%–8.0% |
| 2028 | 942 / 1 | 10 | 64266 / 16542 | 43.4%–44.1% | 7.9%–8.0% |

All exact support, positive-label, per-user and per-item dislike margins passed, including item margins within the context-defined decoder fitting stratum. Full-history null queries retain the exact switched context and add independently switched probe labels. Complete aggregate diagnostics and hashes are in `aggregate.json`.

A first preparation was interrupted after independent review identified a fitted-subset margin issue, before any decoder fit or validation metric. The corrected, documented protocol bases fitting eligibility only on context and retains zero-probe-like target rows. The interrupted record is preserved in ignored runs/.

## Limits

The five switch chains are descriptive perturbations. Structural zeros can disconnect their state space; acceptance and moved-label fractions do not prove uniform sampling or convergence. User-bootstrap intervals are conditional on these draws, are not multiplicity-adjusted, and are not randomization p-values. Repeated splits share users, and these validation users were used in earlier project studies.

Known-dislike rates measure only rated held-out dislikes and cannot certify satisfaction. The full-history mode adds known input history and shifts its length relative to fitting; it is never evaluated as held-out probe reconstruction. Surprise versus shuffled weights alone cannot separate personalized surprise from item-popularity-correlated teacher scores. Hard-negative weighting has prior art; no method-priority or state-of-the-art claim follows from this audit.

## Reproduction

Run `negative_information_experiment.py --source-root runs/research-v2 --ratings PATH/ml-100k.inter --item-metadata PATH/ml-100k.item --out runs/negative-information-v2 --seeds 2026 2027 2028`, then `negative_information_report.py --source runs/negative-information-v2 --out evidence/negative-information-v2`. Use the pinned project environment and set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, VECLIB_MAXIMUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS to 1. Output directories must be new. The default records score hashes without storing large raw score archives; `--save-scores` retains them.

The pre-fit specification is [NEGATIVE_INFORMATION_PROTOCOL.md](../../NEGATIVE_INFORMATION_PROTOCOL.md). Primary references: [LAGCL4Rec](https://aclanthology.org/2025.findings-emnlp.61.pdf) and [Rapallo and Yoshida](https://arxiv.org/abs/0905.4841).
