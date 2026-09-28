# Tail misses mostly concern items already observed in TRAIN

Only **18 of 4,734 tail validation positives (0.38%)** have zero TRAIN support.
**3,819 (80.67%)** have at least 21 TRAIN observations. The poor tail recall
therefore cannot mainly be explained by completely unseen catalog items. This
does not establish which representation or objective would recover those misses.

This is a read-only, post-test exploratory diagnosis of the existing selected
binary and categorical reconstruction models. It uses original TRAIN histories,
VALID pair identities, catalog genres and the same 472 development users per
seed. No raw rating file, TEST payload, fitting, parameter selection or new
assessment was used. All source and score checks passed, and tail totals match
the [earlier diagnosis](results-v1/RESULTS.md) exactly.

The tail is the bottom 80% of the nonpadding catalog: head membership takes the
top `ceil(20% × 1682)` items by TRAIN observation count, with original-token
lexicographic tie breaking. The tail has 1,345 items per split. Its highest TRAIN
support is 80, 81 and 81 in seeds 2026–2028; the head starts at 81, 82 and 81.
The equal-support boundary in the last split is resolved by the declared tie rule.

## Support and exact recommendation hits

Counts below pool the three reused development splits. They represent evaluation
pairs across overlapping splits, **not 13,859 independent observations or unique
people**. Full per-seed counts, conditional macro recall and denominators remain
in [tail-support-v1.json](tail-support-v1.json).

| TRAIN users observing a tail item | Tail VALID positives | Binary hits@10 | Categorical hits@10 | Binary tail recommendation slots | Categorical tail recommendation slots |
|---|---:|---:|---:|---:|---:|
| 0 | 18 | 0 | 0 | 0 | 0 |
| 1–5 | 180 | 0 | 0 | 0 | 0 |
| 6–20 | 717 | 0 | 0 | 0 | 0 |
| 21–50 | 1,881 | 3 | 4 | 32 | 34 |
| 51+ (up to the tail boundary) | 1,938 | 22 | 23 | 116 | 121 |
| Total | 4,734 | 25 | 27 | 148 | 155 |

There are 13,859 total VALID positives and 14,160 recommendation slots across
these splits. The tail contains about 34.16% of the positives but receives only
1.045% of binary slots and 1.095% of categorical slots. Its **pooled** recall is
25/4,734 = 0.528% and 27/4,734 = 0.570%. Conditional macro-user recall is a
different statistic and is retained separately; these denominators must not be
interchanged.

All 915 positives with support at most 20 are missed by both models, and neither
model recommends any item in these support bins. This is observed ranking
concentration, not an assertion that recommending each rare item is beneficial.
Across the entire catalog, binary retrieves 2,475 positives and categorical
retrieves 2,500. Only two of the net 25 extra hits occur in the tail; adding
categories leaves the broad concentration nearly unchanged.

| Seed | All VALID positives | Tail positives | Binary tail hits | Categorical tail hits |
|---|---:|---:|---:|---:|
| 2026 | 4,655 | 1,620 | 6 | 6 |
| 2027 | 4,584 | 1,525 | 8 | 10 |
| 2028 | 4,620 | 1,589 | 11 | 11 |

## Available shared evidence is common

For each missed target, we inspect only its TRAIN co-observations with that
user's TRAIN context. The maximum support is the largest number of TRAIN users
who observed both the target and one context item. Because the evaluated target
is absent from that user's TRAIN history, that user cannot supply its own
target/source co-observation.

Among the **4,709 binary tail misses**:

- 99.62% have a positive co-observation count with at least one context item.
- 96.30% have at least one source/target pair observed by five or more TRAIN users.
- The mean maximum pair support is 31.74; per-seed medians are 32.5, 32 and 32.
- 99.43% share at least one provided catalog genre with the user's TRAIN history.
- Mean target-genre coverage by that history is 98.88%, and mean maximum
  source-item genre Jaccard similarity is 0.902.

The corresponding values for categorical misses are nearly identical. These
figures average missed positive pairs, so users with more missed positives have
more weight; they are not uniform-user estimates. Rich histories and broad
genres can produce high overlap even when candidate discrimination is weak.
Catalog genre overlap is descriptive metadata access, not a trained content
recommender or evidence that genres will fix the problem.

Available co-occurrence likewise does not establish positive conditional
preference evidence. It can reflect item popularity, redundant histories,
exposure or other associations. The binary model already uses collaborative
co-occurrences in a conditional linear solution; counting them does not prove
the model ignored them. What this diagnosis rules out is the narrow explanation
that most misses have no observed item history or no shared context at all.

The remaining question concerns discrimination and ranking under existing
support. Demonstrating a fix requires an explicit model or policy comparison
with the same candidates and assessment, including any loss of head accuracy.
It cannot be inferred from these support counts, and this reused dataset cannot
provide fresh final confirmation.

## Reproduction

```sh
runs/environment-check/.venv/bin/python \
  exploratory/research_diagnosis/tail_support.py \
  --items /path/to/RecBole_DSAIT4335/dataset/ml-100k/ml-100k.item \
  --out exploratory/research_diagnosis/tail-support-reproduction.json
```

Run from the project root. The script verifies the completed global selection
seal, TRAIN/category and ordered-identity hashes, selected-score hashes and
catalog metadata. It refuses to overwrite output. Synthetic aggregation checks
covered conditional versus pooled counting, exclusive model hits and empty
strata. On real outputs it also verified that the five bins partition all
positive pairs and hits, and that both models' tail counts reproduce the earlier
diagnosis. Output is aggregate-only, with no user/item identifiers, individual
histories, score matrices or recommendation lists.
