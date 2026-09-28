# Explicit-rating and preference-contrast development study

The proposed contrast representations failed to produce a competitive recommender. Positive-rating targets explain a more promising improvement; explicitly adding dislike channels does not consistently improve on the positive-only control.

All table entries are **development nDCG@10 with ratings>=4 relevant**, after meta-fit-only selection. The same training/validation interaction splits are used for every model. Equal-grid all-observed EASE selects ridge250 in all three splits. No test results are included.

| Model | 2026 | 2027 | 2028 | Mean | Known dislikes/slot |
|---|---:|---:|---:|---:|---:|
| All-observed EASE (equal tuning grid) | 0.2404 | 0.2421 | 0.2525 | 0.2450 | 1.62% |
| Positive EASE | 0.2482 | 0.2678 | 0.2564 | 0.2575 | 0.85% |
| Likes/dislikes channel ridge | 0.2484 | 0.2658 | 0.2596 | 0.2579 | 0.90% |
| Signed EASE | 0.2284 | 0.2361 | 0.2149 | 0.2264 | 0.54% |
| Contrast-transfer neighbors | 0.1314 | 0.1464 | 0.1299 | 0.1359 | 0.60% |
| Matched pair gate | 0.0968 | 0.0822 | 0.0935 | 0.0908 | 0.14% |
| Permuted counterpart gate | 0.0917 | 0.0762 | 0.0925 | 0.0868 | 0.13% |
| Random dislike-pool gate | 0.0869 | 0.0774 | 0.0894 | 0.0845 | 0.13% |
| Pair-anchor kernel | 0.0804 | 0.0697 | 0.0843 | 0.0781 | 0.20% |
| Pair-signed kernel | 0.0666 | 0.0663 | 0.0761 | 0.0697 | 0.13% |

The relation-neighbor model loses much of the useful liked-item location: its anchor-only ablation beats it substantially. Revision2 explicitly preserves candidate-to-liked-anchor similarity and gates it using the matched dislike. The gate improves its weak anchor baseline, but remains far below ordinary linear recommenders. A low known-dislike rate alone is not success: a poor recommender can avoid observed dislikes by recommending irrelevant, rarely rated items.

The broad random-partner control draws from all of a user's dislikes. A separately reviewed exact permutation control preserves every liked anchor and the entire disliked counterpart multiset, including reuse counts. Both are reported so the influence of pair alignment is not confused with changing which disliked items enter the model.

Positive EASE and the two-channel linear model use the same ridge grid. The per-split paired intervals for their difference include zero, so added dislike channels have not demonstrated an accuracy advantage. The all-observed EASE control uses the identical grid selected on the identical liked-rating objective; its higher observed dislike rate cannot be attributed to a larger tuning budget.

Evaluation uses 472 development users per split for all-observed relevance and 444/434/433 users for liked relevance. Users without a held-out like are excluded only from liked ranking metrics. Known-dislike rates count held-out ratings<=2 per all-user recommendation slot; missing ratings are unknown. All paired bootstrap intervals are descriptive, unadjusted for multiple analyses, and the repeated splits share users.

This is an exploratory negative result about operational liked/disliked contrasts, not a psychological study, a novelty certification or a state-of-the-art claim. Revision2 followed revision1 development evidence. Selection choices and full score hashes must be frozen before the separate test stage.

Reproduce the primary run with `exception_experiment.py --revision 2`; then run its `--audit-reference-root` equal-grid EASE audit and `--pair-permutation-reference-root` exact permutation audit. Use `exception_evaluation.py freeze` before the distinct `evaluate` command. Raw scores, user IDs, recommendation lists and rating records remain in ignored `runs/`; this directory contains aggregate-only evidence.

The unopened `runs/frozen-exception-v2` bundle is retained as provenance. Shared evaluation helpers changed while the main hybrid controls were extended. The replacement `runs/frozen-exception-v3` passed preflight with the updated transitive code fingerprint; all three model bundles are byte-identical to the unopened original, so model choices, scores and development figures are unchanged. Both bundles remain unopened. See `freeze-provenance.json` for the equality proof and hashes.
