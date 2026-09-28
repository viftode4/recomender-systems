# Joint recorded-item and rating likelihood

Exploratory development results on reused validation; final TEST remains closed.

The unchanged categorical field uses joint recorded-event likelihood. Checkpoints minimize meta-fit macro-user joint NLL at epochs 10, 30, 60 or 100.

| Seed | Model | Joint NLL | Conditional rating CE | All-observed nDCG@10 | Liked-record nDCG@10 |
|---|---|---:|---:|---:|---:|
| 2026 | adaptive | 6.99397 | 1.32685 | 0.21139 | 0.20099 |
| 2026 | event_global_category | 7.87525 | 1.46167 | 0.11840 | 0.10745 |
| 2026 | event_item_category | 7.79261 | 1.37903 | 0.11839 | 0.10597 |
| 2026 | event_item_user_tilt | 7.70931 | 1.30708 | 0.11896 | 0.10566 |
| 2026 | fixed_flow | 7.01022 | 1.30710 | 0.19516 | 0.18628 |
| 2027 | adaptive | 6.98757 | 1.26332 | 0.19463 | 0.17807 |
| 2027 | event_global_category | 7.86455 | 1.45299 | 0.12111 | 0.11245 |
| 2027 | event_item_category | 7.77573 | 1.36417 | 0.12111 | 0.11794 |
| 2027 | event_item_user_tilt | 7.65184 | 1.25816 | 0.12425 | 0.12056 |
| 2027 | fixed_flow | 6.97787 | 1.24706 | 0.19494 | 0.17997 |
| 2028 | adaptive | 7.02824 | 1.32517 | 0.18610 | 0.17417 |
| 2028 | event_global_category | 7.87904 | 1.47867 | 0.12544 | 0.11799 |
| 2028 | event_item_category | 7.79725 | 1.39688 | 0.12545 | 0.12226 |
| 2028 | event_item_user_tilt | 7.70014 | 1.31742 | 0.12502 | 0.12010 |
| 2028 | fixed_flow | 7.01916 | 1.32503 | 0.18943 | 0.17419 |

All-observed ranking uses logsumexp of all five logits. Liked-record ranking uses logsumexp of category 4 and 5 logits. Missing pairs compete as possible recorded outcomes, not observed dislikes. This is not an exposure-corrected or satisfaction probability model.

Adaptive versus fixed flow jointly tests recurrent routing and source-gate changes. Shared validation reuse and unadjusted descriptive intervals limit interpretation. No claim of novel likelihood, SOTA, causal meaning, or identified emotions is made.
