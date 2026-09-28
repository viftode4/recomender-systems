# Categorical evidence-field pilot

Exploratory development evidence on reused validation splits. Final TEST remains closed.

Each neural variant selects the smallest macro-user five-category cross-entropy on meta-fit users at epochs 10, 30, 60, 100. Training, initialization and masks share a fixed budget. Count baselines have no validation-selected settings.

| Seed | Model | Macro rating CE | Macro Brier | Expected-rating RMSE | Liked nDCG@10 | All-observed nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| 2026 | adaptive | 1.30573 | 0.68775 | 0.98087 | 0.01170 | 0.01162 |
| 2026 | fixed_flow | 1.30508 | 0.68752 | 0.98017 | 0.01140 | 0.01103 |
| 2026 | global_histogram | 1.46167 | 0.74710 | 1.12615 | 0.02312 | 0.02735 |
| 2026 | hard_clamp | 1.30721 | 0.68809 | 0.97967 | 0.01223 | 0.01271 |
| 2026 | item_histogram | 1.37903 | 0.71709 | 1.03557 | 0.05224 | 0.04899 |
| 2026 | item_user_product | 1.30708 | 0.68815 | 0.98197 | 0.04777 | 0.04575 |
| 2027 | adaptive | 1.23133 | 0.65640 | 0.93279 | 0.00668 | 0.00676 |
| 2027 | fixed_flow | 1.23341 | 0.65730 | 0.93409 | 0.00693 | 0.00703 |
| 2027 | global_histogram | 1.45299 | 0.74522 | 1.12149 | 0.02149 | 0.02205 |
| 2027 | hard_clamp | 1.23508 | 0.65713 | 0.93675 | 0.00689 | 0.00643 |
| 2027 | item_histogram | 1.36417 | 0.71230 | 1.01611 | 0.05051 | 0.04461 |
| 2027 | item_user_product | 1.25816 | 0.66753 | 0.94664 | 0.04860 | 0.04227 |
| 2028 | adaptive | 1.29394 | 0.68138 | 0.98032 | 0.02244 | 0.02085 |
| 2028 | fixed_flow | 1.29316 | 0.68096 | 0.97971 | 0.02400 | 0.02306 |
| 2028 | global_histogram | 1.47867 | 0.75239 | 1.14895 | 0.02232 | 0.02519 |
| 2028 | hard_clamp | 1.29366 | 0.68049 | 0.98111 | 0.01986 | 0.01816 |
| 2028 | item_histogram | 1.39688 | 0.72323 | 1.05907 | 0.05814 | 0.05355 |
| 2028 | item_user_product | 1.31742 | 0.69099 | 1.00267 | 0.05553 | 0.05089 |

Lower CE, Brier and RMSE are better. Ranking uses the declared P(rating>=4) readout and measures a different target; it does not determine checkpoints. Missing ratings are unknown. Categorical metrics condition on rated pairs and cannot establish satisfaction for unobserved items.

The changing routes and source gates are model variables, not identified feelings, trust, or explanations. Shared validation reuse, one small dataset and descriptive unadjusted user intervals preclude claims of independent confirmation or state of the art.
