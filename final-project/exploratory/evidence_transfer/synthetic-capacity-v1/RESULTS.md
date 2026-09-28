# Synthetic pattern-capacity check

This constructed case isolates the two pattern channels. Both candidates have exactly equal first seven evidence features; only pattern count and coverage differ. The richer pattern is labelled positive by construction. Every candidate/item permutation is paired with an ID swap, so catalog tie order is exactly balanced. The frozen shared model has no ID embeddings.

| Seed | Variant | Initial probe NLL | Final probe NLL | Strict wins / 64 | Exact ties / 64 | Tie-aware accuracy |
|---|---|---:|---:|---:|---:|---:|
| 2026 | full_pattern | 0.794753 | 0.000213 | 64 | 0 | 100.00% |
| 2026 | no_pattern | 0.693147 | 0.693147 | 0 | 64 | 50.00% |
| 2027 | full_pattern | 0.630796 | 0.000172 | 64 | 0 | 100.00% |
| 2027 | no_pattern | 0.693147 | 0.693147 | 0 | 64 | 50.00% |
| 2028 | full_pattern | 0.656167 | 0.000208 | 64 | 0 | 100.00% |
| 2028 | no_pattern | 0.693147 | 0.693147 | 0 | 64 | 50.00% |

Both variants use identical initialized parameters per seed, 300 full-batch Adam steps at learning rate 0.01, and the same synthetic examples. There is no checkpoint selection, model selection, or data-dependent budget extension. The no-pattern control necessarily ties because the two candidate inputs become identical.

Probe instances are independently permuted copies of the same structural template. This demonstrates learnability of an intentionally supplied pattern relation; it does not demonstrate generalization to new relation types, useful preference signal in real data, a recommendation improvement, or methodological novelty. Pattern count and coverage co-vary in this toy, so the experiment does not isolate which of those two channels supports learning. No real data or previous evaluation artifact was read.
