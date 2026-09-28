# The pattern features help this model, which still loses to EASE

The new shared evidence interpreter completed all nine declared 100-epoch
training trajectories and 45 checkpoint evaluations. Each seed's model choices
were frozen before the new development comparison. Results use the same reused
MovieLens 100K development population as earlier studies and are exploratory.

| Standalone model/control | Mean nDCG@10 | Tail hits across splits |
|---|---:|---:|
| Full evidence interpreter | 0.245189 | 41 / 4,734 |
| Matched interpreter without pattern count and coverage | 0.238678 | 30 / 4,734 |
| Analytic weighted donor support | 0.215292 | 0 / 4,734 |
| Marginal-support-only interpreter | 0.121652 | 0 / 4,734 |
| Locked binary EASE | 0.261688 | 25 / 4,734 |

The full model improves nDCG by 0.006511, approximately 2.73% relative, versus
the matched no-pattern control. Differences are positive in all three splits:
0.011785, 0.002120 and 0.005628. The full model also improves mean recall, MRR
and catalog coverage against that control. These are descriptive measurements
on overlapping splits, not an independent replication or a causal explanation.
The two pattern inputs are removed together; their separate roles are unresolved.

Against locked EASE, the full model is **6.30% lower in mean nDCG**, and loses
in every split. Mean recall and MRR are also lower. Its higher tail hit count
does not compensate for the overall loss. The declared substantial-pilot target
of a 10% relative gain with positive differences in every split was not met.
No breakthrough, state-of-the-art superiority or novelty priority was established.

## What changed in the research question

Unlike the previous near-null independent pair-feature test, this matched
comparison supplies evidence that the selected pattern features add useful
information within the new model. That is a reason to investigate the mechanism
further, not to substitute a weaker comparator for EASE. The broad shared-local-
evidence approach and the diversity statistic have established prior art.

The model deliberately compresses all evidence into nine prescribed summaries
before a 177-parameter scorer sees it. It cannot learn distinctions those
summaries erase. Conversely, its training budget did not establish that it had
fully exploited even those summaries: all full and no-pattern arms selected
epoch 100, the final allowed checkpoint. Continued training-loss reduction and
slowing meta improvements justify an optimization check, but do not guarantee
that additional training would close the baseline gap.

The next useful decision is to separate optimization from representation:
first assess the fixed model's learning curves and declare any convergence
follow-up separately; then test richer learnable relational evidence against
the same summaries. Simply increasing depth, adding another expert's score, or
renaming the geometric statistic would not resolve the hypothesis. Any further
development exploration remains exposed-data research and needs untouched-data
confirmation before a strong generalization claim.

## Supporting checks and boundaries

- Independent synthetic checks reproduce all nine features with explicit donor
  loops and verify complete query-row exclusion, masking, normalization, parameter
  matching, geometric bounds and checkpoint behavior.
- The [capacity fixture](synthetic-capacity-v1/RESULTS.md) demonstrates that the
  network can learn a deliberately supplied pattern distinction. Its training
  and probe cases reduce to the same two feature vectors under permutations,
  so it supplies no independent generalization or real recommendation evidence.
- The inference-only zero-pattern intervention obtains 0.209724 nDCG. It retains
  41 tail hits while spending 13.602% of slots on the tail, versus 2.394% for the
  full model. This is a distribution-shift sensitivity check, not the trained
  no-pattern ablation or a fairness improvement.
- [Full results](results-v1/RESULTS.md), [declared comparison](analysis-v1/RESULTS.md),
  and [independent audit](audit-v1/RESULTS.md) preserve the evidence. Individual
  records, features and checkpoints remain under ignored `runs/` directories.

The required coursework remains available as a separate review draft. The
broader request for a substantial original advance remains unmet; this study
provides a tested mechanism and a narrower next question, not completion of it.
