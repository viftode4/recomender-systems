# Research beyond the coursework core

Read this after the [project map](PROJECT_MAP.md). These experiments are optional
context for the team meeting; they are not prerequisites for understanding the
required coursework. Different studies use different evaluation partitions, so
their scores do not form one leaderboard.

| Question | Current state | Read when useful |
| --- | --- | --- |
| Can a categorical model use explicit rating evidence? | Implemented and evaluated; no broad ranking win or demonstrated adaptive-routing advantage | [Research overview](../ADAPTIVE_RESEARCH.md), [matched results](../evidence/final-field-comparisons-v1/RESULTS.md) |
| Do ratings recorded together carry useful structure? | Measured structure, but the tested predictor did not show a ranking advantage | [Grouping experiment](../exploratory/framing_search/predictive/README.md) |
| Does addressed evidence improve prediction? | Completed experiment with negative primary results | [Postmortem](../exploratory/addressed_evidence/POSTMORTEM.md) |
| Does conditional graph evidence help? | Resumed 1 October at 12:23 UTC; running with six workers, incomplete, no final result | [Protocol and implementation](../exploratory/conditional_evidence/README.md) |
| Can a model learn how to revise predictions after new feedback? | Design only; not implemented or evaluated | [Revisable state](plans/2026-09-29-revisable-predictive-state-design.md), [useful evidence](plans/2026-09-29-useful-evidence-design.md) |

The [offline synthetic demo](../reports/demo/categorical-field/index.html)
illustrates the earlier categorical model. It shows prediction sensitivity on
a fictional profile, not an interactive study with real users, proof of the
newer design, or a measured gain in recommendation quality.

The conditional-evidence study resumed from saved checkpoints on 1 October
2026 at 12:23 UTC, with six accelerated workers and monitoring. Training,
selection and assessment are still in progress; no final gain is established.
This is a dated status snapshot, not a live dashboard. Operational instructions
are in [the acceleration guide](../operations/TRAINING_ACCELERATION.md); its
older descriptions of the original process are historical. Preserve the sealed
protocol and keep these exploratory results separate from the coursework tables.

The earlier long navigation documents are preserved byte-for-byte under
[docs/archive/2026-10-01](archive/2026-10-01/INDEX.md). Detailed research source
and frozen evidence retain their existing paths for reproducibility.
