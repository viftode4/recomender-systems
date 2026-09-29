# Independent review of the recording audit

29 September 2026. **PASS for the reported descriptive results**, with the
provenance and input-degeneracy limitations below. No model fit, DEV evaluation,
or TEST-label access was performed. Existing code, protocol and evidence were
not modified.

## Evidence independently checked

- All five retained unit tests passed in `runs/environment-check/.venv`.
- A separate synthetic implementation matched literal genre-set enumeration
  against the optimized implementation for observed grouping and 100 shuffles.
  Cases covered unequal user/pair weights, empty genre unions, users without
  ties, gap denominators, and poisoned non-TRAIN rating/timestamp strings.
- All 11 current digests matched: three output artifacts, two source/protocol
  files, and six inputs. The categorical input signature, original source
  manifest, copied/original TRAIN pairs and raw-file digests agree.
- A separate real-data replay reconstructed TRAIN membership, then parsed only
  allowed timestamps. It independently reproduced every descriptive count,
  threshold fraction and quantile. No non-TRAIN rating or timestamp was parsed.
- Literal genre-set intersection/union reproduced the observed statistics and
  all 100 seeded null draws. Maximum absolute discrepancy was
  `4.718447854656915e-16`. Null means, standard deviations, percentiles and
  observed-minus-null differences also matched.
- Source and input digests were checked again after this independent replay.

The replay confirms 80,808 TRAIN records, 943 users, 44,316 exact-time groups,
and 66,203 tied pairs across 942 eligible users. Pair-weighted and macro-user
statistics use distinct, correctly implemented denominators. Shuffling preserves
each user's records and timestamp multiplicities and consumes RNG draws even
for users without a tied pair, as declared.

Reviewed aggregate identity:

```text
aggregates.json
002eddeda09e42a235c1a22e9101fc234ab7beedab847e6e5a56d8e190920733

provenance.json
69a93b65b7a714dd5d81532b95a16230f09cddc16cbcdda54311c264a6e7f74a

audit.py
b767819d740abb87633f7ac30c8a4f86dd7e729edab9499a7acf400ac4928bfd

PROTOCOL.md
7aaa05cd06f778d699680fcd3fe188b876bca7e78e699560c28ed06c8187d072
```

## Limitations retained rather than silently repaired

The original runner records source hashes after computation and does not perform
a final input-drift check. Current hashes and an independent exact numerical
replay support the reported results; they are not a contemporaneous pre-execution
source seal for the original run. The README records this limitation. A future
version should capture source hashes before parsing and recheck all inputs and
sources before writing output.

The implementation does not gracefully render a dataset with no tied pairs or
no positive distinct-time gaps. The actual audited data has both, so this does
not invalidate its results. The protocol's initial `2026-EASE-0` source path
failed the hash gate before timestamp access; the completed run uses the
signature-matching `2026-EASE-1`. The README documents this correction without
changing the preserved protocol.

## Interpretation and disclosure

Output evidence contains aggregate counts, quantiles, aggregate null draws,
paths and hashes, with no user/item ID lists, raw ratings, raw timestamps or
pair-level values. The report accurately labels the shuffle ranges as descriptive
comparisons, not confidence intervals or hypothesis-test results.

The parent framing README makes a correct representational argument: identical
item/rating bags can have different timestamp partitions. The measured genre
coherence supports structure in those partitions, not predictive usefulness when
the target's timestamp/group is hidden. Neither document claims a recovered
screen, a viewing session, causal influence, recommendation improvement or
first-ever novelty. Those boundaries should remain in any later summary.
