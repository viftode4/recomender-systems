# Independent pre-fit review

Status: **pre-fit implementation review passed**. The base model, data,
evaluation, runner and final workflow checks passed. This is not evidence
of recommendation quality. No real training, development outcomes or original
TEST labels were accessed by this reviewer.

## Independent numerical and boundary evidence

`test_independent.py` contains 18 passing synthetic tests. Its expected values
come from finite examples and direct calculations rather than the production
scoring helpers:

- A literal donor panel fixes every node feature, typed edge and all nine
  summary features. A separate scalar cosine calculation verifies donor order,
  deduplication and the sixteen-donor limit.
- Poisoning the entire excluded query row changes no graph input, retrieval
  result or degree feature. Changing another user's candidate rating does change
  its permitted evidence. Visible query degree remains the visible history size.
- Rating-preserving switches retain every donor/history rating margin and the
  query/candidate boundary edges. Candidate ordering and batching preserve scores.
- A NumPy/Python loop computes all three per-rating mean message layers, donor
  pooling and the prediction head independently of the batched graph operations.
- An explicit finite partition verifies the sampled loss and its analytic
  derivatives, exhaustive sampling, no-alternative limit and equal episode
  weighting. Enumerating a two-alternative sampling distribution confirms that
  the partition estimate is unbiased while the logarithmic loss is biased.
- Malformed item fields for an unchosen cohort are ignored before parsing.
  Episode tests separate visible context, every hidden TRAIN target and the
  complete-TRAIN-absent alternative pool.
- Candidate microbatch gradients match one literal full-denominator backward
  pass. A nondeterministic reconstruction is rejected.
- A real tiny synthetic training run resumes with identical losses, selected
  epoch, catalog score digests, model parameters and Adam state. Direct tests
  also restore Python, NumPy and Torch RNG streams and reject changed guards.
- A synthetic complete selection seal rejects changed source, predictions,
  training summary, meta screen, runtime or missing seeds before accessing
  development inputs. Score digests follow catalog order, not set iteration.

Command used from `final-project/`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
runs/environment-check/.venv/bin/python -m unittest \
  exploratory.conditional_evidence.test_independent \
  exploratory.conditional_evidence.test_retrieval_independent \
  exploratory.conditional_evidence.test_workflow -v
```

The latest completed run passed all 32 tests in 1.068 seconds: 18 independent
base checks, eight independently written retrieval checks, and six workflow
tests. The retrieval checks include real tiny fold training with identical model,
Adam state and losses after interruption and resume. The workflow suite includes
the positive gate, actual synthetic utility fitting, every policy's frozen
predictions, completed-stage reuse and required-artifact rejection. This is
additional evidence alongside the owners' model, data, evaluation and retrieval
tests; it does not replace a completed-study audit.

## Findings corrected before scientific fitting

The first graph implementation divided all typed messages by total degree;
the approved protocol requires a mean within each rating type. The implementation
was corrected, and the independent three-layer oracle distinguishes these cases.

The runner review also identified and corrected an accidental 600-epoch resume
after a recorded no-extension decision, unchecked global summary/screen hashes,
metric-only selected-state replay, unordered meta-user score digests and scramble
seeds missing their episode identity. The current runner validates the complete
checkpoint schedule, recomputes extension triggers from the original 300-epoch
prefix, binds runtime and input/source digests, replays exact selected scores,
and requires the retrieval release to agree with the sealed meta gate.

The base assessment supports unchanged-choice interrupted replay and verifies
already completed outputs instead of silently overwriting them. The fold
training callback receives only other-fold category rows, with the hidden fold
absent from both training and the donor bank. Initialization and episode seeds
are recorded separately.

## Workflow completion and recovery

Completed retrieval manifests require the exact seed, base selection seal,
policy set and artifact inventory, including all policy predictions. Every
artifact digest is checked before reuse. The global release precedes development
access. A completed retrieval assessment binds its release and result hashes;
reusing it returns verified results without reopening outcome labels. An
incomplete assessment may replay the same frozen choices, with conflicting
completed output rejected.

The workflow publishes aggregate crossfit/full-bank feature distribution changes,
fold and deployment costs, and the privileged TRAIN-only oracle diagnostic.
Feature comparisons describe action-weighted means and scales; they do not prove
that independently trained latent coordinates align. Per-query deployment costs
are reduced before publication. The base evidence checksum index remains
immutable, with a separate index for reports and supplementary diagnostics.
Only selections at the actual final epoch cap are labelled unconverged. An
operating-system lock prevents concurrent workflows using the same output root.

No implementation blocker remains in the reviewed source. The actual study must
still verify complete selected-state replay, all-seed seals, numerical aggregates
and protected historical artifacts after fitting. Synthetic success does not
substitute for those checks.

## Limits that remain part of the interpretation

The raw and scrambled readers share an architecture and preserved rating margins.
Raw versus summary also changes available rating/degree information and parameter
count, so that contrast alone cannot isolate correspondence. Anonymous mean
message passing cannot distinguish every different graph arrangement; a null
result would concern this finite reader, not all uses of raw evidence.

Candidate-positive donors are permitted TRAIN evidence. Their selection is
explicitly outcome-dependent on other users' records, not on the query's hidden
target. The observed-event endpoint is not a claim that all rated items were
enjoyed, and missing records are not verified dislikes. The reused development
cohort, overlapping splits, search budgets and possible selected epoch cap remain
limitations even when every numerical and provenance check passes.

## Historical artifact boundary

Only this review and `test_independent.py` were written by this reviewer for the
new study. Existing scientific sources, reports, results and packages were left
unchanged. The completed coursework package was independently checked before
this study: 347 safe unique ZIP entries, all 346 payload hashes and CRCs, exact
current PDF bytes, and byte-identical archived framing/original PDFs. No raw
array, rating-table or checkpoint payloads were present.

- Coursework PDF SHA-256:
  `343cfc595db940d7289ff0ea5a11b85f8ab1021de7a14ed41d763e722515c2ee`.
- Coursework ZIP SHA-256:
  `799819d24950dde22c72293f7ddae5a5d2c6ab94ff3949f96a13d3880dfd802e`.

This document does not claim a model win, fresh confirmation, original
architecture, completed real-data fit or external submission.
