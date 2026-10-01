# Independent review of coursework completion

The four additions are established hybrid families. Completing the lecture-family
coverage is a useful coursework deliverable; it does not establish a new
architecture, a new research result, or improvement over the strongest reference.

The assessment reuses the 236 policy-calibration users in each of three existing
splits. Those labels have informed earlier work. Freezing the new choices before
reading them in this runner protects this execution's selection boundary, but does
not restore a fresh holdout. The three seeds reuse people and data; seed means are
descriptive, not independent replications of a new population.

## Tests before execution

The independent model tests compare both meta-level ridge stages with separate
augmented least-squares problems; check representation-basis permutation and zero
observations; verify exact mixed-list quotas, duplicate/seen/PAD skipping and
exhaustion fallback; compute reciprocal-rank fusion from literal eligible lists;
and exercise switching at exact quantile boundaries, tied activity, absent fit
groups and exact expert-utility ties.

Before the actual run, all 21 independent cases passed: twelve model cases and
nine source/selection-boundary cases. The latter dynamically reject any access to
`valid_mask` or `test.tsv`, feed malformed and poisoned skipped calibration rows,
and check source hashes and TRAIN user alignment. They reject an absent third
seed, an omitted cohort hash, changed sealed artifacts, a later tied maximum, and
a selected prediction file that disagrees with the selected candidate.

Review found two seal gaps before execution: a manifest could omit an artifact
hash, and the selected prediction file was not checked against the selected
candidate. The producer fixed both, required the exact seven-artifact inventory,
and validated finite development values, fixed settings, roles, and stages. The
independent tests passed before the first real fit.

## Interpretation limits

The meta-level encoder uses the fixed catalog's genre matrix, including items
without training observations. This is declared transductive item metadata, not
held-out relevance. TRAIN reconstruction objectives and source predictions are
fixed before development chooses settings. Selection occurs separately within
each family; all selected families and fixed references must be reported even
when they perform poorly.

## Completed independent replay

`audit-v1/audit.json` records a passing audit of the actual completed run. The
standalone `verify.py` imports none of the scientific implementations. It checked
all source/input/output hashes and the global selection barrier before parsing
VALID relevance, recomputed development metrics for all 39 saved candidates,
verified the first exact maximum in each family, and independently reconstructed
all nine selected/reference rankings for every seed. Both selected meta-level
stages used augmented least-squares reference solves. No hyperparameter search or
original source-model training was repeated.

The audit independently reproduced all 27 assessment tables, saved per-user
accuracy/novelty values, activity and popularity groups, diversity, calibration,
coverage and discounted exposure, then checked public tables and descriptive seed
means. **43,239 numerical checks passed; maximum absolute difference was
7.105427357601002e-15.** All scientific hashes remained unchanged through the
135.16-second audit. Neither `test.tsv` nor the frozen `valid_mask` array was read.

The verified assessment means are context nDCG@10 0.278133, EASE 0.273765,
mixed 0.216840, meta-level 0.198748, tuned switch 0.274641 and tuned RRF 0.271294.
The tuned and fixed switch means are identical. These results fill concrete
coursework coverage gaps, while providing no evidence that the added methods
improve on the locked context reference. The weaker results are retained.

The receipt binds the source, completed assessment manifest, public artifacts and
selection seal; `audit-v1/SHA256.json` binds that receipt and the independent
verifier's exact source. Reproduction command:

```sh
runs/environment-check/.venv/bin/python coursework_completion/verify.py \
  --run runs/coursework-completion-v1 \
  --evidence coursework_completion/results-v1 \
  --out /tmp/coursework-completion-audit-new
```

Use a new output directory. Full numerical replay requires the ignored private
prediction/source bundles; the public report package preserves aggregate evidence
and its audit receipt.
