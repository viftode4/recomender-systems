# Independent review of the grouped-record prediction study

The experiment passed its independent implementation and artifact checks. It
**did not establish a predictive benefit for the tested grouping mechanism**.
This is an exploratory nested study of previously used MovieLens100K data, not
a fresh population assessment or a novel-architecture claim.

## What was checked

Twenty-four synthetic and boundary tests passed in a fresh post-run execution.
Twelve independently written model tests enumerate the full singleton/pair
feature dictionary, physically delete every target-owned feature, and solve an
augmented primal least-squares reference. They verify production predictions,
the exact EASE limit, target self-exclusion, changed query partitions, item/user
permutation and row-local label invariance, unseen candidates, empty queries,
and zero-column pair dictionaries. Grouped ridge is a known higher-order linear
feature model with a changed input relation; writing its solver ourselves does
not establish architectural novelty.

Ten independent runner tests check the declared ID-only nested partition,
held-out timestamp poisoning, ignored rating values, assessment gating before
file parsing, stable full-catalog ranking and literal metrics. Selection checks
require all 39 declared candidates in order, the first exact maximum for each
role, all six selected roles, and the complete sealed artifact set. Before
fitting, the review identified and corrected a seal omission that could otherwise
leave an assessment file unchecked. The runner's initial population guard also
caught the source catalog's explicit padding user before any fits; two additional
catalog tests verify the resulting explicit padding handling.

The retained [independent auditor](verify_results.py) imports none of the study's
model, runner or metric implementations. Its [receipt](audit-v1/audit.json)
records:

- Source, input, selection-seal and raw-run artifact hashes verified.
- The complete nested split independently reconstructed: 65,518 F records,
  7,645 D records and 7,645 A records over 943 real users.
- F membership and every retained timestamp independently matched to the raw
  original-TRAIN rows. Original VALID/TEST and nested D values were not parsed.
- All 39 candidate specifications and six first-maximum choices checked.
- All six selected prediction arrays independently ranked; selected D metrics,
  four A endpoints per role, activity/head-tail diagnostics, genre calibration,
  coverage/novelty and item exposure independently reproduced.
- Six paired bootstrap comparisons reproduced with their declared seed.
- 2,105 numeric metric checks, maximum absolute discrepancy
  `7.105427357601002e-15`.
- Selected EASE predictions checked against an independent item-space solution,
  maximum absolute score discrepancy `6.5503158452884236e-15`.

The auditor did not refit grouped models or rerun the 39-candidate search.
Unselected D metric values were not independently recomputed from predictions,
because only selected arrays were retained. Their candidate order and selection
use were checked; independent synthetic references cover the solver and metric
algebra. This distinction limits the audit claim precisely.

## Interpretation

| Selected family | Nested A nDCG@10 |
|---|---:|
| Equal-timestamp group pairs | 0.172555419 |
| Bag pairs | 0.172720673 |
| Mean of three shuffled-partition metrics | 0.172818720 |
| EASE | 0.172956063 |

The grouped model is 0.2316% relatively below EASE. Its nDCG differences against
bag, mean shuffled partitions and EASE are negative, and all corresponding
paired descriptive intervals include zero. Neither the grouping-effect criterion
nor the predeclared substantial-improvement threshold was met.

The result concerns same-group pair indicators under this ridge objective and
finite grid. It does not establish that grouping contains no possible predictive
information, that EASE is a ceiling, or that a different data-generating account
cannot help. Earlier genre coherence in timestamp groups was a descriptive
property; this test shows that property alone did not justify the expected
ranking advantage for the specified mechanism.

## Reproduce the audit

From the final-project directory, select a new output directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
runs/environment-check/.venv/bin/python \
  -m exploratory.framing_search.predictive.verify_results \
  --run runs/framing-predictive-v1 \
  --out runs/framing-predictive-audit-replay
```

The audit receipt SHA-256 is
`56a4aabcf7553b085a7a221fc408c89a4397cfc5fbe8503aa73950e5d25ffcdb`.
Its verifier-source digest is embedded in the receipt. No source in the sealed
scientific set was changed by this review. Audit elapsed time was 6.84 seconds;
wall time and receipt hashes differ on a fresh replay.
