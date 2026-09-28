# Learn how to interpret evidence across movies

The research question is whether a shared rule for interpreting collaborative
evidence can recover useful ranking information that our target-specific
reconstruction models missed. The first experiment tests one proposed ingredient:
the structure of the historical evidence behind a candidate, beyond its amount.
This is an exploratory mechanism test, not an established invention or advance.

## Why this question

The preceding diagnosis found that tail items receive about 1% of recommendation
slots despite supplying about one-third of held-out recorded interactions. Most
of those items have training observations and shared context. That availability
does not establish that positives are distinguishable from unrecorded candidates.
Meanwhile, adding independently fitted pair features under squared reconstruction
produced no meaningful improvement. More donor histories helped in a controlled
reduced-data experiment. These facts motivate learning a common evidence rule
across targets; they do not predict that it will work.

Three explanations were considered before this experiment:

1. A complete history mixes incompatible interests. A context decomposition might
   help, but our existing data does not identify current intent, and another
   bottleneck risks discarding useful evidence.
2. Target-specific parameters use sparse evidence inefficiently. A common rule
   could transfer patterns learned around frequent items to less frequent ones.
   This is the selected direction, with support-pattern structure as its first
   explicit hypothesis.
3. Squared reconstruction optimizes the wrong errors for ranking. A matched
   reconstruction-versus-multinomial study remains a separate useful control.
   The current experiment does not independently identify an objective effect.

## Proposed mechanism

For query history H, find how each other user's TRAIN history overlaps H. A
donor's weight is squared cosine similarity to H. For each candidate movie,
collect donors who recorded it and summarize their support, its concentration,
and the query-history patterns behind that support.

Normalize each donor's binary overlap pattern to unit Euclidean norm. If donor
weights are w and normalized patterns are a, define the effective pattern count
as (sum w)^2 / ||sum w a||^2. Identical patterns count as one; mutually disjoint
patterns yield the ordinary weight-based effective count. This is a geometric
redundancy statistic, not a measurement of statistically independent people,
confidence, causal evidence, emotion, or current intent.

A shared small network maps nine features to one candidate logit. Its inputs
contain no user/movie embeddings, movie identifier, pretrained model score, or
teacher prediction. Training hides actual TRAIN records and asks the network
to rank those records among eligible movies. Missing records are competing
outcomes in that task; they are not asserted to be disliked.

The query user's entire row is excluded from donor evidence and all population
normalizers. Otherwise a hidden training target would vote for itself. The
network learns from the resulting queries, while donor histories remain fixed.

## What would support or disprove it

Compare the full interpreter against the same architecture with pattern count
and coverage removed, a marginal-support-only arm, analytic weighted donor
support, and the locked binary reconstruction reference. The two pattern inputs
are removed together, so their individual causal roles are not separated.

If full and no-pattern models tie, this experiment does not support adding
support-pattern structure. If both lose to the reference, the proposed shared
feature representation has not earned replacement of the existing predictor.
If only the marginal arm improves, the evidence points to frequency calibration,
not the proposed mechanism. Track head/tail and total accuracy together so
reallocating slots cannot masquerade as a universal improvement.

A development gain is still exploratory. A substantial advance requires a
fixed method, strong matched controls, and confirmation on untouched data.
The first pilot uses a declared training budget; a loss is not proof that every
version of this mechanism fails or that the optimizer converged.

## Closest prior work and actual scope

[IGMC](https://arxiv.org/abs/1904.12058) learns transferable rating predictors
from local user-item subgraphs. [SEAL](https://arxiv.org/abs/1802.09691) learns
link predictors from enclosing graph patterns. Neighborhood weighting and
effective-count summaries also predate this experiment. The broad idea of
learning a shared local-evidence rule is therefore not new. Our local question
is whether explicit context-pattern redundancy adds ranking information beyond
support amount under the stated exclusion and matched-control protocol.

We are implementing the feature algebra and scorer ourselves to make this
hypothesis inspectable. Using NumPy, SciPy and PyTorch for numerical operations
does not outsource the model design. A successful pilot would justify deeper
learned relational operators and additional datasets; it would not establish
priority or superiority over every existing recommender.
