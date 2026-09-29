# What the records observe, and what we ask them to predict

Conceptual audit, 29 September 2026. Sources: the actual assignment PDF, local
data headers and experimental protocols, and the primary references below.
No fitting, new DEV metrics, or TEST access. Proposed diagnostics remain unrun.

## Two observations that our main target deliberately separates

A MovieLens row records **which movie someone rated and the rating they gave**.
These are different outcomes. Our all-recorded ranking target retains the first:
recover a hidden member of a person's recorded collection. It is legitimate for
that benchmark to reward finding a movie the person rated one star. Calling the
same success evidence that they would enjoy the recommendation adds an
unsupported interpretation.

This is already explicit in our
[categorical reconstruction protocol](../categorical_reconstruction/PROTOCOL.md):
all observed entries are positive targets regardless of category; missing
entries are not confirmed dislikes. The audit identifies a risk in our reasoning
and language, not an undiscovered implementation bug.

The local interaction header contains user ID, item ID, rating and timestamp;
item metadata supplies title, release year and genre, and user metadata supplies
basic demographics. The original dataset contains 100,000 ratings on a 1–5
scale from 943 users, with at least 20 ratings per retained user. It supplies
neither displayed candidate sets nor records of rejection or movie availability.
[Official MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k/README).

Consider A rated 5, B rated 1, and an unrecorded C. Recovering either A or B earns
the same relevance credit in our main benchmark. C could be unknown, unavailable,
watched but not rated, or deliberately avoided. A blank does not distinguish
those explanations. Even unlimited computation cannot recover which explanation
holds from that blank alone; this does **not** establish an accuracy ceiling for
the observable record-completion task.

Two otherwise identical binary profiles with reversed rating values receive
identical EASE predictions under the same fitted model. That is a precise
invariance. It proves that the binary representation cannot distinguish their
recorded judgments, not that distinguishing them must improve record completion.
Our completed rating-aware studies already test some uses of this information;
their small or unsuccessful gains cannot be discarded by renaming the target.

## Time identifies a submission, not a viewing

The dataset creators explicitly warn that rating timestamps can occur years
after consumption, with many old movies rated together. The site's recommendations
can prompt this backfilling. Its early interfaces also supported genre and
release-date filtering. These facts make a reporting circumstance plausible,
but do not identify any particular page, intent, or exposure event.
[Harper and Konstan, pp. 7 and 15](https://files.grouplens.org/papers/harper-tiis2015.pdf).

A and B submitted a second apart could be successive judgments about two old
movies, rather than two successive viewing experiences. Movies sharing an exact
timestamp could have been submitted together; clock resolution or batching also
remain compatible explanations. Neither timing pattern establishes a shared mood.

The proposed **(user, exact submitted timestamp) group** therefore has a defensible
observable meaning without inventing a viewing session. Our completed categorical,
pair, addressed-evidence and evidence-transfer models ignore that partition. The
earlier temporal sensitivity study used time to change the split, rather than
learning from within-query grouping. This is an untested information distinction
in our project, not a claim that modeling the recording process is new.

A TRAIN-only coherence audit should compare real groups with within-user shuffled
groups preserving each user's group sizes. Excess genre coherence would establish
structure beyond the unordered bag. It would **not** establish that a missing
record is predictable when its group identity is unknown. Any later masked-target
experiment must remove the target before constructing groups and statistics;
inference must not receive its timestamp or its true group. The model must work
from visible groups while preserving the existing relevance and candidate rules.

## One diagnostic of the target distinction, without another model

Using fixed existing scores, compare the ordering of two already-observed DEV
movies whose recorded ratings differ. Both movies are known to have generated
ratings, so their comparison asks about recorded judgment rather than finding a
record among unobserved catalog entries.

For each eligible user, calculate the fraction of unequal-rating pairs for which
the higher-rated movie has the higher score; score ties count one half. Average
users equally and report eligible user and pair counts. Freeze the model list,
cohort and formula before computation; do not choose checkpoints on this metric.
Retain the original completion nDCG beside it. Report this as exploratory analysis
of repeatedly used DEV data, not fresh confirmation.

If strong completion scores poorly order these judgments, the two objectives
empirically disagree. If they order them well, the claim that binarization erased
all useful preference information weakens. Neither outcome licenses causal claims
about previously unexposed movies. Selection affects which judgments are present;
the general problem and propensity-based remedies have established prior work.
[Schnabel et al., ICML 2016](https://proceedings.mlr.press/v48/schnabel16.html).

## Assignment fit and the decision boundary

The [actual brief](../../reference/Project-RecSys.pdf), p. 4, requires RecBole,
MovieLens 100K, ranking and separate evaluation. Pages 7–8 request insightful
behavior/group analysis and its implications for hybrids. It does not explicitly
mandate all-ratings-positive relevance, a threshold, random 80/10/10 splitting,
or cutoff 10. These are adopted choices, not permission to compare incompatible
scores as if they measured the same task. Required hybrids, metrics and societal
analyses on pp. 5–10 remain part of the assignment.

The immediate discipline is simple: distinguish **membership in a recorded
collection**, **ordering of recorded judgments**, and **future response to an
offered movie**. They require different evidence. A new framing must specify
which distinction it can exploit and a comparison that could disprove it. The
target distinction alone does not explain our EASE deficit, and a different
endpoint would not constitute beating EASE on the original benchmark.
