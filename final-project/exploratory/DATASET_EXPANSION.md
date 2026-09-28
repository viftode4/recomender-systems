# Fresh-data replication plan

Status: proposed, not executed. Sources and local availability checked on
2026-09-28. No new dataset was parsed, no split was created, and no new holdout
was evaluated in preparing this note.

The next dataset should be **MovieLens 1M**, followed only if justified by a
separately specified question by **Jester Dataset 1**. ML1M is a practical
external replication of a rating-aware method; Jester would test whether a
result survives a different domain and rating interface. Neither requires paid
compute. Actual runtime still needs a training-only pilot.

## Assignment and evidence boundaries

The downloaded assignment, `reference/Project-RecSys.pdf`, page 4 explicitly
requires experiments on MovieLens 100K using RecBole and focuses on ranking.
`reference/ASSIGNMENT.md` records its verified source. The PDF does not explicitly
authorize replacing ML100K with a different dataset. The existing required
models, hybrids, evaluation, and societal analysis therefore remain the assessed
core; fresh data can be presented as supplementary replication.

The ML100K final holdout has already been opened. Any subsequent model design
informed by these outcomes is exploratory on ML100K, even if evaluated on another
random split. Repartitioning those same records cannot restore an independent
confirmation set. Preserve the completed final evidence as a historical result.

ML1M has not appeared in this project's model runs or local data inventory.
It is fresh to this research process, but still comes from MovieLens. Shared
platform, selection mechanisms, and films limit the independence of its domain.
It is not evidence of generalization to all recommendation settings. Do not
join users across releases or attempt identity matching. Describe it as a
separate release and user sample, not as proof of independent population draws.

## Dataset options and verified sources

| Dataset | Available information | Fit to the question | Recommendation |
| --- | --- | --- | --- |
| MovieLens 1M | 1,000,209 ratings, 6,040 users who joined in 2000, approximately 3,900 movies; integer 1–5 scores, rating-entry timestamps, movie genres | Directly preserves positive, neutral, and negative rating categories while increasing data size | First replication dataset |
| HetRec MovieLens | Movie ratings plus IMDb/Rotten Tomatoes metadata; official page lists 2,113 users | Useful if a new mechanism requires richer content; shares the MovieLens domain | Defer until its README and archive are verified |
| HetRec Last.fm | Official page lists 92,800 artist-listening records from 1,892 users; social and tagging information | Different domain, but listening observations do not identify explicit dislikes | Not a direct replication of negative-rating claims |
| Jester Dataset 1 | About 4.1 million continuous ratings from 73,421 users over 100 jokes; −10 to +10, with 99 indicating missing data | Explicit negative preferences in a distinct domain; tiny catalogue is convenient for CPU mechanism experiments | Stronger later domain check, with its own endpoint definition |

ML1M is a stable benchmark with an approximately 6 MB archive. Its files are
`ratings.dat` (`UserID::MovieID::Rating::Timestamp`), `movies.dat`
(`MovieID::Title::Genres`), and `users.dat`. IDs are not all contiguous movie
records. The research license requires acknowledgement, bars implied
endorsement and commercial use without permission, and bars redistribution
without separate permission. Keep raw data out of code and report archives;
provide the official downloader and checksums instead. Sources:
[dataset page](https://grouplens.org/datasets/movielens/1m/),
[official README and license](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt),
[official archive](https://files.grouplens.org/datasets/movielens/ml-1m.zip).

The HetRec official landing page and directory disagree in useful ways: the
page reports only 86,000 MovieLens ratings, while the download directory has
an 18 MB `hetrec2011-movielens-2k-v2.zip` and a 256-byte old archive. Do not use
the landing-page count as a verified archive statistic, or the old archive as
the intended dataset. Both HetRec README responses failed web-tool text
decoding, so exact schema, counts, and license have **not** been verified here.
Last.fm's archive is approximately 2.5 MB. Sources:
[official landing page](https://grouplens.org/datasets/hetrec-2011/),
[official file index](https://files.grouplens.org/datasets/hetrec2011/),
[Last.fm README](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-readme.txt),
[MovieLens README](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-readme.txt).

Jester's primary source permits research use with acknowledgement. Dataset 1
comes in three ZIP files containing XLS matrices; the first column is a rating
count and the next 100 are joke ratings. Ten specific joke columns are nearly
universally rated, so an uncontrolled split can exploit this collection design.
There are no rating timestamps for temporal evaluation. Do not silently map the
continuous scale into MovieLens categories or interpret 99 as a high score.
Source: [Berkeley dataset documentation](https://eigentaste.berkeley.edu/dataset/).

Avoid `ml-latest` / `ml-latest-small` for the main replication: their publisher
calls them development datasets rather than stable shared research benchmarks.
The 10M–32M releases would substantially expand compute and storage, and their
half-star scale changes a five-category model's output space. Neither issue is
necessary to answer the first replication question.
[Official latest-small description](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

## Availability and acquisition

The inventory covered both personal checkouts:

- `/Users/vliftode/personal/recomender-systems`
- `/Users/vliftode/personal/recommender-systems`

The substantive local dataset directory contains only `ml-100k`; RecBole also
contains test fixtures and configuration files mentioning other datasets.
No `ratings.dat`, `ratings.csv`, `user_artists.dat`, `user_ratedmovies.dat`, ML1M
archive, or HetRec archive was found outside environment/cache directories.
Configuration names are not downloaded datasets or completed experiments.

Official sources are readable through the web tool, but an ordinary HTTPS HEAD
request to the official ML1M archive failed with `Could not resolve host` under
the current terminal sandbox. No general binary-download MCP was exposed.
Therefore no archive has been acquired. This is an acquisition limitation, not
a reason to reuse a compromised holdout. Do not route personal data acquisition
through company services or an unrelated connector.

Once an ordinary permitted downloader is available, save the original archive
under an ignored local run directory, record its URL, byte size, acquisition
time, SHA-256, publisher checksum if retrievable, and README/license hash.
Inspect archive member names for unexpected paths before extraction. A locally
computed SHA-256 provides reproducibility but is not a substitute for comparing
the publisher's checksum. Do not publish any raw or transformed rating tables.

## Proposed ML1M protocol, to fix before creating splits

1. **State the claim first.** Fix one candidate mechanism and its matched
   ablations, primary endpoint, baseline set, hyperparameter ranges, optimizer
   budgets, stopping rules, and minimum meaningful improvement before the
   external holdout exists. The purpose is to confirm or falsify a specific
   ML100K-derived hypothesis, not to search ML1M until it succeeds.

2. **Make one deterministic outer split.** Proposed default is a per-user
   rating-independent 70% training, 10% development, 20% reserved allocation,
   using a recorded seed and deterministic rounding/tie rule. Assign by an
   item-pair hash or a documented permutation, never by rating value. This is
   offline rating-record completion, not next-watch prediction. The exact rule
   must be agreed and sealed before execution; no split has been made here.

3. **Use a small trusted preparation step.** It may parse records to route
   them into partitions, but must not expose reserved ratings, distribution
   summaries, per-user reserved activity, or holdout-selected groups to model
   developers. After preparation, training/analysis processes read only the
   exposed training and development files. Hash and reserve the untouched
   evaluation file and remove the raw archive/extracted table from ordinary
   training paths. This is a procedural boundary, not cryptographic isolation
   against the machine's owner.

4. **Separate fitting from selection.** All learned embeddings, item
   popularity, graphs, genre profiles, user groups, data-dependent thresholds,
   normalization, and candidate pools use training records only. The
   development set selects hyperparameters. If a method learns a decoder,
   calibration policy, or surprise score, create its inner fit/calibration
   partitions inside training; do not reuse the same labels to both fit a
   feature generator and claim out-of-sample performance. Target ratings must
   be masked from their own input in denoising or reconstruction objectives.

5. **Keep the final model convention simple.** Proposed first replication
   freezes models trained on the 70% partition without a last-minute refit.
   At development evaluation mask training interactions; at final evaluation
   mask training plus development interactions for every model. Development
   ratings are not added to any method's profile. If refitting on 80% is wanted,
   explicitly replace this convention in the protocol before training, for
   every model and every training-derived feature; do not mix conventions.

6. **Fix the catalogue and missing-item behavior.** Use the same frozen
   metadata catalogue and known-history mask for all models. Represent unseen
   training items explicitly and define a deterministic fallback for methods
   that cannot estimate their scores. Do not silently remove difficult test
   items. Report a secondary warm-item endpoint using a training-defined item
   set, with its own denominator. Avoid filtering users/items by reserved
   counts or requiring a future positive when assigning training cohorts.

7. **Use strong, fairly tuned references.** Include all-observed EASE,
   rating-positive EASE, and SLIMElastic, alongside the appropriate simple
   count baseline and capacity-matched mechanism control. Tune on identical
   development access and disclose search budgets. Choose the objective for
   each reference in advance; do not select a different reference after the
   final results. ML100K's best regularizer need not be ML1M's best regularizer.

8. **Freeze before evaluation.** Seal data/split hashes, code/environment,
   training choices, scores or checkpoints, reference selections, endpoint
   adapters, group definitions, and the comparison list. Reuse the reviewed
   preflight and single-opening pattern: write an exclusive marker before
   the evaluator opens labels and record success/failure and output hashes.
   All prespecified branches must be included, including unsuccessful ones.
   Do not tune further on this holdout after observing the result.

## Endpoints and interpretation

The recommended primary replication endpoint for a rating-aware claim is
full-catalogue **nDCG@10 for reserved ratings ≥4**, macro-averaged over users
with at least one eligible liked target. Fix the cutoff and this denominator
before any result. Report eligible users, target counts, and cold-item counts.
The all-observed nDCG@10 endpoint remains an explicitly separate bridge to the
assignment, with every reserved recorded rating relevant regardless of score.
Also report recall@10 with the same endpoint-specific targets.

Candidate generation must not use reserved positives; do not rank against
sampled negatives for one model and the full catalogue for another. All score
adapters must be fixed from their objectives before evaluation. A conditional
rating probability and the probability that an item receives any rating are
different targets; this distinction was decisive in the earlier experiments.

For the negative-preference mechanism, measure recommendations hitting known
reserved low ratings (≤2) per recommendation slot, labelled as **observed
low-rating hits**. Unrated items are neither verified dislikes nor verified
likes. This quantity alone cannot establish user satisfaction or causal harm.
Calibration and fairness groups must use training histories; group utility,
item exposure, coverage and utility retention are distinct diagnostics.

Predeclare paired model-minus-reference user differences with at least 20,000
bootstrap resamples, multiplicity correction across the full confirmatory
comparison family, and point estimates. Multiple initialization seeds share
one holdout: report their spread, and if averaging them for a contrast, resample
users while retaining each user's whole vector of seed results. Do not treat
seed runs as independent datasets. If no interval excludes zero, report the
precision limit, not equivalence. Aggregate reports should contain no user IDs
or individual rating histories.

Do not compare raw ML1M nDCG numerically with ML100K nDCG as though task
difficulty were constant. Catalogue size, activity and target counts differ.
Compare within-dataset effects against the same reference families, then report
whether direction and magnitude transfer. A later Jester study needs a fresh
definition of positive/negative ratings, a control for universally rated jokes,
and its own sealed evaluation. It cannot be retrofitted into a successful
replication by changing thresholds after observing its results.

Temporal analysis would be a separate prespecified experiment, preferably with
global time boundaries if claiming future-information isolation. Per-user
chronological splitting can still let later events from other users enter
training. In either case MovieLens timestamps indicate when a rating was
entered, not when a movie was watched; batch backfilling prevents claims about
actual consumption order.
[GroupLens primary dataset paper, §3.2](https://files.grouplens.org/papers/harper-tiis2015.pdf).

## CPU feasibility

Using a conservative 4,000-item estimate, one dense float64 item Gram matrix is
about 128 MB and a 6,040-by-4,000 float32 score matrix about 97 MB, before solver
workspaces. This makes a bounded EASE/reference study plausible locally.
Compute item Gram matrices from sparse interactions; score in user batches;
save selected models rather than every dense intermediate. Profile one fit on
training before scheduling a grid, and record elapsed time and peak memory.

A dense five-category expansion has roughly 20,000 features and a 3.2 GB
float64 Gram matrix before copies or factorization. It should not be assumed
cheap merely because ML1M's ZIP is small. Prefer a sparse/low-rank implementation
or a small prespecified dimension where the model permits it. Training-only
runtime profiling may change compute budgets if recorded before comparing
development outcomes; it may not justify weakening only the baselines after
seeing their performance. No hardware capacity or training time has been
verified in this note: sandbox access to the system-memory query was denied.

Immediate next step: agree the hypothesis and the exact split/evaluation
contract, acquire the official ML1M archive through an available personal
download path, then implement and review the preparation boundary on synthetic
data before exposing the training partition.
