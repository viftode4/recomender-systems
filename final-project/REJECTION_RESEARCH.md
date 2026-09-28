# What information is a dislike actually adding?

Literature checked against primary papers on 2026-09-28. This is a research
design note, not a claim of novelty priority or a new evaluation result. No real
test interactions were opened for this audit.

The five-branch design is specified in [NEGATIVE_INFORMATION_PROTOCOL.md](NEGATIVE_INFORMATION_PROTOCOL.md).
This note records its research context and identification limits. The approved
design includes strict context/probe separation, fixed and equally tuned ridge
comparisons, five weight-permutation controls and five placement controls.

The next useful experiment is to isolate personalized rejection from two simpler
explanations: some users assign many low ratings, and some movies receive many
low ratings. A more complicated negative-feedback architecture cannot establish
which information source produced its gain.

## Closest prior work and what it rules out

| Primary source | Relevant mechanism | Implication for this project |
| --- | --- | --- |
| [Frolov and Oseledets, *Fifty Shades of Ratings*, 2016](https://arxiv.org/abs/1607.04228) | Models the rating category as a third tensor dimension; includes negative-only user feedback and avoiding irrelevant recommendations. | Using low ratings, separate rating channels, or evaluating unwanted recommendations is established work. |
| [Huang et al., *Negative Can Be Positive*, 2023](https://doi.org/10.1016/j.ipm.2023.103403) | Analyzes different negative signals and uses signed graph encoders and a sign-aware loss. Negative interactions can rank high under existing models. | A low rating that contradicts a model's high score is not an unexplored observation. |
| [Chen et al., *SIGformer*, SIGIR 2024](https://arxiv.org/abs/2404.11982) | Uses signed graph paths and spectral information to combine positive and negative feedback. | Relational transfer through dislikes and signed higher-order connectivity have close predecessors. |
| [Zheng et al., *LAGCL4Rec*, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.61.pdf) | Separates semantically similar hard negatives from easier negatives; increases hard-negative weights in BPR and contrastive learning. Includes low-rating feedback and LLM augmentation. | Giving extra weight to a dislike near an established positive interest is too broad a novelty claim. Its machinery differs from a small linear model, but the core motivation overlaps. |
| [Ivanova et al., *Benefiting from Negative yet Informative Feedback by Contrasting Opposing Sequential Patterns*, RecSys 2025](https://arxiv.org/abs/2508.14786) | Separate positive/negative sequential encoders with positive and negative cross-entropy and an opposing-pattern contrastive objective. | Contrasting positive and negative preference patterns is established; a new implementation alone is insufficient evidence of a new principle. |
| [Yin et al., *Correct and Weight*, January 2026 preprint](https://arxiv.org/html/2601.04291v1) | Corrects implicit-feedback false negatives and favors confident/easy negatives with a positive-relative score weight. | This is a counterpoint to hard-negative weighting: uncertain implicit negatives may actually be positives. It does not directly establish the right treatment of observed one/two-star ratings. |

These sources establish nearby mechanisms, not a compatible numerical SOTA
ranking for our protocol. Their datasets, relevance thresholds, candidate sets,
splits and resources differ. Published metric values should not be placed beside
ours as if they were controlled comparisons.

## One concrete experiment

Use the existing positive-target signed-channel ridge model as a measuring
instrument. Keep the model, target, all positive interactions, and the set of
observed training pairs fixed. Let `P` indicate ratings at least four, `D`
indicate ratings at most two, and let eligible replacement positions be observed
ratings at most three. Three-star ratings remain a working neutral category, not
a claim about the user's internal attitude.

Construct a controlled perturbation of `D` using checkerboard switches on two
users and two items. All four positions must be observed nonpositive training
pairs, and their dislike indicators must be either `[1, 0; 0, 1]` or
`[0, 1; 1, 0]`. Switch to the other arrangement. Each operation preserves:

- every positive interaction and every observed-pair mask;
- each user's number of dislikes;
- each item's number of dislikes;
- the total amount of negative evidence.

It changes who dislikes which item, conditional on these constraints. Train and
score each perturbed branch with its own perturbed `D`; training on one version
and scoring with another would answer a different question. Compare real `D`,
several independently perturbed versions, and positive-only EASE with the same
relevance target and candidate masks. Positive-only predictions must be exactly
invariant to these perturbations, providing a strong implementation check.

Because the decoder fits only users with at least one context like, switches
must also stay within the eligible and excluded user strata, defined using
context alone. Zero-probe-like fitting users retain all-zero probe targets.
Otherwise, global item dislike counts could remain fixed while counts in the
actual fitting matrix change. Verify item margins within each stratum, not just
across all users. This is a stricter control within the same branch.

Fix a small common regularization grid before running. On the same meta-fit
cohort, either select each branch under identical tuning budgets, or report the
entire fixed grid as a sensitivity analysis. Do not tune only the real branch
and compare it to an arbitrarily weak perturbed branch. Evaluate paired
differences on the development cohort. Freeze the comparison and its selection
rule before any further confirmatory evaluation permitted by the project's
evaluation protocol.

The primary question is whether true dislike placement predicts separate liked
items better than placements that retain user severity and item rejection
frequency. Report liked nDCG and recall, plus observed held-out dislike exposure
with its explicit label denominator. The latter measures exposure to known
dislikes; unlabeled recommended items have unknown acceptability.

If true placement wins consistently, the model has evidence of useful
user-item-specific rejection structure beyond the preserved margins. If the
effect is absent, that rejects its exploitation by this model under this
protocol; it does not prove that dislikes contain no information. If the real
and perturbed branches both improve equally over positive-only EASE, the gain
may come from the retained marginal structure rather than personal rejection.
This interpretation is a hypothesis to inspect, not an automatic causal
identification result.

## The perturbation is not automatically an exact statistical null

The unobserved and positive cells are structural zeros for the switch process.
General 2-by-2 moves need not connect every binary table with the same margins
when structural zeros are present. See [Rapallo and Yoshida, *Markov bases and
subbases for bounded contingency tables*, 2010](https://arxiv.org/abs/0905.4841).

Record proposed/accepted moves, fraction of changed labels, affected users and
items, immovable rows/columns, overlap with the original matrix over time, and
variation between independent chains. Count rejected proposals as steps when
sampling the chain; sampling only after successful moves can bias the
stationary distribution. A stationary-looking overlap trace does not prove
uniform sampling or connectivity. Without those guarantees, call this a
fixed-margin perturbation stress test and report the between-perturbation
spread. Do not present a Monte Carlo rank as an exact conditional p-value.

## Surprise weighting: an additional leakage trap

Suppose the positive teacher produces `S = P @ B`, and the negative features
are `W[u,d] = D[u,d] * g(S[u,d])`. When fitting the reconstruction target
`P[u,j]`, zeroing the decoder's own positive and negative item-j coefficients is
not enough: every other weight can still depend on `P[u,j] * B[j,d]`.
Cross-fitting the teacher by users removes that user's influence on `B`; it
does not remove the target from the query profile `P[u,:]` used to construct
`W`. This is indirect target leakage through derived features.

The approved design divides TRAIN history into disjoint context and probe
targets using pair identities, without consulting rating values. It fits the
teacher using context and computes surprise solely from context positives.
The five branches compare positive features, ordinary dislikes, surprise
weights, within-user permutations of those weights, and the placement controls.
All branches predict the same separate probe likes. Mean-one weighting preserves
each user's total dislike feature mass. This is a controlled predictive
information audit, not a new architecture.

The secondary full-history inference uses the same fitted decoder with all
original training history, including probe observations, and evaluates only
separate validation targets. It must not be reported as held-out probe
reconstruction. The longer query context creates a stated distribution shift.

Users with one dislike receive exactly the ordinary unit weight, and tied
weights can remain identical after permutation. Report the number of users
with at least two dislikes, within-user weight variation, and the number of
actually changed permutation rows. Even a gain over shuffled weights would
show useful weight-item alignment, not by itself prove that the information is
specific to a user's positive profile: item popularity may also influence the
teacher's scores. No extra architecture or branch is implied by these limits.

## Limits that the available data cannot resolve

MovieLens low ratings do not identify why a user disliked a film: genre,
execution, expectations, context and temporary mood can produce the same
record. Genres cannot resolve these latent explanations. The proposed
experiment concerns the predictive information in recorded ratings, not the
causal effect of revealing a dislike on a person's preferences.

The dataset authors explicitly explain that rating timestamps are not viewing
times: users can backfill many old ratings in one session, sometimes prompted
by the site's own recommendations. A temporal holdout tests later rating
entries, not a demonstrated trajectory of changing taste. See [Harper and
Konstan, *The MovieLens Datasets: History and Context*, section 3.2](https://files.grouplens.org/papers/harper-tiis2015.pdf).

The site's interaction-generation mechanism also limits conclusions about
real-world recommenders. [Fan et al., *Our Model Achieves Excellent Performance
on MovieLens: What Does it Mean?*, 2024 revision](https://arxiv.org/abs/2307.09985)
documents how onboarding and recommendations shape the observed interactions.
Repeated splits of these same users and movies do not establish transfer to a
new dataset or real user satisfaction.
