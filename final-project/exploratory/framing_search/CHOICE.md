# Choice, competition, and what MovieLens actually observes

Conceptual investigation, 2026-09-29. No fitting, new metric evaluation, or TEST
access. This is an assessment of a framing, not an experimental result.

## The simple perspective

**A choice is meaningful relative to what it displaced.** A recommendation is
not necessarily an isolated judgment that a movie is good. Two similar movies
may compete for the same evening; one may make another unnecessary. The useful
quantity would then be the change in a person's decision when an alternative
appears, rather than a permanent score attached to each movie.

That is a coherent perspective. It does not yet provide this project with a new
mechanism, because the literature already studies it and MovieLens 100K does not
record the alternatives presented at each decision.

## A precise contrast, without a neural network

Let two movies A and B initially receive equal choice probability. Introduce A2,
a close substitute for A. In a simple hypothetical population, half the people
want the experience represented by A and half want B. The former split evenly
between A and A2; the latter still select B:

| Available alternatives | P(A) | P(A2) | P(B) | A/B odds |
|---|---:|---:|---:|---:|
| A, B | 0.50 | — | 0.50 | 1.00 |
| A, A2, B | 0.25 | 0.25 | 0.50 | 0.50 |

A single softmax over fixed item scores cannot express both aggregate rows: its
A/B odds are always `exp(score(A) - score(B))`, regardless of A2. A mixture of
different stable tastes can produce such an aggregate pattern, so this example
alone does not demonstrate irrationality or changing individual preferences.
It is a structural contrast, not a claim about behavior in MovieLens.

EASE itself produces scores, not choice probabilities. It would be inaccurate
to claim that EASE explicitly assumes this softmax choice law. Also, its signed
item-to-item coefficients already permit an observed history item to reduce a
candidate's score; simply introducing negative influence is not a new
capability. The relevant distinction is dependence on *currently offered
alternatives*, which differ from previously rated history items. Source:
[EASE, equations 1–3](https://arxiv.org/pdf/1905.03375).

Choice-set dependence is established prior art. The context-dependent random
utility model adds effects from the other alternatives and supports testing
departures from fixed relative odds. Pairwise Choice Markov Chains provide
another established route through pairwise transition rates. These are direct
precedents, so names such as "competition flow" would not make the framing new.
Sources: [CDM, ICML 2019](https://proceedings.mlr.press/v97/seshadri19a.html),
[PCMC, NeurIPS 2016](https://arxiv.org/abs/1603.02740).

## Why the required data cannot distinguish the toy worlds

Suppose we only observe 50 people rating A and 50 rating B. At least two
different processes are compatible with these records:

1. All 100 people considered both movies and made a meaningful selection.
2. Each person encountered only their eventually rated movie, so no A-versus-B
   choice occurred at all.

The observed records can be identical while predictions after changing the
available alternatives disagree. This is an identifiability argument, not a
sample-size complaint: more records of the same incomplete kind do not, by
themselves, reveal which world generated them.

MovieLens 100K supplies user, item, rating, and timestamp records plus metadata.
It does not supply the menus shown, items considered and rejected, availability
windows, or an explicit no-choice outcome. An unrated item therefore cannot be
silently labeled a losing alternative. Source:
[official 100K README](https://files.grouplens.org/datasets/movielens/ml-100k/README).

The data creators make a stronger point about time: rating times do not identify
watching times; users can backfill ratings long after viewing, sometimes prompted
by the site's recommendations. Thus timestamps cannot justify a model of what
someone chose for their next evening. Source: [Harper and Konstan, section 3.2](https://files.grouplens.org/papers/harper-tiis2015.pdf).

Even observed menus would not automatically establish causal context effects.
Different people may receive different menus, producing apparent context
dependence from selection. Within-distribution prediction can still benefit, but
counterfactual interpretation requires additional assumptions or controlled menu
variation. Source: [Choice Set Confounding, KDD 2021](https://www.cs.cornell.edu/~arb/papers/choice-set-confounding-KDD-2021.pdf).

## What we *can* compare without inventing exposure

For a person who rated both i and j, their recorded ratings reveal a within-person
ordering, with ties retained as ties. Define `d_u(i,j) = sign(r_ui - r_uj)` only
when both ratings exist. This removes any common additive rating generosity and
is invariant to a person's strictly increasing recoding of the rating scale.
It does not establish that i and j were offered together, or that retrospective
rating order equals prospective choice.

A minimal rule could estimate the fraction of comparable raters who preferred i
to j, optionally conditioning on observed taste disagreements in the query's
remaining history. No missing rating becomes a dislike. In a toy example, two
people who both rated X and Y have identical binary histories, but one preferred
X and the other Y. Binary-history EASE must give them identical candidate scores;
a rule using their ordering can distinguish them. That shows a possible source
of information, not that this information improves the all-recorded endpoint.

This is not a fresh discovery. Slope One uses differences among co-rated items;
collaborative learning to rank from rating-derived pairwise preferences predates
our project. Our existing categorical reconstruction study already used rating
information, so calling another rating-aware model the missing new perspective
would also ignore completed work. Sources:
[Slope One](https://arxiv.org/abs/cs/0702144),
[Learning to Rank for Collaborative Filtering, 2007](https://www.scitepress.org/PublishedPapers/2007/23963/pdf/index.html),
[our earlier study](../categorical_reconstruction/README.md).

## The useful consequence for the search

For the existing random held-out, all-recorded experiment, the immediate object
is **a partially observed collection of rated movies**. Reconstructing a missing
member of that collection differs from predicting which movie a person would
choose from a menu. A low-rated movie can correctly belong to their collection.

That suggests a productive question for the broader framing search:

> What structure makes a missing movie belong to this particular collection,
> beyond the sum of its similarities to the visible members?

This keeps the reasoning tied to the observable target. It leaves open simple
collection mechanisms such as exclusions, overlapping subsets, and distinctive
combinations, but those need separate investigation against prior art. It does
not justify claiming any of them is new or already effective.

Decision: **do not prioritize menu competition as our next ML100K model.** It
currently requires unavailable decision information and has close established
precedents. Retain the distinction between collection membership and preference
as a guide for choosing and interpreting a more suitable simple mechanism.

To study actual competition later, the discriminating evidence would be logged
offered sets and outcomes with reliable chooser context, preferably randomized
additions/removals of substitutes. Static held-out nDCG labels cannot establish
how a changed recommendation slate changes a person's response.
