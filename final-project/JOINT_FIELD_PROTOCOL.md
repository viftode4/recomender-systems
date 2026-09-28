# Joint recorded-item and rating experiment

This extension was authorized before examining any real results from the
conditional categorical field experiment. It changes the training likelihood,
not the architecture or the categorical inputs. The conditional experiment,
its checkpoints and its source files remain untouched.

For a query context C, the unchanged field produces five logits l(i,r) for
every item. Padding and every observed context item are removed from the
normalizing set. A masked TRAIN probe (i,r) has loss

    -l(i,r) + log sum_{j outside C, c=1..5} exp(l(j,c)).

All simultaneously masked probe items remain eligible under the same query.
Losses are averaged within user and then across batch users. TRAIN masking is
identity-only, keeps 80% of each user's observations, and never stratifies on
rating values. Categories 1 through 5 all remain supervised outcomes.

Writing a(i)=logsumexp_r l(i,r), this loss decomposes exactly into conditional
rating cross-entropy and item-event cross-entropy, with unit coefficients.
Conditional rating loss alone is invariant to an arbitrary shared shift of an
item's five logits. Joint loss identifies relative item offsets within the
eligible catalog; a global query-wide logit shift remains unidentifiable.
This removes an output-loss symmetry; it does not establish identifiability of
neural parameters, routing gates, or latent preferences.

The likelihood models one exchangeable recorded event among remaining
item/category outcomes. Unobserved pairs compete in the denominator and receive
gradients. They are not explicitly labelled as dislikes, but this is still an
assumption about competition between possible recorded events. It is not an
unbiased exposure model, a probability that an item will ever be rated, or a
chronological next-event forecast. MovieLens selection effects remain.

## Fixed design before fitting

- Seeds: 2026, 2027, 2028; original EASE-1 TRAIN/validation split and ID order.
- Same CategoricalEvidenceField: dimension 8, 16 ports, four recurrent steps,
  damping 0.5; no pretrained scores or new prediction head.
- Variants: adaptive and fixed_flow. Fixed flow caches initial routing and
  source gates within a query, jointly removing their recurrent changes.
- Identical initial weights, identity-only episode masks and batch order.
- Adam 0.001, batch 16, one CPU thread, 100 epochs each.
- Checkpoints 10, 30, 60, 100. Select minimum macro-user joint NLL on the
  pre-existing meta-fit validation users; exact ties retain the earlier epoch.
- Persist both choices before evaluating the disjoint development users.
- Inference supplies original TRAIN ratings only. Validation ranking and
  likelihood exclude TRAIN items and padding, without validation assimilation.
- All-observed ranking: logsumexp of all five logits. Liked-record ranking:
  logsumexp of logits for categories 4 and 5. The >=4 endpoint is an evaluation
  definition, never a binary training label. Conditional rating distribution
  remains a five-category softmax.
- Fixed count baselines: item-event counts plus one; global category counts
  plus one per category; item/user category distributions smoothed with mass
  10 toward that global distribution. Baseline joint masses are event mass
  times global categories, event mass times item categories, and event mass
  times item categories times user categories divided by global categories.
  The final variant is normalized jointly over eligible item/category pairs;
  its unnormalized category tilt also changes item-event mass, rather than
  leaving item popularity fixed under a normalized conditional distribution.
- Report joint NLL, conditional categorical NLL/Brier/RMSE, both ranking
  adapters with denominators/dislike rates, timing and aggregate routing
  diagnostics. User bootstrap intervals are descriptive on reused validation.
- Freeze all three seeds and both variants before any final TEST access.

This is an objective-alignment experiment in the established multinomial
recommendation family. Relevant predecessors include [Liang et al., 2018,
multinomial VAE/denoising recommendation](https://arxiv.org/abs/1802.05814) and
[Zheng et al., 2016, categorical autoregressive collaborative
filtering](https://proceedings.mlr.press/v48/zheng16.html). Neither this likelihood
nor the attention family establishes a first-ever invention. A benefit of
adaptive versus fixed flow would support the measured recurrent mechanism;
better ranking alone would not identify psychological meaning or feelings.
