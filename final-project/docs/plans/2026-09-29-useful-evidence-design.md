# What evidence would change the recommendation?

Conceptual refinement, 29 September 2026. Extends the
[revisable predictive-state design](2026-09-29-revisable-predictive-state-design.md).
This is a research proposal, not an implementation, result or novelty claim.
No experiment outcomes were opened and no running-study sources were changed.

## The missing capability

The earlier design learns how to revise a belief after feedback. The next useful
capability is to anticipate which feedback would improve a decision, then check
whether that anticipated improvement actually occurs.

One model supplies the response distribution, personal state and update rule.
It can evaluate possible answers using temporary copies of the current state.
Only an actual answer enters the evidence memory. There is no requirement for a
second recommender, committee of scorers or separate reinforcement-learning agent.

## Three routes and the recommended order

| Route | What it would add | Limitation | Decision |
|---|---|---|---|
| Richer item descriptions | More observable distinctions between items | Extra attributes cannot reveal unobserved personal context; interpretation can outrun evidence | Preserve current identity and available attributes; defer extra data until a specific predictive gap warrants it |
| Learn and audit how evidence transfers | Make one response improve predictions on other items, while checking the consistency of hypothetical updates | Consistency alone admits an updater that never learns | First refinement; requires proper target prediction loss and controlled ambiguity tests |
| Choose useful feedback | Use the same update rule to estimate the value of asking about an item | Depends on reliable predictions of answers and their consequences; retrospective ratings are selected observations | Test after the update rule helps under identical acquired evidence |

The relationship worth learning is: **in this history, what does learning the
response to item i change about the response distribution for item j?** Ordinary
content resemblance is one possible source of transfer, not its definition.
This is a contextual relationship and can depend on the answer received.

For example, a low rating of a science-fiction action movie is compatible with
several predictions about other films. An answer about a science-fiction movie
without the action tag may distinguish some of those predictions. It does not
establish the psychological cause of the original rating. Identical observable
data can support multiple explanations, including item-specific dislike.

## A precise consistency condition

Begin with a fixed-time rating-revelation episode. Let H be visible evidence,
i an unqueried item, j a distinct target, Y_i a possible answer, and R_j the
target rating. Shared model parameters, candidate availability and the target
random variable remain fixed throughout this thought experiment.

For exact conditional distributions from one joint model, total probability gives:

```
p(R_j = r | H)
    = sum_y p(Y_i = y | H) * p(R_j = r | H, Y_i = y).
```

For our learned update U, the rightmost term is approximated by the predictor
after U(H, i, y). Measure the discrepancy between the two sides across target
items and ratings. For five-star retrospective data the sum has five terms.
Actual elicitation would need all supported responses, including abstention or
"not seen" where applicable. Record candidate identity and any selection context
in the conditioning information consistently on both sides.

In plain language: averaging forecasts across hypothetical answers should recover
the current forecast. A particular real answer can still change it substantially.
This controls average signed drift, not average movement or every uncertainty
measure. The displayed identity is a property of coherent conditioning; it is
not a new theorem and does not certify that the joint model is correct.

Hypothetical branches must be discarded. They never become observations, replay
records or pseudo-labels. Repeatedly imagining one answer must leave the actual
personal state and evidence ledger unchanged.

This first diagnostic deliberately excludes elapsed-time transitions and actions
that alter preferences. A temporal extension must compare predictions about the
same future variable under the same transition and action assumptions; blindly
equating forecasts at different times is invalid.

An identity update passes the condition. So can incorrect but mutually consistent
predictions. Use it initially as a diagnostic, together with proper predictive
loss on distinct concealed targets after real feedback. A later regularizer
would need a declared loss weight, a controlled comparison and checks for update
collapse. Do not add it to a training objective merely because it sounds principled.

## Choosing a question from its consequences

For a declared fixed candidate set A and bounded rating-derived utility u, define
V(H) as the largest expected utility among the available recommendations. The
estimated value of asking about i is:

```
estimated_value(i) = sum_y p(y | H, i) * V(U(H, i, y)) - V(H) - question_cost(i).
```

Utility and question cost must have matching units. Initially compare equal
one-answer budgets with zero assigned cost; this avoids inventing a monetary or
psychological cost for an offline dataset. Report fewer-question performance
curves later. The queried item is excluded from the scored target candidate set.

Under exact coherent conditioning with unchanged available actions and zero
question cost, the expected value of information is nonnegative: the decision
maker could retain the previous action. A learned approximation has no such
guarantee. It can also overestimate value by becoming overconfident. Measure
predicted value against realized rating-derived utility over many episodes,
including harmful revisions; never score a policy solely using its own belief.
Positive expected value does not require every realized answer to improve utility.

High response entropy is a separate selection rule. An unpredictable answer can
be irrelevant to which recommendation is useful. Conversely, a modest uncertainty
can matter greatly when it changes a close decision. Expected utility is an
evaluation choice, not a claim to have identified satisfaction caused by exposure.

A constructed example makes the distinction precise. Suppose two equally likely
states favor opposite recommendations: each preferred item has success
probability 0.8 and the other 0.2. A question whose positive-answer probabilities
are 0.9 and 0.1 in the two states has a 50/50 marginal answer. After either
answer, the better recommendation has predicted success 0.74. Before asking,
both recommendations have success 0.5. The expected decision gain is 0.24,
although any fixed item's two branch predictions, 0.74 and 0.26, average back
to 0.5. An independent fair-coin answer has the same answer entropy and zero
value. These are exact illustrative calculations, not dataset results.

## The smallest informative experiment

Use permitted development records to construct visible histories, concealed
question pools and disjoint concealed target pools. Keep final assessment
outcomes unavailable to model development. This dataset has already been explored;
these studies remain exploratory and do not create a new untouched benchmark.

For retrospective rating revelation, query and target item identities can be
declared available to all methods, while their ratings are concealed. Pool
membership therefore conditions on an item having been rated. This evaluates
conditional rating prediction and ranking within a known-rated pool, not
full-catalog exposure, next-item prediction or real-time preference change.
Use the same fixed candidate and target pools across policies for each episode.

1. First prescribe identical reveals for every update rule. Score predictions
   before and after one reveal on other targets. This separates revision quality
   from question-selection quality. Include the same accumulated evidence in an
   order-free reconstruction control.
2. With the revision rule held constant, compare one question selected randomly,
   by response entropy, and by estimated decision value. No policy may inspect
   an unrevealed rating or select using assessment outcomes.
3. Report target log loss, rating-derived recommendation utility, harmful-update
   frequency, calibration, update time and the consistency diagnostic. Evaluate
   per-user paired differences and uncertainty across the declared episodes.
4. Inspect whether larger predicted query value corresponds to larger realized
   improvement. Failure here would directly undermine choosing questions with
   this model, even if passive predictions were good.

Keep this fixed-time experiment separate from the earlier temporal-bundle
protocol. Synthetic streams with known causes can test mechanics, but cannot
prove that the same hidden causes exist in MovieLens.

## New observations that would resolve more

If we later collect real responses, a small repeated-measure study could ask
about familiar anchor items across sessions, record the distinction between
general appreciation and a choice for the current situation, and include a
counterbalanced repeated question. No collection is started by this proposal.

Repeated ratings add information about consistency. Explicit context adds an
observed conditioning variable. Neither guarantees a clean separation of mood,
noise and changing taste; one repeated rating is particularly insufficient.
The ambition is to test predictions that current data cannot distinguish, rather
than assign human-readable names to unidentifiable latent variables.

## Prior art and the actual research bet

Active collaborative filtering already selects queries by expected value of
information. Meta-learning already trains inference and adaptation rules.
Predictive martingale methods already connect sequential predictions with
uncertainty and decisions. Combining those phrases is not an originality claim.

The specific hypothesis here is whether an inspectable learned update can provide
useful transfer to other item predictions, remain approximately coherent under
hypothetical revelations, and forecast which real answer will improve a decision.
A contribution would require a specified mechanism and evidence on those tests.
This targeted literature pass is not an exhaustive novelty or state-of-the-art audit.

Primary sources checked:

- [Zemel and Boutilier, An Active Approach to Collaborative Filtering, 2003](https://proceedings.mlr.press/r4/zemel03a.html): recommendation-oriented query selection using expected information value predates this proposal.
- [Vendrov et al., Gradient-based Optimization for Bayesian Preference Elicitation, AAAI 2020](https://arxiv.org/abs/1911.09153): scalable optimization of information value for preference queries, including item comparisons and attributes.
- [Zintgraf et al., VariBAD, ICLR 2020](https://arxiv.org/abs/1910.08348): meta-learned approximate belief inference and decisions conditioned on uncertainty are established in reinforcement learning.
- [Fong, Holmes and Walker, Martingale posterior distributions](https://arxiv.org/abs/2103.15671): develops prediction-based uncertainty inference using martingale structure. The basic total-probability diagnostic above does not inherit this paper's full construction or guarantees.
- [Duran-Martin et al., Martingale Posterior Neural Networks for Fast Sequential Decision Making, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/7f52f6b8f107931127eefe15429ee278-Abstract-Conference.html): recent prior art for neural predictive updates and sequential decisions; not evidence that our proposed mechanism will work.

## Next deliverable

Extend the proposed `predict`, `observe`, `replay` loop with a side-effect-free
`preview` operation. A demonstration should show predictions before feedback,
the possible answer branches, their estimated decision values, the actual
revealed answer, and changes on separate targets. Freeze the concrete model,
losses and split rules in an executable protocol before training. No large new
sweep is justified until this small experiment shows useful revisions.
