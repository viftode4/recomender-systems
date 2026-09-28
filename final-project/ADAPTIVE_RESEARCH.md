# Let the meaning of evidence depend on other evidence

The starting question is not whether dislikes should repel recommendations.
That already assigns an interpretation to a rating. A low score might reflect
a broad preference, one disappointing film, or a person's general use of the
rating scale. MovieLens records the score, not its explanation.

The new prototype asks a narrower, testable question: **does allowing an
observation's influence to change with the surrounding evidence improve
prediction of other recorded ratings?**

## What changes in the model

The model receives movie identities, the five observed rating categories and
a separate observation mask. It learns rating representations from scratch;
it does not convert training ratings into positive, neutral and negative
classes. Missing ratings supply no source value.

Each movie has a small vector of temporary prediction state for the current
user. Learned intermediate vectors let evidence move between any movies,
without preset genre neighborhoods. Over repeated updates, the model can
change both where evidence travels and how strongly an observation influences
the state around it. An observation is a soft source, not an unchangeable
boundary value. The resulting state predicts a distribution over all five
rating categories.

This is an independently trained predictor. It takes no EASE, SLIM or other
recommender predictions as inputs. The established models remain comparison
baselines required by the assignment.

The fluid analogy motivates changing influence and routing. This is not a
simulation of physical fluid, a claim of conserved mass, or evidence that
preferences themselves obey a fluid equation. Attention weights and source
gates are computational variables, not identified feelings or explanations.

## What would count as useful evidence

Training hides some TRAIN ratings and asks the model to predict their five
categories. Hidden values must be absent from all source values and masks.
Validation selects a checkpoint using categorical cross-entropy. A separate
development cohort measures the selected models; the project test stays
closed until all configurations are frozen.

The essential comparisons use the same rating information, architecture size,
training episodes and optimization budget:

- Adaptive routing and soft observation influence.
- Routing and observation influence fixed during the update sequence.
- Adaptive routing with observed states forced back to their learned evidence
  values, testing the cost or benefit of treating observations as fixed.

Global and smoothed item/user rating frequencies are additional sanity
baselines. If those predict ratings better, a complex model has not earned an
accuracy claim. Parameter counts and timing are part of the comparison.

Predicting ratings and predicting which movies receive ratings are different
tasks. Liked-item ranking uses the explicitly declared adapter P(rating >= 4),
without changing the model's categorical training inputs. The assignment's
all-observed ranking remains a separate diagnostic with a different target.
Neither rating likelihood nor a visually interesting changing field proves
better recommendations.

## Prior work and the limit of the originality claim

Learning interactions within a set through intermediate attention vectors has
a direct predecessor in [Set Transformer, Lee et al., ICML 2019](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf).
Reaction and diffusion operators in neural networks are also established;
see [GREAD, Choi et al., ICML 2023](https://proceedings.mlr.press/v202/choi23a/choi23a.pdf).
They have already been applied to recommendation, for example
[RDGCL, Choi et al., 2023 preprint](https://arxiv.org/abs/2312.16563).

Our experiment concerns the particular combination of learned categorical
evidence, recurrent routing and revisable observation influence, with matched
controls on this project protocol. A custom implementation and a different
combination do not establish that nobody has tried it. A positive controlled
result would support this mechanism in the measured setting; a negative result
would be retained and used to reject or narrow the hypothesis.

There are still assumptions: finite state size, shared learnable parameters,
a fixed computation budget and a predictive loss. These make the experiment
computable and testable. The point is to expose those assumptions and test
their consequences instead of silently assigning a psychological meaning to
every rating.
