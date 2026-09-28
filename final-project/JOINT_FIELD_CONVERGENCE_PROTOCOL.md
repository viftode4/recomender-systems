# Post-v1 convergence sensitivity study

Decision recorded after inspecting v1 development results and its meta-fit
learning curves, before this extension is fitted and before any final TEST
access. This is an exploratory extension, not an independent replication.

The first four completed v1 fits all selected the 100-epoch boundary. Their
meta-fit joint NLL improved by 0.097 to 0.184 between epochs 60 and 100. This
justifies checking the sensitivity of the conclusion to optimization budget;
it does not prove that a larger budget will improve held-out results. The
training and validation losses use different context sizes, so their raw gap
is not interpreted as a conventional generalization gap.

Everything in JOINT_FIELD_PROTOCOL.md remains fixed except the following:

- 400 epochs per model, with checkpoints 10, 30, 60, 100, 200, 300, 400.
- Same adaptive and fixed_flow variants, seeds 2026/2027/2028, architecture,
  damping, categorical inputs, Adam learning rate, batch size, identity-only
  episode masks, batch order, objective, count baselines and ranking adapters.
- Minimum macro-user meta-fit joint NLL selects the checkpoint, retaining the
  earlier checkpoint on an exact tie. Development labels do not select it.
- Restart from the original seed. V1 snapshots lack Adam optimizer state, so
  loading its weights and starting fresh Adam at epoch 100 is prohibited.
- Before passing epoch 100, require exact first-100 training losses and mask
  hashes, plus exact model states and TRAIN-query raw logits at checkpoints
  10/30/60/100, against the completed v1 artifacts. Abort on a mismatch.
- Preserve v1 outputs, sources and frozen bundles unchanged. This new entrypoint
  imports immutable helpers and never modifies the old module's globals.
- Record source/runtime hashes and prior-study hashes before fitting. Each
  seed is an independent process with one numerical thread; three seeds may
  run concurrently. The optimization budget stays equal across variants.
- Freeze both variants on all three seeds before opening TEST. The 400-epoch
  cap is final for this investigation: no further model/budget expansion follows.

The v1 result and the extension both remain reportable. A better result after
400 epochs indicates sensitivity to training budget in this reused development
setting; it does not retrospectively validate a world-first architecture or
provide an unbiased psychological interpretation of routing gates.
