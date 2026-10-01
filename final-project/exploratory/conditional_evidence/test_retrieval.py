"""Synthetic boundaries for the gated retrieval extension; no dataset fits."""
import copy
import io
import math
import unittest

import numpy as np
import torch

from exploratory.conditional_evidence import retrieval as r
from exploratory.conditional_evidence.model import EvidenceBank, GraphReader


def gate(enabled=True):
    return r.meta_gate({1: .5 if enabled else .2, 2: .5 if enabled else .2},
        {"scrambled": {1: .3, 2: .3}, "summary": {1: .35, 2: .35}},
        {"EASE": {1: .4, 2: .4}, "SLIM": {1: .38, 2: .38}}, stage="meta_fit")


def fixture(users=24):
    categories = np.zeros((users, 7), dtype=np.uint8)
    for row in range(users):
        categories[row, 1] = row % 5 + 1
        categories[row, 2] = (row+1) % 5 + 1
        categories[row, 3+row % 4] = (row+2) % 5 + 1
    episodes = []
    for row in range(users):
        context = categories[row].copy()
        target = np.flatnonzero(context)[-1]
        context[target] = 0
        episodes.append(r.Episode(row, context, np.array([target], dtype=np.int64)))
    return categories, episodes


class RetrievalTests(unittest.TestCase):
    def test_gate_requires_both_controls_and_strongest_mean_reference(self):
        self.assertTrue(gate().enabled)
        self.assertFalse(gate(False).enabled)
        tied = r.meta_gate({1: .4, 2: .4},
            {"a": {1: .1, 2: .1}, "b": {1: .2, 2: .2}},
            {"ref": {1: .3, 2: .5}}, stage="meta_fit")
        self.assertFalse(tied.enabled)
        for stage in ("development", "test"):
            with self.assertRaises(ValueError):
                r.meta_gate({1: .8}, {"a": {1: .1}, "b": {1: .2}}, {"ref": {1: .3}}, stage=stage)
        with self.assertRaises(ValueError):
            r.meta_gate({1: .8}, {"a": {1: .1}, "b": {2: .2}}, {"ref": {1: .3}}, stage="meta_fit")

    def test_closed_gate_does_not_read_training_inputs_or_invoke_any_fit(self):
        class Forbidden:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Closed gate read training data")
        def fail(_):
            raise AssertionError("Closed gate trained a model")
        result = r.fit_extension(Forbidden(), Forbidden(), gate=gate(False),
            selected_config={}, selected_epochs=100, seed=19, fit_reader=fail)
        self.assertIsNone(result.utility)
        self.assertEqual(result.protocol_summary()["status"], "gate_closed")
        self.assertEqual(result.crossfit.costs.fold_reader_fits, 0)

    def test_exact_full_catalog_loss_and_competitor_changes_determine_reward(self):
        allowed = np.array([False, True, True, True])
        targets = np.array([1, 3])
        base = np.array([np.nan, 0., 0., 0.])
        changed = np.array([np.nan, math.log(2), math.log(4), math.log(3)])
        expected = math.log(9)-(math.log(2)+math.log(3))/2
        self.assertAlmostEqual(r.full_catalog_loss(changed, targets, allowed), expected)
        self.assertAlmostEqual(r.loss_reduction(base, changed, targets, allowed), math.log(3)-expected)
        # The target logit increased, but its probability fell because its rival
        # increased more. A confidence-only reward would get the sign wrong.
        confidence_only_trap = np.array([np.nan, 1., 10., 0.])
        self.assertLess(r.loss_reduction(base, confidence_only_trap, np.array([1]), allowed), 0)
        self.assertAlmostEqual(r.full_catalog_loss(base+1e16, np.array([1]), allowed), math.log(3))
        with self.assertRaises(ValueError):
            r.full_catalog_loss(base, np.array([0]), allowed)

    def test_next_eight_actions_exclude_query_and_disallowed_rows_with_stable_ties(self):
        history = np.zeros((24, 4), dtype=np.uint8)
        history[:, 1:3] = 1
        context = np.array([0, 5, 0, 0], dtype=np.uint8)
        allowed = np.ones(24, dtype=bool)
        allowed[3] = False
        expected = tuple([i for i in range(24) if i not in (1, 3)][8:16])
        self.assertEqual(r.donor_actions(history, context, exclude_row=1, allowed_rows=allowed), expected)
        self.assertEqual(len(expected), 8)
        changed = history.copy()
        changed[1, :] = [0, 0, 0, 5]
        self.assertEqual(r.donor_actions(changed, context, exclude_row=1, allowed_rows=allowed), expected)

    def test_dedup_is_candidate_specific_and_no_action_when_all_predictions_nonpositive(self):
        extras = r.deduplicated_additions([3, 4, 5], [(1, 7), (2, 3), (7, 8)], 7)
        self.assertEqual(extras, {3: (), 4: (7,), 5: ()})
        self.assertIsNone(r.choose_action((7, 9), [0., -1.]))
        self.assertIsNone(r.choose_action((), []))
        self.assertEqual(r.choose_action((7, 9), [.1, .1]), 7)
        first = r.control_actions((7, 9, 12), seed=481, actual_reductions=[-.1, -.2, 0.])
        self.assertIsNone(first["oracle"])
        self.assertEqual(first, r.control_actions((7, 9, 12), seed=481, actual_reductions=[-.1, -.2, 0.]))

    def test_context_probe_has_no_candidate_edges_and_features_ignore_reader_head(self):
        history, episodes = fixture()
        bank, reader = EvidenceBank(history), GraphReader(seed=7)
        episode = episodes[0]
        actions = r.donor_actions(history, episode.context, exclude_row=0)
        batch = bank.build_context_query(episode.context, 0, actions)
        self.assertFalse(torch.isin(batch.candidate_idx, batch.edge_index).any())
        costs = r.Costs()
        values = r.context_action_features(reader, bank, episode.context, 0, actions, costs)
        self.assertEqual(values.shape, (8, 35))
        self.assertEqual(costs.context_graph_passes, 1)
        self.assertEqual(costs.reader_forward_passes, 0)
        for row, donor in enumerate(actions):
            np.testing.assert_array_equal(values[row, -3:], [np.count_nonzero(episode.context),
                np.count_nonzero(history[donor]), np.count_nonzero((episode.context > 0) & (history[donor] > 0))])
        with torch.no_grad():
            for parameter in reader.head.parameters():
                parameter.add_(100)
        repeated = r.context_action_features(reader, bank, episode.context, 0, actions, r.Costs())
        np.testing.assert_array_equal(values, repeated)
        self.assertTrue(all(not p.requires_grad for p in reader.parameters()))

    def test_global_action_scores_match_literal_all_candidate_graphs(self):
        history, episodes = fixture()
        bank, reader, episode = EvidenceBank(history), GraphReader(seed=8), episodes[0]
        action = r.donor_actions(history, episode.context, exclude_row=0)[0]
        costs = r.Costs()
        scores = r.score_catalog(reader, bank, episode.context, 0, action, costs, chunk_size=2)
        items = np.flatnonzero(r.eligible_items(episode.context))
        literal = bank.build_query(episode.context, 0, items, extra_donor={int(i): (action,) for i in items})
        with torch.inference_mode():
            expected = reader(literal).numpy()
        np.testing.assert_allclose(scores[items], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(costs.donor_insertions+costs.donor_already_present, len(items))
        self.assertEqual(costs.scored_candidates, len(items))
        observation = r.collect_query(reader, bank, episode, r.Costs())
        np.testing.assert_array_equal(observation.reductions, observation.base_loss-observation.action_losses)

    def test_crossfit_removes_whole_held_fold_from_reader_queries_and_bank(self):
        history, episodes = fixture(12)
        assignment = r.user_folds(12, seed=91)
        seen = []
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
        def fit(request):
            held = set(np.flatnonzero(assignment == request.fold))
            training = set(request.original_user_rows.tolist())
            self.assertFalse(held & training)
            self.assertEqual(training | held, set(range(12)))
            np.testing.assert_array_equal(request.categories, history[request.original_user_rows])
            self.assertEqual({e.user_row for e in request.episodes}, set(range(len(training))))
            self.assertEqual(request.epochs, 7)
            self.assertEqual(request.selected_config, {"seed": 91, "learning_rate": .001})
            seen.append((request.fold, training))
            return r.ReaderFit(Stub(), 7)
        def collect(reader, bank, episode, costs):
            fold, training = seen[-1]
            self.assertEqual(episode.user_row, -1)
            np.testing.assert_array_equal(bank.categories, history[sorted(training)])
            self.assertTrue(all(not p.requires_grad for p in reader.parameters()))
            return r.QueryObservations((0,), np.zeros((1, 35)), np.array([.2]), 1., np.array([.8]))
        result = r.crossfit_labels(history, episodes, gate=gate(), selected_config={"seed": 91, "learning_rate": .001},
            selected_epochs=7, seed=91, fit_reader=fit, query_collector=collect)
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(result.reductions), 12)
        self.assertEqual(result.costs.fold_reader_fits, 3)
        self.assertEqual(result.costs.declared_fold_reader_epochs, 21)
        self.assertTrue(all(row["held_users_excluded_from_training_and_donors"] for row in result.folds))
        for row in result.folds:
            self.assertEqual(row["reader_initialization_seed"], 91)
            self.assertEqual(row["training_opportunity"]["positive_oracle_queries"], row["held_episodes"])
            self.assertAlmostEqual(row["training_opportunity"]["mean_oracle"], .2)

    def test_crossfit_rejects_context_category_poison_and_incomplete_hidden_targets(self):
        history, episodes = fixture(6)
        original = episodes[0]
        wrong = original.context.copy()
        wrong[1] = 5 if wrong[1] != 5 else 1
        altered = [r.Episode(0, wrong, original.targets), *episodes[1:]]
        with self.assertRaisesRegex(ValueError, "preserve"):
            r.crossfit_labels(history, altered, gate=gate(), selected_config={}, selected_epochs=1,
                seed=1, fit_reader=lambda _: self.fail("must validate before fit"))
        context = original.context.copy()
        context[1] = 0
        with self.assertRaisesRegex(ValueError, "exactly all hidden"):
            r.crossfit_labels(history, [r.Episode(0, context, original.targets), *episodes[1:]], gate=gate(),
                selected_config={}, selected_epochs=1, seed=1, fit_reader=lambda _: self.fail("must validate before fit"))

    def test_fixed_utility_training_is_reproducible_frozen_and_serializable(self):
        rng = np.random.default_rng(84)
        features = rng.normal(size=(20, 35))
        labels = .2*features[:, 0]-.1*features[:, 1]
        data = r.CrossfitData(gate(), features, labels, ({}, {}, {}), r.Costs())
        first = r.fit_utility(data, seed=56)
        second = r.fit_utility(r.CrossfitData(gate(), features, labels, ({}, {}, {}), r.Costs()), seed=56)
        np.testing.assert_array_equal(first.predict(features), second.predict(features))
        np.testing.assert_allclose(first.mean, features.mean(axis=0), rtol=0, atol=0)
        self.assertEqual(data.costs.utility_training_steps, 100)
        self.assertEqual(data.costs.utility_training_examples, 2000)
        self.assertTrue(all(not p.requires_grad for p in first.network.parameters()))
        serialized = io.BytesIO()
        torch.save(first.checkpoint(), serialized)
        serialized.seek(0)
        loaded = r.UtilityModel.from_checkpoint(torch.load(serialized, weights_only=True))
        np.testing.assert_array_equal(first.predict(features), loaded.predict(features))
        bad = copy.deepcopy(first.checkpoint())
        bad["config"]["epochs"] = 101
        with self.assertRaises(ValueError):
            r.UtilityModel.from_checkpoint(bad)

    def test_deploy_never_trains_and_negative_predictions_match_no_action(self):
        history, episodes = fixture()
        reader, bank, episode = GraphReader(seed=9), EvidenceBank(history), episodes[0]
        class NegativePolicy:
            def predict(self, features, costs):
                costs.utility_prediction_examples += len(features)
                return -np.ones(len(features))
        before = {name: tensor.detach().clone() for name, tensor in reader.state_dict().items()}
        learned = r.deploy(reader, bank, episode.context, 0, NegativePolicy())
        fixed = r.deploy(reader, bank, episode.context, 0, None, policy="no_action")
        self.assertIsNone(learned["action"])
        np.testing.assert_array_equal(learned["scores"], fixed["scores"])
        self.assertEqual(learned["costs"].context_graph_passes, 1)
        self.assertEqual(fixed["costs"].context_graph_passes, 1)
        self.assertEqual(learned["costs"].utility_training_steps, 0)
        features = r.context_action_features(reader, bank, episode.context, 0, learned["actions"], r.Costs())
        summary = learned["feature_summary"]
        self.assertEqual(summary["count"], len(features))
        self.assertEqual(len(summary["sum"]), 35)
        self.assertEqual(len(summary["sum_squared"]), 35)
        self.assertTrue(np.isfinite(summary["sum"]).all())
        self.assertTrue(np.isfinite(summary["sum_squared"]).all())
        np.testing.assert_array_equal(summary["sum"], features.sum(axis=0))
        np.testing.assert_array_equal(summary["sum_squared"], np.square(features).sum(axis=0))
        self.assertEqual(summary, fixed["feature_summary"])
        for name, value in reader.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            r.deploy(reader, bank, episode.context, 0, None, policy="oracle")


if __name__ == "__main__":
    unittest.main()
