"""Synthetic evidence/learning invariants; no datasets or outcome files read."""
import io
import unittest

import numpy as np
from scipy import sparse
import torch

from exploratory.evidence_transfer.model import (
    FEATURE_NAMES, FeatureNormalizer, SharedEvidenceScorer, extract_features,
    fit_normalizer, masked_listwise_loss, prepare_donors,
)


def literal_features(donors, query, excluded):
    """Independent donor/candidate loops instead of sparse matrix products."""
    result = np.zeros((len(query), donors.shape[1], 9))
    for row, context in enumerate(query):
        valid = [v for v in range(len(donors)) if v != excluded[row]]
        h = context.sum()
        weights, patterns = {}, {}
        for v in valid:
            intersection = context * donors[v]
            overlap, donor_length = intersection.sum(), donors[v].sum()
            weights[v] = overlap**2 / (h * donor_length) if h * donor_length else 0.
            patterns[v] = intersection / np.sqrt(overlap) if overlap else intersection
        total_weight = sum(weights.values())
        for item in range(donors.shape[1]):
            supporting = [v for v in valid if donors[v, item]]
            count = len(supporting)
            mass = sum(weights[v] for v in supporting)
            mass2 = sum(weights[v]**2 for v in supporting)
            pattern = sum((weights[v] * patterns[v] for v in supporting), np.zeros(donors.shape[1]))
            norm2 = sum(float(value)**2 for value in pattern)
            result[row, item] = [
                np.log1p(h), np.log1p(count), count / len(valid) if valid else 0,
                np.log1p(mass), mass / total_weight if total_weight else 0,
                mass / count if count else 0,
                np.log1p(mass**2 / mass2) if mass2 else 0,
                np.log1p(mass**2 / norm2) if norm2 else 0,
                np.count_nonzero(pattern) / h if h else 0,
            ]
    return result


class EvidenceFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(190)
        self.donors = (rng.random((11, 8)) < .45).astype(float)
        self.contexts = (rng.random((4, 8)) < .4).astype(float)
        self.exclusions = np.array([2, -1, 5, 0])

    def test_sparse_features_match_literal_donor_loops(self):
        actual = extract_features(prepare_donors(self.donors), self.contexts, self.exclusions)
        expected = literal_features(self.donors, self.contexts, self.exclusions)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (4, 8, len(FEATURE_NAMES)))
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)

    def test_excluded_complete_row_has_no_effect_on_any_feature(self):
        context = self.contexts[:1]
        expected = extract_features(prepare_donors(self.donors), context, np.array([2]))
        for replacement in (np.zeros(8), np.ones(8), self.donors[5]):
            mutated = self.donors.copy()
            mutated[2] = replacement
            np.testing.assert_array_equal(
                extract_features(prepare_donors(mutated), context, np.array([2])), expected,
            )

    def test_donor_and_item_permutations_and_batch_chunking(self):
        bank = prepare_donors(self.donors)
        expected = extract_features(bank, self.contexts, self.exclusions)
        donor_order = np.array([2, 7, 5, 1, 8, 9, 0, 6, 3, 4, 10])
        inv_order = np.argsort(donor_order)
        exclusions = np.array([inv_order[v] if v >= 0 else -1 for v in self.exclusions])
        permuted = extract_features(prepare_donors(self.donors[donor_order]), self.contexts, exclusions)
        np.testing.assert_allclose(permuted, expected, atol=2e-7, rtol=2e-7)
        items = np.array([5, 0, 3, 1, 7, 4, 2, 6])
        permuted = extract_features(prepare_donors(self.donors[:, items]), self.contexts[:, items], self.exclusions)
        np.testing.assert_allclose(permuted, expected[:, items], atol=2e-7, rtol=2e-7)
        chunks = np.concatenate([
            extract_features(bank, self.contexts[i:i+1], self.exclusions[i:i+1])
            for i in range(len(self.contexts))
        ])
        np.testing.assert_array_equal(chunks, expected)
        np.testing.assert_array_equal(
            extract_features(prepare_donors(sparse.csr_matrix(self.donors)),
                             sparse.csr_matrix(self.contexts), self.exclusions), expected,
        )

    def test_geometry_identical_orthogonal_and_whole_pool_replication(self):
        query = np.array([[1, 1, 0, 0]], dtype=float)
        identical = np.array([[1, 0, 1, 0]] * 3, dtype=float)
        features = extract_features(prepare_donors(identical), query, np.array([-1]))[0, 2]
        self.assertAlmostEqual(np.expm1(features[6]), 3, places=6)
        self.assertAlmostEqual(np.expm1(features[7]), 1, places=6)
        self.assertAlmostEqual(features[8], .5)
        orthogonal = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=float)
        features = extract_features(prepare_donors(orthogonal), query, np.array([-1]))[0, 2]
        self.assertAlmostEqual(np.expm1(features[6]), 2, places=6)
        self.assertAlmostEqual(np.expm1(features[7]), 2, places=6)
        self.assertEqual(features[8], 1)
        mixed = np.vstack([orthogonal, [1, 1, 1, 1], orthogonal[0]])
        original = extract_features(prepare_donors(mixed), query, np.array([-1]))
        doubled = extract_features(prepare_donors(np.tile(mixed, (2, 1))), query, np.array([-1]))
        np.testing.assert_allclose(doubled[..., 7:9], original[..., 7:9], atol=2e-7)
        # Replicating just one pattern changes its mixture weight; it is not a
        # promised invariant. The statistic is geometric diversity, not people.

    def test_tiny_nonzero_weight_does_not_shrink_effective_counts(self):
        query = np.zeros((1, 1501))
        query[0, :500] = 1
        donor = np.zeros((1, 1501))
        donor[0, 0] = 1
        donor[0, 500:1499] = 1
        features = extract_features(prepare_donors(donor), query, np.array([-1]))[0, 600]
        self.assertAlmostEqual(np.expm1(features[6]), 1., places=6)
        self.assertAlmostEqual(np.expm1(features[7]), 1., places=6)
        self.assertGreater(features[3], 0)

    def test_empty_context_bank_and_total_exclusion_are_finite(self):
        cases = [
            (self.donors, np.zeros((1, 8)), [-1]),
            (np.zeros((3, 8)), np.ones((1, 8)), [-1]),
            (np.zeros((0, 8)), np.ones((1, 8)), [-1]),
            (np.ones((1, 8)), np.ones((1, 8)), [0]),
        ]
        for donors, query, excluded in cases:
            value = extract_features(prepare_donors(donors), query, np.array(excluded))
            self.assertTrue(np.isfinite(value).all())
            np.testing.assert_array_equal(value[..., 3:], 0)
        value = extract_features(prepare_donors(np.ones((1, 8))), np.ones((1, 8)), np.array([0]))
        np.testing.assert_array_equal(value[..., 1:], 0)

    def test_rejects_invalid_binary_sparse_duplicates_and_exclusions(self):
        malformed = [np.array([[.5]]), np.array([[np.nan]]), np.array([1, 0]),
                     sparse.coo_matrix(([.5, .5], ([0, 0], [0, 0])), shape=(1, 2)),
                     sparse.coo_matrix(([1., 1.], ([0, 0], [0, 0])), shape=(1, 2)),
                     sparse.coo_matrix(([1., -1.], ([0, 0], [0, 0])), shape=(1, 2))]
        for value in malformed:
            with self.assertRaises(ValueError):
                prepare_donors(value)
        bank = prepare_donors(self.donors)
        for excluded in (np.array([11] * 4), np.array([-2] * 4), np.zeros(4), np.array([0])):
            with self.assertRaises(ValueError):
                extract_features(bank, self.contexts, excluded)


class SharedScorerTests(unittest.TestCase):
    def test_normalizer_uses_eligible_positions_only(self):
        rng = np.random.default_rng(37)
        values = rng.normal(size=(3, 5, 9)).astype(np.float32)
        eligible = rng.random((3, 5)) < .7
        values[..., 0] = 3
        normalizer = fit_normalizer(values, eligible)
        poisoned = values.copy()
        poisoned[~eligible] = np.nan
        same = fit_normalizer(poisoned, eligible)
        np.testing.assert_array_equal(same.mean, normalizer.mean)
        np.testing.assert_array_equal(same.scale, normalizer.scale)
        transformed = normalizer.transform(values)[eligible]
        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(transformed[:, 1:].std(axis=0), 1, atol=1e-7)
        self.assertEqual(normalizer.scale[0], 1)
        with self.assertRaises(ValueError):
            fit_normalizer(values, np.zeros_like(eligible))

    def test_matched_initialization_and_exact_feature_intervention(self):
        full = SharedEvidenceScorer(seed=219)
        no_pattern = SharedEvidenceScorer(variant="no_pattern", seed=219)
        marginal = SharedEvidenceScorer(variant="marginal_only", seed=219)
        for name, values in full.state_dict().items():
            torch.testing.assert_close(values, no_pattern.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(values, marginal.state_dict()[name], rtol=0, atol=0)
        features = torch.randn(2, 7, 9)
        without_pattern = features.clone()
        without_pattern[..., 7:] = 0
        without_personalized = features.clone()
        without_personalized[..., 3:] = 0
        torch.testing.assert_close(no_pattern(features), full(without_pattern), rtol=0, atol=0)
        torch.testing.assert_close(marginal(features), full(without_personalized), rtol=0, atol=0)
        with torch.no_grad():
            next(full.parameters()).add_(.1)
        no_pattern.load_state_dict(full.state_dict())
        torch.testing.assert_close(no_pattern(features), full(without_pattern), rtol=0, atol=0)
        self.assertNotIn("feature_mask", full.state_dict())

    def test_masked_loss_macro_average_and_no_excluded_gradient(self):
        logits = torch.tensor([[1., 2., 3., 4.], [.5, -.5, 2., 1.]], dtype=torch.float64, requires_grad=True)
        eligible = torch.tensor([[True, True, False, True], [False, True, True, True]])
        positive = torch.tensor([[True, True, False, False], [False, False, True, False]])
        loss = masked_listwise_loss(logits, positive, eligible)
        expected = .5 * (torch.logsumexp(logits[0, [0, 1, 3]], 0) - 1.5
                         + torch.logsumexp(logits[1, [1, 2, 3]], 0) - 2)
        torch.testing.assert_close(loss, expected)
        altered = logits.detach().clone()
        altered[~eligible] = float("nan")
        torch.testing.assert_close(masked_listwise_loss(altered, positive, eligible), loss)
        loss.backward()
        torch.testing.assert_close(logits.grad[~eligible], torch.zeros(2, dtype=torch.float64))
        invalid = positive.clone()
        invalid[0, 2] = True
        with self.assertRaises(ValueError):
            masked_listwise_loss(logits, invalid, eligible)

    def test_train_gradient_and_exact_serialized_replay(self):
        rng = np.random.default_rng(928)
        features = rng.normal(size=(3, 8, 9)).astype(np.float32)
        eligible = np.ones((3, 8), dtype=bool)
        eligible[:, 0] = False
        normalizer = fit_normalizer(features, eligible)
        x = torch.from_numpy(normalizer.transform(features))
        model = SharedEvidenceScorer(seed=432)
        targets = torch.zeros((3, 8), dtype=torch.bool)
        targets[0, [2, 3]] = True
        targets[1, 4] = True
        targets[2, [1, 6, 7]] = True
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss = masked_listwise_loss(model(x), targets, torch.from_numpy(eligible))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        optimizer.step()
        expected = model(x).detach()
        checkpoint = io.BytesIO()
        torch.save({"config": model.config, "state": model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        replay = SharedEvidenceScorer(**saved["config"])
        replay.load_state_dict(saved["state"])
        norm_file = io.BytesIO()
        np.savez(norm_file, mean=normalizer.mean, scale=normalizer.scale)
        norm_file.seek(0)
        with np.load(norm_file, allow_pickle=False) as norm:
            replay_norm = FeatureNormalizer(norm["mean"], norm["scale"])
        torch.testing.assert_close(replay(torch.from_numpy(replay_norm.transform(features))), expected, rtol=0, atol=0)

    def test_model_initialization_preserves_global_rng(self):
        torch.manual_seed(194)
        before = torch.random.get_rng_state()
        SharedEvidenceScorer(seed=2)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
