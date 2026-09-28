"""Independent feature algebra and leakage tests on synthetic histories only."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch

from exploratory.evidence_transfer.model import (
    prepare_donors, extract_features, fit_normalizer, SharedEvidenceScorer, masked_listwise_loss)


def literal_features(donors, contexts, exclusions):
    """Scalar donor loops and an explicit cosine Gram, independent of batching."""
    result = np.zeros((len(contexts), donors.shape[1], 9), dtype=float)
    for query_index, (query, excluded) in enumerate(zip(contexts, exclusions)):
        context = set(np.flatnonzero(query))
        active = [row for row in range(len(donors)) if row != excluded]
        weights, patterns = {}, {}
        for row in active:
            history = set(np.flatnonzero(donors[row]))
            overlap = context & history
            weight = len(overlap)**2/(len(context)*len(history)) if context and history else 0.
            weights[row] = weight
            patterns[row] = {item:1/math.sqrt(len(overlap)) for item in overlap} if overlap else {}
        total_weight = sum(weights.values())
        for item in range(donors.shape[1]):
            supporters = [row for row in active if donors[row,item]]
            mass = sum(weights[row] for row in supporters)
            squared_mass = sum(weights[row]**2 for row in supporters)
            gram_mass = 0.
            for first in supporters:
                for second in supporters:
                    cosine = sum(value*patterns[second].get(source,0.) for source,value in patterns[first].items())
                    gram_mass += weights[first]*weights[second]*cosine
            covered = set().union(*(set(patterns[row]) for row in supporters if weights[row]>0))
            kish = mass*mass/squared_mass if squared_mass else 0.
            pattern = mass*mass/gram_mass if gram_mass else 0.
            result[query_index,item] = [math.log1p(len(context)),math.log1p(len(supporters)),
                len(supporters)/len(active) if active else 0.,math.log1p(mass),
                mass/total_weight if total_weight else 0.,mass/len(supporters) if supporters else 0.,
                math.log1p(kish),math.log1p(pattern),len(covered)/len(context) if context else 0.]
    return result


class IndependentEvidenceTests(unittest.TestCase):
    def test_nine_channels_equal_literal_donor_and_cosine_gram_loops(self):
        rng = np.random.default_rng(622)
        donors = rng.integers(0,2,size=(7,13))
        donors[0] = 0
        contexts = rng.integers(0,2,size=(4,13))
        contexts[0] = 0
        exclusions = np.array([-1,0,3,6])
        actual = extract_features(prepare_donors(donors),contexts,exclusions)
        expected = literal_features(donors,contexts,exclusions)
        self.assertEqual(actual.shape,(4,13,9))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual[contexts==0],expected[contexts==0],atol=3e-7,rtol=3e-6)

    def test_excluded_entire_donor_row_cannot_change_any_feature(self):
        rng = np.random.default_rng(623)
        donors = rng.integers(0,2,size=(6,17))
        contexts = rng.integers(0,2,size=(1,17))
        before = extract_features(prepare_donors(donors),contexts,np.array([2]))
        for replacement in (np.zeros(17),np.ones(17),rng.integers(0,2,size=17)):
            changed = donors.copy(); changed[2] = replacement
            after = extract_features(prepare_donors(changed),contexts,np.array([2]))
            np.testing.assert_array_equal(before[:,contexts[0]==0],after[:,contexts[0]==0])

    def test_duplicate_orthogonal_and_mixed_pattern_geometry(self):
        query = np.array([[1,1,0]])
        cases = (([[1,0,1],[1,0,1]],2.,1.,.5),
                 ([[1,0,1],[0,1,1]],2.,2.,1.),
                 ([[1,0,1],[1,0,1],[0,1,1]],3.,1.8,1.))
        for rows,kish,pattern,coverage in cases:
            with self.subTest(rows=rows):
                donors = np.array(rows)
                feature = extract_features(prepare_donors(donors),query,np.array([-1]))[0,2]
                np.testing.assert_allclose(feature[6:],[np.log1p(kish),np.log1p(pattern),coverage],atol=2e-7)
                repeated = extract_features(prepare_donors(np.tile(donors,(3,1))),query,np.array([-1]))[0,2]
                self.assertAlmostEqual(float(feature[7]),float(repeated[7]),places=6)

    def test_weak_nonzero_support_retains_unit_effective_counts(self):
        query = np.zeros((1,1503)); query[0,1:501] = 1
        donors = np.zeros((1,1503)); donors[0,1] = 1; donors[0,501:1500] = 1
        feature = extract_features(prepare_donors(donors),query,np.array([-1]))[0,501]
        self.assertGreater(feature[3],0)
        np.testing.assert_allclose(feature[6:8],[np.log(2),np.log(2)],atol=2e-7)
        np.testing.assert_allclose(feature,literal_features(donors,query,[-1])[0,501],atol=2e-7,rtol=2e-6)

    def test_geometric_bounds_and_empty_evidence(self):
        rng = np.random.default_rng(624)
        donors = rng.integers(0,2,size=(11,23))
        queries = rng.integers(0,2,size=(5,23))
        queries[0] = 0
        features = extract_features(prepare_donors(donors),queries,np.array([-1,0,3,6,10]))
        for query_index,item in zip(*np.where(queries==0)):
            row = features[query_index,item]
            supporter_count,kish,pattern = np.expm1(row[[1,6,7]])
            self.assertLessEqual(pattern,kish+2e-5)
            self.assertLessEqual(kish,supporter_count+2e-5)
            self.assertTrue(0 <= row[8] <= 1)
            if row[3]>0:
                self.assertGreaterEqual(pattern,1-2e-5)
            else:
                np.testing.assert_array_equal(row[3:],[0,0,0,0,0,0])
        alone = extract_features(prepare_donors(np.ones((1,4))),np.array([[1,0,0,0]]),np.array([0]))
        np.testing.assert_array_equal(alone[0,1:,1:],0)
        self.assertTrue(np.isfinite(alone).all())

    def test_user_item_permutations_sparse_input_and_batch_partition(self):
        rng = np.random.default_rng(625)
        donors = rng.integers(0,2,size=(6,19)); queries = rng.integers(0,2,size=(4,19))
        excluded = np.array([-1,1,4,0])
        original = extract_features(prepare_donors(donors),queries,excluded)
        donor_order,item_order = rng.permutation(6),rng.permutation(19)
        inverse_donors = np.argsort(donor_order)
        changed_excluded = np.array([inverse_donors[index] if index>=0 else -1 for index in excluded])
        changed = extract_features(prepare_donors(donors[donor_order][:,item_order]),queries[:,item_order],changed_excluded)
        np.testing.assert_allclose(changed,original[:,item_order],atol=4e-7,rtol=3e-6)
        bank = prepare_donors(sparse.csr_matrix(donors))
        sparse_result = extract_features(bank,sparse.csr_matrix(queries),excluded)
        np.testing.assert_array_equal(sparse_result,original)
        pieces = np.concatenate([extract_features(bank,queries[index:index+1],excluded[index:index+1])
                                 for index in range(len(queries))])
        np.testing.assert_array_equal(pieces,original)

    def test_loss_is_equal_episode_multinomial_and_context_logits_are_inert(self):
        values = np.array([[12.,.2,-.7,1.3,-4.],[13.,.5,1.1,-.3,.8]])
        eligible = np.array([[False,True,True,True,False],[False,True,True,True,True]])
        positives = np.array([[False,True,False,False,False],[False,True,False,True,True]])
        terms = []
        for score,allowed,target in zip(values,eligible,positives):
            largest = score[allowed].max()
            terms.append(largest+math.log(sum(math.exp(value-largest) for value in score[allowed]))-score[target].mean())
        tensor = torch.tensor(values,dtype=torch.float64,requires_grad=True)
        loss = masked_listwise_loss(tensor,torch.tensor(positives),torch.tensor(eligible))
        self.assertAlmostEqual(float(loss.detach()),sum(terms)/len(terms),places=13)
        loss.backward()
        np.testing.assert_array_equal(tensor.grad.detach().numpy()[~eligible],0)
        np.testing.assert_allclose(tensor.grad.detach().numpy().sum(axis=1),0,atol=1e-15)
        poison = values.copy(); poison[~eligible] = np.nan
        poisoned_loss = masked_listwise_loss(torch.tensor(poison),torch.tensor(positives),torch.tensor(eligible))
        self.assertEqual(float(loss.detach()),float(poisoned_loss))
        invalid = positives.copy();invalid[0,0] = True
        with self.assertRaises(ValueError):
            masked_listwise_loss(tensor,torch.tensor(invalid),torch.tensor(eligible))

    def test_normalizer_ignores_ineligible_values_and_control_masks_follow_scaling(self):
        features = np.random.default_rng(626).normal(size=(4,7,9)).astype(np.float32)
        mask = np.ones((4,7),dtype=bool);mask[:,[0,3]] = False
        normalizer = fit_normalizer(features,mask)
        changed = features.copy();changed[~mask] = np.nan
        other = fit_normalizer(changed,mask)
        np.testing.assert_array_equal(normalizer.mean,other.mean)
        np.testing.assert_array_equal(normalizer.scale,other.scale)
        standardized = normalizer.transform(features)
        full = SharedEvidenceScorer(variant="full_pattern",seed=331)
        control = SharedEvidenceScorer(variant="no_pattern",seed=331)
        for key,value in full.state_dict().items():
            torch.testing.assert_close(value,control.state_dict()[key],atol=0,rtol=0)
        intervention = standardized.copy();intervention[:,:,[7,8]] = 0
        torch.testing.assert_close(control(torch.tensor(standardized)),full(torch.tensor(intervention)),atol=0,rtol=0)
        # Loading a full checkpoint must retain the declared intervention mask.
        control.load_state_dict(full.state_dict())
        torch.testing.assert_close(control(torch.tensor(standardized)),full(torch.tensor(intervention)),atol=0,rtol=0)

    def test_episode_targets_are_exactly_the_hidden_train_records(self):
        from exploratory.evidence_transfer.run_experiment import make_episodes
        history = np.zeros((5,13),dtype=bool)
        for row,count in enumerate((0,1,2,5,11)):
            history[row,1:count+1] = True
        episodes = make_episodes(history,801)
        np.testing.assert_array_equal(episodes["excluded_training_rows"],[0,1])
        np.testing.assert_array_equal(episodes["exclude_rows"],[2,3,4,2,3,4])
        for index,owner in enumerate(episodes["exclude_rows"]):
            context,target = episodes["contexts"][index],episodes["targets"][index]
            self.assertFalse(np.any(context & target))
            np.testing.assert_array_equal(context|target,history[owner])
            self.assertGreater(context.sum(),0);self.assertGreater(target.sum(),0)
            allowed = ~context;allowed[0] = False
            np.testing.assert_array_equal(episodes["eligible"][index],allowed)
            self.assertFalse(np.any(target & ~allowed))

    def test_auditor_rebuilds_episode_scaling_and_rejects_corrupted_feature_hash(self):
        from exploratory.evidence_transfer import run_experiment as runner
        from exploratory.evidence_transfer import verify_results as verifier
        history = np.random.default_rng(627).integers(0,2,(7,18))
        history[:,0] = 0
        episodes,_,query,raw,normalizer,hashes = runner.prepare_inputs(history,813)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez_compressed(directory/"episodes.npz",**episodes)
            np.savez_compressed(directory/"normalizer.npz",mean=normalizer.mean,scale=normalizer.scale)
            hashes["feature_names"] = list(verifier.FEATURES)
            verifier.json_write(directory/"feature-hashes.json",hashes)
            actual,analytic,checks = verifier.rebuild_features(directory,{"matrix":history},813)
            np.testing.assert_array_equal(actual,query)
            np.testing.assert_array_equal(analytic,raw[:,:,4])
            self.assertTrue(checks["normalization_replay_exact"])
            hashes["raw_query_features_sha256"] = "corrupted"
            verifier.json_write(directory/"feature-hashes.json",hashes)
            with self.assertRaisesRegex(ValueError,"Raw feature replay differs"):
                verifier.rebuild_features(directory,{"matrix":history},813)

    def test_auditor_explicit_network_replays_control_and_intervention_checkpoints(self):
        from exploratory.evidence_transfer import run_experiment as runner
        from exploratory.evidence_transfer import verify_results as verifier
        features = np.random.default_rng(628).normal(size=(67,13,9)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            for arm in ("full_pattern","no_pattern","marginal_only"):
                model = SharedEvidenceScorer(variant=arm,seed=814+42001)
                path = Path(temporary)/(arm+".pt")
                torch.save({"config":model.config,"state_dict":model.state_dict()},path)
                actual = verifier.checkpoint_scores(path,features,814,arm)
                np.testing.assert_array_equal(actual,runner.score_model(model,features))
                if arm=="full_pattern":
                    actual = verifier.checkpoint_scores(path,features,814,arm,"no_pattern")
                    np.testing.assert_array_equal(actual,runner.score_model(runner.restore_model(path,"no_pattern"),features))
                with self.assertRaisesRegex(ValueError,"configuration differs"):
                    verifier.checkpoint_scores(path,features,815,arm)


if __name__ == "__main__":
    unittest.main()
