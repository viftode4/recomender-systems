import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from categorical_experiment import array_digest
from field_reference_comparison import identity_digest
from study import digest, partition_users
from exploratory.conditional_evidence import data


class DataTests(unittest.TestCase):
    def fixture(self, root, seed=2026, bad_train=None, missing_train=False):
        source_root, categorical_root = root / "source", root / "categorical"
        source, cohort = source_root / f"{seed}-EASE-1", categorical_root / str(seed)
        source.mkdir(parents=True)
        cohort.mkdir(parents=True)
        users, items = ["a", "b", "c", "d"], ["[PAD]", "one", "two", "three", "four"]
        train = [(u, item) for u in users for item in ("one", "two")]
        meta, dev = partition_users(users, seed)
        # DEV's malformed item payload must remain uninterpreted during selection.
        valid = "user_id\titem_id\n" + "".join(f"{u}\t" + ("three" if u in meta else "POISON\tNO_ITEM") + "\n" for u in users)
        source.joinpath("valid.tsv").write_text(valid)
        cohort.joinpath("valid.tsv").write_text(valid)
        train_text = "user_id\titem_id\n" + "".join(f"{u}\t{i}\n" for u, i in train)
        source.joinpath("train.tsv").write_text(train_text)
        cohort.joinpath("train.tsv").write_text(train_text)
        source.joinpath("ids.json").write_text(json.dumps({"users": ["[PAD]"] + users, "items": items, "padding_index": 0}))
        ratings = root / "ratings.inter"
        rows = ["user_id:token\titem_id:token\trating:float\ttimestamp:float"]
        for index, (user, item) in enumerate(train):
            if missing_train and index == 0:
                continue
            value = bad_train if bad_train is not None and index == 0 else (1 if item == "one" else 5)
            rows.append(f"{user}\t{item}\t{value}\tTRAIN-TIME-NEVER-PARSED")
        rows.extend(f"{u}\tthree\tPOISON-RATING\tPOISON-TIME" for u in users)
        ratings.write_text("\n".join(rows) + "\n")
        manifest = {"status": "complete", "test_evaluated": False, "test_read": False,
                    "validation_used_for_training": False, "data_sha256": {ratings.name: digest(ratings)},
                    "split_sha256": {part: digest(source / f"{part}.tsv") for part in ("train", "valid")}}
        source.joinpath("manifest.json").write_text(json.dumps(manifest))
        cohort.joinpath("cohorts.json").write_text(json.dumps({"meta_fit": sorted(meta), "development": sorted(dev)}))
        categories = np.zeros((4, 5), dtype=np.int64)
        categories[:, 1], categories[:, 2] = 1, 5
        signature = {"source_manifest_sha256": digest(source / "manifest.json"),
            "data_sha256": manifest["data_sha256"], "split_sha256": manifest["split_sha256"],
            "ids_sha256": digest(source / "ids.json"), "ordered_identity_sha256": identity_digest(users, items),
            "cohort_file_sha256": digest(cohort / "cohorts.json"), "training_categories_sha256": array_digest(categories)}
        cohort.joinpath("input-signature.json").write_text(json.dumps(signature))
        # A poisoned cache cannot enter the training input path.
        cohort.joinpath("training-categories.npz").write_text("NOT AN ARRAY; MUST NOT BE OPENED")
        return {"source_root": source_root, "categorical_root": categorical_root, "ratings": ratings}, meta, dev

    def test_meta_loading_ignores_poisoned_development_items_values_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            args, meta, dev = self.fixture(Path(folder))
            inputs = data.load_inputs(2026, **args)
            self.assertEqual(inputs["meta_truth"], {u: {"three"} for u in meta})
            self.assertEqual(set(inputs["meta_users"]), meta)
            self.assertEqual(set(inputs["dev_users"]), dev)
            np.testing.assert_array_equal(inputs["categories"][:, 1:3], [[1, 5]] * 4)
            self.assertFalse(inputs["categories"].flags.writeable)
            with self.assertRaises(ValueError):
                data.load_development(inputs, "unused", verify_global_seal=lambda _: True)

    def test_train_only_benchmark_never_opens_validation_pair_files(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = self.fixture(Path(folder))
            original = Path.open
            def guarded(path, *a, **kw):
                if path.name == "valid.tsv":
                    raise AssertionError("Benchmark opened validation")
                return original(path, *a, **kw)
            with patch.object(Path, "open", guarded):
                inputs = data.load_inputs(2026, load_meta=False, **args)
            self.assertEqual(inputs["meta_validpairs"], [])
            self.assertFalse(inputs["meta_loaded"])

    def test_development_verifier_runs_before_any_file_open(self):
        class Stop(Exception):
            pass
        def deny(path):
            self.assertEqual(path, "all-seed-seal")
            raise Stop()
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected file access")):
            with self.assertRaises(Stop):
                data.load_development({}, "all-seed-seal", verify_global_seal=deny)
            with self.assertRaises(ValueError):
                data.load_development({}, "all-seed-seal", verify_global_seal=lambda _: False)

    def test_missing_and_nonfinite_train_categories_are_rejected(self):
        for value, missing in [("nan", False), ("inf", False), ("2.5", False), (None, True)]:
            with self.subTest(value=value, missing=missing), tempfile.TemporaryDirectory() as folder:
                args, _, _ = self.fixture(Path(folder), bad_train=value, missing_train=missing)
                with self.assertRaises(ValueError):
                    data.load_inputs(2026, **args)

    def test_paired_episodes_exclude_targets_and_use_complete_train_absent_pool(self):
        matrix = np.zeros((3, 301), dtype=np.uint8)
        matrix[0, 1:101] = np.tile(np.arange(1, 6), 20)
        matrix[1, [2, 5, 8]] = [1, 3, 5]
        matrix[2, 20] = 4
        before = matrix.copy()
        episodes = list(data.episodes(matrix, 2, (.8, .9), 123))
        self.assertEqual(len(episodes), 4)
        self.assertEqual([e["retain_fraction"] for e in episodes], [.8, .9, .8, .9])
        self.assertEqual(data.episode_status(matrix)["skipped_tiny_history_users"], 1)
        for episode in episodes:
            row = matrix[episode["user_index"]]
            candidates, targets = episode["candidate_ids"], episode["target_mask"]
            context = episode["context"]
            self.assertEqual(set(candidates[targets]), set(np.flatnonzero(row)) - set(np.flatnonzero(context)))
            self.assertTrue(np.all(row[candidates[~targets]] == 0))
            self.assertTrue(np.all(context[candidates] == 0))
            self.assertEqual(len(set(candidates)), len(candidates))
            self.assertNotIn(0, candidates)
            self.assertEqual(int(episode["sampled_alternatives"]), 128)
            self.assertEqual(episode["unobserved_population_count"], 300 - np.count_nonzero(row))
        episodes[0]["context"][:] = 0
        np.testing.assert_array_equal(matrix, before)
        self.assertEqual(data.episodes_signature(matrix, 2, (.8, .9), 123), data.episodes_signature(matrix, 2, (.8, .9), 123))
        self.assertNotEqual(data.episodes_signature(matrix, 2, (.8, .9), 123), data.episodes_signature(matrix, 3, (.8, .9), 123))

    def test_exhaustive_loss_and_gradient_match_finite_catalog(self):
        logits = torch.tensor([-.3, 1.2, 0., 2.1, -.8], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([True, True, False, False, False])
        actual = data.sampled_multinomial_loss(logits, target, 3)
        expected = torch.logsumexp(logits, dim=0) - logits[:2].mean()
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits)[0], rtol=0, atol=1e-15)

    def test_sample_correction_and_equal_user_weighting(self):
        scores = torch.tensor([[2., 1., -.5], [1., -.2, .7]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        batch = data.sampled_multinomial_loss(scores, mask, torch.tensor([7., 8.]))
        expected = torch.stack([torch.log(torch.exp(scores[0, :2]).sum() + 7 * torch.exp(scores[0, 2])) - scores[0, :2].mean(),
                                torch.log(torch.exp(scores[1, 0]) + 4 * torch.exp(scores[1, 1:]).sum()) - scores[1, 0]]).mean()
        torch.testing.assert_close(batch, expected, rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            data.sampled_multinomial_loss(torch.tensor([float("inf"), 0.]), torch.tensor([True, False]), 1)
        with self.assertRaises(ValueError):
            data.sampled_multinomial_loss(scores[0], mask[0], 0)


if __name__ == "__main__":
    unittest.main()
