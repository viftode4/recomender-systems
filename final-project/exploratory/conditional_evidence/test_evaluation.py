import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from exploratory.conditional_evidence import evaluation as e
from metrics import evaluate as independent_evaluate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.users = ["alice", "bob", "carol"]
        self.items = ["[PAD]", "a", "b", "c", "d", "e", "f", "g"]
        self.train = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
        self.scores = np.array([[100, 99, 6, 5, 4, 3, 2, 1],
                                [100, 99, 98, 6, 5, 4, 3, 2],
                                [100, 99, 98, 97, 0, 0, 0, 0]], dtype=float)
        self.truth = {"alice": {"b", "e"}, "bob": {"d", "f"}, "carol": {"g"}}

    def test_metrics_independent_replay_and_full_mrr(self):
        result = e.evaluate_scores(self.scores, self.users, self.items, self.train, self.truth, self.users, k=3)
        recs = {"alice": ["b", "c", "d"], "bob": ["c", "d", "e"], "carol": ["d", "e", "f"]}
        history = {user: {self.items[i] for i in np.flatnonzero(self.train[row])} for row, user in enumerate(self.users)}
        counts = {item: int(self.train[:, i].sum()) for i, item in enumerate(self.items) if i}
        independent = independent_evaluate(recs, self.truth, history, self.items[1:], counts, 3)
        for metric, value in independent["aggregate"].items():
            self.assertAlmostEqual(value, result["public"]["all_observed"][metric], places=14)
        carol = result["per_user"]["all_observed"]["carol"]
        self.assertEqual(carol["mrr@3"], 0)
        self.assertEqual(carol["mrr_full_catalog"], .25)
        self.assertEqual(result["public"]["denominators"]["all_observed_positive_pairs"], 5)

    def test_liked_denominators_groups_and_no_ids(self):
        ratings = {("alice", "b"): 1, ("alice", "e"): 5,
                   ("bob", "d"): 4, ("bob", "f"): 4, ("carol", "g"): 2}
        public = e.evaluate_scores(self.scores, self.users, self.items, self.train, self.truth,
                                   self.users, ratings, k=3)["public"]
        self.assertEqual(public["denominators"]["liked_users"], 2)
        self.assertEqual(public["denominators"]["liked_positive_pairs"], 3)
        self.assertEqual(public["known_low_rating_hits_per_slot"], 1 / 9)
        groups = public["groups"]["all_observed"]
        self.assertEqual([groups["activity"][name]["users"] for name in e.ACTIVITY_NAMES], [1, 1, 1])
        self.assertEqual(groups["items"]["head"]["positive_pairs"], 1)
        self.assertEqual(groups["items"]["tail"]["positive_pairs"], 4)
        self.assertEqual(sum(row["recommendation_slots"] for row in groups["items"].values()), 9)
        text = json.dumps(public, allow_nan=False)
        for user in self.users:
            self.assertNotIn(user, text)

    def test_other_cohort_poison_never_examined(self):
        truth = {"alice": self.truth["alice"], "bob": object(), "carol": object()}
        ratings = {("alice", "b"): 1, ("alice", "e"): 2, ("bob", "invalid"): object()}
        result = e.evaluate_scores(self.scores, self.users, self.items, self.train, truth, ["alice"], ratings, k=3)
        self.assertIsNone(result["public"]["liked_ratings"])
        self.assertEqual(result["public"]["denominators"]["liked_users"], 0)
        self.assertEqual(e.meta_ndcg(self.scores, self.users, self.items, self.train, truth, ["alice"], k=3),
                         result["public"]["all_observed"]["ndcg@3"])

    def test_invalid_or_seen_truth_rejected_and_scores_not_mutated(self):
        original = self.scores.copy()
        for wrong in ({"alice": {"a"}}, {"alice": {"[PAD]"}}, {"alice": set()}):
            with self.assertRaises(ValueError):
                e.evaluate_scores(self.scores, self.users, self.items, self.train, wrong, ["alice"], k=3)
        self.assertTrue(np.array_equal(original, self.scores))
        broken = self.scores.copy(); broken[0, 3] = np.nan
        with self.assertRaises(ValueError):
            e.evaluate_scores(broken, self.users, self.items, self.train, self.truth, self.users, k=3)

    def test_cosine_donors_exclude_query_and_stable_ties(self):
        train = np.array([[0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=np.uint8)
        score = e.neighbor_scores(train, k=1)
        self.assertTrue(np.array_equal(score[0], train[1]))
        # The unique b record cannot support its own query through a self row.
        self.assertEqual(score[1, 2], 0)
        self.assertTrue(np.isfinite(e.neighbor_scores(np.zeros((1, 5)), k=16)).all())
        self.assertTrue(np.array_equal(train, np.array([[0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 0]])))

    def test_gate_meta_choice_is_not_reselected_on_development(self):
        meta = {2026: {"raw": .4, "summary": .2, "unconditional": .3, "ease": .35, "slim": .34},
                2027: {"raw": .4, "summary": .2, "unconditional": .3, "ease": .35, "slim": .34}}
        screen = e.meta_screen(meta, "raw", ("summary", "unconditional"), ("ease", "slim"))
        self.assertTrue(screen["passed"])
        dev = {seed: {"raw": .40, "summary": .2, "unconditional": .3, "ease": .35, "slim": .9} for seed in meta}
        result = e.development_gate(dev, screen)
        self.assertEqual(result["strongest_meta_reference"], "ease")
        self.assertTrue(result["substantial_target_met"])
        boundary = {seed: {"raw": .33, "summary": .2, "unconditional": .25, "ease": .3} for seed in meta}
        self.assertTrue(e.development_gate(boundary, screen)["substantial_target_met"])
        boundary[2026]["raw"] -= 1e-8
        self.assertFalse(e.development_gate(boundary, screen)["substantial_target_met"])
        dev[2027]["raw"] = .34
        self.assertFalse(e.development_gate(dev, screen)["substantial_target_met"])
        tampered = copy.deepcopy(screen); tampered["strongest_reference"] = "slim"
        with self.assertRaises(ValueError):
            e.development_gate(dev, tampered)
        meta[2026]["raw"] = .3; meta[2027]["raw"] = .3
        self.assertFalse(e.meta_screen(meta, "raw", ("summary", "unconditional"), ("ease", "slim"))["passed"])

    def test_paired_summary_constant_delta_and_mismatched_users(self):
        first = {name: {"ndcg@10": .5} for name in self.users}
        second = {name: {"ndcg@10": .25} for name in self.users}
        result = e.paired_summary(first, second, resamples=50)
        self.assertEqual(result["mean_delta"], .25)
        self.assertEqual(result["descriptive_bootstrap_95_percentile"], [.25, .25])
        self.assertNotIn("alice", json.dumps(result))
        with self.assertRaises(ValueError):
            e.paired_summary(first, {"alice": second["alice"]})


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "2026"
        self.directory.mkdir()
        self.users, self.items = ["u0", "u1"], ["[PAD]", "i0", "i1", "i2"]
        ids = {"users": ["[PAD]", *self.users], "items": self.items, "padding_index": 0}
        write_json(self.directory / "ids.json", ids)
        write_json(self.directory / "cohorts.json", {"meta_fit": ["u0"], "development": ["u1"]})
        self.signature = {"data_sha256": {"ratings": "raw", "metadata": "meta"},
                          "split_sha256": {"train": "train", "valid": "valid"},
                          "ids_sha256": e.digest(self.directory / "ids.json"),
                          "cohort_file_sha256": e.digest(self.directory / "cohorts.json"),
                          "ordered_identity_sha256": e.identity_digest(self.users, self.items)}
        scores = np.arange(8, dtype=float).reshape(2, 4)
        choices = {}
        for name in ("binary_expanded", "categorical"):
            path = self.directory / f"{name}.npy"
            np.save(path, scores)
            choices[name] = {"scores_file": path.name, "scores_file_sha256": e.digest(path),
                "scores_array_sha256": e.array_digest(scores), "selection_cohort": "reused_meta_fit",
                "selection_metric": "all_observed_ndcg@10", "refit_replay_exact": True,
                "score_file_replay_exact": True, "candidate_id": name, "candidate_count": 9,
                "penalty": 100., "category_ratio": None}
        write_json(self.directory / "selection.json", choices)
        np.save(self.directory / "locked-slim-scores.npy", scores)
        self.slim = {"source_path": "unused-path", "manifest_sha256": "source-manifest",
                     "score_file_sha256": "source-scores", "ids_file_sha256": self.signature["ids_sha256"],
                     "ordered_identity_sha256": self.signature["ordered_identity_sha256"],
                     "selection": {"selection_cohort": "meta_fit", "selection_metric": "meta-fit all-observed nDCG@10",
                                   "selection_file_sha256": "original-selected-settings"}}
        write_json(self.directory / "locked-slim.json", self.slim)
        write_json(self.root / "protocol.json", {"seeds": [2026]})
        self.seal()

    def seal(self):
        source = {"exploratory/conditional_evidence/evaluation.py": e.digest(Path(e.__file__))}
        runtime = {"numpy": np.__version__}
        slim = {key: value for key, value in self.slim.items() if key != "source_path"}
        selected = {"status": "selected", "seed": 2026, "test_read": False, "development_evaluated": False,
            "input_signature": self.signature, "source_sha256": source, "runtime": runtime,
            "locked_slim_source": slim, "output_sha256": {p.name: e.digest(p) for p in self.directory.iterdir()
                if p.is_file() and p.name != "selection-manifest.json"}}
        write_json(self.directory / "selection-manifest.json", selected)
        frozen = {"status": "all_selections_frozen", "test_read": False, "development_evaluated": False,
            "seeds": [2026], "source_sha256": source, "runtime": runtime,
            "protocol_sha256": e.digest(self.root / "protocol.json"),
            "selection_manifest_sha256": {"2026": e.digest(self.directory / "selection-manifest.json")}}
        write_json(self.root / "SELECTIONS-FROZEN.json", frozen)
        manifest = {"status": "complete", "test_read": False, "test_evaluated": False,
            "input_signatures": {"2026": self.signature}, "locked_slim_sources": {"2026": slim},
            "selection_freeze_sha256": e.digest(self.root / "SELECTIONS-FROZEN.json"),
            "source_sha256": source, "runtime": runtime,
            "output_sha256": {p.relative_to(self.root).as_posix(): e.digest(p) for p in self.directory.iterdir() if p.is_file()}}
        write_json(self.root / "manifest.json", manifest)

    def load(self, signature=None):
        return e.load_references(2026, self.signature if signature is None else signature,
                                 self.root, self.users, self.items)

    def test_reference_replay_without_outcome_files(self):
        # No TRAIN/VALID/TEST outcomes or old aggregate results even exist here.
        scores, metadata = self.load()
        self.assertEqual(tuple(scores), e.REFERENCE_NAMES)
        self.assertTrue(np.array_equal(scores["EASEexpanded"], scores["SLIM"]))
        self.assertFalse(metadata["provenance"]["outcome_files_read"])

    def test_input_catalog_and_score_changes_rejected(self):
        signature = copy.deepcopy(self.signature); signature["data_sha256"]["ratings"] = "changed"
        with self.assertRaises(ValueError):
            self.load(signature)
        with self.assertRaises(ValueError):
            e.load_references(2026, self.signature, self.root, self.users[::-1], self.items)
        with (self.directory / "locked-slim-scores.npy").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ValueError):
            self.load()

    def test_forbidden_payload_rejected_before_numpy_load(self):
        with self.assertRaises(ValueError):
            e._safe_path(self.directory, "test_labels.npy")
        path = self.directory / "selection.json"
        choices = json.loads(path.read_text())
        choices["binary_expanded"]["scores_file"] = "test.tsv"
        write_json(path, choices)
        self.seal()
        with mock.patch.object(e.np, "load", side_effect=AssertionError("must reject before reading scores")):
            with self.assertRaises(ValueError):
                self.load()

    def test_changed_slim_provenance_rejected(self):
        path = self.directory / "locked-slim.json"
        slim = json.loads(path.read_text()); slim["selection"]["selection_file_sha256"] = "different-model"
        write_json(path, slim)
        self.seal()  # Payload hashes updated but original selected identity retained.
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
