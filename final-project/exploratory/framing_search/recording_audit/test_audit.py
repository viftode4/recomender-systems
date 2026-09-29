import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exploratory.framing_search.recording_audit.audit import (
    check_inputs, describe, digest, genre_pair_cache, genre_statistics,
    read_train_records, require_digest,
)


class RecordingAuditTests(unittest.TestCase):
    def setUp(self):
        self.records = {
            "u1": [("a", 10, 5.), ("b", 10, 2.), ("c", 40, 4.), ("d", 40, 1.)],
            "u2": [("a", 11, 3.), ("b", 11, 2.), ("c", 11, 4.)],
            "u3": [("c", 0, 5.), ("d", 80, 1.)],
        }
        self.genres = {"a": {"comedy"}, "b": {"comedy", "drama"},
                       "c": {"drama"}, "d": set()}

    def literal(self, rng=None):
        pair_values, user_means = [], []
        for history in self.records.values():
            order = list(range(len(history))) if rng is None else rng.permutation(len(history))
            values = []
            for i, j in itertools.combinations(range(len(history)), 2):
                if history[i][1] != history[j][1]:
                    continue
                a, b = self.genres[history[order[i]][0]], self.genres[history[order[j]][0]]
                values.append(len(a & b) / len(a | b) if a | b else 0.)
            if values:
                user_means.append(sum(values) / len(values))
                pair_values.extend(values)
        return dict(pair_weighted=sum(pair_values) / len(pair_values),
                    macro_user=sum(user_means) / len(user_means),
                    tied_pairs=len(pair_values), eligible_users=len(user_means))

    def test_literal_pairs_match_matrix_for_observed_and_shuffles(self):
        cache = genre_pair_cache(self.records, self.genres)
        self.assertEqual(genre_statistics(cache), self.literal())
        for seed in range(10):
            actual = genre_statistics(cache, np.random.default_rng(seed))
            reference = self.literal(np.random.default_rng(seed))
            self.assertEqual(actual["tied_pairs"], reference["tied_pairs"])
            self.assertAlmostEqual(actual["pair_weighted"], reference["pair_weighted"])
            self.assertAlmostEqual(actual["macro_user"], reference["macro_user"])

    def test_descriptive_counts_and_denominators(self):
        result = describe(self.records)
        self.assertEqual(result["records"], 9)
        self.assertEqual(result["users"], 3)
        self.assertEqual(result["exact_time_groups"], 5)
        self.assertEqual(result["adjacent_event_gaps"], 6)
        self.assertEqual(result["distinct_time_gaps"], 2)
        self.assertEqual(result["zero_adjacent_event_gaps"]["count"], 4)
        self.assertEqual(result["group_thresholds"]["2"]["records"]["count"], 7)
        self.assertEqual(result["group_thresholds"]["2"]["users"]["count"], 2)
        self.assertEqual(result["positive_distinct_time_gap_at_most_seconds"]["60"]["count"], 1)

    def test_non_train_values_are_not_interpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.tsv"
            header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
            path.write_text(header + "u\ti\t4\t100\nv\tx\tNOT_A_RATING\tNOT_A_TIME\n")
            records, skipped = read_train_records(path, {("u", "i")})
            self.assertEqual(records, {"u": [("i", 100, 4.)]})
            self.assertEqual(skipped, 1)
            path.write_text(header + "u\ti\t4\t100\nv\tx\tnan\tinf\n")
            self.assertEqual(read_train_records(path, {("u", "i")}), (records, skipped))

    def test_missing_duplicate_and_invalid_train_records_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.tsv"
            header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
            for rows in ("", "u\ti\t4\t1.5\n", "u\ti\tnan\t100\n",
                         "u\ti\t4\t100\nu\ti\t4\t100\n"):
                path.write_text(header + rows)
                with self.assertRaises(ValueError):
                    read_train_records(path, {("u", "i")})

    def test_hash_gate_precedes_raw_value_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            train = root / "train.tsv"
            train.write_text("user_id\titem_id\nu\ti\n")
            (source / "train.tsv").write_bytes(train.read_bytes())
            raw, metadata = root / "data.inter", root / "data.item"
            raw.write_text("Unparsed data")
            metadata.write_text("Unparsed metadata")
            manifest = {"status": "complete", "test_evaluated": False,
                        "validation_used_for_training": False, "settings": {"seed": 2026},
                        "split_sha256": {"train": digest(train)}, "split_sizes": {"train": 1},
                        "data_sha256": {raw.name: digest(raw), metadata.name: digest(metadata)}}
            (source / "manifest.json").write_text(json.dumps(manifest))
            signature = root / "signature.json"
            signature.write_text(json.dumps({"source_manifest_sha256": digest(source / "manifest.json"),
                "split_sha256": manifest["split_sha256"], "data_sha256": manifest["data_sha256"]}))
            _, count = check_inputs(source, signature, train, raw, metadata)
            self.assertEqual(count, 1)
            raw.write_text("Changed data")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                check_inputs(source, signature, train, raw, metadata)
            with self.assertRaises(ValueError):
                require_digest(train, "bad")


if __name__ == "__main__":
    unittest.main()
