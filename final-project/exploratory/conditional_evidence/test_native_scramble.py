import unittest
from unittest.mock import patch

import numpy as np

from exploratory.conditional_evidence import native_scramble as scramble


class NativeScrambleTests(unittest.TestCase):
    def test_splitmix64_known_words(self):
        state = 0
        words = []
        for _ in range(3):
            state, value = scramble._next(state)
            words.append(value)
        self.assertEqual(words, [0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f])

    def test_native_matches_literal_python_reference(self):
        try:
            scramble._load_native()
        except scramble.NativeUnavailable as error:
            self.skipTest(str(error))
        rng = np.random.default_rng(712)
        matrices = [np.zeros((0, 0), dtype=np.uint8), np.zeros((5, 0), dtype=np.uint8),
                    np.ones((1, 8), dtype=np.uint8), np.array([[1, 0], [0, 1]], dtype=np.uint8)]
        matrices += [np.where(rng.random((rows, columns)) < density,
                             rng.integers(1, 6, size=(rows, columns)), 0).astype(np.uint8)
                     for rows, columns in [(2, 5), (8, 30), (16, 120)] for density in (.1, .4, .85)]
        for matrix in matrices:
            before = matrix.copy()
            for seed in (0, 1, 921, (1 << 64) - 1):
                with self.subTest(shape=matrix.shape, seed=seed):
                    native, native_stats = scramble.scramble_history(matrix, seed, backend="native")
                    reference, reference_stats = scramble.scramble_history(matrix, seed, backend="python")
                    np.testing.assert_array_equal(native, reference)
                    self.assertEqual(native_stats, reference_stats)
                    self.assertLessEqual(native_stats["attempts"], 20 * np.count_nonzero(matrix))
                    self.assertLessEqual(native_stats["successes"], native_stats["attempts"])
                    for rating in range(1, 6):
                        np.testing.assert_array_equal((native == rating).sum(0), (matrix == rating).sum(0))
                        np.testing.assert_array_equal((native == rating).sum(1), (matrix == rating).sum(1))
            np.testing.assert_array_equal(matrix, before)

    def test_real_switches_occur_and_every_collision_is_preserved(self):
        matrix = np.array([[1, 0, 2, 0], [0, 1, 0, 2], [2, 0, 1, 0], [0, 2, 0, 1]], dtype=np.uint8)
        result, stats = scramble.scramble_history(matrix, 567, backend="python")
        self.assertGreater(stats["successes"], 0)
        self.assertEqual(stats["attempts"], 160)
        self.assertEqual(np.count_nonzero(result), np.count_nonzero(matrix))
        complete = np.ones((5, 9), dtype=np.uint8)
        result, stats = scramble.scramble_history(complete, 22, backend="python")
        np.testing.assert_array_equal(result, complete)
        self.assertEqual(stats["successes"], 0)

    def test_python_fallback_has_identical_outputs_without_compiler(self):
        matrix = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=np.uint8)
        expected = scramble.scramble_history(matrix, 28, backend="python")
        with patch.object(scramble, "_load_native", side_effect=scramble.NativeUnavailable("no compiler")):
            result, stats = scramble.scramble_history(matrix, 28)
            np.testing.assert_array_equal(result, expected[0])
            self.assertEqual(stats, expected[1])
            self.assertEqual(scramble.native_provenance()["backend"], "python")
            with self.assertRaises(scramble.NativeUnavailable):
                scramble.scramble_history(matrix, 28, backend="native")

    def test_invalid_categories_and_seeds_fail_before_native_access(self):
        with patch.object(scramble, "_load_native", side_effect=AssertionError("native called")):
            for value in (np.array([[np.nan]]), np.array([[1.5]]), np.array([[6]]), np.array([1])):
                with self.assertRaises(ValueError):
                    scramble.scramble_history(value, 0)
            for seed in (-1, 1 << 64, 1.5):
                with self.assertRaises(ValueError):
                    scramble.scramble_history([[1, 0], [0, 1]], seed)

    def test_runtime_metadata_rejects_changed_source_or_library_binding(self):
        record = {"source_sha256": "changed", "wrapper_sha256": "changed",
                  "library_sha256": "changed", "library_path": "unused"}
        with patch.object(scramble, "_load_native", return_value=(None, None, record)):
            with self.assertRaisesRegex(RuntimeError, "changed after initialization"):
                scramble.runtime_metadata()


if __name__ == "__main__":
    unittest.main()
