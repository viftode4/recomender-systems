"""Training-only parsing contract for fixed research source reproduction."""
from pathlib import Path
import tempfile
import unittest

from coursework_completion.rebuild_research import training_ratings


class TrainingRatingTests(unittest.TestCase):
    def test_nontraining_rating_values_are_never_interpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ratings.inter'
            path.write_text('user_id:token\titem_id:token\trating:float\n'
                            'u\ta\t4\n'
                            'u\theld-out\tINVALID-IF-PARSED\n'
                            'v\tb\t2\n')
            self.assertEqual(training_ratings(path, [('v', 'b'), ('u', 'a')]),
                             [('v', 'b', 2.), ('u', 'a', 4.)])

    def test_missing_or_duplicate_training_records_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ratings.inter'
            header = 'user_id:token\titem_id:token\trating:float\n'
            path.write_text(header + 'u\ta\t4\n')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                training_ratings(path, [('u', 'a'), ('v', 'b')])
            path.write_text(header + 'u\ta\t4\nu\ta\t5\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                training_ratings(path, [('u', 'a')])


if __name__ == '__main__':
    unittest.main()
