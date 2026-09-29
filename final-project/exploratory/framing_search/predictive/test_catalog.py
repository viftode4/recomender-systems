import unittest

from exploratory.framing_search.predictive.run import real_catalog_users


class CatalogBoundaryTests(unittest.TestCase):
    def test_explicit_padding_removed_and_real_order_preserved(self):
        pairs = [('b', 'x'), ('a', 'y')]
        self.assertEqual(real_catalog_users({'users': ['[PAD]', 'b', 'a']}, pairs), ['b', 'a'])
        self.assertEqual(real_catalog_users({'users': ['a', 'b']}, pairs), ['a', 'b'])

    def test_missing_extra_and_duplicate_real_users_rejected(self):
        pairs = [('a', 'x'), ('b', 'y')]
        for users in (['a'], ['a', 'b', 'c'], ['a', 'b', 'a']):
            with self.assertRaises(ValueError):
                real_catalog_users({'users': users}, pairs)


if __name__ == '__main__':
    unittest.main()
