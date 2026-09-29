"""Independent information-boundary, selection and ranking checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exploratory.framing_search.predictive import run as runner


class RunnerBoundaryTests(unittest.TestCase):
    def test_nested_split_matches_declared_hash_partition_and_is_order_invariant(self):
        pairs = [(str(user), str(item)) for user, count in ((1, 20), (2, 25), (3, 17), (4, 3))
                 for item in range(1, count + 1)]
        actual = runner.nested_split(pairs, seed=20260929)
        expected = {name: set() for name in ('fit', 'development', 'assessment')}
        for user in sorted({user for user, item in pairs}):
            items = [item for other, item in pairs if other == user]
            ordered = sorted(items, key=lambda item: (
                hashlib.sha256(json.dumps([20260929, user, item],
                    separators=(',', ':')).encode()).hexdigest(), item))
            count = max(1, len(ordered) // 10)
            bounds = (('fit', ordered[:-2 * count]),
                      ('development', ordered[-2 * count:-count]),
                      ('assessment', ordered[-count:]))
            for name, selected in bounds:
                expected[name].update((user, item) for item in selected)
        for name, wanted in expected.items():
            self.assertEqual(set(map(tuple, actual[name])), wanted)
        self.assertEqual(set().union(*expected.values()), set(pairs))
        self.assertFalse(expected['fit'] & expected['development'])
        self.assertFalse(expected['fit'] & expected['assessment'])
        self.assertFalse(expected['development'] & expected['assessment'])
        reverse = runner.nested_split(pairs[::-1], seed=20260929)
        self.assertEqual(actual, reverse)

    def test_nested_split_rejects_duplicates_and_too_short_histories(self):
        with self.assertRaises(ValueError):
            runner.nested_split([('1', '1'), ('1', '1'), ('1', '2')])
        with self.assertRaises(ValueError):
            runner.nested_split([('1', '1'), ('1', '2')])

    def test_fit_loader_never_interprets_heldout_timestamps_or_any_ratings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'records.inter'
            path.write_text('user_id:token\titem_id:token\trating:float\ttimestamp:float\n'
                '1\t1\tRATING_NOT_NEEDED\t100\n'
                '1\t2\tINVALID_HELDOUT_RATING\tDO_NOT_PARSE\n'
                '2\t3\tALSO_NOT_NEEDED\t101\n')
            values = runner.read_fit_timestamps(path, {('1', '1'), ('2', '3')})
            self.assertEqual(values, {('1', '1'): 100, ('2', '3'): 101})

    def test_fit_loader_rejects_missing_duplicate_and_invalid_allowed_timestamps(self):
        header = 'user_id:token\titem_id:token\trating:float\ttimestamp:float\n'
        variants = ('1\t2\t5\t100\n',
                    '1\t1\t5\t100\n1\t1\t5\t101\n',
                    '1\t1\t5\tnan\n',
                    '1\t1\t5\t100.5\n')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'records.inter'
            for body in variants:
                with self.subTest(body=body):
                    path.write_text(header + body)
                    with self.assertRaises(ValueError):
                        runner.read_fit_timestamps(path, {('1', '1')})

    def test_assessment_guard_precedes_any_file_parse(self):
        class BarrierFailure(Exception):
            pass
        with patch.object(runner, 'verify_barrier', side_effect=BarrierFailure('not sealed')) as guard:
            with self.assertRaises(BarrierFailure):
                runner.read_assessment_values(Path('/does-not-exist.inter'),
                                              {('1', '1')}, Path('/unsealed-run'))
            guard.assert_called_once_with(Path('/unsealed-run'))



    def test_ranking_masks_fit_and_padding_with_stable_catalog_ties(self):
        scores = np.array([[999., 9., 3., 3., 2., 8.], [999., 2., 2., 2., 2., 2.]])
        observed = np.array([[False, True, False, False, False, True],
                             [False, False, True, False, False, False]])
        ranks = runner.rank_scores(scores, observed, k=3)
        np.testing.assert_array_equal(ranks, [[2, 3, 4], [1, 3, 4]])
        np.testing.assert_array_equal(scores[:, 0], [999., 999.])
        with self.assertRaises(ValueError):
            runner.rank_scores(scores, np.ones_like(observed), k=3)
        poisoned = scores.copy()
        poisoned[0, 2] = np.nan
        with self.assertRaises(ValueError):
            runner.rank_scores(poisoned, observed, k=3)

    def test_ranking_metrics_match_literal_hand_calculation(self):
        ranks = np.array([[2, 3, 4], [1, 3, 4], [3, 4, 5]])
        truth = [{3, 7}, {1}, {6, 7, 8, 9}]
        actual = runner.metric_rows(ranks, truth)
        expected = {'precision@3': [1/3, 1/3, 0.], 'recall@3': [.5, 1., 0.],
                    'mrr@3': [.5, 1., 0.], 'hit@3': [1., 1., 0.],
                    'ndcg@3': [(1/np.log2(3))/(1+1/np.log2(3)), 1., 0.]}
        self.assertEqual(set(actual), set(expected))
        for name, values in expected.items():
            np.testing.assert_allclose(actual[name], values, atol=1e-14, rtol=0.)

    def make_barrier_fixture(self, directory):
        directory = Path(directory)
        (directory / 'selected-scores').mkdir()
        candidates, selections = [], {}
        for role in runner.ROLES:
            own = []
            for penalty, beta in runner.candidate_specs(role):
                row = {'role': role, 'lambda': penalty, 'beta': beta,
                       'candidate_id': runner.candidate_id(role, penalty, beta),
                       'development': {'users': 1, 'aggregate': {'ndcg@10': .2}}}
                candidates.append(row)
                own.append(row)
            relative = f'selected-scores/{role}.npy'
            np.save(directory / relative, np.zeros((1, 12)), allow_pickle=False)
            selections[role] = dict(own[0], scores_file=relative,
                                    scores_file_sha256=runner.digest(directory / relative))
        runner.save_json(directory / 'protocol.json',
                         {'source_sha256': runner.source_hashes(), 'inputs': {}})
        runner.save_json(directory / 'selections.json', selections)
        runner.save_json(directory / 'candidates.json', candidates)
        for name in ('nested-A.tsv', 'nested-F.tsv', 'nested-D.tsv', 'fit-context.npz', 'ids.json'):
            (directory / name).write_text('independent fixture\n')
        return selections, candidates

    def test_all_role_barrier_rejects_missing_or_incomplete_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            selections, candidates = self.make_barrier_fixture(directory)
            with self.assertRaises((FileNotFoundError, ValueError)):
                runner.verify_barrier(directory)
            with self.assertRaises(ValueError):
                runner.write_barrier(directory, runner.ROLES[:-1])
            del selections[runner.ROLES[-1]]
            runner.save_json(directory / 'selections.json', selections)
            with self.assertRaises(ValueError):
                runner.write_barrier(directory)

    def test_sealed_files_cannot_change_or_be_omitted(self):
        for altered in ('nested-A.tsv', 'selected-scores/true_group.npy', 'omit_assessment_hash'):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.make_barrier_fixture(directory)
                runner.write_barrier(directory)
                runner.verify_barrier(directory)
                if altered == 'omit_assessment_hash':
                    seal = json.loads((directory / 'SELECTIONS-FROZEN.json').read_text())
                    del seal['files_sha256']['nested-A.tsv']
                    runner.save_json(directory / 'SELECTIONS-FROZEN.json', seal)
                    (directory / 'nested-A.tsv').write_text('changed target IDs\n')
                else:
                    with (directory / altered).open('ab') as stream:
                        stream.write(b'changed')
                with self.assertRaises(ValueError):
                    runner.verify_barrier(directory)

    def test_barrier_requires_complete_grid_and_first_exact_argmax(self):
        for alteration in ('missing_candidate', 'wrong_tie_choice', 'nonmaximum_choice'):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                selections, candidates = self.make_barrier_fixture(directory)
                if alteration == 'missing_candidate':
                    candidates.pop()
                elif alteration == 'wrong_tie_choice':
                    choices = [row for row in candidates if row['role'] == 'true_group']
                    selections['true_group'].update(choices[1])
                else:
                    candidates[1]['development']['aggregate']['ndcg@10'] = .3
                runner.save_json(directory / 'candidates.json', candidates)
                runner.save_json(directory / 'selections.json', selections)
                with self.assertRaises(ValueError):
                    runner.write_barrier(directory)


if __name__ == '__main__':
    unittest.main()
