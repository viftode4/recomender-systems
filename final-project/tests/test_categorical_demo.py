import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from categorical_demo import (CANDIDATES, SYNTHETIC_HISTORY, VARIABLE_ITEM,
                              infer_payload, load_catalog, write_demo)
from categorical_field import CategoricalEvidenceField


class CategoricalDemoTests(unittest.TestCase):
    def test_six_scenarios_change_only_one_synthetic_rating(self):
        torch.manual_seed(8)
        items = ['[PAD]']+sorted({i for i, _ in SYNTHETIC_HISTORY} | {VARIABLE_ITEM} | set(CANDIDATES))
        metadata = {i: {'title': 'Fictional display '+i, 'year': '1995'} for i in items[1:]}
        model = CategoricalEvidenceField(len(items), dim=2, ports=2, steps=2)
        inputs = []
        model.register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
        payload = infer_payload(model, items, metadata, {'seed': 8, 'epoch': 1, 'checkpoint_sha256': 'a'*64})
        self.assertEqual(payload['profile_kind'], 'hand_authored_synthetic')
        self.assertEqual(set(payload['scenarios']), set(map(str, range(6))))
        for rating, context in enumerate(inputs):
            self.assertEqual(context[0, items.index(VARIABLE_ITEM)], rating)
            self.assertEqual(int(torch.count_nonzero(context)), len(SYNTHETIC_HISTORY)+(rating != 0))
            for item, expected in SYNTHETIC_HISTORY:
                self.assertEqual(context[0, items.index(item)], expected)
            for candidate in CANDIDATES:
                self.assertEqual(context[0, items.index(candidate)], 0)
            scenario = payload['scenarios'][str(rating)]
            for candidate in scenario['candidates']:
                np.testing.assert_allclose(sum(candidate['probabilities']), 1., atol=2e-6)
                self.assertAlmostEqual(candidate['probability_rating_at_least_four'], sum(candidate['probabilities'][3:]))
        self.assertIsNone(payload['scenarios']['0']['variable_source_weights'])
        self.assertEqual(len(payload['scenarios']['5']['variable_source_weights']), 2)

    def test_html_json_escapes_script_text_and_catalog_ignores_user_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'catalog.npz'
            # Object arrays cannot be loaded without pickle. Reading only items
            # must succeed even when an unrelated users array is unreadable.
            np.savez(archive, items=np.asarray(['[PAD]', '1']), users=np.asarray([object()], dtype=object))
            self.assertEqual(load_catalog(archive), ['[PAD]', '1'])
            payload = {'title': '</script><script>bad()</script>', 'provenance': {'checkpoint_sha256': 'b'*64}}
            write_demo(payload, root/'demo')
            html = (root/'demo'/'index.html').read_text()
            self.assertNotIn(payload['title'], html)
            self.assertEqual(html.count('<script'), 2)
            self.assertEqual(json.loads((root/'demo'/'predictions.json').read_text()), payload)
            manifest = json.loads((root/'demo'/'manifest.json').read_text())
            self.assertFalse(manifest['test_read'])
            self.assertFalse(manifest['real_user_history_read'])


if __name__ == '__main__':
    unittest.main()
