"""Read saved study progress without opening checkpoints or any rating records.

This reports durable artifacts, not whether an operating-system process is alive.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # A progress file can be in the middle of an update.


def status(root):
    protocol = read_json(root / 'protocol.json')
    if protocol is None:
        raise ValueError('No readable study protocol at ' + str(root))
    report = {
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'meaning': 'Saved progress only; this does not establish process liveness.',
        'study_kind': protocol['study_kind'],
        'fresh_confirmation': False,
        'maximum_epochs_per_variant': protocol['training_budget']['max_epochs'],
        'seeds': {},
    }
    for seed in protocol['seeds']:
        folder = root / str(seed)
        manifest = read_json(folder / 'manifest.json')
        row = {'completion_manifest_present': bool(manifest), 'variants': {}}
        for variant in protocol['variants']:
            directory = folder / variant
            grid = read_json(directory / 'selection-grid.json') or []
            trace = read_json(directory / 'training-trace.json') or []
            selection = read_json(directory / 'selection.json')
            info = {
                'last_saved_epoch': trace[-1]['epoch'] if trace else 0,
                'last_train_probe_nll': trace[-1]['macro_train_joint_probe_nll'] if trace else None,
                'last_meta_fit_nll': grid[-1]['meta_fit_macro_joint_nll'] if grid else None,
                'selected_epoch': selection['epoch'] if selection else None,
                'stop_reason': selection['stop_reason'] if selection else None,
            }
            if trace:
                info['trace_modified_at_utc'] = datetime.fromtimestamp(
                    (directory / 'training-trace.json').stat().st_mtime,
                    timezone.utc,
                ).isoformat()
            row['variants'][variant] = info
        report['seeds'][str(seed)] = row
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(status(args.run), indent=2, allow_nan=False))
