"""Combine meta-fit expert selections into one validation-only hybrid study.

Selection files are read in supplied order. A later selection replaces an
earlier selection of the same model, allowing a declared extended budget.
The optional research run adds the three predeclared all-observed experts.
No result metric is inspected here and no test file is opened.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from study import digest, write_json


RESEARCH_EXPERTS = ('positive_ease-all_observed', 'signed_channels-all_observed',
                    'contrast-all_observed')


def selected_paths(selections, research_run=None):
    selected = {}
    for path in selections:
        records = json.loads(Path(path).read_text())
        for model, row in records.items():
            run = Path(row['selected']['path']).resolve()
            manifest = json.loads((run / 'manifest.json').read_text())
            if manifest['model'] != model or manifest['status'] != 'complete':
                raise ValueError(f'Selection model/status mismatch: {run}')
            if manifest.get('test_evaluated') is not False or manifest.get('validation_used_for_training') is not False:
                raise ValueError('Require fixed-budget, validation-only expert sources')
            selected[model] = run
    if research_run is not None:
        for name in RESEARCH_EXPERTS:
            run = Path(research_run) / name
            manifest = json.loads((run / 'manifest.json').read_text())
            choice = manifest['selection']
            if choice['selection_objective'] != 'all_observed' or choice['selection_cohort'] != 'meta_fit':
                raise ValueError('Research expert was selected on another objective/cohort')
            selected[manifest['model']] = run.resolve()
    if len(selected) < 2:
        raise ValueError('Need at least two selected experts')
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selections', type=Path, nargs='+', required=True)
    parser.add_argument('--research-run', type=Path)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--calibration-fraction', type=float, default=.5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    runs = selected_paths(args.selections, args.research_run)
    command = [sys.executable, str(Path(__file__).with_name('study.py')),
               '--runs', *map(str, runs), '--items', str(args.items), '--out', str(args.out),
               '--seed', str(args.seed), '--calibration-fraction', str(args.calibration_fraction)]
    subprocess.run(command, check=True)
    write_json(args.out / 'assembly.json', {
        'seed': args.seed, 'runs': list(map(str, runs)),
        'selection_sha256': {str(p): digest(p) for p in args.selections},
        'research_run': None if args.research_run is None else str(args.research_run.resolve()),
        'command': command, 'test_read': False,
        'note': 'Later selection files replace the same model; no metric is used to choose between files.'})


if __name__ == '__main__':
    main()
