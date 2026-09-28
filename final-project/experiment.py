"""Run a small fixed-budget study across seeds without test evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from study import partition_users

ROOT = Path(__file__).resolve().parent
MODELS = ('ExactPop', 'Random', 'EASE', 'ItemKNN', 'BPR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tune-experts', action='store_true', help='Small predeclared grids, select on meta-fit users only')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out.resolve()
    data = args.data_path.resolve()
    plan = {'seeds': args.seeds, 'models': list(MODELS), 'test_evaluated': False,
            'protocol': 'Fixed expert budget; meta-fit/development user partition; no test evaluation.',
            'status': 'started', 'tune_experts': args.tune_experts,
            'grids': {'EASE': [50.,250.,1000.], 'ItemKNN': [50,100,200], 'BPR': [[32,20],[64,60],[128,100]]}}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for seed in args.seeds:
        config = json.loads((ROOT/'config.json').read_text())
        config['seed'] = seed
        # Explicit reasonable starting points, fixed before this multi-seed run.
        config.update({'reg_weight': 250., 'k': 100, 'knn_method': 'item', 'embedding_size': 64})
        config_path = output/f'config-{seed}.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        paths = []
        selections = {}
        for model in MODELS:
            variants = [{}]
            if args.tune_experts:
                variants = {'EASE': [{'reg_weight': r} for r in (50.,250.,1000.)],
                            'ItemKNN': [{'k': k} for k in (50,100,200)],
                            'BPR': [{'embedding_size': d,'epochs': e} for d,e in ((32,20),(64,60),(128,100))]}.get(model,[{}])
            candidates = []
            for index, variant in enumerate(variants):
                run_path = output/f'{seed}-{model}-{index}'
                variant_config = output/f'config-{seed}-{model}-{index}.json'
                variant_config.write_text(json.dumps({**config, **variant}, indent=2)+'\n')
                command = [sys.executable, str(ROOT/'run.py'), '--model', model,
                           '--data-path', str(data), '--config', str(variant_config),
                           '--out', str(run_path), '--fixed-epochs']
                print(f'Running {seed} {model} {variant}', flush=True)
                with (output/f'{seed}-{model}-{index}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                metrics = json.loads((run_path/'valid-metrics.json').read_text())['per_user']
                fit_users, _ = partition_users(list(metrics), seed)
                score = sum(metrics[u]['ndcg@10'] for u in fit_users)/len(fit_users)
                candidates.append({'path': str(run_path), 'settings': variant, 'meta_fit_ndcg': score})
            winner = max(candidates, key=lambda row: row['meta_fit_ndcg'])
            paths.append(Path(winner['path']))
            selections[model] = {'selected': winner, 'candidates': candidates}
        (output/f'expert-selection-{seed}.json').write_text(json.dumps(selections, indent=2)+'\n')
        command = [sys.executable, str(ROOT/'study.py'), '--runs', *map(str, paths),
                   '--items', str(data/'ml-100k/ml-100k.item'), '--out', str(output/f'study-{seed}'),
                   '--seed', str(seed)]
        print(f'Running hybrid study {seed}', flush=True)
        with (output/f'study-{seed}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    plan['status'] = 'complete'
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(f'Completed {output}', flush=True)


if __name__ == '__main__':
    main()
