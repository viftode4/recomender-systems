"""Run a small fixed-budget study across seeds without test evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from study import partition_users

ROOT = Path(__file__).resolve().parent
MODELS = ('ExactPop', 'Random', 'EASE', 'ItemKNN', 'BPR')
EXTENDED_MODELS = ('UserKNN', 'SLIMElastic', 'FISMCorrected', 'GenreContent', 'LightGCN', 'NeuMF', 'NGCF')
DEFAULTS = {
    'EASE': {'reg_weight': 250.},
    'ItemKNN': {'k': 100, 'knn_method': 'item'},
    'UserKNN': {'k': 100, 'knn_method': 'user'},
    'BPR': {'embedding_size': 64},
    'SLIMElastic': {'alpha': .01, 'l1_ratio': .1, 'hide_item': True, 'positive_only': True},
    'FISMCorrected': {'embedding_size': 64, 'alpha': .5, 'reg_weights': [.001, .001]},
    'GenreContent': {'genre_idf_power': 1.},
    'LightGCN': {'embedding_size': 64, 'n_layers': 2, 'reg_weight': 1e-5},
    'NeuMF': {'mf_embedding_size': 64, 'mlp_embedding_size': 64, 'mlp_hidden_size': [128, 64],
              'use_pretrain': False},
    'NGCF': {'embedding_size': 64, 'hidden_size_list': [64, 64, 64], 'reg_weight': 1e-5,
             'node_dropout': 0., 'message_dropout': .1},
}
GRIDS = {
    'EASE': [{'reg_weight': r} for r in (50., 250., 1000.)],
    'ItemKNN': [{'k': k} for k in (50, 100, 200)],
    'UserKNN': [{'k': k} for k in (50, 100, 200)],
    'BPR': [{'embedding_size': d, 'epochs': e} for d, e in ((32, 20), (64, 60), (128, 100))],
    'SLIMElastic': [{'alpha': a, 'l1_ratio': l1} for a, l1 in ((.001, .1), (.01, .1), (.01, .5))],
    'FISMCorrected': [{'alpha': a} for a in (0., .5, 1.)],
    'GenreContent': [{'genre_idf_power': p} for p in (0., 1.)],
    'LightGCN': [{'epochs': e} for e in (20, 60)],
    'NeuMF': [{'epochs': e} for e in (20, 60)],
    'NGCF': [{'epochs': e} for e in (20, 60)],
}


def variant_settings(model, tune, epochs=None, thorough_lightgcn_budget=False):
    variants = GRIDS.get(model, [{}]) if tune else [{}]
    if model == 'LightGCN' and thorough_lightgcn_budget:
        variants = [{'epochs': e} for e in (20, 60, 100, 200)]
    # A shared override is for explicit smoke/resource budgets, recorded in plan.
    result = [{**DEFAULTS.get(model, {}), **v, **({'epochs': epochs} if epochs is not None else {})}
              for v in variants]
    return list({json.dumps(v, sort_keys=True): v for v in result}.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tune-experts', action='store_true', help='Small predeclared grids, select on meta-fit users only')
    parser.add_argument('--extended', action='store_true', help='Include user CF, sparse linear, factored item, content, graph and neural baselines')
    parser.add_argument('--thorough-lightgcn-budget', action='store_true',
                        help='Explicit LightGCN 20/60/100/200 epoch grid; also applies without --tune-experts')
    parser.add_argument('--models', nargs='+', choices=MODELS + EXTENDED_MODELS,
                        help='Explicit subset; useful for adding coverage without rerunning existing experts')
    parser.add_argument('--epochs', type=int, help='Explicit shared epoch budget (e.g. 1 for a smoke run); recorded in plan')
    parser.add_argument('--skip-study', action='store_true', help='Export standalone runs and selections; do not fit hybrids')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    models = tuple(dict.fromkeys(args.models or (MODELS + EXTENDED_MODELS if args.extended else MODELS)))
    if len([model for model in models if model != 'Random']) < 2 and not args.skip_study:
        parser.error('A hybrid study requires at least two non-Random experts; use --skip-study for a standalone run')
    if args.thorough_lightgcn_budget and 'LightGCN' not in models:
        parser.error('--thorough-lightgcn-budget requires LightGCN in the selected models')
    if args.thorough_lightgcn_budget and args.epochs is not None:
        parser.error('--thorough-lightgcn-budget cannot be combined with --epochs')
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out.resolve()
    data = args.data_path.resolve()
    plan = {'seeds': args.seeds, 'models': list(models), 'test_evaluated': False,
            'protocol': 'Fixed expert budget; meta-fit/development user partition; no test evaluation.',
            'status': 'started', 'tune_experts': args.tune_experts, 'extended': args.extended,
            'thorough_lightgcn_budget': args.thorough_lightgcn_budget,
            'epoch_override': args.epochs, 'skip_study': args.skip_study,
            'grids': {model: variant_settings(model, args.tune_experts, args.epochs, args.thorough_lightgcn_budget)
                      for model in models}}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for seed in args.seeds:
        config = json.loads((ROOT/'config.json').read_text())
        config['seed'] = seed
        # Explicit reasonable starting points, fixed before this multi-seed run.
        config_path = output/f'config-{seed}.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        paths = []
        selections = {}
        for model in models:
            variants = variant_settings(model, args.tune_experts, args.epochs, args.thorough_lightgcn_budget)
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
                run_manifest = json.loads((run_path/'manifest.json').read_text())
                candidates.append({'path': str(run_path), 'settings': variant, 'meta_fit_ndcg': score,
                                   'timing_seconds': run_manifest['timing_seconds']})
            winner = max(candidates, key=lambda row: row['meta_fit_ndcg'])
            paths.append(Path(winner['path']))
            selections[model] = {'selected': winner, 'candidates': candidates}
        (output/f'expert-selection-{seed}.json').write_text(json.dumps(selections, indent=2)+'\n')
        if args.skip_study:
            continue
        command = [sys.executable, str(ROOT/'study.py'), '--runs', *map(str, paths),
                   '--items', str(data/'ml-100k/ml-100k.item'), '--out', str(output/f'study-{seed}'),
                   '--seed', str(seed)]
        print(f'Running hybrid study {seed}', flush=True)
        with (output/f'study-{seed}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    plan['status'] = 'complete'
    plan['timing_seconds'] = time.perf_counter() - started
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(f'Completed {output}', flush=True)


if __name__ == '__main__':
    main()
