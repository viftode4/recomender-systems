"""Explicitly separate freezing from the one-time rating-aware test audit.

Freeze opens train/validation artifacts only. Evaluation requires the frozen
code, source manifests, IDs and full-catalog scores to match every recorded hash.
It evaluates every preselected model/objective; test outcomes select nothing.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from exception_experiment import evaluate_ratings, recommendations
from study import digest, grouped, load_runs, read_pairs, write_json


ROOT = Path(__file__).resolve().parent
CODE_FILES = ('exception_evaluation.py','exception_experiment.py','exception_model.py',
              'metrics.py','study.py','hybrid_constraints.py')
ALLOWED_MODELS = {'ContrastTransfer','PositiveEASE','SignedEASE','SignedChannelsLinear',
                  'PairContrastGate','PairContrastRandomPartners','PairAnchorKernel',
                  'PairSignedKernel','ObservedEASE','PairContrastPermutedPartners'}


def freeze_sources(runs, out, seed):
    users,items,experts,train,valid,sources = load_runs(runs)
    ids = json.loads((runs[0]/'ids.json').read_text())
    if ids.get('padding_index') != 0 or ids['items'] != items:
        raise ValueError('Invalid original ID mapping')
    artifacts = []
    for run in runs:
        source = sources[run.name]['manifest']
        if source['model'] not in ALLOWED_MODELS or source.get('predictor_kind') != 'static_train_fitted_full_catalog':
            raise ValueError('Unsupported frozen predictor')
        if source.get('validation_used_for_training',True):
            raise ValueError('Frozen expert representation must be train-only')
        if source.get('selection',{}).get('selection_cohort') != 'meta_fit':
            raise ValueError('Only meta-fit selections can be frozen')
        if source['selection'].get('seed') != seed:
            raise ValueError('Selection seed differs from frozen split seed')
        if json.loads((run/'ids.json').read_text()) != ids:
            raise ValueError('Original internal ID mappings differ')
        expected_score = source.get('export_sha256',{}).get('valid-scores.npz')
        if expected_score != digest(run/'valid-scores.npz'):
            raise ValueError('Exported score hash differs from producer manifest')
        artifacts.append({'name':run.name,'model':source['model'],'selection':source['selection'],
                          'directory':str(run.resolve()),
                          'sha256':{p:digest(run/p) for p in ('manifest.json','ids.json','valid-scores.npz','train.tsv','valid.tsv')}})
    out.mkdir(parents=True,exist_ok=False)
    for part in ('train','valid'):
        shutil.copyfile(runs[0]/f'{part}.tsv',out/f'{part}.tsv')
    reference = next(iter(sources.values()))['manifest']
    bundle = {'status':'frozen','seed':seed,'users':users,'items':items,'k':10,
              'models':artifacts,'test_read':False,'test_evaluated':False,
              'expected_test_sha256':reference['split_sha256']['test'],
              'split_sha256':reference['split_sha256'],'data_sha256':reference['data_sha256'],
              'protocol':{
                  'fit':'Saved train-only model predictions; no refitting with validation.',
                  'selection':'All preselected meta-fit choices for both objectives evaluated; no test-based model selection.',
                  'mask':'Exclude padding and every train+validation interaction, including dislikes and neutrals.',
                  'relevance':'Report all observed ratings and separately ratings>=4; <=2 are known dislikes; 3 neutral.',
                  'denominators':'All test users for observed metrics; only users with at least one test like for liked metrics; unknown preferences remain unknown.',
                  'inference':'Every frozen predictor is phase-independent; saved full-catalog scores reused.'}}
    write_json(out/'bundle.json',bundle)
    return digest(out/'bundle.json')


def freeze_research(research_root,out,seeds):
    expected = ('contrast','positive_ease','signed_ease','signed_channels','pair_gate',
                'pair_gate_random','pair_anchor','pair_signed')
    out.mkdir(parents=True,exist_ok=False)
    bundles = {}
    for seed in seeds:
        runs = [research_root/str(seed)/f'{name}-{objective}' for name in expected
                for objective in ('all_observed','liked_ratings')]
        runs += [research_root/'implicit-ease-audit'/str(seed)/f'observed_ease-{objective}'
                 for objective in ('all_observed','liked_ratings')]
        runs += [research_root/'pair-permutation-audit'/str(seed)/f'pair_gate_permuted-{objective}'
                 for objective in ('all_observed','liked_ratings')]
        bundles[str(seed)] = freeze_sources(runs,out/str(seed),seed)
    write_json(out/'manifest.json',{'status':'frozen','format_version':1,'seeds':seeds,
                                  'bundle_sha256':bundles,'test_read':False,'test_evaluated':False,
                                  'code_sha256':{name:digest(ROOT/name) for name in CODE_FILES}})


def preflight(frozen_root):
    manifest = json.loads((frozen_root/'manifest.json').read_text())
    if manifest.get('status') != 'frozen' or manifest.get('test_read'):
        raise ValueError('Require an unused frozen bundle')
    for filename,expected in manifest['code_sha256'].items():
        if filename not in CODE_FILES or digest(ROOT/filename) != expected:
            raise ValueError(f'Frozen evaluation code changed: {filename}')
    if set(manifest['code_sha256']) != set(CODE_FILES):
        raise ValueError('Incomplete frozen code signature')
    bundles = []
    for seed in manifest['seeds']:
        folder = frozen_root/str(seed)
        if digest(folder/'bundle.json') != manifest['bundle_sha256'][str(seed)]:
            raise ValueError('Frozen model choices or bundle changed')
        bundle = json.loads((folder/'bundle.json').read_text())
        for part in ('train','valid'):
            if digest(folder/f'{part}.tsv') != bundle['split_sha256'][part]:
                raise ValueError('Frozen train/validation split changed')
        for model in bundle['models']:
            for filename,expected in model['sha256'].items():
                if digest(Path(model['directory'])/filename) != expected:
                    raise ValueError('Frozen model source or score artifact changed')
        bundles.append(bundle)
    return manifest,bundles


def read_test_labels(test_path, ratings_path, expected_test_hash, expected_ratings_hash,
                     history_pairs, users, items):
    """Open the test split only after its supplied hashes and freeze are checked."""
    if digest(test_path) != expected_test_hash or digest(ratings_path) != expected_ratings_hash:
        raise ValueError('Test split or ratings dataset hash mismatch')
    pairs = read_pairs(test_path)
    if len(set(pairs)) != len(pairs) or set(pairs)&set(history_pairs):
        raise ValueError('Test duplicates or history leakage')
    if {u for u,_ in pairs}-set(users) or {i for _,i in pairs}-set(items[1:]):
        raise ValueError('Unknown test user/item')
    selected = set(pairs)
    ratings = {}
    with ratings_path.open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            pair = row['user_id:token'],row['item_id:token']
            if pair in selected:
                value = float(row['rating:float'])
                if pair in ratings or not np.isfinite(value) or not 1<=value<=5:
                    raise ValueError('Invalid test rating')
                ratings[pair] = value
    if set(ratings) != selected:
        raise ValueError('Missing test ratings')
    return ratings


def evaluate_frozen(frozen_root,source_root,ratings_path,out):
    frozen_manifest,bundles = preflight(frozen_root)
    # Marker precedes any test read and refuses replay, even after a failure.
    marker = frozen_root/'TEST-OPENED.json'
    if marker.exists():
        raise ValueError('This frozen audit has already opened test; do not adapt and rerun')
    if out.exists():
        raise ValueError('Refuse to overwrite evaluation output')
    with marker.open('x') as stream:
        json.dump({'status':'test_opening','output':str(out.resolve()),
                   'frozen_manifest_sha256':digest(frozen_root/'manifest.json')},stream)
    out.mkdir(parents=True,exist_ok=False)
    aggregate = {}
    for bundle in bundles:
        seed = bundle['seed']
        folder = frozen_root/str(seed)
        users,items = bundle['users'],bundle['items']
        train,valid = read_pairs(folder/'train.tsv'),read_pairs(folder/'valid.tsv')
        history_pairs = train+valid
        history = grouped(history_pairs)
        ratings = read_test_labels(source_root/f'{seed}-EASE-1'/'test.tsv',ratings_path,
                                   bundle['expected_test_sha256'],bundle['data_sha256'][ratings_path.name],
                                   history_pairs,users,items)
        chosen_users = {u for u,_ in ratings}
        observed = np.zeros((len(users),len(items)),dtype=bool)
        ui,ii = {u:i for i,u in enumerate(users)},{x:i for i,x in enumerate(items)}
        for u,i in history_pairs:
            observed[ui[u],ii[i]] = True
        counts = Counter(i for _,i in train)
        seed_metrics = {}
        for model in bundle['models']:
            with np.load(Path(model['directory'])/'valid-scores.npz',allow_pickle=False) as archive:
                if archive['users'].tolist()!=users or archive['items'].tolist()!=items:
                    raise ValueError('Frozen score IDs mismatch')
                scores = archive['scores']
            recs = recommendations(scores,users,items,observed,bundle['k'])
            seed_metrics[model['name']] = evaluate_ratings(recs,chosen_users,ratings,history,items[1:],counts,bundle['k'],stage='test')
        target = out/str(seed)
        target.mkdir()
        write_json(target/'metrics.json',seed_metrics)
        aggregate[str(seed)] = {name:{'all_observed':row['all_observed']['aggregate'],
                                     'liked_ratings':row['liked_ratings']['aggregate'] if row['liked_ratings'] else None,
                                     'denominators':row['denominators'],
                                     'known_dislike_rate_per_slot':row['known_dislike_rate_per_slot']}
                                for name,row in seed_metrics.items()}
    write_json(out/'summary.json',aggregate)
    output_hashes = {str(path.relative_to(out)):digest(path) for path in [out/'summary.json']+
                     [out/str(bundle['seed'])/'metrics.json' for bundle in bundles]}
    write_json(out/'manifest.json',{'status':'complete','test_read':True,'test_evaluated':True,
                                  'frozen_manifest_sha256':digest(frozen_root/'manifest.json'),
                                  'code_sha256':frozen_manifest['code_sha256'],
                                  'selection_after_test':False,'output_sha256':output_hashes})
    write_json(marker,{'status':'test_evaluated','output':str(out.resolve()),
                       'evaluation_manifest_sha256':digest(out/'manifest.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--research-root',type=Path,required=True)
    freeze.add_argument('--out',type=Path,required=True)
    freeze.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--frozen',type=Path,required=True)
    evaluate.add_argument('--source-root',type=Path,required=True)
    evaluate.add_argument('--ratings',type=Path,required=True)
    evaluate.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='freeze':
        freeze_research(args.research_root,args.out,args.seeds)
    else:
        evaluate_frozen(args.frozen,args.source_root,args.ratings,args.out)


if __name__=='__main__':
    main()
