"""Post-final-test exploratory addressed-evidence study; never opens TEST.

Benchmarking loads TRAIN categories only. Research fitting uses TRAIN masked
probes, chooses checkpoints on reused meta-fit users, and evaluates reused
development users only after both choices have been persisted. These results
are exploratory and cannot constitute fresh held-out confirmation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from categorical_experiment import array_digest, categorical_matrix, masked_episode, state_digest
from exception_experiment import load_source
from joint_field_experiment import aggregate_result, evaluate_logits, joint_metrics, joint_probe_loss, predict_logits
from study import digest, grouped, partition_users, read_pairs, write_json
from exploratory.addressed_evidence.model import AddressedEvidenceModel, build_neighbors


VARIANTS = ('additive', 'pair')
CHECKPOINTS = (10, 30, 60, 100)
MODEL_CONFIG = {'k': 64, 'init_scale': .01, 'pair_bound': 1.0, 'pair_vote_scale': .01}
SOURCE_FILES = ('exploratory/addressed_evidence/run_experiment.py',
    'exploratory/addressed_evidence/model.py', 'exploratory/addressed_evidence/PROTOCOL.md',
    'categorical_experiment.py', 'categorical_field.py', 'joint_field_experiment.py',
    'exception_experiment.py', 'exception_model.py', 'study.py', 'metrics.py', 'hybrid_constraints.py')


def runtime_signature():
    import scipy
    import torch
    return {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': str(torch.__version__)}


def deterministic_runtime():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def training_budget(max_epochs=2000, checkpoints=CHECKPOINTS, evaluation_interval=20,
                    patience=200, min_improvement=1e-4):
    """Explicit pre-run budget; extending requires a new run/output directory."""
    if (type(max_epochs) is not int or max_epochs < 1 or type(evaluation_interval) is not int
            or evaluation_interval < 1 or type(patience) is not int or patience < 0
            or not np.isfinite(min_improvement) or min_improvement < 0):
        raise ValueError('Invalid declared training budget')
    checkpoints = list(checkpoints)
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or any(type(e) is not int or e < 1 for e in checkpoints):
        raise ValueError('Require ordered unique checkpoint epochs')
    grid = [e for e in checkpoints if e <= max_epochs]
    grid += list(range(checkpoints[-1]+evaluation_interval, max_epochs+1, evaluation_interval))
    if max_epochs not in grid:
        grid.append(max_epochs)
    return {'max_epochs': max_epochs, 'checkpoints': grid, 'evaluation_interval': evaluation_interval,
            'early_stopping_patience_epochs': patience, 'meaningful_improvement': float(min_improvement),
            'selection_rule': 'Exact minimum NLL among reached checkpoints; ties retain earlier.',
            'stopping_rule': 'At scheduled checkpoints, stop after patience epochs without cumulative absolute meaningful improvement; zero disables early stopping.'}


def protocol(seeds, budget=None):
    budget = training_budget() if budget is None else budget
    return {'schema_version': 1, 'study_kind': 'post_final_test_exploratory',
        'stage': 'reused_development', 'test_read': False, 'test_evaluated': False,
        'dataset_test_previously_evaluated': True, 'fresh_confirmation': False,
        'seeds': list(seeds), 'variants': list(VARIANTS), 'model_config': MODEL_CONFIG,
        'training_objective': 'joint_recorded_item_rating_multinomial',
        'selection_metric': 'meta_fit_macro_joint_nll',
        'selection_direction': 'minimize; exact ties retain earlier checkpoint',
        'epochs': budget['max_epochs'], 'checkpoints': budget['checkpoints'], 'training_budget': budget,
        'optimizer': 'Adam', 'learning_rate': .001, 'weight_decay': 0., 'batch_size': 16,
        'numerical_threads': 1, 'context_fraction': .8, 'rating_categories': [1, 2, 3, 4, 5],
        'training_reduction': 'Mean masked-probe joint NLL per user, then mean across users.',
        'source_graph': 'Deterministic TRAIN-only neighbor builder; no model-score, validation-label, user-ID embedding or metadata input.',
        'pair_normalization': '.01*sqrt(choose(K,2)); bounded coefficient tanh(pair_raw) times pair_bound',
        'controls': 'Identical allocated tensors, initialization, neighbor graph, episode masks and batch order on shared epochs. Common maximum/checkpoint/stopping policy; realized updates may differ. Pair branch inactive for additive; allocation is not active capacity.',
        'ranking_adapters': {'all_observed': 'logsumexp(all five rating logits)',
                             'liked_record': 'logsumexp(rating 4 and 5 logits)'},
        'normalizer': 'All catalog items except padding and observed context, times five categories.',
        'inference': 'Original TRAIN ratings only; validation never enters query context.',
        'missing_records': 'Competing possible recorded outcomes, not confirmed dislikes.',
        'cohorts': 'Original seeded half-user partition: reused meta-fit selects; disjoint reused development describes performance after both selections.',
        'claims': 'Post-test research using reused validation; not fresh confirmation, SOTA, psychological inference, or a proven novel architecture.',
        'runtime': runtime_signature()}


def load_training_only(source, ratings):
    """Join only TRAIN IDs, without opening validation/test split or ratings."""
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('test_evaluated')
            or manifest.get('validation_used_for_training', True)):
        raise ValueError('Require original TRAIN-fitted validation export')
    if digest(source/'train.tsv') != manifest['split_sha256']['train'] or digest(ratings) != manifest['data_sha256'][ratings.name]:
        raise ValueError('Source TRAIN/raw-data hash mismatch')
    ids = json.loads((source/'ids.json').read_text())
    train = read_pairs(source/'train.tsv')
    present = set(grouped(train))
    users, items = [u for u in ids['users'] if u in present], ids['items']
    if ids.get('padding_index') != 0 or len(users) != len(present) or items[0] != '[PAD]':
        raise ValueError('Original IDs differ')
    if len(set(train)) != len(train) or set(grouped(train)) != set(users):
        raise ValueError('Invalid TRAIN user or pair identities')
    allowed, retained = set(train), {}
    with ratings.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            pair = row['user_id:token'], row['item_id:token']
            if pair in allowed:
                if pair in retained:
                    raise ValueError('Duplicate TRAIN rating')
                retained[pair] = float(row['rating:float'])
    if set(retained) != allowed:
        raise ValueError('Missing TRAIN ratings')
    matrix = categorical_matrix(users, items, [(u, i, retained[u, i]) for u, i in train])
    return users, items, matrix, manifest


def make_model(matrix, neighbors, seed, variant):
    return AddressedEvidenceModel(n_items=matrix.shape[1], seed=seed, variant=variant,
                                  neighbors=neighbors, **MODEL_CONFIG)


def snapshot(model, epoch, seed):
    return {'config': dict(model.config), 'epoch': epoch, 'seed': seed,
            'state_dict': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}


def restore(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = AddressedEvidenceModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint


def tree_digest(value):
    """Stable checksum for resumable tensor/optimizer/RNG payload contents."""
    import torch
    def encode(node):
        if isinstance(node, torch.Tensor):
            return {'tensor_sha256': array_digest(node.detach().cpu().numpy())}
        if isinstance(node, dict):
            return [[type(key).__name__, str(key), encode(child)] for key,child in sorted(node.items(), key=lambda pair:repr(pair[0]))]
        if isinstance(node, (list, tuple)):
            return [encode(child) for child in node]
        return node
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_checkpoint(path, payload):
    import torch
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_snapshot():
    import torch
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {'torch':torch.get_rng_state(), 'python':random.getstate(),
            'numpy':[name,keys.tolist(),position,has_gauss,cached]}


def restore_rng(state):
    import torch
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    name,keys,position,has_gauss,cached=state['numpy']
    np.random.set_state((name,np.asarray(keys,dtype=np.uint32),position,has_gauss,cached))


def train_epoch(model, optimizer, matrix, seed, epoch):
    import torch
    model.train()
    context, mask = masked_episode(matrix, seed, epoch)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 977])).permutation(len(matrix))
    total = 0.
    for start in range(0, len(matrix), 16):
        rows = order[start:start+16]
        optimizer.zero_grad(set_to_none=True)
        loss = joint_probe_loss(model, torch.as_tensor(context[rows]), torch.as_tensor(matrix[rows]),
                                torch.as_tensor(mask[rows]))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite joint training loss')
        loss.backward()
        optimizer.step()
        total += float(loss.detach())*len(rows)
    return {'epoch': epoch, 'macro_train_joint_probe_nll': total/len(matrix),
            'context_sha256': array_digest(context), 'probe_mask_sha256': array_digest(mask),
            'batch_order_sha256': array_digest(order)}


def benchmark(source, ratings, output, seed=2026):
    """One fresh TRAIN-only epoch per variant, used only to assess compute."""
    import torch
    deterministic_runtime()
    sources = source_hashes()
    users, items, matrix, manifest = load_training_only(source, ratings)
    before = time.monotonic()
    neighbors = build_neighbors(matrix, k=MODEL_CONFIG['k'])
    graph_seconds = time.monotonic()-before
    models, results = {}, {}
    for variant in VARIANTS:
        model = make_model(matrix, neighbors, seed, variant)
        initial = state_digest(model)
        models[variant] = initial
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        before = time.monotonic()
        trace = train_epoch(model, optimizer, matrix, seed, 1)
        results[variant] = {'epoch_seconds': time.monotonic()-before,
            'parameter_count': sum(p.numel() for p in model.parameters()), 'initial_state_sha256': initial,
            'training_diagnostic': trace}
    if len(set(models.values())) != 1 or sources != source_hashes():
        raise ValueError('Initial states differ or benchmark sources changed')
    estimated = sum(row['epoch_seconds'] for row in results.values())*max(CHECKPOINTS)*3
    record = {'status': 'complete', 'purpose': 'TRAIN-only runtime estimate; no validation metrics read or evaluated.',
        'test_read': False, 'validation_read': False, 'dataset_test_previously_evaluated': True,
        'seed': seed, 'users': len(users), 'items': len(items)-1, 'graph_seconds': graph_seconds,
        'variants': results, 'estimate_epochs':100, 'estimated_three_seed_serial_fit_seconds': estimated,
        'estimated_three_seed_2000_epoch_serial_fit_seconds':estimated*20,
        'estimate_caution': 'Excludes checkpoint prediction/evaluation/replay and process contention; three workers may reduce wall time.',
        'source_sha256': sources, 'runtime': runtime_signature(),
        'data_sha256': manifest['data_sha256'], 'train_sha256': manifest['split_sha256']['train'],
        'training_categories_sha256': array_digest(matrix), 'neighbors_sha256': array_digest(np.asarray(neighbors))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return record


def train_variant(matrix, neighbors, users, items, validation, fit_users, seed, variant, output, budget=None, resume=False):
    import torch
    started = time.monotonic()
    model = make_model(matrix, neighbors, seed, variant)
    initial = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    budget = training_budget() if budget is None else budget
    trace, grid, best = [], [], None
    best_snapshot, best_logits_sha256, previous_elapsed = None, None, 0.
    meaningful_best, meaningful_epoch, stop_reason = np.inf, 0, 'running'
    signature = {'source_sha256':source_hashes(), 'runtime':runtime_signature(), 'budget':budget,
        'model_config':dict(model.config), 'matrix_sha256':array_digest(matrix),
        'neighbors_sha256':array_digest(np.asarray(neighbors)), 'users':users, 'items':items,
        'meta_labels_sha256':tree_digest(sorted((u,i,r) for (u,i),r in validation.items() if u in fit_users)),
        'fit_users_sha256':tree_digest(sorted(fit_users))}
    if resume:
        latest = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        expected_digest = latest.pop('integrity_sha256')
        if tree_digest(latest) != expected_digest or latest['signature'] != signature:
            raise ValueError('Resume source, protocol, runtime, input or checkpoint seal differs')
        model.load_state_dict(latest['snapshot']['state_dict'],strict=True)
        optimizer.load_state_dict(latest['optimizer'])
        restore_rng(latest['rng'])
        trace,grid,best,best_snapshot = (latest[key] for key in ('trace','grid','best','best_snapshot'))
        best_logits_sha256,previous_elapsed = latest['best_logits_array_sha256'],latest['active_elapsed_seconds']
        meaningful_best,meaningful_epoch,stop_reason = (latest[key] for key in ('meaningful_best','meaningful_epoch','stop_reason'))
        if latest['initial_state_sha256'] != initial or len(trace) != latest['snapshot']['epoch']:
            raise ValueError('Resume initialization or epoch trace differs')
    elif (output/'latest.pt').exists():
        raise ValueError('Existing training state requires explicit resume')
    epochs = range(len(trace)+1,budget['max_epochs']+1) if stop_reason=='running' else ()
    for epoch in epochs:
        trace.append(train_epoch(model, optimizer, matrix, seed, epoch))
        if epoch in budget['checkpoints']:
            logits = predict_logits(model, matrix)
            nll = joint_metrics(logits, users, items, matrix, validation, fit_users)['macro_joint_nll']
            if not np.isfinite(nll):
                raise ValueError('Nonfinite meta-fit objective')
            row = {'epoch': epoch, 'meta_fit_macro_joint_nll': nll}
            grid.append(row)
            if best is None or nll < best['meta_fit_macro_joint_nll']:
                best = dict(row)
                best_snapshot = snapshot(model,epoch,seed)
                best_logits_sha256 = array_digest(logits)
                atomic_checkpoint(output/'best.pt',best_snapshot)
                np.savez_compressed(output/'selected-logits.npz', users=np.asarray(users), items=np.asarray(items), logits=logits)
            print(f'{seed} addressed {variant} epoch {epoch}: reused meta-fit joint NLL {nll:.6f}', flush=True)
            if nll <= meaningful_best-budget['meaningful_improvement']:
                meaningful_best, meaningful_epoch = nll, epoch
            patience = budget['early_stopping_patience_epochs']
            if patience and epoch-meaningful_epoch >= patience:
                stop_reason = 'predeclared_patience'
            elif epoch==budget['max_epochs']:
                stop_reason = 'declared_maximum'
            latest = {'signature':signature, 'snapshot':snapshot(model,epoch,seed),
                'optimizer':optimizer.state_dict(), 'rng':rng_snapshot(), 'trace':trace, 'grid':grid,
                'best':best, 'best_snapshot':best_snapshot, 'meaningful_best':meaningful_best,
                'meaningful_epoch':meaningful_epoch, 'stop_reason':stop_reason, 'initial_state_sha256':initial,
                'best_logits_array_sha256':best_logits_sha256,
                'active_elapsed_seconds':previous_elapsed+time.monotonic()-started}
            latest['integrity_sha256']=tree_digest(latest)
            atomic_checkpoint(output/'latest.pt',latest)
            write_json(output/'training-trace.json',trace)
            write_json(output/'selection-grid.json',grid)
            write_json(output/'progress.json', {'status':stop_reason,'seed':seed,'variant':variant,
                'epoch':epoch,'maximum_epochs':budget['max_epochs'],'best_epoch':best['epoch'],
                'best_meta_fit_macro_joint_nll':best['meta_fit_macro_joint_nll'],
                'active_elapsed_seconds':latest['active_elapsed_seconds'],
                'test_read':False,'fresh_confirmation':False})
            if stop_reason!='running':
                break
    # The atomic latest snapshot is authoritative after an interrupted save;
    # restore its selected state even if a newer unsealed best file survived.
    selected = output/'best.pt'
    atomic_checkpoint(selected,best_snapshot)
    restored, _ = restore(selected)
    replay = predict_logits(restored, matrix)
    if array_digest(replay)!=best_logits_sha256:
        raise AssertionError('Selected checkpoint replay differs from sealed best-logit digest')
    if not resume:
        with np.load(output/'selected-logits.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['logits'], replay):
                raise AssertionError('Selected checkpoint failed exact replay')
    np.savez_compressed(output/'selected-logits.npz', users=np.asarray(users), items=np.asarray(items), logits=replay)
    best.update(checkpoint=selected.name, checkpoint_sha256=digest(selected),
        logits_array_sha256=array_digest(replay), checkpoint_replay_exact=True,
        initial_state_sha256=initial, selection_metric='meta_fit_macro_joint_nll',
        selection_cohort='reused_meta_fit', parameter_count=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=previous_elapsed+time.monotonic()-started, trained_epochs=len(trace), stop_reason=stop_reason,
        elapsed_scope='Cumulative active segments through saved checkpoints plus current segment; interruption-lost unsaved work is not counted.',
        parameter_count_kind='allocated; graph/category support and variant branch affect active capacity',
        resumed=bool(resume))
    write_json(output/'training-trace.json', trace)
    write_json(output/'selection-grid.json', grid)
    write_json(output/'selection.json', best)
    return best


def contribution_diagnostics(model, matrix):
    """Aggregate learned terms over TRAIN-unseen candidates; no labels used."""
    import torch
    sums={'direct':0.,'pair_feature':0.,'applied_pair_contribution':0.}
    count=0
    model.eval()
    with torch.no_grad():
        for start in range(0,len(matrix),16):
            context=torch.as_tensor(matrix[start:start+16])
            diagnostic=model(context,return_diagnostics=True)['diagnostics']
            eligible=context==0
            eligible[:,0]=False
            direct=diagnostic['direct'][eligible].double()
            feature=diagnostic['distinct_pair_feature'][eligible].double()
            applied=diagnostic['distinct_pair_feature']*diagnostic['pair_coefficients'][None]
            if model.variant=='additive':
                applied=torch.zeros_like(applied)
            for name,value in [('direct',direct),('pair_feature',feature),('applied_pair_contribution',applied[eligible].double())]:
                sums[name]+=float(value.square().sum())
            count+=direct.numel()
    rms={name+'_rms':float(np.sqrt(value/max(count,1))) for name,value in sums.items()}
    coefficient=(model.pair_bound*model.pair_raw.detach().tanh())[1:].numpy()
    return {**rms,'candidate_category_count':count,
        'pair_to_direct_rms_ratio':rms['applied_pair_contribution_rms']/max(rms['direct_rms'],1e-15),
        'pair_coefficient_mean_absolute':float(np.abs(coefficient).mean()),
        'pair_coefficient_fraction_abs_above_point95':float(np.mean(np.abs(coefficient)>.95*model.pair_bound)),
        'scope':'Descriptive learned contributions on original TRAIN-unseen candidates; mechanism activity does not establish predictive benefit.'}


def run_seed(source, ratings, metadata, output, seed, budget=None, resume=False):
    deterministic_runtime()
    budget = training_budget() if budget is None else budget
    started, sources = time.monotonic(), source_hashes()
    expected_protocol={**protocol([seed], budget), 'source_sha256': sources}
    if resume:
        if (output/'manifest.json').exists() or json.loads((output/'protocol.json').read_text())!=expected_protocol:
            raise ValueError('Cannot resume a completed run or changed protocol')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'protocol.json', expected_protocol)
    loaded = load_source(source, ratings, metadata)
    users, items, train, valid, training, validation, upstream = (*loaded[:6], loaded[-1])
    del loaded  # Cached baseline predictions and genre metadata are not inputs.
    if upstream.get('validation_used_for_training', True):
        raise ValueError('Source model assimilated validation')
    matrix = categorical_matrix(users, items, training)
    neighbors = build_neighbors(matrix, k=MODEL_CONFIG['k'])
    fit, dev = partition_users(users, seed)
    inputs={'source_manifest_sha256':digest(source/'manifest.json'),
        'data_sha256':upstream['data_sha256'],'split_sha256':upstream['split_sha256'],
        'ids_sha256':digest(source/'ids.json'),'users_array_sha256':array_digest(np.asarray(users)),
        'items_array_sha256':array_digest(np.asarray(items)),'train_categories_sha256':array_digest(matrix),
        'neighbors_sha256':array_digest(np.asarray(neighbors))}
    if resume:
        if json.loads((output/'input-signature.json').read_text())!=inputs:
            raise ValueError('Resume raw-input, source manifest, split or catalog seal differs')
    else:
        write_json(output/'input-signature.json',inputs)
    write_json(output/'cohorts.json', {'meta_fit': sorted(fit), 'development': sorted(dev)})
    np.savez_compressed(output/'training-categories.npz', users=np.asarray(users), items=np.asarray(items), ratings=matrix)
    write_json(output/'catalog.json', {'padding_index': 0, 'items': items})
    for name in ('train.tsv', 'valid.tsv', 'ids.json'):
        shutil.copyfile(source/name, output/name)
    selections = {}
    for variant in VARIANTS:
        folder = output/variant
        folder.mkdir(exist_ok=resume)
        selections[variant] = train_variant(matrix, neighbors, users, items, validation, fit, seed, variant, folder, budget,
                                            resume=resume and (folder/'latest.pt').exists())
    if (len({row['initial_state_sha256'] for row in selections.values()}) != 1
            or len({row['parameter_count'] for row in selections.values()}) != 1):
        raise ValueError('Allocated parameters or initialization differ')
    traces = [json.loads((output/v/'training-trace.json').read_text()) for v in VARIANTS]
    shared_prefix = min(map(len, traces))
    episode_signatures = [[(row['context_sha256'], row['probe_mask_sha256'], row['batch_order_sha256'])
                           for row in trace[:shared_prefix]] for trace in traces]
    if episode_signatures[0] != episode_signatures[1]:
        raise ValueError('Training episode or batch order differs')
    # Both selections are durably written before any development metric call.
    write_json(output/'selection.json', selections)
    evaluated,diagnostics = {},{}
    for variant in VARIANTS:
        with np.load(output/variant/'selected-logits.npz', allow_pickle=False) as saved:
            logits = saved['logits']
        evaluated[variant] = evaluate_logits(logits, users, items, matrix, train, validation, dev)
        restored,_=restore(output/variant/'best.pt')
        diagnostics[variant]=contribution_diagnostics(restored,matrix)
        write_json(output/variant/'diagnostics.json',diagnostics[variant])
    paired = evaluated['pair']['joint']['per_user'], evaluated['additive']['joint']['per_user']
    if set(paired[0]) != set(paired[1]):
        raise ValueError('Development users differ')
    difference = np.asarray([paired[0][u]['joint_nll']-paired[1][u]['joint_nll'] for u in sorted(dev)])
    comparison = {'users': len(dev), 'macro_joint_nll_pair_minus_additive': float(difference.mean()),
        'fraction_users_pair_better': float(np.mean(difference < 0)),
        'interpretation': 'Descriptive reused-development difference only; no fresh held-out or population claim.'}
    write_json(output/'metrics.json', evaluated)
    result = {'models': {name: aggregate_result(row) for name, row in evaluated.items()},
        'selections': selections, 'comparison': comparison, 'diagnostics':diagnostics,
        'cohorts': {'meta_fit': len(fit), 'development': len(dev)},
        'elapsed_seconds': time.monotonic()-started, 'stage': 'reused_development',
        'study_kind': 'post_final_test_exploratory', 'test_read': False, 'fresh_confirmation': False}
    write_json(output/'aggregates.json', result)
    if sources != source_hashes():
        raise RuntimeError('Source changed during exploratory study')
    if (digest(source/'manifest.json')!=inputs['source_manifest_sha256']
            or digest(source/'ids.json')!=inputs['ids_sha256']
            or any(digest(source/f'{part}.tsv')!=inputs['split_sha256'][part] for part in ('train','valid'))
            or any(digest(path)!=inputs['data_sha256'][path.name] for path in (ratings,metadata))):
        raise RuntimeError('Raw input or source manifest changed during exploratory study')
    write_json(output/'manifest.json', {'status': 'complete', 'seed': seed, 'test_read': False,
        'test_evaluated': False, 'dataset_test_previously_evaluated': True, 'fresh_confirmation': False,
        'source_sha256': sources, 'runtime': runtime_signature(), 'source_manifest_sha256': inputs['source_manifest_sha256'],
        'data_sha256': upstream['data_sha256'], 'split_sha256': upstream['split_sha256'],
        'train_categories_sha256': array_digest(matrix), 'neighbors_sha256': array_digest(np.asarray(neighbors)),
        'input_signature_sha256':digest(output/'input-signature.json'),
        'protocol_sha256': digest(output/'protocol.json'),
        'output_sha256': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return result


def curate(research, output, seeds):
    from package_project import check_aggregate_only
    aggregates = {'schema_version': 1, 'stage': 'reused_development', 'test_read': False,
        'dataset_test_previously_evaluated': True, 'fresh_confirmation': False, 'seeds': {}}
    provenance = {'source_sha256': source_hashes(), 'runtime': runtime_signature(), 'seeds': {}}
    budgets = []
    for seed in seeds:
        directory = research/str(seed)
        manifest = json.loads((directory/'manifest.json').read_text())
        seed_protocol=json.loads((directory/'protocol.json').read_text())
        if (manifest['status'] != 'complete' or manifest['test_read'] or manifest['source_sha256'] != source_hashes()
                or manifest['runtime'] != runtime_signature() or manifest['seed']!=seed
                or seed_protocol['seeds']!=[seed]):
            raise ValueError('Exploratory source/runtime provenance differs')
        for name, expected in manifest['output_sha256'].items():
            if not (directory/name).resolve().is_relative_to(directory.resolve()) or digest(directory/name) != expected:
                raise ValueError('Exploratory output hash differs')
        aggregates['seeds'][str(seed)] = json.loads((directory/'aggregates.json').read_text())
        budgets.append(seed_protocol['training_budget'])
        provenance['seeds'][str(seed)] = {'manifest_sha256': digest(directory/'manifest.json'),
            'protocol_sha256': manifest['protocol_sha256'], 'data_sha256': manifest['data_sha256'],
            'split_sha256': manifest['split_sha256'], 'source_manifest_sha256': manifest['source_manifest_sha256']}
    if any(budget != budgets[0] for budget in budgets):
        raise ValueError('Declared budgets differ across seeds')
    check_aggregate_only(aggregates)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'aggregates.json', aggregates)
    write_json(output/'protocol.json', protocol(seeds, budgets[0]))
    write_json(output/'provenance.json', provenance)
    lines=['# Addressed evidence: post-final-test exploratory study','',
        'Standalone operators share TRAIN data, allocated parameters, initialization and a common maximum/checkpoint/stopping policy. Actual training lengths can differ.',
        'Original meta-fit users select checkpoints; original disjoint development users describe results. Both sets are reused.',
        'The dataset TEST was evaluated before this research direction. This runner never opens TEST or final-result artifacts. These are not fresh held-out results.','',
        'All nDCG values below use the reused development cohort: all-observed uses the five-category item-event score; liked uses the liked-record score and only users with at least one observed liked validation rating.','',
        '| Seed | Variant | Joint NLL ↓ | All nDCG@10 | Liked nDCG@10 | Selected epoch | Trained epochs | Stop reason |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    values={variant:[] for variant in VARIANTS}
    for seed in seeds:
        row=aggregates['seeds'][str(seed)]
        for variant in VARIANTS:
            model,selected=row['models'][variant],row['selections'][variant]
            nll=model['joint']['macro_joint_nll']
            observed=model['ranking']['all_observed_adapter']['all_observed']['ndcg@10']
            liked_metrics=model['ranking']['liked_record_adapter']['liked_ratings']
            liked=liked_metrics['ndcg@10'] if liked_metrics else None
            values[variant].append((nll,observed,liked))
            liked_text=f'{liked:.6f}' if liked is not None else 'n/a'
            lines.append(f"| {seed} | {variant} | {nll:.6f} | {observed:.6f} | {liked_text} | {selected['epoch']} | {selected['trained_epochs']} | {selected['stop_reason']} |")
    lines+=['','Descriptive equal-seed means follow. Overlapping splits are not independent datasets.','',
        '| Variant | Mean joint NLL ↓ | Mean all nDCG@10 | Mean liked nDCG@10 |',
        '|---|---:|---:|---:|']
    for variant,rows in values.items():
        liked=[row[2] for row in rows if row[2] is not None]
        liked_text=f'{np.mean(liked):.6f}' if liked else 'n/a'
        lines.append(f'| {variant} | {np.mean([row[0] for row in rows]):.6f} | {np.mean([row[1] for row in rows]):.6f} | {liked_text} |')
    lines+=['','Primary mechanism comparison: pair minus additive macro joint NLL; negative favors pair.','',
        '| Seed | Paired NLL difference | Fraction of users with lower pair NLL |',
        '|---|---:|---:|']
    for seed in seeds:
        comparison=aggregates['seeds'][str(seed)]['comparison']
        lines.append(f"| {seed} | {comparison['macro_joint_nll_pair_minus_additive']:+.6f} | {comparison['fraction_users_pair_better']:.6f} |")
    mean_delta=np.mean([aggregates['seeds'][str(seed)]['comparison']['macro_joint_nll_pair_minus_additive'] for seed in seeds])
    lines+=['',f'Descriptive equal-seed mean paired NLL difference: {mean_delta:+.6f}.',
            'Reaching the maximum is budget-limited; patience stopping does not prove a global optimum. Learned contribution diagnostics are descriptive, not evidence of predictive benefit.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    write_json(output/'SHA256.json', {p.name: digest(p) for p in sorted(output.iterdir())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('benchmark', 'run-seed'):
        command = commands.add_parser(name)
        command.add_argument('--source', type=Path, required=True)
        command.add_argument('--ratings', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--seed', type=int, default=2026)
        if name == 'run-seed':
            command.add_argument('--item-metadata', type=Path, required=True)
            add_budget_arguments(command)
            command.add_argument('--resume',action='store_true')
    running = commands.add_parser('run')
    running.add_argument('--source-root', type=Path, required=True)
    running.add_argument('--ratings', type=Path, required=True)
    running.add_argument('--item-metadata', type=Path, required=True)
    running.add_argument('--out', type=Path, required=True)
    running.add_argument('--evidence', type=Path, required=True)
    running.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    running.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    running.add_argument('--resume',action='store_true')
    add_budget_arguments(running)
    curation = commands.add_parser('curate')
    curation.add_argument('--research', type=Path, required=True)
    curation.add_argument('--out', type=Path, required=True)
    curation.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()
    if args.command == 'benchmark':
        result = benchmark(args.source, args.ratings, args.out, args.seed)
        print(json.dumps({key: result[key] for key in ('variants', 'estimated_three_seed_serial_fit_seconds', 'validation_read')}))
    elif args.command == 'run-seed':
        run_seed(args.source, args.ratings, args.item_metadata, args.out, args.seed, budget_from_args(args),args.resume)
    elif args.command == 'curate':
        curate(args.research, args.out, args.seeds)
    else:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error('Duplicate seeds')
        budget = budget_from_args(args)
        declared={**protocol(args.seeds,budget),'source_sha256':source_hashes()}
        if args.resume:
            if json.loads((args.out/'protocol.json').read_text())!=declared:
                raise ValueError('Resume root protocol differs')
        else:
            args.out.mkdir(parents=True, exist_ok=False)
            write_json(args.out/'protocol.json', declared)
        progress={'status':'running','study_kind':'post_final_test_exploratory','test_read':False,
                  'seeds':args.seeds,'completed_seeds':[],'workers':args.workers}
        write_json(args.out/'progress.json',progress)
        pending=[]
        for seed in args.seeds:
            manifest_path=args.out/str(seed)/'manifest.json'
            if args.resume and manifest_path.exists():
                manifest=json.loads(manifest_path.read_text())
                if manifest['source_sha256']!=source_hashes() or manifest['runtime']!=runtime_signature():
                    raise ValueError('Completed seed provenance differs')
                for name,expected in manifest['output_sha256'].items():
                    if digest(manifest_path.parent/name)!=expected:
                        raise ValueError('Completed seed output differs')
                progress['completed_seeds'].append(seed)
            else:
                pending.append(seed)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_seed, args.source_root/f'{seed}-EASE-1', args.ratings,
                args.item_metadata, args.out/str(seed), seed, budget,
                args.resume and (args.out/str(seed)).exists()):seed for seed in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    progress.update(status='interrupted_or_failed',error=type(error).__name__+': '+str(error))
                    write_json(args.out/'progress.json',progress)
                    raise
                progress['completed_seeds'].append(futures[future])
                progress['completed_seeds'].sort()
                write_json(args.out/'progress.json',progress)
        curate(args.out, args.evidence, args.seeds)
        progress.update(status='complete',aggregate_evidence=str(args.evidence))
        write_json(args.out/'progress.json',progress)


def add_budget_arguments(parser):
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--checkpoints', type=int, nargs='+', default=list(CHECKPOINTS))
    parser.add_argument('--evaluation-interval', type=int, default=20)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--min-improvement', type=float, default=1e-4)


def budget_from_args(args):
    return training_budget(args.max_epochs, args.checkpoints, args.evaluation_interval,
                           args.patience, args.min_improvement)


if __name__ == '__main__':
    main()
