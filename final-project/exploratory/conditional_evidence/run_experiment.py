"""Durable fixed-plan conditional-evidence experiments; original TEST is unused."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'exploratory.conditional_evidence'
SEEDS = (2026, 2027, 2028)
ARMS = ('raw', 'scrambled', 'summary')
LEARNING_RATES = (0.001, 0.0003)
BASE_EPOCHS = 300
EXTENDED_EPOCHS = 600
CHECKPOINT_INTERVAL = 25
BATCH_QUERIES = 8
CANDIDATE_CHUNK = 64
WIDTH = 16


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        with contextlib.suppress(OSError):
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_json(path, value):
    atomic_write(path, lambda stream: stream.write(json_bytes(value)))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != json_bytes(value):
            raise ValueError(f'conflicting durable output: {path.name}')
    else:
        atomic_json(path, value)


def atomic_torch(path, value):
    atomic_write(path, lambda stream: torch.save(value, stream))


def atomic_array(path, value):
    atomic_write(path, lambda stream: np.save(stream, value, allow_pickle=False))


def numerical_setup(seed):
    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def runtime():
    native = importlib.import_module(f'{PACKAGE}.native_scramble')
    return {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
            'system': platform.system(), 'machine': platform.machine(),
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'native_scramble': native.runtime_metadata()}


def source_hashes(require_protocol=True):
    local = ROOT / 'exploratory/conditional_evidence'
    names = ['run_experiment.py', 'model.py', 'data.py', 'evaluation.py']
    if require_protocol:
        names += ['native_scramble.py', 'native_scramble.c', 'retrieval.py', 'workflow.py']
    paths = {local / name for name in names}
    paths |= {path for path in local.iterdir() if path.suffix in ('.py', '.c')
              and not path.name.startswith('test_')}
    protocol = local / 'PROTOCOL.md'
    if require_protocol or protocol.exists():
        paths.add(protocol)
    # Pin the established helpers used by evaluation/data as well.
    paths |= {ROOT / name for name in ('metrics.py', 'societal.py', 'study.py',
              'hybrid_constraints.py', 'categorical_experiment.py',
              'exception_experiment.py', 'exception_model.py',
              'field_reference_comparison.py',
              'exploratory/categorical_reconstruction/run_experiment.py',
              'exploratory/categorical_reconstruction/group_analysis.py',
              'exploratory/categorical_reconstruction/verify_results.py')}
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def check_hashes(mapping):
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if digest(path) != expected:
            raise ValueError(f'changed sealed source/input: {name}')


def rng_state():
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': (state[0], torch.tensor(state[1].astype(np.int64)),
                      state[2], state[3], state[4])}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    name, values, position, gaussian, cached = state['numpy']
    np.random.set_state((name, values.numpy().astype(np.uint32), position, gaussian, cached))


def save_checkpoint(path, model, optimizer, epoch, guard, history):
    payload = {'schema_version': 1, 'epoch': int(epoch), 'guard': guard,
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'rng': rng_state(), 'history': history}
    atomic_torch(path, payload)


def restore_checkpoint(path, model, optimizer, guard):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema_version') != 1 or saved.get('guard') != guard:
        raise ValueError('checkpoint source/input/config guard differs')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return int(saved['epoch']), saved['history']


def best_checkpoint(history):
    rows = history['checkpoints']
    if not rows or any(not np.isfinite(row['meta_ndcg@10']) for row in rows):
        raise ValueError('missing/nonfinite checkpoint scores')
    return min(rows, key=lambda row: (-row['meta_ndcg@10'], row['epoch']))


def choose_configuration(records):
    # Caller supplies learning-rate grid order; earlier checkpoint wins first.
    return min(enumerate(records), key=lambda pair: (
        -pair[1]['selected']['meta_ndcg@10'], pair[1]['selected']['epoch'], pair[0]))[1]


def validate_history(history, target):
    if [row['epoch'] for row in history['epochs']] != list(range(1, target + 1)):
        raise ValueError('training epoch history is incomplete or unordered')
    if [row['epoch'] for row in history['checkpoints']] != list(range(0, target + 1, CHECKPOINT_INTERVAL)):
        raise ValueError('checkpoint search is incomplete or unordered')
    if any(not np.isfinite(row['mean_query_loss']) for row in history['epochs']):
        raise ValueError('nonfinite training history')
    best_checkpoint(history)


def episode_seed(seed, episode, *, epoch=None):
    key = ['conditional-evidence-scramble-v1', int(seed), int(episode['user_index'])]
    key += ['full_catalog'] if epoch is None else ['training', int(epoch), float(episode['retain_fraction'])]
    return int.from_bytes(hashlib.sha256(json_bytes(key)).digest()[:8], 'little')


def modules():
    return tuple(importlib.import_module(f'{PACKAGE}.{name}')
                 for name in ('model', 'data', 'evaluation'))


def categories(inputs):
    for key in ('categories', 'train_categories', 'ratings'):
        if key in inputs:
            value = np.asarray(inputs[key])
            if value.ndim == 2:
                return value
    raise ValueError('data API missing TRAIN category matrix')


def input_hashes(inputs):
    if 'input_hashes' not in inputs:
        raise ValueError('data API missing immutable input hashes')
    return {str(Path(inputs['source_paths'][key]).resolve()): value
            for key, value in inputs['input_hashes'].items()}


def collect_metadata(total, batch):
    metadata = getattr(batch, 'metadata', {})
    for key, value in metadata.items():
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            total[key] = total.get(key, 0.0) + float(value)
    total['graph_build_calls'] = total.get('graph_build_calls', 0) + 1


def build_batch(bank, episode, candidates, arm, seed):
    return bank.build_query(episode['context'], exclude_row=int(episode['user_index']),
                            candidate_ids=candidates, variant=arm,
                            seed=episode.get('scramble_seed', episode_seed(seed, episode)),
                            materialize_graph=arm != 'summary')


def score_episode(model, bank, episode, arm, seed, chunk=CANDIDATE_CHUNK, counters=None):
    ids = np.asarray(episode['candidate_ids'], dtype=np.int64)
    values = []
    with torch.no_grad():
        for start in range(0, len(ids), chunk):
            started = time.perf_counter()
            batch = build_batch(bank, episode, ids[start:start + chunk], arm, seed)
            if counters is not None:
                collect_metadata(counters, batch)
                counters['graph_build_seconds'] = counters.get('graph_build_seconds', 0) + time.perf_counter() - started
            started = time.perf_counter()
            scores = model(batch).reshape(-1)
            if counters is not None:
                counters['forward_seconds'] = counters.get('forward_seconds', 0) + time.perf_counter() - started
            if len(scores) != len(ids[start:start + chunk]) or not torch.isfinite(scores).all():
                raise ValueError('nonfinite/misaligned reader output')
            values.append(scores.detach())
    return torch.cat(values)


def two_pass_backward(model, make_chunks, loss_function, divisor=1.0):
    """Exact chain-rule gradients with bounded activation memory.

    make_chunks() must recreate deterministic inputs in the same order. Readers
    must have no dropout, mutable normalization or other forward state changes.
    No optimizer step occurs between the score and backward passes.
    """
    with torch.no_grad():
        first = torch.cat([model(batch).reshape(-1).detach() for batch in make_chunks()])
    logits = first.clone().requires_grad_(True)
    loss = loss_function(logits)
    derivative, = torch.autograd.grad(loss, logits)
    offset = 0
    for batch in make_chunks():
        scores = model(batch).reshape(-1)
        count = len(scores)
        if not torch.equal(scores.detach(), first[offset:offset + count]):
            raise ValueError('reader/chunk construction is not deterministic between passes')
        torch.autograd.backward(scores, derivative[offset:offset + count] / divisor)
        offset += count
    if offset != len(first):
        raise ValueError('chunk sequence changed between passes')
    return float(loss.detach())


def train_epoch(model, bank, inputs, data_api, optimizer, epoch, seed, arm,
                *, episode_limit=None, chunk=CANDIDATE_CHUNK):
    model.train()
    iterator = iter(data_api.episodes(inputs, epoch=epoch, retain_fraction=(0.8, 0.9),
                                      seed=seed, negatives=128))
    count, loss_sum, counters = 0, 0.0, {}
    started = time.perf_counter()
    while True:
        batch = []
        for _ in range(BATCH_QUERIES):
            if episode_limit is not None and count + len(batch) >= episode_limit:
                break
            episode = next(iterator, None)
            if episode is None:
                break
            batch.append(episode)
        if not batch:
            break
        optimizer.zero_grad(set_to_none=True)
        for episode in batch:
            episode = {**episode, 'scramble_seed': episode_seed(seed, episode, epoch=epoch)}
            ids = np.asarray(episode['candidate_ids'], dtype=np.int64)
            target = torch.as_tensor(episode['target_mask'], dtype=torch.bool)

            def chunks():
                for start in range(0, len(ids), chunk):
                    started = time.perf_counter()
                    graph = build_batch(bank, episode, ids[start:start + chunk], arm, seed)
                    collect_metadata(counters, graph)
                    counters['graph_build_seconds'] = counters.get('graph_build_seconds', 0) + time.perf_counter() - started
                    yield graph

            def loss_fn(logits):
                return data_api.sampled_multinomial_loss(
                    logits, target, episode['unobserved_population_count'])

            loss_sum += two_pass_backward(model, chunks, loss_fn, divisor=len(batch))
            count += 1
        optimizer.step()
    if count == 0:
        raise ValueError('no eligible TRAIN episodes')
    return {'epoch': epoch, 'queries': count, 'mean_query_loss': loss_sum / count,
            'seconds': time.perf_counter() - started, 'resources': counters}


def full_scores(model, bank, inputs, user_indices, arm, seed, chunk=CANDIDATE_CHUNK):
    matrix = categories(inputs)
    scores = np.zeros(matrix.shape, dtype=np.float32)
    counters = {}
    model.eval()
    started = time.perf_counter()
    for index in user_indices:
        ids = np.flatnonzero(matrix[index] == 0)
        ids = ids[ids != 0]
        episode = {'context': matrix[index], 'user_index': int(index), 'candidate_ids': ids}
        scores[index, ids] = score_episode(model, bank, episode, arm, seed, chunk, counters).numpy()
    return scores, {'queries': len(user_indices), 'seconds': time.perf_counter() - started,
                    'resources': counters}


def make_reader(model_api, arm, seed):
    constructor = model_api.SummaryReader if arm == 'summary' else model_api.GraphReader
    return constructor(width=WIDTH, seed=seed)


def load_inputs(args, data_api, seed, load_meta=True):
    return data_api.load_inputs(seed, source_root=args.source_root, ratings=args.ratings,
                                categorical_root=args.categorical_root, load_meta=load_meta)


def user_indices(inputs, name):
    allowed = {str(user) for user in inputs[name]}
    result = [index for index, user in enumerate(inputs['users']) if str(user) in allowed]
    if len(result) != len(allowed):
        raise ValueError('cohort has unknown or repeated identity')
    return result


def meta_truth(inputs):
    for name in ('meta_truth', 'meta_validpairs'):
        if name in inputs:
            return inputs[name]
    raise ValueError('missing meta-fit relevance')


def config_spec(seed, arm, learning_rate):
    return {'seed': seed, 'arm': arm, 'learning_rate': learning_rate, 'width': WIDTH,
            'weight_decay': 1e-4, 'batch_queries': BATCH_QUERIES,
            'candidate_chunk': CANDIDATE_CHUNK, 'retention': [0.8, 0.9],
            'sampled_alternatives': 128, 'base_epochs': BASE_EPOCHS,
            'extension_epochs': EXTENDED_EPOCHS, 'checkpoint_interval': CHECKPOINT_INTERVAL}


def config_id(arm, index):
    return f'{arm}-lr{index}'


def extension_triggers(records):
    result = []
    for seed in map(str, SEEDS):
        for arm in ARMS:
            for index, record in enumerate(records[seed][arm]):
                prefix = {'checkpoints': [row for row in record['history']['checkpoints']
                                         if row['epoch'] <= BASE_EPOCHS]}
                if best_checkpoint(prefix)['epoch'] == BASE_EPOCHS:
                    result.append(f'{seed}/{arm}/lr{index}')
    return result


def train_configuration(directory, inputs, model_api, data_api, evaluation_api,
                        spec, target_epoch, source, *, resume):
    numerical_setup(spec['seed'])
    reader = make_reader(model_api, spec['arm'], spec['seed'])
    optimizer = torch.optim.Adam(reader.parameters(), lr=spec['learning_rate'], weight_decay=1e-4)
    guard = {'source_sha256': source, 'input_sha256': input_hashes(inputs),
             'config': spec, 'runtime': runtime()}
    latest = directory / 'latest.pt'
    if latest.exists():
        if not resume:
            raise ValueError('existing run requires --resume')
        epoch, history = restore_checkpoint(latest, reader, optimizer, guard)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('incomplete nonempty run without durable checkpoint')
        directory.mkdir(parents=True, exist_ok=True)
        epoch = 0
        history = {'config': spec, 'epochs': [], 'checkpoints': [],
                   'parameter_count': sum(p.numel() for p in reader.parameters())}
        save_checkpoint(latest, reader, optimizer, epoch, guard, history)
    bank = model_api.EvidenceBank(categories(inputs), padding=0)
    indices = user_indices(inputs, 'meta_users')

    def validate_checkpoint(at_epoch):
        values, resource = full_scores(reader, bank, inputs, indices, spec['arm'], spec['seed'])
        value = evaluation_api.meta_ndcg(values, inputs['users'], inputs['items'],
                                         categories(inputs), meta_truth(inputs), inputs['meta_users'], k=10)
        row = {'epoch': at_epoch, 'meta_ndcg@10': float(value), 'resources': resource,
               'meta_score_array_sha256': hashlib.sha256(values[indices].tobytes()).hexdigest()}
        history['checkpoints'].append(row)
        checkpoint = directory / 'checkpoints' / f'epoch-{at_epoch:03d}.pt'
        save_checkpoint(checkpoint, reader, optimizer, at_epoch, guard, history)
        row['checkpoint_sha256'] = digest(checkpoint)
        row['checkpoint'] = str(checkpoint.relative_to(directory))
        save_checkpoint(latest, reader, optimizer, at_epoch, guard, history)
        atomic_json(directory / 'history.json', history)
        print(json.dumps({'run': directory.name, 'seed': spec['seed'], **row}), flush=True)

    if (not history['checkpoints'] or
            (epoch % CHECKPOINT_INTERVAL == 0 and
             epoch not in {row['epoch'] for row in history['checkpoints']})):
        validate_checkpoint(epoch)
    for current in range(epoch + 1, target_epoch + 1):
        check_hashes(source)
        row = train_epoch(reader, bank, inputs, data_api, optimizer, current,
                          spec['seed'], spec['arm'])
        history['epochs'].append(row)
        print(json.dumps({'run': directory.name, 'seed': spec['seed'], 'training': row}), flush=True)
        # One epoch is the largest possible lost work on interruption.
        save_checkpoint(latest, reader, optimizer, current, guard, history)
        if current % CHECKPOINT_INTERVAL == 0:
            validate_checkpoint(current)
    if history['epochs'] and history['epochs'][-1]['epoch'] < target_epoch:
        raise ValueError('training budget incomplete')
    validate_history(history, target_epoch)
    record = {'config': spec, 'guard': guard, 'history': history,
              'selected': best_checkpoint(history), 'completed_epoch': target_epoch,
              'peak_process_memory_bytes': peak_memory_bytes(),
              'peak_memory_scope': 'Worker process lifetime peak, including any previous trajectories executed by this process.'}
    atomic_json(directory / 'result.json', record)
    return record


def train_worker(payload):
    """A spawned process owns one trajectory and its global RNG state."""
    model_api, data_api, evaluation_api = modules()
    numerical_setup(payload['spec']['seed'])
    value = data_api.load_inputs(payload['spec']['seed'], source_root=Path(payload['source_root']),
                                 ratings=Path(payload['ratings']),
                                 categorical_root=Path(payload['categorical_root']), load_meta=True)
    check_hashes(payload['source'])
    check_hashes(input_hashes(value))
    return train_configuration(Path(payload['directory']), value, model_api, data_api,
                               evaluation_api, payload['spec'], payload['target'],
                               payload['source'], resume=payload['resume'])


def run_parallel_tasks(payloads, workers, worker=train_worker):
    """Fail promptly and terminate owned workers; durable checkpoints survive.

    Pool's public context-manager contract terminates and joins its processes on
    exceptional exit, including KeyboardInterrupt. No private executor internals
    or unrelated process groups are touched.
    """
    with multiprocessing.get_context('spawn').Pool(processes=workers) as pool:
        return list(pool.imap_unordered(worker, payloads, chunksize=1))


def open_study(args, data_api, *, create=False):
    numerical_setup(SEEDS[0])
    inputs = {seed: load_inputs(args, data_api, seed) for seed in SEEDS}
    plan = {'study': 'conditional_evidence', 'stage': 'reused_development_exploratory',
            'seeds': list(SEEDS), 'arms': list(ARMS), 'learning_rates': list(LEARNING_RATES),
            'source_sha256': source_hashes(), 'runtime': runtime(),
            'inputs': {str(seed): {'signature': value['signature'],
                                  'input_sha256': input_hashes(value)}
                       for seed, value in inputs.items()},
            'original_test_read': False, 'development_read': False}
    path = args.out / 'plan.json'
    if create:
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(path, plan)
    elif not path.exists() or json.loads(path.read_text()) != plan:
        raise ValueError('study source/runtime/input/plan differs; resume refused')
    return inputs, plan


def execute_train(args):
    model_api, data_api, evaluation_api = modules()
    if (args.out / 'SELECTIONS-FROZEN.json').exists():
        raise ValueError('selected study is immutable; training is closed')
    inputs, plan = open_study(args, data_api, create=not args.resume)
    source = plan['source_sha256']
    extension_path = args.out / 'extension.json'
    extending = extension_path.exists()

    def trajectories(target):
        records = {str(seed): {arm: [None] * len(LEARNING_RATES) for arm in ARMS} for seed in SEEDS}
        tasks = []
        for seed in SEEDS:
            for arm in ARMS:
                for index, learning_rate in enumerate(LEARNING_RATES):
                    check_hashes(input_hashes(inputs[seed]))
                    directory = args.out / str(seed) / config_id(arm, index)
                    spec = config_spec(seed, arm, learning_rate)
                    payload = {'directory': str(directory), 'source_root': str(args.source_root),
                               'ratings': str(args.ratings), 'categorical_root': str(args.categorical_root),
                               'spec': spec, 'target': target, 'source': source,
                               'resume': args.resume or directory.exists()}
                    tasks.append((seed, arm, index, payload))
        if args.workers == 1:
            for seed, arm, index, payload in tasks:
                records[str(seed)][arm][index] = train_worker(payload)
        else:
            expected = {(seed, arm, index) for seed, arm, index, _ in tasks}
            for record in run_parallel_tasks([payload for _, _, _, payload in tasks], args.workers):
                spec = record['config']
                seed, arm, index = spec['seed'], spec['arm'], LEARNING_RATES.index(spec['learning_rate'])
                if (seed, arm, index) not in expected:
                    raise ValueError('unexpected or repeated worker result')
                expected.remove((seed, arm, index))
                records[str(seed)][arm][index] = record
            if expected:
                raise ValueError('parallel trajectory results are incomplete')
        return records

    if extending:
        extension = json.loads(extension_path.read_text())
        if extension['plan_sha256'] != digest(args.out / 'plan.json'):
            raise ValueError('extension plan binding differs')
        target = EXTENDED_EPOCHS if extension.get('extend_all') is True else BASE_EPOCHS
        if extension.get('target_epochs') != target or bool(extension.get('triggering_trajectories')) != extension['extend_all']:
            raise ValueError('extension decision is inconsistent')
        records = trajectories(target)
    else:
        records = trajectories(BASE_EPOCHS)
        triggers = extension_triggers(records)
        atomic_json(extension_path, {'plan_sha256': digest(args.out / 'plan.json'),
                    'triggering_trajectories': triggers, 'extend_all': bool(triggers),
                    'target_epochs': EXTENDED_EPOCHS if triggers else BASE_EPOCHS})
        if triggers:
            records = trajectories(EXTENDED_EPOCHS)
    final_extension = json.loads(extension_path.read_text())
    if extension_triggers(records) != final_extension['triggering_trajectories']:
        raise ValueError('recorded extension differs from base-prefix checkpoint maxima')
    check_hashes(source)
    for value in inputs.values():
        check_hashes(input_hashes(value))
    atomic_json(args.out / 'training-summary.json', {
        'status': 'complete', 'plan_sha256': digest(args.out / 'plan.json'),
        'extension': final_extension,
        'selected_by_arm': {seed: {arm: choose_configuration(rows)
                            for arm, rows in arms.items()} for seed, arms in records.items()},
        'trajectory_count': 18, 'development_read': False})


def verified_relative(directory, name, expected):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != expected:
        raise ValueError(f'invalid/changed frozen artifact: {name}')
    return path


def verify_global_seal(root_or_jsonpath):
    candidate = Path(root_or_jsonpath)
    directory = candidate.parent if candidate.name == 'SELECTIONS-FROZEN.json' else candidate
    seal_path = directory / 'SELECTIONS-FROZEN.json'
    if not seal_path.is_file() or (directory / 'SELECTIONS-FROZEN.sha256').read_text().strip() != digest(seal_path):
        raise ValueError('missing/changed global selection seal')
    seal = json.loads(seal_path.read_text())
    if (seal.get('status') != 'frozen' or seal.get('seeds') != list(SEEDS)
            or seal.get('development_read') is not False or
            set(seal.get('seed_manifest_sha256', {})) != {str(seed) for seed in SEEDS}):
        raise ValueError('incomplete all-seed selection seal')
    if seal['plan_sha256'] != digest(directory / 'plan.json'):
        raise ValueError('changed study plan')
    plan = json.loads((directory / 'plan.json').read_text())
    if seal['source_sha256'] != plan['source_sha256']:
        raise ValueError('source seal differs')
    if plan['runtime'] != runtime():
        raise ValueError('runtime differs from sealed execution')
    for key, name in (('training_summary_sha256', 'training-summary.json'),
                      ('meta_screen_sha256', 'meta-screen.json')):
        if seal.get(key) != digest(directory / name):
            raise ValueError(f'changed global selection artifact: {name}')
    check_hashes(seal['source_sha256'])
    for value in plan['inputs'].values():
        check_hashes(value['input_sha256'])
    for seed in SEEDS:
        seed_dir = directory / str(seed)
        manifest_path = seed_dir / 'selection-manifest.json'
        if digest(manifest_path) != seal['seed_manifest_sha256'][str(seed)]:
            raise ValueError('changed seed selection manifest')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('seed') != seed or manifest.get('arms') != list(ARMS):
            raise ValueError('seed selection identity differs')
        required = {'selection.json', 'ids.json', 'reference-metadata.json'} | {
            f'selected-scores/{arm}.npy' for arm in ARMS} | {
            f'selected-scores/{role}.npy' for role in ('EASEexpanded', 'categorical', 'SLIM', 'neighbor')}
        if not required.issubset(manifest['artifacts']):
            raise ValueError('missing selected predictions or provenance')
        for name, expected in manifest['artifacts'].items():
            verified_relative(seed_dir, name, expected)
    return seal


def execute_seal(args):
    model_api, data_api, evaluation_api = modules()
    inputs, plan = open_study(args, data_api)
    if (args.out / 'SELECTIONS-FROZEN.json').exists():
        verify_global_seal(args.out)
        raise ValueError('already sealed; no replacement allowed')
    summary_path = args.out / 'training-summary.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'complete' or summary['plan_sha256'] != digest(args.out / 'plan.json'):
        raise ValueError('all trajectories must complete before selection sealing')
    manifests, meta_values = {}, {}
    for seed in SEEDS:
        value = inputs[seed]
        seed_dir = args.out / str(seed)
        bank = model_api.EvidenceBank(categories(value), padding=0)
        selected, artifacts, metrics = {}, {}, {}
        for arm in ARMS:
            rows = []
            for index, _ in enumerate(LEARNING_RATES):
                path = seed_dir / config_id(arm, index) / 'result.json'
                record = json.loads(path.read_text())
                validate_history(record['history'], summary['extension']['target_epochs'])
                if (record['config'] != config_spec(seed, arm, LEARNING_RATES[index]) or
                        record['selected'] != best_checkpoint(record['history']) or
                        record['completed_epoch'] != summary['extension']['target_epochs']):
                    raise ValueError('trajectory or earliest maximum differs')
                rows.append(record)
                artifacts[str(path.relative_to(seed_dir))] = digest(path)
                for row in record['history']['checkpoints']:
                    saved = verified_relative(path.parent, row['checkpoint'], row['checkpoint_sha256'])
                    artifacts[str(saved.relative_to(seed_dir))] = digest(saved)
            record = choose_configuration(rows)
            if record != summary['selected_by_arm'][str(seed)][arm]:
                raise ValueError('declared selection differs from complete candidate grid')
            lr_index = LEARNING_RATES.index(record['config']['learning_rate'])
            run_dir = seed_dir / config_id(arm, lr_index)
            checkpoint = verified_relative(run_dir, record['selected']['checkpoint'],
                                           record['selected']['checkpoint_sha256'])
            model = make_reader(model_api, arm, seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=record['config']['learning_rate'], weight_decay=1e-4)
            restore_checkpoint(checkpoint, model, optimizer, record['guard'])
            scores, resource = full_scores(model, bank, value, list(range(len(value['users']))), arm, seed)
            meta = evaluation_api.meta_ndcg(scores, value['users'], value['items'], categories(value),
                                            meta_truth(value), value['meta_users'])
            if meta != record['selected']['meta_ndcg@10']:
                raise ValueError('selected checkpoint meta replay differs')
            replay_digest = hashlib.sha256(scores[user_indices(value, 'meta_users')].tobytes()).hexdigest()
            if replay_digest != record['selected']['meta_score_array_sha256']:
                raise ValueError('selected checkpoint exact score replay differs')
            file = seed_dir / 'selected-scores' / f'{arm}.npy'
            atomic_array(file, scores)
            selected[arm] = {'config': record['config'], 'selected': record['selected'],
                             'full_catalog_resources': resource, 'meta_ndcg@10': float(meta)}
            artifacts[str(file.relative_to(seed_dir))] = digest(file)
            artifacts[str(checkpoint.relative_to(seed_dir))] = digest(checkpoint)
            metrics[arm] = float(meta)
        references, reference_metadata = evaluation_api.load_references(
            seed, value, args.reference_root, value['users'], value['items'])
        references['neighbor'] = evaluation_api.neighbor_scores(categories(value), k=16)
        for name, scores in references.items():
            file = seed_dir / 'selected-scores' / f'{name}.npy'
            atomic_array(file, scores)
            artifacts[str(file.relative_to(seed_dir))] = digest(file)
            metrics[name] = float(evaluation_api.meta_ndcg(scores, value['users'], value['items'],
                                  categories(value), meta_truth(value), value['meta_users']))
        atomic_json(seed_dir / 'selection.json', selected)
        atomic_json(seed_dir / 'ids.json', {'users': value['users'], 'items': value['items']})
        atomic_json(seed_dir / 'reference-metadata.json', reference_metadata)
        for name in ('selection.json', 'ids.json', 'reference-metadata.json'):
            artifacts[name] = digest(seed_dir / name)
        manifest = {'seed': seed, 'arms': list(ARMS), 'artifacts': artifacts,
                    'input_signature': value['signature'], 'meta_metrics': metrics,
                    'source_sha256': plan['source_sha256'], 'development_read': False}
        atomic_json(seed_dir / 'selection-manifest.json', manifest)
        manifests[str(seed)] = digest(seed_dir / 'selection-manifest.json')
        meta_values[str(seed)] = metrics
    screen = evaluation_api.meta_screen(meta_values, 'raw', ('summary', 'scrambled'),
                                         ('EASEexpanded', 'categorical', 'SLIM', 'neighbor'))
    atomic_json(args.out / 'meta-screen.json', screen)
    seal = {'status': 'frozen', 'seeds': list(SEEDS), 'seed_manifest_sha256': manifests,
            'source_sha256': plan['source_sha256'], 'plan_sha256': digest(args.out / 'plan.json'),
            'training_summary_sha256': digest(summary_path),
            'meta_screen_sha256': digest(args.out / 'meta-screen.json'),
            'development_read': False, 'original_test_read': False}
    atomic_json(args.out / 'SELECTIONS-FROZEN.json', seal)
    atomic_write(args.out / 'SELECTIONS-FROZEN.sha256',
                 lambda stream: stream.write((digest(args.out / 'SELECTIONS-FROZEN.json') + '\n').encode()))
    verify_global_seal(args.out)


def verify_assessment_release(directory, release_path):
    directory = Path(directory)
    seal = verify_global_seal(directory)
    release = json.loads(Path(release_path).read_text())
    if (release.get('status') != 'ready' or release.get('base_seal_sha256') !=
            digest(directory / 'SELECTIONS-FROZEN.json') or
            release.get('extension_status') not in ('not_triggered', 'complete')):
        raise ValueError('assessment requires the root-owned extension release')
    if release.get('meta_screen_sha256') != seal['meta_screen_sha256']:
        raise ValueError('extension release does not bind the meta-only gate')
    screen = json.loads((directory / 'meta-screen.json').read_text())
    if (release['extension_status'] == 'not_triggered') == bool(screen['passed']):
        raise ValueError('extension release contradicts the frozen meta-only gate')
    if release['extension_status'] == 'complete':
        if not release.get('extension_artifacts_sha256'):
            raise ValueError('extension outputs are not sealed')
        for name, expected in release['extension_artifacts_sha256'].items():
            verified_relative(directory, name, expected)
    return release


def peak_memory_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def execute_benchmark(args, *, smoke=False):
    model_api, data_api, _ = modules()
    numerical_setup(args.seed)
    source = source_hashes(require_protocol=False)
    args.out.mkdir(parents=True, exist_ok=False)
    value = load_inputs(args, data_api, args.seed, load_meta=False)
    started = time.perf_counter()
    bank = model_api.EvidenceBank(categories(value), padding=0)
    bank_seconds = time.perf_counter() - started
    model = make_reader(model_api, args.arm, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATES[0], weight_decay=1e-4)
    training = train_epoch(model, bank, value, data_api, optimizer, 1, args.seed, args.arm,
                           episode_limit=args.episodes, chunk=args.candidate_chunk)
    eligible = [index for index, row in enumerate(categories(value)) if np.count_nonzero(row) >= 2]
    _, inference = full_scores(model, bank, value, eligible[:args.full_users],
                               args.arm, args.seed, args.candidate_chunk)
    guard = {'source_sha256': source, 'input_sha256': input_hashes(value),
             'config': {'nonstudy': True, 'seed': args.seed, 'arm': args.arm}, 'runtime': runtime()}
    if smoke:
        checkpoint = args.out / 'smoke-latest.pt'
        save_checkpoint(checkpoint, model, optimizer, 1, guard, {'training': training})
        copy = make_reader(model_api, args.arm, args.seed)
        copy_optimizer = torch.optim.Adam(copy.parameters(), lr=LEARNING_RATES[0], weight_decay=1e-4)
        restore_checkpoint(checkpoint, copy, copy_optimizer, guard)
        if any(not torch.equal(v, copy.state_dict()[k]) for k, v in model.state_dict().items()):
            raise ValueError('smoke checkpoint replay differs')
    check_hashes(source)
    check_hashes(input_hashes(value))
    episodes_per_epoch = data_api.episode_status(value)['eligible_users'] * 2
    seconds_per_episode = training['seconds'] / training['queries']
    seconds_per_full_user = inference['seconds'] / inference['queries']
    result = {
        'status': 'complete', 'study_result': False,
        'stage': 'nonstudy_tiny_smoke' if smoke else 'nonstudy_train_only_timing',
        'source_sha256': source, 'input_signature': value['signature'], 'runtime': runtime(),
        'seed': args.seed, 'arm': args.arm, 'candidate_chunk': args.candidate_chunk,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'bank_build_seconds': bank_seconds, 'training': training, 'full_catalog': inference,
        'peak_process_memory_bytes': peak_memory_bytes(),
        'extrapolation': {'assumption': 'Linear scaling of this small TRAIN-only timing sample; not a guaranteed runtime.',
                          'episodes_per_epoch': episodes_per_epoch,
                          'one_epoch_seconds': episodes_per_epoch * seconds_per_episode,
                          'one_trajectory_300_epoch_seconds': 300 * episodes_per_epoch * seconds_per_episode,
                          'one_checkpoint_471_user_seconds': 471 * seconds_per_full_user,
                          'one_trajectory_13_meta_checkpoints_seconds': 13 * 471 * seconds_per_full_user,
                          'one_selected_catalog_943_user_seconds': len(value['users']) * seconds_per_full_user},
        'meta_labels_read': False, 'development_read': False, 'original_test_read': False,
    }
    atomic_json(args.out / 'benchmark.json', result)
    print(json.dumps(result), flush=True)


def execute_assess(args):
    numerical_setup(SEEDS[0])
    _, data_api, evaluation_api = modules()
    release = verify_assessment_release(args.out, args.assessment_release)
    inputs, plan = open_study(args, data_api)
    seal_hash = digest(args.out / 'SELECTIONS-FROZEN.json')
    release_hash = digest(args.assessment_release)
    marker_path = args.out / 'DEVELOPMENT-OPENED.json'
    if marker_path.exists() or args.evidence.exists():
        if not getattr(args, 'resume', False) or not marker_path.exists():
            raise ValueError('assessment output exists; exact interrupted replay requires --resume')
        prior = json.loads(marker_path.read_text())
        if (prior.get('status') not in ('started', 'complete') or
                prior.get('base_seal_sha256') != seal_hash or
                prior.get('assessment_release_sha256') != release_hash or
                prior.get('original_test_read') is not False):
            raise ValueError('assessment resume bindings differ')
        if prior['status'] == 'complete':
            if digest(args.out / 'aggregates.json') != prior['aggregate_sha256']:
                raise ValueError('completed assessment aggregate changed')
            manifest_path = args.out / 'manifest.json'
            if manifest_path.exists():
                provenance = json.loads(manifest_path.read_text())
                if (provenance['base_seal_sha256'] != seal_hash or
                        provenance['assessment_release_sha256'] != release_hash or
                        provenance['source_sha256'] != plan['source_sha256'] or
                        provenance['aggregate_sha256'] != prior['aggregate_sha256'] or
                        provenance['development_marker_sha256'] != digest(marker_path)):
                    raise ValueError('completed assessment provenance changed')
                for name, expected in provenance['private_output_sha256'].items():
                    verified_relative(args.out, name, expected)
                export_evidence(args.evidence, json.loads((args.out / 'aggregates.json').read_text()),
                                plan, provenance)
                return
    marker = {'status': 'started', 'base_seal_sha256': seal_hash,
              'assessment_release_sha256': release_hash, 'original_test_read': False}
    # Durable marker precedes the first DEV relevance access.
    if not marker_path.exists():
        atomic_json(marker_path, marker)
    public, private_files, dev_values = {}, {}, {}
    screen = json.loads((args.out / 'meta-screen.json').read_text())
    for seed in SEEDS:
        value = inputs[seed]
        truth = data_api.load_development(value, args.out,
                    verify_global_seal=lambda path: verify_assessment_release(path, args.assessment_release))
        directory = args.out / str(seed)
        selection_manifest = json.loads((directory / 'selection-manifest.json').read_text())
        roles = [*ARMS, 'EASEexpanded', 'categorical', 'SLIM', 'neighbor']
        rows, private = {}, {}
        for role in roles:
            scores = np.load(directory / 'selected-scores' / f'{role}.npy', allow_pickle=False)
            result = evaluation_api.evaluate_scores(scores, value['users'], value['items'],
                categories(value), truth['truth'], value['dev_users'], ratings=truth['ratings'], k=10)
            rows[role], private[role] = result['public'], result['per_user']
        paired = {f'raw_minus_{role}': evaluation_api.paired_summary(
                    private['raw']['all_observed'], private[role]['all_observed'], seed=seed + 8000)
                  for role in roles if role != 'raw'}
        immutable_json(directory / 'development-private.json', private)
        private_files[str(seed) + '/development-private.json'] = digest(directory / 'development-private.json')
        public[str(seed)] = {'models': rows, 'paired_differences': paired,
                            'selections': json.loads((directory / 'selection.json').read_text()),
                            'meta_metrics': selection_manifest['meta_metrics'],
                            'input_signature': value['signature']}
        dev_values[str(seed)] = {role: rows[role]['all_observed']['ndcg@10'] for role in roles}
    verify_assessment_release(args.out, args.assessment_release)
    aggregates = {'status': 'complete', 'study': 'conditional_evidence',
                  'stage': 'previously_used_development_exploratory', 'fresh_confirmation': False,
                  'original_test_read': False, 'seeds': public, 'meta_screen': screen,
                  'development_gate': evaluation_api.development_gate(dev_values, screen),
                  'extension_status': release['extension_status'],
                  'limitations': ['Development users and observations were exposed in earlier research.',
                    'The three splits overlap; paired intervals are descriptive and not multiplicity adjusted.',
                    'Missing records are not confirmed dislikes. No first-ever architecture claim.',
                    'The sampled denominator is unbiased before taking its logarithm; the loss is not unbiased full softmax.']}
    immutable_json(args.out / 'aggregates.json', aggregates)
    marker.update(status='complete', development_read=True,
                  aggregate_sha256=digest(args.out / 'aggregates.json'))
    atomic_json(args.out / 'DEVELOPMENT-OPENED.json', marker)
    provenance = {'status': 'complete', 'source_sha256': plan['source_sha256'], 'runtime': runtime(),
                  'plan_sha256': digest(args.out / 'plan.json'), 'base_seal_sha256': seal_hash,
                  'assessment_release_sha256': digest(args.assessment_release),
                  'aggregate_sha256': digest(args.out / 'aggregates.json'),
                  'development_marker_sha256': digest(args.out / 'DEVELOPMENT-OPENED.json'),
                  'private_output_sha256': private_files, 'original_test_read': False}
    immutable_json(args.out / 'manifest.json', provenance)
    export_evidence(args.evidence, aggregates, plan, provenance)


def export_evidence(directory, aggregates, plan, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory / 'aggregates.json', aggregates)
    immutable_json(directory / 'protocol.json', plan)
    immutable_json(directory / 'provenance.json', provenance)
    immutable_json(directory / 'SHA256.json', {name: digest(directory / name)
                for name in ('aggregates.json', 'protocol.json', 'provenance.json')})


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    commands = main.add_subparsers(dest='command', required=True)
    for name in ('benchmark', 'tiny-smoke', 'train', 'seal', 'assess'):
        command = commands.add_parser(name)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--source-root', type=Path, default=ROOT / 'runs/research-v2')
        command.add_argument('--categorical-root', type=Path, default=ROOT / 'runs/categorical-reconstruction-v1')
        command.add_argument('--ratings', type=Path, required=True)
        if name in ('benchmark', 'tiny-smoke'):
            command.add_argument('--seed', type=int, choices=SEEDS, default=2026)
            command.add_argument('--arm', choices=ARMS, default='raw')
            command.add_argument('--episodes', type=int, default=8)
            command.add_argument('--full-users', type=int, default=1)
            command.add_argument('--candidate-chunk', type=int, default=CANDIDATE_CHUNK)
        elif name == 'train':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--workers', type=int, choices=(1, 3, 6), default=3)
        elif name == 'seal':
            command.add_argument('--reference-root', type=Path, default=ROOT / 'runs/categorical-reconstruction-v1')
        elif name == 'assess':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--assessment-release', type=Path, required=True)
            command.add_argument('--evidence', type=Path, required=True)
    return main


def main():
    args = parser().parse_args()
    args.out = args.out.resolve()
    numerical_setup(SEEDS[0])
    if args.command in ('benchmark', 'tiny-smoke'):
        if min(args.episodes, args.full_users, args.candidate_chunk) < 1:
            raise ValueError('benchmark budgets must be positive')
        execute_benchmark(args, smoke=args.command == 'tiny-smoke')
    elif args.command == 'train':
        execute_train(args)
    elif args.command == 'seal':
        execute_seal(args)
    elif args.command == 'assess':
        execute_assess(args)


if __name__ == '__main__':
    main()
