"""Prepare a reviewable, sealed plan, then reserve one final TEST batch.

The prepare command never opens test files. The run command repeats every
preflight and exclusively creates runs/FINAL-TEST-BATCH.json before launching
any evaluator. A failed or interrupted batch remains reserved: no automatic
retry, changed choices, or replacement outputs are permitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SEEDS = (2026, 2027, 2028)
THREADS = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
          'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
TRACKS = {
    'exception': ('exception_evaluation', 'frozen-exception-v3', 'final-exception-v3'),
    'conditional': ('categorical_evaluation', 'frozen-adaptive-v1', 'final-adaptive-v1'),
    'joint100': ('joint_evaluation', 'frozen-joint-field-v1', 'final-joint-field-v1'),
    'joint400': ('joint_convergence_evaluation', 'frozen-joint-field-convergence-v1', 'final-joint-field-convergence-v1'),
    'references': ('field_reference_evaluation', 'frozen-field-references-v1', 'final-field-references-v1'),
}
ANALYSIS_FILES = ('run_final_batch.py', 'compare_frozen.py', 'compare_fields_final.py', 'audit_study.py',
                  'evidence/EVALUATION_PROTOCOL.md')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity_digest(users, items):
    return hashlib.sha256(json.dumps([users, items], separators=(',', ':')).encode()).hexdigest()


def collect_preflight(root):
    """Validate every frozen family and compare signatures without TEST I/O."""
    from freeze import load_frozen
    from categorical_evaluation import load_training
    from joint_evaluation import runtime_signature
    records, shared_sources, references = {}, {}, {}

    def source_signature(mapping):
        for name, expected in mapping.items():
            if name in shared_sources and shared_sources[name] != expected:
                raise ValueError(f'Frozen families disagree on shared source: {name}')
            shared_sources[name] = expected

    def dataset_signature(seed, data, splits, users, items, folder):
        signature = {'data_sha256': data, 'split_sha256': splits,
                     'ordered_ids_sha256': identity_digest(users, items)}
        if set(splits) != {'train', 'valid', 'test'}:
            raise ValueError('Incomplete original split signature')
        for part in ('train', 'valid'):
            if digest(folder/f'{part}.tsv') != splits[part]:
                raise ValueError('Frozen TRAIN/validation signature differs')
        if seed in references and signature != references[seed]:
            raise ValueError(f'Frozen families disagree on data, split or ordered IDs: {seed}')
        references[seed] = signature

    primary = root/'runs/frozen-v3'
    index = read_json(primary/'manifest.json')
    if index['seeds'] != list(SEEDS) or index.get('test_read'):
        raise ValueError('Invalid primary freeze index')
    primary_bundles = {}
    for seed in SEEDS:
        folder = primary/str(seed)
        bundle, arrays = load_frozen(folder)
        if bundle['seed'] != seed or digest(folder/'freeze.json') != index['bundles'][str(seed)]['freeze_sha256']:
            raise ValueError('Primary freeze index differs from bundle')
        signatures = [value['manifest'] for value in bundle['sources'].values()]
        for signature in signatures:
            dataset_signature(seed, signature['data_sha256'], signature['split_sha256'],
                              arrays['users'].tolist(), arrays['items'].tolist(), folder)
        if bundle['expected_test_sha256'] != references[seed]['split_sha256']['test']:
            raise ValueError('Primary TEST signature differs')
        source_signature(bundle['code_sha256'])
        primary_bundles[str(seed)] = digest(folder/'freeze.json')
    records['primary'] = {'root': str(primary), 'manifest_sha256': digest(primary/'manifest.json'),
                          'bundle_sha256': primary_bundles}
    for name, (module_name, frozen_name, _) in TRACKS.items():
        folder = root/'runs'/frozen_name
        if (folder/'TEST-OPENED.json').exists():
            raise ValueError(f'TEST was already reserved for {name}')
        module = importlib.import_module(module_name)
        manifest, bundles = module.preflight(folder)
        if manifest['seeds'] != list(SEEDS) or [b['seed'] for b in bundles] != list(SEEDS):
            raise ValueError('Frozen seed sets/order differ')
        source_signature(manifest['code_sha256'])
        for bundle in bundles:
            seed = bundle['seed']
            location = folder/str(seed)
            if name == 'exception':
                users, items = bundle['users'], bundle['items']
            elif name == 'references':
                ids = read_json(location/'score-ids.json')
                users, items = ids['users'], ids['items']
            else:
                users, items, *_ = load_training(location)
            dataset_signature(seed, bundle['data_sha256'], bundle['split_sha256'], users, items, location)
            if bundle['expected_test_sha256'] != references[seed]['split_sha256']['test']:
                raise ValueError('Recorded TEST signature differs')
        records[name] = {'root': str(folder), 'manifest_sha256': digest(folder/'manifest.json'),
                         'bundle_sha256': manifest['bundle_sha256']}
    return {'frozen': records, 'per_seed_data': {str(k): v for k, v in references.items()},
            'shared_source_sha256': shared_sources, 'runtime': runtime_signature()}


def build_jobs(root, python, ratings):
    root, python, ratings = Path(root), str(python), str(ratings)
    def path(value):
        return str(root/value)
    tests = [value for seed in SEEDS for value in ('--test', f'{seed}={path(f"runs/research-v2/{seed}-EASE-1/test.tsv")}')]
    jobs = []
    for seed in SEEDS:
        output = path(f'runs/final-v3/{seed}')
        jobs.append({'name': f'primary-{seed}', 'output': output, 'argv': [python,
            path('final_evaluate.py'), '--frozen', path(f'runs/frozen-v3/{seed}'),
            '--test', path(f'runs/research-v2/{seed}-EASE-1/test.tsv'), '--out', output]})
    for name, (module, frozen, final) in TRACKS.items():
        output = path('runs/'+final)
        argv = [python, path(module+'.py'), 'evaluate', '--frozen', path('runs/'+frozen), '--ratings', ratings]
        argv += ['--source-root', path('runs/research-v2')] if name == 'exception' else tests
        jobs.append({'name': name, 'output': output, 'argv': argv+['--out', output]})
    analysis = [{'name': 'primary-comparisons', 'output': path('evidence/final-comparisons-v3'),
        'argv': [python, path('compare_frozen.py'), '--results']+
        [path(f'runs/final-v3/{seed}') for seed in SEEDS]+['--freezes']+
        [path(f'runs/frozen-v3/{seed}') for seed in SEEDS]+
        ['--bootstrap', '20000', '--confidence', '0.95', '--seed', '2026', '--out', path('evidence/final-comparisons-v3')]}]
    argv = [python, path('compare_fields_final.py')]
    for name in ('conditional', 'joint100', 'joint400', 'references'):
        _, frozen, final = TRACKS[name]
        argv += ['--'+name, path('runs/'+frozen), path('runs/'+final)]
    output = path('evidence/final-field-comparisons-v1')
    analysis.append({'name': 'field-comparisons', 'output': output, 'argv': argv+['--out', output]})
    for seed in SEEDS:
        output = path(f'evidence/final-societal-{seed}')
        analysis.append({'name': f'societal-{seed}', 'output': output,
            'argv': [python, path('audit_study.py'), '--final', path(f'runs/final-v3/{seed}'),
                '--frozen', path(f'runs/frozen-v3/{seed}'), '--data', str(Path(ratings).parent),
                '--out', output, '--aggregate-only']})
    return jobs, analysis


def build_plan(root, python, ratings):
    root, python, ratings = Path(root).resolve(), Path(os.path.abspath(python)), Path(ratings).resolve()
    # Preserve the venv path: resolving its interpreter symlink would equate
    # distinct environments that happen to use the same Python binary.
    if python != Path(os.path.abspath(sys.executable)):
        raise ValueError('Requested Python must be the current verified interpreter path')
    marker = root/'runs/FINAL-TEST-BATCH.json'
    if marker.exists():
        raise ValueError('Final batch already reserved; no repeated opening')
    jobs, analysis = build_jobs(root, python, ratings)
    for job in jobs+analysis:
        if Path(job['output']).exists():
            raise ValueError(f'Refuse existing output: {job["output"]}')
    state = collect_preflight(root)
    ratings_hashes = {row['data_sha256'].get(ratings.name) for row in state['per_seed_data'].values()}
    if None in ratings_hashes or len(ratings_hashes) != 1 or not ratings.is_file():
        raise ValueError('Ratings path/name does not match frozen data signatures')
    # Byte checksum only: this does not parse any rating or held-out label.
    if digest(ratings) != next(iter(ratings_hashes)):
        raise ValueError('Ratings dataset bytes differ from the frozen data signature')
    raw_data = {}
    for row in state['per_seed_data'].values():
        for name, expected in row['data_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Raw dataset signature must use a plain filename')
            data_path = ratings.parent/name
            if digest(data_path) != expected:
                raise ValueError('Raw dataset bytes differ from the frozen data signature')
            raw_data[str(data_path)] = expected
    for seed in SEEDS:
        if not (root/f'runs/research-v2/{seed}-EASE-1/test.tsv').is_file():
            raise ValueError('Expected TEST path is absent')
    return {'schema_version': 1, 'stage': 'pre_test', 'test_read': False, 'root': str(root),
            'python': str(python), 'python_executable_sha256': digest(python), 'ratings': str(ratings),
            'seeds': list(SEEDS), 'max_parallel_jobs': 3, 'environment': THREADS,
            'marker': str(marker), 'jobs': jobs, 'analysis_jobs': analysis,
            'analysis_sha256': {name: digest(root/name) for name in ANALYSIS_FILES}, 'raw_data_sha256': raw_data,
            'preflight': state, 'policy': 'One fixed final batch; evaluate every frozen choice; no refit, test selection, automatic retry or changed-choice retry.'}


def prepare(plan_path, python, ratings, root=ROOT):
    plan = build_plan(root, python, ratings)
    path = Path(plan_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, plan, exclusive=True)
    path.with_suffix(path.suffix+'.sha256').write_text(digest(path)+'\n')
    return plan


def run_plan(plan_path):
    path = Path(plan_path).resolve()
    if digest(path) != path.with_suffix(path.suffix+'.sha256').read_text().strip():
        raise ValueError('Final plan digest differs')
    plan = read_json(path)
    if Path(plan['root']).resolve() != ROOT.resolve():
        raise ValueError('Final plan belongs to another workspace')
    expected = build_plan(ROOT, Path(plan['python']), Path(plan['ratings']))
    if plan != expected:
        raise ValueError('Final plan, frozen choices, sources or runtime changed')
    marker = Path(plan['marker'])
    marker.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'reserved_before_test', 'plan': str(path), 'plan_sha256': digest(path),
              'selection_after_test': False, 'results': {}}
    write_json(marker, record, exclusive=True)
    logs = ROOT/'runs/final-batch-logs'
    try:
        logs.mkdir(exist_ok=False)
        environment = {**os.environ, **plan['environment']}
        def execute(job):
            for name, expected_hash in plan['analysis_sha256'].items():
                if digest(ROOT/name) != expected_hash:
                    raise ValueError('Sealed analysis source changed before execution')
            with (logs/(job['name']+'.log')).open('x') as stream:
                result = subprocess.run(job['argv'], cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, check=False)
            output = Path(job['output'])/'manifest.json'
            completed = result.returncode == 0 and output.is_file() and read_json(output).get('status') == 'complete'
            return job['name'], {'returncode': result.returncode, 'complete': completed,
                                'manifest_sha256': digest(output) if output.is_file() else None}
        with ThreadPoolExecutor(max_workers=plan['max_parallel_jobs']) as pool:
            for name, result in pool.map(execute, plan['jobs']):
                record['results'][name] = result
                write_json(marker, record)
        if not all(row['complete'] for row in record['results'].values()):
            raise RuntimeError('A final evaluator failed; batch remains reserved and no retry is performed')
        for job in plan['analysis_jobs']:
            name, result = execute(job)
            record['results'][name] = result
            write_json(marker, record)
            if not result['complete']:
                raise RuntimeError('A fixed comparison failed; batch remains reserved')
        record['status'] = 'complete'
        write_json(marker, record)
    except BaseException as error:
        record['status'], record['error'] = 'failed_reserved', type(error).__name__+': '+str(error)
        write_json(marker, record)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    before = commands.add_parser('prepare')
    before.add_argument('--plan', type=Path, required=True)
    before.add_argument('--python', type=Path, default=Path(sys.executable))
    before.add_argument('--ratings', type=Path, required=True)
    final = commands.add_parser('run')
    final.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.plan, args.python, args.ratings) if args.command == 'prepare' else run_plan(args.plan)
    print(json.dumps({'status': result.get('status', 'prepared'), 'plan': str(args.plan),
                      'test_read': args.command == 'run'}))


if __name__ == '__main__':
    main()
