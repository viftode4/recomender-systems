"""Explicit, provenance-recorded continuation of the sealed conditional study.

No activation occurs on import unless a parent launcher supplied a pinned
execution-overlay manifest. This also installs the overlay in spawn workers.
The optional takeover signals only an exactly verified recorded workflow PID.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'operations.accelerated_conditional_evidence'
ENV_PATH = 'CONDITIONAL_EVIDENCE_OVERLAY_PATH'
ENV_HASH = 'CONDITIONAL_EVIDENCE_OVERLAY_SHA256'
HELPER = 'operations/conditional_evidence_acceleration.py'
BASELINE = 'exploratory/conditional_evidence/run_experiment.py'
PARITY_CHECKS = ('model_state_exact', 'optimizer_state_exact', 'rng_state_exact',
                 'loss_exact', 'gradient_exact', 'logits_exact', 'checkpoint_resume_exact')
_INSTALLED = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def within_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Operational artifacts must stay within this personal project')
    return path


def verify_files(manifest, expected_manifest=None, manifest_path=None):
    if expected_manifest is not None and digest(manifest_path) != expected_manifest:
        raise ValueError('Execution overlay manifest changed')
    study = within_root(manifest['study'])
    if digest(study / 'plan.json') != manifest['plan_sha256']:
        raise ValueError('Original study plan changed')
    plan = read_json(study / 'plan.json')
    if manifest['original_source_sha256'] != plan['source_sha256']:
        raise ValueError('Overlay original-source binding differs')
    for name, expected in {**plan['source_sha256'], **manifest['overlay_source_sha256']}.items():
        if digest(within_root(ROOT / name)) != expected:
            raise ValueError(f'Pinned execution source changed: {name}')
    parity = within_root(manifest['parity_report'])
    if digest(parity) != manifest['parity_report_sha256']:
        raise ValueError('Parity evidence changed')
    validate_parity(parity)
    return plan


def validate_parity(path):
    value = read_json(path)
    if value.get('status') != 'pass' or any(value.get('checks', {}).get(key) is not True for key in PARITY_CHECKS):
        raise ValueError('Complete strict numerical parity evidence is required')
    if (value.get('accelerator_sha256') != digest(ROOT / HELPER)
            or value.get('baseline_sha256') != digest(ROOT / BASELINE)):
        raise ValueError('Parity evidence is for different execution sources')
    return value


def preflight(study, parity_path):
    """Reject mismatches before any interruption or background launch."""
    study = within_root(study)
    parity = validate_parity(within_root(parity_path))
    plan = read_json(study / 'plan.json')
    run = importlib.import_module('exploratory.conditional_evidence.run_experiment')
    run.numerical_setup(2026)
    run.check_hashes(plan['source_sha256'])
    if run.runtime() != plan['runtime'] or parity.get('runtime') != plan['runtime']:
        raise ValueError('Current, tested and sealed runtimes must agree before takeover')
    for entry in plan.get('inputs', {}).values():
        run.check_hashes(entry['input_sha256'])
    pointer = study.parent / (study.name + '-acceleration.json')
    if pointer.exists():
        value = read_json(pointer)
        path = within_root(value['manifest'])
        verify_files(read_json(path), value['manifest_sha256'], path)
    return plan


def lock_held(path):
    path = Path(path)
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            # Unlocking an unacquired lock has no effect on the other owner.
            fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def process_identity(pid):
    result = subprocess.run(['ps', '-p', str(int(pid)), '-o', 'pid=,ppid=,lstart=,args='],
                            capture_output=True, text=True,
                            env={**os.environ, 'LC_ALL': 'C'}, timeout=5)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    if result.returncode != 0:
        raise RuntimeError('Process identity could not be verified; no signal is permitted')
    parts = result.stdout.strip().split(None, 7)
    if len(parts) != 8 or int(parts[0]) != int(pid):
        raise ValueError('Unexpected process identity response')
    return {'pid': int(parts[0]), 'ppid': int(parts[1]),
            'started': ' '.join(parts[2:7]), 'command': shlex.split(parts[7])}


def descendants(pid):
    # Only PID/PPID metadata is enumerated. Unrelated command lines are not read.
    result = subprocess.run(['ps', '-axo', 'pid=,ppid='], capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError('Cannot verify owned process descendants; no signal is permitted')
    parents = {int(a): int(b) for line in result.stdout.splitlines() for a, b in [line.split()]}
    found, frontier = set(), {int(pid)}
    while frontier:
        frontier = {child for child, parent in parents.items() if parent in frontier} - found
        found |= frontier
    return [identity for child in sorted(found) if (identity := process_identity(child)) is not None]


def validate_parent(record, progress, identity, study):
    if identity is None or identity['pid'] != record.get('pid') or progress.get('pid') != identity['pid']:
        raise ValueError('Launch record, progress and live workflow PID must agree')
    command = record.get('command')
    if identity['command'] != command or record.get('cwd') != str(ROOT):
        raise ValueError('Live workflow command differs from the recorded personal launch')
    if (not isinstance(command, list) or command[:4] != [sys.executable, '-u', '-m',
            'exploratory.conditional_evidence.workflow'] or '--out' not in command
            or Path(command[command.index('--out') + 1]).resolve() != study):
        raise ValueError('Recorded command does not identify the exact original study')
    started = datetime.strptime(identity['started'], '%a %b %d %H:%M:%S %Y').astimezone(timezone.utc)
    recorded = datetime.fromisoformat(record['started_utc'])
    if abs((started - recorded).total_seconds()) > 120:
        raise ValueError('Recorded PID may have been reused; takeover refused')


def wait_stopped(identity, children, lock, timeout=60, *, monotonic=time.monotonic, sleep=time.sleep):
    deadline = monotonic() + timeout
    while True:
        live = [old for old in [identity, *children]
                if (current := process_identity(old['pid'])) is not None
                and current['started'] == old['started']]
        if not live and not lock_held(lock):
            return
        if monotonic() >= deadline:
            raise TimeoutError('Owned workflow/workers did not exit within 60 seconds; no forced stop attempted')
        sleep(min(.5, max(0., deadline - monotonic())))


def takeover(study, *, timeout=60):
    study = within_root(study)
    prefix = study.parent / study.name
    lock = Path(str(prefix) + '.lock')
    if not lock_held(lock):
        return {'status': 'already_unlocked', 'signalled': False}
    record = read_json(Path(str(prefix) + '-launch.json'))
    progress = read_json(Path(str(prefix) + '-progress.json'))
    identity = process_identity(record['pid'])
    validate_parent(record, progress, identity, study)
    children = descendants(identity['pid'])
    # Revalidate immediately before the only permitted signal.
    if process_identity(identity['pid']) != identity:
        raise ValueError('Workflow process changed during takeover verification')
    os.kill(identity['pid'], signal.SIGINT)
    wait_stopped(identity, children, lock, timeout)
    return {'status': 'interrupted_and_joined', 'signalled': True,
            'pid': identity['pid'], 'owned_descendants': len(children)}


def prepare_manifest(study, parity_path):
    """Called under the study lock, after the previous owner has exited."""
    study, parity_path = within_root(study), within_root(parity_path)
    pointer = study.parent / (study.name + '-acceleration.json')
    if pointer.exists():
        value = read_json(pointer)
        path = within_root(value['manifest'])
        manifest = read_json(path)
        verify_files(manifest, value['manifest_sha256'], path)
        if manifest['study'] != str(study) or manifest['parity_report'] != str(parity_path):
            raise ValueError('Existing overlay belongs to another study or parity report')
        return path, value['manifest_sha256']
    if (study / 'SELECTIONS-FROZEN.json').exists():
        raise ValueError('Initial acceleration activation is limited to the unfinished training study')
    parity = validate_parity(parity_path)
    plan = read_json(study / 'plan.json')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Original source changed: {name}')
    folder = study.parent / (study.name + '-execution-overlay-v1')
    if folder.exists():
        raise ValueError('An incomplete overlay preparation exists; inspect it before retrying')
    folder.mkdir()
    original_checkpoints, backups = {}, {}
    for path in sorted(study.rglob('*.pt')):
        name = str(path.relative_to(ROOT))
        original_checkpoints[name] = digest(path)
        if path.name == 'latest.pt':
            backup = folder / 'original-checkpoints' / path.relative_to(study)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, backup)
            if digest(backup) != original_checkpoints[name]:
                raise ValueError('Checkpoint snapshot changed while copying')
            backups[name] = {'path': str(backup.relative_to(ROOT)), 'sha256': digest(backup)}
    manifest = {'schema_version': 1, 'purpose': 'equivalent_compute_execution_overlay',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'study': str(study),
        'plan_sha256': digest(study / 'plan.json'), 'original_source_sha256': plan['source_sha256'],
        'runtime': plan['runtime'], 'interpreter': sys.executable,
        'overlay_source_sha256': {name: digest(ROOT / name) for name in (
            'operations/__init__.py', 'operations/accelerated_conditional_evidence.py', HELPER)},
        'parity_report': str(parity_path), 'parity_report_sha256': digest(parity_path),
        'parity_checks': {key: parity['checks'][key] for key in PARITY_CHECKS},
        'original_checkpoint_sha256': original_checkpoints, 'original_latest_backups': backups,
        'scientific_config_changed': False,
        'resource_difference': 'One forward and one graph construction per chunk; activation memory and timing change.',
        'patches': ['run_experiment.two_pass_backward', 'run_experiment.train_epoch',
                    'run_experiment.full_scores', 'run_experiment.check_hashes',
                    'run_experiment.atomic_torch', 'run_experiment.restore_checkpoint']}
    path = folder / 'manifest.json'
    atomic_json(path, manifest)
    expected = digest(path)
    verify_files(manifest, expected, path)
    atomic_json(pointer, {'manifest': str(path), 'manifest_sha256': expected})
    return path, expected


def install_overlay(manifest_path, expected_hash):
    global _INSTALLED
    manifest_path = within_root(manifest_path)
    if _INSTALLED is not None:
        if _INSTALLED != (str(manifest_path), expected_hash):
            raise ValueError('A different execution overlay is already installed')
        verify_files(read_json(manifest_path), expected_hash, manifest_path)
        return
    manifest = read_json(manifest_path)
    plan = verify_files(manifest, expected_hash, manifest_path)
    if os.path.abspath(sys.executable) != os.path.abspath(manifest['interpreter']):
        raise ValueError('Execution interpreter differs from overlay preparation')
    run = importlib.import_module('exploratory.conditional_evidence.run_experiment')
    accelerator = importlib.import_module('operations.conditional_evidence_acceleration')
    accelerator.install(expected_accelerator_sha256=manifest['overlay_source_sha256'][HELPER],
                        expected_baseline_sha256=plan['source_sha256'][BASELINE])
    original = {name: getattr(run, name) for name in
                ('train_epoch', 'full_scores', 'check_hashes', 'atomic_torch', 'restore_checkpoint')}

    def verify():
        verify_files(manifest, expected_hash, manifest_path)
        if run.runtime() != manifest['runtime']:
            raise ValueError('Execution runtime differs from sealed study')

    def checked(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            verify()
            result = function(*args, **kwargs)
            verify()
            return result
        return call

    provenance = {'manifest': str(manifest_path.relative_to(ROOT)), 'manifest_sha256': expected_hash,
                  'plan_sha256': manifest['plan_sha256'],
                  'accelerator_sha256': manifest['overlay_source_sha256'][HELPER]}

    def atomic_torch(path, payload):
        verify()
        if isinstance(payload, dict) and {'schema_version', 'guard', 'model', 'optimizer', 'rng'} <= payload.keys():
            # One atomic checkpoint contains both optimizer/RNG and execution
            # provenance. There is no checkpoint/sidecar crash window.
            payload = {**payload, 'execution_overlay': provenance}
        original['atomic_torch'](path, payload)

    def restore_checkpoint(path, model, optimizer, guard):
        verify()
        payload = run.torch.load(path, map_location='cpu', weights_only=True)
        recorded = payload.get('execution_overlay')
        if recorded is None:
            relative = str(within_root(path).relative_to(ROOT))
            if manifest['original_checkpoint_sha256'].get(relative) != digest(path):
                raise ValueError('Unrecorded original checkpoint cannot enter the accelerated trajectory')
        elif recorded != provenance:
            raise ValueError('Checkpoint execution-overlay provenance differs')
        return original['restore_checkpoint'](path, model, optimizer, guard)

    for name in ('train_epoch', 'full_scores', 'check_hashes'):
        setattr(run, name, checked(original[name]))
    run.atomic_torch = atomic_torch
    run.restore_checkpoint = restore_checkpoint
    _INSTALLED = (str(manifest_path), expected_hash)


def resume_command(study, parity_path, workers=6):
    return shlex.join([sys.executable, '-u', '-m', MODULE, '--out', str(study),
                       '--resume', '--workers', str(workers), '--parity-report', str(parity_path)])


def export_overlay(evidence, manifest_path, expected):
    """Supplement the original export without rewriting its sealed inventories."""
    manifest = read_json(manifest_path)
    verify_files(manifest, expected, manifest_path)
    target = within_root(evidence) / 'execution-overlay'
    target.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for name, source_hash in manifest['overlay_source_sha256'].items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != source_hash:
            raise ValueError('Overlay source changed before public export')
        path.write_bytes(content)
        inventory[name] = source_hash
    description = {
        'purpose': manifest['purpose'], 'plan_sha256': manifest['plan_sha256'],
        'execution_manifest_sha256': expected,
        'original_source_sha256': manifest['original_source_sha256'],
        'overlay_source_sha256': manifest['overlay_source_sha256'],
        'runtime': manifest['runtime'], 'parity_checks': manifest['parity_checks'],
        'scientific_config_changed': False, 'resource_difference': manifest['resource_difference'],
        'resume_command': 'Use operations.accelerated_conditional_evidence --resume with the original execution manifest.',
    }
    atomic_json(target / 'execution.json', description)
    inventory['execution.json'] = digest(target / 'execution.json')
    atomic_json(target / 'SHA256.json', inventory)


def install_workflow_hooks(workflow, study, parity_path, manifest_path, expected):
    original_progress, original_curate = workflow.progress, workflow.curate

    @functools.wraps(original_progress)
    def progress(args, stage, **values):
        if stage in ('failed', 'interrupted'):
            values['resume_command'] = resume_command(study, parity_path, args.workers)
        return original_progress(args, stage, **values)

    @functools.wraps(original_curate)
    def curate(args):
        result = original_curate(args)
        export_overlay(args.evidence, manifest_path, expected)
        return result

    workflow.progress, workflow.curate = progress, curate


def run_foreground(args):
    study = within_root(args.out)
    if not args.resume:
        raise ValueError('This launcher only resumes the existing scientific study')
    preflight(study, args.parity_report)
    if args.takeover:
        print(json.dumps({'takeover': takeover(study)}), flush=True)
    lock_path = study.parent / (study.name + '.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Study still owned; stop its verified workflow or explicitly request --takeover') from error
        manifest, expected = prepare_manifest(study, args.parity_report)
        os.environ[ENV_PATH], os.environ[ENV_HASH] = str(manifest), expected
        install_overlay(manifest, expected)
        atomic_json(study.parent / (study.name + '-acceleration-active.json'), {
            'status': 'overlay_installed_under_exclusive_study_lock', 'pid': os.getpid(),
            'manifest': str(manifest), 'manifest_sha256': expected,
            'activated_utc': datetime.now(timezone.utc).isoformat()})
        workflow = importlib.import_module('exploratory.conditional_evidence.workflow')
        install_workflow_hooks(workflow, study, args.parity_report, manifest, expected)
        original_args = workflow.parser().parse_args([
            '--out', str(study), '--resume', '--workers', str(args.workers)])
        # The same workflow lock is held here. Calling execute would attempt to
        # acquire a second independent descriptor for that same exclusive lock.
        return workflow.execute_locked(original_args)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--out', type=Path, default=ROOT / 'runs/conditional-evidence-v1')
    result.add_argument('--parity-report', type=Path,
                        default=ROOT / 'runs/conditional-evidence-acceleration-v1/parity.json')
    result.add_argument('--resume', action='store_true', required=True)
    result.add_argument('--workers', type=int, choices=(1, 3, 6), default=6)
    result.add_argument('--takeover', action='store_true')
    result.add_argument('--background', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.background:
        return run_foreground(args)
    study = within_root(args.out)
    preflight(study, args.parity_report)
    command = [sys.executable, '-u', '-m', MODULE, '--out', str(study), '--resume',
               '--workers', str(args.workers), '--parity-report', str(within_root(args.parity_report))]
    if args.takeover:
        command.append('--takeover')
    log = study.parent / (study.name + '.log')
    with log.open('ab', buffering=0) as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=stream, start_new_session=True)
    receipt = {'pid': child.pid, 'command': command, 'cwd': str(ROOT), 'log': str(log),
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'background_process_started_not_yet_verified',
               'activation_requires': 'Verified parent exit, lock acquisition, parity and source/runtime checks'}
    # Return an observed startup state, never equate Popen with successful resume.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        code = child.poll()
        if code is not None:
            receipt.update(status='background_process_exited', exit_code=code)
            break
        progress_path = study.parent / (study.name + '-progress.json')
        if progress_path.exists():
            progress = read_json(progress_path)
            if progress.get('pid') == child.pid and progress.get('stage') == 'training':
                receipt['status'] = 'resumed_training_stage_observed'
                break
        time.sleep(.25)
    atomic_json(study.parent / (study.name + '-acceleration-launch.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt.get('exit_code', 0):
        raise RuntimeError(f'Accelerated startup failed; inspect {log}')


# spawn imports this launcher as __mp_main__ before unpickling the canonical
# original worker. Installation must occur there, not only inside main().
if os.environ.get(ENV_PATH) or os.environ.get(ENV_HASH):
    if not (os.environ.get(ENV_PATH) and os.environ.get(ENV_HASH)):
        raise ValueError('Incomplete inherited execution-overlay binding')
    install_overlay(os.environ[ENV_PATH], os.environ[ENV_HASH])

if __name__ == '__main__':
    main()
