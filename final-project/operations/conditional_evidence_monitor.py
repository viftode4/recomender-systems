"""Observe the conditional-evidence study without importing or altering its code.

Writes only sibling monitor status/event files. It never reads ratings, outcome
files, predictions or Torch checkpoints. A separate OS lock prevents duplicate
monitors. This process records alerts locally; it does not send chat messages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time


SEEDS = (2026, 2027, 2028)
ARMS = ('raw', 'summary', 'scrambled')


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def lock_held(path):
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def scan_log(path):
    training, validation, errors = {}, {}, []
    if not path.exists():
        return training, validation, errors
    with path.open() as stream:
        for line in stream:
            # The writer may currently be in the middle of its last line.
            if not line.endswith('\n'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if 'Traceback (' in line or 'Error:' in line or 'Exception:' in line:
                    errors.append(line.strip()[:300])
                continue
            if row.get('stage') == 'training' and 'updated_utc' in row:
                errors = []  # Errors from an earlier, explicitly resumed invocation are historical.
            if 'run' not in row or 'seed' not in row:
                continue
            key = f"{row['seed']}/{row['run']}"
            if 'training' in row:
                value = row['training']
                training[key] = {name: value[name] for name in ('epoch', 'queries', 'seconds', 'mean_query_loss')}
                if not all(math.isfinite(value[name]) for name in ('seconds', 'mean_query_loss')):
                    errors.append(f'{key}: non-finite training value')
                    training[key]['seconds'] = 0 if not math.isfinite(value['seconds']) else value['seconds']
                    training[key]['mean_query_loss'] = None if not math.isfinite(value['mean_query_loss']) else value['mean_query_loss']
            elif 'meta_ndcg@10' in row:
                seconds = row['resources']['seconds']
                validation[key] = {'epoch': row['epoch'], 'seconds': seconds if math.isfinite(seconds) else 0}
                if not math.isfinite(row['meta_ndcg@10']) or not 0 <= row['meta_ndcg@10'] <= 1:
                    errors.append(f'{key}: invalid selection metric')
    return training, validation, errors


def verify_completion(root, study, alerts):
    aggregates = read_json(study / 'aggregates.json', {})
    if aggregates.get('status') != 'complete' or aggregates.get('original_test_read') is not False:
        alerts.append('Completion stage has no valid completed aggregate receipt')
    evidence = root / 'exploratory/conditional_evidence/results-v1'
    for name in ('SHA256.json', 'RESULTS-SHA256.json'):
        inventory = read_json(evidence / name, {})
        if not inventory:
            alerts.append(f'Missing completion inventory: {name}')
        for filename, expected in inventory.items():
            path = (evidence / filename).resolve()
            if not path.is_relative_to(evidence.resolve()) or not path.is_file() or sha256(path) != expected:
                alerts.append(f'Completion artifact differs: {filename}')


def snapshot(study, *, now=None, stale_seconds=1800):
    study = study.resolve()
    now = time.time() if now is None else now
    prefix = study.parent / study.name
    root = study.parent.parent
    progress = read_json(Path(str(prefix) + '-progress.json'), {})
    plan = read_json(study / 'plan.json', {})
    alerts = []
    stage = progress.get('stage', 'unknown')
    held = lock_held(Path(str(prefix) + '.lock'))
    training, validation, errors = scan_log(Path(str(prefix) + '.log'))
    alerts.extend(errors[-10:])
    if not plan:
        alerts.append('Study plan is missing')
    for name, expected in plan.get('source_sha256', {}).items():
        source = root / name
        if not source.is_file() or sha256(source) != expected:
            alerts.append(f'Frozen source changed: {name}')
    extension_path = study / 'extension.json'
    extension = read_json(extension_path, {})
    target = extension.get('target_epochs', 300)
    extension_started = extension_path.stat().st_mtime if target == 600 else 0
    trajectories = {}
    for seed in SEEDS:
        for arm in ARMS:
            for lr in (0, 1):
                key = f'{seed}/{arm}-lr{lr}'
                directory = study / key
                latest = directory / 'latest.pt'
                completed = read_json(directory / 'result.json', {}).get('completed_epoch', 0)
                age = max(0., now - latest.stat().st_mtime) if latest.exists() else None
                if completed >= target:
                    state = 'complete'
                elif not latest.exists() or (extension_started and latest.stat().st_mtime < extension_started):
                    state = 'queued'
                else:
                    state = 'active'
                duration = training.get(key, {}).get('seconds', 0) + validation.get(key, {}).get('seconds', 0)
                threshold = max(stale_seconds, 6*duration)
                if stage == 'training' and state == 'active' and age > threshold:
                    state = 'stale'
                    alerts.append(f'{key}: checkpoint unchanged for {age:.0f}s (threshold {threshold:.0f}s)')
                trajectories[key] = {'state': state, 'target_epoch': target,
                    'last_logged_training_epoch': training.get(key, {}).get('epoch', 0),
                    'last_selection_epoch': validation.get(key, {}).get('epoch'),
                    'checkpoint_age_seconds': age, 'stale_threshold_seconds': threshold,
                    'latest_training': training.get(key)}
    if stage == 'assessment_complete':
        verify_completion(root, study, alerts)
        status = 'completed' if not alerts else 'attention_required'
    elif stage in ('failed', 'interrupted'):
        alerts.append(f"Workflow {stage}: {progress.get('message', 'no detail')}")
        status = 'failed'
    elif not held:
        alerts.append('Workflow lock is not held; process may have stopped')
        status = 'stopped'
    else:
        status = 'healthy' if not alerts else 'attention_required'
    log = Path(str(prefix) + '.log')
    # Acquisition can legitimately be quiet while constructing labels. Its stage
    # is retained, without guessing that a long label-construction pass has hung.
    return {'checked_utc': utc(now), 'monitor_pid': os.getpid(), 'status': status,
        'stage': stage, 'workflow_lock_held': held,
        'log_age_seconds': max(0., now-log.stat().st_mtime) if log.exists() else None,
        'trajectories': trajectories, 'alerts': alerts,
        'monitor_scope': 'Source integrity, log errors, finite training values, base-training checkpoint freshness, final public artifact hashes',
        'limitations': 'Local status only; no chat notifications. Retrieval label passes have no per-query heartbeat.',
        'study_files_modified': False, 'private_outcome_files_read': False}


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('Interval must be at least 10 seconds')
    study = args.study.resolve()
    prefix = study.parent / (study.name + '-monitor')
    with Path(str(prefix)+'.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A monitor already owns this study')
        previous = None
        while True:
            try:
                value = snapshot(study)
            except Exception as error:
                value = {'checked_utc': utc(time.time()), 'monitor_pid': os.getpid(),
                         'status': 'monitor_error', 'alerts': [f'{type(error).__name__}: {error}']}
            value['interval_seconds'] = args.interval
            atomic_json(Path(str(prefix)+'.json'), value)
            identity = (value['status'], value.get('stage'), tuple(value['alerts']))
            if identity != previous:
                event = {key: value[key] for key in ('checked_utc', 'status', 'alerts')}
                with Path(str(prefix)+'-events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event)+'\n')
                print(json.dumps(event), flush=True)
                previous = identity
            if args.once or value['status'] in ('completed', 'failed', 'stopped'):
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
