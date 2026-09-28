"""Export aggregate-only evidence from the completed, single-use final batch.

Reads completed evaluation artifacts, never TEST splits, never models, and
never changes selections or results. Original aggregate and manifest files
are copied byte for byte where an evaluator already provides aggregates.
"""
import argparse
import hashlib
import json
from pathlib import Path

from package_project import check_aggregate_only


ROOT = Path(__file__).resolve().parent
SEEDS = (2026, 2027, 2028)
TRACKS = {
    'exception': ('final-exception-v3', 'summary.json'),
    'conditional': ('final-adaptive-v1', 'aggregates.json'),
    'joint100': ('final-joint-field-v1', 'aggregates.json'),
    'joint400': ('final-joint-field-convergence-v1', 'aggregates.json'),
    'references': ('final-field-references-v1', 'aggregates.json'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    check_aggregate_only(value)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def verify_outputs(directory, manifest):
    if manifest.get('status') != 'complete':
        raise ValueError('Require completed evaluation or analysis')
    for name, expected in manifest['output_sha256'].items():
        path = directory/name
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError('Completed output digest differs')


def copy_aggregate(source, out, manifest, filename):
    path = source/filename
    if manifest['output_sha256'].get(filename) != digest(path):
        raise ValueError('Aggregate digest differs')
    check_aggregate_only(read_json(path), filename)
    (out/filename).write_bytes(path.read_bytes())


def primary_aggregate(results, manifest, selection):
    if set(results) != set(manifest['models']):
        raise ValueError('Primary model set differs from completed manifest')
    models = {name: {key: row[key] for key in ('aggregate', 'groups', 'item_groups', 'item_exposure')}
              for name, row in results.items()}
    result = {'models': models, 'selection': selection,
              'denominators': {'users': manifest['users'], 'interactions': manifest['interactions']}}
    check_aggregate_only(result)
    return result


def finalize(out, provenance, text):
    write_json(out/'provenance.json', provenance)
    (out/'RESULTS.md').write_text(text)
    write_json(out/'SHA256.json', {path.name: digest(path) for path in sorted(out.iterdir())})


def curate(root=ROOT):
    root = Path(root).resolve()
    plan_path, marker_path = root/'runs/final-batch-plan.json', root/'runs/FINAL-TEST-BATCH.json'
    plan, marker = read_json(plan_path), read_json(marker_path)
    if (marker.get('status') != 'complete' or marker.get('selection_after_test') is not False
            or digest(plan_path) != marker['plan_sha256']
            or digest(plan_path) != plan_path.with_suffix('.json.sha256').read_text().strip()
            or Path(plan['root']).resolve() != root):
        raise ValueError('Require the completed sealed batch')
    jobs = {job['name']: job for job in plan['jobs']+plan['analysis_jobs']}
    if set(jobs) != set(marker['results']) or len(plan['jobs']) != 8 or len(plan['analysis_jobs']) != 5:
        raise ValueError('Incomplete final batch')
    for name, job in jobs.items():
        directory = Path(job['output'])
        result = marker['results'][name]
        if (result['returncode'] != 0 or result['complete'] is not True
                or digest(directory/'manifest.json') != result['manifest_sha256']):
            raise ValueError('Batch result differs from recorded completion')
        verify_outputs(directory, read_json(directory/'manifest.json'))
    outputs = [root/'evidence/final-primary-v3']+[root/'evidence'/name for name, _ in TRACKS.values()]
    if any(path.exists() for path in outputs):
        raise ValueError('Refuse to overwrite final evidence')
    common = {'status': 'complete', 'stage': 'test', 'test_read': True,
        'test_split_files_read_by_curator': False, 'individual_user_data_exported': False,
        'post_test_model_selection': False, 'batch_plan_sha256': digest(plan_path),
        'batch_completion_sha256': digest(marker_path),
        'curation_source_sha256': digest(Path(__file__)),
        'aggregate_validator_sha256': digest(ROOT/'package_project.py')}
    out = outputs[0]
    out.mkdir(parents=True)
    aggregates = {'schema_version': 1, 'stage': 'test', 'test_read': True, 'seeds': {}}
    provenance = {**common, 'track': 'primary', 'seeds': {}}
    lines = ['# Frozen primary held-out results', '',
        'Every one of the 63 frozen models is retained per split. No model is selected using TEST.',
        'The displayed roles below were selected before TEST; the full aggregates retain all models.', '',
        '| Split | Frozen best expert | nDCG@10 | Frozen calibrated hybrid | nDCG@10 |',
        '|---|---|---:|---|---:|']
    for seed in SEEDS:
        directory = Path(jobs[f'primary-{seed}']['output'])
        frozen = root/'runs/frozen-v3'/str(seed)
        manifest, bundle = read_json(directory/'manifest.json'), read_json(frozen/'freeze.json')
        if manifest['freeze_sha256'] != digest(frozen/'freeze.json'):
            raise ValueError('Primary evaluation freeze differs')
        result = primary_aggregate(read_json(directory/'results.json'), manifest, bundle['selection'])
        aggregates['seeds'][str(seed)] = result
        provenance['seeds'][str(seed)] = {'evaluation_manifest_sha256': digest(directory/'manifest.json'),
            'evaluation_manifest': manifest, 'frozen_manifest_sha256': digest(frozen/'freeze.json'),
            'source_sha256': bundle['code_sha256'], 'group_policy_audit': read_json(directory/'group-policy-audit.json')}
        expert, hybrid = bundle['selection']['best_expert'], bundle['selection']['families']['calibrated']
        lines.append(f"| {seed} | {expert} | {result['models'][expert]['aggregate']['ndcg@10']:.6f} | {hybrid} | {result['models'][hybrid]['aggregate']['ndcg@10']:.6f} |")
    write_json(out/'aggregates.json', aggregates)
    finalize(out, provenance, '\n'.join(lines)+'\n')
    for track, (name, aggregate_name) in TRACKS.items():
        source, out = Path(jobs[track]['output']), root/'evidence'/name
        manifest = read_json(source/'manifest.json')
        frozen_record = plan['preflight']['frozen'][track]
        if (manifest.get('test_read') is not True or manifest.get('test_evaluated') is not True
                or manifest.get('selection_after_test') is not False
                or manifest['frozen_manifest_sha256'] != frozen_record['manifest_sha256']):
            raise ValueError('Final evaluation differs from the frozen batch')
        out.mkdir(parents=True)
        copy_aggregate(source, out, manifest, aggregate_name)
        check_aggregate_only(manifest)
        (out/'manifest.json').write_bytes((source/'manifest.json').read_bytes())
        provenance = {**common, 'track': track,
            'evaluation_manifest_sha256': digest(source/'manifest.json'),
            'frozen_manifest_sha256': frozen_record['manifest_sha256'],
            'frozen_bundle_sha256': frozen_record['bundle_sha256'],
            'source_sha256': manifest['code_sha256'],
            'copy_policy': 'Aggregate and evaluation manifest bytes preserved exactly; raw individual metrics and recommendation files excluded.'}
        if track in ('conditional', 'joint100', 'joint400'):
            frozen = Path(frozen_record['root'])
            provenance['protocols'] = {str(seed): read_json(frozen/str(seed)/'protocol.json') for seed in SEEDS}
        lines = [f'# {name}: frozen held-out evidence', '',
            f'`{aggregate_name}` and `manifest.json` are exact copies of the completed evaluation artifacts.',
            'All preselected models are reported. No checkpoint, hyperparameter or model was changed using TEST.',
            'Manifest entries for excluded per-user files preserve provenance; those files are not included here.', '']
        if track == 'conditional':
            lines.append('Conditional rating likelihood and the P(rating ≥ 4) ranking diagnostic are distinct objectives.\n')
        elif track == 'joint400':
            lines.append('This is the separately declared post-v1 convergence sensitivity, prompted by development evidence; it is not an independent confirmatory experiment.\n')
        finalize(out, provenance, '\n'.join(lines))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps({'status': 'complete', 'outputs': [str(p) for p in curate(args.root)],
                      'individual_user_data_exported': False}))


if __name__ == '__main__':
    main()
