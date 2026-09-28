"""Record actual interpreter, package versions and imported RecBole source hashes."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import site
import sys


def inspect_environment(requirements=None):
    import recbole
    root = Path(recbole.__file__).resolve().parent
    relevant = [root / 'config/configurator.py', root / 'data/dataset/dataset.py',
                root / 'data/utils.py', root / 'trainer/trainer.py']
    relevant.extend(sorted((root / 'model/general_recommender').glob('*.py')))
    packages = ('recbole', 'numpy', 'scipy', 'torch', 'pandas', 'scikit-learn')
    snapshot = {
        'python': sys.version, 'platform': platform.platform(),
        'executable': sys.executable, 'recbole_root': str(root),
        'versions': {name: metadata.version(name) for name in packages},
        'isolation': {'virtual_environment': sys.prefix != sys.base_prefix,
                      'user_site_enabled': site.ENABLE_USER_SITE,
                      'prefix': sys.prefix, 'base_prefix': sys.base_prefix},
        'optional_versions': {},
        'recbole_source_sha256': {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in relevant},
        'note': 'Observed runtime snapshot, not proof of a clean install or cross-platform reproducibility.'}
    for name in ('matplotlib', 'pypdf', 'reportlab'):
        try:
            snapshot['optional_versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            snapshot['optional_versions'][name] = None
    if requirements is not None:
        from packaging.requirements import Requirement
        checks = {}
        for line in Path(requirements).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            requirement = Requirement(line)
            if requirement.url is not None or len(requirement.specifier) != 1 or not str(requirement.specifier).startswith('=='):
                raise ValueError('Expected exact public-package version pins only')
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                installed = None
            checks[requirement.name] = {'required': str(requirement.specifier),
                                        'installed': installed,
                                        'matches': installed is not None and installed in requirement.specifier}
        snapshot['requirements'] = {'sha256': hashlib.sha256(Path(requirements).read_bytes()).hexdigest(),
                                     'all_match': all(check['matches'] for check in checks.values()),
                                     'checks': checks}
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--requirements', type=Path,
                        help='Verify all exact pins against installed distribution metadata')
    args = parser.parse_args()
    snapshot = inspect_environment(args.requirements)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(snapshot, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'output': str(args.out), 'versions': snapshot['versions']}))
    if snapshot.get('requirements', {}).get('all_match') is False:
        raise SystemExit('Installed environment does not match all requirement pins')


if __name__ == '__main__':
    main()
