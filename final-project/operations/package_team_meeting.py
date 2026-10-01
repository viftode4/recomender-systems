"""Build the local team handoff around an unchanged verified coursework snapshot.

No training, raw-data export, installation, publishing or submission is done.
Run with the prepared report environment (Python 3.11 and Markdown).
"""
from __future__ import annotations

import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile, ZIP_DEFLATED

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/team-meeting-2026-10-01'
OUT = ROOT / 'packages/team-meeting-2026-10-01'
CSS = '''
:root { color-scheme: light; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
 color:#172a31; background:#faf9f5; line-height:1.65; }
body { max-width:1040px; margin:0 auto; padding:28px 32px 70px; }
nav { display:flex; gap:22px; flex-wrap:wrap; padding:10px 0 22px; border-bottom:1px solid #d5ded9; }
nav a { font-size:14px; font-weight:600; }
h1 { font-size:clamp(30px,4vw,48px); line-height:1.15; letter-spacing:-.035em; margin:38px 0 20px; }
h2 { font-size:23px; margin:34px 0 12px; line-height:1.3; }
h3 { font-size:18px; margin-top:28px; }
p,li { max-width:900px; } li { margin:8px 0; }
a { color:#0b6350; text-underline-offset:3px; }
a:hover { color:#003c2f; } a:focus-visible { outline:3px solid #d59328; outline-offset:4px; }
table { width:100%; border-collapse:collapse; font-size:14px; margin:20px 0; }
td,th { text-align:left; vertical-align:top; padding:12px 14px; border-bottom:1px solid #d5ded9; }
th { background:#eaf0eb; } tr:nth-child(even) td { background:#f1f3ee; }
code { font-size:.88em; overflow-wrap:anywhere; }
pre { background:#eaf0eb; padding:18px; overflow:auto; border-radius:5px; line-height:1.5; }
footer { margin-top:45px; border-top:1px solid #d5ded9; padding-top:16px; color:#52656a; font-size:13px; }
@media(max-width:650px) { body { padding:18px; } table { display:block; overflow:auto; } td,th { min-width:140px; } }
@media print { nav { display:none; } body { padding:0; font-size:10pt; background:white; } h1 { font-size:24pt; } h2 { break-after:avoid; } tr { break-inside:avoid; } a { color:inherit; } }
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    return path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src') and value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'packages/coursework-complete-v1/24.zip'
    receipt = json.loads(original.with_name('verification.json').read_text())
    original_hash = sha(original.read_bytes())
    if original_hash != receipt['archive_sha256']:
        raise ValueError('Original coursework archive changed')
    missing_references = set()
    with tempfile.TemporaryDirectory(prefix='team-handoff-') as tmp:
        base = Path(tmp) / 'Group24-team-start'
        base.mkdir()
        mapping = {}
        with ZipFile(original) as old:
            if old.testzip() is not None:
                raise ValueError('Original archive CRC failure')
            manifest = json.loads(old.read('PACKAGE-MANIFEST.json'))
            expected = manifest['artifact_sha256']
            for name, digest in expected.items():
                if sha(old.read(name)) != digest:
                    raise ValueError(f'Original artifact hash failure: {name}')
            for name in old.namelist():
                relative = checked_relative(name)
                target = base / 'coursework' / relative
                if name.endswith('/'):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(old.read(name))
                if name.startswith('code/'):
                    mapping[(ROOT / name[5:]).resolve()] = target
                elif name.startswith('evidence/'):
                    mapping[(ROOT / name).resolve()] = target
            for name in ('report.pdf', 'report.md', 'main-report.pdf', 'grouping-supplement.pdf', 'REPORT-MANIFEST.json'):
                mapping[(ROOT / 'reports/coursework-complete-v1' / name).resolve()] = base / 'coursework' / name
            for source, packaged in [('coursework_completion/results-v1', 'coursework-completion-v1'),
                                     ('coursework_completion/audit-v1', 'coursework-completion-audit-v1')]:
                target = base / 'coursework/evidence' / packaged
                for path in target.rglob('*'):
                    if path.is_file():
                        mapping[(ROOT / source / path.relative_to(target)).resolve()] = path

        copies = [(path, base / 'meeting' / path.name) for path in sorted(DOCS.glob('*.md'))]
        copies += [(ROOT / 'docs' / name, base / 'meeting' / name)
                   for name in ('PROJECT_MAP.md', 'SETUP.md', 'RESEARCH_INDEX.md', 'PROPOSALS.md', 'RESULTS.md')]
        copies += [(path, base / 'research' / path.name) for path in sorted((ROOT / 'docs/plans').glob('*.md'))]
        for source, destination in copies:
            mapping[source.resolve()] = destination
        additional = [
            (ROOT / 'operations/team_smoke_check.py', base / 'tools/team_smoke_check.py'),
            (ROOT / 'reports/demo/categorical-field/index.html', base / 'demo/index.html'),
            (original.with_name('verification.json'), base / 'verification/original-coursework.json'),
        ]
        for source, destination in additional:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            mapping[source.resolve()] = destination

        generated_html = {dest.with_suffix('.html') for _, dest in copies}
        link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        def rewrite(text, source, dest, html=False):
            def replace(match):
                label, raw = match.groups()
                parsed = urlsplit(raw)
                if parsed.scheme or not parsed.path:
                    return match.group()
                resolved = (source.parent / unquote(parsed.path)).resolve()
                target = mapping.get(resolved)
                if target is None:
                    missing_references.add(str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else raw)
                    return f'{label} (source-checkout reference: `{raw}`)'
                if html and target.with_suffix('.html') in generated_html:
                    target = target.with_suffix('.html')
                relative = Path(os.path.relpath(target, dest.parent)).as_posix()
                suffix = ('#' + parsed.fragment) if parsed.fragment else ''
                return f'[{label}]({relative}{suffix})'
            return link_re.sub(replace, text)

        def render(source, dest):
            content = rewrite(source.read_text(), source, dest, html=True)
            title = content.splitlines()[0].lstrip('# ')
            start = Path(os.path.relpath(base / 'START_HERE.html', dest.parent)).as_posix()
            plan = Path(os.path.relpath(base / 'meeting/TEAM_PLAN.html', dest.parent)).as_posix()
            quick = Path(os.path.relpath(base / 'meeting/QUICKSTART.html', dest.parent)).as_posix()
            report = Path(os.path.relpath(base / 'coursework/report.pdf', dest.parent)).as_posix()
            body = markdown.markdown(content, extensions=['tables', 'fenced_code', 'toc'])
            dest.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style><nav aria-label="Project navigation"><a href="{start}">Start here</a><a href="{plan}">Five-person plan</a><a href="{quick}">Run the check</a><a href="{report}">Report draft</a></nav><main>{body}</main><footer>Group 24 · 1 October 2026 · review draft · training remains paused</footer></html>')

        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rewrite(source.read_text(), source, destination))
            render(source, destination.with_suffix('.html'))
        render(DOCS / 'START_HERE.md', base / 'START_HERE.html')
        (base / 'README.md').write_text('# Group 24 team starting point\n\nOpen **START_HERE.html** in a browser. Begin with the brief, report and five-person plan.\n\n- `meeting/`: concise notes and runnable quickstart.\n- `coursework/`: unchanged, verified code/report/evidence snapshot. Start in `coursework/code/`.\n- `demo/`: offline fictional-profile demonstration.\n- `research/`: optional design documents, not implementations.\n- `tools/`: fast synthetic smoke check.\n- `verification/`: previous archive check receipt.\n\nThe new HTML navigation is the team entry point; older snapshot navigation is retained for provenance.\nNo raw ratings, personal recommendation histories, environments or checkpoints are included.\nTraining remains paused. Review draft; not submitted.\n')

        # Every original snapshot byte must survive the packaging unchanged.
        for name, digest in expected.items():
            if sha((base / 'coursework' / name).read_bytes()) != digest:
                raise ValueError(f'Copied snapshot changed: {name}')
        html_paths = [base / 'START_HERE.html', *sorted(generated_html), base / 'demo/index.html']
        for path in html_paths:
            parser = Links()
            parser.feed(path.read_text())
            for href in parser.links:
                parsed = urlsplit(href)
                if parsed.scheme or not parsed.path:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                if not target.is_relative_to(base.resolve()) or not target.exists():
                    raise ValueError(f'Broken packaged link: {path.relative_to(base)} -> {href}')
        artifact_hashes = {str(p.relative_to(base)): sha(p.read_bytes()) for p in sorted(base.rglob('*')) if p.is_file()}
        package_manifest = {
            'status': 'team_review_draft', 'training_started': False, 'submitted': False,
            'original_archive_sha256': original_hash,
            'original_artifacts_unchanged': len(expected),
            'new_navigation_html_files_checked': len(html_paths),
            'source_only_references_labelled': sorted(missing_references),
            'artifact_sha256': artifact_hashes,
        }
        (base / 'TEAM-MANIFEST.json').write_text(json.dumps(package_manifest, indent=2) + '\n')
        destination = OUT / 'Group24-team-start.zip'
        temporary_zip = OUT / '.Group24-team-start.zip.tmp'
        with ZipFile(temporary_zip, 'w', ZIP_DEFLATED) as out:
            for path in sorted(base.rglob('*')):
                if path.is_file():
                    out.write(path, 'Group24-team-start/' + path.relative_to(base).as_posix())
        with ZipFile(temporary_zip) as check:
            if check.testzip() is not None:
                raise ValueError('Generated ZIP CRC failure')
        temporary_zip.replace(destination)
        local_preview = OUT / 'preview'
        # This is the builder-owned derived preview, never an experiment directory.
        if local_preview.exists():
            shutil.rmtree(local_preview)
        shutil.copytree(base, local_preview)
        verification = {k: v for k, v in package_manifest.items() if k != 'artifact_sha256'}
        verification.update({'archive_sha256': sha(destination.read_bytes()),
                             'archive_bytes': destination.stat().st_size,
                             'zip_crc_valid': True, 'artifact_count': len(artifact_hashes) + 1})
        (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
