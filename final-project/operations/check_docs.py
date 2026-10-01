"""Check active teammate Markdown links locally, without importing project code."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ENTRY_DOCS = (
    "README.md", "CONTRIBUTING.md", "final-project/README.md",
    "final-project/HANDOFF.md", "final-project/PLAN.md",
    *(f"final-project/{folder}/README.md" for folder in
      ("operations", "tests", "evidence", "reports", "packages")),
)
DESTINATION = r"(<[^>\n]*>|(?:[^()\s]|\([^()\n]*\))+)"
LINK = re.compile(r"\[[^\]\n]*\]\(\s*" + DESTINATION)
REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*" + DESTINATION)


def prose_lines(text):
    """Keep original line numbers while excluding fenced examples."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            marks, rest = marker.groups()
            if fence is None:
                fence = marks
            elif marks[0] == fence[0] and len(marks) >= len(fence) and not rest.strip():
                fence = None
            continue
        if fence is None:
            yield number, line


def heading_anchors(path):
    """GitHub-style anchors for the ATX headings used by these documents."""
    anchors = set()
    for _, line in prose_lines(path.read_text(encoding="utf-8")):
        heading = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not heading:
            continue
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading[1])
        title = html.unescape(re.sub(r"<[^>]*>", "", title))
        slug = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
        anchor, duplicate = slug, 0
        while anchor in anchors:
            duplicate += 1
            anchor = f"{slug}-{duplicate}"
        anchors.add(anchor)
    return anchors


def check_docs(root):
    root = Path(root).resolve()
    documents = {root / name for name in ENTRY_DOCS}
    for directory in ("final-project/docs", "final-project/docs/team"):
        documents.update((root / directory).glob("*.md"))
    errors, links, anchors_checked, anchor_cache = [], 0, 0, {}
    for document in sorted(documents):
        label = document.relative_to(root).as_posix()
        if not document.is_file():
            errors.append(f"{label}: missing entry document")
            continue
        for number, line in prose_lines(document.read_text(encoding="utf-8")):
            line = re.sub(r"`+[^`]*`+", "", line)
            matches = list(LINK.finditer(line))
            reference = REFERENCE.match(line)
            if reference:
                matches.append(reference)
            for match in matches:
                destination = html.unescape(match[1].strip("<>"))
                url = urlsplit(destination)
                if url.scheme or url.netloc:
                    continue
                links += 1
                path = unquote(url.path)
                target = ((root / path.lstrip("/")) if path.startswith("/") else
                          document.parent / path) if path else document
                target = target.resolve()
                if not target.exists():
                    errors.append(f"{label}:{number}: missing target {destination}")
                elif url.fragment and target.suffix.lower() == ".md":
                    anchors_checked += 1
                    if target not in anchor_cache:
                        anchor_cache[target] = heading_anchors(target)
                    if unquote(url.fragment) not in anchor_cache[target]:
                        errors.append(f"{label}:{number}: missing heading {destination}")
    return {"status": "failed" if errors else "passed", "documents": len(documents),
            "local_links": links, "anchors": anchors_checked, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Repository root containing final-project/")
    result = check_docs(parser.parse_args().root)
    print(json.dumps(result, sort_keys=True))
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
