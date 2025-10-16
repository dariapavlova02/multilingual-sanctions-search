"""Check local Markdown links against files included in the Git publication tree."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\s*\)")


def prose(text: str) -> str:
    """Exclude fenced examples while preserving source line numbers."""
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append("" if fence else line)
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    counts: Counter[str] = Counter()
    result = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose(text), re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = counts[slug]
        result.add(f"{slug}-{suffix}" if suffix else slug)
        counts[slug] += 1
    return result


def main() -> int:
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    files = {
        (ROOT / name.decode()).resolve()
        for name in listed.split(b"\0")
        if name and (ROOT / name.decode()).is_file()
    }
    documents = sorted(path for path in files if path.suffix == ".md")
    errors = []
    checked = 0
    for document in documents:
        for line_no, line in enumerate(prose(document.read_text()).splitlines(), 1):
            for match in LINK.finditer(line):
                url = urlsplit(match[1].strip("<>"))
                if url.scheme or url.netloc:
                    continue
                checked += 1
                target = (
                    (document.parent / unquote(url.path)).resolve()
                    if url.path
                    else document
                )
                prefix = f"{document.relative_to(ROOT)}:{line_no}"
                if target not in files and not any(target in p.parents for p in files):
                    errors.append(f"{prefix}: target is not in Git: {match[1]}")
                elif url.fragment and target.suffix == ".md" and target.is_file():
                    if unquote(url.fragment) not in anchors(target.read_text()):
                        errors.append(f"{prefix}: missing heading: {match[1]}")
    if errors:
        print("\n".join(errors))
        return 1
    print(
        f"Documentation: {len(documents)} Markdown files, {checked} local links, no broken targets or anchors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
