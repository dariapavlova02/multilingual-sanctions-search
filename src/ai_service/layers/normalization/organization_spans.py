"""Conservative spans for named establishments introduced by lexical descriptors."""

import unicodedata
from functools import lru_cache

from ...data.resources import LEXICONS_DIR


@lru_cache(maxsize=1)
def _descriptors():
    lines = (
        (LEXICONS_DIR / "organization_descriptors.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return frozenset(
        line.strip().casefold()
        for line in lines
        if line.strip() and not line.startswith("#")
    )


def _script(token):
    return frozenset(
        unicodedata.name(char, "").split()[0] for char in token if char.isalpha()
    )


def descriptor_spans(tokens):
    result = []
    consumed = set()
    for start, token in enumerate(tokens):
        if start in consumed or token.casefold() not in _descriptors():
            continue
        end = start + 1
        script = None
        while end < len(tokens):
            word = tokens[end]
            if (
                not word
                or not word[0].isupper()
                or not any(char.isalpha() for char in word)
            ):
                break
            current_script = _script(word)
            if script is not None and current_script != script:
                break
            script = current_script
            end += 1
        if end > start + 1:
            result.append((start, end, " ".join(tokens[start:end])))
            consumed.update(range(start, end))
    return result
