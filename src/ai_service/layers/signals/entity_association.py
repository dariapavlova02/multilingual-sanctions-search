"""Associate evidence using source spans without distributing it by entity order."""

import re

from ...utils.source_text_view import SourceTextView, without_format_controls


def evidence_owner(entities, evidence, text, *, max_distance=300):
    position = evidence.get("position")
    if not position:
        raw = str(evidence.get("raw") or evidence.get("value") or "")
        if not raw:
            return None
        matches = list(re.finditer(re.escape(raw), text, re.IGNORECASE))
        if len(matches) != 1:
            return None
        position = matches[0].span()
    if len(position) != 2 or not 0 <= position[0] < position[1] <= len(text):
        return None
    view = SourceTextView.from_text(text)
    start, end = view.matching_span(*position)
    if start >= end:
        return None
    text = view.text
    left = max(text.rfind(boundary, 0, start) for boundary in (";", "|", "\n")) + 1
    right = min(
        (p for boundary in (";", "|", "\n") if (p := text.find(boundary, end)) >= 0),
        default=len(text),
    )
    found = []
    for entity in entities:
        core = entity.core if isinstance(entity.core, str) else " ".join(entity.core)
        names = {
            core,
            getattr(entity, "full_name", None),
            getattr(entity, "full", None),
            *getattr(entity, "source_names", []),
        }
        spans = set()
        for name in filter(None, names):
            name = without_format_controls(name)
            if not name.strip():
                continue
            pattern = (
                r"(?<!\w)"
                + r"\s+".join(re.escape(part) for part in name.split())
                + r"(?!\w)"
            )
            spans.update(
                match.span()
                for match in re.finditer(pattern, text, re.IGNORECASE)
                if left <= match.start() and match.end() <= right
            )
        for entity_start, entity_end in spans:
            distance = start - entity_end if entity_end <= start else entity_start - end
            if 0 <= distance <= max_distance:
                found.append((entity, entity_start, entity_end, distance))
    preceding = [item for item in found if item[2] <= start]
    if preceding:
        latest_end = max(item[2] for item in preceding)
        owners = {id(item[0]): item[0] for item in preceding if item[2] == latest_end}
    else:
        # Leading evidence is attributable only when the clause names one entity.
        owners = {id(item[0]): item[0] for item in found}
    return next(iter(owners.values())) if len(owners) == 1 else None
