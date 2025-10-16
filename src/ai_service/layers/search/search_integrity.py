"""Shared identity and response checks for every retrieval strategy."""

import re

from ...utils.source_text_view import without_format_controls

TAX_IDENTIFIER_TYPES = {"inn", "inn_ua", "inn_ru", "itn", "tax_id", "tin", "edrpou"}


def source_tax_ids(metadata):
    identifiers = set()
    for field in TAX_IDENTIFIER_TYPES | {"itn_import", "tax_number", "taxpayer_id"}:
        identifiers.update(
            re.findall(r"(?<!\d)\d{8,12}(?!\d)", without_format_controls(str(metadata.get(field) or "")))
        )
    return identifiers


def require_complete_response(response):
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError("Elasticsearch returned incomplete screening results")
    if not isinstance(response.get("hits", {}).get("hits"), list):
        raise RuntimeError("Elasticsearch returned an invalid search response")


def source_metadata(source):
    metadata = dict(source.get("metadata") or {})
    for field in ("entity_id", "aliases", "dob", "country", "category"):
        if source.get(field) is not None:
            metadata[field] = source[field]
    if source.get("source_list"):
        metadata["source"] = source["source_list"]
    if source.get("name"):
        metadata["canonical"] = source["name"]
    return metadata


def candidate_identity(candidate):
    """Aliases share an identity only when their source explicitly identifies it."""
    return source_identity(candidate.doc_id, candidate.entity_type, candidate.metadata)


def source_identity(doc_id, entity_type, metadata):
    entity_id = metadata.get("entity_id")
    source = metadata.get("source")
    if source and entity_id is not None and str(entity_id) != "":
        return ("entity", str(source), entity_type, str(entity_id))
    return ("document", doc_id)


def best_per_entity(candidates):
    best = {}
    for candidate in candidates:
        key = candidate_identity(candidate)
        current = best.get(key)
        if current is None or (-candidate.score, candidate.doc_id) < (
            -current.score,
            current.doc_id,
        ):
            best[key] = candidate
    return sorted(
        best.values(), key=lambda candidate: (-candidate.score, candidate.doc_id)
    )


def metadata_matches(doc_id, metadata, filters):
    for key, expected in filters.items():
        if expected is None:
            continue
        if key in {"id", "doc_id"}:
            actual = doc_id
        elif key in {"country", "country_code"}:
            actual = metadata.get("country") or metadata.get("country_code")
        elif key in {"dob", "date_of_birth"}:
            actual = metadata.get("dob") or metadata.get("date_of_birth")
        else:
            actual = metadata.get(key)
        actual = actual if isinstance(actual, list) else [actual]
        expected = expected if isinstance(expected, list) else [expected]
        if not any(value in expected for value in actual):
            return False
    return True
