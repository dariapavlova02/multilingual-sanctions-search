"""Shared index and document contracts for every ingestion entry point."""

import hashlib
import json
import math
from typing import Any

from ...config import EmbeddingConfig
from .config import HybridSearchConfig

SCHEMA_VERSION = 1
SOURCE_COVERAGE_VERSION = 1
_UNCHANGED_MANIFEST = object()


def embedding_contract(config: EmbeddingConfig | None = None) -> dict:
    return (config if config is not None else EmbeddingConfig()).embedding_contract()


def index_mapping(config: HybridSearchConfig, *, vectors: bool = False) -> dict:
    keyword = {"type": "keyword", "fields": {"keyword": {"type": "keyword"}}}
    text = {
        "type": "text",
        "fields": {
            "keyword": {"type": "keyword", "normalizer": "name_keyword"},
            "exact": {"type": "keyword", "normalizer": "name_keyword"},
        },
    }
    properties = {
        key: text
        for key in (
            "normalized_text",
            "original_text",
            "name",
            "text",
            "aliases",
            "legal_names",
            "pattern",
        )
    }
    properties.update(
        {
            key: keyword
            for key in (
                "entity_id",
                "entity_type",
                "category",
                "source_list",
                "country",
                "dob",
                "model_name",
            )
        }
    )
    properties.update(
        {
            "tier": {"type": "integer"},
            "confidence": {"type": "float"},
            "metadata": {"type": "object"},
            "embedding_contract": {"type": "object", "enabled": False},
        }
    )
    meta = {"schema_version": SCHEMA_VERSION}
    if vectors:
        contract = embedding_contract()
        if config.vector_search.vector_dimension != contract["dimension"]:
            raise ValueError("Search and embedding dimensions must agree")
        properties[config.vector_search.vector_field] = {
            "type": "dense_vector",
            "dims": contract["dimension"],
            "index": True,
            "similarity": "cosine",
        }
        meta["embedding_contract"] = contract
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "normalizer": {
                    "name_keyword": {
                        "type": "custom",
                        "filter": ["lowercase"],
                    }
                }
            },
        },
        "mappings": {
            "_meta": meta,
            "properties": properties,
            "dynamic_templates": [
                {"strings": {"match_mapping_type": "string", "mapping": keyword}}
            ],
        },
    }


async def ensure_index(
    client, index: str, config: HybridSearchConfig, *, vectors=False
):
    desired = index_mapping(config, vectors=vectors)
    if not await client.indices.exists(index=index):
        await client.indices.create(index=index, body=desired)
        return
    mappings = await client.indices.get_mapping(index=index)
    for mapping in mappings.values():
        validate_mapping(mapping.get("mappings", {}), desired["mappings"])


def validate_mapping(actual, desired):
    """Metadata alone cannot prove that the fields needed for retrieval exist."""

    def require_subset(required, present, path):
        if not isinstance(present, dict):
            raise ValueError(f"Incompatible index mapping at {path}")
        for key, value in required.items():
            if isinstance(value, dict):
                require_subset(value, present.get(key), f"{path}.{key}")
            elif (
                key == "type"
                and value == "object"
                and present.get("type") is None
                and isinstance(present.get("properties"), dict)
            ):
                # Elasticsearch omits the implicit object type once children exist.
                continue
            elif present.get(key) != value:
                raise ValueError(f"Incompatible index mapping at {path}.{key}")

    for section in ("_meta", "properties"):
        require_subset(desired[section], actual.get(section), section)


async def set_ingestion_status(
    client, index: str, status: str, generation: str, *, source_manifest=_UNCHANGED_MANIFEST,
    source_coverage_verified: bool = False,
):
    mappings = await client.indices.get_mapping(index=index)
    for name, mapping in mappings.items():
        meta = dict(mapping.get("mappings", {}).get("_meta", {}))
        meta.update({"ingestion_status": status, "generation": generation})
        meta.pop("source_coverage_version", None)
        if source_coverage_verified and status == "completed":
            meta["source_coverage_version"] = SOURCE_COVERAGE_VERSION
        if source_manifest is not _UNCHANGED_MANIFEST:
            if source_manifest is None:
                meta.pop("source_manifest", None)
            else:
                meta["source_manifest"] = source_manifest
        await client.indices.put_mapping(index=name, body={"_meta": meta})


def pattern_document(item: dict, category: str, tier: str | int) -> tuple[str, dict]:
    pattern = str(item.get("pattern") or "").strip()
    if not pattern:
        raise ValueError("Each pattern must contain non-empty text")
    tier_number = (
        int(tier.split("_")[1])
        if isinstance(tier, str) and tier.startswith("tier_")
        else int(tier)
    )
    if not 0 <= tier_number <= 4:
        raise ValueError("Unsupported pattern tier")
    metadata = dict(item.get("metadata") or item.get("meta") or {})
    entity_id = item.get("entity_id")
    if entity_id is None:
        entity_id = metadata.get("entity_id")
    if entity_id is None:
        raise ValueError("Each pattern must identify its source entity")
    entity_type = item.get("entity_type") or (
        "organization" if category == "company" else "person"
    )
    canonical = item.get("canonical") or item.get("original_name") or pattern
    source = item.get("source_list") or metadata.get("source") or "api_upload"
    metadata.update(
        {"entity_id": str(entity_id), "source": source, "canonical": canonical}
    )
    doc = {
        "pattern": pattern,
        "normalized_text": pattern,
        "original_text": canonical,
        "name": canonical,
        "entity_id": str(entity_id),
        "entity_type": entity_type,
        "aliases": item.get("aliases") or [],
        "tier": tier_number,
        "category": category,
        "source_list": source,
        "confidence": item.get("confidence", 1.0),
        "metadata": metadata,
        "country": item.get("country")
        or metadata.get("country")
        or metadata.get("nationality"),
        "dob": item.get("dob") or metadata.get("dob"),
    }
    identity = json.dumps(
        [source, category, str(entity_id), tier_number, pattern], ensure_ascii=False
    )
    return hashlib.sha256(identity.encode()).hexdigest(), doc


def vector_document(item: dict, category: str, model_name: str) -> tuple[str, dict]:
    contract = embedding_contract()
    supplied = item.get("embedding_contract") or {}
    if model_name != contract["model_name"] or any(
        supplied.get(k) != v for k, v in contract.items()
    ):
        raise ValueError(
            "Vector model, revision, dimension and preprocessing must match the query contract"
        )
    vector = item.get("vector") or []
    if (
        len(vector) != contract["dimension"]
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)
        or not any(vector)
    ):
        raise ValueError("Invalid embedding vector")
    metadata = item.get("metadata") or {}
    doc_id, doc = pattern_document(
        {
            **item,
            "pattern": item.get("name"),
            "canonical": (item.get("canonical") or item.get("original_name")
                          or item.get("original_text") or metadata.get("canonical")
                          or item.get("name")),
            "entity_id": (
                item.get("entity_id")
                if item.get("entity_id") is not None
                else metadata.get("entity_id")
            ),
        },
        category,
        0,
    )
    doc.update(
        {"vector": vector, "model_name": model_name, "embedding_contract": contract}
    )
    return doc_id, doc


def bulk_counts(response: Any, expected: int) -> tuple[int, int]:
    """A successful HTTP response may still contain failed index operations."""
    items = response.get("items", [])
    succeeded = sum(
        1
        for item in items
        if len(item) == 1
        and 200 <= next(iter(item.values())).get("status", 0) < 300
        and not next(iter(item.values())).get("error")
    )
    failed = max(expected - succeeded, 0)
    if len(items) != expected or response.get("errors"):
        failed = max(failed, 1)
    return succeeded, failed
