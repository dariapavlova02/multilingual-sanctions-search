"""Snapshot publication and cleanup invariants without a running backend."""

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.api import admin_endpoints
from ai_service.api.snapshot_verification import SnapshotVerifier, VerificationLimits
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.index_schema import (
    embedding_contract, index_mapping, pattern_document, set_ingestion_status, vector_document,
)


class SnapshotClient:
    """Small paginated transport double; business logic uses the real verifier."""
    def __init__(self, config, documents):
        self.config = config
        self.documents = {config.elasticsearch.ac_index: dict(documents),
                          config.elasticsearch.vector_index: {}}
        self.mappings = {}
        for index, vectors in [(config.elasticsearch.ac_index, False),
                               (config.elasticsearch.vector_index, True)]:
            mapping = index_mapping(config, vectors=vectors)["mappings"]
            mapping["_meta"].update(ingestion_status="completed", generation="source-one",
                                     source_manifest={"sha256": "source-file"})
            self.mappings[index] = {"mappings": mapping}
        self.indices = SimpleNamespace(get_mapping=AsyncMock(side_effect=self.get_mapping),
            put_mapping=AsyncMock(side_effect=self.put_mapping), refresh=AsyncMock(),
            exists=AsyncMock(return_value=True))
        self.pits = {}
        self.closed = []
        self.delay = 0
        self.partial = False

    def options(self, **kwargs):
        return self

    async def get_mapping(self, *, index):
        return {index: copy.deepcopy(self.mappings[index])}

    async def put_mapping(self, *, index, body):
        self.mappings[index]["mappings"].update(copy.deepcopy(body))

    async def open_point_in_time(self, *, index, **kwargs):
        pit = str(len(self.pits) + len(self.closed))
        self.pits[pit] = copy.deepcopy(list(self.documents[index].items()))
        return {"id": pit}

    async def close_point_in_time(self, *, id):
        del self.pits[id]
        self.closed.append(id)

    async def search(self, *, body, **kwargs):
        await asyncio.sleep(self.delay)
        assert "vector" not in body["_source"]
        start = body.get("search_after", [-1])[0] + 1
        page = self.pits[body["pit"]["id"]][start:start + 1]
        hits = [{"_id": key, "_source": doc, "sort": [start + offset]}
                for offset, (key, doc) in enumerate(page)]
        return {"hits": {"hits": hits}, "timed_out": self.partial, "_shards": {"failed": 0}}

    async def bulk(self, *, operations):
        for action, document in zip(operations[::2], operations[1::2]):
            self.documents[action["index"]["_index"]][action["index"]["_id"]] = document
        return {"errors": False, "items": [{"index": {"status": 201}} for _ in operations[::2]]}


def input_vector(document):
    contract = embedding_contract()
    return vector_document({**document, "name": document["pattern"],
        "vector": [1.0] + [0.0] * (contract["dimension"] - 1),
        "embedding_contract": contract}, document["category"], contract["model_name"])


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_DIR", str(tmp_path))
    config = HybridSearchConfig.from_env()
    records = [pattern_document({"pattern": "Alternate Example", "canonical": "Canonical Example",
        "entity_id": "entity-one", "source_list": "synthetic", "metadata": {"itn": "1234567890"}}, "person", 2),
        pattern_document({"pattern": "Second Example", "entity_id": "entity-two", "source_list": "synthetic"}, "person", 0)]
    client = SnapshotClient(config, records)
    return config, records, client


@pytest.mark.asyncio
async def test_join_preserves_ac_identity_alias_tier_and_metadata(snapshot):
    config, records, client = snapshot
    vector = input_vector(records[0][1])
    vector[1]["metadata"]["itn"] = "untrusted-import"
    vector[1]["original_text"] = "Untrusted canonical"
    with SnapshotVerifier(client, config) as verifier:
        scratch = Path(verifier.scratch.name)
        joined = await verifier.prepare([vector])
        assert joined[0][0] == records[0][0]
        assert {k: joined[0][1][k] for k in records[0][1]} == records[0][1]
        assert joined[0][1]["tier"] == 2
        assert not client.pits
    assert not scratch.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,missing,extra", [("missing", 1, 0), ("changed", 1, 0), ("extra", 0, 1), ("none", 0, 0)])
async def test_coverage_rejects_missing_changed_or_foreign_sources(snapshot, fault, missing, extra):
    config, records, client = snapshot
    vectors = copy.deepcopy(dict(records))
    if fault == "missing":
        del vectors[records[0][0]]
    elif fault == "changed":
        vectors[records[0][0]]["metadata"]["itn"] = "changed"
    elif fault == "extra":
        vectors["foreign-source"] = copy.deepcopy(records[0][1])
    client.documents[config.elasticsearch.vector_index] = vectors
    with SnapshotVerifier(client, config) as verifier:
        await verifier.prepare([input_vector(records[0][1])])
        assert await verifier.coverage() == {"snapshot_ready": fault == "none",
            "missing_or_changed_vectors": missing, "extra_vectors": extra}
    assert len(client.closed) == 2 and not client.pits


@pytest.mark.asyncio
async def test_unknown_vector_does_not_publish_or_leak_scratch(snapshot):
    config, records, client = snapshot
    vector = input_vector(records[0][1])
    vector[1]["entity_id"] = "unknown"
    with SnapshotVerifier(client, config) as verifier:
        scratch = Path(verifier.scratch.name)
        with pytest.raises(ValueError, match="no matching AC"):
            await verifier.prepare([vector])
    assert not client.pits and not scratch.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["partial", "timeout", "bad-source"])
async def test_scan_failure_closes_pit_and_scratch(snapshot, fault):
    config, records, client = snapshot
    if fault == "partial":
        client.partial = True
    elif fault == "timeout":
        client.delay = 1
    else:
        client.documents[config.elasticsearch.ac_index][records[0][0]]["confidence"] = float("nan")
    with SnapshotVerifier(client, config) as verifier:
        scratch = Path(verifier.scratch.name)
        expected = {"partial": RuntimeError, "timeout": TimeoutError, "bad-source": ValueError}[fault]
        with pytest.raises(expected):
            async with asyncio.timeout(0.02):
                await verifier.prepare([input_vector(records[1][1])])
        assert not client.pits
    assert not scratch.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("generation", "changed"), ("ingestion_status", "loading"), ("source_manifest", {"sha256": "different"})])
async def test_source_change_during_verification_rejects_publication(snapshot, field, value):
    config, records, client = snapshot
    with SnapshotVerifier(client, config) as verifier:
        await verifier.prepare([input_vector(records[0][1])])
        client.mappings[config.elasticsearch.ac_index]["mappings"]["_meta"][field] = value
        with pytest.raises(RuntimeError, match="AC source changed"):
            await verifier.coverage()


@pytest.mark.asyncio
async def test_partial_import_is_completed_but_unready_until_remaining_rows_arrive(snapshot, monkeypatch):
    config, records, client = snapshot
    wrapper = SimpleNamespace(client=client, close=AsyncMock())
    monkeypatch.setattr(admin_endpoints, "ElasticsearchClient", lambda: wrapper)
    for offset, record in enumerate(records):
        await admin_endpoints._load_documents([input_vector(record[1])], "vectors", vectors=True)
        status = admin_endpoints.loading_status["vectors"]
        assert status["status"] == "completed"
        assert status["snapshot_ready"] == (offset == 1)
        assert status["missing_or_changed_vectors"] == 1 - offset
        meta = client.mappings[config.elasticsearch.vector_index]["mappings"]["_meta"]
        assert meta["ingestion_status"] == ("incomplete" if offset == 0 else "completed")
        assert meta["generation"] == "source-one"
        assert meta["source_manifest"] == {"sha256": "source-file"}
    assert wrapper.close.await_count == 2


@pytest.mark.asyncio
async def test_ac_upsert_replaces_previous_manifest(snapshot, monkeypatch):
    config, records, client = snapshot
    monkeypatch.setattr(admin_endpoints, "ElasticsearchClient", lambda: SimpleNamespace(client=client, close=AsyncMock()))
    await admin_endpoints._load_documents([records[0]], "ac_patterns")
    meta = client.mappings[config.elasticsearch.ac_index]["mappings"]["_meta"]
    assert meta["generation"] != "source-one"
    assert meta["source_manifest"] == {"kind": "api_upsert", "generation": meta["generation"]}
    await set_ingestion_status(client, config.elasticsearch.ac_index, "completed", "next", source_manifest=None)
    assert "source_manifest" not in client.mappings[config.elasticsearch.ac_index]["mappings"]["_meta"]


@pytest.mark.parametrize("value", ["0", "-1", "3601", "nan", "inf", "invalid"])
def test_invalid_verification_deadline_rejected(monkeypatch, value):
    monkeypatch.setenv("INGESTION_VERIFY_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        VerificationLimits()


def test_vector_alias_retains_canonical_source_fields(snapshot):
    _, records, _ = snapshot
    _, vector = input_vector(records[0][1])
    assert vector["original_text"] == "Canonical Example"
    assert vector["metadata"]["canonical"] == "Canonical Example"
