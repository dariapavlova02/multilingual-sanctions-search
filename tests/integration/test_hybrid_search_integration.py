"""Real Elasticsearch contracts, with synthetic vectors rather than a quality benchmark.

Run through scripts/run_regression_gate.py. Missing infrastructure and backend
errors fail the gate; they are never translated into skipped tests or clearance.
"""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from ai_service.api import admin_endpoints
from ai_service.api.ingestion_jobs import IngestionBusy, IngestionJobStore
from ai_service.config import EmbeddingConfig
from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.contracts.trace_models import SearchTrace
from ai_service.layers.search.contracts import SearchMode, SearchOpts
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.index_schema import embedding_contract, ensure_index, pattern_document

pytestmark = [pytest.mark.integration, pytest.mark.docker]


class SyntheticEncoder:
    config = EmbeddingConfig()

    async def encode_one_async(self, text):
        values = [0.0] * self.config.dimension
        values[1 if text == "Remote Person" else 0] = 1.0
        return values


def normalized(text):
    return NormalizationResult(normalized=text, tokens=text.split(), trace=[])


async def import_vectors(records):
    encoder = SyntheticEncoder()
    vectors = [{**doc, "name": doc["pattern"],
        "vector": await encoder.encode_one_async(doc["pattern"]),
        "embedding_contract": embedding_contract()} for doc in records]
    await admin_endpoints._load_vectors_background(vectors, "person", encoder.config.model_name, 2)
    return dict(admin_endpoints.loading_status["vectors"])


@pytest.fixture
async def prepared_search(owned_elasticsearch):
    backend = owned_elasticsearch
    records = [
        {"pattern": "Source Example", "canonical": "Source Example", "entity_id": "shared-id",
         "source_list": "synthetic-one", "country": "UA", "dob": "1980-01-01", "metadata": {"itn": "1234567890"}},
        {"pattern": "Alternate Example", "canonical": "Source Example", "entity_id": "shared-id",
         "source_list": "synthetic-one", "country": "UA", "dob": "1980-01-01", "metadata": {"itn": "1234567890"}},
        {"pattern": "Remote Person", "entity_id": "shared-id", "source_list": "synthetic-two", "country": "GB"},
    ]
    for tier, docs in [(0, [records[0], records[2]]), (2, [records[1]])]:
        await admin_endpoints._load_ac_patterns_background(docs, "person", tier, 2)
        assert admin_endpoints.loading_status["ac_patterns"]["status"] == "completed"
    service = HybridSearchService(backend.config)
    service._embedding_service = SyntheticEncoder()
    try:
        yield SimpleNamespace(**vars(backend), service=service, records=records)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_empty_and_partial_snapshots_are_not_ready(owned_elasticsearch):
    config = owned_elasticsearch.config
    await ensure_index(owned_elasticsearch.client, config.elasticsearch.ac_index, config)
    service = HybridSearchService(config)
    try:
        with pytest.raises(RuntimeError, match="completed ingestion"):
            await service.readiness(require_vectors=False)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_incremental_vector_import_publishes_only_complete_coverage(prepared_search):
    item = prepared_search
    first = await import_vectors(item.records[:1])
    assert first["status"] == "completed" and first["snapshot_ready"] is False
    assert first["missing_or_changed_vectors"] == 2
    with pytest.raises(RuntimeError, match="completed ingestion"):
        await item.service.readiness()
    assert await item.service.readiness(require_vectors=False)
    second = await import_vectors(item.records[1:])
    assert second["snapshot_ready"] is True and second["extra_vectors"] == 0
    assert len(set((await item.service.readiness()).values())) == 1
    for index in (item.config.elasticsearch.ac_index, item.config.elasticsearch.vector_index):
        assert (await item.client.count(index=index))["count"] == 3
    alias_id, _ = pattern_document(item.records[1], "person", 2)
    ac = (await item.client.get(index=item.config.elasticsearch.ac_index, id=alias_id))["_source"]
    vector = (await item.client.get(index=item.config.elasticsearch.vector_index, id=alias_id))["_source"]
    assert {key: vector[key] for key in ac} == ac
    assert vector["original_text"] == "Source Example" and vector["tier"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,stage", [(SearchMode.AC, "AC"), (SearchMode.FUZZY, "LEXICAL"),
                                       (SearchMode.VECTOR, "SEMANTIC"), (SearchMode.HYBRID, "HYBRID")])
async def test_real_search_modes_preserve_source_evidence_and_trace(prepared_search, mode, stage):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    trace = SearchTrace(enabled=True)
    results = await item.service.find_candidates(normalized("Alternate Example"), "Alternate Example",
        SearchOpts(search_mode=mode, threshold=0.95, vector_min_score=0.95), trace)
    assert len(results) == 1
    hit = results[0]
    assert hit.metadata["entity_id"] == "shared-id"
    assert hit.metadata["source"] == "synthetic-one"
    assert hit.metadata["canonical"] == "Source Example"
    assert hit.metadata["itn"] == "1234567890"
    assert hit.metadata["country"] == "UA" and hit.metadata["dob"] == "1980-01-01"
    assert trace.get_stage_steps(stage)
    assert any(step.hits for step in trace.steps)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [SearchMode.AC, SearchMode.FUZZY, SearchMode.VECTOR, SearchMode.HYBRID])
async def test_entity_and_source_filters_are_enforced(prepared_search, mode):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    for filters in [{"entity_types": ["organization"]}, {"metadata_filters": {"source": "absent-source"}}]:
        assert await item.service.find_candidates(normalized("Alternate Example"), "Alternate Example",
            SearchOpts(search_mode=mode, threshold=0.95, **filters)) == []
    included = await item.service.find_candidates(normalized("Alternate Example"), "Alternate Example",
        SearchOpts(search_mode=mode, threshold=0.95, metadata_filters={"source": "synthetic-one"}))
    assert len(included) == 1 and included[0].metadata["source"] == "synthetic-one"


@pytest.mark.asyncio
async def test_source_update_invalidates_vectors_and_previous_search_cache(prepared_search):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    opts = SearchOpts(search_mode=SearchMode.VECTOR, threshold=0.95)
    assert await item.service.find_candidates(normalized("Source Example"), "Source Example", opts)
    changed = copy.deepcopy(item.records[0])
    changed["dob"] = "1981-02-03"
    await admin_endpoints._load_ac_patterns_background([changed], "person", 0, 1)
    with pytest.raises(RuntimeError, match="snapshot generation"):
        await item.service.find_candidates(normalized("Source Example"), "Source Example", opts)
    partial = await import_vectors(item.records[2:])
    assert partial["snapshot_ready"] is False and partial["missing_or_changed_vectors"] == 1
    assert (await import_vectors([changed]))["snapshot_ready"] is True
    assert await item.service.readiness()


@pytest.mark.asyncio
async def test_unknown_vector_is_rejected_before_changing_ready_snapshot(prepared_search):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    before = await item.service.readiness()
    unknown = {**item.records[0], "entity_id": "unknown-entity"}
    status = await import_vectors([unknown])
    assert status["status"] == "error"
    assert "unknown-entity" not in status["error"]
    assert await item.service.readiness() == before


@pytest.mark.asyncio
async def test_legacy_metadata_requires_revalidation_of_existing_vectors(prepared_search):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    mapping = await item.client.indices.get_mapping(index=item.config.elasticsearch.vector_index)
    meta = mapping[item.config.elasticsearch.vector_index]["mappings"]["_meta"]
    meta.pop("source_coverage_version")
    await item.client.indices.put_mapping(index=item.config.elasticsearch.vector_index, body={"_meta": meta})
    with pytest.raises(RuntimeError, match="source coverage verification"):
        await item.service.readiness()
    assert (await import_vectors(item.records[:1]))["snapshot_ready"]
    assert await item.service.readiness()


@pytest.mark.asyncio
async def test_extra_stale_vector_prevents_publication(prepared_search):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    document = (await item.client.search(index=item.config.elasticsearch.vector_index, size=1))["hits"]["hits"][0]["_source"]
    await item.client.index(index=item.config.elasticsearch.vector_index, id="synthetic-stale", document=document, refresh=True)
    status = await import_vectors(item.records)
    assert status["snapshot_ready"] is False and status["extra_vectors"] == 1
    with pytest.raises(RuntimeError, match="completed ingestion"):
        await item.service.readiness()


@pytest.mark.asyncio
async def test_vector_reservation_blocks_http_ac_writer(prepared_search):
    item = prepared_search
    store = IngestionJobStore()
    job = store.reserve("vectors", item.config.elasticsearch.vector_index, 1,
                        related_indices=[item.config.elasticsearch.ac_index])
    try:
        with pytest.raises(IngestionBusy):
            store.reserve("ac_patterns", item.config.elasticsearch.ac_index, 1)
    finally:
        job.close()


@pytest.mark.asyncio
async def test_closed_index_cannot_return_cached_clearance(prepared_search):
    item = prepared_search
    opts = SearchOpts(search_mode=SearchMode.AC, threshold=0.99)
    assert await item.service.find_candidates(normalized("Unrelated Zzz"), "Unrelated Zzz", opts) == []
    await item.client.indices.close(index=item.config.elasticsearch.ac_index)
    try:
        from elasticsearch import ApiError
        with pytest.raises((ApiError, RuntimeError)):
            await item.service.find_candidates(normalized("Unrelated Zzz"), "Unrelated Zzz", opts)
    finally:
        await item.client.indices.open(index=item.config.elasticsearch.ac_index)


@pytest.mark.asyncio
async def test_search_deadline_includes_model_queue_and_inference(prepared_search):
    item = prepared_search
    assert (await import_vectors(item.records))["snapshot_ready"]
    cancelled = asyncio.Event()
    async def slow_encoding(text):
        try:
            await asyncio.sleep(1)
            return [1.0] + [0.0] * 383
        finally:
            cancelled.set()
    item.service._embedding_service.encode_one_async = slow_encoding
    with pytest.raises(RuntimeError, match="deadline"):
        await item.service.find_candidates(normalized("Slow semantic query"), "Slow semantic query",
            SearchOpts(search_mode=SearchMode.VECTOR, timeout_ms=100))
    assert cancelled.is_set()
