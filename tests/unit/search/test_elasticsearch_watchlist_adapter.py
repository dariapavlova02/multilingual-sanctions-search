"""Legacy entry points must obey canonical source, model and failure contracts.

The retired prototype tests accepted random vectors, unavailable => [], no-op
mutations and snapshot success without inspecting backend replies. The replacement
suite exercises the supported public search API and rejects the obsolete settings.
Actual ingestion and local-snapshot coverage lives in their respective suites.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ai_service.config import EmbeddingConfig
from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.contracts.trace_models import SearchTrace
from ai_service.layers.embeddings.indexing.elasticsearch_watchlist_adapter import (
    ElasticsearchWatchlistAdapter,
    ElasticsearchWatchlistConfig,
    create_elasticsearch_enhanced_adapter,
    create_elasticsearch_watchlist_adapter,
)
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import Candidate, SearchMode, SearchOpts
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.index_schema import SOURCE_COVERAGE_VERSION, index_mapping

FACTORIES = [
    create_elasticsearch_watchlist_adapter,
    create_elasticsearch_enhanced_adapter,
]


def normalized(text="Example Person"):
    return NormalizationResult(
        normalized=text, tokens=text.split(), trace=[], language="en"
    )


def candidate(identity="entity-a", score=0.9, mode=SearchMode.AC):
    return Candidate(
        doc_id="row-" + identity,
        score=score,
        text="Example Person",
        entity_type="person",
        metadata={
            "entity_id": identity,
            "source_id": identity,
            "source": "test-source",
        },
        search_mode=mode,
        match_fields=["normalized_text"],
        confidence=score,
    )


@pytest.fixture
def available():
    config = HybridSearchConfig(
        elasticsearch={
            "hosts": ["http://controlled.invalid:9200"],
            "ac_index": "test_ac",
            "vector_index": "test_vectors",
        }
    )
    service = ElasticsearchWatchlistAdapter(config)
    service._initialized = True
    mappings = {}
    for name, vectors in [
        (config.elasticsearch.ac_index, False),
        (config.elasticsearch.vector_index, True),
    ]:
        mapping = index_mapping(config, vectors=vectors)["mappings"]
        mapping["_meta"].update(
            ingestion_status="completed",
            generation="generation-a",
            source_manifest="manifest-a",
            source_coverage_version=SOURCE_COVERAGE_VERSION,
        )
        mappings[name] = {name: {"mappings": mapping}}
    client = MagicMock()
    client.options.return_value = client
    client.indices.get_mapping = AsyncMock(side_effect=lambda *, index: mappings[index])
    client.count = AsyncMock(return_value={"count": 1})
    factory = SimpleNamespace(
        get_client=AsyncMock(return_value=client),
        get_connection_stats=AsyncMock(return_value={}),
        close=AsyncMock(),
    )
    service._client_factory = factory
    ac = SimpleNamespace(
        _connected=True,
        search=AsyncMock(return_value=[candidate()]),
        health_check=AsyncMock(return_value={"status": "healthy", "connected": True}),
    )
    vector = SimpleNamespace(
        _connected=True,
        search=AsyncMock(return_value=[candidate(mode=SearchMode.VECTOR)]),
        health_check=AsyncMock(return_value={"status": "healthy", "connected": True}),
    )
    service._ac_adapter, service._vector_adapter = ac, vector
    embedding_config = EmbeddingConfig()
    encoder = SimpleNamespace(
        config=embedding_config,
        embedding_contract=embedding_config.embedding_contract(),
        encode_one_async=AsyncMock(
            return_value=[1.0] + [0.0] * (embedding_config.dimension - 1)
        ),
    )
    service._embedding_service = encoder
    return SimpleNamespace(
        service=service,
        client=client,
        factory=factory,
        ac=ac,
        vector=vector,
        encoder=encoder,
        mappings=mappings,
    )


def test_legacy_imports_share_the_canonical_security_boundary():
    assert ElasticsearchWatchlistAdapter is HybridSearchService
    assert ElasticsearchWatchlistConfig is HybridSearchConfig


@pytest.mark.parametrize("factory", FACTORIES)
def test_factory_preserves_configured_cluster_and_owns_configuration(factory):
    config = ElasticsearchWatchlistConfig(
        elasticsearch={
            "hosts": ["https://search.example:9200"],
            "ac_index": "approved_ac",
            "vector_index": "approved_vectors",
        }
    )
    service = factory(config)
    config.elasticsearch.hosts.append("https://different.example:9200")
    assert type(service) is HybridSearchService
    assert service.config.elasticsearch.hosts == ["https://search.example:9200"]
    assert service.config.elasticsearch.verify_certs is True
    assert service.config.elasticsearch.ac_index == "approved_ac"
    assert service.config.elasticsearch.vector_index == "approved_vectors"


@pytest.mark.parametrize("factory", FACTORIES)
def test_factory_reads_canonical_environment(factory, monkeypatch):
    monkeypatch.setenv("ES_HOSTS", "https://approved.example:9200")
    monkeypatch.setenv("ES_INDEX_PREFIX", "approved")
    monkeypatch.delenv("ES_AC_INDEX", raising=False)
    monkeypatch.delenv("ES_VECTOR_INDEX", raising=False)
    service = factory()
    assert service.config.elasticsearch.hosts == ["https://approved.example:9200"]
    assert service.config.elasticsearch.ac_index == "approved_ac_patterns"
    assert service.config.elasticsearch.vector_index == "approved_vectors"


@pytest.mark.parametrize("factory", FACTORIES)
def test_factory_rejects_unrelated_local_fallback_without_using_it(factory):
    fallback = MagicMock()
    with pytest.raises(ValueError, match="Local fallback is not supported"):
        factory(fallback_config=fallback)
    assert not fallback.mock_calls


def test_constructor_rejects_legacy_fallback_without_using_it():
    fallback = MagicMock()
    with pytest.raises(TypeError):
        ElasticsearchWatchlistAdapter(fallback_service=fallback)
    assert not fallback.mock_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("es_url", "https://old.example:9200"),
        ("es_auth", "private:credential"),
        ("es_verify_ssl", False),
        ("persons_index", "old_persons"),
        ("orgs_index", "old_orgs"),
        ("ac_threshold", 0.8),
        ("max_ac_results", 50),
        ("fallback_timeout", 5),
    ],
)
def test_obsolete_configuration_is_not_silently_ignored(field, value):
    with pytest.raises(ValidationError) as error:
        ElasticsearchWatchlistConfig(**{field: value})
    assert "private:credential" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [SearchMode.AC, SearchMode.VECTOR, SearchMode.HYBRID])
async def test_success_uses_canonical_source_records(available, mode):
    trace = SearchTrace(enabled=True)
    rows = await available.service.find_candidates(
        normalized(), "Example Person", SearchOpts(search_mode=mode), trace
    )
    assert len(rows) == 1
    assert rows[0].metadata["source_id"] == "entity-a"
    assert rows[0].metadata["source"] == "test-source"
    assert rows[0].text == "Example Person"
    assert available.service.get_metrics().successful_requests == 1
    assert available.client.indices.get_mapping.await_count >= 2
    assert trace.steps


@pytest.mark.asyncio
async def test_vector_search_uses_pinned_query_embeddings(available):
    opts = SearchOpts(search_mode=SearchMode.VECTOR)
    await available.service.find_candidates(normalized(), "Example Person", opts)
    await available.service.clear_search_cache()
    await available.service.find_candidates(normalized(), "Example Person", opts)
    sent = [call.kwargs["query"] for call in available.vector.search.await_args_list]
    assert sent == [available.encoder.encode_one_async.return_value] * 2
    available.encoder.encode_one_async.assert_awaited_once_with("Example Person")
    assert all(
        call.kwargs["index_name"] == "test_vectors"
        for call in available.vector.search.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[], [0.0] * 384, [float("nan")] * 384])
async def test_invalid_model_output_never_reaches_backend(available, vector):
    available.encoder.encode_one_async.return_value = vector
    with pytest.raises(RuntimeError, match="unavailable"):
        await available.service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.VECTOR)
        )
    available.vector.search.assert_not_awaited()
    assert available.service.get_metrics().failed_requests == 1


@pytest.mark.asyncio
async def test_provider_failure_is_not_successful_empty_screening(available):
    available.encoder.encode_one_async.side_effect = RuntimeError(
        "controlled provider failure"
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        await available.service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.VECTOR)
        )
    available.vector.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_backend_failure_is_not_successful_empty_screening(available):
    available.ac.search.side_effect = RuntimeError("controlled backend failure")
    with pytest.raises(RuntimeError, match="unavailable"):
        await available.service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC)
        )
    assert available.service.get_metrics().failed_requests == 1
    assert available.service.get_metrics().successful_requests == 0


@pytest.mark.asyncio
async def test_disconnected_adapter_cannot_return_success(available):
    available.ac._connected = False
    with pytest.raises(RuntimeError, match="unavailable"):
        await available.service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC)
        )


@pytest.mark.asyncio
async def test_successful_empty_search_requires_available_source(available):
    available.ac.search.return_value = []
    assert (
        await available.service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC)
        )
        == []
    )
    assert available.service.get_metrics().successful_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect", ["incomplete", "generation", "manifest", "coverage", "empty", "provider"]
)
async def test_incoherent_snapshot_or_provider_rejects_search(available, defect):
    meta = available.mappings["test_vectors"]["test_vectors"]["mappings"]["_meta"]
    if defect == "incomplete":
        meta["ingestion_status"] = "loading"
    elif defect == "generation":
        meta["generation"] = "different-generation"
    elif defect == "manifest":
        meta["source_manifest"] = "different-source"
    elif defect == "coverage":
        meta.pop("source_coverage_version")
    elif defect == "empty":
        available.client.count.return_value = {"count": 0}
    else:
        available.encoder.embedding_contract = {
            **available.encoder.embedding_contract,
            "revision": "different",
        }
    with pytest.raises(RuntimeError):
        await available.service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.VECTOR)
        )
    available.vector.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_cached_results_do_not_bypass_current_source_health(available):
    opts = SearchOpts(search_mode=SearchMode.AC)
    await available.service.find_candidates(normalized(), "Example Person", opts)
    available.client.indices.get_mapping.side_effect = RuntimeError(
        "backend unavailable"
    )
    with pytest.raises(RuntimeError):
        await available.service.find_candidates(normalized(), "Example Person", opts)
    assert available.service.get_metrics().failed_requests == 1
    assert available.service.get_metrics().successful_requests == 1


@pytest.mark.asyncio
async def test_failed_health_remains_failed_until_backend_recovers(available):
    available.ac.health_check.return_value = {"status": "unhealthy", "connected": False}
    for _ in range(2):
        health = await available.service.health_check()
        assert health["status"] == "unhealthy"
        assert health["fallback_enabled"] is False
    available.ac.health_check.return_value = {"status": "healthy", "connected": True}
    assert (await available.service.health_check())["status"] == "healthy"


@pytest.mark.asyncio
async def test_deadline_prevents_indefinite_backend_wait(available):
    cancelled = asyncio.Event()

    async def blocked(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    available.ac.search.side_effect = blocked
    with pytest.raises(RuntimeError, match="deadline"):
        await available.service.find_candidates(
            normalized(),
            "Example Person",
            SearchOpts(search_mode=SearchMode.AC, timeout_ms=100),
        )
    assert cancelled.is_set()
    assert available.service.get_metrics().failed_requests == 1


@pytest.mark.asyncio
async def test_close_releases_owned_connection_factory(available):
    await available.service.close()
    available.factory.close.assert_awaited_once()
