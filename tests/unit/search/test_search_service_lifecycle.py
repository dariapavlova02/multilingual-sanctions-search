"""Closing search must release owned resources and prevent late/new success."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ai_service.config import EmbeddingConfig
from ai_service.exceptions import AIServiceException
from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.layers.embeddings import embedding_service
from ai_service.layers.embeddings.indexing.elasticsearch_watchlist_adapter import (
    ElasticsearchWatchlistAdapter,
)
from ai_service.layers.search import elasticsearch_client
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import SearchOpts
from ai_service.layers.search.elasticsearch_client import ElasticsearchClientFactory


def service():
    return ElasticsearchWatchlistAdapter(HybridSearchConfig())


def provider():
    config = EmbeddingConfig()
    return SimpleNamespace(
        config=config,
        embedding_contract=config.embedding_contract(),
        close=Mock(),
        encode_one_async=AsyncMock(return_value=[1.0] + [0.0] * (config.dimension - 1)),
    )


def normalized():
    return NormalizationResult(
        normalized="Example Person", tokens=["Example", "Person"], trace=[]
    )


@pytest.mark.asyncio
async def test_closed_service_cannot_reinitialize():
    search = service()
    await search.close()
    with pytest.raises(AIServiceException, match="closed"):
        search.initialize()
    assert search._client_factory is None


@pytest.mark.asyncio
async def test_closed_service_cannot_create_a_model(monkeypatch):
    search = service()
    constructor = Mock(return_value=provider())
    monkeypatch.setattr(embedding_service, "EmbeddingService", constructor)
    await search.close()
    with pytest.raises(RuntimeError, match="closed"):
        await search._get_embedding_service()
    constructor.assert_not_called()


@pytest.mark.asyncio
async def test_owned_model_and_client_are_closed_once(monkeypatch):
    search, encoder = service(), provider()
    monkeypatch.setattr(
        embedding_service, "EmbeddingService", Mock(return_value=encoder)
    )
    assert await search._get_embedding_service() is encoder
    factory = SimpleNamespace(close=AsyncMock())
    search._client_factory = factory
    await search.close()
    await search.close()
    encoder.close.assert_called_once()
    factory.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_borrowed_runtime_model_is_not_closed():
    search, encoder = service(), provider()
    search._embedding_service = encoder
    await search.close()
    encoder.close.assert_not_called()
    assert search._embedding_service is None


@pytest.mark.asyncio
async def test_cleanup_attempts_clients_when_owned_model_close_fails(monkeypatch):
    search, encoder = service(), provider()
    encoder.close.side_effect = RuntimeError("controlled cleanup failure")
    monkeypatch.setattr(
        embedding_service, "EmbeddingService", Mock(return_value=encoder)
    )
    await search._get_embedding_service()
    factory = SimpleNamespace(close=AsyncMock())
    search._client_factory = factory
    with pytest.raises(RuntimeError, match="cleanup"):
        await search.close()
    factory.close.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        await search._get_embedding_service()


@pytest.mark.asyncio
async def test_close_clears_cached_source_records():
    search = service()
    for name in ("_search_cache", "_embedding_cache", "_query_cache"):
        getattr(search, name)["source"] = object()
    await search.close()
    assert (
        not search._search_cache
        and not search._embedding_cache
        and not search._query_cache
    )


@pytest.mark.asyncio
async def test_closed_health_does_not_query_or_reopen_adapters():
    search = service()
    search._initialized = True
    adapter = SimpleNamespace(
        health_check=AsyncMock(return_value={"status": "healthy", "connected": True})
    )
    search._ac_adapter = search._vector_adapter = adapter
    await search.close()
    health = await search.health_check()
    assert health["status"] == "unhealthy"
    assert health["closed"] is True
    adapter.health_check.assert_not_awaited()


@pytest.mark.asyncio
async def test_inflight_search_cannot_return_success_after_close():
    search = service()
    started, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        started.set()
        await release.wait()
        return []

    search._find_candidates_within_deadline = blocked
    pending = asyncio.create_task(
        search.find_candidates(normalized(), "Example Person", SearchOpts())
    )
    await asyncio.wait_for(started.wait(), 1)
    await search.close()
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    assert search.get_metrics().failed_requests == 1
    assert search.get_metrics().successful_requests == 0


@pytest.mark.asyncio
async def test_inflight_embedding_cannot_publish_after_close():
    search, encoder = service(), provider()
    started, release = asyncio.Event(), asyncio.Event()

    async def encode(text):
        started.set()
        await release.wait()
        return [1.0] + [0.0] * (encoder.config.dimension - 1)

    encoder.encode_one_async = encode
    search._embedding_service = encoder
    pending = asyncio.create_task(
        search._build_query_vector(normalized(), "Example Person")
    )
    await asyncio.wait_for(started.wait(), 1)
    await search.close()
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    assert not search._embedding_cache


@pytest.mark.asyncio
async def test_closed_client_factory_cannot_create_another_connection(monkeypatch):
    client = SimpleNamespace(close=AsyncMock())
    constructor = Mock(return_value=client)
    monkeypatch.setattr(elasticsearch_client, "AsyncElasticsearch", constructor)
    factory = ElasticsearchClientFactory(HybridSearchConfig())
    assert await factory.get_client() is client
    await factory.close()
    with pytest.raises(RuntimeError, match="closed"):
        await factory.get_client()
    constructor.assert_called_once()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,lock_name,cache_name,value",
    [
        ("_cache_embedding", "_cache_lock", "_embedding_cache", [1.0]),
        ("_cache_search_result", "_search_cache_lock", "_search_cache", []),
        ("_cache_query", "_query_cache_lock", "_query_cache", {"source": "record"}),
    ],
)
async def test_waiting_cache_writer_cannot_repopulate_closed_service(
    operation, lock_name, cache_name, value
):
    search = service()
    lock = getattr(search, lock_name)
    await lock.acquire()
    pending = asyncio.create_task(getattr(search, operation)("key", value))
    await asyncio.sleep(0)
    await search.close()
    lock.release()
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    assert not getattr(search, cache_name)


@pytest.mark.asyncio
async def test_all_clients_close_even_if_one_cleanup_fails():
    factory = ElasticsearchClientFactory(HybridSearchConfig())
    first = SimpleNamespace(
        close=AsyncMock(side_effect=RuntimeError("controlled secret detail"))
    )
    second = SimpleNamespace(close=AsyncMock())
    factory._clients = {"first": first, "second": second}
    with pytest.raises(RuntimeError, match="cleanup") as error:
        await factory.close()
    assert "controlled secret detail" not in str(error.value)
    first.close.assert_awaited_once()
    second.close.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        await factory.get_client()


@pytest.mark.asyncio
async def test_concurrent_close_waits_for_same_connection_cleanup():
    factory = ElasticsearchClientFactory(HybridSearchConfig())
    started, release = asyncio.Event(), asyncio.Event()

    async def close():
        started.set()
        await release.wait()

    client = SimpleNamespace(close=AsyncMock(side_effect=close))
    factory._clients = {"cluster": client}
    first = asyncio.create_task(factory.close())
    await asyncio.wait_for(started.wait(), 1)
    second = asyncio.create_task(factory.close())
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    await asyncio.gather(first, second)
    client.close.assert_awaited_once()
