"""Fail-closed screening: dependency failures cannot select unrelated local data.

Replaces the old success-on-outage expectations. Readiness uses real schema checks;
only ES I/O, adapter responses and the embedding model are controlled here.
"""

import asyncio

import pytest

from ai_service.layers.search.contracts import SearchMode, SearchOpts
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.config import HybridSearchConfig
from tests.search_service_support import candidate, normalized, search_service, assert_no_local_search

MODES = [SearchMode.AC, SearchMode.FUZZY, SearchMode.VECTOR, SearchMode.HYBRID]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fallback", [True, False])
async def test_unavailable_source_fails_even_with_local_services(mode, fallback):
    service = search_service(fallback=fallback)
    client = await service._client_factory.get_client()
    client.indices.get_mapping.side_effect = ConnectionError("Backend unavailable")
    with pytest.raises(ConnectionError, match="Backend unavailable"):
        await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=mode))
    assert_no_local_search(service)
    service._ac_adapter.search.assert_not_awaited()
    service._vector_adapter.search.assert_not_awaited()
    metrics = service.get_metrics()
    assert (metrics.total_requests, metrics.successful_requests, metrics.failed_requests) == (1, 0, 1)


@pytest.mark.parametrize("mode", [SearchMode.AC, SearchMode.VECTOR, SearchMode.HYBRID])
@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize("has_hits", [False, True])
async def test_disconnected_adapter_response_is_not_a_completed_screening(mode, fallback, has_hits):
    service = search_service(fallback=fallback)
    adapter = service._vector_adapter if mode == SearchMode.VECTOR else service._ac_adapter
    adapter._connected = False
    adapter.search.return_value = [candidate(mode=mode)] if has_hits else []
    with pytest.raises(RuntimeError, match="search is unavailable"):
        await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=mode))
    assert_no_local_search(service)
    assert (await service.get_search_cache_stats())["cache_size"] == 0


@pytest.mark.parametrize("mode", MODES)
async def test_partial_stage_failure_discards_hits_and_does_not_cache(mode):
    service = search_service()
    async def incomplete_pages(*args, **kwargs):
        yield [{"_id": "a", "_source": {"pattern": "Example Person", "entity_id": "a", "entity_type": "person", "source_list": "active-source"}}]
        raise RuntimeError("Incomplete active source")
    service._ac_adapter.iter_documents = incomplete_pages
    if mode == SearchMode.AC:
        service._ac_adapter.search.side_effect = RuntimeError("Incomplete active source")
    elif mode == SearchMode.VECTOR:
        service._vector_adapter.search.side_effect = RuntimeError("Incomplete active source")
    # Hybrid receives no AC hits, then fails during its active fuzzy scan.
    with pytest.raises(RuntimeError, match="search is unavailable"):
        await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=mode))
    assert_no_local_search(service)
    assert (await service.get_search_cache_stats())["cache_size"] == 0
    assert service.get_metrics().failed_requests == 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("has_hits", [False, True])
async def test_warm_cache_cannot_bypass_source_outage(mode, has_hits):
    service = search_service()
    service._ac_adapter.search.return_value = [candidate()] if has_hits else []
    service._vector_adapter.search.return_value = [candidate(mode=SearchMode.VECTOR)] if has_hits else []
    # An AC document iterator also exercises a populated fuzzy cache.
    async def pages(*args, **kwargs):
        yield [{"_id": "active", "_source": {"pattern": "Example Person", "entity_id": "active", "entity_type": "person", "source_list": "active-source"}}] if has_hits else []
    service._ac_adapter.iter_documents = pages
    opts = SearchOpts(search_mode=mode)
    first = await service.find_candidates(normalized(), "Example Person", opts)
    second = await service.find_candidates(normalized(), "Example Person", opts)
    assert first == second
    assert bool(first) is has_hits
    assert (await service.get_search_cache_stats())["cache_size"] == 1
    client = await service._client_factory.get_client()
    client.indices.get_mapping.side_effect = ConnectionError("Backend unavailable")
    with pytest.raises(ConnectionError, match="Backend unavailable"):
        await service.find_candidates(normalized(), "Example Person", opts)
    assert_no_local_search(service)
    metrics = service.get_metrics()
    assert (metrics.total_requests, metrics.successful_requests, metrics.failed_requests) == (3, 2, 1)


@pytest.mark.parametrize("stage", ["readiness", "adapter", "embedding"])
async def test_request_deadline_cancels_dependency_and_counts_one_failure(stage):
    service = search_service()
    cancelled = asyncio.Event()
    async def stalled(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    mode = SearchMode.AC
    if stage == "readiness":
        service._client_factory.get_client.side_effect = stalled
    elif stage == "adapter":
        service._ac_adapter.search.side_effect = stalled
    else:
        service._embedding_service.encode_one_async.side_effect = stalled
        mode = SearchMode.VECTOR
    with pytest.raises(RuntimeError, match="exceeded its deadline"):
        await asyncio.wait_for(service.find_candidates(normalized(), "Example Person",
            SearchOpts(search_mode=mode, timeout_ms=100)), timeout=2)
    assert cancelled.is_set()
    assert_no_local_search(service)
    metrics = service.get_metrics()
    assert (metrics.total_requests, metrics.successful_requests, metrics.failed_requests) == (1, 0, 1)
    assert (await service.get_search_cache_stats())["cache_size"] == 0


@pytest.mark.parametrize("adapter_name", ["_ac_adapter", "_vector_adapter"])
@pytest.mark.parametrize("state", [{"status": "unhealthy"}, {"status": "healthy", "connected": False}, {}, None])
async def test_health_does_not_hide_adapter_failure(adapter_name, state):
    service = search_service()
    getattr(service, adapter_name).health_check.return_value = state
    health = await service.health_check()
    assert health["status"] == "unhealthy"
    assert health["fallback_enabled"] is False
    assert_no_local_search(service)


async def test_health_before_initialization_is_not_healthy():
    service = HybridSearchService(HybridSearchConfig())
    health = await service.health_check()
    assert health["status"] == "unhealthy"
    assert health["initialized"] is False


@pytest.mark.parametrize("adapter_name", ["_ac_adapter", "_vector_adapter"])
async def test_health_with_missing_required_adapter_is_not_healthy(adapter_name):
    service = search_service()
    setattr(service, adapter_name, None)
    assert (await service.health_check())["status"] == "unhealthy"
