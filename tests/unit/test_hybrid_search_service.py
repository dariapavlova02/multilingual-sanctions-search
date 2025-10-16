"""Public search behavior with current client, model and trace contracts."""

from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ai_service.contracts.trace_models import SearchTrace
from ai_service.exceptions import AIServiceException
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import SearchMode, SearchOpts
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from tests.search_service_support import candidate, normalized, search_service, assert_no_local_search


class TestHybridSearchService:
    @pytest.fixture
    def service(self):
        return search_service()

    def test_initialization_success(self):
        prefix = "ai_service.layers.search.hybrid_search_service."
        with patch(prefix + "ElasticsearchClientFactory") as factory, \
                patch(prefix + "ElasticsearchACAdapter") as ac, \
                patch(prefix + "ElasticsearchVectorAdapter") as vector:
            service = HybridSearchService(HybridSearchConfig())
            service.initialize()
            service.initialize()
            assert service.is_initialized()
            factory.assert_called_once()
            ac.assert_called_once_with(service.config, client_factory=factory.return_value)
            vector.assert_called_once_with(service.config, client_factory=factory.return_value)

    def test_initialization_failure(self):
        with patch("ai_service.layers.search.hybrid_search_service.ElasticsearchClientFactory",
                   side_effect=ConnectionError("Backend unavailable")):
            service = HybridSearchService(HybridSearchConfig())
            with pytest.raises(AIServiceException, match="initialization failed"):
                service.initialize()
            assert not service.is_initialized()

    async def test_find_candidates_ac_mode(self, service):
        service._ac_adapter.search.return_value = [candidate("a", 0.95), candidate("b", 0.85)]
        opts = SearchOpts(search_mode=SearchMode.AC)
        result = await service.find_candidates(normalized(), "Example Person", opts)
        assert [(r.doc_id, r.score, r.search_mode) for r in result] == [
            ("a", 0.95, SearchMode.AC), ("b", 0.85, SearchMode.AC),
        ]
        service._ac_adapter.search.assert_awaited_once_with(
            query="Example Person", opts=opts, index_name="test_ac",
        )
        service._vector_adapter.search.assert_not_awaited()
        assert_no_local_search(service)

    async def test_find_candidates_vector_mode(self, service):
        service._vector_adapter.search.return_value = [candidate("v", 0.85, SearchMode.VECTOR)]
        opts = SearchOpts(search_mode=SearchMode.VECTOR)
        result = await service.find_candidates(normalized(), "Example Person", opts)
        assert [(r.doc_id, r.score, r.search_mode) for r in result] == [("v", 0.85, SearchMode.VECTOR)]
        service._embedding_service.encode_one_async.assert_awaited_once_with("Example Person")
        service._vector_adapter.search.assert_awaited_once_with(
            query=[0.1] * 384, opts=opts, index_name="test_vector",
        )
        assert_no_local_search(service)

    async def test_find_candidates_hybrid_mode(self, service):
        # AC is accepted but below the escalation threshold; the active fuzzy scan is empty.
        service._ac_adapter.search.return_value = [candidate("a", 0.75)]
        service._vector_adapter.search.return_value = [candidate("v", 0.85, SearchMode.VECTOR)]
        opts = SearchOpts(search_mode=SearchMode.HYBRID, escalation_threshold=0.9)
        result = await service.find_candidates(normalized(), "Example Person", opts)
        assert [(r.doc_id, r.search_mode) for r in result] == [
            ("v", SearchMode.VECTOR), ("a", SearchMode.AC),
        ]
        assert service.get_metrics().escalation_triggered == 1
        assert_no_local_search(service)

    async def test_fallback_search(self, service):
        """A failed active query must never select a different local source."""
        service._ac_adapter.search.side_effect = ConnectionError("Elasticsearch unavailable")
        with pytest.raises(RuntimeError, match="search is unavailable"):
            await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC))
        assert_no_local_search(service)

    async def test_embedding_cache(self, service):
        first = await service._build_query_vector(normalized(), "Example Person")
        second = await service._build_query_vector(normalized(), "Example Person")
        assert first == second == [0.1] * 384
        service._embedding_service.encode_one_async.assert_awaited_once_with("Example Person")
        # A new encoder cannot silently query the existing index with another
        # vector space, whether the old query vector is cached or not.
        from ai_service.config import EmbeddingConfig
        service._embedding_service.config = EmbeddingConfig(revision="a" * 40)
        with pytest.raises(RuntimeError, match="Embedding provider contract"):
            await service._build_query_vector(normalized(), "Example Person")
        service._embedding_service.encode_one_async.assert_awaited_once()

    async def test_health_check(self, service):
        health = await service.health_check()
        assert health["status"] == "healthy"
        assert health["ac_adapter"]["status"] == "healthy"
        assert health["vector_adapter"]["status"] == "healthy"
        assert health["embedding_cache"]["cache_size"] == 0
        assert health["fallback_enabled"] is False

    async def test_metrics_collection(self, service):
        service._ac_adapter.search.return_value = [candidate()]
        for _ in range(2):
            await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC))
        metrics = service.get_metrics()
        assert (metrics.total_requests, metrics.successful_requests, metrics.failed_requests) == (2, 2, 0)
        assert metrics.ac_requests == 1  # The second request uses a verified cache entry.
        assert metrics.avg_hybrid_latency_ms > 0

    async def test_error_handling(self, service):
        service._ac_adapter.search.side_effect = ValueError("Malformed response")
        with pytest.raises(RuntimeError, match="search is unavailable") as exc:
            await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC))
        assert isinstance(exc.value.__cause__, ValueError)
        assert_no_local_search(service)
        assert (await service.get_search_cache_stats())["cache_size"] == 0

    def test_configuration_validation(self):
        config = HybridSearchConfig()
        service = HybridSearchService(config)
        assert service.config == config
        assert service.config is not config
        with pytest.raises(ValueError, match="hosts"):
            HybridSearchConfig(elasticsearch={"hosts": []})

    async def test_clear_embedding_cache(self, service):
        await service._build_query_vector(normalized(), "Example Person")
        assert (await service.get_embedding_cache_stats())["cache_size"] == 1
        await service.clear_embedding_cache()
        assert (await service.get_embedding_cache_stats())["cache_size"] == 0
        await service._build_query_vector(normalized(), "Example Person")
        assert service._embedding_service.encode_one_async.await_count == 2

    async def test_search_trace_creation(self, service):
        service._ac_adapter.search.return_value = [candidate()]
        trace = SearchTrace(enabled=True)
        await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC), trace)
        steps = trace.get_stage_steps("AC")
        assert len(steps) == 1
        assert steps[0].query == "Example Person"
        assert steps[0].hits[0].doc_id == "active"
        assert trace.to_dict()["total_hits"] == 1
        assert trace.get_total_time_ms() >= 0

    async def test_threshold_top_k_and_entity_filters_use_active_results(self, service):
        service._ac_adapter.search.return_value = [
            candidate("below", 0.4), candidate("a", 0.8), candidate("b", 0.9),
            replace(candidate("organization", 0.99), entity_type="organization"),
        ]
        result = await service.find_candidates(normalized(), "Example Person",
            SearchOpts(search_mode=SearchMode.AC, threshold=0.7, top_k=1, entity_types=["person"]))
        assert [r.doc_id for r in result] == ["b"]
        assert_no_local_search(service)

    @pytest.mark.parametrize("vector", [[0.0] * 384, [0.1] * 383, [float("nan")] * 384, [float("inf")] * 384])
    async def test_invalid_embedding_is_not_cached_or_searched(self, service, vector):
        service._embedding_service.encode_one_async.return_value = vector
        with pytest.raises(RuntimeError, match="search is unavailable"):
            await service.find_candidates(normalized(), "Example Person", SearchOpts(search_mode=SearchMode.VECTOR))
        service._vector_adapter.search.assert_not_awaited()
        assert (await service.get_embedding_cache_stats())["cache_size"] == 0
        assert_no_local_search(service)

    async def test_expired_embedding_is_rebuilt(self, service):
        await service._build_query_vector(normalized(), "Example Person")
        key = next(iter(service._embedding_cache))
        service._embedding_cache[key] = ([0.2] * 384, datetime.now() - timedelta(days=2))
        assert await service._build_query_vector(normalized(), "Example Person") == [0.1] * 384
        assert service._embedding_service.encode_one_async.await_count == 2
