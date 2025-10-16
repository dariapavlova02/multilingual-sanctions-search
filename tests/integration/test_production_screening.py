"""Run with SANCTIONS_TEST_ES_URL against a disposable Elasticsearch cluster."""

import os
import uuid

import pytest

from ai_service.api import admin_endpoints
from ai_service.api.elasticsearch_wrapper import ElasticsearchClient
from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import SearchMode, SearchOpts
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.index_schema import ensure_index

pytestmark = [pytest.mark.integration, pytest.mark.docker]


@pytest.mark.asyncio
async def test_ingestion_search_idempotence_filters_and_empty_readiness(owned_elasticsearch):
    config = owned_elasticsearch.config
    service = HybridSearchService(config)
    async with ElasticsearchClient() as wrapper:
        try:
            await ensure_index(wrapper.client, config.elasticsearch.ac_index, config)
            with pytest.raises(RuntimeError, match="no completed ingestion"):
                await service.readiness(require_vectors=False)
            documents = [{"pattern": "John Smith", "entity_id": "synthetic-person", "source_list": "synthetic-test"}]
            for _ in range(2):
                await admin_endpoints._load_ac_patterns_background(documents, "person", "tier_0_exact", 1)
                assert admin_endpoints.loading_status["ac_patterns"]["status"] == "completed"
            assert (await wrapper.client.count(index=config.elasticsearch.ac_index))["count"] == 1
            norm = NormalizationResult(normalized="John Smith", tokens=["John", "Smith"], trace=[])
            results = await service.find_candidates(norm, "John Smith", SearchOpts(search_mode=SearchMode.AC, threshold=0.95))
            assert len(results) == 1
            assert results[0].metadata["entity_id"] == "synthetic-person"
            assert results[0].confidence == 1.0
            filtered = await service.find_candidates(norm, "John Smith", SearchOpts(
                search_mode=SearchMode.AC, entity_types=["organization"], threshold=0.95))
            assert filtered == []
            miss = NormalizationResult(normalized="Unrelated Example", tokens=["Unrelated", "Example"], trace=[])
            assert await service.find_candidates(miss, "Unrelated Example", SearchOpts(search_mode=SearchMode.AC)) == []
        finally:
            await service.close()
            for index in (config.elasticsearch.ac_index, config.elasticsearch.vector_index):
                if await wrapper.client.indices.exists(index=index):
                    await wrapper.client.indices.delete(index=index)
