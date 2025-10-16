"""
Unit tests for configuration error handling
"""

import pytest
from unittest.mock import patch, MagicMock

from ai_service.config.settings import SearchConfig
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.config import HybridSearchConfig


class TestSearchConfigErrorHandling:
    """Test SearchConfig error handling"""
    
    def test_validation_error_handling(self):
        """Test validation error handling"""
        with pytest.raises(Exception) as exc_info:
            SearchConfig(
                es_hosts=[],  # Invalid: empty hosts
                es_timeout=-1,  # Invalid: negative timeout
                escalation_threshold=1.5,  # Invalid: > 1.0
                vector_dimension=0,  # Invalid: zero dimension
                max_concurrent_requests=-1,  # Invalid: negative requests
                request_timeout_ms=0,  # Invalid: zero timeout
                embedding_cache_size=-1,  # Invalid: negative size
                embedding_cache_ttl_seconds=0  # Invalid: zero TTL
            )
        
        # Should contain multiple validation errors
        error_str = str(exc_info.value)
        assert "At least one Elasticsearch host must be specified" in error_str
        assert "Timeout must be positive" in error_str
        assert "Thresholds must be between 0.0 and 1.0" in error_str
        assert "Vector dimension must be positive" in error_str
        assert "Max concurrent requests must be positive" in error_str
        assert "Request timeout must be positive" in error_str
        assert "Embedding cache size must be positive" in error_str
        assert "Embedding cache TTL must be positive" in error_str
    
    def test_consistency_validation_error_handling(self):
        """Test consistency validation error handling"""
        with pytest.raises(Exception) as exc_info:
            SearchConfig(
                enable_escalation=True,
                escalation_threshold=0.4,  # Too low for escalation
                enable_fallback=True,
                fallback_threshold=0.05,  # Too low for fallback
                vector_similarity_threshold=0.2,  # Too low for similarity
                enable_embedding_cache=True,
                embedding_cache_size=50  # Too small for cache
            )
        
        # Should contain multiple consistency validation errors
        error_str = str(exc_info.value)
        assert "Escalation threshold should be greater than 0.5" in error_str
        assert "Fallback threshold should be greater than 0.1" in error_str
        assert "Vector similarity threshold should be greater than 0.3" in error_str
        assert "Embedding cache size should be at least 100" in error_str
    
    def test_reload_configuration_error_handling(self):
        """A malformed replacement must preserve all active settings."""
        config = SearchConfig(es_hosts=["previous.invalid:9243"], es_timeout=47)
        previous = config.model_dump()
        with patch.dict('os.environ', {'ES_HOSTS': 'replacement.invalid:9200', 'ES_TIMEOUT': 'invalid'}):
            with pytest.raises(ValueError):
                config._reload_configuration()
        assert config.model_dump() == previous


class TestHybridSearchServiceErrorHandling:
    """Revalidate corrupted model instances at the service boundary."""

    @staticmethod
    def invalid_config():
        config = HybridSearchConfig()
        config.elasticsearch.hosts = []
        config.elasticsearch.timeout = -1
        config.escalation_threshold = 1.5
        config.vector_search.vector_dimension = 0
        config.max_concurrent_requests = -1
        config.request_timeout_ms = 0
        config.embedding_cache_size = -1
        config.embedding_cache_ttl_seconds = 0
        return config

    def test_validate_configuration_error_handling(self):
        service = HybridSearchService()
        with pytest.raises(ValueError) as error:
            service._validate_configuration(self.invalid_config())
        for name in ('elasticsearch.hosts', 'elasticsearch.timeout', 'escalation_threshold',
                     'vector_dimension', 'request_timeout_ms', 'embedding_cache_size'):
            assert name in str(error.value)

    def test_validate_configuration_host_format_error(self):
        config = HybridSearchConfig()
        config.elasticsearch.hosts = ["localhost"]
        with pytest.raises(ValueError, match="Host must include port or scheme"):
            HybridSearchService()._validate_configuration(config)

    def test_validate_configuration_port_error(self):
        config = HybridSearchConfig()
        config.elasticsearch.hosts = ["localhost:invalid"]
        with pytest.raises(ValueError, match="Invalid port number"):
            HybridSearchService()._validate_configuration(config)

    def test_validate_configuration_threshold_error(self):
        # Canonical thresholds have explicit [0, 1] bounds, not the legacy
        # SearchConfig's additional heuristic minimums.
        config = HybridSearchConfig()
        config.escalation_threshold = 1.4
        config.fallback_threshold = -0.1
        config.vector_search.similarity_threshold = 1.2
        with pytest.raises(ValueError) as error:
            HybridSearchService()._validate_configuration(config)
        for field in ('escalation_threshold', 'fallback_threshold', 'similarity_threshold'):
            assert field in str(error.value)

    def test_validate_configuration_cache_error(self):
        config = HybridSearchConfig()
        config.embedding_cache_size = 50
        with pytest.raises(ValueError, match="embedding_cache_size"):
            HybridSearchService()._validate_configuration(config)

    @pytest.mark.asyncio
    async def test_update_configuration_error_handling(self):
        service = HybridSearchService()
        with pytest.raises(ValueError):
            await service.update_configuration(self.invalid_config())

    @pytest.mark.asyncio
    async def test_update_configuration_rollback(self):
        service = HybridSearchService()
        original = service.config
        original_dump = original.model_dump()
        with pytest.raises(ValueError):
            await service.update_configuration(self.invalid_config())
        assert service.config is original
        assert service.config.model_dump() == original_dump

    @pytest.mark.asyncio
    async def test_update_configuration_partial_failure(self):
        service = HybridSearchService()
        original = service.config
        existing_client = MagicMock()
        with patch.object(service, '_client_factory', existing_client):
            with pytest.raises(RuntimeError, match="recreating the search service"):
                await service.update_configuration(HybridSearchConfig(
                    elasticsearch={"hosts": ["replacement.invalid:9200"]}))
            assert service._client_factory is existing_client
            assert service.config is original
