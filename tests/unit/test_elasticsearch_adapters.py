"""
Unit tests for Elasticsearch adapters.

Tests the AC and Vector adapters for Elasticsearch integration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from ai_service.layers.search.elasticsearch_adapters import (
    ElasticsearchACAdapter,
    ElasticsearchVectorAdapter,
)
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import SearchOpts, SearchMode, Candidate
from elasticsearch.exceptions import ApiError, ConnectionError, ConnectionTimeout
from elastic_transport import ApiResponseMeta, NodeConfig


def api_error():
    return ApiError(
        "Search failed",
        ApiResponseMeta(400, "1.1", {}, 0, NodeConfig("http", "localhost", 9200)),
        {},
    )


class TestElasticsearchACAdapter:
    """Test cases for ElasticsearchACAdapter"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return HybridSearchConfig(
            elasticsearch={
                "hosts": ["localhost:9200"],
                "timeout": 10,
                "max_retries": 3,
                "retry_on_timeout": True,
                "verify_certs": False,
                "ac_index": "test_ac",
                "vector_index": "test_vector",
                "smoke_test_timeout": 5,
            }
        )

    @pytest.fixture
    def mock_client(self):
        """Create a mock Elasticsearch client"""
        client = AsyncMock()
        client.indices.exists.return_value = False
        client.indices.stats.return_value = {}
        client.cluster.health.return_value = {"status": "green"}
        client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.95,
                        "_source": {
                            "normalized_text": "Test Name",
                            "aliases": ["Test Name Variant"],
                            "legal_names": ["Legal Test Name"],
                            "entity_type": "person",
                        },
                    },
                    {
                        "_id": "2",
                        "_score": 0.85,
                        "_source": {
                            "normalized_text": "Another Test Name",
                            "aliases": [],
                            "legal_names": [],
                            "entity_type": "person",
                        },
                    },
                ],
            }
        }
        return client

    @pytest.fixture
    def mock_client_factory(self, mock_client):
        """Create a mock client factory"""
        factory = AsyncMock()
        factory.get_client.return_value = mock_client
        factory.health_check.return_value = {"status": "healthy"}
        return factory

    @pytest.fixture
    async def ac_adapter(self, mock_config, mock_client_factory):
        """Create an AC adapter instance for testing"""
        adapter = ElasticsearchACAdapter(
            mock_config, client_factory=mock_client_factory
        )
        await adapter._ensure_connection()
        return adapter

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_config, mock_client_factory):
        """Test successful adapter initialization"""
        adapter = ElasticsearchACAdapter(
            mock_config, client_factory=mock_client_factory
        )
        await adapter._ensure_connection()

        assert adapter._connected is True
        assert adapter._client is not None
        assert adapter._index_manager is not None

    @pytest.mark.asyncio
    async def test_search_ac_patterns(self, ac_adapter, mock_client):
        """Test AC patterns search"""
        opts = SearchOpts(top_k=10, threshold=0.7)
        result = await ac_adapter.search_ac_patterns("test query", opts)

        assert len(result) == 2
        assert result[0]["score"] == 0.95
        assert result[0]["tier"] == 0
        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_candidates(self, ac_adapter, mock_client):
        """Test search for candidates"""
        opts = SearchOpts(top_k=10, threshold=0.7)
        result = await ac_adapter.search("Test Name", opts)

        assert len(result) == 2
        assert isinstance(result[0], Candidate)
        assert result[0].doc_id == "1"
        assert result[0].score == pytest.approx(1.0)
        assert result[0].text == "Test Name"
        assert result[0].entity_type == "person"

    @pytest.mark.asyncio
    async def test_search_with_retry(self, ac_adapter, mock_client):
        """Test search with retry logic"""
        # Mock first call to fail, second to succeed
        mock_client.search.side_effect = [
            ConnectionError("Connection failed"),
            {
                "hits": {
                    "total": {"value": 1},
                    "hits": [
                        {
                            "_id": "1",
                            "_score": 0.95,
                            "_source": {
                                "normalized_text": "Test Name",
                                "aliases": [],
                                "legal_names": [],
                                "entity_type": "person",
                            },
                        }
                    ],
                }
            },
        ]

        opts = SearchOpts(top_k=10, threshold=0.7)
        result = await ac_adapter.search("Test Name", opts)

        assert len(result) == 1
        assert result[0].doc_id == "1"
        assert mock_client.search.call_count == 2

    @pytest.mark.asyncio
    async def test_search_error_handling(self, ac_adapter, mock_client):
        """Test error handling in search operations"""
        mock_client.search.side_effect = api_error()

        opts = SearchOpts(top_k=10, threshold=0.7)
        with pytest.raises(ApiError):
            await ac_adapter.search("Test Name", opts)

    @pytest.mark.asyncio
    async def test_health_check(self, ac_adapter, mock_client):
        """Test health check functionality"""
        health = await ac_adapter.health_check()

        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert "indices" in health

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, ac_adapter, mock_client):
        """Test connection error handling"""
        mock_client.search.side_effect = ConnectionError("Connection failed")

        opts = SearchOpts(top_k=10, threshold=0.7)
        with pytest.raises(ConnectionError):
            await ac_adapter.search("Test Name", opts)
        assert ac_adapter._connected is False

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, ac_adapter, mock_client):
        """Test timeout error handling"""
        mock_client.search.side_effect = ConnectionTimeout("Request timeout")

        opts = SearchOpts(top_k=10, threshold=0.7)
        with pytest.raises(ConnectionTimeout):
            await ac_adapter.search("Test Name", opts)
        assert ac_adapter._connected is False


class TestElasticsearchVectorAdapter:
    """Test cases for ElasticsearchVectorAdapter"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return HybridSearchConfig(
            elasticsearch={
                "hosts": ["localhost:9200"],
                "timeout": 10,
                "max_retries": 3,
                "retry_on_timeout": True,
                "verify_certs": False,
                "ac_index": "test_ac",
                "vector_index": "test_vector",
                "smoke_test_timeout": 5,
            }
        )

    @pytest.fixture
    def mock_client(self):
        """Create a mock Elasticsearch client"""
        client = AsyncMock()
        client.indices.exists.return_value = False
        client.indices.stats.return_value = {}
        client.cluster.health.return_value = {"status": "green"}
        client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.95,
                        "_source": {
                            "normalized_text": "Test Name",
                            "entity_type": "person",
                            "metadata": {"dob": "1990-01-01"},
                        },
                    },
                    {
                        "_id": "2",
                        "_score": 0.85,
                        "_source": {
                            "normalized_text": "Another Test Name",
                            "entity_type": "person",
                            "metadata": {"dob": "1990-01-01"},
                        },
                    },
                ],
            }
        }
        return client

    @pytest.fixture
    def mock_client_factory(self, mock_client):
        """Create a mock client factory"""
        factory = AsyncMock()
        factory.get_client.return_value = mock_client
        factory.health_check.return_value = {"status": "healthy"}
        return factory

    @pytest.fixture
    async def vector_adapter(self, mock_config, mock_client_factory):
        """Create a vector adapter instance for testing"""
        adapter = ElasticsearchVectorAdapter(
            mock_config, client_factory=mock_client_factory
        )
        await adapter._ensure_connection()
        return adapter

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_config, mock_client_factory):
        """Test successful adapter initialization"""
        adapter = ElasticsearchVectorAdapter(
            mock_config, client_factory=mock_client_factory
        )
        await adapter._ensure_connection()

        assert adapter._connected is True
        assert adapter._client is not None
        assert adapter._index_manager is not None

    @pytest.mark.asyncio
    async def test_search_vector_fallback(self, vector_adapter, mock_client):
        """Test vector search fallback"""
        query_vector = [0.1] * 384
        opts = SearchOpts(top_k=10, threshold=0.7)
        result = await vector_adapter.search_vector_fallback(
            query_vector, "test query", opts
        )

        assert len(result) == 2
        assert isinstance(result[0], Candidate)
        assert result[0].doc_id == "1"
        assert result[0].score == pytest.approx(0.9)
        assert result[0].search_mode == SearchMode.VECTOR

    @pytest.mark.asyncio
    async def test_search_candidates(self, vector_adapter, mock_client):
        """Test search for candidates"""
        query_vector = [0.1] * 384
        opts = SearchOpts(top_k=10, threshold=0.7)
        result = await vector_adapter.search(query_vector, opts)

        assert len(result) == 2
        assert isinstance(result[0], Candidate)
        assert result[0].doc_id == "1"
        assert result[0].score == pytest.approx(0.9)
        assert result[0].text == "Test Name"
        assert result[0].entity_type == "person"

    @pytest.mark.asyncio
    async def test_search_with_retry(self, vector_adapter, mock_client):
        """Test search with retry logic"""
        # Mock first call to fail, second to succeed
        mock_client.search.side_effect = [
            ConnectionError("Connection failed"),
            {
                "hits": {
                    "total": {"value": 1},
                    "hits": [
                        {
                            "_id": "1",
                            "_score": 0.95,
                            "_source": {
                                "normalized_text": "Test Name",
                                "entity_type": "person",
                                "metadata": {"dob": "1990-01-01"},
                            },
                        }
                    ],
                }
            },
        ]

        query_vector = [0.1] * 384
        opts = SearchOpts(top_k=10, threshold=0.7)
        result = await vector_adapter.search(query_vector, opts)

        assert len(result) == 1
        assert result[0].doc_id == "1"
        assert mock_client.search.call_count == 2

    @pytest.mark.asyncio
    async def test_search_error_handling(self, vector_adapter, mock_client):
        """Test error handling in search operations"""
        mock_client.search.side_effect = api_error()

        query_vector = [0.1] * 384
        opts = SearchOpts(top_k=10, threshold=0.7)
        with pytest.raises(ApiError):
            await vector_adapter.search(query_vector, opts)

    @pytest.mark.asyncio
    async def test_health_check(self, vector_adapter, mock_client):
        """Test health check functionality"""
        health = await vector_adapter.health_check()

        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert "indices" in health

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, vector_adapter, mock_client):
        """Test connection error handling"""
        mock_client.search.side_effect = ConnectionError("Connection failed")

        query_vector = [0.1] * 384
        opts = SearchOpts(top_k=10, threshold=0.7)
        with pytest.raises(ConnectionError):
            await vector_adapter.search(query_vector, opts)
        assert vector_adapter._connected is False

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, vector_adapter, mock_client):
        """Test timeout error handling"""
        mock_client.search.side_effect = ConnectionTimeout("Request timeout")

        query_vector = [0.1] * 384
        opts = SearchOpts(top_k=10, threshold=0.7)
        with pytest.raises(ConnectionTimeout):
            await vector_adapter.search(query_vector, opts)
        assert vector_adapter._connected is False


class TestElasticsearchIndexManager:
    """Test cases for ElasticsearchIndexManager"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return HybridSearchConfig(
            elasticsearch={
                "hosts": ["localhost:9200"],
                "timeout": 10,
                "max_retries": 3,
                "retry_on_timeout": True,
                "verify_certs": False,
                "ac_index": "test_ac",
                "vector_index": "test_vector",
                "smoke_test_timeout": 5,
            }
        )

    @pytest.fixture
    def mock_client(self):
        """Create a mock Elasticsearch client"""
        client = AsyncMock()
        client.indices.create.return_value = {"acknowledged": True}
        client.indices.exists.return_value = False
        client.cluster.health.return_value = {"status": "green"}
        return client

    @pytest.fixture
    def index_manager(self, mock_config, mock_client):
        """Create an index manager instance for testing"""
        from ai_service.layers.search.elasticsearch_index_manager import (
            ElasticsearchIndexManager,
        )

        return ElasticsearchIndexManager(mock_config, mock_client)

    @pytest.mark.asyncio
    async def test_create_ac_patterns_index(self, index_manager, mock_client):
        """Test AC patterns index creation"""
        result = await index_manager.create_ac_patterns_index()

        assert result is True
        mock_client.indices.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_vector_index(self, index_manager, mock_client):
        """Test vector index creation"""
        result = await index_manager.create_vector_index()

        assert result is True
        mock_client.indices.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_index_health(self, index_manager, mock_client):
        """Test index health check"""
        health = await index_manager.get_index_health()

        assert "indices" in health
        assert "test_ac" in health["indices"]
        assert "test_vector" in health["indices"]
