"""
Simple Elasticsearch wrapper for admin endpoints.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.exceptions import RequestError, NotFoundError
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    AsyncElasticsearch = None
    RequestError = Exception
    NotFoundError = Exception
    ELASTICSEARCH_AVAILABLE = False

# Configuration will be read from environment variables

class SimpleElasticsearchClient:
    """Simple Elasticsearch client wrapper for admin operations."""

    def __init__(self):
        self.client = None
        self._closed = False

        if ELASTICSEARCH_AVAILABLE:
            from ..layers.search.config import HybridSearchConfig
            from ..layers.search.elasticsearch_client import build_client_kwargs

            self.config = HybridSearchConfig.from_env()
            self.client = AsyncElasticsearch(**build_client_kwargs(self.config.elasticsearch))

        else:
            raise ImportError("elasticsearch package not available")

    async def close(self):
        """Close the client connection."""
        if self.client and not self._closed:
            await self.client.close()
            self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def health_check(self) -> bool:
        """Check if Elasticsearch is healthy."""
        if not self.client:
            return False

        try:
            health = await self.client.cluster.health()
            return health.get("status") in ["green", "yellow"]
        except Exception:
            return False

# Alias for compatibility
ElasticsearchClient = SimpleElasticsearchClient