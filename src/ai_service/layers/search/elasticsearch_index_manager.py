"""
Elasticsearch index management utilities.

Provides index creation, mapping management, and health monitoring for
AC and Vector search indices.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from elasticsearch import AsyncElasticsearch, ApiError as ElasticsearchException

from ...utils.logging_config import get_logger
from .config import HybridSearchConfig


class ElasticsearchIndexManager:
    """Manages Elasticsearch indices for search functionality."""
    
    def __init__(self, config: HybridSearchConfig, client: AsyncElasticsearch):
        self.config = config
        self.client = client
        self.logger = get_logger(__name__)
        
        # Index names
        self.ac_index = config.elasticsearch.ac_index
        self.vector_index = config.elasticsearch.vector_index
        self.ac_patterns_index = self.ac_index
        
    async def create_ac_index(self) -> bool:
        from .index_schema import ensure_index
        await ensure_index(self.client, self.ac_index, self.config)
        return True

    async def create_vector_index(self) -> bool:
        from .index_schema import ensure_index
        await ensure_index(self.client, self.vector_index, self.config, vectors=True)
        return True

    async def create_ac_patterns_index(self) -> bool:
        return await self.create_ac_index()

    async def create_all_indices(self) -> Dict[str, bool]:
        """Create all required indices."""
        results = {}
        
        results["ac_index"] = await self.create_ac_index()
        results["vector_index"] = await self.create_vector_index()
        results["ac_patterns_index"] = await self.create_ac_patterns_index()
        
        return results
    
    async def _index_exists(self, index_name: str) -> bool:
        return bool(await self.client.indices.exists(index=index_name))

    def _get_ac_index_mapping(self):
        from .index_schema import index_mapping
        return index_mapping(self.config)

    def _get_vector_index_mapping(self):
        from .index_schema import index_mapping
        return index_mapping(self.config, vectors=True)

    def _get_ac_patterns_index_mapping(self):
        return self._get_ac_index_mapping()

    async def get_index_health(self) -> Dict[str, Any]:
        """Get health status of all indices."""
        health_info = {
            "timestamp": datetime.now().isoformat(),
            "indices": {}
        }
        
        indices = list(dict.fromkeys([self.ac_index, self.vector_index, self.ac_patterns_index]))
        
        for index_name in indices:
            try:
                if await self._index_exists(index_name):
                    stats = await self.client.indices.stats(index=index_name)
                    health = await self.client.cluster.health(index=index_name)
                    
                    health_info["indices"][index_name] = {
                        "exists": True,
                        "status": health.get("status", "unknown"),
                        "doc_count": stats["indices"][index_name]["total"]["docs"]["count"],
                        "size_in_bytes": stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
                    }
                else:
                    health_info["indices"][index_name] = {
                        "exists": False,
                        "status": "missing"
                    }
            except ElasticsearchException as exc:
                health_info["indices"][index_name] = {
                    "exists": False,
                    "status": "error",
                    "error": str(exc)
                }
        
        return health_info
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete an index (use with caution)."""
        try:
            if await self._index_exists(index_name):
                await self.client.indices.delete(index=index_name)
                self.logger.warning(f"Deleted index: {index_name}")
                return True
            else:
                self.logger.info(f"Index {index_name} does not exist")
                return False
        except ElasticsearchException as exc:
            self.logger.error(f"Failed to delete index {index_name}: {exc}")
            return False
    
    async def refresh_index(self, index_name: str) -> bool:
        """Refresh an index to make recent changes searchable."""
        try:
            if await self._index_exists(index_name):
                await self.client.indices.refresh(index=index_name)
                self.logger.debug(f"Refreshed index: {index_name}")
                return True
            else:
                self.logger.warning(f"Cannot refresh non-existent index: {index_name}")
                return False
        except ElasticsearchException as exc:
            self.logger.error(f"Failed to refresh index {index_name}: {exc}")
            return False
