"""Elasticsearch client factory using AsyncElasticsearch."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch

from ...utils.logging_config import get_logger
from .config import HybridSearchConfig, ElasticsearchConfig


def build_client_kwargs(config: ElasticsearchConfig) -> Dict[str, Any]:
    """One connection contract for search, administration and ingestion."""
    kwargs = {
        "hosts": config.normalized_hosts(),
        "request_timeout": config.timeout,
        "max_retries": config.max_retries,
        "retry_on_timeout": config.retry_on_timeout,
        "verify_certs": config.verify_certs,
        "connections_per_node": 50,
        "http_compress": True,
        "sniff_on_start": False,
        "sniff_on_node_failure": False,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    elif config.username and config.password:
        kwargs["basic_auth"] = (config.username, config.password)
    if config.ca_certs:
        kwargs["ca_certs"] = config.ca_certs
    return kwargs


class ElasticsearchClientFactory:
    """Factory that provides configured AsyncElasticsearch clients."""

    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = HybridSearchConfig.validated_copy(config)
        self.es_config: ElasticsearchConfig = self.config.elasticsearch
        self._clients: Dict[str, AsyncElasticsearch] = {}
        self._lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    def _build_es_config(self) -> Dict[str, Any]:
        """Build Elasticsearch client configuration with connection pooling."""
        return build_client_kwargs(self.es_config)

    def validate_connection_configuration(self, config: HybridSearchConfig) -> None:
        if build_client_kwargs(config.elasticsearch) != self._build_es_config():
            raise ValueError("Elasticsearch adapter and client connection settings differ")

    def get_hosts(self) -> List[str]:
        """Return normalized host list used by the factory."""
        return self.es_config.normalized_hosts()

    async def _create_client(self, host: Optional[str] = None) -> AsyncElasticsearch:
        """Create a new AsyncElasticsearch client for the given host."""
        es_config = self._build_es_config()
        if host is not None:
            es_config["hosts"] = [host]
        
        return AsyncElasticsearch(**es_config)

    async def get_client(self, host: Optional[str] = None) -> AsyncElasticsearch:
        """Return cached client for host, creating it on first use."""
        if host is not None and host not in self.get_hosts():
            raise ValueError("Explicit Elasticsearch host is not in the configured cluster")
        cache_key = host if host is not None else "cluster"

        async with self._lock:
            if self._closed:
                raise RuntimeError("Elasticsearch client factory is closed")
            client = self._clients.get(cache_key)
            if client is None:
                client = await self._create_client(host)
                self._clients[cache_key] = client
        return client

    async def close(self) -> None:
        """Close every managed client and reject future connection creation."""
        async with self._close_lock:
            async with self._lock:
                if self._closed:
                    return
                self._closed = True
                clients = list(self._clients.values())
                self._clients.clear()
            failed = False
            for client in clients:
                try:
                    await client.close()
                except Exception:
                    failed = True
                    self.logger.warning("Elasticsearch client cleanup failed")
            if failed:
                raise RuntimeError("Elasticsearch client cleanup failed")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check against configured hosts."""
        async def check_host(host):
            host_result: Dict[str, Any] = {"host": host, "status": "unhealthy"}
            start = datetime.now()
            try:
                timeout = self.es_config.smoke_test_timeout
                async with asyncio.timeout(timeout):
                    client = await self.get_client(host)
                    response = await client.options(request_timeout=timeout).cluster.health(
                        timeout=f"{math.ceil(timeout * 1000)}ms"
                    )
                if response.get("status") in ["green", "yellow"]:
                    host_result["status"] = "healthy"
                    host_result["details"] = response
                else:
                    host_result["error"] = "Cluster health is not green or yellow"
            except Exception as exc:
                host_result["error"] = "Elasticsearch health check failed"
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and 100 <= status <= 599:
                    host_result["status_code"] = status
            host_result["elapsed_ms"] = (datetime.now() - start).total_seconds() * 1000
            return host_result

        results = await asyncio.gather(*(check_host(host) for host in self.get_hosts()))
        return {
            "status": "healthy" if any(result["status"] == "healthy" for result in results) else "unhealthy",
            "hosts": results,
            "timestamp": datetime.now().isoformat(),
        }

    async def smoke_test(self) -> bool:
        """Run a lightweight connectivity check against Elasticsearch."""
        health = await self.health_check()
        return health.get("status") == "healthy"
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = {
            "total_clients": len(self._clients),
            "clients": {}
        }
        
        for host, client in self._clients.items():
            client_stats = {
                "host": host,
                "connected": True
            }
            
            try:
                # Try to get connection pool info
                if hasattr(client, 'transport') and hasattr(client.transport, 'connection_pool'):
                    pool = client.transport.connection_pool
                    client_stats.update({
                        "pool_size": len(pool.connections) if hasattr(pool, 'connections') else 0,
                        "maxsize": getattr(pool, 'maxsize', 25),
                        "dead_count": len(pool.dead) if hasattr(pool, 'dead') else 0,
                        "live_count": len(pool.live) if hasattr(pool, 'live') else 0
                    })
            except Exception as e:
                client_stats["error"] = str(e)
                client_stats["connected"] = False
            
            stats["clients"][host] = client_stats
        
        return stats
