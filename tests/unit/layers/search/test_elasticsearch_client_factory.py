import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.elasticsearch_client import ElasticsearchClientFactory


def _write_settings(path: Path) -> None:
    data = {
        "search": {
            "elasticsearch": {
                "hosts": ["http://example.com:9200"],
                "verify_certs": False,
                "timeout": 3,
                "smoke_test_timeout": 1,
            }
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@asynccontextmanager
async def _cluster_server(status=200, payload=None, stall=False):
    requests = []
    handlers = set()
    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            requests.append(request.split(b"\r\n", 1)[0].decode())
            if stall:
                await asyncio.Event().wait()
            body = json.dumps(payload if payload is not None else {"status": "green"}).encode()
            headers = (f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nX-Elastic-Product: Elasticsearch\r\n"
                "Connection: close\r\n\r\n").encode()
            writer.write(headers + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)
    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    try:
        port = server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", requests
    finally:
        server.close()
        tasks = list(handlers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.wait_closed()


def test_hybrid_config_from_env(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    _write_settings(settings_path)
    monkeypatch.setenv("AI_SEARCH_SETTINGS_PATH", str(settings_path))
    monkeypatch.setenv("ES_HOSTS", "https://override:443")
    monkeypatch.setenv("ES_TIMEOUT", "15")

    cfg = HybridSearchConfig.from_env()

    assert cfg.elasticsearch.timeout == 15
    assert cfg.elasticsearch.normalized_hosts() == ["https://override:443"]


@pytest.mark.asyncio
async def test_elasticsearch_factory_health_check_success(tmp_path):
    settings_path = tmp_path / "settings.json"
    _write_settings(settings_path)
    async with _cluster_server() as (host, requests):
        config = HybridSearchConfig.from_env(settings_path=settings_path, env={"ES_HOSTS": host})
        factory = ElasticsearchClientFactory(config)
        try:
            health = await factory.health_check()
            assert health["status"] == "healthy"
            assert health["hosts"][0]["status"] == "healthy"
            assert requests == ["GET /_cluster/health?timeout=1000ms HTTP/1.1"]
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_elasticsearch_factory_health_check_failure():
    async with _cluster_server(503, {"error": "private-backend-marker", "status": 503}) as (host, requests):
        factory = ElasticsearchClientFactory(HybridSearchConfig(elasticsearch={
            "hosts": [host], "max_retries": 0, "smoke_test_timeout": 1}))
        try:
            health = await factory.health_check()
            assert health["status"] == "unhealthy"
            assert health["hosts"][0]["status_code"] == 503
            assert "private-backend-marker" not in json.dumps(health)
            assert len(requests) == 1
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_health_check_deadline_bounds_a_stalled_real_http_response():
    async with _cluster_server(stall=True) as (host, requests):
        factory = ElasticsearchClientFactory(HybridSearchConfig(elasticsearch={
            "hosts": [host], "max_retries": 0, "smoke_test_timeout": 0.1}))
        try:
            start = time.monotonic()
            health = await asyncio.wait_for(factory.health_check(), timeout=0.8)
            assert time.monotonic() - start < 0.8
            assert health["status"] == "unhealthy"
            assert requests == ["GET /_cluster/health?timeout=100ms HTTP/1.1"]
        finally:
            await factory.close()


@pytest.mark.asyncio
async def test_default_client_can_retry_another_configured_node(monkeypatch):
    from ai_service.layers.search import elasticsearch_client as module
    real_client = module.AsyncElasticsearch
    monkeypatch.setattr(module, "AsyncElasticsearch",
        lambda **kwargs: real_client(**kwargs, randomize_nodes_in_pool=False))
    async with _cluster_server(503, {"error": "unavailable", "status": 503}) as (failed, failed_requests):
        async with _cluster_server() as (healthy, healthy_requests):
            factory = ElasticsearchClientFactory(HybridSearchConfig(elasticsearch={
                "hosts": [failed, healthy], "max_retries": 1, "timeout": 1}))
            try:
                client = await factory.get_client()
                result = await client.cluster.health(timeout="1s")
                assert result["status"] == "green"
                assert failed_requests and healthy_requests
            finally:
                await factory.close()


@pytest.mark.asyncio
async def test_red_cluster_is_unhealthy_despite_http_success():
    async with _cluster_server(payload={"status": "red"}) as (host, requests):
        factory = ElasticsearchClientFactory(HybridSearchConfig(elasticsearch={"hosts": [host]}))
        try:
            health = await factory.health_check()
            assert health["status"] == "unhealthy"
            assert health["hosts"][0]["status"] == "unhealthy"
        finally:
            await factory.close()
