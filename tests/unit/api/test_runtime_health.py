"""Health must describe executable dependencies, not just a constructed object."""
from types import SimpleNamespace
import asyncio
import secrets
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main


def healthy_runtime():
    def model_service():
        return SimpleNamespace(
            initialize_runtime=AsyncMock(),
            runtime_health_check=Mock(return_value={"status": "healthy"}),
            close=Mock(),
        )
    return SimpleNamespace(
        validation_service=object(), language_service=object(), unicode_service=object(),
        signals_service=object(), normalization_service=model_service(),
        enable_search=True, enable_embeddings=True, enable_variants=True,
        enable_smart_filter=False, enable_decision_engine=False,
        embeddings_service=model_service(), variants_service=model_service(),
        search_service=SimpleNamespace(
            health_check=AsyncMock(return_value={"status": "healthy"}),
            readiness=AsyncMock(return_value={"ac": "generation", "vectors": "generation"}),
            close=AsyncMock(),
        ),
        get_processing_stats=Mock(return_value={"total_processed": 0, "successful": 0}),
    )


@pytest.fixture
def runtime_api(monkeypatch):
    runtime = healthy_runtime()
    monkeypatch.setattr(main, "orchestrator", runtime)
    monkeypatch.setitem(main.app.dependency_overrides, main.verify_admin_token, lambda: "test")
    return TestClient(main.app), runtime


@pytest.mark.parametrize("path", ["/health", "/health/ready", "/health/detailed"])
def test_search_failure_propagates_to_all_health_routes(runtime_api, path):
    client, runtime = runtime_api
    runtime.search_service.health_check.return_value = {"status": "unhealthy"}
    assert client.get(path).status_code == 503
    assert client.get("/health/live").status_code == 200


@pytest.mark.parametrize("component", ["normalization_service", "embeddings_service", "variants_service"])
@pytest.mark.parametrize("path", ["/health", "/health/ready", "/health/detailed"])
def test_model_or_queue_failure_propagates(runtime_api, component, path):
    client, runtime = runtime_api
    getattr(runtime, component).runtime_health_check.return_value = {"status": "unhealthy"}
    assert client.get(path).status_code == 503


def test_unpublished_runtime_is_unavailable(runtime_api, monkeypatch):
    client, _ = runtime_api
    monkeypatch.setattr(main, "orchestrator", None)
    for path in ("/health", "/health/ready", "/health/detailed"):
        assert client.get(path).status_code == 503
    assert client.get("/health/live").status_code == 200


@pytest.mark.parametrize("report", [None, [], {}, {"status": "unknown"},
                                     {"status": "healthy", "connected": False}])
def test_malformed_search_health_fails_closed(runtime_api, report):
    client, runtime = runtime_api
    runtime.search_service.health_check.return_value = report
    assert client.get("/health/ready").status_code == 503


@pytest.mark.parametrize("generations", [None, {}, {"ac": ""}, {"ac": "g"},
                                         {"ac": "g", "vectors": "other"},
                                         {"ac": "g", "vectors": None}])
def test_incomplete_generation_fails_closed(runtime_api, generations):
    client, runtime = runtime_api
    runtime.search_service.readiness.return_value = generations
    assert client.get("/health/ready").status_code == 503


@pytest.mark.parametrize("component", ["search_service", "normalization_service",
                                       "embeddings_service", "variants_service", "signals_service"])
def test_missing_required_provider_fails_closed(runtime_api, component):
    client, runtime = runtime_api
    setattr(runtime, component, None)
    assert client.get("/health/ready").status_code == 503


def test_disabled_optional_providers_are_not_required(runtime_api):
    client, runtime = runtime_api
    runtime.enable_embeddings = runtime.enable_variants = runtime.enable_search = False
    runtime.embeddings_service = runtime.variants_service = runtime.search_service = None
    assert client.get("/health/ready").status_code == 200


def test_healthy_probes_do_not_execute_models(runtime_api):
    client, runtime = runtime_api
    for path in ("/health", "/health/ready", "/health/detailed"):
        assert client.get(path).status_code == 200
    runtime.normalization_service.initialize_runtime.assert_not_awaited()
    runtime.embeddings_service.initialize_runtime.assert_not_awaited()
    assert runtime.search_service.readiness.await_count == 3


def test_detailed_health_serializes_provider_timestamps(runtime_api):
    from datetime import datetime, timezone
    client, runtime = runtime_api
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runtime.search_service.health_check.return_value = {"status": "healthy", "last_check": timestamp}
    response = client.get("/health/detailed")
    assert response.status_code == 200
    assert response.json()["components"]["search_service"]["last_check"] == timestamp.isoformat()


@pytest.mark.parametrize("method", ["readiness", "health_check"])
async def test_backend_probe_has_a_bounded_deadline(monkeypatch, method):
    from ai_service.api import runtime_health
    runtime = healthy_runtime()
    cancelled = asyncio.Event()
    async def stalled(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    getattr(runtime.search_service, method).side_effect = stalled
    monkeypatch.setattr(runtime_health, "HEALTH_TIMEOUT_SECONDS", 0.02)
    result = await asyncio.wait_for(runtime_health.collect_runtime_health(runtime), timeout=0.5)
    assert result["status"] == "unhealthy"
    assert cancelled.is_set()


async def test_probe_cancellation_is_not_swallowed():
    from ai_service.api.runtime_health import collect_runtime_health
    runtime = healthy_runtime()
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    async def stalled(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    runtime.search_service.readiness.side_effect = stalled
    task = asyncio.create_task(collect_runtime_health(runtime))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_worker_closure_during_backend_probe_is_observed():
    from ai_service.api.runtime_health import collect_runtime_health
    runtime = healthy_runtime()
    async def close_worker(**kwargs):
        runtime.embeddings_service.runtime_health_check.return_value = {"status": "unhealthy"}
        return {"ac": "g", "vectors": "g"}
    runtime.search_service.readiness.side_effect = close_worker
    assert (await collect_runtime_health(runtime))["status"] == "unhealthy"


@pytest.mark.parametrize("path", ["/health", "/health/ready", "/health/detailed"])
def test_dependency_exception_is_private(runtime_api, path):
    client, runtime = runtime_api
    secret = secrets.token_hex(24)
    runtime.search_service.readiness.side_effect = RuntimeError(secret)
    response = client.get(path)
    assert response.status_code == 503
    assert secret not in response.text


def test_optional_diagnostics_failure_is_explicit(runtime_api, monkeypatch):
    client, _ = runtime_api
    monkeypatch.setattr("ai_service.utils.http_client_pool.get_http_pool", Mock(side_effect=RuntimeError()))
    assert client.get("/health/ready").status_code == 200
    response = client.get("/health/detailed")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


@pytest.fixture
def lifecycle_runtime(monkeypatch):
    runtime = healthy_runtime()
    runtime.enable_search = False
    monkeypatch.setattr(main, "orchestrator", None)
    monkeypatch.setattr(main.OrchestratorFactory, "create_production_orchestrator", AsyncMock(return_value=runtime))
    monkeypatch.setattr("ai_service.layers.normalization.ner_gateways.close_global_gateway", Mock())
    monkeypatch.setattr("ai_service.utils.async_model_loader._model_loader.close", Mock())
    return runtime


async def test_runtime_is_published_only_after_model_verification(lifecycle_runtime):
    runtime = lifecycle_runtime
    entered, release = asyncio.Event(), asyncio.Event()
    async def initialize():
        entered.set()
        await release.wait()
    runtime.normalization_service.initialize_runtime.side_effect = initialize
    task = asyncio.create_task(main.startup_event())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert main.orchestrator is None
        runtime.embeddings_service.initialize_runtime.assert_awaited_once()
    finally:
        release.set()
        await task
    assert main.orchestrator is runtime


@pytest.mark.parametrize("component", ["normalization_service", "embeddings_service"])
async def test_failed_warmup_does_not_publish_or_leak_workers(lifecycle_runtime, component):
    runtime = lifecycle_runtime
    getattr(runtime, component).initialize_runtime.side_effect = RuntimeError("probe failed")
    runtime.search_service.close.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="probe failed"):
        await main.startup_event()
    assert main.orchestrator is None
    runtime.search_service.close.assert_awaited_once()
    runtime.embeddings_service.close.assert_called_once()
    runtime.variants_service.close.assert_called_once()


async def test_shutdown_unpublishes_before_closing_and_closes_every_worker(lifecycle_runtime):
    runtime = lifecycle_runtime
    main.orchestrator = runtime
    async def fail_close():
        assert main.orchestrator is None
        raise RuntimeError("controlled shutdown failure")
    runtime.search_service.close.side_effect = fail_close
    await main.shutdown_event()
    runtime.embeddings_service.close.assert_called_once()
    runtime.variants_service.close.assert_called_once()
