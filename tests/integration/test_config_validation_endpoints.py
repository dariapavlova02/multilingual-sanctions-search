"""Authenticated configuration diagnostics against the production search contract."""

import secrets
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.hybrid_search_service import HybridSearchService


@pytest.fixture
def config_api(monkeypatch):
    token = secrets.token_hex(32)
    monkeypatch.setattr(main.SECURITY_CONFIG, "admin_api_key", token)
    service = HybridSearchService(HybridSearchConfig())
    service.readiness = AsyncMock(return_value={"ac": "generation", "vectors": "generation"})
    monkeypatch.setattr(main, "orchestrator", SimpleNamespace(search_service=service))
    return TestClient(main.app), {"Authorization": f"Bearer {token}"}, service


ENDPOINTS = [("POST", "/validate-config"), ("GET", "/config-status"), ("POST", "/reload-config")]


@pytest.mark.parametrize("method,path", ENDPOINTS)
def test_configuration_endpoints_require_authentication(config_api, method, path):
    client, _, service = config_api
    response = client.request(method, path)
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authenticated"
    service.readiness.assert_not_awaited()


@pytest.mark.parametrize("method,path", ENDPOINTS)
def test_configuration_endpoints_reject_invalid_token(config_api, method, path):
    client, _, service = config_api
    response = client.request(method, path, headers={"Authorization": "Bearer invalid-token"})
    assert response.status_code == 401
    assert "Invalid API key" in response.text
    service.readiness.assert_not_awaited()


@pytest.mark.parametrize("method,path", ENDPOINTS)
def test_configuration_endpoints_require_orchestrator(config_api, monkeypatch, method, path):
    client, headers, _ = config_api
    monkeypatch.setattr(main, "orchestrator", None)
    response = client.request(method, path, headers=headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Orchestrator not initialized"


def test_validate_configuration_and_actual_index_readiness(config_api):
    client, headers, service = config_api
    response = client.post("/validate-config", headers=headers)
    assert response.status_code == 200
    assert response.json()["search_service"] == {
        "enabled": True, "validation_passed": True, "runtime_ready": True,
        "errors": [], "warnings": [],
    }
    service.readiness.assert_awaited_once_with()


def test_validation_rechecks_mutated_nested_values_without_echoing_secrets(config_api):
    client, headers, service = config_api
    sensitive_value = "https://user:private-password@example.invalid:9243"
    service.config.elasticsearch.hosts = [sensitive_value]
    response = client.post("/validate-config", headers=headers)
    assert response.status_code == 200
    result = response.json()["search_service"]
    assert not result["validation_passed"]
    assert not result["runtime_ready"]
    assert any("elasticsearch.hosts" in error for error in result["errors"])
    assert "private-password" not in response.text
    service.readiness.assert_not_awaited()


def test_valid_settings_do_not_imply_available_sanctions_indices(config_api):
    client, headers, service = config_api
    service.readiness.side_effect = RuntimeError("credential-or-private-index-detail")
    response = client.post("/validate-config", headers=headers)
    result = response.json()["search_service"]
    assert result["validation_passed"]
    assert not result["runtime_ready"]
    assert result["warnings"]
    assert "credential-or-private-index-detail" not in response.text


def test_disabled_search_is_explicit(config_api, monkeypatch):
    client, headers, _ = config_api
    monkeypatch.setattr(main, "orchestrator", SimpleNamespace(search_service=None))
    result = client.post("/validate-config", headers=headers).json()["search_service"]
    assert not result["enabled"]
    assert not result["validation_passed"]
    assert not result["runtime_ready"]
    assert not client.get("/config-status", headers=headers).json()["search_service"]["enabled"]


def test_status_does_not_advertise_an_inactive_config_watcher(config_api):
    client, headers, _ = config_api
    result = client.get("/config-status", headers=headers).json()["search_service"]
    assert result["enabled"]
    assert result["hot_reload"] is False
    assert result["change_application"] == "restart_required"
    assert result["reload_stats"] == {}


def test_reload_cannot_report_success_without_replacing_runtime_components(config_api):
    client, headers, service = config_api
    previous = service.config.model_dump()
    response = client.post("/reload-config", headers=headers)
    assert response.status_code == 409
    assert "recreate the API service" in response.json()["detail"]
    assert service.config.model_dump() == previous
    service.readiness.assert_not_awaited()
