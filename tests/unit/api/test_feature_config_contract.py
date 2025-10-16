"""HTTP flags reach processing intact; invalid policy never starts processing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main
from ai_service.contracts.base_contracts import UnifiedProcessingResult
from ai_service.utils.feature_flags import FeatureFlagManager


@pytest.fixture
def flag_api(monkeypatch):
    manager = FeatureFlagManager()
    manager.update_flags(
        strict_stopwords=True, debug_tracing=True
    )
    result = UnifiedProcessingResult(
        original_text="Synthetic Example",
        normalized_text="Synthetic Example",
        language="en",
        language_confidence=1,
        tokens=["Synthetic", "Example"],
        trace=[],
        success=True,
    )
    service = SimpleNamespace(
        process=AsyncMock(return_value=result), enable_search=False
    )
    monkeypatch.setattr(main, "orchestrator", service)
    monkeypatch.setattr(main, "get_feature_flag_manager", lambda: manager)
    return TestClient(main.app), service, manager


@pytest.mark.parametrize("endpoint", ["/normalize", "/process"])
@pytest.mark.parametrize(
    "options", [None, {"flags": {}}, {"flags": {"debug_tracing": False}}]
)
def test_http_preserves_global_policy_and_applies_only_explicit_flags(
    flag_api, endpoint, options
):
    client, service, manager = flag_api
    payload = {"text": "Synthetic Example"}
    if options is not None:
        payload["options"] = options
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    flags = service.process.await_args.kwargs["feature_flags"]
    assert flags.strict_stopwords is True
    assert flags.debug_tracing is (options != {"flags": {"debug_tracing": False}})
    flags.strict_stopwords = False
    assert manager.get_flags().strict_stopwords is True


@pytest.mark.parametrize("endpoint", ["/normalize", "/process"])
@pytest.mark.parametrize(
    "flags",
    [
        {"strict_stopwords": "false"},
        {"strict_stopwords": None},
        {"unknown_flag": True},
        {"factory_rollout_percentage": -1},
        {"max_latency_threshold_ms": 0},
        {"min_confidence_threshold": 1.1},
        {"language_overrides": {"unsupported": "legacy"}},
        {"ascii_fastpath": False, "enable_ascii_fastpath": True},
    ],
)
def test_invalid_http_flags_rejected_before_processing(flag_api, endpoint, flags):
    client, service, _ = flag_api
    response = client.post(
        endpoint, json={"text": "Synthetic Example", "options": {"flags": flags}}
    )
    assert response.status_code == 422
    service.process.assert_not_awaited()


def test_processing_options_returns_an_independent_validated_snapshot():
    from ai_service.api.models import ProcessingOptions, FlagOverrides
    from ai_service.utils.feature_flags import FeatureFlags

    base = FeatureFlags(debug_tracing=True)
    plain = ProcessingOptions().get_effective_flags(base)
    partial = ProcessingOptions(
        flags=FlagOverrides(strict_stopwords=True)
    ).get_effective_flags(base)
    plain.debug_tracing = False
    partial.debug_tracing = False
    assert base.debug_tracing is True
    assert partial.strict_stopwords is True
    assert base.strict_stopwords is False
