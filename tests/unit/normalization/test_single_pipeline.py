"""One production/direct service; removed rollout settings cannot be accepted."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import ai_service.main as main
from ai_service.api.models import FlagOverrides
from ai_service.core.orchestrator_factory import OrchestratorFactory
from ai_service.layers.normalization.normalization_service import NormalizationService
from ai_service.utils.feature_flags import FeatureFlagManager, FeatureFlags


REMOVED_SETTINGS = [
    ("normalization_implementation", "legacy"),
    ("factory_rollout_percentage", 100),
    ("use_factory_normalizer", True),
    ("language_overrides", {"ru": "factory"}),
    ("enable_dual_processing", False),
    ("enable_performance_fallback", True),
    ("max_latency_threshold_ms", 100),
    ("enable_accuracy_monitoring", True),
    ("min_confidence_threshold", 0.8),
    ("log_implementation_choice", True),
]


@pytest.fixture
async def runtime():
    return await OrchestratorFactory.create_orchestrator(enable_smart_filter=False)


@pytest.mark.parametrize("text,language", [
    ("Иван Петров", "ru"),
    ("Іван Коваленко", "uk"),
    ("John Fitzgerald Kennedy", "en"),
    ("J. Smith", "en"),
])
async def test_direct_and_orchestrated_normalization_share_the_contract(runtime, text, language):
    assert type(runtime.normalization_service) is NormalizationService
    flags = FeatureFlags(enable_spacy_ner=False, enable_spacy_uk_ner=False,
                         enable_spacy_en_ner=False)
    direct = await NormalizationService().normalize_async(text, language=language, feature_flags=flags)
    result = await runtime.process(text, language_hint=language, screen=False,
                                   feature_flags=flags, cache_result=False)
    assert direct.success and result.success
    assert result.normalized_text == direct.normalized
    assert result.tokens == direct.tokens


@pytest.mark.parametrize("endpoint", ["/normalize", "/process"])
@pytest.mark.parametrize("name,value", REMOVED_SETTINGS)
async def test_removed_http_options_fail_before_real_normalization(runtime, monkeypatch, endpoint, name, value):
    monkeypatch.setattr(main, "orchestrator", runtime)
    before = runtime.normalization_service.get_statistics()["total_requests"]
    client = TestClient(main.app)
    response = client.post(endpoint, json={
        "text": "John Smith", "options": {"flags": {name: value}},
    })
    assert response.status_code == 422
    assert runtime.normalization_service.get_statistics()["total_requests"] == before


@pytest.mark.parametrize("name,value", REMOVED_SETTINGS)
def test_removed_library_options_are_rejected(name, value):
    with pytest.raises(TypeError):
        FeatureFlags(**{name: value})
    with pytest.raises(ValidationError):
        FlagOverrides(**{name: value})
    with pytest.raises(ValueError, match="removed"):
        FeatureFlagManager().get_flags({name: value})


@pytest.mark.parametrize("name,value", REMOVED_SETTINGS)
@pytest.mark.parametrize("prefix", ["", "AISVC_FLAG_"])
def test_removed_environment_options_cannot_silently_start(name, value, prefix, monkeypatch):
    monkeypatch.setenv(prefix + name.upper(), str(value))
    with pytest.raises(ValueError):
        FeatureFlagManager()


def test_removed_yaml_options_cannot_silently_start(tmp_path, monkeypatch):
    path = tmp_path / "flags.yaml"
    path.write_text("development:\n  feature_flags:\n    normalization_implementation: factory\n")
    monkeypatch.setenv("AISVC_FEATURE_FLAGS_FILE", str(path))
    with pytest.raises(ValueError):
        FeatureFlagManager()


@pytest.mark.parametrize("language", ["RU", "UK", "EN"])
def test_removed_language_environment_override_is_rejected(language, monkeypatch):
    monkeypatch.setenv("NORMALIZATION_IMPLEMENTATION_" + language, "factory")
    with pytest.raises(ValueError, match="removed"):
        FeatureFlagManager()


async def test_removed_direct_policy_fails_before_processing():
    service = NormalizationService()
    with pytest.raises(ValueError, match="removed"):
        await service.normalize_async("John Smith", feature_flags={"normalization_implementation": "legacy"})
    assert service.get_statistics()["total_requests"] == 0


@pytest.mark.parametrize("fails", [False, True])
async def test_canonical_service_retains_runtime_readiness_and_initialization_errors(fails):
    service = NormalizationService()
    initialize = AsyncMock(side_effect=RuntimeError("unavailable model") if fails else None)
    service.normalization_factory.ner_gateway = SimpleNamespace(
        initialize_runtime=initialize,
        runtime_health_check=lambda: {"status": "unhealthy" if fails else "healthy"},
    )
    if fails:
        with pytest.raises(RuntimeError, match="unavailable model"):
            await service.initialize_runtime()
    else:
        await service.initialize_runtime()
    initialize.assert_awaited_once()
    assert service.runtime_health_check()["status"] == ("unhealthy" if fails else "healthy")


async def test_http_nickname_policy_changes_real_output_and_cache(runtime, monkeypatch):
    monkeypatch.setattr(main, "orchestrator", runtime)
    outputs = []
    # Disable only the optional NER/fastpath to isolate the supported nickname policy.
    common = {"enable_spacy_ner": False, "enable_spacy_uk_ner": False,
              "enable_spacy_en_ner": False, "enable_ascii_fastpath": False}
    client = TestClient(main.app)
    for expand in (True, False, True, False):
        response = client.post("/normalize", json={
            "text": "Bill Gates", "language": "en",
            "options": {"flags": {**common, "enable_en_nicknames": expand,
                                   "enable_en_nickname_expansion": expand}},
        })
        assert response.status_code == 200, response.text
        outputs.append(response.json()["normalized_text"])
    assert outputs == ["William Gates", "Bill Gates", "William Gates", "Bill Gates"]
