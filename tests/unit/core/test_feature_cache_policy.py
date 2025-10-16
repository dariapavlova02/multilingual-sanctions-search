"""Exercise real orchestrator cache reads/writes across distinct flag policies."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.contracts.base_contracts import NormalizationResult, SignalsResult
from ai_service.core.cache_service import CacheService
from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.utils.feature_flags import FeatureFlags


@pytest.fixture
def runtime():
    async def normalize(text, feature_flags, **kwargs):
        # This layer double makes its effective input visible. The cache and
        # orchestrator are real; this is not a normalization quality assertion.
        output = (
            "Dictionary Example"
            if feature_flags.use_diminutives_dictionary_only
            else "Expanded Example"
        )
        return NormalizationResult(
            normalized=output, tokens=output.split(), trace=[], success=True
        )

    normalizer = SimpleNamespace(normalize_async=AsyncMock(side_effect=normalize))
    service = UnifiedOrchestrator(
        validation_service=SimpleNamespace(
            validate_and_sanitize=AsyncMock(
                return_value={"is_valid": True, "sanitized_text": "Synthetic Example"}
            )
        ),
        language_service=SimpleNamespace(
            detect_language_config_driven=lambda *a, **k: SimpleNamespace(
                language="en", confidence=1.0
            )
        ),
        unicode_service=SimpleNamespace(normalize_unicode=lambda text: text),
        normalization_service=normalizer,
        signals_service=SimpleNamespace(
            extract_signals=AsyncMock(return_value=SignalsResult())
        ),
        enable_search=False,
        enable_smart_filter=False,
        enable_variants=False,
        enable_embeddings=False,
        enable_decision_engine=False,
    )
    service.cache_service = CacheService()
    return service, normalizer


async def test_changed_policy_reprocesses_while_repeated_policy_uses_cache(runtime):
    service, normalizer = runtime
    results = []
    for dictionary_only in (False, True, False, True):
        result = await service.process(
            "Synthetic Example",
            screen=False,
            cache_result=True,
            feature_flags=FeatureFlags(use_diminutives_dictionary_only=dictionary_only),
        )
        assert result.success, result.errors
        results.append(result.normalized_text)
    assert results == ["Expanded Example", "Dictionary Example"] * 2
    assert normalizer.normalize_async.await_count == 2


def test_mutated_flag_object_cannot_enter_processing(runtime):
    service, _ = runtime
    flags = FeatureFlags()
    flags.require_tin_dob_gate = "false"
    with pytest.raises(ValueError):
        service._validate_and_normalize_flags(flags)
