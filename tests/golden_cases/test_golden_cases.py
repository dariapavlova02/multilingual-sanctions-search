"""Golden canary tests for normalization layer."""

import json
from pathlib import Path

import pytest

from ai_service.layers.normalization.normalization_service import NormalizationService

GOLDEN_CASES_PATH = Path(__file__).with_name("golden_cases.json")
GOLDEN_CASES = json.loads(GOLDEN_CASES_PATH.read_text())



def _expected_normalized(case: dict) -> str:
    personas = case.get("expected_personas", [])
    if not personas:
        return ""
    return " | ".join(persona["normalized"] for persona in personas)


@pytest.fixture(scope="module")
def normalization_service() -> NormalizationService:
    return NormalizationService()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case["id"] for case in GOLDEN_CASES])
def test_golden_cases_sync(normalization_service: NormalizationService, case: dict) -> None:
    """Check synchronous normalization against golden expectations."""
    expected = _expected_normalized(case)

    result = normalization_service.normalize(
        case["input"],
        language=case["language"],
        remove_stop_words=True,
        preserve_names=True,
        enable_advanced_features=True,
    )

    assert result.normalized == expected, (
        f"Mismatch for {case['id']}: expected '{expected}', got '{result.normalized}'"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case["id"] for case in GOLDEN_CASES])
async def test_golden_cases_async(case: dict) -> None:
    """Check asynchronous normalization against the same golden expectations."""
    service = NormalizationService()
    expected = _expected_normalized(case)

    result = await service.normalize_async(
        case["input"],
        language=case["language"],
        remove_stop_words=True,
        preserve_names=True,
        enable_advanced_features=True,
    )

    assert result.normalized == expected, (
        f"Factory mismatch for {case['id']}: expected '{expected}', got '{result.normalized}'"
    )
