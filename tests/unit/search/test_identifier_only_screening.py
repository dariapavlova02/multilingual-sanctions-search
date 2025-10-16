"""Identifier-only screening must not require a fabricated name embedding."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.layers.normalization.normalization_service import NormalizationService
from ai_service.layers.search.contracts import Candidate, SearchMode, SearchOpts
from ai_service.layers.signals.signals_service import SignalsService


@pytest.fixture(scope="module")
def normalization():
    return NormalizationService()


def orchestrator(found=True):
    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    hit = Candidate(doc_id="synthetic-source", text="Source Person", entity_type="person",
        score=1.0, confidence=1.0, metadata={"tin": "1234567890"},
        search_mode=SearchMode.AC, match_fields=["inn"])
    service.enable_search = True
    service.metrics_service = None
    service.homoglyph_detector = None
    service.search_service = SimpleNamespace(
        readiness=AsyncMock(return_value="generation-one"),
        _verify_dataset_version=AsyncMock(),
        find_by_identifier=AsyncMock(return_value=[hit] if found else []),
        find_candidates=AsyncMock(side_effect=ValueError("No semantic name vector")))
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("language,text", [
    ("ru", "ИНН 1234567890"), ("uk", "ІПН 1234567890"), ("en", "INN 1234567890"),
    ("ru", "\u200bИНН 12345\u200b67890."),
])
@pytest.mark.parametrize("found", [False, True])
@pytest.mark.parametrize("mode", list(SearchMode))
async def test_real_identifier_extraction_completes_without_name_search(normalization, language, text, found, mode):
    norm = await normalization.normalize_async(text, language=language)
    assert norm.normalized == ""
    signals = await SignalsService().extract_signals(text, norm, language)
    service = orchestrator(found)
    errors = []
    result = await service._handle_search_layer(norm, None, errors, text,
        signals_result=signals, search_options=SearchOpts(search_mode=mode))
    assert errors == [] and result is not None
    assert result["total_hits"] == (1 if found else 0)
    service.search_service.find_candidates.assert_not_awaited()
    service.search_service.find_by_identifier.assert_awaited_once()
    require_vectors = mode not in (SearchMode.AC, SearchMode.FUZZY)
    service.search_service.readiness.assert_awaited_once_with(require_vectors=require_vectors)
    service.search_service._verify_dataset_version.assert_awaited_once_with("generation-one", require_vectors=require_vectors)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["readiness", "lookup", "generation"])
async def test_identifier_only_backend_and_generation_failures_are_not_empty_success(normalization, failure):
    text = "INN 1234567890"
    norm = await normalization.normalize_async(text, language="en")
    signals = await SignalsService().extract_signals(text, norm, "en")
    service = orchestrator()
    if failure == "readiness":
        service.search_service.readiness.side_effect = RuntimeError("Snapshot unavailable")
    elif failure == "lookup":
        service.search_service.find_by_identifier.side_effect = RuntimeError("Backend unavailable")
    else:
        service.search_service._verify_dataset_version.side_effect = RuntimeError("Generation changed")
    errors = []
    result = await service._handle_search_layer(norm, None, errors, text,
        signals_result=signals, search_options=SearchOpts())
    assert result is None and errors
    service.search_service.find_candidates.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("text,span,raw,id_type", [
    ("Other Name INN 1234567890", (11, 25), "INN 1234567890", "inn"),
    ("INN 1234567890", (0, 3), "INN 1234567890", "inn"),
    ("INN 1234567890", (-1, 14), "INN 1234567890", "inn"),
    ("INN 1234567890", (0, 14), "INN 1234567890", "passport_rf"),
    ("Other Name INN 1234567890", (0, 25), "Other Name INN 1234567890", "inn"),
    ("INN 1234567890 42", (0, 14), "INN 1234567890", "inn"),
    ("INN 1234567890 +", (0, 14), "INN 1234567890", "inn"),
    ("INN 001234567890", (0, 16), "INN 001234567890", "inn"),
])
async def test_missing_name_does_not_allow_ignoring_other_text_or_unverified_id_spans(text, span, raw, id_type):
    service = orchestrator()
    signals = SimpleNamespace(persons=[], organizations=[], extras={"unassigned_ids": [
        {"type": id_type, "value": "1234567890", "position": span, "raw": raw}]})
    errors = []
    result = await service._handle_search_layer(SimpleNamespace(normalized=""), None, errors, text,
        signals_result=signals, search_options=SearchOpts())
    assert result is None and errors
    service.search_service.find_candidates.assert_awaited_once()


@pytest.mark.asyncio
async def test_nonempty_normalized_name_still_requires_name_screening(normalization):
    text = "INN 1234567890"
    norm = await normalization.normalize_async(text, language="en")
    signals = await SignalsService().extract_signals(text, norm, "en")
    service = orchestrator()
    errors = []
    result = await service._handle_search_layer(SimpleNamespace(normalized="Source Person"), None, errors, text,
        signals_result=signals, search_options=SearchOpts())
    assert result is None and errors
    service.search_service.find_candidates.assert_awaited_once()
