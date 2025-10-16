"""An exact identifier hit does not resolve its owner in a multi-entity input."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main
from ai_service.contracts.base_contracts import SignalsExtras, UnifiedProcessingResult
from ai_service.contracts.decision_contracts import RiskLevel
from ai_service.contracts.search_contracts import Candidate, SearchInfo, SearchType
from ai_service.core.decision_engine import DecisionEngine
from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.layers.signals.signals_service import SignalsService


def candidate(identifier="1234567890", entity_id="source-one"):
    return Candidate(entity_id=entity_id, entity_type="person", normalized_name="Example Person",
        aliases=[], country="", dob="1980-01-01", meta={"tin": identifier},
        final_score=1.0, ac_score=1.0, vector_score=0, features={}, search_type=SearchType.EXACT)


def query(signals, candidates=None):
    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    values = candidates if candidates is not None else [candidate()]
    service._create_search_info_from_results = lambda _: SearchInfo(
        total_matches=len(values), fusion_candidates=values,
        has_exact_matches=bool(values), exact_confidence=1.0 if values else 0.0)
    return service._create_decision_input(
        SimpleNamespace(metadata={}, original_text="Synthetic input", language="en"),
        SimpleNamespace(normalized_text="Example Person"), signals, {"query": "Synthetic input"})


def signals_with_unassigned(identifier="1234567890", id_type="inn", people=None):
    return SimpleNamespace(persons=people or [], organizations=[], confidence=0.8,
        extras={"unassigned_ids": [{"type": id_type, "value": identifier}]})


@pytest.mark.parametrize("people", [[], [
    {"ids": [], "dob": "1980-01-01", "confidence": 0.8},
    {"ids": [], "dob": "1990-01-01", "confidence": 0.8},
]])
def test_unassigned_exact_id_remains_high_risk_and_requires_ownership_review(people):
    inp = query(signals_with_unassigned(people=people))
    value = inp.search.fusion_candidates[0]
    result = DecisionEngine().decide(inp)
    assert result.risk is RiskLevel.HIGH
    assert result.review_required is True
    assert result.required_additional_fields == []
    assert value.features["id_match"] is True
    assert value.features["unassigned_id_match"] is True
    assert value.features["identity_pair_match"] is False
    assert result.details["normalized_features"]["unassigned_id_match"] is True
    assert any("ownership" in reason.lower() for reason in result.reasons)
    assert not any("TIN+DOB" in reason for reason in result.reasons)


@pytest.mark.parametrize("extras_kind", ["dict", "contract"])
def test_query_supports_contract_and_service_extras(extras_kind):
    signals = signals_with_unassigned()
    if extras_kind == "contract":
        signals.extras = SignalsExtras(unassigned_ids=signals.extras["unassigned_ids"])
    inp = query(signals)
    assert inp.search.fusion_candidates[0].features["unassigned_id_match"] is True


@pytest.mark.parametrize("identifier,id_type", [
    ("001234567890", "inn"), ("9999999999", "inn"), ("1234567890", "passport_rf"),
])
def test_unrelated_or_different_kind_identifiers_do_not_invent_ownership_matches(identifier, id_type):
    inp = query(signals_with_unassigned(identifier, id_type))
    assert inp.search.fusion_candidates[0].features["unassigned_id_match"] is False
    assert inp.search.fusion_candidates[0].features["id_match"] is False
    assert DecisionEngine().decide(inp).details["normalized_features"]["unassigned_id_match"] is False


def test_one_confirmed_person_cannot_clear_another_unassigned_source_hit():
    signals = signals_with_unassigned("9999999999", people=[
        {"ids": [{"type": "inn", "value": "1234567890"}], "dob": "1980-01-01"}])
    inp = query(signals, [candidate(), candidate("9999999999", "source-two")])
    result = DecisionEngine().decide(inp)
    assert result.risk is RiskLevel.HIGH and result.review_required is True
    first, second = inp.search.fusion_candidates
    assert first.features["identity_pair_match"] is True
    assert first.features["unassigned_id_match"] is False
    assert second.features["identity_pair_match"] is False
    assert second.features["unassigned_id_match"] is True


def test_associated_identity_keeps_existing_confirmation_behavior():
    signals = signals_with_unassigned("9999999999", people=[
        {"ids": [{"type": "inn", "value": "1234567890"}], "dob": "1980-01-01"}])
    inp = query(signals)
    result = DecisionEngine().decide(inp)
    assert result.risk is RiskLevel.HIGH and result.review_required is False
    assert inp.search.fusion_candidates[0].features["unassigned_id_match"] is False
    assert any("TIN+DOB" in reason for reason in result.reasons)


@pytest.mark.parametrize("separator", ["; ", "\n", " | "])
@pytest.mark.asyncio
async def test_real_extraction_and_decision_preserve_unassigned_id_ownership(separator):
    text = "Example Person DOB 1980-01-01" + separator + "INN 12345\u200b67890"
    signals = await SignalsService().extract_signals(text,
        {"persons_core": [["Example", "Person"]], "organizations_core": []}, "en")
    assert not signals.persons[0].ids
    item, = signals.extras["unassigned_ids"]
    assert item["value"] == "1234567890"
    start, end = item["position"]
    assert item["raw"] == text[start:end]
    inp = query(signals)
    result = DecisionEngine().decide(inp)
    assert result.risk is RiskLevel.HIGH and result.review_required is True
    assert not any("TIN+DOB" in reason for reason in result.reasons)


@pytest.mark.parametrize("path", ["/process", "/process-batch"])
@pytest.mark.asyncio
async def test_real_unassigned_evidence_survives_single_and_batch_http(monkeypatch, path):
    text = "Example Person DOB 1980-01-01; INN 12345\u200b67890"
    signals = await SignalsService().extract_signals(text,
        {"persons_core": [["Example", "Person"]], "organizations_core": []}, "en")
    decision = DecisionEngine().decide(query(signals))
    result = UnifiedProcessingResult(original_text=text, normalized_text="Example Person",
        language="en", language_confidence=1.0, tokens=[], trace=[], signals=signals,
        decision=decision, success=True, search_results={"results": []})
    service = SimpleNamespace(process=AsyncMock(return_value=result),
        process_batch=AsyncMock(return_value=[result]), enable_search=True)
    monkeypatch.setattr(main, "orchestrator", service)
    client = TestClient(main.app)
    response = client.post(path, json={"text": text} if path == "/process" else {"texts": [text]})
    assert response.status_code == 200
    row = response.json() if path == "/process" else response.json()["results"][0]
    assert row["signals"]["persons"][0]["ids"] == []
    assert row["decision"]["review_required"] is True
    item, = row["signals"]["extras"]["unassigned_ids"]
    assert item["value"] == "1234567890"
    start, end = item["position"]
    assert item["raw"] == text[start:end]


@pytest.mark.parametrize("kind", ["missing", "dict", "contract"])
def test_public_extras_have_known_fields_without_exposing_arbitrary_attributes(kind):
    signals = SimpleNamespace(persons=[], organizations=[], confidence=0.0)
    if kind != "missing":
        extras = {"dates": [{"value": "1980-01-01"}], "amounts": [], "unassigned_ids": []}
        signals.extras = SignalsExtras(**extras) if kind == "contract" else extras
        if kind == "dict":
            signals.extras["private_debug"] = "not-a-public-field"
    row = main._extract_signals_dict(SimpleNamespace(signals=signals))
    assert set(row["extras"]) == {"dates", "amounts", "unassigned_ids"}
    assert row["extras"]["unassigned_ids"] == []
    assert row["extras"]["dates"] == ([] if kind == "missing" else [{"value": "1980-01-01"}])
