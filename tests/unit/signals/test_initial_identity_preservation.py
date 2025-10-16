"""Initials distinguish source entities and must survive signal extraction."""

from types import SimpleNamespace

import pytest

from ai_service.contracts.search_contracts import Candidate, SearchInfo, SearchType
from ai_service.core.decision_engine import DecisionEngine
from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.layers.normalization.normalization_service import NormalizationService
from ai_service.layers.signals.signals_service import SignalsService
from ai_service.main import _extract_signals_dict


@pytest.fixture(scope="module")
def normalization():
    return NormalizationService()


NAME_CASES = [
    ("ru", "И. Петров", "С. Петров"),
    ("ru", "Петров И.", "Петров С."),
    ("ru", "И. И. Петров", "С. С. Петров"),
    ("uk", "І. Коваленко", "С. Коваленко"),
    ("uk", "Коваленко І.", "Коваленко С."),
    ("uk", "І. І. Коваленко", "С. С. Коваленко"),
    ("en", "J. Smith", "A. Smith"),
    ("en", "Smith J.", "Smith A."),
    ("en", "J. J. Smith", "A. A. Smith"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("language,first,second", NAME_CASES)
@pytest.mark.parametrize("controls", [False, True])
async def test_distinct_initials_keep_separate_ids_dates_and_public_names(normalization, language, first, second, controls):
    text = f"{first} INN 1234567890 DOB 1980-01-01; {second} INN 001234567890 DOB 1990-01-01"
    if controls:
        text = text.replace(".", ".\u200b").replace("12345", "12345\u2066").replace("1980-", "1980-\u2069")
    norm = await normalization.normalize_async(text, language=language)
    assert norm.normalized == first + " | " + second
    result = await SignalsService().extract_signals(text, norm, language)
    assert [p.full_name for p in result.persons] == [first, second]
    assert [p.core for p in result.persons] == [first.split(), second.split()]
    assert [[item["value"] for item in p.ids] for p in result.persons] == [["1234567890"], ["001234567890"]]
    assert [p.dob for p in result.persons] == ["1980-01-01", "1990-01-01"]
    public = _extract_signals_dict(SimpleNamespace(signals=result))
    assert [p["full_name"] for p in public["persons"]] == [first, second]
    assert public["extras"]["unassigned_ids"] == []
    for person in public["persons"]:
        item, = person["ids"]
        start, end = item["position"]
        assert item["raw"] == text[start:end]
        start, end = person["dob_position"]
        assert person["dob_raw"] == text[start:end]


@pytest.mark.asyncio
@pytest.mark.parametrize("second", ["A. Smith", "J. Smith"])
async def test_unresolved_evidence_is_not_assigned_by_person_order(normalization, second):
    if second == "J. Smith":
        text = "J. Smith INN 1234567890; J. Smith INN 001234567890"
        expected = ["1234567890", "001234567890"]
    else:
        text = "J. Smith; A. Smith; INN 1234567890"
        expected = ["1234567890"]
    norm = await normalization.normalize_async(text, language="en")
    result = await SignalsService().extract_signals(text, norm, "en")
    assert len(result.persons) == 2
    assert [p.full_name for p in result.persons] == ["J. Smith", second]
    assert all(not p.ids for p in result.persons)
    assert [item["value"] for item in _extract_signals_dict(SimpleNamespace(signals=result))["extras"]["unassigned_ids"]] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("date,paired", [("1980-01-01", True), ("1990-01-01", False)])
async def test_initials_do_not_allow_combining_different_peoples_identity_evidence(normalization, date, paired):
    text = "J. Smith INN 1234567890 DOB 1980-01-01; A. Smith INN 001234567890 DOB 1990-01-01"
    norm = await normalization.normalize_async(text, language="en")
    signals = await SignalsService().extract_signals(text, norm, "en")
    candidate = Candidate(entity_id="synthetic-person", entity_type="person", normalized_name="Source Person",
        aliases=[], country="", dob=date, meta={"tin": "1234567890"}, final_score=1.0,
        ac_score=1.0, vector_score=0, features={}, search_type=SearchType.EXACT)
    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    service._create_search_info_from_results = lambda _: SearchInfo(total_matches=1, fusion_candidates=[candidate])
    query = service._create_decision_input(SimpleNamespace(metadata={}, original_text=text, language="en"),
        SimpleNamespace(normalized_text=norm.normalized), signals, {"query": norm.normalized})
    assert candidate.features["id_match"] is True
    assert candidate.features["date_match"] is True
    assert candidate.features["identity_pair_match"] is paired
    assert candidate.features["identity_conflict"] is not paired
    assert candidate.features["unassigned_id_match"] is False
    assert DecisionEngine().decide(query).review_required is not paired


@pytest.mark.parametrize("token,valid", [("Ё.", True), ("Ї.", True), ("É.", True),
    ("1.", False), ("..", False), ("A..", False), ("Dr.", False), ("INN", False)])
def test_normalized_initial_is_a_letter_not_a_numeric_or_document_abbreviation(token, valid):
    assert SignalsService()._is_valid_person_token(token, "en") is valid
