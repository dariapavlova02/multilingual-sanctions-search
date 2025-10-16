from types import SimpleNamespace

from ai_service.contracts.decision_contracts import DecisionInput, RiskLevel, SignalsInfo
from ai_service.contracts.search_contracts import Candidate, SearchInfo, SearchType
from ai_service.core.decision_engine import DecisionEngine


def make_query(score=0.81):
    candidate = Candidate(
        entity_id="synthetic-identity", entity_type="person", normalized_name="Example Person",
        aliases=[], country="", dob="1970-01-01", meta={"tin": "different-id"},
        final_score=score, ac_score=score, vector_score=0, features={}, search_type=SearchType.PHRASE,
    )
    return DecisionInput(
        text="Example Person", signals=SignalsInfo(0.7, 0, id_match=False, date_match=False),
        search=SearchInfo(total_matches=1, fusion_candidates=[candidate]),
    )


def test_identifiers_present_on_record_do_not_imply_a_match():
    result = DecisionEngine().decide(make_query())
    assert result.risk != RiskLevel.HIGH
    assert not any("TIN+DOB" in reason for reason in result.reasons)


def test_strong_name_match_with_contradictory_identity_requires_review():
    query = make_query(0.99)
    query.search.has_exact_matches = True
    query.search.exact_confidence = 0.99
    result = DecisionEngine().decide(query)
    assert result.risk == RiskLevel.HIGH
    assert result.review_required
    assert not any("TIN+DOB" in reason for reason in result.reasons)


def test_normalization_evidence_is_retained_for_scoring():
    query = make_query()
    engine = DecisionEngine()
    baseline = engine.decide(query)
    query.normalization = SimpleNamespace(homoglyph_detected=True)
    assert engine.decide(query).score > baseline.score


def test_identifiers_and_dates_from_different_people_are_not_combined():
    from ai_service.core.unified_orchestrator import UnifiedOrchestrator

    candidate = make_query().search.fusion_candidates[0]
    candidate.meta = {"tin": "1234567890"}
    search = SearchInfo(total_matches=1, fusion_candidates=[candidate])
    orchestrator = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    orchestrator._create_search_info_from_results = lambda results: search
    signals = SimpleNamespace(
        persons=[
            {"ids": [{"type": "tin", "value": "1234567890"}], "dob": "1985-01-01"},
            {"ids": [{"type": "tin", "value": "9999999999"}], "dob": "1970-01-01"},
        ], organizations=[], confidence=0.9,
    )
    query = orchestrator._create_decision_input(
        SimpleNamespace(metadata={}, original_text="Two people", language="en"),
        SimpleNamespace(normalized_text="Two people"), signals, {"query": "Two people"},
    )
    assert candidate.features["id_match"]
    assert candidate.features["date_match"]
    assert not candidate.features["identity_pair_match"]
    assert candidate.features["identity_conflict"]
    result = DecisionEngine().decide(query)
    assert result.review_required
    assert not any("TIN+DOB" in reason for reason in result.reasons)
