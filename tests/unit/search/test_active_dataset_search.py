"""Retrieval must preserve active-source identity and reject incomplete evidence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import Candidate, SearchMode, SearchOpts
from ai_service.layers.search.elasticsearch_adapters import (
    ElasticsearchACAdapter,
    ElasticsearchVectorAdapter,
)
from ai_service.layers.search.fuzzy_search_service import (
    FuzzyConfig,
    FuzzySearchService,
)
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.search_integrity import best_per_entity


def candidate(doc_id, entity_id, score=0.9, source="fixture", mode=SearchMode.AC):
    return Candidate(
        doc_id,
        score,
        "Example Person",
        "person",
        {"entity_id": entity_id, "source": source},
        mode,
        ["name"],
        score,
    )


def hit(doc_id, entity_id, pattern, source="fixture", sort=1):
    return {
        "_id": doc_id,
        "sort": [sort],
        "_source": {
            "entity_id": entity_id,
            "entity_type": "person",
            "pattern": pattern,
            "name": pattern,
            "source_list": source,
            "country": ["UA", "GB"],
            "metadata": {"dob": "1980-01-01"},
        },
    }


def normalized():
    return NormalizationResult(
        normalized="Example Person",
        tokens=["Example", "Person"],
        trace=[],
        language="en",
        confidence=1.0,
    )


def test_aliases_merge_but_same_name_and_ids_in_different_sources_do_not():
    values = [
        candidate("alias", "0", 0.8),
        candidate("canonical", "0", 0.9),
        candidate("different-person", "1"),
        candidate("different-source", "0", source="other"),
    ]
    assert {c.doc_id for c in best_per_entity(values)} == {
        "canonical",
        "different-person",
        "different-source",
    }


@pytest.mark.asyncio
async def test_fuzzy_uses_active_index_and_keeps_colliding_aliases_across_pages():
    service = HybridSearchService(HybridSearchConfig())
    service._fuzzy_service.config.max_results = 1

    async def pages(opts, batch_size):
        yield [hit("a1", "one", "Example Person"), hit("a2", "one", "Example Person")]
        yield [
            hit("b", "two", "Example Person"),
            hit("c", "one", "Example Person", source="other"),
        ]

    service._ac_adapter = SimpleNamespace(iter_documents=pages)
    results = await service._fuzzy_search("example person", SearchOpts(top_k=3))
    assert len(results) == 3
    assert {r.doc_id for r in results} == {"a1", "b", "c"}
    assert all(r.metadata["country"] == ["UA", "GB"] for r in results)
    assert all(r.metadata["dob"] == "1980-01-01" for r in results)


@pytest.mark.asyncio
async def test_snapshot_updates_pit_id_and_closes_on_partial_response():
    client = SimpleNamespace(
        open_point_in_time=AsyncMock(return_value={"id": "first"}),
        close_point_in_time=AsyncMock(),
        search=AsyncMock(
            side_effect=[
                {
                    "pit_id": "second",
                    "hits": {"hits": [hit("a", "one", "Example Person")]},
                },
                {"pit_id": "third", "timed_out": True, "hits": {"hits": []}},
            ]
        ),
    )
    client.options = Mock(return_value=client)
    adapter = ElasticsearchACAdapter(
        HybridSearchConfig(elasticsearch={"ac_index": "active_fixture"})
    )
    adapter._ensure_connection = AsyncMock(return_value=client)
    with pytest.raises(RuntimeError, match="incomplete"):
        async for _ in adapter.iter_documents(
            SearchOpts(metadata_filters={"entity_id": "one"}), batch_size=2
        ):
            pass
    client.close_point_in_time.assert_awaited_once_with(id="third")
    assert client.open_point_in_time.await_args.kwargs["index"] == "active_fixture"
    second = client.search.await_args_list[1].kwargs
    assert second["body"]["search_after"] == [1]
    assert second["body"]["pit"]["id"] == "second"
    assert second["body"]["query"]["bool"]["filter"] == [{"term": {"entity_id": "one"}}]
    assert second["allow_partial_search_results"] is False


@pytest.mark.asyncio
async def test_fuzzy_timeout_closes_snapshot_without_returning_partial_matches():
    closed = False

    async def pages(opts, batch_size):
        nonlocal closed
        try:
            yield [hit("a", "one", "Example Person")]
            await asyncio.sleep(1)
        finally:
            closed = True

    service = HybridSearchService(HybridSearchConfig())
    service._ac_adapter = SimpleNamespace(iter_documents=pages)
    with pytest.raises(TimeoutError):
        await service._fuzzy_search("Example Person", SearchOpts(timeout_ms=100))
    assert closed


@pytest.mark.parametrize(
    "adapter_class", [ElasticsearchACAdapter, ElasticsearchVectorAdapter]
)
@pytest.mark.parametrize("failure", [{"timed_out": True}, {"_shards": {"failed": 1}}])
def test_all_retrieval_modes_reject_partial_es_results(adapter_class, failure):
    adapter = adapter_class(HybridSearchConfig())
    with pytest.raises(RuntimeError, match="incomplete"):
        adapter._parse_candidates({**failure, "hits": {"hits": []}})


@pytest.mark.asyncio
async def test_fuzzy_algorithm_failure_propagates(monkeypatch):
    from ai_service.layers.search import fuzzy_search_service

    service = FuzzySearchService(FuzzyConfig())
    monkeypatch.setattr(
        fuzzy_search_service.process, "extract", Mock(side_effect=ValueError("failure"))
    )
    with pytest.raises(RuntimeError, match="Fuzzy matching failed"):
        await service.search_async("Example Person", ["Example Person"])


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_hit", [False, True])
async def test_changed_generation_cannot_publish_or_cache_a_result(cache_hit):
    service = HybridSearchService(HybridSearchConfig())
    service._initialized = True
    service.readiness = AsyncMock(
        side_effect=[{"active": "before"}, {"active": "after"}]
    )
    service._ac_search_only = AsyncMock(return_value=[candidate("a", "one")])
    service._get_cached_search_result = AsyncMock(
        return_value=[] if cache_hit else None
    )
    service._cache_search_result = AsyncMock()
    with pytest.raises(RuntimeError) as caught:
        await service.find_candidates(
            normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC)
        )
    error = caught.value if cache_hit else caught.value.__cause__
    assert "changed during screening" in str(error)
    service._cache_search_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_fuzzy_mode_needs_only_ac_index_and_does_not_dispatch_hybrid():
    service = HybridSearchService(HybridSearchConfig())
    service._initialized = True
    service.readiness = AsyncMock(return_value={"active": "generation"})
    service._fuzzy_search = AsyncMock(
        return_value=[candidate("a", "one", mode=SearchMode.FUZZY)]
    )
    service._hybrid_search = AsyncMock(side_effect=AssertionError("wrong dispatch"))
    result = await service.find_candidates(
        normalized(), "Example Person", SearchOpts(search_mode=SearchMode.FUZZY)
    )
    assert len(result) == 1
    assert all(
        call.kwargs == {"require_vectors": False}
        for call in service.readiness.await_args_list
    )


@pytest.mark.asyncio
async def test_vector_escalation_preserves_fuzzy_hits_and_does_not_repeat_vector_query():
    service = HybridSearchService(HybridSearchConfig())
    service._ac_search_only = AsyncMock(return_value=[])
    service._fuzzy_search = AsyncMock(
        return_value=[candidate("fuzzy", "one", 0.6, mode=SearchMode.FUZZY)]
    )
    service._vector_search_only = AsyncMock(
        return_value=[candidate("vector", "two", mode=SearchMode.VECTOR)]
    )
    service._fuzzy_results_sufficient = Mock(return_value=False)
    result = await service._hybrid_search(
        normalized(), "Example Person", SearchOpts(threshold=0.5)
    )
    assert {c.doc_id for c in result} == {"fuzzy", "vector"}
    service._vector_search_only.assert_awaited_once()


def test_missing_other_strategy_does_not_reduce_score_and_aliases_do_not_boost_it():
    service = HybridSearchService(HybridSearchConfig())
    ac = candidate("ac", "one", 0.9)
    assert service._combine_results([ac], [], SearchOpts())[0].score == 0.9
    vectors = [
        candidate("vector1", "one", 0.8, mode=SearchMode.VECTOR),
        candidate("vector2", "one", 0.8, mode=SearchMode.VECTOR),
    ]
    result = service._combine_results([ac], vectors, SearchOpts())
    assert len(result) == 1
    assert 0.8 <= result[0].score <= 0.9
    assert result[0].metadata["retrieval_scores"] == {"lexical": 0.9, "vector": 0.8}


def test_post_filter_matches_es_multi_value_semantics():
    service = HybridSearchService(HybridSearchConfig())
    value = candidate("document", "0")
    value.metadata["country"] = ["UA", "GB"]
    assert service._matches_metadata_filters(
        value, {"country": ["UA"], "entity_id": "0"}
    )
    assert not service._matches_metadata_filters(value, {"entity_id": "document"})


@pytest.mark.asyncio
async def test_identifier_lookup_reads_active_metadata_and_does_not_match_substrings():
    service = HybridSearchService(HybridSearchConfig())
    service._initialized = True
    service.readiness = AsyncMock(return_value={"active": "one"})
    first = hit("a", "one", "Example Person")
    first["_source"]["metadata"]["itn_import"] = "1234567890; 9876543210"
    alias = hit("b", "one", "Example Alias")
    alias["_source"]["metadata"]["itn"] = "1234567890"
    other = hit("c", "two", "Other Person")
    other["_source"]["metadata"]["itn"] = "001234567890"

    async def pages(opts, batch_size=1000):
        yield [first, alias, other]

    service._ac_adapter = SimpleNamespace(iter_documents=pages)
    results = await service.find_by_identifier("1234567890", "inn_ua")
    assert len(results) == 1
    assert results[0].metadata["entity_id"] == "one"
    assert results[0].trace["id_match"] is True
    assert results[0].metadata["source"] == "fixture"


@pytest.mark.asyncio
async def test_orchestrator_identifier_lookup_uses_configured_search_service():
    from ai_service.core.unified_orchestrator import UnifiedOrchestrator

    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    value = candidate("active-document", "one")
    service.search_service = SimpleNamespace(
        find_by_identifier=AsyncMock(return_value=[value])
    )
    result = await service._find_candidates_by_id("1234567890", "inn_ua")
    assert result[0]["doc_id"] == "active-document"
    assert result[0]["search_mode"] == "id_exact"
    assert (
        "itn" not in result[0]["metadata"]
    )  # Query data must not be fabricated as source evidence.
    service.search_service.find_by_identifier.assert_awaited_once_with(
        "1234567890", "inn_ua", None
    )


@pytest.mark.asyncio
async def test_identifier_evidence_survives_validation_and_name_normalization():
    from ai_service.core.unified_orchestrator import UnifiedOrchestrator
    from ai_service.layers.signals.signals_service import SignalsService
    from ai_service.utils.input_validation import InputValidator
    raw = 'Example Person INN 1234567890 DOB 1980-01-01'
    sanitized = InputValidator().validate_and_sanitize(raw).sanitized_text
    assert '1234567890' in sanitized
    assert '1980-01-01' in sanitized
    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    service.signals_service = SignalsService()
    service.metrics_service = None
    result = await service._handle_signals_layer(
        'Example Person INN l2e4s6789o',
        {'persons_core': [['Example', 'Person']], 'organizations_core': [], 'trace': []},
        SimpleNamespace(original_text=raw, language='en'),
    )
    assert any(item['value'] == '1234567890' for item in result.persons[0].ids)
    assert result.persons[0].dob == '1980-01-01'


@pytest.mark.asyncio
async def test_real_name_normalization_retains_identity_evidence_boundaries():
    from ai_service.core.unified_orchestrator import UnifiedOrchestrator
    from ai_service.layers.normalization.normalization_service import NormalizationService
    from ai_service.layers.signals.signals_service import SignalsService
    from ai_service.utils.feature_flags import FeatureFlags

    raw = 'Replacement Example INN 1234567890 DOB 1980-01-01'
    normalization = await NormalizationService().normalize_async(
        raw, language='en', feature_flags=FeatureFlags()
    )
    assert normalization.normalized == 'Replacement Example'
    assert normalization.persons_core == [['Replacement', 'Example']]
    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    service.signals_service = SignalsService()
    service.metrics_service = None
    signals = await service._handle_signals_layer(
        raw, normalization, SimpleNamespace(original_text=raw, language='en')
    )
    assert len(signals.persons) == 1
    assert signals.persons[0].dob == '1980-01-01'
    assert [item['value'] for item in signals.persons[0].ids] == ['1234567890']


@pytest.mark.parametrize('text', [
    'Для Жені Галича з групи O.Torvald',
    'Іван Петренко INN 1234567890',
    'Иван Петров DOB 1980-01-01',
])
def test_validation_preserves_separate_latin_and_cyrillic_words(text):
    from ai_service.utils.input_validation import InputValidator
    assert InputValidator().validate_and_sanitize(text).sanitized_text == text


def test_normalization_trace_cannot_invent_identifier_source_positions():
    from ai_service.layers.signals.signals_service import SignalsService
    service = SignalsService()
    trace = {'trace': [{'role': 'id', 'token': '1234567890'}]}
    assert service._extract_ids_from_normalization_trace(trace, 'Example Person') == {
        'person_ids': [], 'organization_ids': []
    }


def test_iso_birthdate_captures_and_repeated_source_occurrences():
    from ai_service.data.patterns.dates import extract_birthdates_from_text
    result = extract_birthdates_from_text('Person One DOB 1980-03-04; Person Two DOB 1980-03-04')
    assert [item['iso_format'] for item in result] == ['1980-03-04', '1980-03-04']
    assert result[0]['position'] != result[1]['position']
    assert not extract_birthdates_from_text('Person DOB 1980-02-31')
