"""Identity evidence must remain attached to its source entity under concurrency."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.layers.signals.signals_service import PersonSignal, SignalsService


def person(name):
    return PersonSignal(core=name.split(), full_name=name)


def identifier(text, value):
    start = text.index(value)
    return dict(
        type="inn_ua",
        value=value,
        raw=value,
        confidence=0.9,
        valid=True,
        position=(start, start + len(value)),
    )


def test_unlinked_id_is_preserved_without_assigning_it_to_every_person(monkeypatch):
    service = SignalsService()
    text = "John Smith; Jane Doe; INN 1234567890"
    item = identifier(text, "1234567890")
    people = [person("John Smith"), person("Jane Doe")]
    monkeypatch.setattr(
        service.identifier_extractor, "extract_person_ids", lambda _: [item]
    )
    monkeypatch.setattr(
        service.identifier_extractor, "extract_organization_ids", lambda _: []
    )
    unresolved = service._enrich_with_identifiers(text, people, [], None)
    assert unresolved == [item]
    assert all(not value.ids for value in people)


def test_ids_stay_with_the_person_in_their_source_clause():
    service = SignalsService()
    text = "John Smith INN 1234567890; Jane Doe INN 9999999999"
    people = [person("John Smith"), person("Jane Doe")]
    service._link_ids_to_persons_by_proximity(
        people,
        [identifier(text, value) for value in ("1234567890", "9999999999")],
        text,
    )
    assert [[item["value"] for item in value.ids] for value in people] == [
        ["1234567890"],
        ["9999999999"],
    ]


def test_repeated_date_value_can_belong_to_two_distinct_people():
    service = SignalsService()
    text = "John Smith DOB 1980-01-01; Jane Doe DOB 1980-01-01"
    people = [person("John Smith"), person("Jane Doe")]
    date = "1980-01-01"
    offsets = [text.index(date), text.rindex(date)]
    service._enrich_persons_with_birthdates(
        people,
        [
            dict(raw=date, iso_format=date, position=(start, start + len(date)))
            for start in offsets
        ],
        text,
    )
    assert [value.dob for value in people] == [date, date]


def test_conflicting_birthdates_are_not_silently_overwritten():
    service = SignalsService()
    text = "John Smith DOB 1980-01-01 or 1990-01-01"
    value = person("John Smith")
    value.dob = "1980-01-01"
    value.dob_raw = "1980-01-01"
    value.dob_position = (15, 25)
    dates = [
        dict(
            raw=date,
            iso_format=date,
            position=(text.index(date), text.index(date) + len(date)),
        )
        for date in ("1980-01-01", "1990-01-01")
    ]
    service._enrich_persons_with_birthdates([value], dates, text)
    assert value.dob is None
    assert value.dob_raw is None and value.dob_position is None
    assert "conflicting_birthdates" in value.evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("language,text,expected", [
    ("ru", "Саши Пушкина ИНН 1234567890 ДР 1980-01-01", "Александр Пушкин"),
    ("uk", "Жені Галича ІПН 1234567890 ДН 1980-01-01", "Євген Галич"),
    ("en", "Bill Smith INN 1234567890 DOB 1980-01-01", "William Smith"),
])
async def test_inflected_and_expanded_names_keep_their_source_evidence(language, text, expected):
    from ai_service.layers.normalization.normalization_service import NormalizationService

    normalization = await NormalizationService().normalize_async(text, language=language)
    assert normalization.normalized == expected
    result = await SignalsService().extract_signals(text, normalization, language)
    assert len(result.persons) == 1
    assert result.persons[0].dob == "1980-01-01"
    assert [item["value"] for item in result.persons[0].ids] == ["1234567890"]


def test_source_name_alias_cannot_cross_an_entity_boundary():
    text = "Саши Пушкина; Other Person INN 1234567890"
    people = [person("Александр Пушкин"), person("Other Person")]
    people[0].source_names = ["Саши Пушкина"]
    SignalsService()._link_ids_to_persons_by_proximity(people, [identifier(text, "1234567890")], text)
    assert not people[0].ids
    assert [item["value"] for item in people[1].ids] == ["1234567890"]


@pytest.mark.asyncio
async def test_concurrent_signal_requests_do_not_share_text(monkeypatch):
    service = SignalsService()
    barrier = threading.Barrier(2, timeout=5)

    def cores(text, normalization, language):
        barrier.wait()
        return [text.split(" INN ")[0].split()], []

    monkeypatch.setattr(service, "_get_entity_cores", cores)
    monkeypatch.setattr(service, "_create_organization_signals", lambda *args: [])
    monkeypatch.setattr(
        service.identifier_extractor,
        "extract_person_ids",
        lambda text: [identifier(text, text.split(" INN ")[1])],
    )
    monkeypatch.setattr(
        service.identifier_extractor, "extract_organization_ids", lambda _: []
    )
    results = await asyncio.gather(
        service.extract_async("John Smith INN 1234567890"),
        service.extract_async("Jane Doe INN 9999999999"),
    )
    assert [
        [item["value"] for item in result["persons"][0]["ids"]] for result in results
    ] == [["1234567890"], ["9999999999"]]


@pytest.mark.asyncio
async def test_unassigned_identifiers_still_reach_screening():
    service = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    service._find_candidates_by_id = AsyncMock(
        return_value=[{"doc_id": "source-match"}]
    )
    signals = SimpleNamespace(
        persons=[],
        organizations=[],
        extras={"unassigned_ids": [{"type": "inn_ua", "value": "1234567890"}]},
    )
    result = await service._search_by_extracted_ids(signals, None)
    assert result == [{"doc_id": "source-match"}]
    service._find_candidates_by_id.assert_awaited_once_with("1234567890", "inn_ua", None)
