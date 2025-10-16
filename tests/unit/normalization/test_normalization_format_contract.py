"""Formatting must survive gender preservation and remain visible in the trace."""

import pytest

from ai_service.layers.normalization.normalization_service import NormalizationService
from ai_service.layers.normalization.processors.normalization_factory import NormalizationConfig
from ai_service.layers.normalization.processors.token_processor import TokenProcessor
from ai_service.layers.signals.signals_service import SignalsService
from ai_service.utils.feature_flags import FeatureFlags


@pytest.fixture(scope="module")
def service():
    return NormalizationService()


@pytest.mark.asyncio
@pytest.mark.parametrize("language,expected", [
    ("ru", "Петров-Сидоров"), ("ru", "Петрова-Сидорова"),
    ("ru", "Петров"), ("ru", "Петрова"),
    ("uk", "Коваленко-Петренко"), ("uk", "Ковальська-Левицька"),
])
@pytest.mark.parametrize("casing", ["lower", "upper"])
async def test_standalone_surname_casing_survives_gender_preservation(service, language, expected, casing):
    text = getattr(expected, casing)()
    result = await service.normalize_async(text, language=language)
    assert result.success
    assert result.normalized == expected
    assert result.persons_core == [[expected]]
    assert result.persons[0]["original_tokens"] == [text]
    again = await service.normalize_async(result.normalized, language=language)
    assert again.normalized == expected


@pytest.mark.parametrize("source", ["петрова-сидорова", "ПЕТРОВА-СИДОРОВА"])
def test_formatting_does_not_replace_feminine_source_with_masculine_lemma(service, source):
    person = service.normalization_factory._finalize_person(
        [(source, "surname")], ["Петров-Сидоров"], "ru")
    assert person["tokens"] == ["Петрова-Сидорова"]
    assert person["original_tokens"] == [source]


@pytest.mark.asyncio
@pytest.mark.parametrize("language,text,expected,initial", [
    ("ru", "И.. И. Петров", "И. И. Петров", "И"),
    ("uk", "І.. І. Коваленко", "І. І. Коваленко", "І"),
    ("en", "J.. Smith", "J. Smith", "J"),
])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fsm", [False, True])
async def test_initial_repair_has_actual_before_after_trace_with_or_without_cache_and_fsm(
        service, language, text, expected, initial, cached, fsm):
    config = NormalizationConfig(language=language, enable_cache=cached, enable_fsm_tuned_roles=fsm)
    flags = FeatureFlags(enable_fsm_tuned_roles=fsm, fix_initials_double_dot=True, debug_tracing=False)
    for _ in range(2):
        result = await service.normalization_factory.normalize_text(text, config, flags)
        assert result.success and result.normalized == expected
        assert any(item.rule == "collapse_double_dots" and item.token == initial + ".."
                   and item.output == initial + "." for item in result.trace)


@pytest.mark.parametrize("initial", ["И", "І", "J", "Ё"])
def test_token_processor_records_initial_transformation_without_changing_ellipsis(initial):
    traces = []
    result = TokenProcessor()._fix_initials_double_dot([initial + "..", "...", "и.о.", initial + "."], traces)
    assert result == [initial + ".", "...", "и.о.", initial + "."]
    events = [trace for trace in traces if isinstance(trace, dict)]
    assert len(events) == 1
    assert events[0]["rule"] == "collapse_double_dots"
    assert events[0]["before"] == initial + ".."
    assert events[0]["after"] == initial + "."


@pytest.mark.asyncio
@pytest.mark.parametrize("language,text", [("ru", "И. И. Петров"), ("uk", "І. Коваленко"), ("en", "J. Smith")])
async def test_already_clean_initial_does_not_report_a_repair(service, language, text):
    result = await service.normalize_async(text, language=language)
    assert result.success
    assert not any("collapse_double_dots" in item.rule for item in result.trace)


@pytest.mark.asyncio
async def test_formatted_compound_names_keep_separate_source_identifiers(service):
    text = "петров-сидоров INN 1234567890; кузнецов-морозов INN 9999999999"
    normalized = await service.normalize_async(text, language="ru")
    assert normalized.normalized == "Петров-Сидоров | Кузнецов-Морозов"
    signals = await SignalsService().extract_signals(text, normalized, "ru")
    assert len(signals.persons) == 2
    assert [[item["value"] for item in person.ids] for person in signals.persons] == [["1234567890"], ["9999999999"]]
    for person in signals.persons:
        item, = person.ids
        start, end = item["position"]
        assert item["raw"] == text[start:end]


@pytest.mark.asyncio
@pytest.mark.parametrize("language,marker", [("ru", "ИНН"), ("uk", "ІПН"), ("en", "INN")])
@pytest.mark.parametrize("prefix", ["", "Петров-Сидоров; "])
async def test_identifier_marker_cannot_become_a_person_or_claim_an_unassigned_id(service, language, marker, prefix):
    text = prefix + marker + " 1234567890"
    result = await service.normalize_async(text, language=language)
    assert result.success
    assert result.normalized == ("Петров-Сидоров" if prefix else "")
    assert not any(item.role in {"given", "surname", "patronymic"} and item.token.casefold() == marker.casefold()
                   for item in result.trace)
    signals = await SignalsService().extract_signals(text, result, language)
    assert len(signals.persons) == (1 if prefix else 0)
    assert all(not person.ids for person in signals.persons)
    item, = signals.extras["unassigned_ids"]
    assert item["value"] == "1234567890"
    start, end = item["position"]
    assert item["raw"] == text[start:end]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,expected", [("Smith-Jones", "Smith-Jones"), ("O'Connor", "O’Connor")])
async def test_english_compound_survives_normalization_to_signals(service, name, expected):
    text = name + " INN 1234567890"
    result = await service.normalize_async(text, language="en")
    assert result.normalized == expected
    signals = await SignalsService().extract_signals(text, result, "en")
    person, = signals.persons
    # The existing en_apostrophe golden uses a typographic display apostrophe
    # and an ASCII lemma; signal cores retain that lemma spelling.
    assert person.full_name == name
    item, = person.ids
    assert item["value"] == "1234567890"
    start, end = item["position"]
    assert item["raw"] == text[start:end]


@pytest.mark.parametrize("value", ["Smith-123", "--Smith", "Smith-", "INN", "..."])
def test_compound_signal_validation_does_not_admit_labels_or_malformed_names(value):
    assert not SignalsService()._is_valid_person_token(value, "en")


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fsm", [False, True])
async def test_each_repeated_initial_has_one_transformation_event(service, cached, fsm):
    result = await service.normalization_factory.normalize_text("И.. И.. Петров",
        NormalizationConfig(language="ru", enable_cache=cached, enable_fsm_tuned_roles=fsm),
        FeatureFlags(enable_fsm_tuned_roles=fsm, fix_initials_double_dot=True))
    assert result.normalized == "И. И. Петров"
    events = [item for item in result.trace
        if item.rule == "collapse_double_dots" and item.token == "И.." and item.output == "И."]
    assert len(events) == 2
