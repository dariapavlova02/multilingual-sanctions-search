"""Request-dependent morphology must not inherit a previous request's policy."""

import pytest

from ai_service.layers.normalization.morphology_adapter import MorphologyAdapter
from ai_service.utils.feature_flags import FeatureFlags


@pytest.fixture
def adapter():
    return MorphologyAdapter(cache_size=2)


def test_cached_form_respects_role_and_custom_rules(adapter):
    expanded = FeatureFlags(morphology_custom_rules_first=True)
    literal = FeatureFlags(morphology_custom_rules_first=False)
    assert adapter.to_nominative_cached("Петрика", "uk", expanded, "given")[0] == "Петро"
    assert adapter.to_nominative_cached("Петрика", "uk", expanded, "surname")[0] == "Петрик"
    assert adapter.to_nominative_cached("Петрика", "uk", literal, "given")[0] == "Петрик"
    assert adapter.to_nominative_cached("Петрика", "uk", expanded, "given")[0] == "Петро"


def test_cached_form_preserves_each_requests_case(adapter):
    flags = FeatureFlags()
    for token, expected in [("Сашка", "Олександр"), ("сашка", "олександр"), ("САШКА", "ОЛЕКСАНДР")]:
        assert adapter.to_nominative_cached(token, "uk", flags, "given")[0] == expected


def test_flagged_cache_is_bounded_and_cleared(adapter):
    for token in ("Петрика", "Сашка", "Жені"):
        adapter.to_nominative_cached(token, "uk", FeatureFlags(), "given")
    assert len(adapter._to_nominative_cached_with_flags) == 2
    adapter.clear_cache()
    assert not adapter._to_nominative_cached_with_flags


@pytest.mark.parametrize("token", ["В'ячеслава", "Іванова-Петренка", "В’ячеслава"])
def test_punctuation_does_not_make_cyrillic_names_mixed_script(adapter, token):
    assert not adapter._is_mixed_script(token)


def test_actual_mixed_script_remains_detected(adapter):
    assert adapter._is_mixed_script("O'Брайен")


def test_given_name_with_apostrophe_is_declined(adapter):
    assert adapter.to_nominative_cached("В'ячеслава", "uk", FeatureFlags(), "given")[0] == "В'ячеслав"


def test_diminutive_policy_can_be_disabled_after_cache_warmup(adapter):
    expanded = FeatureFlags(enable_enhanced_diminutives=True)
    literal = FeatureFlags(enable_enhanced_diminutives=False)
    assert adapter.to_nominative_cached("Сашка", "uk", expanded, "given")[0] == "Олександр"
    assert adapter.to_nominative_cached("Сашка", "uk", literal, "given")[0] == "Сашка"


@pytest.mark.parametrize("language,text", [
    ("uk", "Для Жені Галича"),
    ("uk", "Переказ від Вовчика Зеленського В. О."),
    ("ru", "Платёж от Димы Медведева"),
    ("en", "Transfer to Stephen E. King for services"),
])
@pytest.mark.asyncio
async def test_cache_setting_does_not_change_normalization(language, text):
    from ai_service.layers.normalization.normalization_service import NormalizationService
    from ai_service.layers.normalization.processors.normalization_factory import NormalizationConfig

    service = NormalizationService()
    outputs = []
    for use_cache in (True, False, True):
        result = await service.normalization_factory.normalize_text(
            text, NormalizationConfig(language=language, enable_cache=use_cache)
        )
        assert result.success
        outputs.append((result.normalized, result.tokens, result.persons))
    assert outputs[0] == outputs[1] == outputs[2]


@pytest.mark.parametrize("text", [
    "ПЕТР СЕРГЕЕВИЧ КОЗЛОВ", "петр сергеевич козлов",
    "Петр Сергеевич Козлов", "Петра Сергеевича Козлова",
])
@pytest.mark.parametrize("yo_strategy", ["preserve", "fold"])
@pytest.mark.asyncio
async def test_given_spelling_is_stable_across_case_inflection_and_repeated_passes(text, yo_strategy):
    from ai_service.layers.normalization.processors.normalization_factory import NormalizationConfig, NormalizationFactory

    factory = NormalizationFactory()
    config = NormalizationConfig(language="ru", yo_strategy=yo_strategy)
    first = await factory.normalize_text(text, config)
    second = await factory.normalize_text(first.normalized, config)
    assert first.success and second.success
    expected = "Пётр Сергеевич Козлов" if yo_strategy == "preserve" else "Петр Сергеевич Козлов"
    assert first.normalized == second.normalized == expected
