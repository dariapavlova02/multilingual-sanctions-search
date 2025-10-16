"""Configuration integrity at file, environment, request and cache boundaries."""

import json
from dataclasses import fields

import pytest

from ai_service.utils.feature_flags import FeatureFlagManager, FeatureFlags


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    import os

    prefixes = ("AISVC_", "NORMALIZATION_IMPLEMENTATION")
    names = {
        "FACTORY_ROLLOUT_PERCENTAGE",
        "ENABLE_PERFORMANCE_FALLBACK",
        "MAX_LATENCY_THRESHOLD_MS",
        "ENABLE_ACCURACY_MONITORING",
        "MIN_CONFIDENCE_THRESHOLD",
        "ENABLE_DUAL_PROCESSING",
        "LOG_IMPLEMENTATION_CHOICE",
        "FIX_INITIALS_DOUBLE_DOT",
        "PRESERVE_HYPHENATED_CASE",
        "USE_DIMINUTIVES_DICTIONARY_ONLY",
        "DIMINUTIVES_ALLOW_CROSS_LANG",
        "APP_ENV",
    }
    for name in list(os.environ):
        if name.startswith(prefixes) or name in names:
            monkeypatch.delenv(name)


def test_serialization_covers_every_stored_field():
    flags = FeatureFlags()
    result = flags.to_dict()
    assert set(result) == {field.name for field in fields(flags)}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_diminutives_dictionary_only", True),
        ("diminutives_allow_cross_lang", True),
        ("morphology_custom_rules_first", False),
        ("debug_tracing", True),
    ],
)
def test_effective_flags_distinguish_processing_cache_keys(field, value):
    baseline = FeatureFlags().to_dict()
    changed = FeatureFlags(**{field: value}).to_dict()
    assert changed != baseline


def test_selected_file_environment_and_partial_request_precedence(
    tmp_path, monkeypatch
):
    path = tmp_path / "flags.yaml"
    path.write_text(
        "production:\n  feature_flags:\n    strict_stopwords: true\n"
        "    enable_spacy_en_ner: false\n    debug_tracing: true\n"
    )
    monkeypatch.setenv("AISVC_FEATURE_FLAGS_FILE", str(path))
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AISVC_FLAG_STRICT_STOPWORDS", "false")
    manager = FeatureFlagManager()
    flags = manager.get_flags({"debug_tracing": False})
    assert flags.strict_stopwords is False
    assert flags.enable_spacy_en_ner is False
    assert flags.debug_tracing is False
    assert manager.get_flags().debug_tracing is True


@pytest.mark.parametrize("value", ["garbage", "", "null"])
def test_invalid_boolean_environment_fails(value, monkeypatch):
    monkeypatch.setenv("AISVC_FLAG_STRICT_STOPWORDS", value)
    with pytest.raises(ValueError):
        FeatureFlagManager()


@pytest.mark.parametrize(
    "name,value",
    [
        ("FACTORY_ROLLOUT_PERCENTAGE", "101"),
        ("FACTORY_ROLLOUT_PERCENTAGE", "-1"),
        ("MAX_LATENCY_THRESHOLD_MS", "nan"),
        ("MAX_LATENCY_THRESHOLD_MS", "0"),
        ("MIN_CONFIDENCE_THRESHOLD", "inf"),
        ("MIN_CONFIDENCE_THRESHOLD", "1.1"),
        ("NORMALIZATION_IMPLEMENTATION", "typo"),
        ("NORMALIZATION_IMPLEMENTATION_RU", "typo"),
        ("AISVC_FLAG_STRICT_STOPWORD", "true"),
    ],
)
def test_invalid_environment_cannot_silently_change_policy(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        FeatureFlagManager()


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "",
        "development: []",
        "development: {feature_flags: []}",
        "production: {feature_flags: {strict_stopwords: true}}",
        "development: {feature_flags: {strict_stopword: true}}",
        "development: {feature_flags: {strict_stopwords: null}}",
        "development: {feature_flags: {strict_stopwords: true, strict_stopwords: false}}",
    ],
)
def test_invalid_explicit_file_fails(document, tmp_path, monkeypatch):
    path = tmp_path / "flags.yaml"
    path.write_text(document)
    monkeypatch.setenv("AISVC_FEATURE_FLAGS_FILE", str(path))
    with pytest.raises(ValueError):
        FeatureFlagManager()


def test_missing_explicit_file_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("AISVC_FEATURE_FLAGS_FILE", str(tmp_path / "absent.yaml"))
    with pytest.raises(ValueError):
        FeatureFlagManager()


def test_update_is_validated_before_publishing_any_field():
    manager = FeatureFlagManager()
    before = manager.get_current_config()
    with pytest.raises(ValueError):
        manager.update_flags(strict_stopwords=True, enable_spacy_ner="false")
    assert manager.get_current_config() == before


def test_invalid_update_field_is_not_ignored():
    with pytest.raises(ValueError):
        FeatureFlagManager().update_flags(strict_stopword=True)


def test_request_and_returned_configs_do_not_mutate_global_flags():
    manager = FeatureFlagManager()
    manager.update_flags(strict_stopwords=True)
    first = manager.get_flags()
    first.strict_stopwords = False
    result = manager.get_current_config()
    result["strict_stopwords"] = False
    assert manager.get_flags().strict_stopwords is True


def test_request_values_are_validated_without_mutating_base():
    manager = FeatureFlagManager()
    before = manager.get_current_config()
    with pytest.raises(ValueError):
        manager.get_flags({"strict_stopwords": "false"})
    assert manager.get_current_config() == before


def test_aliases_merge_before_canonical_defaults():
    manager = FeatureFlagManager()
    assert manager.get_flags({"ascii_fastpath": False}).enable_ascii_fastpath is False
    with pytest.raises(ValueError):
        manager.get_flags({"ascii_fastpath": False, "enable_ascii_fastpath": True})


def test_all_boolean_flags_can_be_selected_by_environment(monkeypatch):
    defaults = FeatureFlags()
    expected = {}
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        if type(value) is bool:
            expected[field.name] = not value
            monkeypatch.setenv("AISVC_FLAG_" + field.name.upper(), str(not value))
    actual = FeatureFlagManager().get_current_config()
    assert {key: actual[key] for key in expected} == expected
