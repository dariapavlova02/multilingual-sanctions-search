"""Malformed deployment settings must not silently change screening or TLS."""

import json

import pytest

from ai_service.config.settings import SearchConfig
from ai_service.layers.search.config import ElasticsearchConfig, HybridSearchConfig
from ai_service.layers.search.elasticsearch_client import build_client_kwargs


@pytest.mark.parametrize("key", ["ES_VERIFY_CERTS", "ES_RETRY_ON_TIMEOUT"])
@pytest.mark.parametrize("value", ["tru", "disabled", "", "  "])
def test_invalid_connection_boolean_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        ElasticsearchConfig.from_sources(env={key: value})


@pytest.mark.parametrize("key", ["ES_VERIFY_CERTS", "ENABLE_HYBRID_SEARCH",
    "ENABLE_ESCALATION", "ENABLE_FALLBACK", "ENABLE_EMBEDDING_CACHE"])
def test_invalid_legacy_search_boolean_is_rejected(monkeypatch, key):
    monkeypatch.setenv(key, "not-a-boolean")
    with pytest.raises(ValueError, match=key):
        SearchConfig()


@pytest.mark.parametrize("value, expected", [
    ("true", True), (" TRUE ", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), (" FALSE ", False), ("0", False), ("no", False), ("off", False),
])
def test_tls_boolean_forms_have_the_same_effect_in_both_loaders(monkeypatch, value, expected):
    monkeypatch.setenv("ES_VERIFY_CERTS", value)
    active = ElasticsearchConfig.from_sources(env={"ES_VERIFY_CERTS": value})
    legacy = SearchConfig().elasticsearch
    assert build_client_kwargs(active)["verify_certs"] is expected
    assert build_client_kwargs(legacy)["verify_certs"] is expected


@pytest.mark.parametrize("key", ["ES_HOSTS", "ELASTICSEARCH_HOSTS"])
@pytest.mark.parametrize("value", ["", "  ", "backend.invalid:9200,"])
def test_invalid_explicit_hosts_cannot_fall_back_to_yaml_or_default(key, value):
    with pytest.raises(ValueError):
        ElasticsearchConfig.from_sources({"hosts": ["previous.invalid:9200"]}, {key: value})


@pytest.mark.parametrize("key", ["ES_TIMEOUT", "ES_MAX_RETRIES", "ES_SMOKE_TEST_TIMEOUT"])
@pytest.mark.parametrize("value", ["", "invalid-setting-marker"])
def test_explicit_invalid_numeric_connection_setting_is_rejected_without_echo(key, value):
    with pytest.raises(ValueError) as error:
        ElasticsearchConfig.from_sources(env={key: value})
    assert key in str(error.value)
    if value:
        assert value not in str(error.value)


@pytest.mark.parametrize("value", ["", "invalid", "nan"])
def test_invalid_vector_dimension_override_cannot_fall_back(value):
    with pytest.raises(ValueError, match="ES_VECTOR_DIMENSION"):
        HybridSearchConfig.from_env(env={"ES_VECTOR_DIMENSION": value})


@pytest.mark.parametrize("value", ["", "normalized_text", "normalized_text:bad",
    "normalized_text:2,aliases:bad", "normalized_text:2,", "normalized_text:2,normalized_text:3",
    "normalized_text:nan", "normalized_text:inf", "normalized_text:-1"])
def test_invalid_field_weights_cannot_be_partially_applied(value):
    with pytest.raises(ValueError):
        HybridSearchConfig.from_env(env={"ES_AC_FIELD_WEIGHTS": value})


@pytest.mark.parametrize("via_environment", [True, False])
def test_explicit_missing_settings_file_is_an_error(tmp_path, via_environment):
    missing = tmp_path / "missing.yaml"
    with pytest.raises((ValueError, OSError)):
        if via_environment:
            HybridSearchConfig.from_env(env={"AI_SEARCH_SETTINGS_PATH": str(missing)})
        else:
            HybridSearchConfig.from_env(env={}, settings_path=missing)


@pytest.mark.parametrize("payload", [[], {"search": []}, {"search": None}, {"search": {"elasticsearch": []}}])
def test_invalid_yaml_structure_is_rejected(tmp_path, payload):
    path = tmp_path / "settings.yaml"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        HybridSearchConfig.from_env(env={}, settings_path=path)


@pytest.mark.parametrize("scheme", ["ftp", "https://", ""])
def test_explicit_invalid_connection_scheme_is_rejected(scheme):
    with pytest.raises(ValueError):
        ElasticsearchConfig.from_sources(env={"ES_SCHEME": scheme})


def test_valid_overrides_preserve_other_yaml_settings_and_source(tmp_path):
    payload = {"search": {"elasticsearch": {"hosts": ["https://source.invalid:9243"],
        "timeout": 17, "verify_certs": True}, "ac_search": {"min_score": 0.8},
        "vector_search": {"vector_dimension": 512}}}
    path = tmp_path / "settings.yaml"
    raw = json.dumps(payload)
    path.write_text(raw)
    config = HybridSearchConfig.from_env(settings_path=path, env={
        "ES_TIMEOUT": "23", "ES_VERIFY_CERTS": "yes", "ES_VECTOR_DIMENSION": "384",
        "ES_AC_FIELD_WEIGHTS": "normalized_text:2,aliases:0"})
    assert config.elasticsearch.hosts == ["https://source.invalid:9243"]
    assert config.elasticsearch.timeout == 23 and config.elasticsearch.verify_certs is True
    assert config.vector_search.vector_dimension == 384
    assert config.ac_search.field_weights == {"normalized_text": 2.0, "aliases": 0.0}
    assert config.ac_search.min_score == 0.8
    assert path.read_text() == raw


def test_rejected_tls_reload_preserves_verified_connection(monkeypatch):
    config = SearchConfig(es_hosts=["https://previous.invalid:9243"], es_verify_certs=True)
    previous = config.model_dump()
    monkeypatch.setenv("ES_VERIFY_CERTS", "typo")
    monkeypatch.setenv("ES_HOSTS", "replacement.invalid:9200")
    with pytest.raises(ValueError):
        config._on_config_changed()
    assert config.model_dump() == previous
    assert config.get_reload_stats()["reload_count"] == 0
