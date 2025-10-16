"""Deployment configuration must reach clients without silent conflicts or mutation."""

import json
from unittest.mock import AsyncMock

import pytest

from ai_service.layers.search.config import ElasticsearchConfig, HybridSearchConfig
from ai_service.layers.search.elasticsearch_adapters import ElasticsearchACAdapter, ElasticsearchVectorAdapter
from ai_service.layers.search.elasticsearch_client import ElasticsearchClientFactory, build_client_kwargs
from ai_service.layers.search.hybrid_search_service import HybridSearchService


@pytest.mark.parametrize("payload", [
    {"elasticsearch": {"host": ["other.invalid:9200"]}},
    {"elasticsearch": {"verify_cert": False}},
    {"ac_search": {"min_scor": 0.8}},
    {"vector_search": {"vector_dimensions": 512}},
    {"enable_escalations": False},
    {"enable_elasticsearch_auth": True},
    {"es_auth_type": "basic"},
])
def test_unknown_or_obsolete_settings_are_not_silently_ignored(payload):
    with pytest.raises(ValueError):
        HybridSearchConfig(**payload)


def test_connection_aliases_reach_the_canonical_client_options():
    config = HybridSearchConfig(es_hosts=["https://other.invalid:9243"], es_timeout=7,
        es_username="fixture-user", es_password="fixture-password", es_ca_certs="/tmp/fixture-ca.pem",
        enable_ssl_verification=False)
    kwargs = build_client_kwargs(config.elasticsearch)
    assert kwargs["hosts"] == ["https://other.invalid:9243"]
    assert kwargs["request_timeout"] == 7
    assert kwargs["basic_auth"] == ("fixture-user", "fixture-password")
    assert kwargs["verify_certs"] is False and kwargs["ca_certs"] == "/tmp/fixture-ca.pem"
    assert not any(key in config.model_dump() for key in ("es_hosts", "es_password", "enable_ssl_verification"))


def test_api_key_alias_reaches_the_client_without_basic_auth():
    config = HybridSearchConfig(es_api_key="fixture-api-key")
    kwargs = build_client_kwargs(config.elasticsearch)
    assert kwargs["api_key"] == "fixture-api-key" and "basic_auth" not in kwargs


@pytest.mark.parametrize("alias, field, old, new", [
    ("es_hosts", "hosts", ["old.invalid:9200"], ["new.invalid:9200"]),
    ("es_timeout", "timeout", 8, 9),
    ("es_api_key", "api_key", "old-marker", "new-marker"),
    ("enable_ssl_verification", "verify_certs", True, False),
])
def test_conflicting_connection_aliases_fail_without_echoing_values(alias, field, old, new):
    with pytest.raises(ValueError) as error:
        HybridSearchConfig(**{alias: old, "elasticsearch": {field: new}})
    assert alias in str(error.value)
    assert "old-marker" not in str(error.value) and "new-marker" not in str(error.value)


@pytest.mark.parametrize("data", [
    {"username": "fixture-user"}, {"password": "fixture-password"},
    {"username": "", "password": "fixture-password"},
    {"username": "fixture-user", "password": ""}, {"api_key": ""},
    {"username": "fixture-user", "password": "fixture-password", "api_key": "fixture-api-key"},
])
def test_incomplete_or_ambiguous_credentials_cannot_be_discarded(data):
    with pytest.raises(ValueError):
        ElasticsearchConfig(**data)


@pytest.mark.parametrize("field", ["elasticsearch", "ac_search", "vector_search"])
@pytest.mark.parametrize("value", [None, [], "wrong-shape"])
def test_direct_invalid_nested_shapes_are_validation_errors(field, value):
    with pytest.raises(ValueError):
        HybridSearchConfig(**{field: value})


def test_environment_overrides_are_applied_after_translating_yaml_aliases(tmp_path):
    path = tmp_path / "search.yaml"
    data = {"search": {"es_hosts": ["old.invalid:9200"], "es_timeout": 7,
        "vector_dimension": 512, "es_api_key": "fixture-api-key"}}
    raw = json.dumps(data)
    path.write_text(raw)
    config = HybridSearchConfig.from_env(settings_path=path, env={
        "ES_HOSTS": "https://new.invalid:9243", "ES_TIMEOUT": "11", "ES_VECTOR_DIMENSION": "384"})
    assert config.elasticsearch.hosts == ["https://new.invalid:9243"]
    assert config.elasticsearch.timeout == 11 and config.vector_dimension == 384
    assert build_client_kwargs(config.elasticsearch)["api_key"] == "fixture-api-key"
    assert path.read_text() == raw


@pytest.mark.parametrize("factory", [HybridSearchService, ElasticsearchClientFactory,
    ElasticsearchACAdapter, ElasticsearchVectorAdapter])
def test_mutating_the_callers_config_cannot_change_a_constructed_service(factory):
    config = HybridSearchConfig(elasticsearch={"hosts": ["https://original.invalid:9243"]})
    service = factory(config)
    config.elasticsearch.hosts[0] = "https://changed.invalid:9243"
    config.elasticsearch.verify_certs = False
    config.ac_search.field_weights["normalized_text"] = 0
    assert service.config.elasticsearch.hosts == ["https://original.invalid:9243"]
    assert service.config.elasticsearch.verify_certs is True
    assert service.config.ac_search.field_weights["normalized_text"] == 2.0


@pytest.mark.parametrize("factory", [HybridSearchService, ElasticsearchClientFactory,
    ElasticsearchACAdapter, ElasticsearchVectorAdapter])
def test_mutated_invalid_configuration_is_revalidated_at_construction(factory):
    config = HybridSearchConfig()
    config.elasticsearch.hosts.clear()
    with pytest.raises(ValueError):
        factory(config)


@pytest.mark.parametrize("adapter", [ElasticsearchACAdapter, ElasticsearchVectorAdapter])
def test_injected_client_factory_cannot_silently_use_another_connection(adapter):
    client = ElasticsearchClientFactory(HybridSearchConfig(
        elasticsearch={"hosts": ["https://unrelated.invalid:9243"]}))
    with pytest.raises(ValueError, match="connection settings differ"):
        adapter(HybridSearchConfig(), client_factory=client)


def test_empty_environment_credentials_can_explicitly_switch_to_api_key_auth():
    config = ElasticsearchConfig.from_sources({"username": "old-user", "password": "old-password"},
        {"ES_USERNAME": "", "ES_PASSWORD": "", "ES_API_KEY": "new-fixture-api-key"})
    kwargs = build_client_kwargs(config)
    assert kwargs["api_key"] == "new-fixture-api-key" and "basic_auth" not in kwargs


@pytest.mark.asyncio
async def test_default_client_keeps_all_configured_nodes(monkeypatch):
    created = []
    def make_client(**kwargs):
        created.append(kwargs)
        return AsyncMock()
    monkeypatch.setattr("ai_service.layers.search.elasticsearch_client.AsyncElasticsearch", make_client)
    factory = ElasticsearchClientFactory(HybridSearchConfig(
        elasticsearch={"hosts": ["first.invalid:9200", "second.invalid:9200"]}))
    try:
        pooled = await factory.get_client()
        assert await factory.get_client() is pooled
        assert created[0]["hosts"] == ["http://first.invalid:9200", "http://second.invalid:9200"]
        specific = await factory.get_client("http://second.invalid:9200")
        assert specific is not pooled
        assert created[1]["hosts"] == ["http://second.invalid:9200"]
    finally:
        await factory.close()


@pytest.mark.asyncio
async def test_explicit_client_host_must_belong_to_the_configured_cluster():
    factory = ElasticsearchClientFactory(HybridSearchConfig())
    try:
        with pytest.raises(ValueError):
            await factory.get_client("http://unrelated.invalid:9200")
    finally:
        await factory.close()
