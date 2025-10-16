from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.api import admin_endpoints
from ai_service.layers.search.config import ElasticsearchConfig, HybridSearchConfig
from ai_service.layers.search.contracts import SearchOpts
from ai_service.layers.search.elasticsearch_client import build_client_kwargs
from ai_service.layers.search.fuzzy_search_service import (
    FuzzyConfig,
    FuzzySearchService,
)
from ai_service.layers.search.hybrid_search_service import HybridSearchService


def test_yaml_connection_survives_absent_environment_overrides():
    config = ElasticsearchConfig.from_sources(
        {"hosts": ["https://example.invalid:9243"]}, {}
    )
    assert config.hosts == ["https://example.invalid:9243"]
    assert build_client_kwargs(config)["verify_certs"] is True


def test_search_cache_distinguishes_escalation_normalization_and_data_generation():
    service = HybridSearchService(HybridSearchConfig())
    normal = service._generate_search_cache_key("Example", SearchOpts())
    assert normal != service._generate_search_cache_key(
        "Example", SearchOpts(enable_escalation=False)
    )
    assert normal != service._generate_search_cache_key(
        "Example", SearchOpts(), normalized="Different"
    )
    assert normal != service._generate_search_cache_key(
        "Example", SearchOpts(), dataset_version={"ac": "new"}
    )


@pytest.mark.asyncio
async def test_fuzzy_matching_does_not_drop_names_after_first_batch():
    service = FuzzySearchService(FuzzyConfig(max_candidates=2))
    results = await service.search_async(
        "Unique Example", ["Other", "Unrelated", "Unique Example"]
    )
    assert any(result.matched_text == "Unique Example" for result in results)


@pytest.mark.asyncio
async def test_bulk_item_failure_is_visible_and_client_is_closed(monkeypatch):
    client = SimpleNamespace(
        bulk=AsyncMock(
            return_value={
                "errors": True,
                "items": [
                    {"index": {"status": 201}},
                    {
                        "index": {
                            "status": 400,
                            "error": {"type": "mapper_parsing_exception"},
                        }
                    },
                ],
            }
        ),
        indices=SimpleNamespace(refresh=AsyncMock()),
    )
    wrapper = SimpleNamespace(client=client, close=AsyncMock())
    monkeypatch.setattr(admin_endpoints, "ElasticsearchClient", lambda: wrapper)
    monkeypatch.setattr(admin_endpoints, "ensure_index", AsyncMock())
    monkeypatch.setattr(admin_endpoints, "set_ingestion_status", AsyncMock())
    await admin_endpoints._load_ac_patterns_background(
        [
            {"pattern": "Example One", "entity_id": "one"},
            {"pattern": "Example Two", "entity_id": "two"},
        ],
        "person",
        "tier_0_exact",
        2,
    )
    status = admin_endpoints.loading_status["ac_patterns"]
    assert status["status"] == "error"
    assert status["progress"] == 1
    assert status["failed"] == 1
    wrapper.close.assert_awaited_once()
    client.indices.refresh.assert_not_awaited()


def test_schema_metadata_cannot_hide_missing_search_fields():
    import json
    from ai_service.layers.search.index_schema import index_mapping, validate_mapping

    expected = index_mapping(HybridSearchConfig())["mappings"]
    corrupted = json.loads(json.dumps(expected))
    del corrupted["properties"]["pattern"]["fields"]["keyword"]
    with pytest.raises(ValueError, match="pattern.fields.keyword"):
        validate_mapping(corrupted, expected)


def test_zero_source_identifier_is_preserved():
    from ai_service.layers.search.index_schema import pattern_document

    _, document = pattern_document(
        {"pattern": "Source Entity", "entity_id": 0}, "person", 0
    )
    assert document["entity_id"] == "0"


@pytest.mark.asyncio
async def test_search_retry_does_not_retry_programming_errors():
    from ai_service.layers.search.elasticsearch_adapters import retry_elasticsearch

    calls = 0

    @retry_elasticsearch(delay=0)
    async def operation():
        nonlocal calls
        calls += 1
        raise ValueError("Invalid search contract")

    with pytest.raises(ValueError, match="Invalid search contract"):
        await operation()
    assert calls == 1


def test_mapping_accepts_elasticsearch_implicit_object_type():
    import json
    from ai_service.layers.search.index_schema import index_mapping, validate_mapping

    expected = index_mapping(HybridSearchConfig())["mappings"]
    actual = json.loads(json.dumps(expected))
    actual["properties"]["metadata"] = {"properties": {"source": {"type": "keyword"}}}
    validate_mapping(actual, expected)


def test_rejected_environment_reload_preserves_the_complete_previous_configuration(monkeypatch):
    from ai_service.config.settings import SearchConfig
    config = SearchConfig(es_hosts=['previous.invalid:9200'], es_timeout=30)
    before = config.model_dump()
    monkeypatch.setenv('ES_HOSTS', 'replacement.invalid:9200')
    monkeypatch.setenv('ES_TIMEOUT', '-1')
    with pytest.raises(ValueError):
        config._on_config_changed()
    assert config.model_dump() == before
    assert config.get_reload_stats()['reload_count'] == 0


def test_explicit_tls_url_and_ipv6_are_validated_without_exposing_credentials():
    from ai_service.config.settings import SearchConfig
    from ai_service.layers.search.config import ElasticsearchConfig
    hosts = ['https://example.invalid:9243', 'http://[::1]:9200']
    assert SearchConfig(es_hosts=hosts).es_hosts == hosts
    assert ElasticsearchConfig(hosts=hosts).hosts == hosts
    with pytest.raises(ValueError, match='credentials separately'):
        ElasticsearchConfig(hosts=['https://user:secret@example.invalid:9243'])


def test_vector_compatibility_fields_cannot_silently_disagree_with_canonical_fields():
    with pytest.raises(ValueError, match='Conflicting vector'):
        HybridSearchConfig(vector_dimension=512, vector_search={'vector_dimension': 384})


@pytest.mark.asyncio
async def test_pending_configuration_is_validated_and_copied_before_initialization():
    service = HybridSearchService(HybridSearchConfig())
    replacement = HybridSearchConfig(elasticsearch={'hosts': ['https://example.invalid:9243']})
    await service.update_configuration(replacement)
    replacement.elasticsearch.hosts.append('http://unwanted.invalid:9200')
    assert service.config.elasticsearch.hosts == ['https://example.invalid:9243']
    assert service._client_factory is None


@pytest.mark.asyncio
async def test_invalid_pending_configuration_preserves_previous_settings():
    service = HybridSearchService(HybridSearchConfig())
    previous = service.config
    invalid = HybridSearchConfig()
    invalid.vector_search.vector_dimension = 0
    with pytest.raises(ValueError, match='vector_dimension'):
        await service.update_configuration(invalid)
    assert service.config is previous


@pytest.mark.asyncio
async def test_live_config_update_preserves_all_adapters_clients_and_caches():
    service = HybridSearchService(HybridSearchConfig())
    service.initialize()
    before = (service.config, service._client_factory, service._ac_adapter, service._vector_adapter)
    service._search_cache['sentinel'] = object()
    cached = service._search_cache['sentinel']
    with pytest.raises(RuntimeError, match='recreating'):
        await service.update_configuration(HybridSearchConfig())
    after = (service.config, service._client_factory, service._ac_adapter, service._vector_adapter)
    assert all(old is new for old, new in zip(before, after))
    assert service._search_cache['sentinel'] is cached
    await service.close()
