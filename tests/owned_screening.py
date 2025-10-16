"""Explicit ownership and isolated namespaces for tests that write to Elasticsearch."""

import os
import re
from types import SimpleNamespace
from urllib.parse import urlsplit
import uuid

import pytest

from ai_service.api.elasticsearch_wrapper import ElasticsearchClient
from ai_service.layers.search.config import HybridSearchConfig


def validate_owned_target(url, cluster_name):
    """Allow only an explicitly named runner cluster on a local TCP endpoint."""
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port is None or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or not re.fullmatch(r"sanctions-regression-[0-9a-f]{32}", cluster_name)):
        raise ValueError("Use the isolated loopback cluster created by scripts/run_regression_gate.py")


@pytest.fixture
async def owned_elasticsearch(monkeypatch, tmp_path):
    required = ["SANCTIONS_TEST_ES_URL", "SANCTIONS_TEST_ES_USERNAME",
                "SANCTIONS_TEST_ES_PASSWORD", "SANCTIONS_TEST_CLUSTER_NAME"]
    if not all(os.getenv(key) for key in required):
        pytest.fail("A disposable owned cluster is required: use scripts/run_regression_gate.py")
    validate_owned_target(os.environ["SANCTIONS_TEST_ES_URL"], os.environ["SANCTIONS_TEST_CLUSTER_NAME"])
    # Clear competing explicit index/auth/file overrides before setting the test namespace.
    for key in ("ES_AC_INDEX", "ES_VECTOR_INDEX", "ES_API_KEY", "ES_CA_CERTS", "AI_SEARCH_SETTINGS_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ES_HOSTS", os.environ["SANCTIONS_TEST_ES_URL"])
    monkeypatch.setenv("ES_USERNAME", os.environ["SANCTIONS_TEST_ES_USERNAME"])
    monkeypatch.setenv("ES_PASSWORD", os.environ["SANCTIONS_TEST_ES_PASSWORD"])
    monkeypatch.setenv("ES_INDEX_PREFIX", "regression_" + uuid.uuid4().hex)
    monkeypatch.setenv("APP_STATE_DIR", str(tmp_path))
    config = HybridSearchConfig.from_env()
    async with ElasticsearchClient() as wrapper:
        info = await wrapper.client.info()
        assert info["cluster_name"] == os.environ["SANCTIONS_TEST_CLUSTER_NAME"], "Unowned test cluster"
        try:
            yield SimpleNamespace(config=config, client=wrapper.client)
        finally:
            for index in (config.elasticsearch.ac_index, config.elasticsearch.vector_index):
                if await wrapper.client.indices.exists(index=index):
                    await wrapper.client.indices.delete(index=index)


@pytest.fixture(scope="session")
def screening_embeddings():
    """The pinned real model; workflow tests never synthesize query vectors."""
    from ai_service.config import EmbeddingConfig
    from ai_service.layers.embeddings.embedding_service import EmbeddingService
    service = EmbeddingService(EmbeddingConfig(device="cpu"))
    yield service
    service.close()


@pytest.fixture
async def active_screening(owned_elasticsearch, screening_embeddings):
    """Build a complete synthetic snapshot through the production ingestion code."""
    from ai_service.api.admin_endpoints import _load_documents, loading_status
    from ai_service.layers.search.index_schema import pattern_document, vector_document, embedding_contract
    from ai_service.layers.search.hybrid_search_service import HybridSearchService

    rows = [
        {"pattern": "Replacement Example", "canonical": "Replacement Example", "entity_id": "primary",
         "source_list": "synthetic-regression", "entity_type": "person", "country": "UA",
         "metadata": {"tax_id": "1234567890", "dob": "1980-01-01"}},
        {"pattern": "Replacement Alias", "canonical": "Replacement Example", "entity_id": "primary",
         "source_list": "synthetic-regression", "entity_type": "person", "country": "UA",
         "metadata": {"tax_id": "1234567890", "dob": "1980-01-01"}},
        {"pattern": "Distant Other Entity", "entity_id": "secondary", "source_list": "synthetic-regression",
         "entity_type": "person", "country": "GB", "metadata": {"tax_id": "001234567890", "dob": "1990-01-01"}},
        {"pattern": "Example Trading LLC", "entity_id": "company", "source_list": "synthetic-regression",
         "entity_type": "organization", "country": "GB", "metadata": {}},
    ]
    categories = ["company" if row["entity_type"] == "organization" else "person" for row in rows]
    documents = [pattern_document(row, category, 0) for row, category in zip(rows, categories)]
    await _load_documents(documents, "ac_patterns")
    assert loading_status["ac_patterns"]["status"] == "completed", loading_status["ac_patterns"]
    embeddings = await screening_embeddings.encode_batch_async([row["pattern"] for row in rows])
    contract = embedding_contract()
    vectors = [vector_document({**row, "name": row["pattern"], "vector": vector,
        "embedding_contract": contract}, category, contract["model_name"])
        for row, vector, category in zip(rows, embeddings, categories)]
    await _load_documents(vectors, "vectors", vectors=True)
    assert loading_status["vectors"]["status"] == "completed", loading_status["vectors"]
    assert loading_status["vectors"]["snapshot_ready"] is True, loading_status["vectors"]
    service = HybridSearchService(owned_elasticsearch.config)
    service._embedding_service = screening_embeddings
    generations = await service.readiness()
    try:
        yield SimpleNamespace(**vars(owned_elasticsearch), search=service,
            embeddings=screening_embeddings, rows=rows, generations=generations)
    finally:
        await service.close()


@pytest.fixture
async def screening_api(active_screening, monkeypatch):
    """Real ASGI routes, services, pinned embeddings and owned Elasticsearch."""
    import httpx
    import ai_service.main as main
    from ai_service.core.orchestrator_factory import OrchestratorFactory
    from ai_service.config.settings import DecisionConfig
    from ai_service.core.decision_engine import DecisionEngine

    orchestrator = await OrchestratorFactory.create_orchestrator(
        enable_smart_filter=True, enable_variants=True, enable_embeddings=True,
        enable_decision_engine=True, enable_search=True, allow_smart_filter_skip=False,
        search_service=active_screening.search, embeddings_service=active_screening.embeddings,
        decision_engine=DecisionEngine(DecisionConfig()),
    )
    monkeypatch.setattr(main, "orchestrator", orchestrator)
    # This fixture checks dependencies explicitly; ASGITransport has no server lifespan.
    from ai_service.api.runtime_health import initialize_runtime_models
    await initialize_runtime_models(orchestrator)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://screening.test") as client:
        try:
            yield SimpleNamespace(**vars(active_screening), api=client, orchestrator=orchestrator)
        finally:
            orchestrator.variants_service.close()
