"""Small, explicit clients for search service contract tests; no live model or ES."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from ai_service.config import EmbeddingConfig
from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import Candidate, SearchMode
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.index_schema import index_mapping, SOURCE_COVERAGE_VERSION


def normalized():
    return NormalizationResult(
        normalized="Example Person", tokens=["Example", "Person"],
        trace=[], language="en", confidence=1.0,
    )


def candidate(doc_id="active", score=0.9, mode=SearchMode.AC):
    return Candidate(
        doc_id, score, "Example Person", "person",
        {"entity_id": doc_id, "source": "active-source", "dob": "1980-01-01"},
        mode, ["name"], score,
    )


def search_service(*, fallback=True):
    config = HybridSearchConfig(
        elasticsearch={"ac_index": "test_ac", "vector_index": "test_vector"},
        enable_fallback=fallback,
    )
    service = HybridSearchService(config)
    mappings = {}
    for name, vectors in [("test_ac", False), ("test_vector", True)]:
        mapping = index_mapping(config, vectors=vectors)["mappings"]
        mapping["_meta"].update(
            ingestion_status="completed", generation="snapshot-one",
            source_manifest={"active-source": "fixture-digest"},
            source_coverage_version=SOURCE_COVERAGE_VERSION,
        )
        mappings[name] = {name: {"mappings": mapping}}

    async def get_mapping(*, index):
        return deepcopy(mappings[index])

    client = SimpleNamespace(
        indices=SimpleNamespace(get_mapping=AsyncMock(side_effect=get_mapping)),
        count=AsyncMock(return_value={"count": 3}),
    )
    # The Elasticsearch options method is synchronous and returns a client.
    client.options = Mock(return_value=client)
    service._client_factory = SimpleNamespace(
        get_client=AsyncMock(return_value=client),
        get_connection_stats=AsyncMock(return_value={"active_connections": 1}),
    )

    async def no_documents(*args, **kwargs):
        yield []

    def adapter():
        return SimpleNamespace(
            _connected=True, search=AsyncMock(return_value=[]),
            iter_documents=no_documents,
            health_check=AsyncMock(return_value={"status": "healthy", "connected": True}),
        )

    service._ac_adapter = adapter()
    service._vector_adapter = adapter()
    service._embedding_service = SimpleNamespace(
        config=EmbeddingConfig(), encode_one_async=AsyncMock(return_value=[0.1] * 384),
    )
    # These are deliberately a different source. No public request may consult them.
    service._fallback_watchlist_service = SimpleNamespace(
        ready=Mock(return_value=True), search=Mock(return_value=[("local", 0.99, {})]),
        get_doc=Mock(return_value=SimpleNamespace(
            full_name="Unrelated Local Person", entity_type="person", extras={},
        )),
    )
    service._fallback_vector_service = SimpleNamespace(search=Mock(return_value=[]))
    service._initialized = True
    return service


def assert_no_local_search(service):
    service._fallback_watchlist_service.search.assert_not_called()
    service._fallback_vector_service.search.assert_not_called()
