from unittest.mock import AsyncMock

import pytest

from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.contracts import SearchMode, SearchOpts
from ai_service.layers.search.elasticsearch_adapters import ElasticsearchACAdapter, ElasticsearchVectorAdapter


def _response(doc_id, score, text, country):
    return {"timed_out": False, "_shards": {"failed": 0}, "hits": {"max_score": score,
        "hits": [{"_id": doc_id, "_score": score, "_source": {"normalized_text": text,
            "entity_type": "person", "country": country, "metadata": {"country": country}}}]}}


@pytest.mark.asyncio
async def test_ac_adapter_builds_query_and_parses_results(monkeypatch):
    config = HybridSearchConfig()
    adapter = ElasticsearchACAdapter(config)
    client = AsyncMock()
    client.search.return_value = _response("doc-1", 1.2, "Source Example", "RU")
    monkeypatch.setattr(adapter, "_ensure_connection", AsyncMock(return_value=client))
    opts = SearchOpts(top_k=5, metadata_filters={"country": "RU"})
    results = await adapter.search("Source Example", opts)
    call = client.search.await_args.kwargs
    assert call["index"] == config.elasticsearch.ac_index
    body = call["body"]
    should = body["query"]["bool"]["should"]
    assert any("multi_match" in clause for clause in should)
    assert any("pattern.keyword" in clause.get("term", {}) for clause in should)
    assert body["min_score"] == opts.ac_min_score
    assert {"term": {"country.keyword": "RU"}} in body["query"]["bool"]["filter"]
    assert len(results) == 1 and results[0].doc_id == "doc-1"
    assert results[0].search_mode == SearchMode.AC and results[0].metadata["country"] == "RU"


@pytest.mark.asyncio
async def test_vector_adapter_search(monkeypatch):
    config = HybridSearchConfig()
    adapter = ElasticsearchVectorAdapter(config)
    client = AsyncMock()
    client.search.return_value = _response("vec-1", 0.9, "Source Example", "US")
    monkeypatch.setattr(adapter, "_ensure_connection", AsyncMock(return_value=client))
    opts = SearchOpts(top_k=3, metadata_filters={"country": "US", "id": "vec-1"})
    vector = [0.1] * config.vector_dimension
    results = await adapter.search(vector, opts)
    call = client.search.await_args.kwargs
    assert call["index"] == config.elasticsearch.vector_index
    body = call["body"]
    assert body["knn"]["field"] == config.vector_search.vector_field
    assert body["knn"]["query_vector"] == vector and body["knn"]["k"] == 3
    assert body["knn"]["num_candidates"] >= 3
    assert "ef_search" not in body["knn"]
    assert {"ids": {"values": ["vec-1"]}} in body["knn"]["filter"]
    assert len(results) == 1 and results[0].doc_id == "vec-1"
    assert results[0].search_mode == SearchMode.VECTOR
    assert results[0].match_fields == [config.vector_search.vector_field]
    assert results[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_vector_adapter_validates_dimension(monkeypatch):
    adapter = ElasticsearchVectorAdapter(HybridSearchConfig())
    connection = AsyncMock()
    monkeypatch.setattr(adapter, "_ensure_connection", connection)
    with pytest.raises(ValueError, match="dimension"):
        await adapter.search([0.1, 0.2], SearchOpts(top_k=2))
    connection.assert_not_awaited()
