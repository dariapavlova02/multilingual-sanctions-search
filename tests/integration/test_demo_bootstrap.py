"""The documented demo must traverse real ingestion, models and HTTP search."""

from pathlib import Path

import pytest

from ai_service.scripts.bootstrap import bootstrap

pytestmark = [pytest.mark.integration, pytest.mark.docker]


async def test_demo_snapshot_is_searchable_and_refuses_implicit_replacement(
    screening_api,
):
    demo = Path(__file__).resolve().parents[2] / "examples" / "demo"
    # Replace only the fixture-owned synthetic indices, never a user's cluster.
    await bootstrap(
        ingest=True, vectors=True, data_dir=demo, batch_size=2, replace=True
    )
    assert (await screening_api.api.get("/health/ready")).status_code == 200

    for query, entity_id in [
        ("John Smith", "demo-person-1"),
        ("John A. Smith", "demo-person-1"),
        ("Ivan Petrenko", "demo-person-2"),
        ("Example Trading LLC", "demo-company-1"),
    ]:
        response = await screening_api.api.post(
            "/search", json={"query": query, "search_mode": "ac", "top_k": 10}
        )
        assert response.status_code == 200, response.text
        matches = response.json()["results"]
        assert any(
            hit["metadata"]["entity_id"] == entity_id for hit in matches
        ), response.text
        assert all(hit["metadata"]["source"] == "custom" for hit in matches)
        assert all(hit["metadata"]["synthetic"] is True for hit in matches)

    for mode in ("vector", "hybrid"):
        response = await screening_api.api.post(
            "/search", json={"query": "John Smith", "search_mode": mode, "top_k": 10}
        )
        assert response.status_code == 200, response.text
        assert any(
            hit["metadata"]["entity_id"] == "demo-person-1"
            for hit in response.json()["results"]
        ), response.text

    with pytest.raises(ValueError, match="Existing snapshot is nonempty"):
        await bootstrap(ingest=True, data_dir=demo)
    assert (await screening_api.api.get("/health/ready")).status_code == 200
