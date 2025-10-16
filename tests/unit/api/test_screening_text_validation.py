"""Empty screening requests must be rejected before invoking the pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main


@pytest.mark.parametrize("path", ["/normalize", "/process", "/process-batch", "/search"])
@pytest.mark.parametrize("text", ["", " \t\r\n", "\u200b\u2066", "\x01", "\u200b \n\u2066"],
                         ids=["empty", "whitespace", "format", "control", "mixed"])
def test_effectively_empty_request_does_not_dispatch(monkeypatch, path, text):
    service = SimpleNamespace(process=AsyncMock(), process_batch=AsyncMock(),
                              search_service=object(), enable_search=True)
    monkeypatch.setattr(main, "orchestrator", service)
    key = "texts" if path == "/process-batch" else "query" if path == "/search" else "text"
    response = TestClient(main.app).post(path, json={key: [text] if key == "texts" else text})
    assert response.status_code == 422, response.text
    service.process.assert_not_called()
    service.process_batch.assert_not_called()
    assert "decision" not in response.json()


@pytest.mark.parametrize("model,key", [(main.ProcessTextRequest, "text"),
    (main.ProcessBatchRequest, "texts"), (main.SearchRequest, "query")])
def test_visible_input_keeps_formatting_for_evidence_offsets(model, key):
    text = "Example Exa\u200bmple INN 12345\u206667890"
    value = [text] if key == "texts" else text
    request = model(**{key: value})
    assert getattr(request, key) == value
