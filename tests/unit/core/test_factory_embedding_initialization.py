"""The production factory must await initialization and propagate its failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ai_service.core.orchestrator_factory import OrchestratorFactory
from ai_service.exceptions import ServiceInitializationError


@pytest.mark.parametrize("initialization_fails", [False, True])
async def test_factory_awaits_embedding_initialization(monkeypatch, initialization_fails):
    initialize = AsyncMock(side_effect=RuntimeError("model initialization failed") if initialization_fails else None)
    embedding = SimpleNamespace(initialize=initialize)
    monkeypatch.setattr("ai_service.layers.embeddings.embedding_service.EmbeddingService", Mock(return_value=embedding))
    arguments = dict(enable_embeddings=True, enable_smart_filter=False,
        validation_service=Mock(), language_service=Mock(), unicode_service=Mock(),
        normalization_service=Mock(), signals_service=Mock())
    if initialization_fails:
        with pytest.raises(ServiceInitializationError, match="embedding service could not initialize"):
            await OrchestratorFactory.create_orchestrator(**arguments)
    else:
        orchestrator = await OrchestratorFactory.create_orchestrator(**arguments)
        assert orchestrator.embeddings_service is embedding and orchestrator.enable_embeddings
    initialize.assert_awaited_once()
