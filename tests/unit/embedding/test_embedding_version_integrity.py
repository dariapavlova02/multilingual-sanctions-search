"""Model identity must survive configuration, loading and cached retrieval."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from ai_service.config import EmbeddingConfig
from ai_service.layers.embeddings.embedding_service import EmbeddingService
from ai_service.layers.search.contracts import SearchMode, SearchOpts
from tests.search_service_support import normalized, search_service

ALTERNATIVE = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATIVE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch):
    for key in ("EMBEDDING_MODEL", "EMBEDDING_MODEL_REVISION", "ES_VECTOR_DIMENSION"):
        monkeypatch.delenv(key, raising=False)


def test_alternative_resolves_its_own_pin():
    config = EmbeddingConfig(model_name=ALTERNATIVE)
    assert config.revision == ALTERNATIVE_REVISION
    assert config.revision != EmbeddingConfig().revision


def test_environment_model_resolves_its_own_pin(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", ALTERNATIVE)
    assert EmbeddingConfig().revision == ALTERNATIVE_REVISION


def test_explicit_model_does_not_borrow_environment_pair(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", EmbeddingConfig().revision)
    monkeypatch.setenv("ES_VECTOR_DIMENSION", "768")
    config = EmbeddingConfig(model_name=ALTERNATIVE)
    assert (config.revision, config.dimension) == (ALTERNATIVE_REVISION, 384)


@pytest.mark.parametrize("revision", ["main", "v1", "", "a" * 39, "z" * 40])
def test_mutable_or_invalid_revision_rejected(revision):
    with pytest.raises(ValidationError):
        EmbeddingConfig(revision=revision)


@pytest.mark.parametrize("values", [
    {"model_name": "custom/model"},
    {"model_name": "custom/model", "revision": "a" * 40},
    {"model_name": "custom/model", "dimension": 768},
    {"model_name": "../local/model", "revision": "a" * 40, "dimension": 768},
    {"extra_models": ["custom/model"]},
    {"preprocessing_version": "unimplemented-v999"},
])
def test_unverifiable_configuration_rejected(values):
    with pytest.raises(ValidationError):
        EmbeddingConfig(**values)


def test_custom_model_requires_explicit_contract():
    config = EmbeddingConfig(model_name="custom/model", revision="a" * 40, dimension=768)
    assert (config.model_name, config.revision, config.dimension) == ("custom/model", "a" * 40, 768)


@pytest.mark.parametrize("field,value", [
    ("model_name", ALTERNATIVE), ("revision", "a" * 40),
    ("dimension", 768), ("preprocessing_version", "other"),
])
def test_running_configuration_is_immutable(field, value):
    config = EmbeddingConfig()
    with pytest.raises(ValidationError):
        setattr(config, field, value)


def test_loader_rejects_unapproved_model_before_download():
    service = EmbeddingService(EmbeddingConfig())
    with patch("sentence_transformers.SentenceTransformer") as loader:
        with pytest.raises(ValueError, match="allowlist"):
            service._load_model(ALTERNATIVE)
        loader.assert_not_called()
    service.close()


def test_allowlisted_alternative_is_pinned_and_dimension_checked():
    service = EmbeddingService(EmbeddingConfig(extra_models=[ALTERNATIVE]))
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 384
    with patch("sentence_transformers.SentenceTransformer", return_value=model) as loader:
        assert service._load_model(ALTERNATIVE) is model
        assert service._load_model(ALTERNATIVE) is model
        loader.assert_called_once()
        assert loader.call_args.kwargs["revision"] == ALTERNATIVE_REVISION
        assert loader.call_args.kwargs["trust_remote_code"] is False
        assert loader.call_args.kwargs["model_kwargs"] == {"use_safetensors": True}
    service.close()


def test_alternative_wrong_dimension_not_cached():
    service = EmbeddingService(EmbeddingConfig(extra_models=[ALTERNATIVE]))
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 768
    with patch("sentence_transformers.SentenceTransformer", return_value=model):
        with pytest.raises(ValueError, match="dimension"):
            service._load_model(ALTERNATIVE)
    assert not service.model_cache
    service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_cache", [False, True])
async def test_foreign_same_dimension_encoder_rejected_before_results(warm_cache):
    service = search_service()
    opts = SearchOpts(search_mode=SearchMode.VECTOR, timeout_ms=1000)
    if warm_cache:
        assert await service.find_candidates(normalized(), "Example Person", opts) == []
    service._embedding_service = SimpleNamespace(
        config=EmbeddingConfig(model_name=ALTERNATIVE),
        encode_one_async=AsyncMock(return_value=[0.2] * 384),
    )
    with pytest.raises((ValueError, RuntimeError), match="[Ee]mbedding.*contract"):
        await service.find_candidates(normalized(), "Example Person", opts)
    service._embedding_service.encode_one_async.assert_not_called()


@pytest.mark.asyncio
async def test_encoder_replacement_during_encoding_fails_closed():
    service = search_service()

    async def replace_encoder(text):
        service._embedding_service = SimpleNamespace(
            config=EmbeddingConfig(model_name=ALTERNATIVE),
            encode_one_async=AsyncMock(return_value=[0.2] * 384),
        )
        return [0.1] * 384

    service._embedding_service.encode_one_async.side_effect = replace_encoder
    with pytest.raises((ValueError, RuntimeError), match="[Ee]mbedding.*contract"):
        await service._build_query_vector(normalized(), "Example Person")
    assert not service._embedding_cache


@pytest.mark.asyncio
async def test_embedding_cache_does_not_expose_owned_vector():
    service = search_service()
    first = await service._build_query_vector(normalized(), "Example Person")
    first[0] = float("nan")
    second = await service._build_query_vector(normalized(), "Example Person")
    assert second[0] == 0.1
    service._embedding_service.encode_one_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_lexical_search_does_not_require_a_matching_encoder():
    service = search_service()
    service._embedding_service = SimpleNamespace(config=EmbeddingConfig(model_name=ALTERNATIVE))
    assert await service.find_candidates(
        normalized(), "Example Person", SearchOpts(search_mode=SearchMode.AC, timeout_ms=1000)
    ) == []


def test_batch_override_does_not_mutate_running_config():
    service = EmbeddingService(EmbeddingConfig(batch_size=64))
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.return_value = np.array([[0.1] * 384])
    with patch("sentence_transformers.SentenceTransformer", return_value=model):
        assert len(service.encode_batch(["Example Person"], batch_size=7)) == 1
        assert model.encode.call_args.kwargs["batch_size"] == 7
        service.encode_batch(["Example Person"])
        assert model.encode.call_args.kwargs["batch_size"] == 64
    assert service.config.batch_size == 64
    service.close()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_override_rejects_invalid_sizes(batch_size):
    service = EmbeddingService(EmbeddingConfig())
    with pytest.raises(ValueError, match="positive integer"):
        service.encode_batch(["Example Person"], batch_size=batch_size)
    assert not service.model_cache
    service.close()


def test_extra_model_list_cannot_change_after_construction():
    names = [ALTERNATIVE]
    config = EmbeddingConfig(extra_models=names)
    names.append("custom/unpinned")
    assert config.extra_models == (ALTERNATIVE,)


def test_service_config_cannot_be_replaced_or_retagged():
    service = EmbeddingService(EmbeddingConfig())
    original = service.embedding_contract
    with pytest.raises(AttributeError):
        service.config = EmbeddingConfig(model_name=ALTERNATIVE)
    exposed = service.embedding_contract
    exposed["revision"] = "a" * 40
    assert service.embedding_contract == original
    service.close()


@pytest.mark.asyncio
async def test_new_revision_of_same_model_is_rejected_by_existing_index():
    service = search_service()
    service._embedding_service.config = EmbeddingConfig(revision="a" * 40)
    with pytest.raises(RuntimeError, match="Embedding provider contract"):
        await service.readiness()


@pytest.mark.asyncio
async def test_environment_contract_change_requires_restart(monkeypatch):
    service = search_service()
    monkeypatch.setenv("EMBEDDING_MODEL", ALTERNATIVE)
    with pytest.raises(RuntimeError, match="restart and reindex"):
        await service.readiness()


@pytest.mark.asyncio
async def test_encoder_change_during_index_readiness_is_detected():
    service = search_service()
    client = await service._client_factory.get_client()

    async def change_encoder(**kwargs):
        service._embedding_service.config = EmbeddingConfig(model_name=ALTERNATIVE)
        return {"count": 3}

    client.count.side_effect = change_encoder
    with pytest.raises(RuntimeError, match="Embedding provider contract"):
        await service.readiness()


def test_vector_export_uses_batch_override_and_actual_contract(tmp_path):
    import json
    import importlib.util
    from pathlib import Path

    # Full collection also imports a different package named scripts. Load this
    # repository's CLI by its path, independent of collection/import order.
    source_path = Path(__file__).resolve().parents[3] / "scripts" / "generate_vectors.py"
    spec = importlib.util.spec_from_file_location("embedding_contract_vector_generator", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = module.VectorGenerator()
    source = tmp_path / "patterns.json"
    source.write_text(json.dumps({"patterns": [{
        "pattern": "Example Person", "tier": 0, "entity_id": "example", "entity_type": "person"
    }]}))
    target = tmp_path / "vectors.json"
    with patch.object(generator.service, "encode_batch", return_value=[[0.1] * 384]) as encode:
        assert generator.generate_vectors_from_patterns(source, target) == 1
        encode.assert_called_once_with(["Example Person"], batch_size=64)
    assert json.loads(target.read_text())[0]["embedding_contract"] == generator.service.embedding_contract
    generator.service.close()
