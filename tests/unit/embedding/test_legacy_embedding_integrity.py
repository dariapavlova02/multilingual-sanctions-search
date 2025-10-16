"""Legacy entry points must retain the production model and failure contracts."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from ai_service.config import EmbeddingConfig
from ai_service.layers.embeddings.optimized_embedding_service import OptimizedEmbeddingService
from ai_service.layers.embeddings.models.embedding_model_manager import EmbeddingModelManager
from ai_service.layers.embeddings.models.model_config import ModelConfig, get_model_config
from ai_service.layers.search.hybrid_search_service_refactored import HybridSearchServiceRefactored as LegacySearch
from ai_service.utils.inference_queue import InferenceUnavailableError
from tests.search_service_support import normalized


@pytest.fixture
def model():
    value = Mock()
    value.get_sentence_embedding_dimension.return_value = 384
    value.device = "cpu"
    value.max_seq_length = 128

    def encode(texts, **kwargs):
        scale = 1.0 if kwargs.get("normalize_embeddings", True) else 2.0
        return np.array([[scale] + [0.0] * 383 for _ in texts])

    value.encode.side_effect = encode
    return value


@pytest.fixture
def service(model):
    with patch("sentence_transformers.SentenceTransformer", return_value=model):
        value = OptimizedEmbeddingService(enable_gpu=False, precompute_common_patterns=False)
        yield value
        value.close()


def test_inherited_single_encoding_uses_same_loader(service):
    assert len(service.encode_one("Example Person")) == 384


def test_optimized_encoding_uses_shared_preprocessing(service, model):
    result = service.get_embeddings_optimized(["Example Person 1980-01-01 INN1234567890"])
    assert result["success"]
    encoded_text = model.encode.call_args.args[0][0]
    assert "1980" not in encoded_text and "1234567890" not in encoded_text


def test_cached_vectors_distinguish_normalization_mode(service):
    first = service.get_embeddings_optimized(["Example Person"], normalize=False)
    second = service.get_embeddings_optimized(["Example Person"], normalize=True)
    assert first["embeddings"][0][0] == 2.0
    assert second["embeddings"][0][0] == 1.0
    assert second["cache_misses"] == 1


def test_cached_vectors_are_owned_copies(service):
    first = service.get_embeddings_optimized(["Example Person"])
    first["embeddings"][0][0] = float("nan")
    second = service.get_embeddings_optimized(["Example Person"])
    assert second["embeddings"][0][0] == 1.0


def test_empty_batch_has_a_defined_result(service):
    result = service.get_embeddings_optimized([])
    assert result["success"] and result["embeddings"] == []


def test_blank_row_does_not_change_vector_ownership(service):
    result = service.get_embeddings_optimized(["Example Person", " ", "Another Person"])
    assert result["success"] is False
    assert result["embeddings"] == []


@pytest.mark.parametrize("bad_vectors", [[], [[1.0] * 384], [[0.0] * 384] * 2,
    [[float("nan")] * 384] * 2, [[1.0] * 12] * 2])
def test_invalid_or_missing_model_rows_are_not_success(service, model, bad_vectors):
    model.encode.side_effect = None
    model.encode.return_value = bad_vectors
    result = service.get_embeddings_optimized(["Example Person", "Another Person"])
    assert not result["success"] and result["embeddings"] == []
    assert not service.embedding_cache


def test_unapproved_alternate_is_rejected_without_download(service):
    with patch("sentence_transformers.SentenceTransformer") as loader:
        result = service.get_embeddings_optimized(["Example Person"], model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert not result["success"]
        loader.assert_not_called()


def test_public_error_does_not_echo_model_exception(service, model):
    model.encode.side_effect = RuntimeError("SECRET_SENTINEL Private input")
    result = service.get_embeddings_optimized(["Example Person"])
    assert not result["success"] and "SECRET_SENTINEL" not in str(result)


def test_model_exception_does_not_leak_into_logs(service, model, caplog):
    model.encode.side_effect = RuntimeError("PRIVATE_MODEL_ERROR_SENTINEL")
    assert not service.get_embeddings_optimized(["Example Person"])["success"]
    assert "PRIVATE_MODEL_ERROR_SENTINEL" not in caplog.text


def test_warm_cache_cannot_bypass_closed_queue(service):
    assert service.get_embeddings_optimized(["Example Person"])["success"]
    service.close()
    with pytest.raises(InferenceUnavailableError):
        service.get_embeddings_optimized(["Example Person"])


def test_loading_is_inside_the_inference_budget(model):
    started, release = threading.Event(), threading.Event()
    value = OptimizedEmbeddingService(enable_gpu=False, precompute_common_patterns=False)

    def load(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return model

    try:
        with patch("sentence_transformers.SentenceTransformer", side_effect=load), ThreadPoolExecutor(1) as pool:
            future = pool.submit(value.get_embeddings_optimized, ["Example Person"])
            try:
                assert started.wait(5)
                assert value.get_inference_stats()["active"] == 1
            finally:
                release.set()
            assert future.result(5)["success"]
    finally:
        release.set()
        value.close()


def test_manager_pins_the_model_and_disallows_pickle(model):
    manager = EmbeddingModelManager(enable_gpu=False)
    config = EmbeddingConfig()
    try:
        with patch("sentence_transformers.SentenceTransformer", return_value=model) as loader:
            manager.get_model(config.model_name)
            assert loader.call_args.kwargs.get("revision") == config.revision
            assert loader.call_args.kwargs.get("trust_remote_code") is False
            assert loader.call_args.kwargs.get("model_kwargs") == {"use_safetensors": True}
    finally:
        manager.shutdown()


def test_manager_model_name_cannot_hide_a_different_path(model):
    manager = EmbeddingModelManager(enable_gpu=False)
    try:
        with patch("sentence_transformers.SentenceTransformer", return_value=model) as loader:
            with pytest.raises(ValueError):
                config = ModelConfig(name=EmbeddingConfig().model_name, model_path="sentence-transformers/all-MiniLM-L6-v2")
                manager.get_model(config.name, config)
            loader.assert_not_called()
    finally:
        manager.shutdown()


def test_manager_cached_model_must_match_requested_dimension(model):
    manager = EmbeddingModelManager(enable_gpu=False)
    name = EmbeddingConfig().model_name
    try:
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            manager.get_model(name)
            with pytest.raises(ValueError):
                manager.get_model(name, ModelConfig(name=name, dimension=768))
    finally:
        manager.shutdown()


def test_manager_cannot_load_after_shutdown(model):
    manager = EmbeddingModelManager(enable_gpu=False)
    manager.shutdown()
    with patch("sentence_transformers.SentenceTransformer", return_value=model) as loader:
        with pytest.raises(InferenceUnavailableError):
            manager.get_model(EmbeddingConfig().model_name)
        loader.assert_not_called()


def test_named_model_defaults_cannot_be_mutated_for_another_caller():
    first = get_model_config("english")
    with pytest.raises((AttributeError, ValueError)):
        first.dimension = 12


@pytest.mark.asyncio
async def test_old_search_import_does_not_generate_a_fake_query_vector():
    value = LegacySearch()
    value._get_embedding_service = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="unavailable"):
        await value._build_query_vector(normalized(), "Example Person")


@pytest.mark.asyncio
async def test_old_search_import_propagates_query_failure():
    from ai_service.layers.search.contracts import SearchOpts
    value = LegacySearch()
    value._initialized = True
    value.readiness = AsyncMock(side_effect=ConnectionError("source unavailable"))
    with pytest.raises((ConnectionError, RuntimeError)):
        await value.find_candidates(normalized(), "Example Person", SearchOpts())


@pytest.mark.parametrize("label,value", [("INN", "1234567890"), ("ИНН", "1234567890"),
    ("ІПН", "1234567890"), ("ідентифікаційний номер:", "1234567890"),
    ("идентификационный номер:", "123456789012"), ("TIN", "123456789"),
    ("taxpayer identification number:", "123456789"), ("IN\u200bN", "12345\u200b67890")])
def test_preprocessing_v2_uses_explicit_tax_identifier_rules(label, value):
    from ai_service.services.embedding_preprocessor import EmbeddingPreprocessor
    assert EmbeddingPreprocessor().normalize_for_embedding(f"Example Person {label}{value}") == "Example Person"


@pytest.mark.parametrize("text", ["Company 12345678", "Company 12345678 note INN",
    "INN123", "Company INN1234567890Corp"])
def test_preprocessing_does_not_remove_unclassified_numeric_names(text):
    from ai_service.services.embedding_preprocessor import EmbeddingPreprocessor
    assert EmbeddingPreprocessor().normalize_for_embedding(text) == text


def test_preprocessing_version_change_rejects_old_vectors():
    from copy import deepcopy
    from ai_service.layers.search.config import HybridSearchConfig
    from ai_service.layers.search.index_schema import index_mapping, validate_mapping
    desired = index_mapping(HybridSearchConfig(), vectors=True)["mappings"]
    old = deepcopy(desired)
    old["_meta"]["embedding_contract"]["preprocessing_version"] = "embedding-preprocessor-v1"
    assert desired["_meta"]["embedding_contract"]["preprocessing_version"] == "embedding-preprocessor-v2"
    with pytest.raises(ValueError, match="preprocessing_version"):
        validate_mapping(old, desired)


def test_manager_caches_distinct_revisions_separately(model):
    manager = EmbeddingModelManager(enable_gpu=False)
    name = EmbeddingConfig().model_name
    other = Mock()
    other.get_sentence_embedding_dimension.return_value = 384
    try:
        with patch("sentence_transformers.SentenceTransformer", side_effect=[model, other]) as loader:
            assert manager.get_model(name) is model
            assert manager.get_model(name, ModelConfig(name=name, revision="a" * 40)) is other
            assert manager.get_model(name) is model
            assert loader.call_count == 2
            assert manager.get_model_info(name)["loaded_variants"] == 2
    finally:
        manager.shutdown()


def test_manager_information_is_an_owned_copy(model):
    manager = EmbeddingModelManager(enable_gpu=False)
    name = EmbeddingConfig().model_name
    try:
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            manager.get_model(name)
        first = manager.get_model_info(name)
        first["config"]["model_kwargs"]["use_safetensors"] = False
        first["config"]["revision"] = "a" * 40
        second = manager.get_model_info(name)
        assert second["config"]["revision"] == EmbeddingConfig().revision
        assert second["config"]["model_kwargs"]["use_safetensors"] is True
    finally:
        manager.shutdown()


def test_shutdown_during_preload_is_bounded_and_does_not_publish_model(model):
    entered, release = threading.Event(), threading.Event()
    manager = EmbeddingModelManager(enable_gpu=False, max_pending_loads=0)

    def load(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return model

    try:
        with patch("sentence_transformers.SentenceTransformer", side_effect=load):
            first = manager.preload_model(EmbeddingConfig().model_name)
            assert entered.wait(5)
            with pytest.raises(InferenceUnavailableError, match="capacity"):
                manager.preload_model(EmbeddingConfig().model_name)
            manager.shutdown()
            release.set()
            with pytest.raises(InferenceUnavailableError, match="closed"):
                first.result(5)
            assert manager.list_loaded_models() == []
    finally:
        release.set()
        manager.shutdown()


def test_cache_clear_during_loading_prevents_repopulation(model):
    entered, release = threading.Event(), threading.Event()
    manager = EmbeddingModelManager(enable_gpu=False)

    def load(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return model

    try:
        with patch("sentence_transformers.SentenceTransformer", side_effect=load):
            future = manager.preload_model(EmbeddingConfig().model_name)
            assert entered.wait(5)
            manager.clear_cache()
            release.set()
            assert future.result(5) is model
            assert manager.list_loaded_models() == []
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.parametrize("options", [{"use_safetensors": False}, {"trust_remote_code": True}])
def test_legacy_model_options_cannot_disable_safe_loading(options):
    with pytest.raises(ValueError, match="safe loading contract"):
        ModelConfig(name=EmbeddingConfig().model_name, model_kwargs=options)


@pytest.mark.parametrize("factory", [OptimizedEmbeddingService, EmbeddingModelManager])
def test_removed_pool_setting_is_explicitly_rejected(factory):
    with pytest.raises(ValueError, match="no longer supported"):
        factory(thread_pool_size=8)


@pytest.mark.asyncio
async def test_async_cached_calls_cannot_bypass_closed_queue(service):
    assert service.get_embeddings_optimized(["Example Person"])["success"]
    service.close()
    with pytest.raises(InferenceUnavailableError):
        await service.get_embeddings_async_optimized(["Example Person"])


@pytest.mark.asyncio
async def test_cancelled_async_waiter_does_not_encode_later(service, model):
    entered, release = threading.Event(), threading.Event()

    def encode(texts, **kwargs):
        entered.set()
        assert release.wait(5)
        return np.array([[1.0] + [0.0] * 383 for _ in texts])

    model.encode.side_effect = encode
    first = asyncio.create_task(service.get_embeddings_async_optimized(["Example Person"]))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(service.get_embeddings_async_optimized(["Another Person"]))
        for _ in range(100):
            if service.get_inference_stats()["pending"]:
                break
            await asyncio.sleep(0)
        assert service.get_inference_stats()["pending"] == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        assert (await first)["success"]
        assert model.encode.call_count == 1
        assert service.get_inference_stats()["pending"] == 0
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


def test_faiss_fallback_reports_actual_execution(service):
    with patch("ai_service.layers.embeddings.optimized_embedding_service.faiss") as faiss:
        faiss.IndexFlatIP.side_effect = RuntimeError("Unavailable acceleration")
        result = service.find_similar_texts_optimized("Example Person", [f"Candidate {i}" for i in range(101)])
    assert result["success"] and not result["faiss_accelerated"]
    assert [row["text"] for row in result["results"]] == [f"Candidate {i}" for i in range(10)]


@pytest.mark.model
def test_real_legacy_and_primary_vectors_agree_after_preprocessing():
    from ai_service.layers.embeddings.embedding_service import EmbeddingService
    primary = EmbeddingService(EmbeddingConfig())
    legacy = OptimizedEmbeddingService(enable_gpu=False, precompute_common_patterns=False)
    texts = ["Example Person INN1234567890", "Another Person 1980-01-01"]
    try:
        expected = primary.encode_batch(texts)
        result = legacy.get_embeddings_optimized(texts)
        assert result["success"]
        np.testing.assert_allclose(result["embeddings"], expected, atol=1e-6)
        assert result["embedding_contract"] == primary.embedding_contract
        assert legacy.runtime_health_check()["model_validated"]
    finally:
        primary.close()
        legacy.close()
