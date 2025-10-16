"""Actual provider lifecycle and queue health, with controlled failed models."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ai_service.config import EmbeddingConfig
from ai_service.layers.embeddings.embedding_service import EmbeddingService
from ai_service.layers.normalization.ner_gateways.unified_spacy_gateway import (
    UnifiedSpacyGateway, SupportedLanguage, NERUnavailableError,
)
from ai_service.utils.inference_queue import InferenceQueue, InferenceUnavailableError


class LanguageModel:
    pipe_names = ["ner"]

    def __init__(self):
        self.calls = 0
        self.failure = False

    def __call__(self, text):
        self.calls += 1
        if self.failure:
            raise RuntimeError("Controlled model failure")
        return SimpleNamespace(ents=[])


@pytest.fixture
def ner_models(monkeypatch):
    models = {}
    def load(name, **kwargs):
        models[name] = LanguageModel()
        return models[name]
    monkeypatch.setattr("spacy.load", load)
    gateway = UnifiedSpacyGateway()
    yield gateway, models
    gateway.close()


async def test_ner_readiness_requires_a_forward_pass_for_each_model(ner_models):
    gateway, models = ner_models
    for language in SupportedLanguage:
        gateway.is_available(language)
    assert gateway.runtime_health_check()["status"] == "unhealthy"
    assert all(m.calls == 0 for m in models.values())
    await gateway.initialize_runtime()
    assert gateway.runtime_health_check()["status"] == "healthy"
    assert all(m.calls == 1 for m in models.values())
    for _ in range(5):
        assert gateway.runtime_health_check()["status"] == "healthy"
    assert all(m.calls == 1 for m in models.values())


@pytest.mark.parametrize("failure", ["missing", "no_ner", "inference"])
async def test_ner_startup_rejects_unusable_models(ner_models, monkeypatch, failure):
    gateway, models = ner_models
    if failure == "missing":
        def missing(*args, **kwargs):
            raise OSError("Model package exists but weights are absent")
        monkeypatch.setattr("spacy.load", missing)
    else:
        gateway.is_available(SupportedLanguage.ENGLISH)
        model = models["en_core_web_sm"]
        if failure == "no_ner":
            model.pipe_names = []
        else:
            model.failure = True
    with pytest.raises(NERUnavailableError):
        await gateway.initialize_runtime()
    assert gateway.runtime_health_check()["status"] == "unhealthy"


async def test_ner_failure_clear_and_close_invalidate_readiness(ner_models):
    gateway, models = ner_models
    await gateway.initialize_runtime()
    model = models["en_core_web_sm"]
    model.failure = True
    with pytest.raises(NERUnavailableError):
        await gateway.get_ner_hints_async("Runtime probe", "en")
    assert gateway.runtime_health_check()["status"] == "unhealthy"
    model.failure = False
    await gateway.get_ner_hints_async("Runtime probe", "en")
    assert gateway.runtime_health_check()["status"] == "healthy"
    gateway.clear_cache("en")
    assert gateway.runtime_health_check()["status"] == "unhealthy"
    await gateway.initialize_runtime()
    gateway.close()
    assert gateway.runtime_health_check()["status"] == "unhealthy"


async def test_unsupported_request_does_not_poison_healthy_ner(ner_models):
    gateway, _ = ner_models
    await gateway.initialize_runtime()
    with pytest.raises(NERUnavailableError):
        await gateway.get_ner_hints_async("東京", "en")
    assert gateway.runtime_health_check()["status"] == "healthy"


def test_health_does_not_wait_on_model_loading_lock(ner_models):
    gateway, _ = ner_models
    entered, release = threading.Event(), threading.Event()
    def hold_lock():
        with gateway._model_lock:
            entered.set()
            release.wait(2)
    worker = threading.Thread(target=hold_lock)
    worker.start()
    try:
        assert entered.wait(1)
        # A separate thread holding the model lock must not block this probe.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(gateway.runtime_health_check).result(timeout=0.5)["status"] == "unhealthy"
    finally:
        release.set()
        worker.join(1)


class EmbeddingModel:
    def __init__(self):
        self.result = np.ones((1, 384), dtype=np.float32)
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def embedding_model(monkeypatch):
    service = EmbeddingService(EmbeddingConfig())
    model = EmbeddingModel()
    service._model = model
    monkeypatch.setattr(service, "_load_model", lambda *args: model)
    monkeypatch.setattr(service, "_get_cached_preprocessing", lambda value: value)
    yield service, model
    service.close()


async def test_embedding_requires_successful_vector_and_observes_close(embedding_model):
    service, model = embedding_model
    assert service.runtime_health_check()["status"] == "unhealthy"
    await service.initialize_runtime()
    assert service.runtime_health_check()["status"] == "healthy"
    for _ in range(5):
        service.runtime_health_check()
    assert model.calls == 1
    service.close()
    assert service.runtime_health_check()["status"] == "unhealthy"


@pytest.mark.parametrize("bad_result", [[], [[1.0]], [[float("nan")] * 384],
                                         [[float("inf")] * 384], [[0.0] * 384]])
async def test_invalid_probe_vectors_cannot_mark_model_ready(embedding_model, bad_result):
    service, model = embedding_model
    model.result = bad_result
    with pytest.raises(InferenceUnavailableError):
        await service.initialize_runtime()
    assert service.runtime_health_check()["status"] == "unhealthy"


async def test_embedding_runtime_failure_and_recovery_are_visible(embedding_model):
    service, model = embedding_model
    await service.initialize_runtime()
    model.result = RuntimeError("Controlled inference failure")
    with pytest.raises(RuntimeError):
        await service.encode_one_async("Runtime probe")
    assert service.runtime_health_check()["status"] == "unhealthy"
    model.result = np.ones((1, 384))
    await service.encode_one_async("Runtime probe")
    assert service.runtime_health_check()["status"] == "healthy"


def test_queue_reports_overdue_native_work_until_it_finishes():
    queue = InferenceQueue(1, 0.02)
    entered, release = threading.Event(), threading.Event()
    def stalled():
        entered.set()
        release.wait(2)
    try:
        with pytest.raises(InferenceUnavailableError, match="timed out"):
            queue.run(stalled)
        assert entered.is_set()
        health = queue.health_check()
        assert health["active"] == 1 and health["overdue"]
        assert health["status"] == "unhealthy"
    finally:
        release.set()
        queue.close()


def test_queue_temporary_occupancy_does_not_mark_a_worker_failed():
    queue = InferenceQueue(0, 2)
    entered, release = threading.Event(), threading.Event()
    def work():
        entered.set()
        release.wait(2)
    future = queue.submit(work)
    try:
        assert entered.wait(1)
        assert queue.health_check()["status"] == "healthy"
    finally:
        release.set()
        future.result(timeout=1)
        queue.close()


def test_variant_provider_reports_initialization_and_closed_worker():
    from ai_service.layers.variants.variant_generation_service import VariantGenerationService
    service = VariantGenerationService()
    try:
        assert service.runtime_health_check()["status"] == "unhealthy"
        service.initialize()
        assert service.runtime_health_check()["status"] == "healthy"
        service.close()
        assert service.runtime_health_check()["status"] == "unhealthy"
    finally:
        service.close()
