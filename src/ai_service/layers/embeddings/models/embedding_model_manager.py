"""Bounded model loading and caching using the production model contract."""
from collections import OrderedDict
from dataclasses import replace
import threading
import time

from ....utils.inference_queue import InferenceQueue, InferenceUnavailableError
from .loader import load_embedding_model
from .model_config import DEFAULT_MODELS, get_model_config


class EmbeddingModelManager:
    """Cache loaded models by their full specification, never just a display name.

    Returned model objects are intended for trusted library callers. Inference
    scheduling belongs to EmbeddingService; this manager bounds model loading.
    """

    def __init__(self, max_models=3, enable_gpu=False, cache_dir=None,
                 thread_pool_size=None, *, max_pending_loads=16, loading_timeout=30):
        if type(max_models) is not int or max_models < 1:
            raise ValueError("max_models must be a positive integer")
        if thread_pool_size is not None:
            raise ValueError("thread_pool_size is no longer supported; configure max_pending_loads")
        self.max_models = max_models
        self.enable_gpu = enable_gpu
        self.cache_dir = cache_dir
        self.device = self._detect_device()
        self._models = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False
        self._generation = 0
        self._loading = InferenceQueue(max_pending_loads, loading_timeout, label="Model loading")

    def _detect_device(self):
        if self.enable_gpu:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _resolve(self, model_name, config):
        config = get_model_config(model_name) if config is None else replace(config)
        canonical_name = get_model_config(model_name).name if model_name in DEFAULT_MODELS else model_name
        if config.name != canonical_name:
            raise ValueError("Requested model and configuration disagree")
        if config.enable_gpu and not self.enable_gpu:
            raise ValueError("Automatic GPU selection is disabled on this manager")
        embedding = config.embedding_config(self.device)
        folder = config.cache_dir if config.cache_dir is not None else self.cache_dir
        key = (embedding.model_name, embedding.revision, embedding.dimension,
               embedding.preprocessing_version, embedding.device, config.use_fp16,
               config.max_sequence_length, folder)
        return config, embedding, folder, key

    def get_model(self, model_name, config=None):
        return self._loading.run(self._get_model, model_name, config)

    def _get_model(self, model_name, config=None):
        config, embedding, folder, key = self._resolve(model_name, config)
        with self._lock:
            if self._closed:
                raise InferenceUnavailableError("Model loading service is closed")
            generation = self._generation
            if key in self._models:
                model, saved, _ = self._models.pop(key)
                self._models[key] = (model, saved, time.time())
                return model
        model = load_embedding_model(embedding, cache_folder=folder,
            use_fp16=config.use_fp16, expected_max_sequence_length=config.max_sequence_length)
        with self._lock:
            if self._closed:
                raise InferenceUnavailableError("Model loading service is closed")
            if generation == self._generation:
                self._models[key] = (model, config, time.time())
                while len(self._models) > self.max_models:
                    self._models.popitem(last=False)
        return model

    def preload_model(self, model_name, config=None):
        """Return an observable, cancellable future; a full queue rejects the call."""
        return self._loading.submit(self._get_model, model_name, config)

    def get_model_info(self, model_name):
        canonical = get_model_config(model_name).name if model_name in DEFAULT_MODELS else model_name
        with self._lock:
            matches = [(model, config, used) for key, (model, config, used) in self._models.items() if key[0] == canonical]
            if not matches:
                return {"loaded": False}
            model, config, used = matches[-1]
            return {"loaded": True, "config": config.to_dict(), "last_used": used,
                    "device": str(model.device), "loaded_variants": len(matches)}

    def list_loaded_models(self):
        with self._lock:
            return list(dict.fromkeys(key[0] for key in self._models))

    def clear_cache(self):
        with self._lock:
            self._generation += 1
            self._models.clear()

    def shutdown(self):
        with self._lock:
            self._closed = True
            self._generation += 1
            self._models.clear()
        self._loading.close()

    def __del__(self):
        if hasattr(self, "_loading"):
            self.shutdown()
