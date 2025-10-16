"""Cached legacy API over the canonical embedding loader and bounded worker."""
from collections import OrderedDict
from datetime import datetime
import threading
import time
from typing import Optional

import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

from ...config import EmbeddingConfig
from .embedding_service import EmbeddingService


class OptimizedEmbeddingService(EmbeddingService):
    def __init__(self, default_model=None, max_cache_size=1000,
                 enable_batch_optimization=True, enable_gpu=True, thread_pool_size=None,
                 precompute_common_patterns=True, *, config: Optional[EmbeddingConfig] = None):
        if type(max_cache_size) is not int or max_cache_size < 0:
            raise ValueError("Cache capacity must be a nonnegative integer")
        if thread_pool_size is not None:
            raise ValueError("thread_pool_size is no longer supported; configure max_pending_calls on EmbeddingConfig")
        self.enable_gpu = enable_gpu
        auto_gpu = self._check_gpu_availability()
        if config is None:
            values = {"device": "cuda" if auto_gpu else "cpu"}
            if default_model is not None:
                values["model_name"] = default_model
            config = EmbeddingConfig(**values)
        elif default_model is not None and default_model != config.model_name:
            raise ValueError("default_model and explicit model contract disagree")
        super().__init__(config)
        self.max_cache_size = max_cache_size
        self.enable_batch_optimization = enable_batch_optimization
        self.precompute_common_patterns = precompute_common_patterns
        self.gpu_available = self.config.device != "cpu"
        self.embedding_cache = OrderedDict()
        self.cache_lock = threading.RLock()
        self.performance_metrics = {"cache_hits": 0, "cache_misses": 0,
            "batch_optimizations": 0, "total_embeddings_generated": 0,
            "total_processing_time": 0.0, "gpu_accelerated": 0}
        if precompute_common_patterns:
            self._precompute_common_patterns()

    def _check_gpu_availability(self):
        if not self.enable_gpu:
            return False
        import torch
        return torch.cuda.is_available()

    def _precompute_common_patterns(self):
        # Retain the existing optional warmup vocabulary; use the same preprocessor
        # as real input and exclude terms that have no embeddable content.
        patterns = ["payment", "платіж", "платеж", "company", "компанія", "компания",
                    "person", "персона", "особа", "contract", "договір", "договор",
                    "invoice", "рахунок", "счет", "transfer", "переказ", "перевод"]
        patterns = [text for text in patterns if self.preprocessor.normalize_for_embedding(text)]
        self.warm_up_cache(patterns)

    def _get_cache_key(self, text, model_name, normalize=True):
        config = self._selected_model_config(model_name)
        return (config.model_name, config.revision, config.dimension,
                config.preprocessing_version, normalize, text)

    def _get_cached_embedding(self, text, model_name, normalize=True):
        key = self._get_cache_key(text, model_name, normalize)
        with self.cache_lock:
            saved = self.embedding_cache.pop(key, None)
            if saved is not None and time.monotonic() - saved[1] < 3600:
                self.embedding_cache[key] = saved
                self.performance_metrics["cache_hits"] += 1
                return list(saved[0])
            self.performance_metrics["cache_misses"] += 1
        return None

    def _cache_embedding(self, text, model_name, embedding, normalize=True):
        key = self._get_cache_key(text, model_name, normalize)
        with self.cache_lock:
            if not self.max_cache_size or self._inference.snapshot()["closed"]:
                return
            self.embedding_cache.pop(key, None)
            self.embedding_cache[key] = (tuple(embedding), time.monotonic())
            while len(self.embedding_cache) > self.max_cache_size:
                self.embedding_cache.popitem(last=False)

    def _load_model_optimized(self, model_name=None):
        return super()._load_model(model_name)

    def _load_model(self, model_name=None):
        return self._load_model_optimized(model_name)

    @staticmethod
    def _snapshot_texts(texts):
        return list(texts) if isinstance(texts, (list, tuple)) else texts

    def get_embeddings_optimized(self, texts, model_name=None, normalize=True,
                                 batch_size=32, use_cache=True):
        return self._inference.run(self._get_embeddings, self._snapshot_texts(texts),
                                   model_name, normalize, batch_size, use_cache)

    def _get_embeddings(self, texts, model_name=None, normalize=True, batch_size=32, use_cache=True):
        started = time.monotonic()
        try:
            config = self._selected_model_config(model_name)
            if type(normalize) is not bool or type(batch_size) is not int or batch_size < 1:
                raise ValueError("Invalid encoding options")
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
                raise ValueError("Expected a string or list of strings")
            if any(not self._get_cached_preprocessing(text) for text in texts):
                raise ValueError("Every requested row must have embeddable text")
            vectors = [None] * len(texts)
            missing = []
            for index, text in enumerate(texts):
                cached = self._get_cached_embedding(text, config.model_name, normalize) if use_cache else None
                if cached is None:
                    missing.append(index)
                else:
                    vectors[index] = cached
            effective_batch_size = batch_size
            if missing:
                if self.enable_batch_optimization and self.gpu_available and len(missing) > batch_size * 2:
                    effective_batch_size = min(batch_size * 2, 64)
                    with self.cache_lock:
                        self.performance_metrics["batch_optimizations"] += 1
                # _encode is the common synchronous worker implementation. Calling
                # it directly here avoids nesting a second queue behind this job.
                computed = self._encode([texts[i] for i in missing], batch_size=effective_batch_size,
                    model_name=config.model_name, normalize_embeddings=normalize)
                array = np.asarray(computed, dtype=np.float32)
                if (array.shape != (len(missing), config.dimension) or not np.isfinite(array).all()
                        or not np.any(array != 0, axis=1).all()):
                    raise ValueError("Model output does not preserve the row/vector contract")
                computed = array.tolist()
                for index, vector in zip(missing, computed):
                    vectors[index] = vector
                    if use_cache:
                        self._cache_embedding(texts[index], config.model_name, vector, normalize)
            elapsed = time.monotonic() - started
            with self.cache_lock:
                self.performance_metrics["total_embeddings_generated"] += len(missing)
                self.performance_metrics["total_processing_time"] += elapsed
                if self.gpu_available:
                    self.performance_metrics["gpu_accelerated"] += len(missing)
            return {"success": True, "embeddings": vectors, "model_name": config.model_name,
                "embedding_contract": config.embedding_contract(), "text_count": len(texts),
                "embedding_dimension": config.dimension, "processing_time": elapsed,
                "normalized": normalize, "batch_size": effective_batch_size,
                "timestamp": datetime.now().isoformat(), "cache_hits": len(texts)-len(missing),
                "cache_misses": len(missing), "gpu_accelerated": self.gpu_available and bool(missing)}
        except Exception:
            self.logger.error("Legacy embedding generation failed")
            return self._create_error_result("Embedding generation is unavailable")

    async def get_embeddings_async_optimized(self, texts, model_name=None, normalize=True, batch_size=32):
        return await self._inference.run_async(self._get_embeddings, self._snapshot_texts(texts),
                                             model_name, normalize, batch_size, True)

    def get_embeddings(self, texts, model_name=None, normalize=True, batch_size=32):
        return self.get_embeddings_optimized(texts, model_name, normalize, batch_size)

    def find_similar_texts_optimized(self, query, candidates, model_name=None,
                                    threshold=0.7, top_k=10, metric="cosine", use_faiss=True):
        return self._inference.run(self._find_similar, query, self._snapshot_texts(candidates),
                                   model_name, threshold, top_k, metric, use_faiss)

    def _find_similar(self, query, candidates, model_name, threshold, top_k, metric, use_faiss):
        started = time.monotonic()
        try:
            if (not isinstance(candidates, list) or type(top_k) is not int or top_k < 1
                    or metric not in {"cosine", "dot", "euclidean"} or not np.isfinite(threshold)):
                raise ValueError("Invalid similarity options")
            encoded = self._get_embeddings([query] + candidates, model_name)
            if not encoded["success"]:
                return self._create_error_result("Embedding generation is unavailable")
            query_vector, candidate_vectors = encoded["embeddings"][0], encoded["embeddings"][1:]
            accelerated = False
            if use_faiss and faiss is not None and metric == "cosine" and len(candidates) > 100:
                try:
                    results = self._faiss_similarity_search(query_vector, candidate_vectors, candidates, top_k, threshold)
                    accelerated = True
                except Exception:
                    results = self._numpy_similarity_search(query_vector, candidate_vectors, candidates, top_k, threshold, metric)
            else:
                results = self._numpy_similarity_search(query_vector, candidate_vectors, candidates, top_k, threshold, metric)
            return {"success": True, "query": query, "total_candidates": len(candidates),
                "threshold": threshold, "top_k": top_k, "metric": metric, "results": results,
                "model_name": encoded["model_name"], "embedding_contract": encoded["embedding_contract"],
                "processing_time": time.monotonic()-started, "optimized": True,
                "faiss_accelerated": accelerated, "timestamp": datetime.now().isoformat()}
        except Exception:
            self.logger.error("Legacy similarity calculation failed")
            return self._create_error_result("Similarity calculation is unavailable")

    def _faiss_similarity_search(self, query_embedding, candidate_embeddings, candidates, top_k, threshold):
        query = np.array([query_embedding], dtype=np.float32)
        matrix = np.array(candidate_embeddings, dtype=np.float32)
        faiss.normalize_L2(query)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, indices = index.search(query, min(top_k, len(candidates)))
        results = []
        for score, position in zip(scores[0], indices[0]):
            if position >= 0 and score >= threshold:
                results.append({"text": candidates[position], "similarity_score": float(score), "rank": len(results)+1})
        return results

    def _numpy_similarity_search(self, query_embedding, candidate_embeddings, candidates, top_k, threshold, metric):
        if not candidates:
            return []
        query, matrix = np.asarray(query_embedding), np.asarray(candidate_embeddings)
        if metric == "cosine":
            scores = (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        elif metric == "dot":
            scores = matrix @ query
        elif metric == "euclidean":
            scores = 1.0 / (1.0 + np.linalg.norm(matrix-query, axis=1))
        else:
            raise ValueError("Unsupported similarity metric")
        positions = [i for i in np.argsort(-scores, kind="stable") if scores[i] >= threshold][:top_k]
        return [{"text": candidates[i], "similarity_score": float(scores[i]), "rank": rank+1}
                for rank, i in enumerate(positions)]

    def find_similar_texts(self, query, candidates, model_name=None, threshold=0.7, top_k=10, metric="cosine"):
        return self.find_similar_texts_optimized(query, candidates, model_name, threshold, top_k, metric)

    async def find_similar_texts_async_optimized(self, query, candidates, model_name=None, threshold=0.7, top_k=10, metric="cosine"):
        return await self._inference.run_async(self._find_similar, query, self._snapshot_texts(candidates),
                                             model_name, threshold, top_k, metric, True)

    def get_performance_metrics(self):
        with self.cache_lock:
            metrics = dict(self.performance_metrics)
            total = metrics["cache_hits"] + metrics["cache_misses"]
            generated = metrics["total_embeddings_generated"]
            return {"cache_hit_rate": metrics["cache_hits"]/total if total else 0.0,
                "cache_size": len(self.embedding_cache), "max_cache_size": self.max_cache_size,
                "gpu_available": self.gpu_available, "gpu_accelerated_embeddings": metrics["gpu_accelerated"],
                "batch_optimizations": metrics["batch_optimizations"], "total_embeddings_generated": generated,
                "average_processing_time": metrics["total_processing_time"]/generated if generated else 0.0,
                "total_processing_time": metrics["total_processing_time"]}

    def clear_cache(self):
        with self.cache_lock:
            self.embedding_cache.clear()

    def warm_up_cache(self, texts, model_name=None):
        return self.get_embeddings_optimized(texts, model_name, use_cache=True)

    @staticmethod
    def _create_error_result(message):
        return {"success": False, "embeddings": [], "error": message}

    def close(self):
        super().close()
        self.clear_cache()
