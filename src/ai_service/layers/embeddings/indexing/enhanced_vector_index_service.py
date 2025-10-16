"""Local lexical/semantic retrieval with one bounded operation queue and atomic state."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Optional

import numpy as np

from ....config import EmbeddingConfig
from ....utils.inference_queue import InferenceUnavailableError
from .vector_index_service import VectorIndexConfig, CharTfidfVectorIndex, _LexicalState
from ..optimized_embedding_service import OptimizedEmbeddingService


@dataclass(frozen=True)
class EnhancedVectorIndexConfig(VectorIndexConfig):
    use_semantic_embeddings: bool = True
    semantic_weight: float = 0.6
    embedding_model: Optional[str] = None
    enable_hybrid_search: bool = True
    min_semantic_similarity: float = 0.3
    max_candidates_for_reranking: int = 100
    embedding_revision: Optional[str] = None
    embedding_dimension: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("use_semantic_embeddings", "enable_hybrid_search"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Semantic index switches must be booleans")
        for name in ("semantic_weight", "min_semantic_similarity"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "Semantic weights and thresholds must be between zero and one"
                )
        if (
            type(self.max_candidates_for_reranking) is not int
            or self.max_candidates_for_reranking < 1
        ):
            raise ValueError("Reranking capacity must be positive")
        values = {}
        if self.embedding_model is not None:
            values["model_name"] = self.embedding_model
        if self.embedding_revision is not None:
            values["revision"] = self.embedding_revision
        if self.embedding_dimension is not None:
            values["dimension"] = self.embedding_dimension
        selected = EmbeddingConfig(**values)
        object.__setattr__(self, "embedding_model", selected.model_name)
        object.__setattr__(self, "embedding_revision", selected.revision)
        object.__setattr__(self, "embedding_dimension", selected.dimension)


@dataclass(frozen=True)
class _SemanticState:
    matrix: np.ndarray
    contract: tuple
    faiss_index: object = None


def embedding_configuration(config):
    return EmbeddingConfig(
        model_name=config.embedding_model,
        revision=config.embedding_revision,
        dimension=config.embedding_dimension,
        device="cpu",
        max_pending_calls=0,
        inference_timeout=config.operation_timeout,
    )


def new_embedding_service(config):
    return OptimizedEmbeddingService(
        config=embedding_configuration(config),
        enable_gpu=False,
        precompute_common_patterns=False,
    )


class EnhancedVectorIndex(CharTfidfVectorIndex):
    def __init__(self, config=None, *, embedding_service=None):
        if config is not None and not isinstance(config, EnhancedVectorIndexConfig):
            raise TypeError("Expected EnhancedVectorIndexConfig")
        super().__init__(config or EnhancedVectorIndexConfig())
        self._semantic_state = None
        self._embedding_config = embedding_configuration(self.cfg)
        self._owns_embedding_service = embedding_service is None
        self.embedding_service = None
        self.search_metrics = {
            name: 0
            for name in (
                "total_searches",
                "lexical_searches",
                "semantic_searches",
                "hybrid_searches",
                "failed_searches",
            )
        }
        self.search_metrics["avg_search_time"] = 0.0
        if self.cfg.use_semantic_embeddings:
            self.embedding_service = (
                embedding_service
                if embedding_service is not None
                else new_embedding_service(self.cfg)
            )
            self._require_provider()

    @property
    def semantic_embeddings(self):
        with self._state_lock:
            return (
                None
                if self._semantic_state is None
                else self._semantic_state.matrix.copy()
            )

    @property
    def semantic_faiss_index(self):
        return (
            None if self._semantic_state is None else self._semantic_state.faiss_index
        )

    @property
    def embedding_contract(self):
        return (
            self._embedding_config.embedding_contract()
            if self.cfg.use_semantic_embeddings
            else None
        )

    @staticmethod
    def _contract_key(contract):
        return tuple(sorted(contract.items()))

    def _require_provider(self):
        if (
            self.embedding_service is None
            or self.embedding_service.embedding_contract != self.embedding_contract
        ):
            raise InferenceUnavailableError(
                "Local semantic provider contract differs from index"
            )

    def _vectors(self, texts):
        self._require_provider()
        result = self.embedding_service.get_embeddings_optimized(
            list(texts), batch_size=64, use_cache=True
        )
        self._require_provider()
        if (
            not result.get("success")
            or result.get("embedding_contract") != self.embedding_contract
        ):
            raise InferenceUnavailableError("Local semantic generation failed")
        matrix = np.asarray(result.get("embeddings"), dtype=np.float32)
        if (
            matrix.shape != (len(texts), self._embedding_config.dimension)
            or not np.isfinite(matrix).all()
            or not np.any(matrix != 0, axis=1).all()
        ):
            raise InferenceUnavailableError("Invalid local semantic vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or not (norms > 0).all():
            raise InferenceUnavailableError("Invalid local semantic vector norms")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    def _rebuild(self, docs, deadline):
        try:
            if not docs:
                lexical, semantic = _LexicalState(), None
            else:
                lexical = (
                    self._fit_state(docs)
                    if self.cfg.enable_hybrid_search
                    or not self.cfg.use_semantic_embeddings
                    else _LexicalState(
                        tuple(x[0] for x in docs), tuple(x[1] for x in docs)
                    )
                )
                semantic = None
                if self.cfg.use_semantic_embeddings:
                    matrix = self._vectors([text for _, text in docs])
                    semantic = _SemanticState(
                        matrix,
                        self._contract_key(self.embedding_contract),
                        self._make_faiss(matrix),
                    )
            with self._state_lock:
                self._check_commit(deadline)
                self._state, self._semantic_state = lexical, semantic
        except InferenceUnavailableError:
            raise
        except Exception:
            raise InferenceUnavailableError(
                "Local semantic index rebuild failed"
            ) from None

    def ready(self):
        with self._state_lock:
            if (
                self._closed
                or not self._state.doc_ids
                or self._operations.health_check()["status"] != "healthy"
            ):
                return False
            if (
                self.cfg.enable_hybrid_search or not self.cfg.use_semantic_embeddings
            ) and self._state.matrix is None:
                return False
            if self.cfg.use_semantic_embeddings:
                try:
                    self._require_provider()
                    return (
                        self._semantic_state is not None
                        and self._semantic_state.contract
                        == self._contract_key(self.embedding_contract)
                        and self.embedding_service.runtime_health_check()["status"]
                        == "healthy"
                    )
                except Exception:
                    return False
            return True

    def _search(self, query, top_k):
        started = time.monotonic()
        try:
            if not self.ready():
                raise InferenceUnavailableError("Local index is not ready")
            if self.cfg.use_semantic_embeddings and self.cfg.enable_hybrid_search:
                candidate_k = min(
                    self.cfg.max_candidates_for_reranking, max(top_k * 3, 50)
                )
                lexical = self._search_state(self._state, query, candidate_k)
                semantic = self._semantic_search(query, candidate_k)
                result = self._combine_and_rerank(lexical, semantic, top_k)
                kind = "hybrid_searches"
            elif self.cfg.use_semantic_embeddings:
                result = self._semantic_search(query, top_k)
                kind = "semantic_searches"
            else:
                result = self._search_state(self._state, query, top_k)
                kind = "lexical_searches"
            with self._state_lock:
                self.search_metrics[kind] += 1
                self.search_metrics["total_searches"] += 1
                total = self.search_metrics["total_searches"]
                self.search_metrics["avg_search_time"] += (
                    time.monotonic() - started - self.search_metrics["avg_search_time"]
                ) / total
            return result
        except Exception:
            with self._state_lock:
                self.search_metrics["failed_searches"] += 1
            raise InferenceUnavailableError("Local index search unavailable") from None

    def _semantic_search(self, query, top_k):
        state = self._semantic_state
        if state is None or state.contract != self._contract_key(
            self.embedding_contract
        ):
            raise InferenceUnavailableError("Local semantic state is unavailable")
        query_vector = self._vectors([query])
        # Exact cosine is also the fallback for unavailable optional acceleration.
        if state.faiss_index is not None:
            try:
                scores, ids = state.faiss_index.search(
                    query_vector, min(top_k, len(self._state.doc_ids))
                )
                values = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]
                if (
                    len(values) != min(top_k, len(self._state.doc_ids))
                    or len({i for i, _ in values}) != len(values)
                    or any(
                        i < 0 or i >= len(self._state.doc_ids) or not math.isfinite(s)
                        for i, s in values
                    )
                ):
                    raise ValueError("Invalid accelerated semantic result")
            except Exception:
                self.logger.warning(
                    "Local semantic acceleration unavailable; using exact cosine"
                )
                values = list(enumerate((state.matrix @ query_vector.T).reshape(-1)))
        else:
            values = list(enumerate((state.matrix @ query_vector.T).reshape(-1)))
        if any(not math.isfinite(float(s)) for _, s in values):
            raise InferenceUnavailableError("Invalid semantic scores")
        results = [
            (self._state.doc_ids[i], min(1.0, max(-1.0, float(score))))
            for i, score in values
            if score >= self.cfg.min_semantic_similarity
        ]
        return sorted(results, key=lambda x: (-x[1], x[0]))[:top_k]

    def _combine_and_rerank(self, lexical_results, semantic_results, top_k):
        lexical, semantic = dict(lexical_results), dict(semantic_results)
        result = []
        for identity in lexical.keys() | semantic.keys():
            left, right = max(0.0, lexical.get(identity, 0.0)), max(
                0.0, semantic.get(identity, 0.0)
            )
            score = (
                1 - self.cfg.semantic_weight
            ) * left + self.cfg.semantic_weight * right
            if left > 0 and right > 0:
                score *= 1.2
            result.append((identity, min(1.0, score)))
        return sorted(result, key=lambda x: (-x[1], x[0]))[:top_k]

    def get_index_statistics(self):
        with self._state_lock:
            return {
                "document_count": len(self._state.doc_ids),
                "ready": self.ready(),
                "lexical_index_built": self._state.matrix is not None,
                "semantic_index_built": self._semantic_state is not None,
                "faiss_lexical_available": self._state.faiss_index is not None,
                "faiss_semantic_available": self.semantic_faiss_index is not None,
                "embedding_contract": self.embedding_contract,
                "search_metrics": dict(self.search_metrics),
                "operations": self._operations.snapshot(),
            }

    def optimize_index(self):
        return self._operations.run(
            self._optimize, time.monotonic() + self.cfg.operation_timeout
        )

    def _optimize(self, deadline):
        if not self.ready():
            raise InferenceUnavailableError("Local index is not ready for optimization")
        lexical = self._state
        semantic = self._semantic_state
        if lexical.matrix is not None:
            lexical = replace(lexical, faiss_index=self._make_faiss(lexical.matrix))
        if semantic is not None:
            semantic = replace(semantic, faiss_index=self._make_faiss(semantic.matrix))
        with self._state_lock:
            self._check_commit(deadline)
            self._state, self._semantic_state = lexical, semantic
        return {
            **self.get_index_statistics(),
            "acceleration_rebuilt": lexical.faiss_index is not None
            or (semantic is not None and semantic.faiss_index is not None),
        }

    def warm_up(self, sample_queries=None):
        for query in sample_queries or ["Model readiness verification"]:
            self.search(query, top_k=5)

    def close(self):
        super().close()
        with self._state_lock:
            self._semantic_state = None
        if self.embedding_service is not None and self._owns_embedding_service:
            self.embedding_service.close()
