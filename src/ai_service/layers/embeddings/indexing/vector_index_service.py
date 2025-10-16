"""Bounded local TF-IDF retrieval with atomic publication of complete generations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    import faiss
except ImportError:
    faiss = None

from ....utils.inference_queue import InferenceQueue, InferenceUnavailableError
from ....utils.logging_config import get_logger


@dataclass(frozen=True)
class VectorIndexConfig:
    ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 1
    sublinear_tf: bool = True
    norm: str = "l2"
    use_svd: bool = True
    svd_dim: int = 128
    use_faiss: bool = True
    hnsw_m: int = 32
    ef_search: int = 96
    max_features: int = 100_000
    max_documents: int = 50_000
    max_text_length: int = 4096
    max_corpus_bytes: int = 25 * 1024 * 1024
    max_dense_values: int = 8_000_000
    max_pending_operations: int = 16
    operation_timeout: float = 30.0

    def __post_init__(self):
        if (
            not isinstance(self.ngram_range, (tuple, list))
            or len(self.ngram_range) != 2
            or any(type(x) is not int or x < 1 for x in self.ngram_range)
            or self.ngram_range[0] > self.ngram_range[1]
        ):
            raise ValueError("Invalid character n-gram range")
        object.__setattr__(self, "ngram_range", tuple(self.ngram_range))
        for name in (
            "min_df",
            "svd_dim",
            "hnsw_m",
            "ef_search",
            "max_features",
            "max_documents",
            "max_text_length",
            "max_corpus_bytes",
            "max_dense_values",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("Index capacity settings must be positive integers")
        for name in ("sublinear_tf", "use_svd", "use_faiss"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Index switches must be booleans")
        if self.norm not in ("l1", "l2"):
            raise ValueError("Unsupported TF-IDF norm")
        if (
            type(self.max_pending_operations) is not int
            or self.max_pending_operations < 0
        ):
            raise ValueError("Pending index capacity must be nonnegative")
        if (
            type(self.operation_timeout) not in (int, float)
            or not math.isfinite(self.operation_timeout)
            or self.operation_timeout <= 0
        ):
            raise ValueError("Index timeout must be positive and finite")


@dataclass(frozen=True)
class _LexicalState:
    doc_ids: tuple = ()
    doc_texts: tuple = ()
    vectorizer: Any = None
    svd: Any = None
    matrix: Any = None
    faiss_index: Any = None


class CharTfidfVectorIndex:
    def __init__(self, config: Optional[VectorIndexConfig] = None):
        if config is not None and not isinstance(config, VectorIndexConfig):
            raise TypeError("Expected VectorIndexConfig")
        self._cfg = replace(config) if config is not None else VectorIndexConfig()
        self.logger = get_logger(__name__)
        self._state = _LexicalState()
        self._state_lock = threading.RLock()
        self._closed = False
        self._operations = InferenceQueue(
            self.cfg.max_pending_operations,
            self.cfg.operation_timeout,
            label="Local index",
        )

    @property
    def cfg(self):
        return self._cfg

    @property
    def doc_ids(self):
        with self._state_lock:
            return list(self._state.doc_ids)

    @property
    def doc_texts(self):
        with self._state_lock:
            return list(self._state.doc_texts)

    @property
    def vectorizer(self):
        return self._state.vectorizer

    @property
    def svd(self):
        return self._state.svd

    @property
    def X_vec(self):
        with self._state_lock:
            return None if self._state.matrix is None else self._state.matrix.copy()

    @property
    def faiss_index(self):
        return self._state.faiss_index

    def _snapshot_docs(self, docs):
        if not isinstance(docs, (list, tuple)) or len(docs) > self.cfg.max_documents:
            raise ValueError("Invalid local index corpus size")
        result = []
        identities = set()
        size = 0
        for row in docs:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("Expected document ID and text")
            identity, text = row
            if (
                not isinstance(identity, str)
                or not identity.strip()
                or len(identity) > self.cfg.max_text_length
                or not isinstance(text, str)
                or not text.strip()
                or len(text) > self.cfg.max_text_length
            ):
                raise ValueError("Invalid local index document")
            if identity in identities:
                raise ValueError("Duplicate document ID")
            identities.add(identity)
            size += len(identity.encode("utf-8")) + len(text.encode("utf-8"))
            if size > self.cfg.max_corpus_bytes:
                raise ValueError("Local index corpus exceeds configured byte limit")
            result.append((identity, text))
        return tuple(result)

    def _build_vectorizer(self):
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=self.cfg.ngram_range,
            min_df=self.cfg.min_df,
            sublinear_tf=self.cfg.sublinear_tf,
            norm=self.cfg.norm,
            max_features=self.cfg.max_features,
            dtype=np.float32,
        )

    def _make_faiss(self, matrix):
        if faiss is None or not self.cfg.use_faiss:
            return None
        if sparse.issparse(matrix):
            if matrix.shape[0] * matrix.shape[1] > self.cfg.max_dense_values:
                return None  # Exact sparse cosine remains available without allocating a dense copy.
            matrix = matrix.toarray()
        try:
            index = faiss.IndexHNSWFlat(
                matrix.shape[1], self.cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efSearch = self.cfg.ef_search
            index.add(np.ascontiguousarray(matrix, dtype=np.float32))
            return index
        except Exception:
            self.logger.warning(
                "Local index acceleration unavailable; using exact cosine"
            )
            return None

    def _fit_state(self, docs):
        if not docs:
            return _LexicalState()
        vectorizer = self._build_vectorizer()
        matrix = vectorizer.fit_transform([text for _, text in docs])
        svd = None
        if self.cfg.use_svd and matrix.shape[1] > 1:
            svd = TruncatedSVD(
                n_components=min(self.cfg.svd_dim, matrix.shape[1] - 1), random_state=0
            )
            matrix = svd.fit_transform(matrix)
        matrix = normalize(matrix, norm="l2").astype(np.float32)
        values = matrix.data if sparse.issparse(matrix) else matrix
        row_norms = (
            np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1)
            if sparse.issparse(matrix)
            else np.linalg.norm(matrix, axis=1)
        )
        if (
            matrix.shape[0] != len(docs)
            or not np.isfinite(values).all()
            or not (row_norms > 0).all()
        ):
            raise ValueError("Invalid lexical matrix")
        return _LexicalState(
            tuple(x[0] for x in docs),
            tuple(x[1] for x in docs),
            vectorizer,
            svd,
            matrix,
            self._make_faiss(matrix),
        )

    def _check_commit(self, deadline):
        if self._closed or time.monotonic() >= deadline:
            raise InferenceUnavailableError("Local index update is no longer available")

    def rebuild(self, docs):
        prepared = self._snapshot_docs(docs)
        deadline = time.monotonic() + self.cfg.operation_timeout
        return self._operations.run(self._rebuild, prepared, deadline)

    def _rebuild(self, docs, deadline):
        try:
            state = self._fit_state(docs)
            with self._state_lock:
                self._check_commit(deadline)
                self._state = state
        except InferenceUnavailableError:
            raise
        except Exception:
            raise InferenceUnavailableError("Local index rebuild failed") from None

    @staticmethod
    def _validate_query(query, top_k):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must contain text")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

    def search(self, query, top_k=10):
        self._validate_query(query, top_k)
        if len(query) > self.cfg.max_text_length:
            raise ValueError("Search query exceeds configured length")
        return self._operations.run(self._search, query, top_k)

    def _search(self, query, top_k):
        try:
            return self._search_state(self._state, query, top_k)
        except InferenceUnavailableError:
            raise
        except Exception:
            raise InferenceUnavailableError("Local lexical search failed") from None

    def _search_state(self, state, query, top_k):
        if not state.doc_ids or state.matrix is None:
            return []
        vector = state.vectorizer.transform([query])
        if state.svd is not None:
            vector = state.svd.transform(vector)
        vector = normalize(vector, norm="l2").astype(np.float32)
        data = vector.data if sparse.issparse(vector) else vector
        if not np.isfinite(data).all():
            raise InferenceUnavailableError("Invalid lexical query vector")
        if not np.any(data):
            return []
        if state.faiss_index is not None:
            try:
                dense = vector.toarray() if sparse.issparse(vector) else vector
                scores, positions = state.faiss_index.search(
                    np.ascontiguousarray(dense), min(top_k, len(state.doc_ids))
                )
                pairs = [(int(i), float(s)) for i, s in zip(positions[0], scores[0])]
                if (
                    len(pairs) != min(top_k, len(state.doc_ids))
                    or len({i for i, _ in pairs}) != len(pairs)
                    or any(
                        i < 0 or i >= len(state.doc_ids) or not math.isfinite(s)
                        for i, s in pairs
                    )
                ):
                    raise ValueError("Invalid acceleration result")
                return sorted(
                    ((state.doc_ids[i], min(1.0, max(-1.0, s))) for i, s in pairs),
                    key=lambda x: (-x[1], x[0]),
                )
            except Exception:
                self.logger.warning(
                    "Local accelerated search unavailable; using exact cosine"
                )
        similarities = state.matrix @ vector.T
        if sparse.issparse(similarities):
            similarities = similarities.toarray()
        similarities = np.asarray(similarities).reshape(-1)
        if (
            similarities.shape != (len(state.doc_ids),)
            or not np.isfinite(similarities).all()
        ):
            raise InferenceUnavailableError("Invalid lexical search results")
        pairs = [
            (identity, min(1.0, max(-1.0, float(score))))
            for identity, score in zip(state.doc_ids, similarities)
        ]
        return sorted(pairs, key=lambda x: (-x[1], x[0]))[:top_k]

    def ready(self):
        with self._state_lock:
            return (
                not self._closed
                and bool(self._state.doc_ids)
                and self._state.matrix is not None
                and self._operations.health_check()["status"] == "healthy"
            )

    def close(self):
        with self._state_lock:
            self._closed = True
            self._state = _LexicalState()
        self._operations.close()
