"""Atomic active/overlay watchlists and bounded, data-only snapshot persistence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from ....contracts.trace_models import SearchTrace, SearchTraceHit, SearchTraceStep
from ....utils.inference_queue import InferenceQueue, InferenceUnavailableError
from ....utils.logging_config import get_logger
from ....utils.source_text_view import without_format_controls
from .enhanced_vector_index_service import (
    EnhancedVectorIndex,
    EnhancedVectorIndexConfig,
    new_embedding_service,
)
from .local_index_snapshot import (
    FORMAT,
    VERSION,
    FILENAME,
    canonical_bytes,
    read_json,
    read_snapshot,
    write_snapshot,
)


@dataclass(frozen=True)
class WatchlistDoc:
    doc_id: str
    text: str
    entity_type: str
    metadata: dict


class _LegacySnapshot(ValueError):
    pass


class WatchlistIndexService:
    def __init__(self, cfg=None):
        if cfg is not None and not isinstance(cfg, EnhancedVectorIndexConfig):
            raise TypeError("Expected EnhancedVectorIndexConfig")
        self._cfg = replace(cfg) if cfg is not None else EnhancedVectorIndexConfig()
        self.logger = get_logger(__name__)
        self._lock = threading.RLock()
        self._closed = False
        self._operations = InferenceQueue(
            self.cfg.max_pending_operations,
            self.cfg.operation_timeout,
            label="Watchlist",
        )
        # One owned encoder serves active, overlay and unpublished candidate indices.
        self._embedding_service = (
            new_embedding_service(self.cfg)
            if self.cfg.use_semantic_embeddings
            else None
        )
        self._active = self._new_index()
        self._overlay = None
        self._docs, self._overlay_docs = {}, {}
        self._active_id = self._overlay_id = None

    @property
    def cfg(self):
        return self._cfg

    def _new_index(self):
        return EnhancedVectorIndex(self.cfg, embedding_service=self._embedding_service)

    def ready(self):
        with self._lock:
            return (
                not self._closed
                and bool(self._docs)
                and self._active.ready()
                and (self._overlay is None or self._overlay.ready())
                and self._operations.health_check()["status"] == "healthy"
            )

    def _snapshot_corpus(self, corpus):
        if (
            not isinstance(corpus, (list, tuple))
            or len(corpus) > self.cfg.max_documents
        ):
            raise ValueError("Invalid watchlist corpus size")
        prepared = []
        size = 0
        for row in corpus:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ValueError("Expected document ID, text, entity type and metadata")
            identity, text, entity_type, metadata = row
            if (
                not isinstance(entity_type, str)
                or not entity_type.strip()
                or len(entity_type) > self.cfg.max_text_length
                or type(metadata) is not dict
            ):
                raise ValueError("Invalid watchlist document metadata")
            try:
                data = canonical_bytes(metadata)
                owned = json.loads(data)
                size += len(canonical_bytes([identity, text, entity_type, owned]))
            except (TypeError, ValueError, RecursionError):
                raise ValueError(
                    "Watchlist metadata must be finite JSON data"
                ) from None
            if size > self.cfg.max_corpus_bytes:
                raise ValueError("Watchlist corpus exceeds configured byte limit")
            prepared.append((identity, text, entity_type, owned))
        self._active._snapshot_docs(
            [(identity, text) for identity, text, _, _ in prepared]
        )
        return tuple(prepared)

    @staticmethod
    def _index_id(value):
        if value is None:
            return str(uuid.uuid4())
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise ValueError("Invalid watchlist generation identifier")
        return value

    def build_from_corpus(self, corpus, index_id=None):
        rows = self._snapshot_corpus(corpus)
        return self._operations.run(
            self._replace,
            rows,
            self._index_id(index_id),
            False,
            time.monotonic() + self.cfg.operation_timeout,
        )

    def set_overlay_from_corpus(self, corpus, overlay_id=None):
        rows = self._snapshot_corpus(corpus)
        return self._operations.run(
            self._replace,
            rows,
            self._index_id(overlay_id),
            True,
            time.monotonic() + self.cfg.operation_timeout,
        )

    def _replace(self, rows, index_id, overlay, deadline):
        candidate = self._new_index()
        published = False
        try:
            candidate.rebuild([(identity, text) for identity, text, _, _ in rows])
            documents = {
                identity: WatchlistDoc(identity, text, entity_type, metadata)
                for identity, text, entity_type, metadata in rows
            }
            with self._lock:
                if self._closed or time.monotonic() >= deadline:
                    raise InferenceUnavailableError(
                        "Watchlist update is no longer available"
                    )
                if overlay:
                    previous = self._overlay
                    self._overlay = candidate if rows else None
                    self._overlay_docs = documents
                    self._overlay_id = index_id if rows else None
                else:
                    previous = self._active
                    self._active, self._docs, self._active_id = (
                        candidate,
                        documents,
                        index_id,
                    )
                published = bool(rows) or not overlay
            if previous is not None:
                previous.close()
        finally:
            if not published:
                candidate.close()

    def clear_overlay(self):
        return self.set_overlay_from_corpus([])

    def search(self, query, top_k=50, trace=None):
        self._active._validate_query(query, top_k)
        if len(query) > self.cfg.max_text_length:
            raise ValueError("Watchlist query exceeds configured length")
        if trace is not None and not isinstance(trace, SearchTrace):
            raise TypeError("Expected SearchTrace")
        try:
            result, local_trace = self._operations.run(
                self._search, query, top_k, bool(trace and trace.enabled)
            )
        except Exception:
            if trace is not None:
                trace.note("Watchlist search unavailable")
            raise InferenceUnavailableError("Watchlist search unavailable") from None
        if trace is not None and trace.enabled:
            trace.steps.extend(local_trace.steps)
            trace.notes.extend(local_trace.notes)
        return result

    def _search(self, query, top_k, trace_enabled):
        if not self.ready():
            raise InferenceUnavailableError("Watchlist is not ready")
        trace = SearchTrace(enabled=trace_enabled)
        stage = (
            ("HYBRID" if self.cfg.enable_hybrid_search else "SEMANTIC")
            if self.cfg.use_semantic_embeddings
            else "LEXICAL"
        )
        candidates, signals_by_id = {}, {}
        overlay_hits = 0
        for label, index, documents, index_id in (
            ("overlay", self._overlay, self._overlay_docs, self._overlay_id),
            ("active", self._active, self._docs, self._active_id),
        ):
            if index is None:
                trace.note("No overlay index available")
                continue
            started = time.monotonic()
            # An overlay replaces every old row with that ID, even when its new
            # name does not match. Retrieve enough base candidates to fill top_k.
            requested = min(
                len(documents),
                top_k + (len(self._overlay_docs) if label == "active" else 0),
            )
            results = index.search(query, max(1, requested))
            hits = []
            for identity, score in results:
                if (
                    identity not in documents
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not -1 <= score <= 1
                ):
                    raise InferenceUnavailableError(
                        "Watchlist index and metadata disagree"
                    )
                if score <= 0 or (label == "active" and identity in self._overlay_docs):
                    continue
                candidates[identity] = float(score)
                signals = self._extract_signals(identity, documents[identity], query)
                signals_by_id[identity] = signals
                hits.append(
                    SearchTraceHit(
                        identity, float(score), len(hits) + 1, stage, signals
                    )
                )
            if label == "overlay":
                overlay_hits = len(hits)
            trace.add_step(
                SearchTraceStep(
                    stage=stage,
                    query=query,
                    topk=top_k,
                    took_ms=(time.monotonic() - started) * 1000,
                    hits=hits,
                    meta={
                        "overlay_id": index_id if label == "overlay" else None,
                        "active_id": index_id if label == "active" else None,
                        "search_type": label,
                        "index_type": "hybrid" if stage == "HYBRID" else stage.lower(),
                    },
                )
            )
        items = sorted(candidates.items(), key=lambda row: (-row[1], row[0]))[:top_k]
        trace.add_step(
            SearchTraceStep(
                "RERANK",
                query,
                top_k,
                0.0,
                [
                    SearchTraceHit(
                        identity, score, rank, "RERANK", signals_by_id[identity]
                    )
                    for rank, (identity, score) in enumerate(items, 1)
                ],
                {
                    "total_candidates": len(candidates),
                    "final_results": len(items),
                    "merge_strategy": "overlay_replaces_base",
                    "overlay_hits": overlay_hits,
                },
            )
        )
        trace.limit_payload_size(max_size_kb=200)
        trace.note(
            f"Watchlist search completed: {len(items)} results from {len(candidates)} candidates"
        )
        return items, trace

    @staticmethod
    def _literal_present(value, query):
        if not isinstance(value, str) or not value.strip():
            return False
        value, query = (
            without_format_controls(value).casefold(),
            without_format_controls(query).casefold(),
        )
        return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", query) is not None

    @staticmethod
    def _full_date(value):
        if not isinstance(value, str):
            return None
        for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                pass
        return None

    def _extract_signals(self, identity, document, query):
        # These are literal source-field occurrences, not same-person evidence.
        dob = self._full_date(document.metadata.get("dob"))
        dates = re.findall(
            r"(?<!\d)(?:\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}|\d{2}/\d{2}/\d{4})(?!\d)",
            without_format_controls(query),
        )
        return {
            "source_dob_present_in_query": dob is not None
            and any(self._full_date(value) == dob for value in dates),
            "document_id_present_in_query": self._literal_present(identity, query),
            "source_text_present_in_query": self._literal_present(document.text, query),
            "entity_type": document.entity_type,
        }

    def get_doc(self, doc_id, trace=None):
        try:
            with self._lock:
                if self._closed:
                    raise InferenceUnavailableError("Watchlist is closed")
                document = self._overlay_docs.get(doc_id) or self._docs.get(doc_id)
                if document is None:
                    if trace is not None:
                        trace.note(f"Document {doc_id} not found in any index")
                    return None
                if trace is not None:
                    trace.note(
                        f'Document {doc_id} found in {"overlay" if doc_id in self._overlay_docs else "active"} index'
                    )
                return WatchlistDoc(
                    document.doc_id,
                    document.text,
                    document.entity_type,
                    json.loads(canonical_bytes(document.metadata)),
                )
        except Exception:
            if trace is not None:
                trace.note("Watchlist metadata unavailable")
            raise InferenceUnavailableError("Watchlist metadata unavailable") from None

    @contextmanager
    def _commit_guard(self, deadline):
        with self._lock:
            if self._closed or time.monotonic() >= deadline:
                raise InferenceUnavailableError(
                    "Snapshot publication is no longer available"
                )
            yield

    def _payload(self, rows, index_id):
        return {
            "format": FORMAT,
            "version": VERSION,
            "config": asdict(self.cfg),
            "embedding_contract": self._active.embedding_contract,
            "index_id": index_id,
            "documents": [
                {
                    "doc_id": identity,
                    "text": text,
                    "entity_type": entity_type,
                    "metadata": metadata,
                }
                for identity, text, entity_type, metadata in rows
            ],
        }

    def save_snapshot(self, snapshot_dir, as_overlay=False):
        if type(as_overlay) is not bool:
            raise ValueError("as_overlay must be boolean")
        try:
            return self._operations.run(
                self._save,
                str(snapshot_dir),
                as_overlay,
                time.monotonic() + self.cfg.operation_timeout,
            )
        except InferenceUnavailableError:
            raise
        except Exception:
            raise RuntimeError("Watchlist snapshot save failed") from None

    def _save(self, snapshot_dir, overlay, deadline):
        index, documents, index_id = (
            (self._overlay, self._overlay_docs, self._overlay_id)
            if overlay
            else (self._active, self._docs, self._active_id)
        )
        if index is None or not index.ready():
            raise InferenceUnavailableError("No ready watchlist index to save")
        rows = [
            (doc.doc_id, doc.text, doc.entity_type, doc.metadata)
            for doc in documents.values()
        ]
        digest = write_snapshot(
            snapshot_dir,
            self._payload(rows, index_id),
            self.cfg.max_corpus_bytes,
            commit_guard=self._commit_guard(deadline),
        )
        return {
            "saved": True,
            "path": snapshot_dir,
            "sha256": digest,
            "format_version": VERSION,
        }

    def _read_payload(self, snapshot_dir, expected_sha256):
        directory = Path(snapshot_dir)
        if not (directory / FILENAME).exists() and (
            (directory / "meta.json").exists() or (directory / "docs.json").exists()
        ):
            raise _LegacySnapshot(
                "Legacy snapshot requires explicit data-only migration"
            )
        payload, digest = read_snapshot(
            directory, self.cfg.max_corpus_bytes, expected_sha256
        )
        if (
            type(payload["config"]) is not dict
            or set(payload["config"]) != set(asdict(self.cfg))
            or EnhancedVectorIndexConfig(**payload["config"]) != self.cfg
        ):
            raise ValueError("Snapshot and configured retrieval policies disagree")
        if payload["embedding_contract"] != self._active.embedding_contract:
            raise ValueError("Snapshot model contract differs from configured provider")
        if not isinstance(payload["index_id"], str):
            raise ValueError("Snapshot generation identifier is required")
        self._index_id(payload["index_id"])
        if not isinstance(payload["documents"], list):
            raise ValueError("Snapshot documents must be a list")
        rows = []
        for doc in payload["documents"]:
            if type(doc) is not dict or set(doc) != {
                "doc_id",
                "text",
                "entity_type",
                "metadata",
            }:
                raise ValueError("Unsupported snapshot document")
            rows.append(
                (doc["doc_id"], doc["text"], doc["entity_type"], doc["metadata"])
            )
        return self._snapshot_corpus(rows), payload["index_id"], digest

    def reload_snapshot(
        self, snapshot_dir, as_overlay=False, trace=None, *, expected_sha256=None
    ):
        if type(as_overlay) is not bool:
            raise ValueError("as_overlay must be boolean")
        try:
            result = self._operations.run(
                self._reload,
                str(snapshot_dir),
                as_overlay,
                expected_sha256,
                time.monotonic() + self.cfg.operation_timeout,
            )
        except _LegacySnapshot:
            result = {
                "error": "Legacy snapshots require migration from JSON source records"
            }
        except FileNotFoundError:
            result = {"error": "Snapshot file not found"}
        except Exception:
            self.logger.error("Watchlist snapshot reload failed")
            result = {"error": "Watchlist snapshot reload failed"}
        if trace is not None:
            trace.note(
                "Watchlist snapshot reload failed"
                if "error" in result
                else "Watchlist snapshot loaded"
            )
        return result

    def _reload(self, directory, overlay, expected_sha256, deadline):
        started = time.monotonic()
        rows, index_id, digest = self._read_payload(directory, expected_sha256)
        if not rows:
            raise ValueError("Cannot activate an empty snapshot")
        self._replace(rows, index_id, overlay, deadline)
        label = "overlay" if overlay else "active"
        return {
            label + "_loaded": True,
            label + "_count": len(rows),
            "index_id": index_id,
            "path": directory,
            "sha256": digest,
            "load_time_ms": (time.monotonic() - started) * 1000,
        }

    def migrate_legacy_snapshot(self, source_dir, destination_dir):
        """Rebuild from old JSON source rows only; never inspect pickle/NumPy/FAISS files."""
        return self._operations.run(
            self._migrate,
            str(source_dir),
            str(destination_dir),
            time.monotonic() + self.cfg.operation_timeout,
        )

    def _migrate(self, source_dir, destination_dir, deadline):
        source, destination = Path(source_dir), Path(destination_dir)
        if (
            source.resolve() == destination.resolve()
            or (destination / FILENAME).exists()
        ):
            raise ValueError(
                "Migration requires a separate unused snapshot destination"
            )
        meta, _ = read_json(source / "meta.json", self.cfg.max_corpus_bytes)
        documents, _ = read_json(source / "docs.json", self.cfg.max_corpus_bytes)
        if (
            type(meta) is not dict
            or type(documents) is not dict
            or type(meta.get("index_config")) is not dict
        ):
            raise ValueError("Invalid legacy JSON source")
        configured = asdict(self.cfg)
        configured.update(meta["index_config"])
        if EnhancedVectorIndexConfig(**configured) != self.cfg:
            raise ValueError(
                "Legacy index configuration differs from the selected policy"
            )
        identities, texts = meta.get("doc_ids"), meta.get("doc_texts")
        if (
            not isinstance(identities, list)
            or not isinstance(texts, list)
            or len(identities) != len(texts)
            or set(identities) != set(documents)
        ):
            raise ValueError("Legacy index and metadata document sets disagree")
        rows = []
        for identity, text in zip(identities, texts):
            document = documents[identity]
            if type(document) is not dict or document.get("text") != text:
                raise ValueError("Legacy source text differs between files")
            rows.append(
                (
                    identity,
                    text,
                    document.get("entity_type"),
                    document.get("metadata", {}),
                )
            )
        rows = self._snapshot_corpus(rows)
        if not rows:
            raise ValueError("Cannot migrate an empty snapshot")
        candidate = self._new_index()
        try:
            candidate.rebuild([(identity, text) for identity, text, _, _ in rows])
            if self._closed or time.monotonic() >= deadline:
                raise InferenceUnavailableError("Snapshot migration expired")
            digest = write_snapshot(
                destination,
                self._payload(rows, str(uuid.uuid4())),
                self.cfg.max_corpus_bytes,
                commit_guard=self._commit_guard(deadline),
            )
        finally:
            candidate.close()
        return {
            "migrated": True,
            "documents": len(rows),
            "path": destination_dir,
            "sha256": digest,
            "model_objects_deserialized": False,
        }

    def status(self):
        with self._lock:
            return {
                "active": {"id": self._active_id, "count": len(self._docs)},
                "overlay": {"id": self._overlay_id, "count": len(self._overlay_docs)},
                "ready": self.ready(),
                "operations": self._operations.snapshot(),
            }

    def close(self):
        with self._lock:
            self._closed = True
            self._operations.close()
            self._active.close()
            if self._overlay is not None:
                self._overlay.close()
            if self._embedding_service is not None:
                self._embedding_service.close()
