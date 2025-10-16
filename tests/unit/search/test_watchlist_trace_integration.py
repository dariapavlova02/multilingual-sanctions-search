"""Trace contracts use complete ready local indices and actual snapshot files."""

import pytest
from unittest.mock import patch

from ai_service.layers.embeddings.indexing.watchlist_index_service import (
    WatchlistIndexService,
)
from ai_service.layers.embeddings.indexing.enhanced_vector_index_service import (
    EnhancedVectorIndexConfig,
)
from ai_service.contracts.trace_models import SearchTrace
from ai_service.utils.inference_queue import InferenceUnavailableError


class TestWatchlistTraceIntegration:
    def setup_method(self):
        self.service = WatchlistIndexService(
            EnhancedVectorIndexConfig(
                use_semantic_embeddings=False, use_svd=False, use_faiss=False
            )
        )
        self.trace = SearchTrace(enabled=True)

    def teardown_method(self):
        self.service.close()

    def build(self, rows=None):
        self.service.build_from_corpus(
            rows or [("doc1", "test text", "person", {})], "test_active"
        )

    def test_search_with_trace_enabled(self):
        self.build(
            [
                ("doc1", "test text", "person", {}),
                ("doc2", "another text", "person", {}),
            ]
        )
        with patch.object(
            self.service._active, "search", return_value=[("doc1", 0.9), ("doc2", 0.8)]
        ):
            results = self.service.search("test query", top_k=10, trace=self.trace)
        assert results == [("doc1", 0.9), ("doc2", 0.8)]
        assert len(self.trace.steps) == 2
        active, rerank = self.trace.steps
        assert (
            active.stage == "LEXICAL"
            and active.query == "test query"
            and active.topk == 10
        )
        assert [(h.doc_id, h.score) for h in active.hits] == results
        assert active.hits[0].source == "LEXICAL"
        assert (
            active.meta["active_id"] == "test_active"
            and active.meta["search_type"] == "active"
        )
        assert (
            rerank.stage == "RERANK"
            and rerank.query == "test query"
            and rerank.topk == 10
        )
        assert [(h.doc_id, h.score) for h in rerank.hits] == results
        assert rerank.meta["merge_strategy"] == "overlay_replaces_base"

    def test_search_with_overlay_and_active(self):
        self.build([("doc1", "active text", "person", {})])
        self.service.set_overlay_from_corpus(
            [("doc2", "overlay text", "person", {})], "test_overlay"
        )
        with (
            patch.object(self.service._active, "search", return_value=[("doc1", 0.9)]),
            patch.object(self.service._overlay, "search", return_value=[("doc2", 0.8)]),
        ):
            results = self.service.search("test query", top_k=10, trace=self.trace)
        assert results == [("doc1", 0.9), ("doc2", 0.8)]
        assert len(self.trace.steps) == 3
        assert self.trace.steps[0].meta["overlay_id"] == "test_overlay"
        assert self.trace.steps[0].meta["search_type"] == "overlay"
        assert self.trace.steps[1].meta["active_id"] == "test_active"
        assert self.trace.steps[2].stage == "RERANK"
        assert self.trace.steps[2].meta["total_candidates"] == 2

    def test_search_with_empty_index(self):
        with pytest.raises(InferenceUnavailableError):
            self.service.search("test query", trace=self.trace)
        assert self.trace.steps == []
        assert self.trace.notes == ["Watchlist search unavailable"]

    def test_search_with_trace_disabled(self):
        self.build()
        trace = SearchTrace(enabled=False)
        with patch.object(self.service._active, "search", return_value=[("doc1", 0.9)]):
            assert self.service.search("test query", trace=trace) == [("doc1", 0.9)]
        assert trace.steps == [] and trace.notes == []

    def test_search_with_no_trace(self):
        self.build()
        with patch.object(self.service._active, "search", return_value=[("doc1", 0.9)]):
            assert self.service.search("test query") == [("doc1", 0.9)]

    def test_search_with_exception(self):
        self.build()
        with patch.object(
            self.service._active,
            "search",
            side_effect=RuntimeError("private provider details"),
        ):
            with pytest.raises(
                InferenceUnavailableError, match="^Watchlist search unavailable$"
            ):
                self.service.search("test query", trace=self.trace)
        assert self.trace.steps == []
        assert self.trace.notes == ["Watchlist search unavailable"]

    def test_get_doc_with_trace_enabled(self):
        self.build()
        self.service.set_overlay_from_corpus([("doc2", "overlay text", "person", {})])
        assert self.service.get_doc("doc2", self.trace).text == "overlay text"
        assert self.service.get_doc("doc1", self.trace).text == "test text"
        assert self.service.get_doc("doc3", self.trace) is None
        assert self.trace.notes == [
            "Document doc2 found in overlay index",
            "Document doc1 found in active index",
            "Document doc3 not found in any index",
        ]

    def test_get_doc_with_exception(self):
        class UnavailableMetadata(dict):
            def get(self, *args):
                raise RuntimeError("private metadata details")

        with patch.object(self.service, "_overlay_docs", UnavailableMetadata()):
            with pytest.raises(
                InferenceUnavailableError, match="^Watchlist metadata unavailable$"
            ):
                self.service.get_doc("doc1", self.trace)
        assert self.trace.notes == ["Watchlist metadata unavailable"]

    def test_reload_snapshot_with_trace_enabled(self, tmp_path):
        self.build()
        self.service.save_snapshot(str(tmp_path))
        result = self.service.reload_snapshot(str(tmp_path), trace=self.trace)
        assert result["active_loaded"] is True and result["active_count"] == 1
        assert "load_time_ms" in result
        assert self.trace.notes == ["Watchlist snapshot loaded"]
        assert self.service.get_doc("doc1").text == "test text"

    def test_reload_snapshot_with_missing_directory(self, tmp_path):
        result = self.service.reload_snapshot(
            str(tmp_path / "absent"), trace=self.trace
        )
        assert result == {"error": "Snapshot file not found"}
        assert self.trace.notes == ["Watchlist snapshot reload failed"]

    def test_reload_snapshot_with_exception(self, tmp_path):
        self.build()
        with patch.object(
            self.service,
            "_read_payload",
            side_effect=RuntimeError("private storage details"),
        ):
            result = self.service.reload_snapshot(str(tmp_path), trace=self.trace)
        assert result == {"error": "Watchlist snapshot reload failed"}
        assert self.trace.notes == ["Watchlist snapshot reload failed"]
        assert self.service.ready()
