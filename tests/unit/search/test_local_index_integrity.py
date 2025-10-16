"""Integrity regressions for local vector state and data-only snapshots."""

import copy
import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai_service.config import EmbeddingConfig
from ai_service.layers.embeddings.indexing import (
    enhanced_vector_index_service as enhanced,
)
from ai_service.layers.embeddings.indexing.vector_index_service import (
    CharTfidfVectorIndex,
    VectorIndexConfig,
)
from ai_service.layers.embeddings.indexing.watchlist_index_service import (
    WatchlistIndexService,
)
from ai_service.utils.inference_queue import InferenceUnavailableError


@pytest.fixture
def encoder(monkeypatch):
    control = SimpleNamespace(fail=False, override=None, contract=None, instances=[])

    class Encoder:
        def __init__(self, **kwargs):
            self.config = kwargs.get("config") or EmbeddingConfig(
                model_name=kwargs.get("default_model")
            )
            self.closed = False
            control.instances.append(self)

        @property
        def embedding_contract(self):
            return control.contract or self.config.embedding_contract()

        def get_embeddings_optimized(self, texts, **kwargs):
            if control.fail:
                return {
                    "success": False,
                    "embeddings": [],
                    "error": "secret provider detail",
                }
            rows = []
            for text in texts:
                row = np.zeros(self.config.dimension, dtype=np.float32)
                digest = hashlib.sha256(text.encode()).digest()
                row[list(digest)] = 1
                row /= np.linalg.norm(row)
                rows.append(row.tolist())
            return {
                "success": True,
                "embeddings": (
                    control.override if control.override is not None else rows
                ),
                "embedding_contract": self.embedding_contract,
            }

        def runtime_health_check(self):
            return {
                "status": "unhealthy" if self.closed else "healthy",
                "model_validated": not self.closed,
            }

        def get_performance_metrics(self):
            return {}

        def close(self):
            self.closed = True

        def clear_cache(self):
            pass

    monkeypatch.setattr(enhanced, "OptimizedEmbeddingService", Encoder)
    return control


def config(**kwargs):
    return enhanced.EnhancedVectorIndexConfig(use_svd=False, use_faiss=False, **kwargs)


def corpus(name="Alpha Example", identity="a"):
    return [
        (
            identity,
            name,
            "person",
            {"source": "synthetic-unit", "nested": {"value": "original"}},
        )
    ]


def test_failed_lexical_rebuild_preserves_previous_generation():
    index = CharTfidfVectorIndex(VectorIndexConfig(use_svd=False, use_faiss=False))
    index.rebuild([("old", "Alpha Example")])
    before = index.search("Alpha Example")
    with pytest.raises(Exception):
        index.rebuild([("new", "")])
    assert index.doc_ids == ["old"]
    assert index.search("Alpha Example") == before


def test_failed_semantic_rebuild_cannot_pair_new_ids_with_old_vectors(encoder):
    index = enhanced.EnhancedVectorIndex(config())
    index.rebuild([("old", "Alpha Example")])
    before = index.search("Alpha Example")
    encoder.fail = True
    with pytest.raises(InferenceUnavailableError):
        index.rebuild([("new", "Beta Example")])
    encoder.fail = False
    assert index.doc_ids == ["old"]
    assert index.search("Alpha Example") == before


def test_empty_rebuild_clears_semantic_state(encoder):
    index = enhanced.EnhancedVectorIndex(config())
    index.rebuild([("a", "Alpha Example")])
    index.rebuild([])
    assert index.semantic_embeddings is None
    assert not index.ready()


def test_semantic_failure_is_not_successful_lexical_fallback(encoder):
    index = enhanced.EnhancedVectorIndex(config())
    index.rebuild([("a", "Alpha Example")])
    encoder.fail = True
    with pytest.raises(InferenceUnavailableError):
        index.search("Alpha Example")


@pytest.mark.parametrize("bad", [[], [[1.0]], [[float("nan")] * 384], [[0.0] * 384]])
def test_invalid_semantic_rows_do_not_publish(encoder, bad):
    index = enhanced.EnhancedVectorIndex(config())
    encoder.override = bad
    with pytest.raises(InferenceUnavailableError):
        index.rebuild([("a", "Alpha Example")])
    assert not index.ready()


def test_changed_provider_contract_is_rejected_before_cached_search(encoder):
    index = enhanced.EnhancedVectorIndex(config())
    index.rebuild([("a", "Alpha Example")])
    encoder.contract = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ).embedding_contract()
    assert encoder.contract["model_name"] != index.cfg.embedding_model
    with pytest.raises(InferenceUnavailableError):
        index.search("Alpha Example")


def test_index_configuration_is_owned_and_immutable(encoder):
    selected = config()
    index = enhanced.EnhancedVectorIndex(selected)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        selected.semantic_weight = 0.1
    assert index.cfg.semantic_weight == 0.6


def test_closed_local_index_rejects_search_and_rebuild(encoder):
    index = enhanced.EnhancedVectorIndex(config())
    index.rebuild([("a", "Alpha Example")])
    index.close()
    for action in (
        lambda: index.search("Alpha Example"),
        lambda: index.rebuild([("b", "Beta Example")]),
    ):
        with pytest.raises(InferenceUnavailableError):
            action()


def test_failed_watchlist_update_keeps_metadata_and_vectors_together(encoder):
    service = WatchlistIndexService(config())
    service.build_from_corpus(corpus(), "before")
    encoder.fail = True
    with pytest.raises(InferenceUnavailableError):
        service.build_from_corpus(corpus("Beta Example", "b"), "after")
    encoder.fail = False
    assert service.get_doc("a").text == "Alpha Example"
    assert service.get_doc("b") is None
    assert service.status()["active"]["id"] == "before"


def test_watchlist_metadata_is_owned_on_input_and_output(encoder):
    service = WatchlistIndexService(config())
    rows = corpus()
    service.build_from_corpus(rows)
    rows[0][3]["nested"]["value"] = "input mutation"
    output = service.get_doc("a")
    assert output.metadata["nested"]["value"] == "original"
    output.metadata["nested"]["value"] = "output mutation"
    assert service.get_doc("a").metadata["nested"]["value"] == "original"


def test_watchlist_empty_and_failed_search_are_unavailable(encoder):
    service = WatchlistIndexService(config())
    with pytest.raises(InferenceUnavailableError):
        service.search("Alpha Example")
    service.build_from_corpus(corpus())
    encoder.fail = True
    with pytest.raises(InferenceUnavailableError):
        service.search("Alpha Example")


def test_overlay_replacement_cannot_reuse_old_name_hit(encoder):
    service = WatchlistIndexService(config(use_semantic_embeddings=False))
    service.build_from_corpus(corpus("abcdefgh", "a"))
    service.set_overlay_from_corpus(corpus("ijklmnop", "a"))
    assert service.get_doc("a").text == "ijklmnop"
    assert service.search("abcdefgh") == []


def test_snapshot_roundtrip_retains_semantic_mode_and_source(encoder, tmp_path):
    source = WatchlistIndexService(config())
    source.build_from_corpus(corpus(), "source-generation")
    before = source.search("Alpha Example")
    result = source.save_snapshot(str(tmp_path))
    assert result["saved"] is True
    restored = WatchlistIndexService(config())
    loaded = restored.reload_snapshot(str(tmp_path))
    assert loaded.get("active_loaded") is True, loaded
    assert isinstance(restored._active, enhanced.EnhancedVectorIndex)
    assert restored._active.semantic_embeddings is not None
    assert restored.search("Alpha Example") == before
    assert restored.get_doc("a").metadata == source.get_doc("a").metadata
    assert not list(tmp_path.glob("*.pkl"))


_PICKLE_CALLS = []


def _record_pickle_execution():
    _PICKLE_CALLS.append("executed")
    return None


class _PickleProbe:
    def __reduce__(self):
        return _record_pickle_execution, ()


def legacy_files(directory):
    directory.mkdir(exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {"doc_ids": ["a"], "doc_texts": ["Alpha Example"], "index_config": {}}
        )
    )
    (directory / "docs.json").write_text(
        json.dumps(
            {"a": {"text": "Alpha Example", "entity_type": "person", "metadata": {}}}
        )
    )
    (directory / "vectorizer.pkl").write_bytes(pickle.dumps(_PickleProbe()))
    (directory / "svd.pkl").write_bytes(pickle.dumps(None))


def test_legacy_pickle_is_never_executed(encoder, tmp_path):
    _PICKLE_CALLS.clear()
    legacy_files(tmp_path)
    service = WatchlistIndexService(config(use_semantic_embeddings=False))
    result = service.reload_snapshot(str(tmp_path))
    assert _PICKLE_CALLS == []
    assert "error" in result
    assert not service.ready()


def test_snapshot_failure_preserves_live_generation(encoder, tmp_path):
    service = WatchlistIndexService(config())
    service.build_from_corpus(corpus(), "current")
    (tmp_path / "snapshot.json").write_text("{malformed")
    result = service.reload_snapshot(str(tmp_path))
    assert "error" in result
    assert service.status()["active"]["id"] == "current"
    assert service.search("Alpha Example")[0][0] == "a"


def test_duplicate_document_ids_are_rejected(encoder):
    service = WatchlistIndexService(config())
    with pytest.raises(ValueError):
        service.build_from_corpus(corpus() + corpus("Different Name"))


def test_legacy_data_only_migration_does_not_execute_objects(encoder, tmp_path):
    _PICKLE_CALLS.clear()
    source, destination = tmp_path / "old", tmp_path / "new"
    legacy_files(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    service = WatchlistIndexService(config(use_semantic_embeddings=False))
    result = service.migrate_legacy_snapshot(str(source), str(destination))
    assert result["migrated"] is True
    assert _PICKLE_CALLS == []
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert (
        not service.ready()
    )  # Migration creates an artifact, not an implicit active swap.
    assert service.reload_snapshot(str(destination), expected_sha256=result["sha256"])[
        "active_loaded"
    ]
    assert service.search("Alpha Example")[0][0] == "a"


def test_changed_snapshot_digest_cannot_replace_active_data(encoder, tmp_path):
    service = WatchlistIndexService(config(use_semantic_embeddings=False))
    service.build_from_corpus(corpus(), "old")
    saved = service.save_snapshot(str(tmp_path))
    payload = json.loads((tmp_path / "snapshot.json").read_text())
    payload["payload"]["documents"][0]["text"] = "Tampered Example"
    from ai_service.layers.embeddings.indexing.local_index_snapshot import (
        canonical_bytes,
    )

    payload["payload_sha256"] = hashlib.sha256(
        canonical_bytes(payload["payload"])
    ).hexdigest()
    (tmp_path / "snapshot.json").write_bytes(canonical_bytes(payload))
    assert "error" in service.reload_snapshot(
        str(tmp_path), expected_sha256=saved["sha256"]
    )
    assert service.get_doc("a").text == "Alpha Example"
    assert service.status()["active"]["id"] == "old"


@pytest.mark.parametrize(
    "kind",
    [
        "digest",
        "version",
        "model",
        "config",
        "duplicate_id",
        "metadata",
        "extra_field",
        "empty",
        "missing_config",
        "null_generation",
    ],
)
def test_invalid_snapshot_is_rejected_before_live_swap(encoder, tmp_path, kind):
    service = WatchlistIndexService(config())
    service.build_from_corpus(corpus(), "old")
    service.save_snapshot(str(tmp_path))
    path = tmp_path / "snapshot.json"
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if kind == "digest":
        payload["index_id"] = "changed"
    elif kind == "version":
        payload["version"] = 999
    elif kind == "model":
        payload["embedding_contract"]["revision"] = "0" * 40
    elif kind == "config":
        payload["config"]["semantic_weight"] = 0.1
    elif kind == "duplicate_id":
        payload["documents"].append(copy.deepcopy(payload["documents"][0]))
    elif kind == "metadata":
        payload["documents"][0]["metadata"] = []
    elif kind == "extra_field":
        payload["executable"] = "forbidden"
    elif kind == "empty":
        payload["documents"] = []
    elif kind == "missing_config":
        del payload["config"]["semantic_weight"]
    elif kind == "null_generation":
        payload["index_id"] = None
    from ai_service.layers.embeddings.indexing.local_index_snapshot import (
        canonical_bytes,
    )

    if kind != "digest":
        envelope["payload_sha256"] = hashlib.sha256(
            canonical_bytes(payload)
        ).hexdigest()
    path.write_bytes(canonical_bytes(envelope))
    assert "error" in service.reload_snapshot(str(tmp_path))
    assert service.status()["active"]["id"] == "old"
    assert service.search("Alpha Example")[0][0] == "a"


def test_snapshot_symlink_is_not_followed(encoder, tmp_path):
    service = WatchlistIndexService(config(use_semantic_embeddings=False))
    service.build_from_corpus(corpus())
    service.save_snapshot(str(tmp_path / "source"))
    target = tmp_path / "target"
    target.mkdir()
    (target / "snapshot.json").symlink_to(tmp_path / "source" / "snapshot.json")
    assert "error" in service.reload_snapshot(str(target))


def test_snapshot_size_and_duplicate_json_fields_rejected(tmp_path):
    from ai_service.layers.embeddings.indexing.local_index_snapshot import read_json

    path = tmp_path / "data.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path, 100)
    with pytest.raises(ValueError, match="limit"):
        read_json(path, 5)
    path.write_text('{"a":NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        read_json(path, 100)


def test_snapshot_atomic_write_failure_keeps_previous_file(
    encoder, tmp_path, monkeypatch
):
    from ai_service.layers.embeddings.indexing import local_index_snapshot as snapshots

    service = WatchlistIndexService(config(use_semantic_embeddings=False))
    service.build_from_corpus(corpus())
    service.save_snapshot(str(tmp_path))
    before = (tmp_path / "snapshot.json").read_bytes()
    service.build_from_corpus(corpus("Beta Example", "b"))
    monkeypatch.setattr(
        snapshots.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("private filesystem details")),
    )
    with pytest.raises(RuntimeError, match="Watchlist snapshot save failed"):
        service.save_snapshot(str(tmp_path))
    assert (tmp_path / "snapshot.json").read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_active_and_overlay_share_one_owned_encoder(encoder):
    service = WatchlistIndexService(config())
    service.build_from_corpus(corpus())
    service.set_overlay_from_corpus(corpus("Beta Example", "b"))
    assert len(encoder.instances) == 1
    assert service._active.embedding_service is service._overlay.embedding_service
    service.clear_overlay()
    assert not encoder.instances[0].closed
    assert service.search("Alpha Example")[0][0] == "a"
    service.close()
    assert encoder.instances[0].closed
    assert not service.ready()


def test_failed_overlay_update_preserves_old_overlay(encoder):
    service = WatchlistIndexService(config())
    service.build_from_corpus(corpus())
    service.set_overlay_from_corpus(corpus("Beta Example", "b"), "before")
    encoder.fail = True
    with pytest.raises(InferenceUnavailableError):
        service.set_overlay_from_corpus(corpus("Gamma Example", "c"), "after")
    encoder.fail = False
    assert service.status()["overlay"]["id"] == "before"
    assert service.get_doc("b") is not None and service.get_doc("c") is None


@pytest.mark.parametrize(
    "values",
    [
        {"semantic_weight": float("nan")},
        {"semantic_weight": 2},
        {"min_semantic_similarity": -1},
        {"max_pending_operations": -1},
        {"operation_timeout": 0},
        {"operation_timeout": float("inf")},
        {"use_semantic_embeddings": "false"},
        {"max_features": 0},
        {"max_documents": 0},
        {"ngram_range": [5, 3]},
        {"embedding_revision": "main"},
    ],
)
def test_invalid_index_configuration_rejected(values):
    with pytest.raises(ValueError):
        enhanced.EnhancedVectorIndexConfig(**values)


def test_configuration_cannot_be_replaced_on_live_objects(encoder):
    for value in (
        enhanced.EnhancedVectorIndex(config()),
        WatchlistIndexService(config()),
    ):
        with pytest.raises(AttributeError):
            value.cfg = config(semantic_weight=0.1)


def test_sparse_lexical_index_does_not_force_dense_allocation():
    from scipy import sparse

    index = CharTfidfVectorIndex(
        VectorIndexConfig(use_svd=False, use_faiss=True, max_dense_values=1)
    )
    index.rebuild([("a", "Alpha Example"), ("b", "Beta Example")])
    assert sparse.issparse(index.X_vec)
    assert index.faiss_index is None
    assert index.search("Alpha Example")[0][0] == "a"


def test_faiss_hnsw_uses_cosine_inner_product():
    from ai_service.layers.embeddings.indexing import vector_index_service as lexical

    if lexical.faiss is None:
        pytest.skip("Optional FAISS is unavailable")
    index = CharTfidfVectorIndex(VectorIndexConfig(use_svd=False))
    index.rebuild([("a", "Alpha Example"), ("b", "Beta Example")])
    assert index.faiss_index.metric_type == lexical.faiss.METRIC_INNER_PRODUCT
    result = index.search("Alpha Example")
    assert result[0][0] == "a"
    assert result[0][1] == pytest.approx(1, abs=1e-5)
    assert result[0][1] > result[1][1]


def test_semantic_hnsw_uses_cosine_inner_product(encoder):
    from ai_service.layers.embeddings.indexing import vector_index_service as lexical

    if lexical.faiss is None:
        pytest.skip("Optional FAISS is unavailable")
    index = enhanced.EnhancedVectorIndex(
        enhanced.EnhancedVectorIndexConfig(use_svd=False, enable_hybrid_search=False)
    )
    index.rebuild([("a", "Alpha Example"), ("b", "Beta Example")])
    assert index.semantic_faiss_index.metric_type == lexical.faiss.METRIC_INNER_PRODUCT
    assert index.search("Alpha Example")[0] == pytest.approx(("a", 1))


def test_trace_reports_occurrence_without_inventing_identity(encoder):
    from ai_service.contracts.trace_models import SearchTrace

    service = WatchlistIndexService(config())
    service.build_from_corpus(
        [("123", "Alpha Example", "person", {"dob": "1980-01-01"})]
    )
    trace = SearchTrace(enabled=True)
    service.search("Alpha Example 123456 DOB 1980-01-02", trace=trace)
    assert trace.steps[0].stage == "HYBRID"
    signals = trace.steps[-1].hits[0].signals
    assert signals["document_id_present_in_query"] is False
    assert signals["source_dob_present_in_query"] is False
    assert "dob_match" not in signals and "id_match" not in signals


def test_invalid_source_rows_are_rejected_before_rebuild(encoder):
    service = WatchlistIndexService(
        config(max_documents=1, max_text_length=50, max_corpus_bytes=1000)
    )
    for rows in (
        corpus() * 2,
        corpus("x" * 51),
        [("a", "Alpha Example", "person", {"v": float("nan")})],
    ):
        with pytest.raises(ValueError):
            service.build_from_corpus(rows)
    assert not service.ready()


def test_index_search_admission_is_bounded(encoder, monkeypatch):
    import threading

    index = enhanced.EnhancedVectorIndex(config(max_pending_operations=0))
    index.rebuild([("a", "Alpha Example")])
    entered, release = threading.Event(), threading.Event()
    original = index.embedding_service.get_embeddings_optimized

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(index.embedding_service, "get_embeddings_optimized", blocked)
    result = []
    worker = threading.Thread(
        target=lambda: result.append(index.search("Alpha Example"))
    )
    worker.start()
    try:
        assert entered.wait(2)
        with pytest.raises(InferenceUnavailableError, match="capacity"):
            index.search("Alpha Example")
        assert index.get_index_statistics()["operations"]["pending"] == 0
    finally:
        release.set()
        worker.join(2)
        index.close()
    assert not worker.is_alive()
    assert result[0][0][0] == "a"


def test_expired_rebuild_never_publishes_late(encoder, monkeypatch):
    import threading

    index = enhanced.EnhancedVectorIndex(config(operation_timeout=0.2))
    index.rebuild([("old", "Alpha Example")])
    release, finished = threading.Event(), threading.Event()
    fit, rebuild = index._fit_state, index._rebuild

    def blocked(rows):
        assert release.wait(2)
        return fit(rows)

    def tracked(*args):
        try:
            return rebuild(*args)
        finally:
            finished.set()

    monkeypatch.setattr(index, "_fit_state", blocked)
    monkeypatch.setattr(index, "_rebuild", tracked)
    try:
        with pytest.raises(InferenceUnavailableError):
            index.rebuild([("new", "Beta Example")])
        assert index.doc_ids == ["old"]
    finally:
        release.set()
    assert finished.wait(2)
    assert index.doc_ids == ["old"]
    index.close()


def test_close_during_rebuild_prevents_publication(encoder, monkeypatch):
    import threading

    index = enhanced.EnhancedVectorIndex(config())
    entered, release = threading.Event(), threading.Event()
    original = index._fit_state

    def blocked(rows):
        entered.set()
        assert release.wait(2)
        return original(rows)

    monkeypatch.setattr(index, "_fit_state", blocked)
    errors = []

    def run():
        try:
            index.rebuild([("new", "Alpha Example")])
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(2)
        index.close()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], InferenceUnavailableError)
    assert index.doc_ids == [] and not index.ready()


def test_slow_watchlist_build_keeps_old_metadata_visible(encoder, monkeypatch):
    import threading

    service = WatchlistIndexService(config())
    service.build_from_corpus(corpus(), "before")
    entered, release = threading.Event(), threading.Event()
    original = service._new_index

    def candidate():
        value = original()
        fit = value._fit_state

        def blocked(rows):
            entered.set()
            assert release.wait(2)
            return fit(rows)

        value._fit_state = blocked
        return value

    monkeypatch.setattr(service, "_new_index", candidate)
    worker = threading.Thread(
        target=lambda: service.build_from_corpus(corpus("Beta Example", "b"), "after")
    )
    worker.start()
    try:
        assert entered.wait(2)
        assert service.get_doc("a").text == "Alpha Example"
        assert service.get_doc("b") is None
        assert service.status()["active"]["id"] == "before"
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert service.get_doc("a") is None and service.get_doc("b").text == "Beta Example"
    assert service.status()["active"]["id"] == "after"
    service.close()


def test_snapshot_timeout_cannot_replace_file_later(encoder, tmp_path, monkeypatch):
    import threading
    from ai_service.layers.embeddings.indexing import local_index_snapshot as snapshots

    service = WatchlistIndexService(
        config(use_semantic_embeddings=False, operation_timeout=0.2)
    )
    service.build_from_corpus(corpus())
    service.save_snapshot(str(tmp_path))
    before = (tmp_path / "snapshot.json").read_bytes()
    service.build_from_corpus(corpus("Beta Example", "b"))
    release, finished = threading.Event(), threading.Event()
    original = snapshots.canonical_bytes
    save = service._save

    def blocked(value):
        assert release.wait(2)
        return original(value)

    def tracked(*args):
        try:
            return save(*args)
        finally:
            finished.set()

    monkeypatch.setattr(snapshots, "canonical_bytes", blocked)
    monkeypatch.setattr(service, "_save", tracked)
    try:
        with pytest.raises(InferenceUnavailableError):
            service.save_snapshot(str(tmp_path))
    finally:
        release.set()
    assert finished.wait(2)
    assert (tmp_path / "snapshot.json").read_bytes() == before
    service.close()


def test_migration_cli_reads_json_source_only(tmp_path):
    import subprocess
    import os
    import sys
    from dataclasses import asdict

    source, destination = tmp_path / "source", tmp_path / "migrated"
    legacy_files(source)
    (source / "vectorizer.pkl").write_bytes(b"not a pickle")
    selected = tmp_path / "config.json"
    selected.write_text(json.dumps(asdict(config(use_semantic_embeddings=False))))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_service.scripts.migrate_watchlist_snapshot",
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--config",
            str(selected),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_objects_deserialized"] is False
    assert (destination / "snapshot.json").is_file()
    assert (source / "vectorizer.pkl").read_bytes() == b"not a pickle"


@pytest.mark.model
def test_real_primary_model_survives_data_only_snapshot_roundtrip(tmp_path):
    primary = EmbeddingConfig()
    selected = config()
    service = WatchlistIndexService(selected)
    try:
        rows = [
            (
                "company",
                "Alpine Research Group",
                "organization",
                {"source": "synthetic-model-check"},
            ),
            (
                "person",
                "Elena Kovalenko",
                "person",
                {"source": "synthetic-model-check"},
            ),
        ]
        service.build_from_corpus(rows, "real-model-generation")
        before = service.search("Elena Kovalenko")
        saved = service.save_snapshot(str(tmp_path))
        restored = service.reload_snapshot(
            str(tmp_path), expected_sha256=saved["sha256"]
        )
        assert restored["active_loaded"] is True
        assert service._active.embedding_contract == primary.embedding_contract()
        assert service.search("Elena Kovalenko") == before
        assert before[0][0] == "person"
        assert service.get_doc("person").metadata == rows[1][3]
    finally:
        service.close()


def test_optimization_rebuilds_acceleration_without_changing_source(encoder):
    index = enhanced.EnhancedVectorIndex(
        enhanced.EnhancedVectorIndexConfig(use_svd=False)
    )
    index.rebuild([("a", "Alpha Example"), ("b", "Beta Example")])
    before = index.search("Alpha Example")
    old_accelerator = index.semantic_faiss_index
    report = index.optimize_index()
    if old_accelerator is None:
        assert report["acceleration_rebuilt"] is False
    else:
        assert report["acceleration_rebuilt"] is True
        assert index.semantic_faiss_index is not old_accelerator
    assert index.search("Alpha Example") == before
    assert index.doc_ids == ["a", "b"]
    index.close()
