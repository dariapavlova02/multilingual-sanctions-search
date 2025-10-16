"""Durable status and concurrent writer protection on the production state volume."""

import pytest

from ai_service.api.ingestion_jobs import IngestionJobStore, IngestionBusy


def test_writer_lock_and_progress_survive_store_recreation(tmp_path):
    store = IngestionJobStore(tmp_path)
    job = store.reserve("ac_patterns", "sanctions_ac_patterns", 2)
    try:
        job.update(status="loading", progress=1)
        second_store = IngestionJobStore(tmp_path)
        assert second_store.get(job.job_id)["progress"] == 1
        with pytest.raises(IngestionBusy):
            second_store.reserve("ac_patterns", "sanctions_ac_patterns", 3)
        job.update(status="completed", progress=2)
    finally:
        job.close()
    assert IngestionJobStore(tmp_path).get(job.job_id)["status"] == "completed"


def test_abandoned_worker_is_interrupted_instead_of_completed(tmp_path):
    store = IngestionJobStore(tmp_path)
    job = store.reserve("vectors", "sanctions_vectors", 5)
    job.update(status="loading", progress=2)
    job.close()  # Equivalent to kernel releasing the lock after worker exit.
    recovered = IngestionJobStore(tmp_path).get(job.job_id)
    assert recovered["status"] == "interrupted"
    assert recovered["progress"] == 2
    retry = store.reserve("vectors", "sanctions_vectors", 5)
    assert retry.job_id != job.job_id
    retry.close()


@pytest.mark.parametrize("index", ["ac", "vectors"])
def test_vector_reservation_excludes_both_source_and_vector_writers(tmp_path, index):
    store = IngestionJobStore(tmp_path)
    job = store.reserve("vectors", "vectors", 2, related_indices=["ac"])
    try:
        with pytest.raises(IngestionBusy):
            IngestionJobStore(tmp_path).reserve("maintenance", index, 0)
        assert set(store.get(job.job_id)["locked_indices"]) == {"ac", "vectors"}
    finally:
        job.close()
    next_job = store.reserve("maintenance", index, 0)
    next_job.close()


def test_failed_multi_index_reservation_releases_already_acquired_lock(tmp_path):
    store = IngestionJobStore(tmp_path)
    active = store.reserve("vectors", "z-vectors", 1)
    try:
        with pytest.raises(IngestionBusy):
            store.reserve("vectors", "z-vectors", 1, related_indices=["a-ac"])
        independent = store.reserve("ac_patterns", "a-ac", 1)
        independent.close()
    finally:
        active.close()
