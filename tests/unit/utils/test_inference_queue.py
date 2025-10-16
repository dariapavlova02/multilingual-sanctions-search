"""Concurrency limits must survive cancellation, timeout and worker failure."""

import asyncio
from concurrent.futures import CancelledError
import threading

import pytest

from ai_service.utils.inference_queue import InferenceQueue, InferenceUnavailableError


def blocking_call():
    entered, release = threading.Event(), threading.Event()
    def work():
        entered.set()
        assert release.wait(3), "Test did not release worker"
        return "finished"
    return work, entered, release


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_capacity_and_cancelled_queue_entries_are_bounded():
    queue = InferenceQueue(max_pending=1, timeout=1)
    work, entered, release = blocking_call()
    first = queue.submit(work)
    assert entered.wait(1)
    executed = []
    try:
        for _ in range(200):
            pending = queue.submit(lambda: executed.append("cancelled work ran"))
            with pytest.raises(InferenceUnavailableError, match="capacity"):
                queue.submit(lambda: None)
            assert pending.cancel()
            assert queue.snapshot()['pending'] == 0
        assert executed == []
        last = queue.submit(lambda: "last")
    finally:
        release.set()
    assert first.result(timeout=1) == "finished"
    assert last.result(timeout=1) == "last"
    queue.close()


@pytest.mark.asyncio
async def test_cancelled_active_caller_does_not_release_model_slot():
    queue = InferenceQueue(max_pending=0, timeout=1)
    work, entered, release = blocking_call()
    first = asyncio.create_task(queue.run_async(work))
    try:
        await wait_until(entered.is_set)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert queue.snapshot()['active'] == 1
        with pytest.raises(InferenceUnavailableError, match="capacity"):
            await queue.run_async(lambda: None)
    finally:
        release.set()
    await wait_until(lambda: queue.snapshot()['active'] == 0)
    assert await queue.run_async(lambda: "next") == "next"
    queue.close()


@pytest.mark.asyncio
async def test_queued_deadline_removes_work_before_model_execution():
    queue = InferenceQueue(max_pending=1, timeout=0.02)
    work, entered, release = blocking_call()
    first = queue.submit(work)
    executed = []
    try:
        await wait_until(entered.is_set)
        with pytest.raises(InferenceUnavailableError, match="timed out"):
            await queue.run_async(lambda: executed.append(True))
        assert queue.snapshot()['pending'] == 0
    finally:
        release.set()
    await wait_until(first.done)
    assert executed == []
    queue.close()


def test_sync_deadline_keeps_active_slot_until_native_work_finishes():
    queue = InferenceQueue(max_pending=0, timeout=0.02)
    work, entered, release = blocking_call()
    try:
        with pytest.raises(InferenceUnavailableError, match="timed out"):
            queue.run(work)
        assert entered.is_set()
        assert queue.snapshot()['active'] == 1
        with pytest.raises(InferenceUnavailableError, match="capacity"):
            queue.submit(lambda: None)
    finally:
        release.set()
        queue.close()


def test_worker_failure_does_not_poison_following_jobs():
    queue = InferenceQueue(max_pending=1, timeout=1)
    def fail():
        raise ValueError("synthetic failure")
    with pytest.raises(ValueError, match="synthetic failure"):
        queue.run(fail)
    assert queue.run(lambda: 42) == 42
    queue.close()


def test_completed_call_releases_capacity_before_returning_result():
    queue = InferenceQueue(max_pending=0, timeout=1)
    assert [queue.run(lambda: 1) for _ in range(200)] == [1] * 200
    assert queue.snapshot()['completed'] == 200
    assert queue.snapshot()['rejected'] == 0
    queue.close()


def test_queue_can_be_shared_across_independent_event_loops():
    from concurrent.futures import ThreadPoolExecutor
    queue = InferenceQueue(max_pending=4, timeout=1)
    with ThreadPoolExecutor(max_workers=4) as callers:
        results = list(callers.map(lambda value: asyncio.run(queue.run_async(lambda: value * 2)), range(4)))
    assert results == [0, 2, 4, 6]
    queue.close()


def test_close_cancels_pending_work_and_rejects_new_calls():
    queue = InferenceQueue(max_pending=1, timeout=1)
    work, entered, release = blocking_call()
    first = queue.submit(work)
    assert entered.wait(1)
    pending = queue.submit(lambda: pytest.fail("Closed pending work executed"))
    try:
        queue.close()
        with pytest.raises(CancelledError):
            pending.result()
        with pytest.raises(InferenceUnavailableError, match="closed"):
            queue.submit(lambda: None)
        assert queue.snapshot()['active'] == 1
    finally:
        release.set()
    assert first.result(timeout=1) == "finished"


@pytest.mark.asyncio
async def test_sync_and_async_service_interfaces_share_one_model_worker(monkeypatch):
    from ai_service.config import EmbeddingConfig
    from ai_service.layers.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService(EmbeddingConfig(max_pending_calls=8))
    active, maximum = 0, 0
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    class Encoder:
        def encode(self, texts, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                entered.set()
                assert release.wait(3)
                return [[float(len(text))] for text in texts]
            finally:
                with lock:
                    active -= 1
    monkeypatch.setattr(service, '_load_model', lambda *args: Encoder())
    monkeypatch.setattr(service, '_get_cached_preprocessing', lambda text: text)
    first = asyncio.create_task(service.encode_one_async("First"))
    await wait_until(entered.is_set)
    calls = [asyncio.create_task(service.encode_batch_async(["Second", "Third"])),
             asyncio.create_task(asyncio.to_thread(service.encode_one, "Fourth")),
             asyncio.create_task(asyncio.to_thread(service.encode, "Fifth"))]
    try:
        await wait_until(lambda: service._inference.snapshot()['pending'] == 3)
    finally:
        release.set()
    assert await asyncio.gather(first, *calls) == [[5.0], [[6.0], [5.0]], [6.0], [5.0]]
    assert maximum == 1
    service.close()


@pytest.mark.parametrize('environment,value', [
    ('EMBEDDING_MAX_PENDING', '-1'), ('EMBEDDING_MAX_PENDING', '129'),
    ('EMBEDDING_TIMEOUT_SECONDS', '0'), ('EMBEDDING_TIMEOUT_SECONDS', 'nan'),
])
def test_invalid_environment_queue_limits_fail_startup(monkeypatch, environment, value):
    from ai_service.config import EmbeddingConfig
    from pydantic import ValidationError
    monkeypatch.setenv(environment, value)
    with pytest.raises(ValidationError):
        EmbeddingConfig()
