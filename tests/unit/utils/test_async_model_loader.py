import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ai_service.utils.async_model_loader import AsyncModelLoader


@pytest.mark.asyncio
async def test_concurrent_loads_share_one_worker_and_survive_cancellation(monkeypatch):
    loader = AsyncModelLoader()
    started, release = threading.Event(), threading.Event()
    model = object()
    calls = []

    def load(*args):
        calls.append(args)
        started.set()
        assert release.wait(3)
        return model

    monkeypatch.setattr(loader, "_load_spacy_model", load)
    first = asyncio.create_task(loader.load_model_async("en", "model"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        others = [asyncio.create_task(loader.load_model_async("en", "model")) for _ in range(8)]
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*others), 3) == [model] * 8
        assert len(calls) == 1
        assert loader.get_model_sync("en") is model
    finally:
        release.set()
        loader.close()


def test_model_load_can_be_shared_by_separate_event_loops(monkeypatch):
    loader = AsyncModelLoader()
    model = object()
    monkeypatch.setattr(loader, "_load_spacy_model", lambda *args: model)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(asyncio.run, loader.load_model_async("en", "model")) for _ in range(2)]
            assert all(future.result(timeout=3) is model for future in futures)
    finally:
        loader.close()
