"""Synchronous entry points for async service APIs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor


def run_sync(coroutine):
    """Run a coroutine without nesting or replacing the caller's event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()
