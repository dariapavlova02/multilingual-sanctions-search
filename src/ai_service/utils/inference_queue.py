"""Bounded, serial model execution shared by synchronous and async callers."""

import asyncio
from collections import deque
from concurrent.futures import Future
from contextvars import copy_context
import threading
import time


class InferenceUnavailableError(RuntimeError):
    """Model capacity is exhausted or the caller's execution deadline expired."""


class InferenceQueue:
    """One active model call and a bounded number of waiting jobs.

    Cancelling a queued future removes the job immediately. Cancelling a running
    caller cannot stop native code: the worker retains its slot until the call
    actually finishes. The worker exits when the queue drains, so unused service
    instances do not retain idle threads. No queue belongs to an asyncio loop.
    """

    def __init__(self, max_pending: int, timeout: float, *, label: str = "Embedding"):
        if max_pending < 0 or timeout <= 0:
            raise ValueError("Invalid inference queue limits")
        self.max_pending = max_pending
        self.timeout = timeout
        self.label = label
        self._lock = threading.Lock()
        self._pending = deque()
        self._worker = None
        self._active = False
        self._active_started = None
        self._closed = False
        self._submitted = self._completed = self._failed = 0
        self._rejected = self._cancelled = self._timed_out = 0

    def submit(self, function, *args, **kwargs):
        future = Future()
        context = copy_context()
        job = (future, context, function, args, kwargs)
        with self._lock:
            if self._closed:
                self._rejected += 1
                raise InferenceUnavailableError(f"{self.label} service is closed")
            if len(self._pending) + int(self._active) >= self.max_pending + 1:
                self._rejected += 1
                raise InferenceUnavailableError(f"{self.label} capacity reached")
            self._pending.append(job)
            self._submitted += 1
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._drain, name=f"{self.label}Inference", daemon=True
                )
                self._worker.start()

        def remove_cancelled(done):
            if done.cancelled():
                with self._lock:
                    self._cancelled += 1
                    try:
                        self._pending.remove(job)
                    except ValueError:
                        pass  # Worker already claimed the job.

        future.add_done_callback(remove_cancelled)
        return future

    def _drain(self):
        while True:
            with self._lock:
                if not self._pending:
                    self._worker = None
                    return
                future, context, function, args, kwargs = self._pending.popleft()
                self._active = True
                self._active_started = time.monotonic()
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        value = context.run(function, *args, **kwargs)
                    except BaseException as exc:
                        with self._lock:
                            self._failed += 1
                            self._active = False
                        future.set_exception(exc)
                    else:
                        with self._lock:
                            self._completed += 1
                            self._active = False
                        future.set_result(value)
            finally:
                with self._lock:
                    self._active = False

    def run(self, function, *args, **kwargs):
        future = self.submit(function, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError as exc:
            future.cancel()
            with self._lock:
                self._timed_out += 1
            raise InferenceUnavailableError(f"{self.label} execution timed out") from exc

    async def run_async(self, function, *args, **kwargs):
        future = self.submit(function, *args, **kwargs)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=self.timeout)
        except (asyncio.CancelledError, TimeoutError) as exc:
            future.cancel()
            wrapped.cancel()
            if isinstance(exc, asyncio.CancelledError):
                raise
            with self._lock:
                self._timed_out += 1
            raise InferenceUnavailableError(f"{self.label} execution timed out") from exc

    def close(self):
        """Reject new work and cancel pending jobs; active native work may finish."""
        with self._lock:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
        for future, *_ in pending:
            future.cancel()

    def snapshot(self):
        with self._lock:
            return {"active": int(self._active), "pending": len(self._pending),
                    "max_active": 1, "max_pending": self.max_pending,
                    "closed": self._closed, "submitted": self._submitted,
                    "completed": self._completed, "failed": self._failed,
                    "rejected": self._rejected, "cancelled": self._cancelled,
                    "timed_out": self._timed_out}

    def health_check(self):
        """A brief burst is healthy; closed or overdue native work is not.

        This does not submit a job or acquire a model's loading lock. A caller
        timeout cannot free a still-running native worker, so track its age.
        """
        with self._lock:
            elapsed = (time.monotonic() - self._active_started
                       if self._active and self._active_started is not None else 0.0)
            overdue = elapsed > self.timeout
            return {"status": "unhealthy" if self._closed or overdue else "healthy",
                    "closed": self._closed, "overdue": overdue,
                    "active": int(self._active), "pending": len(self._pending),
                    "max_active": 1, "max_pending": self.max_pending,
                    "active_seconds": elapsed, "timeout_seconds": self.timeout}
