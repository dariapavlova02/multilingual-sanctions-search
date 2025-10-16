"""Bound HTTP request bodies before JSON or multipart parsing."""

import asyncio
import os
import secrets

from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse


class RequestLimitsConfig(BaseModel):
    """Validated startup limits; changing them requires recreating the service."""

    model_config = ConfigDict(validate_default=True)
    max_body_bytes: int = Field(default_factory=lambda: int(os.getenv("MAX_REQUEST_BYTES", str(1024 * 1024))), ge=1, le=16 * 1024 * 1024)
    max_upload_bytes: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(26 * 1024 * 1024))), ge=1, le=128 * 1024 * 1024)
    body_timeout: float = Field(default_factory=lambda: float(os.getenv("HTTP_BODY_TIMEOUT_SECONDS", "30")), gt=0, le=300)
    processing_timeout: float = Field(default_factory=lambda: float(os.getenv("HTTP_PROCESSING_TIMEOUT_SECONDS", "30")), gt=0, le=300)
    max_readers: int = Field(default_factory=lambda: int(os.getenv("HTTP_MAX_INFLIGHT", "4")), ge=1, le=32)


class RequestLimitsMiddleware:
    def __init__(
        self,
        app,
        get_admin_key,
        max_body_bytes=1024 * 1024,
        max_upload_bytes=26 * 1024 * 1024,
        body_timeout=30,
        max_readers=4,
        processing_timeout=30,
    ):
        limits = RequestLimitsConfig(max_body_bytes=max_body_bytes,
            max_upload_bytes=max_upload_bytes, body_timeout=body_timeout,
            max_readers=max_readers, processing_timeout=processing_timeout)
        self.app = app
        self.get_admin_key = get_admin_key
        self.max_body_bytes = max_body_bytes
        self.max_upload_bytes = max_upload_bytes
        self.body_timeout = limits.body_timeout
        self.processing_timeout = limits.processing_timeout
        self.readers = asyncio.Semaphore(limits.max_readers)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Liveness must not depend on screening capacity. Other routes fail fast
        # when full instead of retaining an unbounded number of waiting bodies.
        if scope.get("method") == "GET" and scope["path"] == "/health/live":
            return await self.app(scope, receive, send)
        if self.readers.locked():
            return await JSONResponse(
                {"detail": "Request capacity reached"}, status_code=503
            )(scope, receive, send)
        await self.readers.acquire()
        try:
            await self._handle_http(scope, receive, send)
        finally:
            self.readers.release()

    async def _handle_http(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        admin = scope["path"].startswith("/admin/")
        if admin:
            authorization = headers.get(b"authorization", b"").decode("latin-1")
            parts = authorization.split(None, 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                return await JSONResponse(
                    {"detail": "Not authenticated"}, status_code=403
                )(scope, receive, send)
            expected = self.get_admin_key()
            if (
                not expected
                or len(expected) < 32
                or not secrets.compare_digest(parts[1], expected)
            ):
                return await JSONResponse(
                    {"detail": "Invalid credentials"}, status_code=401
                )(scope, receive, send)
        limit = self.max_upload_bytes if admin else self.max_body_bytes
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse(
                {"detail": "Invalid Content-Length"}, status_code=400
            )(scope, receive, send)
        if declared < 0 or declared > limit:
            return await JSONResponse(
                {"detail": "Request body too large"}, status_code=413
            )(scope, receive, send)
        chunks = []
        total = 0
        try:
            async with asyncio.timeout(self.body_timeout):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body = message.get("body", b"")
                    total += len(body)
                    if total > limit:
                        return await JSONResponse(
                            {"detail": "Request body too large"}, status_code=413
                        )(scope, receive, send)
                    chunks.append(body)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse(
                {"detail": "Request body timeout"}, status_code=408
            )(scope, receive, send)
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {
                    "type": "http.request",
                    "body": b"".join(chunks),
                    "more_body": False,
                }
            return await receive()

        # Admin responses schedule persistent ingestion jobs. They retain their
        # own lifecycle; a response deadline must not abort accepted ingestion.
        if admin:
            return await self.app(scope, bounded_receive, send)

        deadline = asyncio.timeout(self.processing_timeout)
        async def deadline_send(message):
            if message["type"] == "http.response.start":
                # JSON response preparation is complete. Do not cancel an
                # already-started response or its subsequent background tasks.
                deadline.reschedule(None)
            await send(message)

        try:
            async with deadline:
                await self.app(scope, bounded_receive, deadline_send)
        except TimeoutError:
            if not deadline.expired():
                raise
            return await JSONResponse(
                {"detail": "Request processing timed out"}, status_code=503
            )(scope, receive, send)
