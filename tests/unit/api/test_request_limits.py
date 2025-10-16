import asyncio
import json

import pytest

from ai_service.api.request_limits import RequestLimitsMiddleware


def scope(path='/process', headers=(), method='POST'):
    return {'type': 'http', 'path': path, 'headers': list(headers), 'method': method}


async def empty_body():
    return {'type': 'http.request', 'body': b'', 'more_body': False}


async def respond(send):
    await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    await send({'type': 'http.response.body', 'body': b'{}'})


@pytest.mark.asyncio
async def test_chunked_body_limit_is_enforced_before_parsing():
    called = False
    sent = []
    chunks = iter([{"type": "http.request", "body": b"1234", "more_body": True},
                   {"type": "http.request", "body": b"5678", "more_body": False}])
    async def receive():
        return next(chunks)
    async def send(message):
        sent.append(message)
    async def app(*args):
        nonlocal called
        called = True
    middleware = RequestLimitsMiddleware(app, lambda: None, max_body_bytes=5)
    await middleware({"type": "http", "path": "/process", "headers": []}, receive, send)
    assert sent[0]["status"] == 413
    assert not called


@pytest.mark.asyncio
async def test_admin_authentication_happens_before_reading_upload():
    sent = []
    async def receive():
        pytest.fail("Unauthenticated body was read")
    async def send(message):
        sent.append(message)
    async def app(*args):
        pytest.fail("Unauthenticated request reached the application")
    middleware = RequestLimitsMiddleware(app, lambda: "test-key-" * 8)
    await middleware({"type": "http", "path": "/admin/ac-patterns/upload", "headers": []}, receive, send)
    assert sent[0]["status"] == 403


@pytest.mark.asyncio
async def test_processing_deadline_cancels_work_and_releases_http_capacity():
    cancelled = asyncio.Event()
    sent = []
    async def send(message):
        sent.append(message)
    async def slow_app(scope, receive, send):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    middleware = RequestLimitsMiddleware(slow_app, lambda: None, processing_timeout=0.02, max_readers=1)
    await middleware(scope(), empty_body, send)
    assert cancelled.is_set()
    assert sent[0]['status'] == 503
    assert json.loads(sent[-1]['body']) == {'detail': 'Request processing timed out'}
    async def fast_app(scope, receive, send):
        await respond(send)
    middleware.app = fast_app
    sent.clear()
    await middleware(scope(), empty_body, send)
    assert sent[0]['status'] == 200


@pytest.mark.asyncio
async def test_overload_rejects_before_body_read_but_liveness_is_available():
    entered, release = asyncio.Event(), asyncio.Event()
    async def app(scope, receive, send):
        if scope['path'] != '/health/live':
            entered.set()
            await release.wait()
        await respond(send)
    sent = []
    async def send(message):
        sent.append(message)
    async def unread_body():
        pytest.fail('Capacity-rejected request body was read')
    middleware = RequestLimitsMiddleware(app, lambda: None, max_readers=1)
    first = asyncio.create_task(middleware(scope(), empty_body, send))
    await entered.wait()
    try:
        for _ in range(50):
            sent.clear()
            await middleware(scope(), unread_body, send)
            assert sent[0]['status'] == 503
        sent.clear()
        await middleware(scope('/health/live', method='GET'), unread_body, send)
        assert sent[0]['status'] == 200
    finally:
        release.set()
        await first


@pytest.mark.asyncio
@pytest.mark.parametrize('admin', [False, True])
async def test_response_background_work_is_not_aborted_by_processing_deadline(admin):
    completed = False
    sent = []
    key = 'synthetic-admin-key-' * 3
    async def send(message):
        sent.append(message)
    async def app(scope, receive, send):
        nonlocal completed
        if admin:
            await asyncio.sleep(0.03)  # Includes pre-response ingestion setup.
        await respond(send)
        await asyncio.sleep(0.03)
        completed = True
    middleware = RequestLimitsMiddleware(app, lambda: key, processing_timeout=0.01)
    headers = [(b'authorization', ('Bearer ' + key).encode())] if admin else []
    await middleware(scope('/admin/ac-patterns/bulk' if admin else '/process', headers), empty_body, send)
    assert completed
    assert [message['status'] for message in sent if message['type'] == 'http.response.start'] == [200]


@pytest.mark.asyncio
async def test_application_timeout_error_is_not_misreported_as_middleware_deadline():
    async def app(scope, receive, send):
        raise TimeoutError('Synthetic internal timeout')
    async def send(message):
        pytest.fail('Middleware replaced an unrelated application exception')
    middleware = RequestLimitsMiddleware(app, lambda: None)
    with pytest.raises(TimeoutError, match='Synthetic internal timeout'):
        await middleware(scope(), empty_body, send)


@pytest.mark.asyncio
async def test_http_deadline_does_not_allow_overlapping_model_calls():
    import threading
    from ai_service.utils.inference_queue import InferenceQueue, InferenceUnavailableError
    from fastapi import FastAPI, HTTPException
    from httpx import ASGITransport, AsyncClient

    entered, release = threading.Event(), threading.Event()
    queue = InferenceQueue(max_pending=0, timeout=1)
    def encode():
        entered.set()
        assert release.wait(3)
        return [1.0]
    app = FastAPI()
    @app.get('/encode')
    async def endpoint():
        try:
            return await queue.run_async(encode)
        except InferenceUnavailableError:
            raise HTTPException(status_code=503, detail='Model capacity reached')
    # Match the production combination of ASGI limits and HTTP middleware.
    @app.middleware('http')
    async def passthrough(request, call_next):
        return await call_next(request)
    app.add_middleware(RequestLimitsMiddleware, get_admin_key=lambda: None,
                       processing_timeout=0.03, max_readers=1)
    try:
        async with AsyncClient(transport=ASGITransport(app), base_url='http://test') as client:
            first = await client.get('/encode')
            assert first.status_code == 503
            assert first.json()['detail'] == 'Request processing timed out'
            assert entered.is_set()
            assert queue.snapshot()['active'] == 1
            second = await client.get('/encode')
            assert second.status_code == 503
            assert second.json()['detail'] == 'Model capacity reached'
    finally:
        release.set()
        queue.close()


@pytest.mark.parametrize('environment,value', [
    ('MAX_REQUEST_BYTES', '0'), ('MAX_UPLOAD_BYTES', '-1'),
    ('HTTP_BODY_TIMEOUT_SECONDS', 'nan'), ('HTTP_PROCESSING_TIMEOUT_SECONDS', '0'),
    ('HTTP_PROCESSING_TIMEOUT_SECONDS', 'inf'), ('HTTP_MAX_INFLIGHT', '0'),
])
def test_invalid_http_environment_limits_fail_startup(monkeypatch, environment, value):
    from ai_service.api.request_limits import RequestLimitsConfig
    from pydantic import ValidationError
    monkeypatch.setenv(environment, value)
    with pytest.raises(ValidationError):
        RequestLimitsConfig()
