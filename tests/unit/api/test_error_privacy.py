"""Public errors and stored ingestion failures cannot disclose private values."""

import secrets
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main
from ai_service.api import admin_endpoints


@pytest.fixture
def private_api(monkeypatch):
    secret = secrets.token_hex(24)
    service = SimpleNamespace(
        process=AsyncMock(side_effect=RuntimeError(secret)),
        enable_search=True,
        enable_embeddings=True,
        search_service=SimpleNamespace(readiness=AsyncMock(side_effect=RuntimeError(secret))),
    )
    monkeypatch.setattr(main, 'orchestrator', service)
    return TestClient(main.app, raise_server_exceptions=False), service, secret


@pytest.mark.parametrize('path,payload,status', [
    ('/process', {'text': 'Private Example'}, 500),
    ('/normalize', {'text': 'Private Example'}, 500),
    ('/health/ready', None, 503),
])
def test_runtime_errors_are_private(private_api, path, payload, status):
    client, _, secret = private_api
    response = client.get(path) if payload is None else client.post(path, json=payload)
    assert response.status_code == status
    assert secret not in response.text


def test_detailed_health_requires_admin_credentials(private_api):
    client, _, _ = private_api
    assert client.get('/health/detailed').status_code == 403


def test_similarity_capacity_exhaustion_is_unavailable_not_success(private_api):
    from ai_service.utils.inference_queue import InferenceUnavailableError
    client, service, _ = private_api
    service.search_similar_names = AsyncMock(side_effect=InferenceUnavailableError('Embedding capacity reached'))
    response = client.post('/search-similar', json={'query': 'Synthetic Example', 'candidates': ['Other Example']})
    assert response.status_code == 503
    assert response.json()['detail'] == 'Embedding service temporarily unavailable'


def test_request_validation_does_not_echo_the_complete_input(private_api):
    client, _, secret = private_api
    response = client.post('/search', json={'query': 'Private Example', 'top_k': secret})
    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()['errors'][0]['loc'][-1] == 'top_k'


@pytest.mark.parametrize('path,payload', [
    ('/search', {'query': 'Synthetic Example', 'top_k': 'invalid'}),
    ('/normalize', {'text': 'Synthetic Example', 'language': 'unsupported'}),
    ('/process-batch', {'texts': [123]}),
])
def test_validation_response_matches_published_contract(private_api, path, payload):
    client, _, _ = private_api
    response = client.post(path, json=payload)
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {'detail', 'errors'}
    assert isinstance(body['detail'], str)
    assert body['errors']
    for issue in body['errors']:
        assert set(issue) == {'loc', 'msg', 'type'}
        assert isinstance(issue['loc'], list)
        assert all(isinstance(part, (str, int)) for part in issue['loc'])
        assert isinstance(issue['msg'], str)
        assert isinstance(issue['type'], str)

    document = client.get('/openapi.json').json()
    schema_ref = document['paths'][path]['post']['responses']['422']['content']['application/json']['schema']['$ref']
    schema = document['components']['schemas'][schema_ref.rsplit('/', 1)[-1]]
    assert schema['properties']['detail']['type'] == 'string'
    assert schema['properties']['detail']['const'] == body['detail']
    assert schema['properties']['errors']['type'] == 'array'
    issue_ref = schema['properties']['errors']['items']['$ref']
    issue_schema = document['components']['schemas'][issue_ref.rsplit('/', 1)[-1]]
    assert set(issue_schema['properties']) == set(body['errors'][0])


def test_validation_contract_also_covers_admin_router(private_api):
    client, _, _ = private_api
    document = client.get('/openapi.json').json()
    for path, routes in document['paths'].items():
        for operation in routes.values():
            if 'requestBody' in operation:
                schema = operation['responses']['422']['content']['application/json']['schema']
                assert schema == {'$ref': '#/components/schemas/RequestValidationResponse'}, path


def test_malformed_json_has_safe_validation_structure(private_api):
    client, _, secret = private_api
    response = client.post('/search', content='{"query": "' + secret, headers={'Content-Type': 'application/json'})
    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()['errors'][0]['type'] == 'json_invalid'


def test_internal_validation_is_not_reported_as_a_caller_error(private_api):
    from pydantic import BaseModel

    client, service, secret = private_api
    class InternalConfig(BaseModel):
        internal_port: int
    try:
        InternalConfig(internal_port=secret)
    except Exception as exc:
        service.process.side_effect = exc
    response = client.post('/process', json={'text': 'Private Example'})
    assert response.status_code == 500
    assert secret not in response.text


@pytest.mark.parametrize('path', ['/normalize', '/process'])
def test_failed_result_objects_do_not_bypass_error_privacy(private_api, path):
    client, service, secret = private_api
    service.process.side_effect = None
    service.process.return_value = SimpleNamespace(success=False, errors=[secret])
    response = client.post(path, json={'text': 'Private Example'})
    assert response.status_code in {500, 503}
    assert secret not in response.text


def test_batch_failed_item_does_not_return_internal_exception_details(private_api):
    client, service, secret = private_api
    service.process_batch = AsyncMock(return_value=[SimpleNamespace(
        success=False, errors=[secret], original_text='Private Example',
        normalized_text='', language='en', language_confidence=1.0,
        variants=[], processing_time=0.1,
    )])
    response = client.post('/process-batch', json={'texts': ['Private Example']})
    assert response.status_code == 200
    assert response.json()['results'][0]['success'] is False
    assert secret not in response.text


@pytest.mark.asyncio
async def test_ingestion_diagnostics_store_failure_type_without_payload(monkeypatch):
    secret = secrets.token_hex(24)
    job = SimpleNamespace(job_id='synthetic-job', update=Mock(), close=Mock())
    wrapper = SimpleNamespace(client=SimpleNamespace(), close=AsyncMock())
    monkeypatch.setattr(admin_endpoints, 'ElasticsearchClient', lambda: wrapper)
    monkeypatch.setattr(admin_endpoints, 'ensure_index', AsyncMock(side_effect=RuntimeError(secret)))
    monkeypatch.setattr(admin_endpoints, 'loading_status', {})
    await admin_endpoints._load_ac_patterns_background(
        [{'pattern': 'Synthetic Example', 'entity_id': 'synthetic'}], 'person', 'tier_0_exact', 1, job=job)
    status = admin_endpoints.loading_status['ac_patterns']
    assert status['status'] == 'error'
    assert 'RuntimeError' in status['error']
    assert secret not in repr(status)
    assert secret not in repr(job.update.call_args_list)
    job.close.assert_called_once()
    wrapper.close.assert_awaited_once()
