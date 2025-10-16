"""Screen actual HTTP routes against an owned synthetic source and real models.

These are workflow/identity tests, not population precision or recall estimates.
No orchestrator, normalization, retrieval result or decision is mocked.
"""

import asyncio
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker]


@pytest.mark.parametrize("mode", ["ac", "fuzzy", "vector", "hybrid"])
async def test_search_routes_retain_source_identity_and_aliases(screening_api, mode):
    response = await screening_api.api.post('/search', json={
        'query': 'Replacement Example', 'search_mode': mode, 'threshold': 0.99})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['success'] is True and body['total_hits'] == 1
    hit, = body['results']
    assert hit['metadata']['source'] == 'synthetic-regression'
    assert hit['metadata']['entity_id'] == 'primary'
    assert hit['metadata']['tax_id'] == '1234567890'
    assert hit['metadata']['dob'] == '1980-01-01'
    assert body['normalized_query'] == 'Replacement Example'


@pytest.mark.parametrize('text', [
    'Replacement Example INN 1234567890 DOB 1980-01-01',
    'Replacement Exa\u200bmple INN 12345\u206667890 DOB 1980-\u206901-01',
])
async def test_confirmed_identifier_and_date_survive_entire_pipeline(screening_api, text):
    response = await screening_api.api.post('/process', json={'text': text, 'generate_variants': False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['success'] is True
    decision = body['decision']
    assert decision['risk_level'] == 'high' and decision['review_required'] is False
    features = decision['decision_details']['normalized_features']
    assert features['id_match'] is True and features['date_match'] is True
    person, = body['signals']['persons']
    item, = person['ids']
    assert item['value'] == '1234567890'
    assert item['raw'] == text[slice(*item['position'])]
    assert person['dob_raw'] == text[slice(*person['dob_position'])]


async def test_identifier_only_retains_leading_zeroes_and_requires_owner_review(screening_api):
    response = await screening_api.api.post('/process', json={'text': 'INN 001234567890', 'generate_variants': False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['normalized_text'] == ''
    assert body['signals']['persons'] == []
    identifier, = body['signals']['extras']['unassigned_ids']
    assert identifier['value'] == '001234567890'
    result, = body['search_results']['results']
    assert result['metadata']['entity_id'] == 'secondary'
    assert body['decision']['risk_level'] == 'high'
    assert body['decision']['review_required'] is True


async def test_same_surname_people_do_not_exchange_dates_or_identifiers(screening_api):
    text = 'J. Smith INN 1234567890 DOB 1990-01-01; A. Smith INN 001234567890 DOB 1980-01-01'
    response = await screening_api.api.post('/process', json={'text': text, 'generate_variants': False})
    assert response.status_code == 200, response.text
    body = response.json()
    people = body['signals']['persons']
    assert [p['full_name'] for p in people] == ['J. Smith', 'A. Smith']
    assert [[i['value'] for i in p['ids']] for p in people] == [['1234567890'], ['001234567890']]
    assert [p['dob'] for p in people] == ['1990-01-01', '1980-01-01']
    assert body['decision']['risk_level'] == 'high'
    assert body['decision']['review_required'] is True
    assert not any('TIN+DOB' in reason for reason in body['decision']['decision_reasons'])


@pytest.mark.parametrize('mode', ['ac', 'fuzzy', 'vector', 'hybrid'])
async def test_empty_result_is_from_completed_snapshot(screening_api, mode):
    response = await screening_api.api.post('/search', json={
        'query': 'Quasar Unlisted', 'search_mode': mode, 'threshold': 0.99})
    assert response.status_code == 200, response.text
    assert response.json()['results'] == []
    assert response.json()['success'] is True
    assert (await screening_api.api.get('/health/ready')).status_code == 200


@pytest.mark.parametrize('path', ['/search', '/process'])
async def test_warm_result_is_not_served_when_snapshot_becomes_incomplete(screening_api, path):
    payload = ({'query': 'Replacement Example', 'search_mode': 'ac'} if path == '/search'
        else {'text': 'INN 1234567890', 'generate_variants': False})
    first = await screening_api.api.post(path, json=payload)
    assert first.status_code == 200, first.text
    index = screening_api.config.elasticsearch.ac_index
    mappings = await screening_api.client.indices.get_mapping(index=index)
    meta = mappings[index]['mappings']['_meta']
    await screening_api.client.indices.put_mapping(index=index, _meta={**meta, 'ingestion_status': 'loading'})
    failed = await screening_api.api.post(path, json=payload)
    assert failed.status_code == 503, failed.text
    assert 'decision' not in failed.json() and 'results' not in failed.json()
    assert (await screening_api.api.get('/health/ready')).status_code == 503
    assert (await screening_api.api.get('/health/live')).status_code == 200
    await screening_api.client.indices.put_mapping(index=index, _meta=meta)
    recovered = await screening_api.api.post(path, json=payload)
    assert recovered.status_code == 200, recovered.text


async def test_batch_preserves_order_and_evidence_under_concurrency(screening_api):
    texts = ['INN 1234567890', 'Replacement Example INN 1234567890 DOB 1980-01-01',
             'INN 001234567890', 'J. Smith; INN 1234567890']
    started = time.perf_counter()
    response = await screening_api.api.post('/process-batch', json={
        'texts': texts, 'generate_variants': False, 'max_concurrent': 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['successful'] == len(texts)
    assert [r['original_text'] for r in body['results']] == texts
    assert [r['decision']['review_required'] for r in body['results']] == [True, False, True, True]
    assert all(r['decision']['risk_level'] == 'high' for r in body['results'])
    assert time.perf_counter() - started < 10, 'The four-row warm batch exceeded the existing 10 s workflow budget'


async def test_requested_generated_values_use_real_services(screening_api):
    response = await screening_api.api.post('/process', json={
        'text': 'Replacement Example', 'generate_variants': True, 'generate_embeddings': True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body['embedding']) == 384 and any(body['embedding'])
    assert body['variants'] and all(isinstance(v, str) for v in body['variants'])
    assert body['search_results']['total_hits'] == 1


@pytest.mark.parametrize('text', ['', '  ', '\u200b\u2066', 'Test ' * 10000],
    ids=['empty', 'whitespace', 'format-only', 'oversized'])
async def test_invalid_inputs_are_rejected_before_screening(screening_api, text):
    response = await screening_api.api.post('/process', json={'text': text, 'generate_variants': False})
    assert response.status_code == 422, response.text
    assert 'decision' not in response.json()


async def test_empty_name_embedding_request_cannot_publish_partial_decision(screening_api):
    response = await screening_api.api.post('/process', json={
        'text': 'INN 1234567890', 'generate_variants': False, 'generate_embeddings': True})
    assert response.status_code == 503, response.text
    assert 'decision' not in response.json()
