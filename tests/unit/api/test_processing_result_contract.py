"""Optional results and per-row screening evidence must survive API serialization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import ai_service.main as main
from ai_service.contracts.base_contracts import UnifiedProcessingResult


def processing_result(text='Synthetic Example', **kwargs):
    values=dict(original_text=text, normalized_text=text, language='en', language_confidence=1.0,
                tokens=text.split(), trace=[], variants=[text, 'Synthetic E.'], embeddings=[1.0, 0.0],
                success=True, search_results={'results':[{'doc_id':'source-one','score':1.0}]},
                decision=SimpleNamespace(risk=SimpleNamespace(value='high'), score=1.0,
                    reasons=['source_match'], details={'entity_id':'source-one'}, review_required=True,
                    required_additional_fields=[]))
    values.update(kwargs)
    return UnifiedProcessingResult(**values)


@pytest.fixture
def result_api(monkeypatch):
    result=processing_result()
    service=SimpleNamespace(process=AsyncMock(return_value=result),
        process_batch=AsyncMock(return_value=[result]), enable_search=True)
    monkeypatch.setattr(main,'orchestrator',service)
    return TestClient(main.app),service


@pytest.mark.parametrize('path', ['/process','/process-batch'])
@pytest.mark.parametrize('requested', [False,True])
def test_generation_flags_control_exposed_payload(result_api,path,requested):
    client,service=result_api
    body={'text':'Synthetic Example'} if path=='/process' else {'texts':['Synthetic Example']}
    body.update(generate_variants=requested,generate_embeddings=requested)
    response=client.post(path,json=body)
    assert response.status_code==200
    item=response.json() if path=='/process' else response.json()['results'][0]
    assert item['variants']==(['Synthetic Example','Synthetic E.'] if requested else None)
    assert item['embedding']==([1.0,0.0] if requested else None)
    assert item['decision']['risk_level']=='high'
    assert item['search_results']['results'][0]['doc_id']=='source-one'
    if path=='/process-batch':
        assert item['variants_count']==(2 if requested else 0)


def test_batch_preserves_row_evidence_and_suppresses_partial_failed_decision(result_api):
    client,service=result_api
    private='private-backend-diagnostic'
    service.process_batch.return_value=[processing_result(),processing_result('Second Example',
        success=False,errors=[private],normalized_text=private)]
    response=client.post('/process-batch',json={'texts':['Synthetic Example','Second Example']})
    assert response.status_code==200
    body=response.json()
    assert body['total_texts']==2 and body['successful']==1
    good,bad=body['results']
    assert good['original_text']=='Synthetic Example' and good['decision']['review_required'] is True
    assert bad['original_text']=='Second Example' and bad['success'] is False
    assert bad['normalized_text']=='' and bad['tokens']==[] and bad['trace']==[]
    assert all(bad[field] is None for field in ['decision','signals','search_results','variants','embedding'])
    assert bad['variants_count']==0 and private not in response.text


@pytest.mark.parametrize('results', [[], [processing_result('Different row')]])
def test_batch_contract_rejects_missing_or_misassociated_rows(result_api,results):
    client,service=result_api
    service.process_batch.return_value=results
    assert client.post('/process-batch',json={'texts':['Synthetic Example']}).status_code==500


def test_openapi_describes_the_same_result_fields_for_single_and_batch(result_api):
    client,_=result_api
    document=client.get('/openapi.json').json()
    schemas=document['components']['schemas']
    single=schemas['ProcessResponse']['properties']
    assert single['variants']
    ref=document['paths']['/process-batch']['post']['responses']['200']['content']['application/json']['schema']['$ref']
    batch=schemas[ref.rsplit('/',1)[-1]]
    item_ref=batch['properties']['results']['items']['$ref']
    item=schemas[item_ref.rsplit('/',1)[-1]]
    assert set(single).issubset(item['properties'])


@pytest.mark.parametrize('path', ['/process', '/process-batch'])
def test_source_spans_survive_real_signal_extraction_and_http_serialization(result_api, path):
    from ai_service.layers.signals.signals_service import SignalsService

    client, service = result_api
    text = 'Synthetic Exa\u2066mple INN 12345\u200b67890 DOB 1980-\u206901-01'
    extracted = SignalsService().extract(text,
        {'persons_core': [['Synthetic', 'Example']], 'organizations_core': []}, language='en')
    signals = SimpleNamespace(persons=[SimpleNamespace(**item) for item in extracted['persons']],
        organizations=[], confidence=extracted['confidence'])
    result = processing_result(text, signals=signals)
    service.process.return_value = result
    service.process_batch.return_value = [result]
    body = {'text': text} if path == '/process' else {'texts': [text]}
    response = client.post(path, json=body)
    assert response.status_code == 200
    item = response.json() if path == '/process' else response.json()['results'][0]
    assert len(item['signals']['persons']) == 1
    person = item['signals']['persons'][0]
    assert person['dob'] == '1980-01-01'
    start, end = person['dob_position']
    assert person['dob_raw'] == text[start:end]
    assert len(person['ids']) == 1
    identifier = person['ids'][0]
    assert identifier['value'] == '1234567890'
    start, end = identifier['position']
    assert identifier['raw'] == text[start:end]
