"""Processing options must control generated values and cached results."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.contracts.base_contracts import NormalizationResult, ProcessingContext, SignalsResult
from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.utils.feature_flags import FeatureFlags


class MemoryCache:
    def __init__(self):
        self.values={}
    def get(self,key):
        return self.values.get(key)
    def set(self,key,value):
        self.values[key]=value


def isolated_orchestrator():
    service=UnifiedOrchestrator(validation_service=SimpleNamespace(),language_service=SimpleNamespace(),
        unicode_service=SimpleNamespace(),normalization_service=SimpleNamespace(),signals_service=SimpleNamespace(),
        enable_search=False,enable_smart_filter=False,enable_variants=False,enable_embeddings=False,
        enable_decision_engine=False)
    service._handle_validation_layer=AsyncMock(return_value=None)
    service._handle_smart_filter_layer=AsyncMock(return_value=None)
    async def detect(context,hint):
        context.language=hint or 'en';context.language_confidence=1.0
    service._handle_language_detection_layer=detect
    service._handle_unicode_normalization_layer=AsyncMock(return_value='Synthetic Example')
    service._handle_name_normalization_layer=AsyncMock(return_value=NormalizationResult(
        normalized='Synthetic Example',tokens=['Synthetic','Example'],trace=[]))
    service._handle_signals_layer=AsyncMock(return_value=SignalsResult(confidence=0.0))
    service.variants_service=SimpleNamespace(generate_variants=AsyncMock(return_value=['Synthetic Example','Synthetic E.']))
    service.embeddings_service=SimpleNamespace(generate_embeddings=AsyncMock(return_value=[1.0,0.0]))
    service.cache_service=MemoryCache()
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [
    {'language_hint':'uk'}, {'clean_unicode':False}, {'enable_advanced_features':False},
    {'feature_flags':FeatureFlags(strict_stopwords=True)}, {'generate_variants':True},
    {'generate_embeddings':True}, {'screen':True},
])
async def test_changed_processing_options_do_not_reuse_an_incompatible_result(changed):
    service=isolated_orchestrator()
    base=dict(screen=False,cache_result=True,generate_variants=False,generate_embeddings=False)
    first=await service.process('Synthetic Example',**base)
    assert first.success
    service._handle_name_normalization_layer.return_value=NormalizationResult(
        normalized='Recomputed Example',tokens=['Recomputed','Example'],trace=[])
    second=await service.process('Synthetic Example',**{**base,**changed})
    assert second.success and second.normalized_text=='Recomputed Example'
    if changed.get('generate_variants'):
        assert second.variants==['Synthetic Example','Synthetic E.']
    if changed.get('generate_embeddings'):
        assert second.embeddings==[1.0,0.0]


@pytest.mark.asyncio
async def test_equivalent_options_hit_cache_without_sharing_mutable_results():
    service=isolated_orchestrator()
    options=dict(screen=False,cache_result=True,generate_variants=True,generate_embeddings=False)
    first=await service.process('Synthetic Example',**options)
    first.variants.append('Caller mutation')
    second=await service.process('Synthetic Example',**options)
    assert second.variants==['Synthetic Example','Synthetic E.']
    assert service._handle_name_normalization_layer.await_count==1
    second.variants.clear()
    third=await service.process('Synthetic Example',**options)
    assert third.variants==['Synthetic Example','Synthetic E.']


@pytest.mark.asyncio
async def test_shared_cache_cannot_mix_different_service_instances():
    first=isolated_orchestrator();second=isolated_orchestrator()
    second.cache_service=first.cache_service
    options=dict(screen=False,cache_result=True,generate_variants=False,generate_embeddings=False)
    await first.process('Synthetic Example',**options)
    second._handle_name_normalization_layer.return_value=NormalizationResult(normalized='Other Policy',tokens=[],trace=[])
    assert (await second.process('Synthetic Example',**options)).normalized_text=='Other Policy'


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['variants','embeddings'])
async def test_requested_unavailable_generation_cannot_report_success(stage):
    service=isolated_orchestrator()
    setattr(service,stage+'_service',None)
    result=await service.process('Synthetic Example',screen=False,**{'generate_'+stage:True})
    assert result.success is False
    assert getattr(result,stage) is None and result.errors


@pytest.mark.asyncio
@pytest.mark.parametrize('generated', [None, {'count':2}, {'variants':['Valid',123]}])
async def test_malformed_variant_payload_is_failed_not_dictionary_keys(generated):
    service=isolated_orchestrator()
    service.variants_service.generate_variants.return_value=generated
    errors=[]
    variants=await service._handle_variants_layer(NormalizationResult(normalized='Synthetic Example',tokens=[],trace=[]),
        ProcessingContext(original_text='Synthetic Example',language='en'),True,errors)
    assert variants is None and errors


@pytest.mark.asyncio
async def test_variant_dictionary_yields_the_actual_list():
    service=isolated_orchestrator()
    service.variants_service.generate_variants.return_value={'variants':['One','Two'],'count':2,'language':'en'}
    result=await service.process('Synthetic Example',screen=False,generate_variants=True)
    assert result.success and result.variants==['One','Two']
