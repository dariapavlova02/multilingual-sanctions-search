"""The real variant service uses one bounded worker for sync and async calls."""

import asyncio
import threading

import pytest

from ai_service.layers.variants.config import VariantExecutionConfig
from ai_service.layers.variants.variant_generation_service import VariantGenerationService
from ai_service.utils.inference_queue import InferenceQueue, InferenceUnavailableError


@pytest.mark.asyncio
async def test_sync_async_variant_calls_share_capacity_and_cancellation_retains_active_slot():
    # Skip unrelated dictionary initialization; exercise the actual public methods.
    service=VariantGenerationService.__new__(VariantGenerationService)
    service._generation_queue=InferenceQueue(0,2,label='Variants')
    started=threading.Event();release=threading.Event();finished=threading.Event()
    def blocked(*args,**kwargs):
        started.set()
        try:
            assert release.wait(2)
            return {'variants':['Synthetic Example'],'count':1}
        finally:
            finished.set()
    service._generate_variants=blocked
    task=asyncio.create_task(service.generate_variants_async('Synthetic Example','en'))
    try:
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(.005)
        assert service.get_generation_stats()['active']==1
        with pytest.raises(InferenceUnavailableError,match='Variants capacity'):
            await service.generate_variants_async('Second Example','en')
        with pytest.raises(InferenceUnavailableError,match='Variants capacity'):
            service.generate_variants('Second Example','en')
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.get_generation_stats()['active']==1
        assert not finished.is_set()
    finally:
        release.set()
        service.close()
    async with asyncio.timeout(1):
        while service.get_generation_stats()['active']:
            await asyncio.sleep(.005)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_real_generator_async_output_is_a_bounded_string_list():
    service=VariantGenerationService()
    try:
        result=await service.generate_variants_async('Synthetic Example','en',max_variants=10)
        assert isinstance(result['variants'],list)
        assert 1<=result['count']==len(result['variants'])<=10
        assert all(isinstance(value,str) for value in result['variants'])
    finally:
        service.close()


@pytest.mark.parametrize('key,value',[('VARIANTS_MAX_PENDING','-1'),('VARIANTS_MAX_PENDING','129'),
    ('VARIANTS_TIMEOUT_SECONDS','0'),('VARIANTS_TIMEOUT_SECONDS','nan'),('VARIANTS_TIMEOUT_SECONDS','301')])
def test_invalid_variant_execution_limits_fail_validation(monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError):
        VariantExecutionConfig()


def test_typo_budget_prevents_building_all_full_length_alternatives():
    import json
    from pathlib import Path
    import subprocess
    import sys

    # Other services/tests can leave process-wide tracing active. Measure in a
    # fresh interpreter so their historical peak cannot change this result.
    script = '''import json, sys, tracemalloc
sys.path.insert(0, sys.argv[1])
from ai_service.layers.variants.variant_generation_service import VariantGenerationService
service = VariantGenerationService.__new__(VariantGenerationService)
service.typo_patterns = {}
text = 'а' * 2500
tracemalloc.start()
result = service._generate_typo_variants(text, max_typos=3)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({'count': len(result), 'peak': peak}))
'''
    completed = subprocess.run([sys.executable, '-c', script, str(Path(__file__).resolve().parents[3] / 'src')],
        capture_output=True, text=True, check=True, timeout=30)
    measured = json.loads(completed.stdout)
    assert measured['count'] == 3
    assert measured['peak'] < 256_000, 'Three variants should not require materializing thousands of long strings'
