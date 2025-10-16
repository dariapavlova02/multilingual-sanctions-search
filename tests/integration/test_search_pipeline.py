"""Exercise real retrieval/model/index contracts on a disposable source snapshot.

Synthetic hit/miss checks do not measure population recall or compare a legacy model.
"""

import json
import math
import time

import pytest

from ai_service.contracts.base_contracts import NormalizationResult
from ai_service.contracts.trace_models import SearchTrace
from ai_service.layers.search.contracts import SearchMode, SearchOpts

pytestmark = [pytest.mark.integration, pytest.mark.docker]


def normalized(text):
    return NormalizationResult(normalized=text, tokens=text.split(), trace=[], language='en', confidence=1.0)


class TestSearchPipelineIntegration:
    @pytest.mark.parametrize('mode', [SearchMode.AC, SearchMode.FUZZY, SearchMode.VECTOR, SearchMode.HYBRID])
    async def test_known_name_uses_active_source_and_trace(self, active_screening, mode):
        trace = SearchTrace(enabled=True)
        results = await active_screening.search.find_candidates(normalized('Replacement Example'),
            'Replacement Example', SearchOpts(search_mode=mode, threshold=0.99), trace)
        assert len(results) == 1
        hit = results[0]
        assert hit.metadata['entity_id'] == 'primary'
        assert hit.metadata['source'] == 'synthetic-regression'
        assert hit.metadata['tax_id'] == '1234567890'
        assert trace.steps and any(h.doc_id == hit.doc_id for step in trace.steps for h in step.hits)
        assert json.loads(json.dumps(hit.to_dict()))['doc_id'] == hit.doc_id
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in [hit.score, hit.confidence])

    async def test_company_type_and_country_filters(self, active_screening):
        opts = SearchOpts(search_mode=SearchMode.AC, entity_types=['organization'], metadata_filters={'country':'GB'})
        result = await active_screening.search.find_candidates(normalized('Example Trading LLC'), 'Example Trading LLC', opts)
        assert len(result) == 1 and result[0].metadata['entity_id'] == 'company'
        assert result[0].entity_type == 'organization'
        excluded = await active_screening.search.find_candidates(normalized('Example Trading LLC'),
            'Example Trading LLC', opts.model_copy(update={'metadata_filters':{'country':'UA'}}))
        assert excluded == []

    async def test_fuzzy_spelling_error_keeps_source_identity(self, active_screening):
        results = await active_screening.search.find_candidates(normalized('Replacment Example'),
            'Replacment Example', SearchOpts(search_mode=SearchMode.FUZZY, threshold=0.85))
        assert len(results) == 1 and results[0].metadata['entity_id'] == 'primary'
        assert results[0].search_mode == SearchMode.FUZZY

    async def test_hybrid_empty_result_executes_real_vector_escalation(self, active_screening):
        trace = SearchTrace(enabled=True)
        results = await active_screening.search.find_candidates(normalized('Quasar Unlisted'),
            'Quasar Unlisted', SearchOpts(search_mode=SearchMode.HYBRID, threshold=0.99), trace)
        assert results == []
        assert active_screening.search.get_metrics().vector_requests == 1
        assert trace.get_stage_steps('SEMANTIC')

    async def test_identifier_metadata_matches_exactly_with_leading_zeroes(self, active_screening):
        results = await active_screening.search.find_by_identifier('001234567890', 'tax_id')
        assert len(results) == 1 and results[0].metadata['entity_id'] == 'secondary'
        assert results[0].metadata['tax_id'] == '001234567890'
        assert await active_screening.search.find_by_identifier('01234567890', 'tax_id') == []

    async def test_same_entity_id_in_different_sources_remains_distinct(self, active_screening):
        from ai_service.api.admin_endpoints import _load_documents, loading_status
        from ai_service.layers.search.index_schema import pattern_document
        other = {**active_screening.rows[0], 'source_list': 'another-synthetic-source'}
        await _load_documents([pattern_document(other, 'person', 0)], 'ac_patterns')
        assert loading_status['ac_patterns']['status'] == 'completed'
        result = await active_screening.search.find_candidates(normalized('Replacement Example'),
            'Replacement Example', SearchOpts(search_mode=SearchMode.AC, threshold=0.99))
        assert len(result) == 2
        assert {hit.metadata['source'] for hit in result} == {'synthetic-regression','another-synthetic-source'}
        assert {hit.metadata['entity_id'] for hit in result} == {'primary'}
        # AC upsert invalidated the previous vector generation.
        with pytest.raises(RuntimeError, match='coherent snapshot'):
            await active_screening.search.readiness(require_vectors=True)

    async def test_generation_change_invalidates_cached_metadata(self, active_screening):
        from ai_service.api.admin_endpoints import _load_documents, loading_status
        from ai_service.layers.search.index_schema import pattern_document
        opts = SearchOpts(search_mode=SearchMode.AC, threshold=0.99)
        service = active_screening.search
        first = await service.find_candidates(normalized('Replacement Example'), 'Replacement Example', opts)
        assert first[0].metadata['dob'] == '1980-01-01'
        changed = {**active_screening.rows[0], 'metadata':{'tax_id':'1234567890','dob':'1981-01-01'}}
        await _load_documents([pattern_document(changed, 'person', 0)], 'ac_patterns')
        assert loading_status['ac_patterns']['status'] == 'completed'
        result = await service.find_candidates(normalized('Replacement Example'), 'Replacement Example', opts)
        assert result[0].metadata['dob'] == '1981-01-01'
        assert first[0].metadata['dob'] == '1980-01-01'

    @pytest.mark.performance
    async def test_search_performance_sla(self, active_screening, record_property):
        """Keep the existing 50 ms local search budget, now with actual I/O/model work."""
        latencies = {}
        for query in ['Replacement Example', 'Replacement Alias', 'Example Trading LLC',
                      'Quasar Unlisted', 'Distant Other Entity', 'Replacment Example']:
            await active_screening.search.clear_search_cache()
            started = time.perf_counter()
            await active_screening.search.find_candidates(normalized(query), query, SearchOpts())
            latencies[query] = (time.perf_counter() - started) * 1000
        record_property('search_latency_ms', json.dumps(latencies, sort_keys=True))
        assert max(latencies.values()) <= 50, latencies
        assert sum(latencies.values()) / len(latencies) <= 50, latencies
