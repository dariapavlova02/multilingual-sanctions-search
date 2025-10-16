"""A ready hybrid search must use one coherent AC/vector snapshot."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_service.layers.search.config import HybridSearchConfig
from ai_service.layers.search.hybrid_search_service import HybridSearchService
from ai_service.layers.search.index_schema import index_mapping, SOURCE_COVERAGE_VERSION


def service_with_snapshots(ac_generation, vector_generation, ac_manifest=None, vector_manifest=None):
    config = HybridSearchConfig()
    service = HybridSearchService(config)
    mappings = {}
    for index, vectors, generation, manifest in [
        (config.elasticsearch.ac_index, False, ac_generation, ac_manifest),
        (config.elasticsearch.vector_index, True, vector_generation, vector_manifest),
    ]:
        mapping = index_mapping(config, vectors=vectors)['mappings']
        mapping['_meta'].update(ingestion_status='completed', generation=generation)
        if vectors:
            mapping['_meta']['source_coverage_version'] = SOURCE_COVERAGE_VERSION
        if manifest is not None:
            mapping['_meta']['source_manifest'] = manifest
        mappings[index] = {'mappings': mapping}
    async def get_mapping(*, index):
        return {index: mappings[index]}
    client = SimpleNamespace(indices=SimpleNamespace(get_mapping=AsyncMock(side_effect=get_mapping)),
                             count=AsyncMock(return_value={'count': 1}))
    client.options = lambda **kwargs: client
    service._client_factory = SimpleNamespace(get_client=AsyncMock(return_value=client))
    service._initialized = True
    return service


@pytest.mark.asyncio
async def test_independently_completed_generations_are_not_hybrid_ready():
    service = service_with_snapshots('source-revision', 'different-revision')
    with pytest.raises(RuntimeError, match='generation|snapshot'):
        await service.readiness()


@pytest.mark.asyncio
async def test_ac_only_readiness_does_not_require_a_vector_generation():
    service = service_with_snapshots('source-revision', 'different-revision')
    assert await service.readiness(require_vectors=False) == {service.config.elasticsearch.ac_index: 'source-revision'}


@pytest.mark.asyncio
async def test_shared_generation_with_different_source_manifests_is_rejected():
    service = service_with_snapshots('same', 'same', [{'sha256': 'first'}], [{'sha256': 'second'}])
    with pytest.raises(RuntimeError, match='manifest|snapshot'):
        await service.readiness()


@pytest.mark.asyncio
async def test_matching_snapshot_is_ready():
    service = service_with_snapshots('same', 'same', [{'sha256': 'first'}], [{'sha256': 'first'}])
    assert set((await service.readiness()).values()) == {'same'}


@pytest.mark.asyncio
async def test_empty_mapping_is_not_ac_ready():
    service = service_with_snapshots('same', 'same')
    client = await service._client_factory.get_client()
    client.indices.get_mapping.side_effect = None
    client.indices.get_mapping.return_value = {}
    with pytest.raises(RuntimeError, match='concrete source snapshot'):
        await service.readiness(require_vectors=False)


@pytest.mark.asyncio
async def test_legacy_completed_snapshot_requires_coverage_verification():
    service = service_with_snapshots('same', 'same')
    client = await service._client_factory.get_client()
    original = client.indices.get_mapping.side_effect
    async def legacy_mapping(*, index):
        result = await original(index=index)
        result[index]['mappings']['_meta'].pop('source_coverage_version', None)
        return result
    client.indices.get_mapping.side_effect = legacy_mapping
    with pytest.raises(RuntimeError, match='source coverage verification'):
        await service.readiness()
