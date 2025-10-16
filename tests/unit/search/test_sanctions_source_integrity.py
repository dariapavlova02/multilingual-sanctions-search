import json

import pytest

from ai_service.layers.search.sanctions_data_loader import SanctionsDataLoader


@pytest.mark.asyncio
async def test_empty_sources_do_not_invent_sanctions(tmp_path):
    loader = SanctionsDataLoader(tmp_path, cache_dir=tmp_path / "cache")
    result = await loader.load_dataset()
    assert result.total_entries == 0
    assert result.all_names == []
    assert result.sources == []


@pytest.mark.asyncio
async def test_alias_collisions_preserve_entities_and_source_changes_invalidate_cache(tmp_path):
    source = tmp_path / "sanctioned_persons.json"
    source.write_text(json.dumps([
        {"id": 1, "name": "Alpha", "aliases": ["Shared"]},
        {"id": 2, "name": "Beta", "aliases": ["Shared"]},
    ]))
    loader = SanctionsDataLoader(tmp_path, cache_dir=tmp_path / "cache")
    first = await loader.load_dataset()
    assert first.all_names == ["Alpha", "Beta", "Shared"]
    assert len(await loader.find_entries("Shared")) == 2
    restored = await SanctionsDataLoader(tmp_path, cache_dir=tmp_path / "cache").load_dataset()
    assert len(restored.name_to_entries["Shared"]) == 2
    source.write_text(json.dumps([{"id": 3, "name": "Gamma"}]))
    updated = await loader.load_dataset()
    assert updated.all_names == ["Gamma"]
    assert updated.source_manifest != first.source_manifest


@pytest.mark.asyncio
async def test_corrupted_source_cannot_be_reported_as_success(tmp_path):
    (tmp_path / "sanctioned_companies.json").write_text("invalid json")
    loader = SanctionsDataLoader(tmp_path, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="Failed to load sanctions source"):
        await loader.load_dataset()


@pytest.mark.asyncio
async def test_bundled_sources_are_available_and_demo_is_excluded(tmp_path):
    from ai_service.data.resources import PACKAGE_DATA_DIR

    loader = SanctionsDataLoader(cache_dir=tmp_path)
    dataset = await loader.load_dataset(force_reload=True)
    expected = sum(len(json.loads((PACKAGE_DATA_DIR / name).read_text())) for name in (
        "sanctioned_persons.json", "sanctioned_companies.json", "terrorism_black_list.json"
    ))
    assert dataset.total_entries == expected
    assert all(e.source != "sample" for e in dataset.persons + dataset.organizations)
    assert len(dataset.source_manifest) == 3
