"""Source metadata and accepted tax-identifier types must share their field names."""

import pytest

from ai_service.layers.search.search_integrity import source_tax_ids


@pytest.mark.parametrize("field", [
    "inn", "inn_ua", "inn_ru", "itn", "tax_id", "tin", "edrpou",
    "itn_import", "tax_number", "taxpayer_id",
])
def test_source_identifier_alias_preserves_exact_value(field):
    assert source_tax_ids({field: "001234567890"}) == {"001234567890"}


def test_source_identifiers_do_not_read_unrelated_fields_or_match_longer_numbers():
    assert source_tax_ids({"name": "001234567890", "tax_id": "00123456789099"}) == set()
