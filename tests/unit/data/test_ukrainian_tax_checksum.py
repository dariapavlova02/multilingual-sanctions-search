"""Checksum arithmetic is separate from source matching and registry ownership.

The first two cases are the contrasting published python-stdnum RNTRC examples:
https://github.com/arthurdejong/python-stdnum/blob/5d4ad17cae8abeab21f446b5569f85d185566330/stdnum/ua/rntrc.py
The remaining sparse-digit cases exercise leading zeroes and negative weights.
No example establishes that an identifier is registered to a person.
"""

import pytest

from ai_service.data.patterns.identifiers import _validate_ukrainian_inn, validate_inn
from ai_service.layers.normalization.normalization_service import NormalizationService
from ai_service.layers.signals.extractors.identifier_extractor import IdentifierExtractor
from ai_service.layers.signals.signals_service import SignalsService


CASES = [("1759013776", True), ("1759013770", False),
         ("0000000017", True), ("0000000011", False), ("9000000002", True)]


@pytest.mark.parametrize("value,expected", CASES)
def test_ukrainian_checksum_reference_and_arithmetic_boundaries(value, expected):
    assert _validate_ukrainian_inn(value) is expected
    assert validate_inn(value) is expected


@pytest.mark.parametrize("value", [None, 1759013776, "", "12345", "175901377600",
    "175901377A", "175901377²", "１７５９０１３７７６", "١٧٥٩٠١٣٧٧٦"])
def test_ukrainian_checksum_requires_ten_ascii_digits(value):
    assert _validate_ukrainian_inn(value) is False


@pytest.mark.parametrize("value,expected", CASES)
@pytest.mark.parametrize("controls", [False, True])
def test_extraction_retains_identifiers_even_when_the_checksum_is_invalid(value, expected, controls):
    text = "ІПН " + value
    if controls:
        text = text[:7] + "\u200b" + text[7:]
    item, = IdentifierExtractor().extract_person_ids(text)
    assert item["value"] == value
    assert item["valid"] is expected
    start, end = item["position"]
    assert item["raw"] == text[start:end]


@pytest.fixture(scope="module")
def normalization():
    return NormalizationService()


@pytest.mark.asyncio
@pytest.mark.parametrize("language,name,marker", [("ru", "И. Петров", "ИНН"),
    ("uk", "І. Коваленко", "ІПН"), ("en", "J. Smith", "INN")])
@pytest.mark.parametrize("value,expected", [("1759013776", True), ("1759013770", False)])
async def test_trace_and_regex_extraction_agree_without_discarding_invalid_ids(normalization, language, name, marker, value, expected):
    text = f"{name} {marker} {value}"
    norm = await normalization.normalize_async(text, language=language)
    result = await SignalsService().extract_signals(text, norm, language)
    person, = result.persons
    assert person.full_name == name
    item, = person.ids
    assert item["value"] == value and item["valid"] is expected
    start, end = item["position"]
    assert item["raw"] == text[start:end]
