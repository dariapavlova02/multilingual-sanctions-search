"""Search text cleanup must be stable without corrupting visible source letters."""

import unicodedata

import pytest

from ai_service.layers.unicode.unicode_service import UnicodeService
from ai_service.utils.input_validation import InputValidator


@pytest.mark.parametrize('control', ['\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad',
    '\u202e', '\u2066', '\u2069', '\x00', '\x80', '\x9f'])
@pytest.mark.parametrize('text', ['Synthetic Example INN 001234567890', 'синтетичний приклад', 'контрольная строка'])
def test_injected_format_and_control_codes_do_not_change_visible_search_key(control, text):
    service = UnicodeService()
    position = len(text) // 2
    obfuscated = text[:position] + control + text[position:]
    result = service.normalize_text(obfuscated)
    assert result['original'] == obfuscated
    assert result['normalized'] == service.normalize_text(text)['normalized']
    assert service.normalize_text(result['normalized'])['normalized'] == result['normalized']


@pytest.mark.parametrize('text', ['Café José', 'Іван Їжак Йосип Ґанок', 'Иван Йосиф', 'שלום محمد'])
def test_canonically_equivalent_texts_have_the_same_key(text):
    service = UnicodeService()
    composed = service.normalize_text(unicodedata.normalize('NFC', text))['normalized']
    decomposed = service.normalize_text(unicodedata.normalize('NFD', text))['normalized']
    assert composed == decomposed


def test_cleanup_preserves_non_latin_letters_case_and_numeric_evidence():
    text = 'Іван Їжак Ґанок Йосип שלום محمد INN 001234567890 DOB 1980-01-01'
    assert UnicodeService().normalize_text(text)['normalized'] == text


@pytest.mark.parametrize('aggressive', [False, True])
def test_emoji_policy_is_applied_even_when_input_is_already_nfc(aggressive):
    service = UnicodeService()
    result = service.normalize_text('synthetic 🌟 example', aggressive=aggressive)['normalized']
    assert ('🌟' not in result) if aggressive else ('🌟' in result)
    assert service.normalize_text(result, aggressive=aggressive)['normalized'] == result


def test_typographic_quotes_are_real_codepoint_mappings():
    result = UnicodeService().normalize_text('O\u2019Connor \u201cExample\u201d')['normalized']
    assert result == 'O\'Connor "Example"'


def test_whitespace_cleanup_is_stable_after_control_removal():
    service = UnicodeService()
    first = service.normalize_text(' \u200b  Example\t\n \u2066  Value  \u2069 ')['normalized']
    assert first == 'Example Value'
    assert service.normalize_text(first)['normalized'] == first


@pytest.mark.parametrize('text,expected', [('Ñandú', 'Ñandu'), ('\x80Ñ', 'Ñ'), ('И\u200b\u0306', 'Й')])
def test_decoded_unicode_does_not_invent_an_encoding_or_destroy_composed_letters(text, expected):
    service = UnicodeService()
    result = service.normalize_text(text)['normalized']
    assert result == expected
    assert service.normalize_text(result)['normalized'] == expected


@pytest.mark.parametrize('text', ['\u200b\u200d', '\u2066\u2069', '\x80\x00'])
def test_control_only_input_cannot_be_validated_as_successful_content(text):
    result = InputValidator().validate_and_sanitize(text)
    assert result.is_valid is False and result.sanitized_text == ''
