"""Formatting cleanup must preserve exact raw spans and per-entity ownership."""

import pytest

from ai_service.layers.search.search_integrity import source_tax_ids
from ai_service.layers.signals.extractors.birthdate_extractor import BirthdateExtractor
from ai_service.layers.signals.extractors.identifier_extractor import IdentifierExtractor
from ai_service.layers.signals.signals_service import PersonSignal, SignalsService
from ai_service.utils.source_text_view import SourceTextView


@pytest.mark.parametrize('control', ['\u200b', '\u2066', '\u2069', '\x80'])
def test_obfuscated_ids_and_dates_retain_exact_source_spans(control):
    text = f'Synthetic Example INN 00123{control}4567890 DOB 1980-{control}01-01'
    identifiers = IdentifierExtractor().extract_person_ids(text)
    assert len(identifiers) == 1
    assert identifiers[0]['value'] == '001234567890'
    dates = BirthdateExtractor().extract(text)
    assert len(dates) == 1 and dates[0]['iso_format'] == '1980-01-01'
    for evidence in [*identifiers, *dates]:
        start, end = evidence['position']
        assert evidence['raw'] == text[start:end]
        assert control in evidence['raw']


@pytest.mark.parametrize('separator', ['; ', '\n', ' | '])
def test_control_removal_keeps_identity_evidence_in_the_right_clause(separator):
    first = PersonSignal(core=['Synthetic', 'Example'], full_name='Synthetic Example')
    second = PersonSignal(core=['Other', 'Example'], full_name='Other Example')
    text = ('Synthetic Exa\u2066mple INN 12345\u200b67890 DOB 1980-\u206901-01'
        + separator + 'Other Exa\u2066mple INN 00123\u200b4567890 DOB 1990-\u206901-01')
    service = SignalsService()
    unassigned = service._enrich_with_identifiers(text, [first, second], [], None)
    service._enrich_with_birthdates(text, [first, second])
    assert not unassigned
    assert [[entry['value'] for entry in item.ids] for item in [first, second]] == [
        ['1234567890'], ['001234567890']]
    assert [first.dob, second.dob] == ['1980-01-01', '1990-01-01']
    for person in [first, second]:
        start, end = person.dob_position
        assert person.dob_raw == text[start:end]
        for entry in person.ids:
            start, end = entry['position']
            assert entry['raw'] == text[start:end]


def test_repeated_id_occurrences_are_not_discarded_before_ownership_resolution():
    text = 'First Example INN 1234567890; Second Example INN 1234567890'
    people = [PersonSignal(core=name.split(), full_name=name) for name in ['First Example', 'Second Example']]
    identifiers = IdentifierExtractor().extract_person_ids(text)
    assert len(identifiers) == 2
    assert identifiers[0]['position'] != identifiers[1]['position']
    assert SignalsService()._enrich_with_identifiers(text, people, [], None) == []
    assert all([entry['value'] for entry in person.ids] == ['1234567890'] for person in people)


def test_source_tax_ids_apply_the_same_formatting_policy_without_losing_leading_zeroes():
    source = {'itn': '00123\u20664567890'}
    assert source_tax_ids(source) == {'001234567890'}
    assert source['itn'] == '00123\u20664567890'


def test_matching_view_preserves_newlines_and_reversible_spans():
    view = SourceTextView.from_text('\u2066ab\u200bcd\u2069\nef')
    assert view.text == 'abcd\nef'
    assert view.original_span(1, 3) == (2, 5)
    assert view.original[2:5] == 'b\u200bc'
    assert view.matching_span(2, 5) == (1, 3)
    with pytest.raises(ValueError):
        view.original_span(0, 0)


def test_a_control_only_entity_name_cannot_claim_evidence():
    person = PersonSignal(core=['\u2066'], full_name='\u200b')
    unassigned = SignalsService()._enrich_with_identifiers('INN 1234567890', [person], [], None)
    assert not person.ids
    assert [entry['value'] for entry in unassigned] == ['1234567890']
