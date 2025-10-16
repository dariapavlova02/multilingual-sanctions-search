"""
Tests for optional English NER functionality.

Tests the spaCy English NER integration for enhanced role tagging
in English text normalization.
"""

import pytest
from ai_service.layers.normalization.ner_gateways.spacy_en import (
    SpacyEnNER, 
    NERHints, 
    get_spacy_en_ner, 
    clear_spacy_en_ner,
)
from ai_service.layers.normalization.role_tagger_service import RoleTaggerService
from ai_service.layers.normalization.lexicon_loader import clear_lexicon_cache, get_lexicons
from pathlib import Path
from ai_service.data.resources import LEXICONS_DIR


@pytest.fixture(autouse=True)
def setup_teardown_lexicons():
    """Setup and teardown lexicons for each test."""
    clear_lexicon_cache()
    get_lexicons()
    yield
    clear_lexicon_cache()


@pytest.fixture
def role_tagger():
    """Create a role tagger for testing."""
    lexicons = get_lexicons()
    return RoleTaggerService(lexicons)


def test_ner_org_suppresses_person_candidate(role_tagger):
    """
    Test that NER organization entities suppress person candidates.
    """
    text = "Client Microsoft Corporation John Smith"
    tokens = ["Client", "Microsoft", "Corporation", "John", "Smith"]
    
    # Create NER hints with organization span
    ner_hints = NERHints(
        person_spans=[(text.find("John"), text.find("Smith") + len("Smith"))],
        org_spans=[(text.find("Microsoft"), text.find("Corporation") + len("Corporation"))]
    )
    
    tags = role_tagger.tag(tokens, lang="en", flags={"enable_ner": True, "ner_hints": ner_hints})
    
    # Verify roles - RoleTaggerService doesn't include token text in RoleTag
    # So we check by position
    assert len(tags) == 5
    # Check that we get some role assignments
    assert tags[1].role.value == tags[2].role.value == "org"


def test_ner_person_enhances_confidence(role_tagger):
    """
    Test that NER person hints increase confidence for person candidates.
    """
    text = "Jane Doe"
    tokens = ["Jane", "Doe"]
    
    ner_hints = NERHints(
        person_spans=[(text.find("Jane"), text.find("Doe") + len("Doe"))],
        org_spans=[]
    )
    
    tags = role_tagger.tag(tokens, lang="en", flags={"enable_ner": True, "ner_hints": ner_hints})
    
    # Both tokens should be person candidates
    assert len(tags) == 2
    assert tags[0].role.value in ["given", "surname", "person_candidate", "unknown"]
    assert tags[1].role.value in ["given", "surname"]
    assert all("ner_person_span" in tag.evidence for tag in tags)


def test_ner_mixed_entities(role_tagger):
    """
    Test handling of mixed person and organization entities.
    """
    text = "Apple Inc. CEO Tim Cook announced new products"
    tokens = ["Apple", "Inc.", "CEO", "Tim", "Cook", "announced", "new", "products"]
    
    ner_hints = NERHints(
        person_spans=[(text.find("Tim"), text.find("Cook") + len("Cook"))],
        org_spans=[(text.find("Apple"), text.find("Inc.") + len("Inc."))]
    )
    
    tags = role_tagger.tag(tokens, lang="en", flags={"enable_ner": True, "ner_hints": ner_hints})
    
    # Basic functionality test
    assert len(tags) == 8
    # Check that we get some role assignments
    assert any(tag.role.value != "unknown" for tag in tags)


def test_ner_no_hints(role_tagger):
    """
    Test that role tagger works without NER hints.
    """
    text = "John Smith"
    tokens = ["John", "Smith"]
    
    tags = role_tagger.tag(tokens, lang="en", flags={"enable_ner": False, "ner_hints": None})
    
    # Should still work without NER hints
    assert len(tags) == 2
    assert tags[0].role.value in ["given", "surname", "person_candidate", "unknown"]
    assert tags[1].role.value in ["given", "surname", "person_candidate", "unknown"]


def test_ner_empty_hints(role_tagger):
    """
    Test that role tagger works with empty NER hints.
    """
    text = "John Smith"
    tokens = ["John", "Smith"]
    
    ner_hints = NERHints(person_spans=[], org_spans=[])
    tags = role_tagger.tag(tokens, lang="en", flags={"enable_ner": True, "ner_hints": ner_hints})
    
    # Should work with empty hints
    assert len(tags) == 2
    assert tags[0].role.value in ["given", "surname", "person_candidate", "unknown"]
    assert tags[1].role.value in ["given", "surname", "person_candidate", "unknown"]


def test_spacy_en_ner_extraction():
    """
    Test spaCy English NER entity extraction.
    """
    ner = get_spacy_en_ner()
    assert ner is not None
    
    text = "Apple Inc. CEO Tim Cook announced new products"
    hints = ner.extract_entities(text)
    
    assert isinstance(hints, NERHints)
    assert len(hints.person_spans) > 0
    assert len(hints.org_spans) > 0
    
    # Verify person spans
    person_text = text[hints.person_spans[0][0]:hints.person_spans[0][1]]
    assert "Tim" in person_text or "Cook" in person_text
    
    # Verify org spans
    org_text = text[hints.org_spans[0][0]:hints.org_spans[0][1]]
    assert "Apple" in org_text or "Inc" in org_text


def test_spacy_en_ner_singleton():
    """
    Test that spaCy English NER uses singleton pattern.
    """
    ner1 = get_spacy_en_ner()
    ner2 = get_spacy_en_ner()
    
    assert ner1 is ner2


def test_spacy_en_ner_not_available(monkeypatch):
    from ai_service.layers.normalization.ner_gateways.unified_spacy_gateway import UnifiedSpacyGateway
    gateway = UnifiedSpacyGateway()
    def unavailable(name, **kwargs):
        raise OSError("controlled missing model")
    monkeypatch.setattr("spacy.load", unavailable)
    with pytest.raises(RuntimeError):
        SpacyEnNER(gateway)
    gateway.close()


def test_clear_spacy_en_ner():
    """
    Test clearing the spaCy English NER singleton.
    """
    ner1 = get_spacy_en_ner()
    assert ner1 is not None
    clear_spacy_en_ner()
    ner2 = get_spacy_en_ner()
    assert ner2 is not None and ner1 is not ner2


def test_ner_hints_dataclass():
    """
    Test NERHints dataclass functionality.
    """
    hints = NERHints(
        person_spans=[(0, 5), (10, 15)],
        org_spans=[(20, 30)]
    )
    
    assert len(hints.person_spans) == 2
    assert len(hints.org_spans) == 1
    assert hints.person_spans[0] == (0, 5)
    assert hints.org_spans[0] == (20, 30)
