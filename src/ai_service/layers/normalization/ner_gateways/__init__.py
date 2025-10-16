"""One NER contract and model owner, with language-specific public facades."""
from .unified_spacy_gateway import (
    UnifiedSpacyGateway, SupportedLanguage, NEREntity, NERHints, NERUnavailableError,
    get_global_gateway, close_global_gateway, create_russian_gateway,
    create_ukrainian_gateway, create_english_gateway,
)
from .spacy_en import SpacyEnNER, get_spacy_en_ner, clear_spacy_en_ner
from .spacy_ru import SpacyRuNER, get_spacy_ru_ner, clear_spacy_ru_ner
from .spacy_uk import SpacyUkNER, get_spacy_uk_ner, clear_ner_cache as clear_spacy_uk_ner

NERHintsRu = NERHints
NERHintsEn = NERHints


def clear_ner_cache():
    clear_spacy_en_ner()
    clear_spacy_ru_ner()
    clear_spacy_uk_ner()
