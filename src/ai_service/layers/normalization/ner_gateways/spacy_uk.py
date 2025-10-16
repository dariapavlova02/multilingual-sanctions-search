"""Ukrainian facade over the shared NER implementation."""
from .language_adapter import LanguageNERAdapter
from .unified_spacy_gateway import NEREntity, NERHints, SupportedLanguage, get_global_gateway


class SpacyUkNER(LanguageNERAdapter):
    language = SupportedLanguage.UKRAINIAN

    def is_available(self):
        return self.gateway.is_available(self.language)


_spacy_uk_ner = None


def get_spacy_uk_ner():
    global _spacy_uk_ner
    if _spacy_uk_ner is None or _spacy_uk_ner.gateway is not get_global_gateway():
        if not get_global_gateway().is_available(SupportedLanguage.UKRAINIAN):
            return None
        _spacy_uk_ner = SpacyUkNER()
    return _spacy_uk_ner


def clear_ner_cache():
    global _spacy_uk_ner
    _spacy_uk_ner = None
    get_global_gateway().clear_cache(SupportedLanguage.UKRAINIAN)
