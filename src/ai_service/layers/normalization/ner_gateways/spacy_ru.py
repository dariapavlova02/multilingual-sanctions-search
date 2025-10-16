"""Russian facade over the shared NER implementation."""
from .language_adapter import LanguageNERAdapter
from .unified_spacy_gateway import NEREntity, NERHints, SupportedLanguage, get_global_gateway


class SpacyRuNER(LanguageNERAdapter):
    language = SupportedLanguage.RUSSIAN

    @property
    def is_available(self):
        return self.gateway.is_available(self.language)


_spacy_ru_ner = None


def get_spacy_ru_ner():
    global _spacy_ru_ner
    if _spacy_ru_ner is None or _spacy_ru_ner.gateway is not get_global_gateway():
        if not get_global_gateway().is_available(SupportedLanguage.RUSSIAN):
            return None
        _spacy_ru_ner = SpacyRuNER()
    return _spacy_ru_ner


def clear_spacy_ru_ner():
    global _spacy_ru_ner
    _spacy_ru_ner = None
    get_global_gateway().clear_cache(SupportedLanguage.RUSSIAN)
