"""English facade over the shared NER implementation."""
from .language_adapter import LanguageNERAdapter
from .unified_spacy_gateway import NEREntity, NERHints, SupportedLanguage, get_global_gateway


class SpacyEnNER(LanguageNERAdapter):
    language = SupportedLanguage.ENGLISH

    def __init__(self, gateway=None):
        super().__init__(gateway)
        if not self.gateway.is_available(self.language):
            raise RuntimeError('spaCy English NER model is not available')

    def is_available(self):
        return self.gateway.is_available(self.language)


_spacy_en_ner = None


def get_spacy_en_ner():
    global _spacy_en_ner
    if _spacy_en_ner is None or _spacy_en_ner.gateway is not get_global_gateway():
        if not get_global_gateway().is_available(SupportedLanguage.ENGLISH):
            return None
        _spacy_en_ner = SpacyEnNER()
    return _spacy_en_ner


def clear_spacy_en_ner():
    global _spacy_en_ner
    _spacy_en_ner = None
    get_global_gateway().clear_cache(SupportedLanguage.ENGLISH)
