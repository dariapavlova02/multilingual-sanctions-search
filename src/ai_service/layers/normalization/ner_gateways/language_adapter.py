"""Language-specific entry points backed by the shared model gateway."""
from .unified_spacy_gateway import get_global_gateway


class LanguageNERAdapter:
    language = None

    def __init__(self, gateway=None):
        self.gateway = gateway or get_global_gateway()

    def extract_entities(self, text):
        return self.gateway.get_ner_hints(text, self.language)

    def __call__(self, text):
        return self.extract_entities(text)

    def get_entity_at_position(self, text, start, end=None):
        end = start + 1 if end is None else end
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            return None
        return next((e for e in self.extract_entities(text).entities if e.start < end and start < e.end), None)

    def _contains(self, text, start, end, label):
        if not 0 <= start < end <= len(text):
            return False
        return any(e.label == label and e.start <= start and end <= e.end for e in self.extract_entities(text).entities)

    def is_person_entity(self, text, start, end):
        return self._contains(text, start, end, 'PER')

    def is_org_entity(self, text, start, end):
        return self._contains(text, start, end, 'ORG')

    def get_statistics(self):
        info = self.gateway.get_model_info()[self.language.value]
        return {**info, "model_available": info["available"], "spacy_available": info["loaded"]}
