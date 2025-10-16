"""Shared spaCy models, character-span contracts and bounded NER execution."""
from dataclasses import dataclass, field
from enum import Enum
import threading
import unicodedata
from typing import Any

from ....utils.inference_queue import InferenceQueue
from ....utils.logging_config import get_logger


class SupportedLanguage(str, Enum):
    RUSSIAN = 'ru'
    UKRAINIAN = 'uk'
    ENGLISH = 'en'


@dataclass
class NEREntity:
    text: str
    label: str
    start: int
    end: int
    # spaCy does not expose a calibrated probability for each entity.
    confidence: float | None = None


@dataclass
class NERHints:
    """Spans refer to the exact text passed to the gateway, including duplicates."""
    person_spans: list[tuple[int, int]] = field(default_factory=list)
    org_spans: list[tuple[int, int]] = field(default_factory=list)
    entities: list[NEREntity] = field(default_factory=list)

    @property
    def persons(self) -> set[str]:
        return {e.text.lower() for e in self.entities if e.label in {'PER', 'PERSON'}}

    @property
    def organizations(self) -> set[str]:
        return {e.text.lower() for e in self.entities if e.label == 'ORG'}

    @property
    def locations(self) -> set[str]:
        return {e.text.lower() for e in self.entities if e.label == 'LOC'}

    @property
    def confidence(self) -> None:
        """Entity counts are not recognition probabilities."""
        return None


class NERUnavailableError(RuntimeError):
    """Requested NER could not complete; empty hints would hide that failure."""


class UnifiedSpacyGateway:
    # The pinned small pipelines have an independent NER tok2vec submodel.
    # Exclude other components instead of loading and merely disabling them.
    NON_NER_COMPONENTS = ['tok2vec', 'tagger', 'morphologizer', 'parser',
                          'senter', 'attribute_ruler', 'lemmatizer']
    MODEL_CONFIG = {
        SupportedLanguage.RUSSIAN: {'model_name': 'ru_core_news_sm'},
        SupportedLanguage.UKRAINIAN: {'model_name': 'uk_core_news_sm'},
        SupportedLanguage.ENGLISH: {'model_name': 'en_core_web_sm'},
    }

    def __init__(self, *, max_pending: int = 8, timeout: float = 10.0):
        self.logger = get_logger(__name__)
        self._models: dict[SupportedLanguage, Any] = {}
        self._availability: dict[SupportedLanguage, bool] = {}
        self._validated_models: dict[SupportedLanguage, Any] = {}
        self._model_lock = threading.RLock()
        self._queue = InferenceQueue(max_pending, timeout, label='NER')

    def _load_spacy_model(self, language: SupportedLanguage):
        language = SupportedLanguage(language)
        with self._model_lock:
            if language in self._availability:
                return self._models.get(language), self._availability[language]
            try:
                import spacy
                model = spacy.load(self.MODEL_CONFIG[language]['model_name'],
                                   exclude=self.NON_NER_COMPONENTS)
            except (ImportError, OSError):
                model = None
                self.logger.warning('Required NER model unavailable for %s; run make download-models', language.value)
            self._models[language] = model
            self._availability[language] = model is not None
            return model, model is not None

    def is_available(self, language: SupportedLanguage) -> bool:
        return self._load_spacy_model(language)[1]

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = label.upper()
        return {'PERSON': 'PER', 'ORGANIZATION': 'ORG', 'LOCATION': 'LOC', 'GPE': 'LOC'}.get(label, label)

    @staticmethod
    def _languages_for_text(text, preferred):
        names = {unicodedata.name(c, "") for c in text if c.isalpha()}
        selected = set()
        if any("LATIN" in name for name in names):
            selected.add(SupportedLanguage.ENGLISH)
        if any("CYRILLIC" in name for name in names):
            if any(c.lower() in "іїєґ" for c in text):
                selected.add(SupportedLanguage.UKRAINIAN)
            elif preferred in {"ru", "uk"}:
                selected.add(SupportedLanguage(preferred))
            else:
                selected.update((SupportedLanguage.RUSSIAN, SupportedLanguage.UKRAINIAN))
        return selected

    def _extract_entities(self, text: str, language, required: bool):
        if not text or not text.strip():
            return []
        try:
            languages = self._languages_for_text(text, language)
            if not languages and any(c.isalpha() for c in text):
                raise NERUnavailableError("NER does not support this script")
            entities, seen = [], set()
            for selected in sorted(languages, key=lambda item:item.value):
                model, available = self._load_spacy_model(selected)
                if not available:
                    raise NERUnavailableError("Requested NER model is unavailable")
                try:
                    model_entities = list(model(text).ents)
                    if any(not (0 <= e.start_char < e.end_char <= len(text))
                           or text[e.start_char:e.end_char] != e.text for e in model_entities):
                        raise NERUnavailableError("NER returned inconsistent source positions")
                except Exception:
                    self._validated_models.pop(selected, None)
                    raise
                self._validated_models[selected] = model
                for entity in model_entities:
                    start, end = entity.start_char, entity.end_char
                    if not (0 <= start < end <= len(text)) or text[start:end] != entity.text:
                        raise NERUnavailableError("NER returned inconsistent source positions")
                    # A Cyrillic model must not classify a Latin name (or vice
                    # versa) merely because both occur in one payment string.
                    if selected not in self._languages_for_text(entity.text, language):
                        continue
                    key = (self._normalize_label(entity.label_), start, end)
                    if key not in seen:
                        seen.add(key)
                        entities.append(NEREntity(entity.text, key[0], start, end))
            return sorted(entities, key=lambda entity:(entity.start, entity.end, entity.label))
        except Exception as exc:
            self.logger.warning("NER extraction failed for %s", language)
            if required:
                raise NERUnavailableError("Requested NER extraction failed") from exc
            return []

    @staticmethod
    def _hints(entities: list[NEREntity]) -> NERHints:
        return NERHints(
            person_spans=[(e.start, e.end) for e in entities if e.label == 'PER'],
            org_spans=[(e.start, e.end) for e in entities if e.label == 'ORG'],
            entities=entities,
        )

    def extract_entities(self, text: str, language: SupportedLanguage, *, required=False) -> list[NEREntity]:
        return self._queue.run(self._extract_entities, text, language, required)

    def get_ner_hints(self, text: str, language: SupportedLanguage, *, required=False) -> NERHints:
        return self._hints(self.extract_entities(text, language, required=required))

    async def get_ner_hints_async(self, text: str, language: SupportedLanguage) -> NERHints:
        entities = await self._queue.run_async(self._extract_entities, text, language, True)
        return self._hints(entities)

    async def initialize_runtime(self):
        """Exercise each supported pipeline before the API accepts traffic."""
        probes = {"en": "Model readiness verification.",
                  "ru": "Проверка готовности модели.",
                  "uk": "Перевірка готовності моделі."}
        for language, text in probes.items():
            await self._queue.run_async(self._verify_language, text, SupportedLanguage(language))
        if self.runtime_health_check()["status"] != "healthy":
            raise NERUnavailableError("Required NER pipelines are not ready")

    def _verify_language(self, text, language):
        model, available = self._load_spacy_model(language)
        if not available or "ner" not in model.pipe_names:
            self._validated_models.pop(language, None)
            raise NERUnavailableError("Required NER component is unavailable")
        self._extract_entities(text, language, True)

    def runtime_health_check(self):
        """Read completed inference state without waiting for a loading model."""
        models = {}
        for language in SupportedLanguage:
            model = self._models.get(language)
            models[language.value] = {
                "loaded": model is not None,
                "validated": model is not None and self._validated_models.get(language) is model,
            }
        queue = self._queue.health_check()
        ready = queue["status"] == "healthy" and all(m["validated"] for m in models.values())
        return {"status": "healthy" if ready else "unhealthy", "models": models, "queue": queue}

    def get_supported_languages(self):
        return list(SupportedLanguage)

    def get_model_info(self):
        """Diagnostics must not load models as a side effect."""
        with self._model_lock:
            return {lang.value: {'model_name': data['model_name'],
                    'available': self._availability.get(lang),
                    'loaded': self._models.get(lang) is not None}
                    for lang, data in self.MODEL_CONFIG.items()}

    def clear_cache(self, language=None):
        """Clear models in the same serial queue as inference."""
        return self._queue.run(self._clear_cached_models, language)

    def _clear_cached_models(self, language):
        with self._model_lock:
            languages = [SupportedLanguage(language)] if language is not None else list(SupportedLanguage)
            for lang in languages:
                self._models.pop(lang, None)
                self._availability.pop(lang, None)
                self._validated_models.pop(lang, None)

    def close(self):
        self._queue.close()


_global_gateway = None
_global_lock = threading.Lock()


def get_global_gateway():
    global _global_gateway
    with _global_lock:
        if _global_gateway is None:
            _global_gateway = UnifiedSpacyGateway()
        return _global_gateway


def close_global_gateway():
    global _global_gateway
    with _global_lock:
        if _global_gateway is not None:
            _global_gateway.close()
            _global_gateway = None


def create_russian_gateway():
    return UnifiedSpacyGateway()


def create_ukrainian_gateway():
    return UnifiedSpacyGateway()


def create_english_gateway():
    return UnifiedSpacyGateway()
