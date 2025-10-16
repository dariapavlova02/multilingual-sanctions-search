"""
Signals Service - структурирование и обогащение сущностей.

Отвечает за:
- Выделение юридических форм и полных названий организаций
- Парсинг идентификаторов (ИНН/ЄДРПОУ/ОГРН/VAT)
- Извлечение дат рождения
- Скоринг сущностей и формирование evidence
- Реконструкцию полных данных персон и организаций

НЕ отвечает за:
- Нормализацию текста (это делает NormalizationService)
- Решение "пропускать/не пропускать" (это делает SmartFilter)
- Генерацию эмбеддингов (это делает EmbeddingService с EmbeddingPreprocessor)
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils.logging_config import get_logger
from .entity_association import evidence_owner
from ...utils.source_text_view import SourceTextView, without_format_controls
from .extractors import (
    BirthdateExtractor,
    IdentifierExtractor,
    OrganizationExtractor,
    PersonExtractor,
)


# Confidence scoring constants
class ConfidenceScoring:
    """Constants for confidence scoring algorithm."""

    BASE_CONFIDENCE = 0.5

    # Person scoring bonuses
    BIRTHDATE_BONUS = 0.15
    VALID_ID_BONUS = 0.2
    INVALID_ID_BONUS = 0.1
    NAME_PATTERN_BONUS = 0.1

    # Organization scoring bonuses
    LEGAL_FORM_BONUS = 0.3
    QUOTED_CORE_BONUS = 0.2
    NORM_MATCH_BONUS = 0.2
    ADJACENT_NAME_BONUS = 0.1
    ORG_CORE_BONUS = 0.1
    CONTEXT_BONUS = 0.15  # Bonus for contextual organization signals
    ADDRESS_BONUS = 0.1   # Bonus for address-like context

    # Multi-evidence bonuses
    PERSON_MAX_MULTI_BONUS = 0.2
    ORG_MAX_MULTI_BONUS = 0.25
    MULTI_EVIDENCE_INCREMENT = 0.05

    MAX_CONFIDENCE = 1.0


@dataclass
class PersonSignal:
    """Сигнал о персоне"""

    core: List[str]  # Токены имени из normalization
    full_name: str
    dob: Optional[str] = None  # ISO YYYY-MM-DD
    dob_raw: Optional[str] = None
    ids: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    source_names: List[str] = field(default_factory=list, repr=False)
    dob_position: Optional[tuple[int, int]] = None


@dataclass
class OrganizationSignal:
    """Сигнал об организации"""

    core: str  # Ядро названия из normalization
    legal_form: Optional[str] = None
    full: Optional[str] = None  # Реконструированное полное название
    ids: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class ExtraSignal:
    """Дополнительные сигналы"""

    type: str
    value: str
    raw: str
    confidence: float = 0.0


class SignalsService:
    """
    Сервис сигналов для структурирования сущностей.

    Принимает исходный текст + результат нормализации,
    возвращает структурированные персоны и организации
    с метаданными (юр. формы, ID, даты рождения).
    """

    def __init__(self):
        self.logger = get_logger(__name__)

        # Initialize specialized extractors
        self.person_extractor = PersonExtractor()
        self.organization_extractor = OrganizationExtractor()
        self.identifier_extractor = IdentifierExtractor()
        self.birthdate_extractor = BirthdateExtractor()


        self.logger.info("SignalsService initialized with extractors")

    def extract(
        self,
        text: str,
        normalization_result: Optional[Dict[str, Any]] = None,
        language: str = "uk",
    ) -> Dict[str, Any]:
        """
        Извлечение и структурирование сигналов из текста.

        Args:
            text: Исходный текст
            normalization_result: Результат работы NormalizationService (опциональный)
            language: Язык текста

        Returns:
            Dict с ключами:
            - persons: List[PersonSignal]
            - organizations: List[OrganizationSignal]
            - extras: Dict с дополнительными сигналы
        """
        if not text or not text.strip():
            return self._empty_result()

        # Debug logging for input validation
        self.logger.debug(f"SIGNALS DEBUG: extract_signals called with text='{text}', language='{language}'")
        self.logger.debug(f"SIGNALS DEBUG: normalization_result type: {type(normalization_result)}")

        # Convert NormalizationResult object to dict if needed
        if normalization_result and hasattr(normalization_result, 'to_dict'):
            self.logger.debug(f"SIGNALS DEBUG: Converting NormalizationResult to dict")
            normalization_result = normalization_result.to_dict()
        elif normalization_result and not isinstance(normalization_result, dict):
            # If it's already a dict, use as is, otherwise try to convert
            try:
                normalization_result = dict(normalization_result)
            except (TypeError, ValueError):
                self.logger.warning(f"Cannot convert normalization_result to dict: {type(normalization_result)}")
                normalization_result = None
        elif normalization_result and not isinstance(normalization_result, dict):
            # If it's not a dict and doesn't have to_dict, convert it
            try:
                normalization_result = dict(normalization_result)
            except (TypeError, ValueError):
                self.logger.warning(f"Cannot convert normalization_result to dict: {type(normalization_result)}")
                normalization_result = None

        try:
            # 1. Извлекаем базовые токены сущностей
            matching_text = without_format_controls(text)
            persons_core, organizations_core = self._get_entity_cores(
                matching_text, normalization_result, language
            )

            # 2. Создаем персоны из токенов
            persons = self._create_person_signals(persons_core)
            self._attach_source_names(persons, normalization_result)

            # 3. Создаем организации из токенов (используем persons_core, чтобы не захватывать ФИО)
            organizations = self._create_organization_signals(matching_text, organizations_core, persons_core)

            # 4. Обогащаем сущности идентификаторами
            unassigned_ids = self._enrich_with_identifiers(text, persons, organizations, normalization_result)

            # 5. Обогащаем персоны датами рождения
            self._enrich_with_birthdates(text, persons)

            # 6. Финальный скоринг сущностей
            self._score_entities(persons, organizations)

            # 7. Формируем результат
            result = self._build_result(persons, organizations)
            result["extras"]["unassigned_ids"] = unassigned_ids

            self.logger.debug(
                f"Extracted {len(persons)} persons, {len(organizations)} organizations"
            )

            return result

        except Exception as e:
            self.logger.exception("Signal extraction failed")
            raise RuntimeError("Signal extraction failed") from e

    def _get_entity_cores(
        self,
        text: str,
        normalization_result: Optional[Dict[str, Any]],
        language: str,
    ) -> tuple[List[List[str]], List[str]]:
        """Извлекает базовые токены персон и организаций."""
        # Извлекаем персоны - ПРИОРИТЕТ нормализованным данным
        if normalization_result and "persons_core" in normalization_result:
            persons_core = normalization_result["persons_core"]
            self.logger.info(f"🟢 SIGNALS FIX: Using normalized persons_core: {persons_core}")
            self.logger.debug(f"🟢 SIGNALS FIX: Using normalized persons_core: {persons_core}")

            # ДИАГНОСТИКА: проверим что в persons_core и отфильтруем неправильные токены
            filtered_persons_core = []
            for person_tokens in persons_core:
                filtered_tokens = []
                for token in person_tokens:
                    # Проверяем не является ли токен payment/stopword
                    if self._is_valid_person_token(token, language):
                        filtered_tokens.append(token)
                    else:
                        self.logger.warning(f"🔴 FILTERING OUT invalid person token: '{token}'")
                        self.logger.debug(f"🔴 FILTERING OUT invalid person token: '{token}'")

                if filtered_tokens:  # Добавляем только если остались валидные токены
                    filtered_persons_core.append(filtered_tokens)

            persons_core = filtered_persons_core
            self.logger.info(f"🟢 AFTER FILTERING: persons_core: {persons_core}")
            self.logger.debug(f"🟢 AFTER FILTERING: persons_core: {persons_core}")

            # FALLBACK: если после фильтрации ничего не осталось, попробуем PersonExtractor
            if not persons_core:
                self.logger.warning(f"[WARN] EMPTY AFTER FILTERING: Trying PersonExtractor as fallback")
                self.logger.debug(f"[WARN] EMPTY AFTER FILTERING: Trying PersonExtractor as fallback")
                fallback_persons = self.person_extractor.extract(text, language)

                # Применяем такую же фильтрацию к fallback результатам
                for person_tokens in fallback_persons:
                    filtered_tokens = []
                    for token in person_tokens:
                        if self._is_valid_person_token(token, language):
                            filtered_tokens.append(token)
                    if filtered_tokens:
                        persons_core.append(filtered_tokens)

                self.logger.info(f"[WARN] FALLBACK RESULT: persons_core: {persons_core}")
                self.logger.debug(f"[WARN] FALLBACK RESULT: persons_core: {persons_core}")
        else:
            # FALLBACK: используем PersonExtractor только если нет нормализованных данных
            self.logger.warning(f"🔴 SIGNALS FALLBACK: No persons_core in normalization_result, falling back to PersonExtractor. normalization_result keys: {list(normalization_result.keys()) if normalization_result else 'None'}")
            self.logger.debug(f"🔴 SIGNALS FALLBACK: No persons_core in normalization_result, falling back to PersonExtractor")
            persons_core = self.person_extractor.extract(text, language)

        # Ensure persons_core is not None
        if persons_core is None:
            persons_core = []

        # Извлекаем организации - ПРИОРИТЕТ нормализованным данным
        if normalization_result and "organizations_core" in normalization_result:
            organizations_core = normalization_result["organizations_core"]
            self.logger.debug(f"Using normalized organizations_core: {organizations_core}")
        else:
            # FALLBACK: используем OrganizationExtractor только если нет нормализованных данных
            self.logger.warning("No organizations_core in normalization_result, falling back to OrganizationExtractor")
            organizations_core = self.organization_extractor.extract(text, language)

        # Ensure organizations_core is not None
        if organizations_core is None:
            organizations_core = []

        return persons_core, organizations_core

    def _is_valid_person_token(self, token: str, language: str) -> bool:
        """Проверяет является ли токен валидным для персоны (не stopword/payment context)."""
        if not token or len(token) < 2:
            return False

        # Initials already retained by name normalization distinguish people with
        # the same surname. Dropping them also loses their source-name association.
        if len(token) == 2 and token[0].isalpha() and token[0].isupper() and token[1] == ".":
            return True

        token_lower = token.lower()

        # Проверяем payment context слова
        payment_words = {
            "сплата", "платеж", "оплата", "платіж", "договор", "договору", "контракт",
            "соглашение", "угода", "абон", "плата", "плати", "услуг", "послуг",
        }
        if token_lower in payment_words:
            return False

        # Проверяем stopwords
        stopwords = {
            "по", "от", "для", "в", "на", "с", "к", "у", "з", "від", "та", "і",
            "и", "а", "но", "или", "либо", "чи", "або", "№", "номер",
        }
        if token_lower in stopwords:
            return False

        # Проверяем даты и номера
        if re.match(r'^\d+[\.\-/]\d+[\.\-/]\d+', token) or re.match(r'^\d{8,}', token):
            return False

        # Дополнительная проверка - это ли имя?
        if self._looks_like_person_name(token, language):
            return True

        # A normalized compound can capitalize each hyphen/apostrophe segment.
        # Requiring the entire tail to be lowercase discards names such as
        # Smith-Jones and O'Connor after normalization has already accepted them.
        parts = re.split(r"[-'’]", token)
        return len(token) >= 3 and all(
            part.isalpha() and part[0].isupper()
            and (len(part) == 1 or part[1:].islower())
            for part in parts
        )

    def _looks_like_person_name(self, token: str, language: str) -> bool:
        """Проверяет похож ли токен на имя человека."""
        if not token:
            return False

        token_lower = token.lower()

        # Известные украинские и русские имена
        common_names = {
            # Женские имена
            "катерина", "екатерина", "анна", "марія", "мария", "ольга", "наталія", "наталья",
            "юлія", "юлия", "ірина", "ирина", "світлана", "светлана", "тетяна", "татьяна",
            "людмила", "валерия", "валентина", "оксана", "лариса", "віра", "вера", "надія", "надежда",
            # Мужские имена
            "олександр", "александр", "дмитро", "дмитрий", "андрій", "андрей", "сергій", "сергей",
            "володимир", "владимир", "іван", "иван", "петро", "петр", "максим", "михайло", "михаил",
            "артем", "роман", "віталій", "виталий", "игорь", "ігор", "юрій", "юрий", "євген", "евгений",
        }

        if token_lower in common_names:
            return True

        # Проверка на типичные окончания имен
        if language in ["uk", "ru"]:
            # Женские окончания (включая -ова для фамилий)
            if token_lower.endswith(("на", "ина", "ова", "ева", "ія", "я", "ка", "ла")):
                return len(token) >= 4
            # Мужские окончания (включая -ов/-ев для фамилий)
            if token_lower.endswith(("ов", "ев", "ій", "ич", "им", "ро", "ко", "ан", "ін")):
                return len(token) >= 4

        return False

    def _create_person_signals(
        self, persons_core: List[List[str]]
    ) -> List[PersonSignal]:
        """Создает PersonSignal объекты из токенов."""
        persons = []
        for person_tokens in persons_core:
            person = PersonSignal(
                core=person_tokens,
                full_name=" ".join(person_tokens),
                confidence=ConfidenceScoring.BASE_CONFIDENCE,
                evidence=["name_pattern"],
            )
            persons.append(person)
        return persons

    @staticmethod
    def _attach_source_names(persons, normalization_result):
        """Keep original name forms for evidence ownership after inflection."""
        if not normalization_result:
            return
        source_names = {}
        for person in normalization_result.get("persons", []) or []:
            if not isinstance(person, dict):
                continue
            normalized = person.get("tokens", [])
            original = person.get("original_tokens", [])
            if (not isinstance(normalized, list) or not isinstance(original, list)
                    or not normalized or not original
                    or not all(isinstance(token, str) for token in [*normalized, *original])):
                continue
            key = tuple(token.casefold() for token in normalized)
            source_names.setdefault(key, set()).add(" ".join(original))
        for person in persons:
            person.source_names = sorted(source_names.get(tuple(token.casefold() for token in person.core), set()))

    def _create_organization_signals(
        self, text: str, organizations_core: List[str], persons_core: List[List[str]]
    ) -> List[OrganizationSignal]:
        """Создает OrganizationSignal объекты из токенов и юридических форм."""
        # Извлекаем юридические формы и создаем организации
        legal_forms_found = self._extract_legal_forms(text, organizations_core, persons_core)

        # Объединяем организации из нормализации с найденными юридическими формами
        org_dict = {}

        # Сначала добавляем базовые организации из нормализации
        for org_core in organizations_core:
            org_dict[org_core.upper()] = OrganizationSignal(
                core=org_core.upper(),
                confidence=ConfidenceScoring.BASE_CONFIDENCE,
                evidence=["org_core"],
            )

        # Обновляем информацией о юридических формах
        for legal_org in legal_forms_found:
            core_key = legal_org["core"]
            if core_key in org_dict:
                # Обновляем существующую организацию
                org_dict[core_key].legal_form = legal_org["legal_form"]
                org_dict[core_key].full = legal_org["full"]
                org_dict[core_key].evidence.extend(legal_org["evidence"])
            else:
                # Добавляем новую организацию
                org_dict[core_key] = OrganizationSignal(
                    core=core_key,
                    legal_form=legal_org["legal_form"],
                    full=legal_org["full"],
                    confidence=ConfidenceScoring.BASE_CONFIDENCE,
                    evidence=legal_org["evidence"],
                )

        # Добавляем контекстные сигналы для улучшения precision
        self._enrich_with_context_signals(text, list(org_dict.values()))

        return list(org_dict.values())

    def _enrich_with_context_signals(self, text: str, organizations: List[OrganizationSignal]):
        """
        Обогащает организации контекстными сигналами для улучшения precision.

        Ищет в тексте слова, которые указывают на деловую/организационную активность:
        - Финансовые термины: банк, кредит, счёт, платеж
        - Организационные: предприятие, компания, фирма
        - Адресные: улица, проспект, город, офис
        - Деятельность: услуги, товары, договор, поставка
        """
        if not organizations or not text:
            return

        # Контекстные паттерны (регионально-нейтральные) с учётом словоформ
        context_patterns = {
            "financial": r"\b(?:банк[а-я]*|bank|кредит[а-я]*|credit|счет[а-я]*|account|платеж[а-я]*|payment|перевод[а-я]*|transfer)\b",
            "business": r"\b(?:предприяти[а-я]*|enterprise|компани[а-я]*|company|фирм[а-я]*|firm|организаци[а-я]*|organization|корпораци[а-я]*|corporation)\b",
            "address": r"\b(?:улиц[а-я]*|ул\.|проспект[а-я]*|просп\.|город[а-я]*|г\.|офис[а-я]*|office|адрес[а-я]*|address)\b",
            "activity": r"\b(?:услуг[а-я]*|services|товар[а-я]*|goods|договор[а-я]*|contract|поставк[а-я]*|supply|реализаци[а-я]*|implementation)\b"
        }

        # Компилируем паттерны
        compiled_patterns = {}
        for category, pattern in context_patterns.items():
            try:
                compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue

        # Проверяем каждую организацию
        for org in organizations:
            context_matches = []

            # Ищем контекстные сигналы в тексте
            for category, pattern in compiled_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    context_matches.extend([(category, match) for match in matches])

            # Добавляем evidence на основе найденных контекстов
            if context_matches:
                # Группируем по категориям
                found_categories = set(match[0] for match in context_matches)

                for category in found_categories:
                    if category == "financial":
                        org.evidence.append("financial_context")
                    elif category == "business":
                        org.evidence.append("business_context")
                    elif category == "address":
                        org.evidence.append("address_context")
                    elif category == "activity":
                        org.evidence.append("activity_context")

                self.logger.debug(
                    f"Organization '{org.core}' enhanced with context signals: {found_categories}"
                )

    def _enrich_with_identifiers(
        self,
        text: str,
        persons: List[PersonSignal],
        organizations: List[OrganizationSignal],
        normalization_result: Optional[Dict[str, Any]] = None,
    ):
        """Обогащает сущности найденными идентификаторами."""
        # 1. Извлекаем ID из trace нормализации (приоритетный источник)
        trace_ids = self._extract_ids_from_normalization_trace(normalization_result, text)

        # 2. Дополняем regex-извлечением из текста (fallback и дополнение)
        org_ids = self.identifier_extractor.extract_organization_ids(text)
        person_ids = self.identifier_extractor.extract_person_ids(text)

        # 3. Объединяем ID из trace с regex-извлеченными
        all_org_ids = trace_ids.get('organization_ids', []) + org_ids
        all_person_ids = trace_ids.get('person_ids', []) + person_ids

        # 4. Удаляем дубликаты по значению
        unique_org_ids = self._deduplicate_ids(all_org_ids, text)
        unique_person_ids = self._deduplicate_ids(all_person_ids, text)

        self.logger.debug(f"[CHECK] ID ENRICHMENT: Found {len(unique_person_ids)} person IDs, {len(unique_org_ids)} org IDs")
        if unique_person_ids:
            self.logger.debug(f"[CHECK] PERSON IDS: {[(p.get('type'), p.get('value'), p.get('source')) for p in unique_person_ids[:3]]}")

        unassigned = []
        all_entities = [*persons, *organizations]
        all_ids = self._deduplicate_ids(unique_person_ids + unique_org_ids, text)
        for id_info in all_ids:
            owner = evidence_owner(all_entities, id_info, text)
            eligible = unique_person_ids if isinstance(owner, PersonSignal) else unique_org_ids
            if owner is not None and id_info in eligible:
                self._assign_id_to_person(owner, id_info)
            else:
                unassigned.append(id_info)
        # Identifier lookup belongs to screening. Never attach a hit to the first entity.
        return unassigned

    def _enrich_with_birthdates(self, text: str, persons: List[PersonSignal]):
        """Обогащает персоны найденными датами рождения."""
        birthdates = self.birthdate_extractor.extract(text)
        self._enrich_persons_with_birthdates(persons, birthdates, text)

    def _build_result(
        self, persons: List[PersonSignal], organizations: List[OrganizationSignal]
    ) -> Dict[str, Any]:
        """Формирует финальный результат."""
        # Calculate overall confidence as average of entity confidences
        all_confidences = []
        if persons:
            all_confidences.extend([p.confidence for p in persons])
        if organizations:
            all_confidences.extend([o.confidence for o in organizations])

        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # Check for sanctioned IDs (FAST PATH markers)
        sanctioned_id_found = False
        sanctioned_count = 0

        for person in persons:
            for id_info in person.ids:
                if id_info.get('sanctioned', False):
                    sanctioned_id_found = True
                    sanctioned_count += 1

        for org in organizations:
            for id_info in org.ids:
                if id_info.get('sanctioned', False):
                    sanctioned_id_found = True
                    sanctioned_count += 1

        result = {
            "persons": [self._person_to_dict(p) for p in persons],
            "organizations": [self._org_to_dict(o) for o in organizations],
            "extras": {"dates": [], "amounts": []},
            "confidence": overall_confidence,
        }

        # Add fast path sanctions info
        if sanctioned_id_found:
            result["fast_path_sanctions"] = {
                "sanctioned_ids_found": sanctioned_count,
                "cache_hit": True
            }
            self.logger.warning(f"🚨 FAST PATH: {sanctioned_count} sanctioned IDs found via cache")

        return result

    def _extract_legal_forms(
        self, text: str, organizations_core: List[str], persons_core: List[List[str]]
    ) -> List[Dict]:
        """
        Детектор юридических форм и реконструкция полных названий.

        Args:
            text: Исходный текст
            organizations_core: Ядра организаций из нормализации

        Returns:
            List[Dict]: Список найденных организаций с legal_form и full
        """
        from ...data.patterns.legal_forms import (
            LEGAL_FORM_REGEX,
            ORGANIZATION_NAME_REGEX,
            QUOTED_CORE_REGEX,
            normalize_legal_form,
        )

        organizations = []
        # Prepare flattened set of person tokens to avoid mixing persons into organization names
        person_tokens_upper = set()
        try:
            for person in persons_core or []:
                for tok in person or []:
                    if isinstance(tok, str) and tok.strip():
                        person_tokens_upper.add(tok.strip().upper())
        except Exception:
            person_tokens_upper = set()

        # Ищем юридические формы в тексте
        for legal_match in LEGAL_FORM_REGEX.finditer(text):
            legal_form_raw = legal_match.group(0)
            legal_form_normalized = normalize_legal_form(legal_form_raw)

            # Позиция юридической формы
            form_start = legal_match.start()
            form_end = legal_match.end()

            # Ищем quoted core рядом с формой (±100 символов)
            search_start = max(0, form_start - 100)
            search_end = min(len(text), form_end + 100)
            search_region = text[search_start:search_end]

            core = None
            full = None
            evidence = ["legal_form_hit"]

            # Сначала пытаемся найти quoted название
            quoted_matches = list(QUOTED_CORE_REGEX.finditer(search_region))

            # Сортируем quoted matches по расстоянию до юридической формы
            def distance_to_legal_form(quoted_match):
                # Расстояние от quoted match до legal form в search_region
                quoted_start_in_search = quoted_match.start()
                quoted_end_in_search = quoted_match.end()

                # Позиция legal form в search_region
                legal_start_in_search = form_start - search_start
                legal_end_in_search = form_end - search_start

                # Расстояние между центрами
                quoted_center = (quoted_start_in_search + quoted_end_in_search) / 2
                legal_center = (legal_start_in_search + legal_end_in_search) / 2
                return abs(quoted_center - legal_center)

            quoted_matches.sort(key=distance_to_legal_form)

            for quoted_match in quoted_matches:
                quoted_core = quoted_match.group(1).strip()

                # Проверяем, есть ли это ядро в результатах нормализации
                if quoted_core.upper() in [org.upper() for org in organizations_core]:
                    core = quoted_core.upper()
                    # Используем нормализованное название для консистентности
                    normalized_core = next(
                        org
                        for org in organizations_core
                        if org.upper() == quoted_core.upper()
                    )
                    full = f'{legal_form_normalized} "{normalized_core}"'
                    evidence.append("quoted_core")
                    evidence.append("norm_match")
                    break
                elif len(quoted_core) >= 3:  # Минимум 3 символа для валидного названия
                    core = quoted_core.upper()
                    full = (
                        f'{legal_form_normalized} "{core}"'  # Используем uppercase core
                    )
                    evidence.append("quoted_core")
                    break

            # Если не нашли в кавычках, ищем рядом с формой
            if not core:
                # 1) Пытаемся найти название СПРАВА от юр. формы (паттерн: `ТОВ <название>`)
                right_region = text[form_end : min(len(text), form_end + 50)].strip()
                if right_region:
                    words = right_region.split()
                    # Пропускаем предлоги/служебные слова в начале
                    skip_heads = {"від", "от", "для", "за", "на", "of", "for", "by", "to"}
                    while words and words[0].lower() in skip_heads:
                        words.pop(0)
                    if words:
                        for i in range(min(5, len(words)), 0, -1):
                            candidate = " ".join(words[:i]).strip()
                            if (
                                len(candidate) >= 3
                                and any(c.isalpha() for c in candidate)
                                and len(candidate) <= 50
                            ):
                                # Skip candidate if it heavily overlaps with person tokens (>=2 tokens)
                                cand_words = [w for w in candidate.replace('"', ' ').split() if w]
                                overlap = sum(1 for w in cand_words if w.upper() in person_tokens_upper)
                                if overlap >= 2:
                                    continue
                                if candidate.upper() in [org.upper() for org in organizations_core]:
                                    core = candidate.upper()
                                    normalized_core = next(
                                        org for org in organizations_core if org.upper() == candidate.upper()
                                    )
                                    full = f"{legal_form_normalized} {normalized_core}"
                                    evidence.extend(["adjacent_name", "norm_match"])
                                    break
                                else:
                                    core = candidate.upper()
                                    full = f"{legal_form_normalized} {candidate}"
                                    evidence.append("adjacent_name")
                                    break

                # 2) Если справа не нашли — пробуем слева (последние 1–3 слова)
                if not core:
                    left_region = text[max(0, form_start - 50) : form_start].strip()
                    if left_region:
                        words = left_region.split()
                        # Пропускаем предлоги/служебные слова в конце левой области
                        skip_tails = {"від", "от", "для", "за", "на", "of", "for", "by", "to"}
                        while words and words[-1].lower() in skip_tails:
                            words.pop()
                        if words:
                            for i in range(min(3, len(words)), 0, -1):
                                candidate = " ".join(words[-i:]).strip()
                                if (
                                    len(candidate) >= 3
                                    and any(c.isalpha() for c in candidate)
                                    and len(candidate) <= 50
                                ):
                                    # Skip candidate if it heavily overlaps with person tokens (>=2 tokens)
                                    cand_words = [w for w in candidate.replace('"', ' ').split() if w]
                                    overlap = sum(1 for w in cand_words if w.upper() in person_tokens_upper)
                                    if overlap >= 2:
                                        continue
                                    if candidate.upper() in [org.upper() for org in organizations_core]:
                                        core = candidate.upper()
                                        normalized_core = next(
                                            org for org in organizations_core if org.upper() == candidate.upper()
                                        )
                                        full = f"{legal_form_normalized} {normalized_core}"
                                        evidence.extend(["adjacent_name", "norm_match"])
                                        break
                                    else:
                                        core = candidate.upper()
                                        full = f"{legal_form_normalized} {candidate}"
                                        evidence.append("adjacent_name")
                                        break

            # Если нашли core, добавляем организацию
            if core:
                org_info = {
                    "core": core,
                    "legal_form": legal_form_normalized,
                    "full": full,
                    "evidence": evidence,
                    "legal_form_raw": legal_form_raw,
                }
                organizations.append(org_info)

        # Удаляем дубликаты по core
        seen_cores = set()
        unique_orgs = []
        for org in organizations:
            if org["core"] not in seen_cores:
                seen_cores.add(org["core"])
                unique_orgs.append(org)

        return unique_orgs

    def _extract_org_ids(self, text: str) -> List[Dict]:
        """Детектор организационных ID"""
        from ...data.patterns.identifiers import (
            get_compiled_patterns_cached,
            get_validation_function,
            normalize_identifier,
        )

        org_id_types = {
            "edrpou",
            "inn_ru",
            "ogrn",
            "ogrnip",
            "kpp",
            "vat_eu",
            "lei",
            "ein",
        }

        found_ids = []

        for pattern, compiled_regex in get_compiled_patterns_cached():
            if pattern.type not in org_id_types:
                continue

            for match in compiled_regex.finditer(text):
                raw_value = match.group(1)
                normalized_value = normalize_identifier(raw_value, pattern.type)

                # Дополнительная валидация если есть
                validator = get_validation_function(pattern.type)
                is_valid = True
                if validator:
                    is_valid = validator(normalized_value)

                confidence = 0.9 if is_valid else 0.6

                id_info = {
                    "type": pattern.type,
                    "value": normalized_value,
                    "raw": match.group(0),  # Полное совпадение
                    "name": pattern.name,
                    "confidence": confidence,
                    "position": match.span(),
                    "valid": is_valid,
                }
                found_ids.append(id_info)

        # Удаляем дубликаты по value
        seen_values = set()
        unique_ids = []
        for id_info in found_ids:
            if id_info["value"] not in seen_values:
                seen_values.add(id_info["value"])
                unique_ids.append(id_info)

        return unique_ids

    def _extract_person_ids(self, text: str) -> List[Dict]:
        """Детектор личных ID"""
        from ...data.patterns.identifiers import (
            get_compiled_patterns_cached,
            get_validation_function,
            normalize_identifier,
        )

        person_id_types = {"inn_ua", "inn_ru", "snils", "ssn", "passport_ua"}

        found_ids = []

        for pattern, compiled_regex in get_compiled_patterns_cached():
            if pattern.type not in person_id_types:
                continue

            for match in compiled_regex.finditer(text):
                raw_value = match.group(1)
                normalized_value = normalize_identifier(raw_value, pattern.type)

                # Дополнительная валидация если есть
                validator = get_validation_function(pattern.type)
                is_valid = True
                if validator:
                    is_valid = validator(normalized_value)

                confidence = 0.9 if is_valid else 0.6

                id_info = {
                    "type": pattern.type,
                    "value": normalized_value,
                    "raw": match.group(0),  # Полное совпадение
                    "name": pattern.name,
                    "confidence": confidence,
                    "position": match.span(),
                    "valid": is_valid,
                }
                found_ids.append(id_info)

        # Удаляем дубликаты по value
        seen_values = set()
        unique_ids = []
        for id_info in found_ids:
            if id_info["value"] not in seen_values:
                seen_values.add(id_info["value"])
                unique_ids.append(id_info)

        return unique_ids

    def _extract_birthdates(self, text: str) -> List[Dict]:
        """Детектор дат рождения"""
        from ...data.patterns.dates import extract_birthdates_from_text

        return extract_birthdates_from_text(text)

    def _enrich_organizations_with_ids(self, organizations, org_ids, text=""):
        for id_info in org_ids:
            owner = evidence_owner(organizations, id_info, text)
            if owner is not None:
                self._assign_id_to_person(owner, id_info)

    def _enrich_persons_with_ids(self, persons, person_ids, text=""):
        self._link_ids_to_persons_by_proximity(persons, person_ids, text)

    def _link_ids_to_persons_by_proximity(self, persons, person_ids, text):
        for id_info in person_ids:
            owner = evidence_owner(persons, id_info, text)
            if owner is not None:
                self._assign_id_to_person(owner, id_info)

    def _assign_id_to_person(self, person: PersonSignal, id_info: Dict):
        """Присваивает ID конкретной персоне"""
        if any(existing["type"] == id_info["type"] and existing["value"] == id_info["value"]
               for existing in person.ids):
            return
        person.ids.append(
            {
                "type": id_info["type"],
                "value": id_info["value"],
                "raw": id_info["raw"],
                "confidence": id_info["confidence"],
                "valid": id_info["valid"],
                "position": id_info.get("position"),
            }
        )

        # Добавляем evidence (скоринг будет в _score_entities)
        if id_info["valid"]:
            person.evidence.append(f"valid_{id_info['type']}")
        else:
            person.evidence.append(f"invalid_{id_info['type']}")

    def _enrich_persons_with_birthdates(self, persons, birthdates, text):
        assignments = {}
        for date_info in birthdates:
            owner = evidence_owner(persons, date_info, text, max_distance=200)
            if owner is not None:
                assignments.setdefault(id(owner), (owner, []))[1].append(date_info)
        for owner, values in assignments.values():
            distinct = {value["iso_format"] for value in values}
            if len(distinct) == 1:
                owner.dob = values[0]["iso_format"]
                owner.dob_raw = values[0]["raw"]
                owner.dob_position = values[0].get("position")
                owner.evidence.append("birthdate_found")
            else:
                owner.dob = owner.dob_raw = owner.dob_position = None
                owner.evidence.append("conflicting_birthdates")

    def _score_entities(
        self, persons: List[PersonSignal], organizations: List[OrganizationSignal]
    ):
        """
        Финальный скоринг сущностей на основе собранных evidence.

        Алгоритм скоринга:
        - Базовая уверенность от нормализации: 0.5
        - Юридическая форма с валидным core: +0.3
        - Валидный ID: +0.2
        - Невалидный ID: +0.1
        - Дата рождения: +0.15
        - Множественные evidence: бонус +0.05 за каждый дополнительный
        """

        # Скоринг персон
        for person in persons:
            base_confidence = ConfidenceScoring.BASE_CONFIDENCE
            bonus = 0.0
            evidence_count = len(person.evidence)

            # Анализируем evidence для бонусов
            for ev in person.evidence:
                if ev == "birthdate_found":
                    bonus += ConfidenceScoring.BIRTHDATE_BONUS
                elif ev.startswith("valid_"):
                    bonus += ConfidenceScoring.VALID_ID_BONUS
                elif ev.startswith("invalid_"):
                    bonus += ConfidenceScoring.INVALID_ID_BONUS
                elif ev == "name_pattern":
                    bonus += ConfidenceScoring.NAME_PATTERN_BONUS

            # Бонус за множественные evidence
            if evidence_count > 1:
                multi_bonus = min(
                    ConfidenceScoring.PERSON_MAX_MULTI_BONUS,
                    (evidence_count - 1) * ConfidenceScoring.MULTI_EVIDENCE_INCREMENT,
                )
                bonus += multi_bonus

            # Обновляем уверенность с учетом максимума
            person.confidence = min(
                ConfidenceScoring.MAX_CONFIDENCE, base_confidence + bonus
            )

        # Скоринг организаций
        for org in organizations:
            base_confidence = ConfidenceScoring.BASE_CONFIDENCE
            bonus = 0.0
            evidence_count = len(org.evidence)

            # Анализируем evidence для бонусов
            for ev in org.evidence:
                if ev == "legal_form_hit":
                    bonus += ConfidenceScoring.LEGAL_FORM_BONUS
                elif ev == "quoted_core":
                    bonus += ConfidenceScoring.QUOTED_CORE_BONUS
                elif ev == "norm_match":
                    bonus += ConfidenceScoring.NORM_MATCH_BONUS
                elif ev == "adjacent_name":
                    bonus += ConfidenceScoring.ADJACENT_NAME_BONUS
                elif ev.startswith("valid_"):
                    bonus += ConfidenceScoring.VALID_ID_BONUS
                elif ev.startswith("invalid_"):
                    bonus += ConfidenceScoring.INVALID_ID_BONUS
                elif ev == "org_core":
                    bonus += ConfidenceScoring.ORG_CORE_BONUS
                # Новые контекстные бонусы
                elif ev in ["financial_context", "business_context", "activity_context"]:
                    bonus += ConfidenceScoring.CONTEXT_BONUS
                elif ev == "address_context":
                    bonus += ConfidenceScoring.ADDRESS_BONUS

            # Бонус за множественные evidence
            if evidence_count > 1:
                multi_bonus = min(
                    ConfidenceScoring.ORG_MAX_MULTI_BONUS,
                    (evidence_count - 1) * ConfidenceScoring.MULTI_EVIDENCE_INCREMENT,
                )
                bonus += multi_bonus

            # Обновляем уверенность с учетом максимума
            org.confidence = min(
                ConfidenceScoring.MAX_CONFIDENCE, base_confidence + bonus
            )

    def _person_to_dict(self, person: PersonSignal) -> Dict[str, Any]:
        """Конвертация PersonSignal в словарь"""
        return {
            "core": person.core,
            "full_name": person.full_name,
            "dob": person.dob,
            "dob_raw": person.dob_raw,
            "dob_position": person.dob_position,
            "ids": person.ids,
            "confidence": person.confidence,
            "evidence": person.evidence,
        }

    def _org_to_dict(self, org: OrganizationSignal) -> Dict[str, Any]:
        """Конвертация OrganizationSignal в словарь"""
        return {
            "core": org.core,
            "legal_form": org.legal_form,
            "full": org.full,
            "ids": org.ids,
            "confidence": org.confidence,
            "evidence": org.evidence,
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Пустой результат при ошибке или пустом вводе"""
        return {
            "persons": [],
            "organizations": [],
            "extras": {"dates": [], "amounts": []},
            "confidence": 0.0,
        }

    # Backward compatibility methods for tests
    def _extract_persons(self, text: str, normalization_result: Optional[Dict] = None) -> List[Dict]:
        """Backward compatibility method for tests"""
        persons_core, _ = self._get_entity_cores(text, normalization_result, "auto")
        person_signals = self._create_person_signals(persons_core)
        return [
            {
                "core": signal.core,
                "full_name": signal.full_name,
                "confidence": signal.confidence,
                "evidence": signal.evidence,
                "dob": signal.dob,
                "ids": signal.ids
            }
            for signal in person_signals
        ]

    def _extract_organizations(self, text: str, normalization_result: Optional[Dict] = None) -> List[Dict]:
        """Backward compatibility method for tests"""
        persons_core, organizations_core = self._get_entity_cores(text, normalization_result, "auto")
        org_signals = self._create_organization_signals(text, organizations_core, persons_core)
        return [
            {
                "core": signal.core,
                "legal_form": signal.legal_form,
                "full": signal.full,
                "confidence": signal.confidence,
                "evidence": signal.evidence,
                "ids": signal.ids
            }
            for signal in org_signals
        ]

    def _extract_persons_from_normalization(self, normalization_result: Dict) -> List[Dict]:
        """Backward compatibility method for tests"""
        return self._extract_persons("", normalization_result)

    def _extract_organizations_from_normalization(self, normalization_result: Dict) -> List[Dict]:
        """Backward compatibility method for tests"""
        return self._extract_organizations("", normalization_result)

    def _extract_organization_ids(self, text: str) -> List[Dict]:
        """Backward compatibility method for tests"""
        return self._extract_org_ids(text)

    def _extract_extras(self, text: str) -> Dict[str, List]:
        """Backward compatibility method for tests"""
        birthdates = self._extract_birthdates(text)
        return {
            "dates": birthdates,
            "amounts": []  # Amounts not implemented yet
        }

    async def extract_signals(self, text: str, normalization_result: Optional[Dict] = None,
                            language: str = "auto"):
        """Backward compatibility method for tests - returns result with object attributes"""
        result_dict = await self.extract_async(text, normalization_result, language)

        # Create a simple object wrapper for backward compatibility
        class ResultWrapper:
            def __init__(self, result_dict):
                # Convert person dicts to simple objects with attributes
                self.persons = []
                for person_dict in result_dict.get("persons", []):
                    person_obj = type('PersonObj', (), {})()
                    for key, value in person_dict.items():
                        setattr(person_obj, key, value)
                    self.persons.append(person_obj)

                # Convert organization dicts to simple objects with attributes
                self.organizations = []
                for org_dict in result_dict.get("organizations", []):
                    org_obj = type('OrgObj', (), {})()
                    for key, value in org_dict.items():
                        # Map 'full' to 'full_name' for backward compatibility
                        if key == "full":
                            setattr(org_obj, "full_name", value)
                            setattr(org_obj, key, value)  # Keep original too
                        else:
                            setattr(org_obj, key, value)

                    # Ensure 'full' attribute always exists for backward compatibility
                    if not hasattr(org_obj, 'full'):
                        setattr(org_obj, 'full', None)
                    if not hasattr(org_obj, 'full_name'):
                        setattr(org_obj, 'full_name', None)

                    self.organizations.append(org_obj)

                # Copy other attributes directly
                for key, value in result_dict.items():
                    if key not in ["persons", "organizations"]:
                        setattr(self, key, value)

                # Add backward compatibility attributes for test expectations
                # Extract numbers (IDs) from persons and organizations
                self.numbers = {}
                all_ids = []

                # Collect IDs from persons
                for person in self.persons:
                    if hasattr(person, 'ids') and person.ids:
                        all_ids.extend(person.ids)

                # Collect IDs from organizations
                for org in self.organizations:
                    if hasattr(org, 'ids') and org.ids:
                        all_ids.extend(org.ids)

                all_ids.extend(result_dict.get("extras", {}).get("unassigned_ids", []))

                # Organize IDs by type
                for id_item in all_ids:
                    if isinstance(id_item, dict):
                        id_type = id_item.get('type', 'unknown')
                        id_value = id_item.get('value', id_item.get('raw', ''))
                        if id_type not in self.numbers:
                            self.numbers[id_type] = []
                        self.numbers[id_type].append(id_value)

                # Extract dates from persons
                self.dates = {}
                birth_dates = []

                for person in self.persons:
                    # Check for both dob and birth_date attributes for compatibility
                    dob = None
                    if hasattr(person, 'dob') and person.dob:
                        dob = person.dob
                    elif hasattr(person, 'birth_date') and person.birth_date:
                        dob = person.birth_date

                    if dob:
                        birth_dates.append(str(dob))

                # Also check extras for dates
                extras_dates = result_dict.get("extras", {}).get("dates", [])
                birth_dates.extend(extras_dates)

                if birth_dates:
                    self.dates['birth_dates'] = birth_dates

        return ResultWrapper(result_dict)

    def _extract_person_tokens(self, text: str, language: str) -> List[List[str]]:
        """
        Извлечение токенов персон из текста.

        Использует простую эвристику для поиска имен:
        - Слова с заглавной буквы, не являющиеся юридическими формами
        - Последовательности из 2-3 капитализированных слов
        """
        # Простая эвристика: ищем последовательности капитализированных слов
        import re

        # Паттерн для поиска имен (2-3 капитализированных слова подряд)
        # Поддерживаем украинский и русский алфавиты (только заглавные в начале)
        name_pattern = r"\b[А-ЯЁЇІЄҐ][а-яёїієґ]+(?:\s+[А-ЯЁЇІЄҐ][а-яёїієґ]+){1,2}\b"

        # Также ищем латинские имена
        latin_name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"

        found_names = []

        # Украинские/русские имена
        for match in re.finditer(name_pattern, text):
            name_tokens = match.group(0).split()
            # Фильтруем юридические формы
            if not self._contains_legal_form(name_tokens):
                found_names.append(name_tokens)

        # Латинские имена
        for match in re.finditer(latin_name_pattern, text):
            name_tokens = match.group(0).split()
            if not self._contains_legal_form(name_tokens):
                found_names.append(name_tokens)

        return found_names

    def _extract_organization_tokens(self, text: str, language: str) -> List[str]:
        """
        Извлечение токенов организаций из текста.

        Использует детекцию юридических форм для поиска организаций.
        """
        from ...data.patterns.legal_forms import get_legal_forms_regex

        # Ищем юридические формы в тексте
        legal_forms_regex = get_legal_forms_regex()
        organizations = []

        for match in re.finditer(legal_forms_regex, text, re.IGNORECASE):
            # Найдена юридическая форма, теперь ищем название организации
            org_text = self._extract_org_name_near_legal_form(text, match)
            if org_text:
                organizations.append(org_text.strip().upper())

        return list(set(organizations))  # Убираем дубликаты

    def _contains_legal_form(self, tokens: List[str]) -> bool:
        """Проверяет, содержат ли токены юридическую форму"""
        from ...data.patterns.legal_forms import get_legal_forms_set

        legal_forms = get_legal_forms_set()
        for token in tokens:
            if token.upper() in legal_forms:
                return True
        return False

    def _extract_org_name_near_legal_form(self, text: str, legal_form_match) -> str:
        """
        Извлекает название организации рядом с найденной юридической формой.

        Это упрощенная версия логики из _extract_legal_forms.
        """
        legal_form = legal_form_match.group(0)
        start = legal_form_match.start()
        end = legal_form_match.end()

        # Ищем кавычки рядом с юридической формой
        quoted_pattern = r'["\u201c\u201d\u00ab\u00bb]([^"\u201c\u201d\u00ab\u00bb]+)["\u201c\u201d\u00ab\u00bb]'

        # Ищем в окрестности юридической формы (±100 символов)
        search_start = max(0, start - 100)
        search_end = min(len(text), end + 100)
        search_area = text[search_start:search_end]

        quoted_matches = list(re.finditer(quoted_pattern, search_area))

        if quoted_matches:
            # Берем ближайшее кавычки к юридической форме
            legal_form_pos_in_area = start - search_start
            closest_match = min(
                quoted_matches, key=lambda m: abs(m.start() - legal_form_pos_in_area)
            )
            return closest_match.group(1)

        # Если кавычек нет, возвращаем заглушку
        return "UNNAMED_ORG"

    # ==================== ASYNC METHODS ====================

    async def extract_async(
        self,
        text: str,
        normalization_result: Optional[Dict[str, Any]] = None,
        language: str = "uk",
    ) -> Dict[str, Any]:
        """
        Async version of extract using thread pool executor
        
        Args:
            text: Исходный текст
            normalization_result: Результат работы NormalizationService (опциональный)
            language: Язык текста

        Returns:
            Dict с ключами:
            - persons: List[PersonSignal]
            - organizations: List[OrganizationSignal]
            - extras: Dict с дополнительными сигналы
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Use default thread pool executor
            self.extract,
            text, normalization_result, language
        )
    
    def _extract_ids_from_normalization_trace(
        self, normalization_result: Optional[Dict[str, Any]], text: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Извлекает ID из trace нормализации.

        Args:
            normalization_result: Результат нормализации с trace
            text: Исходный текст для определения позиций

        Returns:
            Dict с ключами 'person_ids' и 'organization_ids'
        """
        # Import INN validation for proper Russian/Ukrainian validation
        from ...data.patterns.identifiers import validate_inn
        
        if not normalization_result or 'trace' not in normalization_result:
            self.logger.debug("[CHECK] ID TRACE: No trace in normalization_result")
            return {'person_ids': [], 'organization_ids': []}

        trace = normalization_result['trace']
        person_ids = []
        organization_ids = []

        self.logger.debug(f"[CHECK] ID TRACE: Processing {len(trace)} trace entries")

        for entry in trace:
            # Ищем токены с ролью 'id'
            if entry.get('role') == 'id' or entry.get('type') == 'role' and entry.get('role') == 'id':
                token_text = entry.get('token', '')

                if token_text and token_text.isdigit():
                    # Найдем позицию токена в исходном тексте
                    import re
                    matches = list(re.finditer(r"(?<!\d)" + re.escape(token_text) + r"(?!\d)", text))
                    if len(matches) != 1:
                        continue  # Regex extraction retains real per-occurrence spans.
                    position = matches[0].span()

                    # Определяем тип ID на основе длины и контекста
                    id_length = len(token_text)

                    # Универсальный ID для персон и организаций
                    id_info = {
                        "type": "numeric_id",  # Общий тип для всех numeric ID из trace
                        "value": token_text,
                        "raw": token_text,
                        "name": f"Numeric ID ({id_length} digits)",
                        "confidence": 0.95,  # Высокая уверенность - найдено нормализацией
                        "position": position,
                        "valid": False,  # A numeric token has not passed identifier validation.
                        "source": "normalization_trace"  # Отметка что из trace
                    }

                    # Добавляем и в person_ids и в organization_ids
                    # так как из trace неясно к чему относится ID
                    person_ids.append(id_info.copy())
                    organization_ids.append(id_info.copy())

                    self.logger.debug(f"[CHECK] ID TRACE: Found numeric ID '{token_text}' in trace")
            
            # ИЩЕМ ИНН В NOTES - это фикс для проблемы когда ИНН отсекается нормализацией
            notes = entry.get('notes', '')
            if 'marker_инн_nearby' in notes or 'marker_inn_nearby' in notes:
                # Извлекаем ИНН из текста по контексту
                token_text = entry.get('token', '')
                if token_text and token_text.isdigit() and len(token_text) >= 10:
                    # Ищем ИНН в исходном тексте рядом с этим токеном
                    import re
                    inn_pattern = r'(?:(?:ИНН|инн|INN)\s*[\:\:]?\s*)?(\d{10,12})'
                    inn_matches = list(re.finditer(inn_pattern, text))

                    inn_found = False
                    for match in inn_matches:
                        inn_value = match.group(1)
                        if inn_value == token_text:
                            position = match.span()  # Span and raw both include the marker.

                            # Создаем ID для ИНН с правильной валидацией
                            is_valid = validate_inn(inn_value)
                            inn_id_info = {
                                "type": "inn",
                                "value": inn_value,
                                "raw": match.group(0),  # Весь матч включая "ИНН"
                                "name": "Taxpayer ID (INN)",
                                "confidence": 0.9 if is_valid else 0.6,
                                "position": position,
                                "valid": is_valid,  # Используем правильную валидацию RU + UA
                                "source": "normalization_trace_inn"
                            }

                            person_ids.append(inn_id_info.copy())
                            self.logger.warning(f"[CHECK] ID TRACE: Found INN '{inn_value}' from marker_инн_nearby in trace (valid={is_valid})")
                            inn_found = True
                            break

                    # Если не нашли ИНН pattern, НЕ добавляем как numeric_id
                    # (это предотвращает дубликаты и неправильную типизацию)
                    if not inn_found:
                        self.logger.debug(f"[CHECK] ID TRACE: Token '{token_text}' with marker_инн_nearby but no INN pattern match")

        self.logger.debug(f"[CHECK] ID TRACE: Extracted {len(person_ids)} person IDs, {len(organization_ids)} org IDs from trace")
        return {'person_ids': person_ids, 'organization_ids': organization_ids}

    def _deduplicate_ids(self, ids: List[Dict[str, Any]], text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Удаляет дубликаты ID по значению, сохраняя приоритет trace > regex.

        Args:
            ids: Список ID для дедупликации

        Returns:
            Список уникальных ID
        """
        if not ids:
            return []

        # Группируем по значению
        id_groups = {}
        for id_info in ids:
            span = tuple(id_info.get('position') or ())
            value_text = str(id_info.get('value', ''))
            if text and value_text and len(span) == 2 and 0 <= span[0] < span[1] <= len(text):
                # Regex evidence may include its label ("ИНН 123...") while
                # normalization records only the digits. Both describe one
                # source occurrence, not two independent identifiers.
                view = SourceTextView.from_text(text[span[0]:span[1]])
                matches = list(re.finditer(r"(?<!\w)" + re.escape(value_text) + r"(?!\w)", view.text))
                if len(matches) == 1:
                    local_start, local_end = view.original_span(*matches[0].span())
                    span = (span[0] + local_start, span[0] + local_end)
            value = (value_text, span)
            if value not in id_groups:
                id_groups[value] = []
            id_groups[value].append(id_info)

        # Выбираем лучший ID из каждой группы (INN extractor > trace > other regex)
        unique_ids = []
        for value, group in id_groups.items():
            # Сортируем: INN типы первыми, потом trace, потом по confidence
            group.sort(key=lambda x: (
                x.get('type') not in ['inn', 'inn_ua', 'inn_ru'],  # INN типы первые (False < True)
                x.get('source') != 'normalization_trace',  # trace вторые (False < True)  
                -x.get('confidence', 0)  # потом по убывающей confidence
            ))

            # Берем лучший
            best_id = group[0]
            unique_ids.append(best_id)

            self.logger.debug(f"[CHECK] ID DEDUP: Selected {best_id.get('source', 'regex')} source for ID '{value}'")

        return unique_ids

    def _is_mixed_language_text(self, text: str) -> bool:
        """
        Check if text contains mixed language (both Cyrillic and Latin scripts)
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text contains both Cyrillic and Latin characters
        """
        if not text:
            return False
            
        # Count Cyrillic characters
        cyrillic_count = len(re.findall(r"[а-яёіїєґ]", text, re.IGNORECASE))
        
        # Count Latin characters
        latin_count = len(re.findall(r"[a-z]", text, re.IGNORECASE))
        
        # Consider mixed if both scripts are present
        return cyrillic_count > 0 and latin_count > 0
