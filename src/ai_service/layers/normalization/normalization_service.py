#!/usr/bin/env python3
"""
Text normalization service - thin facade over NormalizationFactory.

This module provides a clean interface for normalizing person names
from Ukrainian, Russian, and English texts. The heavy lifting is done
by the NormalizationFactory and its processors.
"""

import json
import time
import unicodedata
from dataclasses import fields
from typing import Dict, Optional, Set, List, Tuple

from ...config import LANGUAGE_CONFIG
from ...contracts.base_contracts import NormalizationResult
from ...data.patterns.identifiers import (
    get_validation_function,
    normalize_identifier,
)
from ...utils.logging_config import get_logger
from ...utils.feature_flags import get_feature_flag_manager, FeatureFlags
from ..language.language_detection_service import LanguageDetectionService
from .morphology_adapter import get_global_adapter
from .processors.normalization_factory import NormalizationFactory, NormalizationConfig
from .homoglyph_detector import HomoglyphDetector

# Check for optional dependencies
try:
    from ...data.dicts import russian_names, ukrainian_names
    DICTIONARIES_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy dependency
    DICTIONARIES_AVAILABLE = False
    russian_names = None  # type: ignore
    ukrainian_names = None  # type: ignore


class NormalizationService:
    """
    Thin facade for name normalization.

    Responsibilities:
    - Input validation and sanitization
    - Language detection
    - Configuration management
    - Factory orchestration
    - Statistics aggregation
    """

    def __init__(self):
        """Initialize normalization service with minimal dependencies."""
        self.logger = get_logger(__name__)

        # Feature flag manager
        self.feature_flags = get_feature_flag_manager()

        # Core services
        self.language_service = LanguageDetectionService()
        # Use global adapter for better caching across requests
        self.morphology_adapter = get_global_adapter()
        self.homoglyph_detector = HomoglyphDetector()

        # Load resources for factory
        name_dictionaries = self._load_name_dictionaries()
        diminutive_maps = self._load_diminutive_maps()

        # Initialize the processor factory
        self.normalization_factory = NormalizationFactory(
            name_dictionaries,
            diminutive_maps,
        )

        # Statistics tracking
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'languages_detected': {'ru': 0, 'uk': 0, 'en': 0, 'mixed': 0, 'unknown': 0},
        }

        self.logger.info("NormalizationService initialized as thin facade with feature flags")


    def _apply_language_specific_apostrophe_normalization(
        self,
        result: NormalizationResult,
        language: Optional[str],
    ) -> NormalizationResult:
        """Apply language-specific apostrophe normalization."""
        if not language:
            return result

        lang = language.lower()

        # For English, convert ASCII apostrophe to typographic apostrophe
        if lang == "en":
            if result.normalized:
                result.normalized = result.normalized.replace("'", "\u2019")

            # Also update tokens
            if result.tokens:
                result.tokens = [token.replace("'", "\u2019") for token in result.tokens]

        return result


    async def initialize_runtime(self):
        await self.normalization_factory.ner_gateway.initialize_runtime()

    def runtime_health_check(self):
        return self.normalization_factory.ner_gateway.runtime_health_check()

    async def normalize_async(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        remove_stop_words: bool = True,
        preserve_names: bool = True,
        enable_advanced_features: bool = True,
        enable_morphology: Optional[bool] = None,
        strict_stopwords: Optional[bool] = None,
        # Ukrainian-specific flags
        preserve_feminine_suffix_uk: bool = False,
        enable_spacy_uk_ner: bool = False,
        # English-specific flags
        en_use_nameparser: bool = True,
        enable_en_nickname_expansion: bool = True,
        enable_spacy_en_ner: bool = False,
        # Russian-specific flags
        ru_yo_strategy: str = "preserve",
        enable_ru_nickname_expansion: bool = True,
        enable_spacy_ru_ner: bool = False,
        # Processing feature flags
        feature_flags: Optional[FeatureFlags] = None,
    ) -> NormalizationResult:
        """
        Async normalization entrypoint.

        Args:
            text: Input text to normalize
            language: Language code or 'auto' for detection
            remove_stop_words: Remove stop words during tokenization
            preserve_names: Preserve name-specific punctuation
            enable_advanced_features: Enable morphology and advanced processing
            preserve_feminine_suffix_uk: Preserve Ukrainian feminine suffixes (-ська/-цька)
            enable_spacy_uk_ner: Enable spaCy Ukrainian NER
            en_use_nameparser: Use nameparser for English names
            enable_en_nickname_expansion: Expand English nicknames
            enable_spacy_en_ner: Enable spaCy English NER
            ru_yo_strategy: Russian 'ё' policy ('preserve' or 'fold')
            enable_ru_nickname_expansion: Expand Russian nicknames
            enable_spacy_ru_ner: Enable spaCy Russian NER

        Returns:
            NormalizationResult with normalized text and metadata
        """
        effective_flags = self.feature_flags.get_flags(feature_flags)
        start_time = time.time()
        self._stats['total_requests'] += 1

        # Initialize homoglyph analysis (will be populated in try block)
        homoglyph_analysis = None

        try:
            # Input validation
            validation_result = self._validate_input(text)
            if validation_result:
                self._stats['failed_requests'] += 1
                return validation_result

            # Homoglyph detection and normalization (security preprocessing)
            homoglyph_analysis = self.homoglyph_detector.detect_homoglyphs(text)
            normalized_text = text  # Keep original text for now
            if homoglyph_analysis.get('has_homoglyphs', False):
                # Log security warnings for potential homoglyph attacks
                suspicious_chars = homoglyph_analysis.get('suspicious_chars', [])
                self.logger.warning(f"Homoglyph attack detected: {len(suspicious_chars)} suspicious characters")

            # Use the normalized text for further processing
            text = normalized_text

            # Language detection
            detected_language = self._detect_language(text, language)
            # Safe increment for languages_detected stats
            if detected_language in self._stats['languages_detected']:
                self._stats['languages_detected'][detected_language] += 1
            else:
                # Log unexpected language for monitoring
                self.logger.warning(f"Unexpected language detected: {detected_language}")
                self._stats['languages_detected'].setdefault(detected_language, 0)
                self._stats['languages_detected'][detected_language] += 1

            if strict_stopwords is not None:
                effective_flags.strict_stopwords = strict_stopwords
            if enable_morphology is not None:
                enable_advanced_features = enable_morphology
            
            result = await self._process_with_factory(
                text, detected_language, remove_stop_words,
                preserve_names, enable_advanced_features,
                preserve_feminine_suffix_uk, enable_spacy_uk_ner,
                en_use_nameparser, enable_en_nickname_expansion, enable_spacy_en_ner,
                ru_yo_strategy, enable_ru_nickname_expansion, enable_spacy_ru_ner,
                effective_flags,
            )

            # Add homoglyph analysis to result
            if homoglyph_analysis:
                result.homoglyph_detected = homoglyph_analysis.get('has_homoglyphs', False)
                result.homoglyph_analysis = {
                    'has_homoglyphs': homoglyph_analysis.get('has_homoglyphs', False),
                    'confidence': homoglyph_analysis.get('confidence', 0.0),
                    'suspicious_chars': homoglyph_analysis.get('suspicious_chars', []),
                    'details': homoglyph_analysis.get('details', [])
                }

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=result.success)

            return result

        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=False)
            self._stats['failed_requests'] += 1

            return NormalizationResult(
                normalized="",
                tokens=[],
                trace=[],
                errors=[str(e)],
                language=language or "unknown",
                confidence=0.0,
                original_length=len(text),
                normalized_length=0,
                token_count=0,
                processing_time=processing_time,
                success=False,
                original_text=text,
                token_variants={},
                total_variants=0,
                persons_core=[],
                organizations_core=[],
                persons=[],
            )

    def normalize_sync(self, text: str, *, language=None, **options) -> NormalizationResult:
        """Normalize synchronously using the same options and pipeline as async calls."""
        from ...utils.async_bridge import run_sync

        return run_sync(self.normalize_async(text, language=language, **options))

    def normalize(self, text: str, language=None, remove_stop_words=True,
                  preserve_names=True, enable_advanced_features=True, **options):
        """Compatibility entry point for positional normalization options."""
        return self.normalize_sync(
            text, language=language, remove_stop_words=remove_stop_words,
            preserve_names=preserve_names, enable_advanced_features=enable_advanced_features,
            **options,
        )

    _normalize_sync = normalize_sync

    def _tag_roles(self, tokens, language="ru", quoted_segments=None):
        return self.normalization_factory.role_classifier.tag_tokens(
            tokens, language, quoted_segments
        )[0]

    def _morph_nominal(self, token, language="ru"):
        return self.morphology_adapter.to_nominative(token, language)

    def infer_gender(self, elements, language="ru"):
        return self.normalization_factory.gender_processor.infer_gender_scores(elements, language)

    def adjust_surname_gender(self, lemma, language, gender, gap, original=None):
        processor = self.normalization_factory.gender_processor
        return processor.adjust_surname_with_evidence(lemma, language, gender, gap, original)

    def group_persons(self, tagged_tokens, language="ru"):
        tokens = [token for token, _ in tagged_tokens]
        return self.normalization_factory._extract_persons(
            tagged_tokens, tokens, [role for _, role in tagged_tokens], language
        )

    def validate_identifier(self, identifier: str, identifier_type: str) -> bool:
        """Validate identifier value using checksum-aware validators."""
        validator = get_validation_function(identifier_type)
        if validator is None:
            raise ValueError(f"Unsupported identifier type: {identifier_type}")

        normalized_value = normalize_identifier(identifier, identifier_type)
        if not normalized_value:
            return False

        return validator(normalized_value)

    def _validate_input(self, text: str) -> Optional[NormalizationResult]:
        """Validate input text and return error result if invalid."""
        if not isinstance(text, str):
            return NormalizationResult(
                normalized="",
                tokens=[],
                trace=[],
                errors=["Input must be a string"],
                language="unknown",
                confidence=0.0,
                original_length=0,
                normalized_length=0,
                token_count=0,
                processing_time=0.0,
                success=False,
                original_text=str(text),
                token_variants={},
                total_variants=0,
                persons_core=[],
                organizations_core=[],
                persons=[],
            )

        if len(text) > 10000:
            return NormalizationResult(
                normalized="",
                tokens=[],
                trace=[],
                errors=[f"Input too long: {len(text)} characters (max 10,000)"],
                language="unknown",
                confidence=0.0,
                original_length=len(text),
                normalized_length=0,
                token_count=0,
                processing_time=0.0,
                success=False,
                original_text=text,
                token_variants={},
                total_variants=0,
                persons_core=[],
                organizations_core=[],
                persons=[],
            )

        try:
            unicodedata.normalize("NFC", text)
        except Exception as e:
            return NormalizationResult(
                normalized="",
                tokens=[],
                trace=[],
                errors=[f"Invalid Unicode input: {e}"],
                language="unknown",
                confidence=0.0,
                original_length=len(text),
                normalized_length=0,
                token_count=0,
                processing_time=0.0,
                success=False,
                original_text=text,
                token_variants={},
                total_variants=0,
                persons_core=[],
                organizations_core=[],
                persons=[],
            )

        return None  # Valid input

    def _detect_language(self, text: str, language: Optional[str]) -> str:
        """Detect language or use provided language."""
        if language and language != "auto":
            return language

        try:
            lang_result = self.language_service.detect_language_config_driven(text, LANGUAGE_CONFIG)
            return lang_result.language
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return "ru"  # Default fallback

    def _update_stats(self, processing_time: float, success: bool):
        """Update processing statistics."""
        if success:
            self._stats['successful_requests'] += 1
        else:
            self._stats['failed_requests'] += 1

        # Update average processing time
        total = self._stats['total_requests']
        current_avg = self._stats['avg_processing_time']
        self._stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total

    def get_statistics(self) -> Dict:
        """Get service statistics."""
        return {
            **self._stats,
            'factory_stats': self.normalization_factory.get_statistics()
        }

    def clear_caches(self):
        """Clear all caches."""
        self.normalization_factory.clear_caches()
        self.morphology_adapter.clear_cache()
        self.logger.info("All caches cleared")

    def warmup_morphology_cache(self, samples: List[Tuple[str, str]] = None):
        """
        Warm up morphology cache with common names.
        
        Args:
            samples: List of (token, language) tuples. If None, uses default samples.
        """
        if samples is None:
            # Default common names for warmup
            samples = [
                # Russian names
                ("Анна", "ru"), ("Мария", "ru"), ("Иван", "ru"), ("Сергей", "ru"),
                ("Иванова", "ru"), ("Петрова", "ru"), ("Сидоров", "ru"), ("Кузнецов", "ru"),
                # Ukrainian names
                ("Олена", "uk"), ("Ірина", "uk"), ("Марія", "uk"), ("Іван", "uk"),
                ("Ковальська", "uk"), ("Шевченко", "uk"), ("Петренко", "uk"), ("Кравцівська", "uk"),
            ]
        
        self.morphology_adapter.warmup(samples)
        self.logger.info(f"Morphology cache warmed up with {len(samples)} samples")

    def _load_name_dictionaries(self) -> Dict[str, Set[str]]:
        """Load name dictionaries for processors."""
        dictionaries = {}

        if DICTIONARIES_AVAILABLE:
            try:
                # Load Russian names
                if hasattr(russian_names, 'RUSSIAN_NAMES'):
                    # Extract given names from RUSSIAN_NAMES
                    given_names = set()
                    surnames = set()
                    diminutives = set()
                    
                    for name, props in russian_names.RUSSIAN_NAMES.items():
                        given_names.add(name)
                        if 'variants' in props:
                            given_names.update(props['variants'])
                        if 'diminutives' in props:
                            diminutives.update(props['diminutives'])
                        if 'declensions' in props:
                            given_names.update(props['declensions'])
                    
                    dictionaries['given_names_ru'] = given_names
                    dictionaries['diminutives_ru'] = diminutives
                    dictionaries['surnames_ru'] = surnames

                # Load Ukrainian names
                if hasattr(ukrainian_names, 'UKRAINIAN_NAMES'):
                    # Extract given names from UKRAINIAN_NAMES
                    given_names = set()
                    surnames = set()
                    diminutives = set()
                    
                    for name, props in ukrainian_names.UKRAINIAN_NAMES.items():
                        given_names.add(name)
                        if 'variants' in props:
                            given_names.update(props['variants'])
                        if 'diminutives' in props:
                            diminutives.update(props['diminutives'])
                        if 'declensions' in props:
                            given_names.update(props['declensions'])
                    
                    dictionaries['given_names_uk'] = given_names
                    dictionaries['diminutives_uk'] = diminutives
                    dictionaries['surnames_uk'] = surnames

                self.logger.info(f"Loaded {len(dictionaries)} name dictionaries")
            except Exception as e:
                self.logger.warning(f"Failed to load name dictionaries: {e}")

        return dictionaries

    def _load_diminutive_maps(self) -> Dict[str, Dict[str, str]]:
        """Load diminutive to full name mappings from JSON dictionaries."""
        maps: Dict[str, Dict[str, str]] = {}

        from ...data.resources import PACKAGE_DATA_DIR
        data_dir = PACKAGE_DATA_DIR

        for lang in ("ru", "uk"):
            path = data_dir / f"diminutives_{lang}.json"
            if not path.exists():
                self.logger.warning("Diminutive dictionary missing for %s: %s", lang, path)
                continue

            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                maps[lang] = {
                    unicodedata.normalize("NFC", key).lower(): unicodedata.normalize("NFC", value).lower()
                    for key, value in raw.items()
                }
            except json.JSONDecodeError as exc:
                self.logger.error("Invalid JSON in diminutive dictionary %s: %s", lang, exc)
            except Exception as exc:  # pragma: no cover - IO errors logged
                self.logger.error("Failed to load diminutive dictionary %s: %s", lang, exc)

        # Preserve English nicknames behaviour if available
        if DICTIONARIES_AVAILABLE:
            try:
                from ...data.dicts.english_names import NICKNAMES_EN
                maps['en'] = {
                    unicodedata.normalize("NFC", key).lower(): unicodedata.normalize("NFC", value).lower()
                    for key, value in NICKNAMES_EN.items()
                }
            except ImportError:
                pass
            except Exception as exc:
                self.logger.warning("Failed to load English nicknames: %s", exc)

        self.logger.info(f"Loaded diminutive maps for {len(maps)} languages")
        return maps

    async def _process_with_factory(
        self,
        text: str,
        language: str,
        remove_stop_words: bool,
        preserve_names: bool,
        enable_advanced_features: bool,
        preserve_feminine_suffix_uk: bool = False,
        enable_spacy_uk_ner: bool = False,
        en_use_nameparser: bool = True,
        enable_en_nickname_expansion: bool = True,
        enable_spacy_en_ner: bool = False,
        ru_yo_strategy: str = "preserve",
        enable_ru_nickname_expansion: bool = True,
        enable_spacy_ru_ner: bool = False,
        feature_flags: Optional[FeatureFlags] = None,
    ) -> NormalizationResult:
        """Process with factory implementation."""
        config = NormalizationConfig(
            remove_stop_words=remove_stop_words,
            preserve_names=preserve_names,
            enable_advanced_features=enable_advanced_features,
            enable_morphology=enable_advanced_features,
            language=language,
            preserve_feminine_suffix_uk=preserve_feminine_suffix_uk,
            enable_spacy_uk_ner=enable_spacy_uk_ner,
            en_use_nameparser=en_use_nameparser,
            enable_nameparser_en=en_use_nameparser,
            enable_en_nickname_expansion=enable_en_nickname_expansion,
            enable_en_nicknames=enable_en_nickname_expansion,
            enable_spacy_en_ner=enable_spacy_en_ner,
            ru_yo_strategy=ru_yo_strategy,
            yo_strategy=ru_yo_strategy,  # Use ru_yo_strategy for yo_strategy
            enable_ru_nickname_expansion=enable_ru_nickname_expansion,
            enable_spacy_ru_ner=enable_spacy_ru_ner,
        )
        # Use the same effective configuration for direct and orchestrated calls.
        # Request-level processing options remain explicit arguments above.
        for field in fields(config):
            if hasattr(feature_flags, field.name):
                setattr(config, field.name, getattr(feature_flags, field.name))
        result = await self.normalization_factory.normalize_text(text, config, feature_flags)

        # Apply language-specific apostrophe normalization
        result = self._apply_language_specific_apostrophe_normalization(result, language)

        # Apply language-specific character conversion (e.g., Russian to Ukrainian)
        result = self._apply_language_specific_character_conversion(result, language)

        return result

    def _apply_language_specific_character_conversion(
        self,
        result: NormalizationResult,
        language: Optional[str],
    ) -> NormalizationResult:
        """Apply language-specific character conversion (e.g., Russian to Ukrainian)."""
        if not language or not DICTIONARIES_AVAILABLE:
            return result

        lang = language.lower()

        # For Ukrainian language, convert Russian variants to Ukrainian canonical forms
        if lang == "uk":
            try:
                from ...data.dicts.ukrainian_names import UKRAINIAN_NAMES

                # Build reverse mapping: Russian variant -> Ukrainian canonical
                # Only map variants that contain distinctly Russian characters
                russian_to_ukrainian = {}
                for ukrainian_name, props in UKRAINIAN_NAMES.items():
                    if 'variants' in props:
                        for variant in props['variants']:
                            # Only map if variant contains Russian-specific characters
                            # and is not already a valid Ukrainian name itself
                            has_russian_chars = 'И' in variant or 'Ы' in variant or 'Э' in variant or 'Ё' in variant
                            is_not_ukrainian_canonical = variant not in UKRAINIAN_NAMES
                            if has_russian_chars or is_not_ukrainian_canonical:
                                russian_to_ukrainian[variant] = ukrainian_name

                # Convert normalized text
                if result.normalized:
                    words = result.normalized.split()
                    converted_words = []
                    for word in words:
                        # Try to convert each word
                        converted_word = russian_to_ukrainian.get(word, word)
                        converted_words.append(converted_word)
                    result.normalized = " ".join(converted_words)

                # Convert tokens
                if result.tokens:
                    converted_tokens = []
                    for token in result.tokens:
                        converted_token = russian_to_ukrainian.get(token, token)
                        converted_tokens.append(converted_token)
                    result.tokens = converted_tokens

            except ImportError:
                pass
            except Exception as e:
                # Log but don't fail
                self.logger.warning(f"Failed to apply Russian-to-Ukrainian conversion: {e}")

        return result


    def _build_error_result(self, text: str, error_msg: str) -> NormalizationResult:
        """Build error result for failed processing."""
        return NormalizationResult(
            normalized="",
            tokens=[],
            trace=[],
            errors=[error_msg],
            language="unknown",
            confidence=0.0,
            original_length=len(text),
            normalized_length=0,
            token_count=0,
            processing_time=0.0,
            success=False,
            original_text=text,
            token_variants={},
            total_variants=0,
            persons_core=[],
            organizations_core=[],
            persons=[],
        )
