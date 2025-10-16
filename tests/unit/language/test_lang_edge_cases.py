"""
Unit tests for language detection edge cases.

Tests robust handling of short strings, uppercase text, noise,
long words, and numeric/punctuation dominance.
"""

import pytest
from ai_service.layers.language.language_detection_service import LanguageDetectionService
from ai_service.config import LANGUAGE_CONFIG
from ai_service.utils.types import LanguageDetectionResult


class TestLanguageDetectionEdgeCases:
    """Test edge cases for language detection"""

    @pytest.fixture
    def language_service(self):
        """Create language detection service instance"""
        return LanguageDetectionService()

    def test_short_text_edge_cases(self, language_service):
        """Test handling of very short text"""
        # Very short text should be unknown with low confidence
        result = language_service.detect_language_config_driven("ОО", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence <= 0.5
        assert result.details["reason"] == "insufficient_alphabetic_chars"
        
        # Single character
        result = language_service.detect_language_config_driven("А", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence <= 0.5
        
        # Two characters
        result = language_service.detect_language_config_driven("LL", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence <= 0.5

    def test_acronym_detection_with_confidence_penalty(self, language_service):
        """Test detection of acronyms with confidence penalty"""
        # Russian acronym
        result = language_service.detect_language_config_driven("ООО", LANGUAGE_CONFIG)
        assert result.language in ["ru", "unknown"]
        assert result.confidence <= 0.6
        assert result.details.get("is_likely_acronym", False)
        assert result.details.get("uppercase_penalty", 0) == 0.4
        
        # English acronym
        result = language_service.detect_language_config_driven("LLC", LANGUAGE_CONFIG)
        assert result.language in ["en", "unknown"]
        assert result.confidence <= 0.6
        assert result.details.get("is_likely_acronym", False)
        
        # Ukrainian acronym
        result = language_service.detect_language_config_driven("ТОВ", LANGUAGE_CONFIG)
        assert result.language in ["uk", "unknown"]
        assert result.confidence <= 0.6
        assert result.details.get("is_likely_acronym", False)

    def test_uppercase_text_confidence_penalty(self, language_service):
        """Test uppercase text gets confidence penalty"""
        # All uppercase Russian text
        result = language_service.detect_language_config_driven("ПЛАТЕЖ", LANGUAGE_CONFIG)
        assert result.language in ["ru", "unknown"]
        assert result.confidence <= 0.7  # Should be lower due to uppercase penalty
        assert result.details.get("is_likely_acronym", False)  # Not an acronym (too long)
        
        # All uppercase English text
        result = language_service.detect_language_config_driven("MEMBERSHIP", LANGUAGE_CONFIG)
        assert result.language in ["en", "unknown"]
        assert result.confidence <= 0.7

    def test_numeric_punctuation_dominance(self, language_service):
        """Test text dominated by numbers and punctuation"""
        # Mostly numbers and punctuation
        result = language_service.detect_language_config_driven("— — — 12345", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence <= 0.3
        assert result.details["reason"] == "excessive_non_alphabetic_chars"
        
        # Mixed with some letters
        result = language_service.detect_language_config_driven("12345 $$$$$$$ ABC", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence <= 0.3
        
        # Phone number format
        result = language_service.detect_language_config_driven("+380-50-123-45-67", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence <= 0.3

    def test_normal_text_not_affected(self, language_service):
        """Test that normal text is not affected by edge case rules"""
        # Normal Russian text
        result = language_service.detect_language_config_driven("Платеж Иванову", LANGUAGE_CONFIG)
        assert result.language == "ru"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False
        
        # Normal English text
        result = language_service.detect_language_config_driven("Membership fee", LANGUAGE_CONFIG)
        assert result.language == "en"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False
        
        # Normal Ukrainian text
        result = language_service.detect_language_config_driven("Переказ коштів", LANGUAGE_CONFIG)
        assert result.language == "uk"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False

    def test_mixed_case_acronyms(self, language_service):
        """Test mixed case acronyms and abbreviations"""
        # Mixed case acronym (should not trigger penalty)
        result = language_service.detect_language_config_driven("iPhone", LANGUAGE_CONFIG)
        assert result.language == "en"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False
        
        # Abbreviation with periods
        result = language_service.detect_language_config_driven("Mr. Smith", LANGUAGE_CONFIG)
        assert result.language == "en"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False

    def test_long_words_not_affected(self, language_service):
        """Test that long words are not incorrectly flagged as acronyms"""
        # Long Russian word
        result = language_service.detect_language_config_driven("ПЕРЕВОДОПОЛУЧАТЕЛЬ", LANGUAGE_CONFIG)
        assert result.language == "ru"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False
        
        # Long English word
        result = language_service.detect_language_config_driven("INTERNATIONALIZATION", LANGUAGE_CONFIG)
        assert result.language == "en"
        assert result.confidence >= 0.6
        assert result.details.get("is_likely_acronym", False) is False

    def test_edge_case_confidence_ranges(self, language_service):
        """Test that edge cases have appropriate confidence ranges"""
        # Short text should have low confidence
        result = language_service.detect_language_config_driven("ОО", LANGUAGE_CONFIG)
        assert 0.0 <= result.confidence <= 0.5
        
        # Acronyms should have reduced confidence
        result = language_service.detect_language_config_driven("ООО", LANGUAGE_CONFIG)
        assert 0.0 <= result.confidence <= 0.6
        
        # Noisy text should have very low confidence
        result = language_service.detect_language_config_driven("— — — 12345", LANGUAGE_CONFIG)
        assert 0.0 <= result.confidence <= 0.3
        
        # Normal text should have good confidence
        result = language_service.detect_language_config_driven("Платеж", LANGUAGE_CONFIG)
        assert result.confidence >= 0.6

    def test_empty_and_whitespace_text(self, language_service):
        """Test empty and whitespace-only text"""
        # Empty string
        result = language_service.detect_language_config_driven("", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.details["reason"] == "empty_text"
        
        # Whitespace only
        result = language_service.detect_language_config_driven("   \t\n  ", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.details["reason"] == "empty_text"

    def test_special_characters_and_unicode(self, language_service):
        """Test text with special characters and unicode"""
        # Text with emojis and special chars
        result = language_service.detect_language_config_driven("Платеж 💰 от Иванова", LANGUAGE_CONFIG)
        assert result.language == "ru"
        assert result.confidence >= 0.6
        
        # Text with unicode symbols
        result = language_service.detect_language_config_driven("Payment → John Smith", LANGUAGE_CONFIG)
        assert result.language == "en"
        assert result.confidence >= 0.6

    def test_threshold_boundary_cases(self, language_service):
        """Test cases at the boundary of thresholds"""
        # Exactly 3 characters (should not be short text)
        result = language_service.detect_language_config_driven("ООО", LANGUAGE_CONFIG)
        # This might be detected as acronym, but should not be short text
        assert result.details["total_letters"] >= 3
        
        # Exactly 70% non-alphabetic (should trigger noisy text)
        result = language_service.detect_language_config_driven("1234567ABC", LANGUAGE_CONFIG)
        assert result.language == "unknown"
        assert result.details["reason"] == "excessive_non_alphabetic_chars"
        
        # Just under 70% non-alphabetic (should not trigger)
        result = language_service.detect_language_config_driven("123456ABC", LANGUAGE_CONFIG)
        assert result.language != "unknown" or result.details["reason"] != "excessive_non_alphabetic_chars"
