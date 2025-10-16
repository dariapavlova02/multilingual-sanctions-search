"""Reject invalid explicit policy; valid request flags are independent copies.

The old silent-fallback and shared-object expectations are retired. An invalid
policy cannot become a successful request using a different configuration.
"""

from unittest.mock import MagicMock

import pytest

from ai_service.core.unified_orchestrator import UnifiedOrchestrator
from ai_service.utils.feature_flags import FeatureFlags


class TestDefensiveFlagsHandling:
    @pytest.fixture
    def mock_orchestrator(self):
        return UnifiedOrchestrator(
            validation_service=MagicMock(),
            language_service=MagicMock(),
            unicode_service=MagicMock(),
            normalization_service=MagicMock(),
            signals_service=MagicMock(),
            default_feature_flags=FeatureFlags(),
        )

    def test_none_flags_uses_defaults(self, mock_orchestrator):
        result = mock_orchestrator._validate_and_normalize_flags(None)
        assert result == mock_orchestrator.default_feature_flags
        assert result is not mock_orchestrator.default_feature_flags

    def test_partial_mapping_preserves_other_defaults(self, mock_orchestrator):
        mock_orchestrator.default_feature_flags.strict_stopwords = True
        result = mock_orchestrator._validate_and_normalize_flags(
            {"fix_initials_double_dot": True}
        )
        assert result.fix_initials_double_dot is True
        assert result.strict_stopwords is True
        assert mock_orchestrator.default_feature_flags.fix_initials_double_dot is False

    @pytest.mark.parametrize("invalid", ["false", None, 123])
    def test_invalid_boolean_values_rejected(self, mock_orchestrator, invalid):
        flags = FeatureFlags()
        flags.strict_stopwords = invalid
        with pytest.raises(ValueError):
            mock_orchestrator._validate_and_normalize_flags(flags)
        assert flags.strict_stopwords == invalid

    def test_valid_flags_passed_through(self, mock_orchestrator):
        flags = FeatureFlags(
            fix_initials_double_dot=True,
            preserve_hyphenated_case=False,
            strict_stopwords=True,
            enable_ac_tier0=False,
            enable_vector_fallback=False,
        )
        result = mock_orchestrator._validate_and_normalize_flags(flags)
        assert result.to_dict() == flags.to_dict()
        assert result is not flags
        result.strict_stopwords = False
        assert flags.strict_stopwords is True

    def test_unsupported_object_rejected(self, mock_orchestrator):
        class BrokenFlags:
            def to_dict(self):
                raise RuntimeError("not a configuration")

        with pytest.raises(ValueError):
            mock_orchestrator._validate_and_normalize_flags(BrokenFlags())

    def test_mixed_valid_invalid_flags_rejected_atomically(self, mock_orchestrator):
        with pytest.raises(ValueError):
            mock_orchestrator._validate_and_normalize_flags(
                {"strict_stopwords": True, "fix_initials_double_dot": "invalid"}
            )
        assert mock_orchestrator.default_feature_flags.strict_stopwords is False

    def test_flags_with_unknown_attributes_rejected(self, mock_orchestrator):
        flags = FeatureFlags()
        flags.unknown_flag = True
        with pytest.raises(ValueError):
            mock_orchestrator._validate_and_normalize_flags(flags)

    def test_empty_feature_flags_object(self, mock_orchestrator):
        flags = FeatureFlags()
        result = mock_orchestrator._validate_and_normalize_flags(flags)
        assert result == flags
        assert result is not flags
        assert result.enable_ac_tier0 is True
        assert result.enable_vector_fallback is True

    async def test_flags_validation_in_process_method(self, mock_orchestrator):
        with pytest.raises(ValueError):
            await mock_orchestrator.process(
                "Synthetic Example", feature_flags={"invalid": True}
            )
        mock_orchestrator.validation_service.validate_and_sanitize.assert_not_called()
        mock_orchestrator.normalization_service.normalize_async.assert_not_called()
