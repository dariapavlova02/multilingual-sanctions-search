"""
Unit tests for FeatureFlags system.
"""

import pytest
import os
from unittest.mock import patch

from ai_service.utils.feature_flags import (
    FeatureFlags,
    FeatureFlagManager,
    get_feature_flag_manager,
)


class TestFeatureFlags:
    """Tests for FeatureFlags dataclass."""

    def test_default_values(self):
        """Test default feature flag values."""
        flags = FeatureFlags()

        # Test key defaults based on current settings
        assert flags.ascii_fastpath is True
        assert flags.enable_ac_tier0 is True
        assert flags.enable_vector_fallback is True
        assert flags.enforce_nominative is True
        assert flags.preserve_feminine_surnames is True

    def test_to_dict_method(self):
        """Test feature flags to_dict serialization."""
        flags = FeatureFlags()
        flags_dict = flags.to_dict()

        # Check required fields are present
        required_fields = [
            'enable_ascii_fastpath',
            'enable_ac_tier0',
            'enable_vector_fallback',
            'enforce_nominative',
            'preserve_feminine_surnames'
        ]

        for field in required_fields:
            assert field in flags_dict
            assert isinstance(flags_dict[field], bool)



class TestFeatureFlagManager:
    """Tests for FeatureFlagManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = FeatureFlagManager()
        assert hasattr(manager, '_flags')
        assert isinstance(manager._flags, FeatureFlags)

    @patch.dict(os.environ, {
        'AISVC_FLAG_ASCII_FASTPATH': 'false'
    })
    def test_environment_loading(self):
        """Test loading flags from environment variables."""
        manager = FeatureFlagManager()

        assert manager._flags.ascii_fastpath is False





    def test_update_flags(self):
        """Test programmatic flag updates."""
        manager = FeatureFlagManager()

        original_value = manager._flags.ascii_fastpath
        manager.update_flags(ascii_fastpath=not original_value)

        assert manager._flags.ascii_fastpath != original_value

    def test_current_config(self):
        """Test current configuration retrieval."""
        manager = FeatureFlagManager()
        config = manager.get_current_config()

        assert 'strict_stopwords' in config
        assert 'enable_spacy_ner' in config
        assert all(type(value) is bool for value in config.values())


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_feature_flag_manager_singleton(self):
        """Test global manager is singleton."""
        manager1 = get_feature_flag_manager()
        manager2 = get_feature_flag_manager()

        assert manager1 is manager2
