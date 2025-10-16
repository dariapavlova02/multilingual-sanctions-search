"""
Validated configuration for the supported processing features.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import asdict, dataclass, fields
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from threading import Lock, RLock

import yaml
from .config_aliases import accept_flag_aliases, canonical_flag_names, FLAG_ALIASES


REMOVED_NORMALIZATION_OPTIONS = frozenset({
    'enable_accuracy_monitoring',
    'enable_dual_processing',
    'enable_performance_fallback',
    'factory_rollout_percentage',
    'language_overrides',
    'log_implementation_choice',
    'max_latency_threshold_ms',
    'min_confidence_threshold',
    'normalization_implementation',
    'use_factory_normalizer',
})

@accept_flag_aliases
@dataclass
class FeatureFlags:
    """Feature flags configuration."""

    # Diagnostic tracing
    debug_tracing: bool = False  # Enable debug tracing

    # Normalization and search behavior
    fix_initials_double_dot: bool = False  # Collapse И.. → И.
    preserve_hyphenated_case: bool = False  # Петрова-сидорова → Петрова-Сидорова
    strict_stopwords: bool = False  # Filter stopwords from tokens
    enable_ac_tier0: bool = True
    enable_vector_fallback: bool = True

    # Morphology flags
    morphology_custom_rules_first: bool = True  # Apply custom rules before pymorphy3

    # English-specific flags
    enable_nameparser_en: bool = True  # Enable nameparser for English name parsing
    enable_en_nicknames: bool = True  # Enable English nickname resolution

    # Validation and NER flags
    enable_spacy_ner: bool = True  # Enable spaCy NER processing - ВАЖНО для качества!
    enable_spacy_uk_ner: bool = (
        True  # Enable spaCy Ukrainian NER - КРИТИЧНО для украинского!
    )
    enable_spacy_en_ner: bool = (
        True  # Enable spaCy English NER - ВАЖНО для английского!
    )
    enable_fsm_tuned_roles: bool = (
        True  # Use FSM-tuned role detection - улучшенная логика
    )
    enable_enhanced_diminutives: bool = True  # Enhanced diminutive handling
    enable_enhanced_gender_rules: bool = (
        True  # Enhanced gender rule processing - ВАЖНО!
    )
    preserve_feminine_suffix_uk: bool = (
        True  # Preserve Ukrainian feminine suffixes - КРИТИЧНО!
    )
    en_use_nameparser: bool = True  # Use nameparser for English names
    enable_en_nickname_expansion: bool = True  # Expand English nicknames
    filter_titles_suffixes: bool = True  # Filter out titles and suffixes from EN names

    # Business gates
    require_tin_dob_gate: bool = True  # Require TIN/DOB for strong name matches

    # Nominative and gender enforcement flags
    enforce_nominative: bool = True
    preserve_feminine_surnames: bool = True

    # ASCII fastpath optimization
    enable_ascii_fastpath: bool = True

    # Diminutive resolution
    use_diminutives_dictionary_only: bool = False
    diminutives_allow_cross_lang: bool = False


    def to_dict(self) -> Dict[str, Any]:
        """Return every stored field, with owned JSON-compatible values.

        This representation is used in processing cache keys as well as diagnostics.
        Omitting a flag here can reuse a result produced under a different policy.
        """
        return asdict(self)


def validated_feature_flags(value) -> FeatureFlags:
    """Validate a complete or partial configuration without coercing JSON booleans."""
    if isinstance(value, FeatureFlags):
        if set(vars(value)) - {field.name for field in fields(FeatureFlags)}:
            raise ValueError("Unknown feature flag attribute")
        values = {
            field.name: deepcopy(getattr(value, field.name))
            for field in fields(FeatureFlags)
        }
    elif isinstance(value, Mapping):
        values = canonical_flag_names(deepcopy(dict(value)))
    else:
        raise ValueError("Feature flags must be a configuration object")
    definitions = {field.name: field for field in fields(FeatureFlags)}
    if set(values) & REMOVED_NORMALIZATION_OPTIONS:
        raise ValueError("Normalization mode selection was removed; remove legacy routing options")
    if set(values) - definitions.keys():
        raise ValueError("Unknown feature flag")
    for value in values.values():
        if type(value) is not bool:
            raise ValueError("Feature flag boolean must be true or false")
    return FeatureFlags(**values)


def merge_feature_flags(base: FeatureFlags, overrides=None) -> FeatureFlags:
    """Copy a validated base, then apply only explicitly supplied override fields."""
    base = validated_feature_flags(base)
    if overrides is None:
        return base
    if isinstance(overrides, FeatureFlags):
        return validated_feature_flags(overrides)
    if not isinstance(overrides, Mapping):
        raise ValueError("Feature flag overrides must be an object")
    overrides = canonical_flag_names(deepcopy(dict(overrides)))
    values = base.to_dict()
    values.update(overrides)
    return validated_feature_flags(values)


class _FlagYamlLoader(yaml.SafeLoader):
    """Small explicit configuration files may not use aliases or duplicate keys."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported in feature flags")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("Feature flag YAML requires unique string keys")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_ENV_ALIASES = {
    "fix_initials_double_dot": "FIX_INITIALS_DOUBLE_DOT",
    "preserve_hyphenated_case": "PRESERVE_HYPHENATED_CASE",
    "use_diminutives_dictionary_only": "USE_DIMINUTIVES_DICTIONARY_ONLY",
    "diminutives_allow_cross_lang": "DIMINUTIVES_ALLOW_CROSS_LANG",
}


def _environment_value(raw):
    raw = raw.strip().lower()
    if raw in {"true", "1", "yes", "on", "y"}:
        return True
    if raw in {"false", "0", "no", "off", "n"}:
        return False
    raise ValueError("Invalid feature flag environment boolean")


class FeatureFlagManager:
    """Validated process configuration with owned, atomic snapshots.

    Precedence: runtime defaults < explicitly selected YAML < environment < request.
    Deployment changes require process recreation; update_flags is a library API.
    """

    def __init__(self):
        self._lock = RLock()
        self._flags = self._load_from_environment()

    def _load_from_environment(self) -> FeatureFlags:
        # Preserve the established runtime profile. Bare FeatureFlags remains the
        # explicit library constructor; deployment consumers all use this loader.
        removed_env = {name.upper() for name in REMOVED_NORMALIZATION_OPTIONS}
        if any(key in removed_env or key.startswith("NORMALIZATION_IMPLEMENTATION_")
               for key in os.environ):
            raise ValueError("Normalization mode selection was removed; remove legacy environment options")
        flags = FeatureFlags(use_diminutives_dictionary_only=True)
        path = os.getenv("AISVC_FEATURE_FLAGS_FILE")
        if path is not None:
            try:
                if not path.strip():
                    raise ValueError("Empty feature flags path")
                with Path(path).open("rb") as stream:
                    content = stream.read(65537)
                if len(content) > 65536:
                    raise ValueError("Feature flags file exceeds 64 KiB")
                document = yaml.load(content, Loader=_FlagYamlLoader)
                environment = os.getenv("APP_ENV", "development")
                if not isinstance(document, dict) or environment not in document:
                    raise ValueError("Missing feature flags environment")
                profile = document[environment]
                if not isinstance(profile, dict) or set(profile) != {"feature_flags"}:
                    raise ValueError("Invalid feature flags profile")
                if not isinstance(profile["feature_flags"], dict):
                    raise ValueError("Invalid feature flags section")
                flags = merge_feature_flags(flags, profile["feature_flags"])
            except (OSError, ValueError, yaml.YAMLError, RecursionError):
                raise ValueError(
                    "Invalid explicitly selected feature flags file"
                ) from None

        definitions = {field.name: field for field in fields(FeatureFlags)}
        aliases = {
            "AISVC_FLAG_" + old.upper(): new for old, new in FLAG_ALIASES.items()
        }
        canonical = {"AISVC_FLAG_" + name.upper(): name for name in definitions}
        if any(
            key.startswith("AISVC_FLAG_")
            and key not in canonical
            and key not in aliases
            for key in os.environ
        ):
            raise ValueError("Unknown feature flag environment variable")
        overrides = {}
        for name, field in definitions.items():
            candidates = ["AISVC_FLAG_" + name.upper()]
            candidates.extend(key for key, target in aliases.items() if target == name)
            if name in _ENV_ALIASES:
                candidates.append(_ENV_ALIASES[name])
            for key in candidates:
                if key in os.environ:
                    overrides[name] = _environment_value(os.environ[key])
                    break
        return merge_feature_flags(flags, overrides)

    def get_flags(self, overrides=None) -> FeatureFlags:
        """Return a validated independent snapshot, optionally with request overrides."""
        with self._lock:
            return merge_feature_flags(self._flags, overrides)


    def update_flags(self, **kwargs) -> None:
        """Validate the whole update before atomically publishing a new snapshot."""
        with self._lock:
            self._flags = merge_feature_flags(self._flags, kwargs)

    def get_current_config(self) -> Dict[str, Any]:
        """Return all effective fields as an owned JSON-compatible dictionary."""
        return self.get_flags().to_dict()

    def use_diminutives_dictionary_only(self) -> bool:
        """Return whether diminutive resolution should rely solely on dictionaries."""
        return self._flags.use_diminutives_dictionary_only

    def allow_diminutives_cross_lang(self) -> bool:
        """Return whether cross-language diminutive lookup is permitted."""
        return self._flags.diminutives_allow_cross_lang


# Global feature flag manager instance
_feature_flag_manager = None
_manager_lock = Lock()


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get global feature flag manager instance."""
    global _feature_flag_manager
    with _manager_lock:
        if _feature_flag_manager is None:
            _feature_flag_manager = FeatureFlagManager()
        return _feature_flag_manager
