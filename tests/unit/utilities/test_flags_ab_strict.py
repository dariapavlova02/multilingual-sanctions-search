#!/usr/bin/env python3
"""
Strict flag-branching tests for NormalizationService (generic).

Checks:
 1) enable_advanced_features=False → "Сергея Петрова" stays inflected (does NOT become "Сергей Петров").
 2) preserve_names=False → "O'Brien" gets split.
 3) remove_stop_words=False → STOP_ALL words are not removed.
"""

import pytest

from ai_service.layers.normalization.normalization_service import NormalizationService
from ai_service.data.dicts.stopwords import STOP_ALL


def test_enable_advanced_features_false_no_morph():
    service = NormalizationService()
    text = "Сергея Петрова"
    result = service.normalize(
        text,
        language="ru",
        enable_advanced_features=False,
        remove_stop_words=False,
        preserve_names=True,
    )

    norm = result.normalized
    # Should not morph both tokens to nominative (lenient check)
    assert "Сергей Петров" not in norm, f"Unexpected morphing to nominative: {norm}"


def test_preserve_names_false_splits_apostrophe():
    service = NormalizationService()
    text = "Переказ коштів на ім'я O'Brien Петро-Іванович Коваленко"
    result = service.normalize(
        text,
        language="uk",
        preserve_names=False,
        remove_stop_words=False,
        enable_advanced_features=False,
    )

    norm = result.normalized.lower()
    tokens = [t.lower() for t in result.tokens]

    # O'Brien should be split
    assert "o'brien" not in norm, f"Expected O'Brien to be split when preserve_names=False: {result.normalized}"
    assert any(t == "o" for t in tokens), f"Expected token 'O' present: {result.tokens}"
    assert any("brien" == t or "бриен" == t for t in tokens), f"Expected token 'Brien' present: {result.tokens}"


def test_remove_stop_words_false_keeps_stopwords():
    service = NormalizationService()
    text = "Переказ коштів від імені Петро Іванович Коваленко"
    result = service.normalize(
        text,
        language="uk",
        remove_stop_words=False,
        preserve_names=True,
        enable_advanced_features=False,
    )

    tokens_lower = [t.lower() for t in result.tokens]
    found = [w for w in tokens_lower if w in STOP_ALL]
    assert found, f"Expected STOP_ALL words to remain when remove_stop_words=False, tokens={result.tokens}"
