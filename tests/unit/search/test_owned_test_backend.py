"""Regression fixtures must reject unowned deployment targets before writing."""

import pytest

from tests.owned_screening import validate_owned_target

NAME = "sanctions-regression-" + "a" * 32


@pytest.mark.parametrize("url", ["http://127.0.0.1:39200", "http://localhost:39200/", "http://[::1]:39200"])
def test_owned_loopback_endpoint(url):
    validate_owned_target(url, NAME)


@pytest.mark.parametrize("url,name", [
    ("http://production.example:9200", NAME), ("http://0.0.0.0:9200", NAME),
    ("http://127.0.0.1:9200", "production"), ("http://127.0.0.1:9200", "sanctions-regression-"),
    ("http://127.0.0.1", NAME), ("http://user:secret@127.0.0.1:9200", NAME),
    ("http://127.0.0.1:9200/private", NAME), ("http://127.0.0.1:9200?index=production", NAME),
    ("http://127.0.0.1:9200#production", NAME), ("https://127.0.0.1:9200", NAME),
])
def test_unowned_or_ambiguous_endpoint_is_rejected(url, name):
    with pytest.raises(ValueError):
        validate_owned_target(url, name)
