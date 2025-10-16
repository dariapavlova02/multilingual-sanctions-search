"""Explicit environment values must be valid before changing runtime behavior."""

import os


def parse_boolean(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}")


def environment_boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else parse_boolean(value, name=name)
