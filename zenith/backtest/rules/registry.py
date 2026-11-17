"""Plug-and-play strategy registry.

Each strategy module defines a subclass of ``Strategy`` and decorates it with
``@register``. The Streamlit UI reads ``REGISTRY`` to list strategies and
build parameter controls automatically — adding a new rule needs no UI edits.
"""

from __future__ import annotations

from .base import Strategy

REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a Strategy to the global registry."""
    key = cls.key or cls.__name__.lower()
    if key in REGISTRY:
        raise ValueError(f"Duplicate strategy key {key!r}.")
    REGISTRY[key] = cls
    return cls


def get(key: str) -> type[Strategy]:
    return REGISTRY[key]


def available() -> list[str]:
    return sorted(REGISTRY)
