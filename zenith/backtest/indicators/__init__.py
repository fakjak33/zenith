"""Indicator library. Importing this package registers every indicator in
``CATALOG`` (see ``catalog.py`` for the entry format)."""

from . import trend, momentum, meanrev, volatility, volume, calendar, cross, structure  # noqa: F401
from .catalog import CATALOG, FAMILIES, Indicator, by_family  # noqa: F401
