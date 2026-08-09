"""Source adapters — one module per fund company's holdings page.

An adapter exposes two functions:

    fetch(fund)        -> (html | None, meta)
    parse(html)        -> (rows, meta)

`rows` are raw-but-coerced dicts with the keys ``value_date, security_name,
cusip, ticker, qty, notional, weight``; everything downstream is generic.
Keeping fetch and parse separate is what lets `wayback.py` replay archived
copies of the very same page through the very same parser.
"""

from __future__ import annotations

from importlib import import_module


def get_adapter(name: str):
    """Import an adapter module by its registry `adapter` name."""
    return import_module(f"{__name__}.{name}")
