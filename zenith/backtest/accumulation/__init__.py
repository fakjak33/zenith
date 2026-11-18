"""Accumulation / DCA lab: long-term buy/sell trigger strategies vs DCA & buy-hold."""

from .engine import AccumConfig, run_accumulation, benchmarks  # noqa: F401
from .strategies import BUY_RULES, SELL_RULES  # noqa: F401
