"""Strategy library. Importing this package registers all built-in rules."""

from . import ema, sma, gmma, crossover, tsmom, xsmom, breakout, meanrev, carry  # noqa: F401
from . import turtle, strev, tom, lowvol  # noqa: F401
from .registry import REGISTRY, register, get, available  # noqa: F401
