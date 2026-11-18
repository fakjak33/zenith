"""Execution delay: wait N bars after a signal before acting.

Models the common "confirmation" rule — e.g. wait 5 candles after an EMA cross
before entering. Exposure-*increasing* moves (opening/adding) are delayed by
``entry_delay`` bars; exposure-*reducing* moves (closing/trimming) by
``exit_delay``. A no-op when both are 0.

Implementation: walk the target position; the *applied* position only steps
toward a new target after the required number of bars have elapsed since that
target first appeared, so transient one-bar blips don't trigger early.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_delay(position: pd.Series, entry_delay: int = 0, exit_delay: int = 0) -> pd.Series:
    """Return a delayed copy of ``position``.

    Moves that increase |position| are held back ``entry_delay`` bars; moves that
    decrease |position| are held back ``exit_delay`` bars. The new level must
    persist for the full delay window before it is applied.
    """
    entry_delay = max(0, int(entry_delay))
    exit_delay = max(0, int(exit_delay))
    if entry_delay == 0 and exit_delay == 0:
        return position

    tgt = position.to_numpy(dtype=float)
    out = np.zeros_like(tgt)
    applied = 0.0
    pending = tgt[0]
    pending_age = 0

    for i in range(len(tgt)):
        if tgt[i] != pending:           # target changed → restart the timer
            pending = tgt[i]
            pending_age = 0
        else:
            pending_age += 1
        increasing = abs(pending) > abs(applied)
        need = entry_delay if increasing else exit_delay
        if pending_age >= need:
            applied = pending
        out[i] = applied
    return pd.Series(out, index=position.index)
