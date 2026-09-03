"""ETF MOMENTUM history — the ETF store bound to MOMENTUM's history logic.

There is no second implementation here. `mom/history.py` owns the sharded
yearly append, the decile pick tracker and the sign-adjusted SPY-excess
evaluation; those functions take optional `history_dir` / `load_fn` / `save_fn`
arguments precisely so this package can point them at data/etfmom/ instead of
data/mom/. Forking ~200 lines to change three paths would have duplicated the
fiddliest logic in the feature (the short-side `-raw` sign flip, the
`(date, ticker)` dedupe, the matured-horizon walk) for no benefit.

`make_pick_rows` and `summarize` are already pure and are re-exported as-is.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..config import ETFMOM_HISTORY_DIR
from ..mom import history as mom_history
from ..mom.history import PICK_HORIZONS_TD, make_pick_rows, summarize  # noqa: F401
from . import load, save


def append_history(rows: list[dict], today: date, full: bool | None = None) -> int:
    return mom_history.append_history(rows, today, full=full, history_dir=ETFMOM_HISTORY_DIR)


def series_for(ticker: str, start_year: int | None = None,
               end_year: int | None = None) -> list[dict]:
    return mom_history.series_for(ticker, start_year=start_year, end_year=end_year,
                                  history_dir=ETFMOM_HISTORY_DIR)


def append_picks(new_rows: list[dict]) -> int:
    return mom_history.append_picks(new_rows, load_fn=load, save_fn=save)


def evaluate_pending(px: dict, spy_close: pd.Series, today: date) -> int:
    return mom_history.evaluate_pending(px, spy_close, today, load_fn=load, save_fn=save)
