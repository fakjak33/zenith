"""Draw a random blind chart: (ticker, decision index) such that warm-up +
window + horizon fit inside the ticker's history.

Sampling is class-balanced (pick a class uniformly among the selected ones,
then a ticker uniformly within it) -- otherwise the ~1,000 Russell names
would drown out the eleven precious-metal funds. Windows with bad data are
rejected and redrawn (VELA's windowIsClean, generalised per timeframe), and a
chart overlapping one the player has already seen (same ticker/timeframe,
decision within one window length) is skipped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import MAX_BAR_MOVE, MAX_BAR_MOVE_DEFAULT, MIN_DOLLAR_VOLUME, NO_VOLUME_CLASSES, WARMUP
from . import regime

CRISES = {"2000–02 dot-com": ("2000-03-01", "2002-10-31"),
          "2008 GFC": ("2007-10-01", "2009-03-31"),
          "2020 COVID": ("2020-02-15", "2020-06-30"),
          "2022 bear": ("2022-01-01", "2022-10-31")}

# largest tolerated calendar gap between consecutive bars, per timeframe
MAX_GAP_DAYS = {"1H": 5, "4H": 5, "Daily": 10, "Weekly": 21, "Monthly": 70}
MAX_TRIES = 60


class NoChartError(RuntimeError):
    """No eligible chart for these settings."""


@dataclass
class Chart:
    ticker: str
    name: str
    cls: str
    sector: str
    tf: str
    lookback: int
    horizon: int
    ts: np.ndarray            # timestamps, warm-up + window + horizon
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    t: int                    # decision index INTO these arrays (last visible bar)
    extra: dict = field(default_factory=dict)

    @property
    def vis0(self) -> int:
        return self.t - self.lookback + 1

    @property
    def entry(self) -> float:
        return float(self.c[self.t])

    @property
    def decision_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.ts[self.t])

    @property
    def window_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.ts[self.vis0])

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.ts[self.t + self.horizon])

    @property
    def key(self) -> str:
        return chart_key(self.ticker, self.tf, self.decision_date)

    def future(self) -> dict[str, np.ndarray]:
        s = slice(self.t + 1, self.t + 1 + self.horizon)
        return {"o": self.o[s], "h": self.h[s], "l": self.l[s], "c": self.c[s]}


def chart_key(ticker: str, tf: str, decision) -> str:
    return f"{ticker}|{tf}|{pd.Timestamp(decision).strftime('%Y-%m-%d %H:%M')}"


def window_is_clean(o, h, l, c, v, ts, cls: str, tf: str) -> bool:
    """Reject bad prints, unadjusted splits, stale runs, zero-volume runs, gaps."""
    arr = np.vstack([o, h, l, c])
    if not np.isfinite(arr).all() or (arr <= 0).any():
        return False
    rets = np.abs(np.diff(np.log(c)))
    if rets.size and np.expm1(rets.max()) > MAX_BAR_MOVE.get(cls, MAX_BAR_MOVE_DEFAULT) * (
            3 if tf in ("Weekly", "Monthly") else 1):
        return False
    gaps = np.diff(ts).astype("timedelta64[s]").astype(np.int64) / 86400.0
    if gaps.size and gaps.max() > MAX_GAP_DAYS.get(tf, 10):
        return False
    if _max_run(np.diff(c) == 0) >= 5:          # stale / forward-filled prices
        return False
    if cls not in NO_VOLUME_CLASSES and cls != "Indices":
        if _max_run(v <= 0) >= 5:
            return False
        if tf in ("Daily", "Weekly", "Monthly") and np.median(c * v) / (
                {"Daily": 1, "Weekly": 5, "Monthly": 21}[tf]) < MIN_DOLLAR_VOLUME:
            return False
    return True


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _era_ok(d: np.datetime64, min_year: int | None, crisis_only: bool) -> bool:
    ts = pd.Timestamp(d)
    if min_year and ts.year < min_year:
        return False
    if crisis_only:
        return any(pd.Timestamp(a) <= ts <= pd.Timestamp(b) for a, b in CRISES.values())
    return True


def make_chart(df: pd.DataFrame, row: dict, tf: str, lookback: int, horizon: int, t: int,
               check_clean: bool = True) -> Chart | None:
    """Chart whose decision bar is df row t (warm-up kept for indicators), or
    None when the window fails the bad-data filter."""
    if t < lookback - 1 or t + horizon >= len(df):
        return None
    ts = df.index.values
    s0 = max(0, t - lookback + 1 - WARMUP)
    s1 = t + horizon + 1
    o, h, l, c, v = (df[k].to_numpy(np.float64)[s0:s1] for k in ("open", "high", "low", "close", "volume"))
    w0 = t - lookback + 1 - s0
    if check_clean and not window_is_clean(o[w0:], h[w0:], l[w0:], c[w0:], v[w0:], ts[s0:s1][w0:],
                                           row["cls"], tf):
        return None
    full = [df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close")]
    return Chart(ticker=row["ticker"], name=row.get("name", row["ticker"]), cls=row["cls"],
                 sector=row.get("sector", ""), tf=tf, lookback=lookback, horizon=horizon,
                 ts=ts[s0:s1], o=o, h=h, l=l, c=c, v=v, t=t - s0,
                 extra={"regime": regime.tags(*full, t, tf)})


def chart_at(store, row: dict, tf: str, lookback: int, horizon: int, decision) -> Chart | None:
    """Rebuild the chart whose decision bar is at (or just before) `decision`."""
    df = store.bars(row["ticker"], tf)
    if df is None:
        return None
    t = int(np.searchsorted(df.index.values, np.datetime64(pd.Timestamp(decision)), side="right")) - 1
    return make_chart(df, row, tf, lookback, horizon, t, check_clean=False)


def eligible(universe: list[dict], counts: dict[str, int], classes, lookback: int,
             horizon: int) -> dict[str, list[dict]]:
    need = lookback + horizon + 1
    out: dict[str, list[dict]] = {}
    for r in universe:
        if r["cls"] in classes and counts.get(r["ticker"], 0) >= need:
            out.setdefault(r["cls"], []).append(r)
    return out


def draw(store, universe: list[dict], *, classes, tf: str = "Daily", lookback: int = 120,
         horizon: int = 10, rng: random.Random | None = None, seen: dict | None = None,
         min_year: int | None = None, crisis_only: bool = False) -> Chart:
    """One clean, unseen chart. `seen`: {(ticker, tf): [decision datetimes]}."""
    rng = rng or random.Random()
    seen = seen or {}
    pool = eligible(universe, store.counts(tf), classes, lookback, horizon)
    if not pool:
        raise NoChartError("No ticker in the selected classes has enough history for this "
                           "lookback + horizon at this timeframe.")
    cls_list = sorted(pool)
    for _ in range(MAX_TRIES):
        row = rng.choice(pool[rng.choice(cls_list)])
        df = store.bars(row["ticker"], tf)
        if df is None or len(df) < lookback + horizon + 1:
            continue
        ts = df.index.values
        lo, hi = lookback - 1, len(df) - horizon - 1      # valid decision indices
        t = rng.randint(lo, hi)
        if not _era_ok(ts[t], min_year, crisis_only):
            # one directed retry inside the era before giving up on this ticker
            ok = [i for i in range(lo, hi + 1, max(1, (hi - lo) // 400)) if _era_ok(ts[i], min_year, crisis_only)]
            if not ok:
                continue
            t = rng.choice(ok)
        prior = seen.get((row["ticker"], tf))
        if prior:
            pidx = np.searchsorted(ts, np.array(prior, dtype="datetime64[ns]"))
            if np.any(np.abs(pidx - t) < lookback):
                continue
        ch = make_chart(df, row, tf, lookback, horizon, t)
        if ch is not None:
            return ch
    raise NoChartError("Could not find a clean, unseen chart for these settings — "
                       "widen the universe or the era filter.")
