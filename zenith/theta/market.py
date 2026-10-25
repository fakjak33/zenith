"""The simulated market: aligned daily arrays per ticker plus option pricing.

`Market` is pure (no Streamlit). The view builds one from the release parquet
(`Market.load()`, cached by the caller); tests build one from synthetic frames
(`Market(frames, universe, earnings)`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ivmodel
from .engine import pricing

FIELDS = ("open", "high", "low", "close", "volume", "rv21", "iv_base", "em", "next_earn",
          "iv30", "iv_rank", "iv_pct", "skew", "beta")
SPREAD_PCT = {"A": 0.03, "B": 0.06, "C": 0.12}      # option bid/ask width as a share of mid
MIN_TICK = 0.01


def strike_step(S: float) -> float:
    for lim, step in ((25, 0.5), (100, 1.0), (250, 2.5), (1000, 5.0)):
        if S < lim:
            return step
    return 10.0


def round_strike(K: float, S: float) -> float:
    st = strike_step(S)
    return round(round(K / st) * st, 2)


class Market:
    def __init__(self, frames: dict[str, pd.DataFrame], universe: list[dict],
                 earnings: dict[str, list[str]] | None = None):
        vix = frames["_VIX"]
        self.dates = pd.DatetimeIndex(vix.index)
        self.vix = vix["close"].to_numpy(float)
        rate = frames.get("_RATE")
        self.rate = (rate["close"].reindex(self.dates).ffill().fillna(0.0).to_numpy(float)
                     if rate is not None else np.zeros(len(self.dates)))
        self.meta = {u["ticker"]: u for u in universe if u["ticker"] in frames}
        self.tickers = [u["ticker"] for u in universe if u["ticker"] in frames]
        self.data: dict[str, dict[str, np.ndarray]] = {}
        for t in self.tickers:
            df = frames[t].reindex(self.dates)
            self.data[t] = {c: df[c].to_numpy(float) if c in df else np.full(len(self.dates), np.nan)
                            for c in FIELDS}
        self.earn_idx: dict[str, np.ndarray] = {}
        for t, ds in (earnings or {}).items():
            if t in self.data and ds:
                ix = self.dates.searchsorted(pd.DatetimeIndex(pd.to_datetime(ds)), side="left")
                self.earn_idx[t] = np.unique(ix[ix < len(self.dates)])
        self._exp_cache: dict[bool, np.ndarray] = {}

    # ------------------------------------------------------------ loading ----
    @classmethod
    def load(cls, path=None) -> "Market | None":
        from ..config import THETA_ASSET, THETA_FILES
        from ..ephemeris.store_px import ensure_local
        from . import load as tload
        p = path or ensure_local(THETA_ASSET)
        if p is None:
            return None
        df = pd.read_parquet(p)
        frames = {t: g.drop(columns="ticker").set_index("ts").sort_index()
                  for t, g in df.groupby("ticker", sort=False)}
        uni = tload("universe", {}).get("rows", [])
        earn = tload("earnings", {}).get("dates", {})
        return cls(frames, uni, earn)

    # ------------------------------------------------------------- basics ----
    def __len__(self) -> int:
        return len(self.dates)

    def idx(self, date) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date), side="left"))

    def get(self, t: str, field: str, i: int) -> float:
        return float(self.data[t][field][i])

    def spot(self, t: str, i: int) -> float:
        return self.get(t, "close", i)

    def live(self, t: str, i: int) -> bool:
        return t in self.data and np.isfinite(self.data[t]["close"][i]) and \
            np.isfinite(self.data[t]["iv_base"][i])

    def days_between(self, i: int, j: int) -> int:
        return int((self.dates[j] - self.dates[i]).days)

    def tier(self, t: str) -> str:
        return self.meta.get(t, {}).get("tier", "B")

    def kind(self, t: str) -> str:
        return self.meta.get(t, {}).get("kind", "stock")

    # ---------------------------------------------------------- expiries ----
    def expiries(self, weekly: bool) -> np.ndarray:
        """Indices of expiration days: the last trading day of each week (weekly)
        or of each third-Friday week (monthly)."""
        if weekly not in self._exp_cache:
            d = self.dates
            week = d.to_period("W-FRI")
            last_of_week = np.flatnonzero(np.r_[week[1:] != week[:-1], True])
            if not weekly:
                fri = d[last_of_week]
                third = [(x.day + (4 - x.weekday()) % 7 - 1) // 7 == 2 for x in fri]
                last_of_week = last_of_week[np.array(third, dtype=bool)]
            self._exp_cache[weekly] = last_of_week
        return self._exp_cache[weekly]

    def expiry_choices(self, t: str, i: int, max_dte: int = 120) -> list[int]:
        weekly = self.tier(t) == "A"
        ex = self.expiries(weekly)
        out = [int(e) for e in ex if e > i and self.days_between(i, int(e)) <= max_dte]
        return out

    def nearest_expiry(self, t: str, i: int, dte: int) -> int | None:
        choices = [e for e in self.expiries(False) if e > i]
        if not choices:
            return None
        return int(min(choices, key=lambda e: abs(self.days_between(i, e) - dte)))

    # ---------------------------------------------------------- earnings ----
    def earnings_between(self, t: str, i: int, j: int) -> bool:
        ix = self.earn_idx.get(t)
        if ix is None or not len(ix):
            return False
        k = np.searchsorted(ix, i, side="right")
        return bool(k < len(ix) and ix[k] <= j)

    def next_report(self, t: str, i: int) -> int | None:
        ix = self.earn_idx.get(t)
        if ix is None:
            return None
        k = np.searchsorted(ix, i, side="right")
        return int(ix[k]) if k < len(ix) else None

    # ------------------------------------------------------------ options ----
    def T(self, i: int, expiry_i: int) -> float:
        return max(self.days_between(i, expiry_i), 0) / 365.0

    def iv(self, t: str, i: int, K, expiry_i: int, flat: bool = False):
        T = self.T(i, expiry_i)
        base = self.get(t, "iv_base", i)
        if flat:
            return np.full(np.shape(K), base) if np.ndim(K) else base
        em = self.get(t, "em", i)
        earn = self.earnings_between(t, i, expiry_i)
        atm = ivmodel.atm_vol(base, 0.0 if not np.isfinite(em) else em, T, earn)
        skew = self.get(t, "skew", i)
        return ivmodel.smile(atm, self.spot(t, i), K, T, 0.0 if not np.isfinite(skew) else skew)

    def option_mid(self, t: str, i: int, call: bool, K, expiry_i: int, flat: bool = False):
        T = self.T(i, expiry_i)
        sig = self.iv(t, i, K, expiry_i, flat)
        return pricing.price(call, self.spot(t, i), K, T, sig, self.rate[i])

    def half_spread(self, t: str, mid: float) -> float:
        return max(MIN_TICK / 2, SPREAD_PCT[self.tier(t)] * float(mid) / 2.0)

    def strike_for_delta(self, t: str, i: int, expiry_i: int, delta: float, call: bool,
                         flat: bool = False) -> float:
        """Listed strike nearest the target |delta| under the smile (3 fixed-point passes)."""
        S, T, r = self.spot(t, i), max(self.T(i, expiry_i), 1 / 365), self.rate[i]
        sig = self.get(t, "iv_base", i)
        K = S
        for _ in range(3):
            K = pricing.strike_for_delta(S, T, float(sig), r, delta, call)
            sig = float(self.iv(t, i, K, expiry_i, flat))
        return round_strike(K, S)

    # ----------------------------------------------------------- screener ----
    def snapshot(self, i: int) -> list[dict]:
        rows = []
        for t in self.tickers:
            if not self.live(t, i):
                continue
            d = self.data[t]
            S, iv30 = d["close"][i], d["iv30"][i]
            nx = int(d["next_earn"][i]) if np.isfinite(d["next_earn"][i]) else -1
            em30 = S * iv30 * np.sqrt(30 / 365.0)
            rows.append({"ticker": t, "kind": self.kind(t), "tier": self.tier(t),
                         "sector": self.meta[t].get("sector", ""), "price": float(S),
                         "iv": float(iv30), "ivr": float(d["iv_rank"][i]),
                         "ivp": float(d["iv_pct"][i]), "rv": float(d["rv21"][i]),
                         "iv_rv": float(iv30 / d["rv21"][i]) if d["rv21"][i] > 0 else np.nan,
                         "em_30d": float(em30), "earn_days": nx, "beta": float(d["beta"][i])})
        return rows
