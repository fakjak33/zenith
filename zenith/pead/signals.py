"""PEAD signal math — pure functions, no network (fully unit-testable).

Component conventions
---------------------
Every component produces a RAW value first, then a 0-100 percentile RANK
against a trailing pool of recent announcements (same-day cross-sections are
too small — 5-50 Russell 1000 reporters — so ranking uses the pool of all
signals from the trailing ~63 trading days, one earnings cycle).

Ranks are direction-signed: a LONG's rank rewards big positive values of
SUE/EAR/RUE, a SHORT's rank rewards big negative ones (rank flips to 100-r).
Abnormal volume amplifies drift in BOTH directions (Garfinkel & Sokobin 2006)
so its rank is never flipped. The 52-week-high anchor flips because good news
NEAR the high (bad news FAR from it) shows the most underreaction (George,
Hwang & Li 2015).

    composite = 0.30*SUE + 0.30*EAR + 0.15*AVOL + 0.15*ANCHOR + 0.10*RUE
"""

from __future__ import annotations

import math
import re
from datetime import date

import pandas as pd

from ..pretom import calendar as cal

# Composite weights (documented in the tab's methodology expander).
WEIGHTS = {"sue": 0.30, "ear": 0.30, "avol": 0.15, "anchor": 0.15, "rue": 0.10}
CONVICTION = 60.0            # composite >= this = conviction tier
NEUTRAL_RANK = 50.0          # used when a component is unavailable (flagged)

# Eligibility gates.
MIN_PRICE = 5.0              # penny-stock filter
MIN_ADV_USD = 5e6            # median 60-td dollar volume
MIN_ABS_EAR = 0.005          # |announcement reaction| must clear 0.5%

# Context tier cuts (USD market cap).
CAP_TIERS = ((25e9, "large"), (5e9, "mid"), (0.0, "small"))

# Pool bookkeeping.
POOL_WINDOW_TD = 63          # trailing trading days a pool row stays in scope
POOL_MIN = 100               # below this the view shows a thin-pool caveat

POOL_KEYS = ("sue", "ear", "rue", "avol", "nearness")

_EPS_RX = re.compile(r"-?\$?\(?\s*-?\$?(\d[\d,]*(?:\.\d+)?)\s*\)?")


def parse_eps(s) -> float | None:
    """Parse Nasdaq-style EPS strings: '$1.23', '($0.45)' (negative), '(0.45)'."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return None if pd.isna(s) else float(s)
    txt = str(s).strip()
    if not txt or txt.upper() in ("N/A", "NA", "-", "--"):
        return None
    m = _EPS_RX.fullmatch(txt)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    neg = ("(" in txt and ")" in txt) or txt.lstrip("$").startswith("-")
    return -v if neg else v


def reaction_day(report_date: date, time_str: str) -> date:
    """First trading session in which the market can react to the report.

    Pre-market report -> that same day (or next trading day if closed);
    after-hours / unknown timing -> the next trading day. Nasdaq encodes the
    slot as 'time-pre-market' / 'time-after-hours' / 'time-not-supplied'.
    """
    pre = "pre" in (time_str or "").lower()
    if pre and cal.is_trading_day(report_date):
        return report_date
    return cal.next_trading_day(report_date)


def prev_trading_day(d: date) -> date:
    from datetime import timedelta
    d -= timedelta(days=1)
    while not cal.is_trading_day(d):
        d -= timedelta(days=1)
    return d


# --- component raw values ----------------------------------------------------

def sue_raw(actual: float | None, consensus: float | None,
            close_pre: float | None) -> float | None:
    """Price-scaled EPS surprise (Livnat & Mendenhall 2006 form). Scaling by
    price instead of |EPS| sidesteps zero/near-zero-EPS blowups."""
    if actual is None or consensus is None or not close_pre or close_pre <= 0:
        return None
    return (actual - consensus) / close_pre


def _asof_pos(index, d: date) -> int | None:
    """Position of the last bar with date <= d, or None."""
    dates = [ts.date() if hasattr(ts, "date") else ts for ts in index]
    lo = None
    for i, x in enumerate(dates):
        if x <= d:
            lo = i
        else:
            break
    return lo


def ear(px_close: pd.Series, spy_close: pd.Series, r_day: date) -> float | None:
    """Earnings announcement return (Brandt et al. 2008): stock close-to-close
    over the reaction session minus SPY over the same window. The pre-close ->
    reaction-close span covers both a pre-market gap and an after-hours gap."""
    out = []
    for s in (px_close, spy_close):
        if s is None or len(s) < 2:
            return None
        i = _asof_pos(s.index, r_day)
        if i is None or i == 0:
            return None
        d_i = s.index[i].date() if hasattr(s.index[i], "date") else s.index[i]
        if d_i != r_day:
            return None                     # reaction bar not printed yet
        prev, cur = float(s.iloc[i - 1]), float(s.iloc[i])
        if prev <= 0:
            return None
        out.append(cur / prev - 1.0)
    return out[0] - out[1]


def rue(quarterly_revenue: pd.Series) -> float | None:
    """Time-series revenue surprise (Jegadeesh & Livnat 2006, seasonal form):
    latest YoY quarterly revenue growth vs the mean/std of the prior YoY
    growths. Needs >= 8 quarters (>= 4 YoY observations)."""
    if quarterly_revenue is None:
        return None
    rev = quarterly_revenue.dropna().astype(float)
    if len(rev) < 8:
        return None
    yoy = rev.pct_change(4).dropna()
    if len(yoy) < 4:
        return None
    latest, prior = float(yoy.iloc[-1]), yoy.iloc[:-1]
    sd = float(prior.std(ddof=0))
    if not math.isfinite(latest) or sd == 0 or not math.isfinite(sd):
        return None
    return (latest - float(prior.mean())) / sd


def abnormal_volume(volume: pd.Series, r_day: date, lookback: int = 60) -> float | None:
    """log(reaction-day volume / trailing-median volume) — Garfinkel & Sokobin
    2006 use unexplained volume; the log-ratio vs the 60-td median is the free
    -data analogue."""
    if volume is None or volume.empty:
        return None
    i = _asof_pos(volume.index, r_day)
    if i is None or i < 20:
        return None
    d_i = volume.index[i].date() if hasattr(volume.index[i], "date") else volume.index[i]
    if d_i != r_day:
        return None
    win = volume.iloc[max(0, i - lookback):i].dropna()
    med = float(win.median()) if len(win) else 0.0
    v = float(volume.iloc[i])
    if med <= 0 or v <= 0:
        return None
    return math.log(v / med)


def nearness_52w(px: pd.DataFrame, pre_day: date) -> float | None:
    """close_pre / trailing-252-td high (George, Hwang & Li 2015 anchor)."""
    if px is None or px.empty:
        return None
    col = "high" if "high" in px.columns else "close"
    i = _asof_pos(px.index, pre_day)
    if i is None or i < 60:
        return None
    hi = float(px[col].iloc[max(0, i - 251):i + 1].max())
    c = float(px["close"].iloc[i])
    if hi <= 0 or c <= 0:
        return None
    return c / hi


def gap_held(px: pd.DataFrame, r_day: date, side: str) -> bool | None:
    """Did the reaction candle close in the signal direction (close >= open for
    longs, close <= open for shorts)? Context flag only — no score effect."""
    if px is None or px.empty or "open" not in px.columns:
        return None
    i = _asof_pos(px.index, r_day)
    if i is None:
        return None
    d_i = px.index[i].date() if hasattr(px.index[i], "date") else px.index[i]
    if d_i != r_day:
        return None
    o, c = float(px["open"].iloc[i]), float(px["close"].iloc[i])
    if o <= 0:
        return None
    return c >= o if side == "long" else c <= o


def momentum_12_1(px_close: pd.Series, pre_day: date) -> float | None:
    """12-1 month price momentum as of the pre-announcement close."""
    if px_close is None or px_close.empty:
        return None
    i = _asof_pos(px_close.index, pre_day)
    if i is None or i < 231:
        return None
    past, recent = float(px_close.iloc[i - 231]), float(px_close.iloc[i - 21])
    if past <= 0:
        return None
    return recent / past - 1.0


def adv_usd(px: pd.DataFrame, pre_day: date, lookback: int = 60) -> float | None:
    """Median trailing dollar volume through the pre-announcement close."""
    if px is None or px.empty or "volume" not in px.columns:
        return None
    i = _asof_pos(px.index, pre_day)
    if i is None or i < 20:
        return None
    win = px.iloc[max(0, i - lookback + 1):i + 1]
    dv = (win["close"] * win["volume"]).dropna()
    return float(dv.median()) if len(dv) else None


def cap_tier(mktcap: float | None) -> str | None:
    if mktcap is None or mktcap <= 0:
        return None
    for cut, tier in CAP_TIERS:
        if mktcap >= cut:
            return tier
    return "small"


# --- direction + gates -------------------------------------------------------

def side_of(sue: float | None, ear_v: float | None) -> str:
    """Two-confirmation direction: beat AND positive reaction -> long; miss AND
    negative reaction -> short; anything else -> mixed (never tracked)."""
    if sue is None or ear_v is None:
        return "mixed"
    if sue > 0 and ear_v > 0:
        return "long"
    if sue < 0 and ear_v < 0:
        return "short"
    return "mixed"


def gates(row: dict) -> str | None:
    """First failed eligibility gate, or None if the row is tradeable."""
    raw = row.get("raw", {})
    ctx = row.get("context", {})
    if row.get("eps_actual") is None:
        return "no_eps"
    if row.get("eps_consensus") is None:
        return "no_consensus"
    if raw.get("ear") is None:
        return "no_prices"
    if raw.get("sue") == 0:
        return "zero_surprise"
    if (ctx.get("close_pre") or 0) <= MIN_PRICE:
        return "penny"
    if (ctx.get("adv_usd") or 0) < MIN_ADV_USD:
        return "illiquid"
    if row.get("side") == "mixed":
        return "mixed"
    if abs(raw.get("ear") or 0) < MIN_ABS_EAR:
        return "weak_reaction"
    return None


# --- pool ranking + composite ------------------------------------------------

def pool_rank(value: float | None, pool: list[float]) -> float | None:
    """Self-inclusive percentile (0-100) of value within pool."""
    if value is None:
        return None
    vals = [v for v in pool if v is not None and math.isfinite(v)]
    if value not in vals:
        vals = vals + [value]
    if len(vals) == 1:
        return NEUTRAL_RANK
    below = sum(1 for v in vals if v < value)
    equal = sum(1 for v in vals if v == value)
    return round(100.0 * (below + 0.5 * equal) / len(vals), 1)


def component_ranks(raw: dict, side: str, pool: dict) -> tuple[dict, dict]:
    """Direction-signed 0-100 ranks per component + missing-component flags."""
    flip = side == "short"
    ranks, flags = {}, {}
    for key, pool_key, flips in (("sue", "sue", True), ("ear", "ear", True),
                                 ("rue", "rue", True), ("avol", "avol", False),
                                 ("anchor", "nearness", True)):
        r = pool_rank(raw.get(pool_key), pool.get(pool_key, []))
        if r is None:
            ranks[key] = NEUTRAL_RANK
            flags[f"{key}_missing"] = True
            continue
        ranks[key] = round(100.0 - r, 1) if (flip and flips) else r
    return ranks, flags


def composite(ranks: dict) -> float:
    return round(sum(WEIGHTS[k] * ranks.get(k, NEUTRAL_RANK) for k in WEIGHTS), 1)


# --- sheet assembly ----------------------------------------------------------

def build_signal_sheet(reporters: list[dict], px: dict, spy_close: pd.Series,
                       universe_by_ticker: dict, pool: dict, r_day: date,
                       revenues: dict | None = None,
                       next_earnings: dict | None = None) -> list[dict]:
    """One scored row per reporter for reaction day r_day.

    reporters: [{ticker, name, report_date, time, eps_actual, eps_consensus,
                 n_estimates, sue_source, mktcap}] — EPS fields already resolved
    px: {ticker: DataFrame[open,high,low,close,volume]}; revenues: {ticker:
    quarterly revenue Series}; next_earnings: {ticker: iso date}.
    """
    revenues = revenues or {}
    next_earnings = next_earnings or {}
    pre_day = prev_trading_day(r_day)
    rows = []
    for rep in reporters:
        t = rep["ticker"]
        frame = px.get(t)
        close = frame["close"] if frame is not None and not frame.empty else None
        i_pre = _asof_pos(close.index, pre_day) if close is not None else None
        close_pre = (float(close.iloc[i_pre])
                     if close is not None and i_pre is not None else None)

        raw = {
            "sue": sue_raw(rep.get("eps_actual"), rep.get("eps_consensus"), close_pre),
            "ear": ear(close, spy_close, r_day) if close is not None else None,
            "rue": rue(revenues.get(t)),
            "avol": (abnormal_volume(frame["volume"], r_day)
                     if frame is not None and not frame.empty else None),
            "nearness": nearness_52w(frame, pre_day),
        }
        raw = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in raw.items()}
        side = side_of(raw["sue"], raw["ear"])

        mom = momentum_12_1(close, pre_day)
        uni = universe_by_ticker.get(t, {})
        nxt = next_earnings.get(t)
        row = {
            "ticker": t, "name": rep.get("name") or uni.get("name", ""),
            "sector": uni.get("sector", ""),
            "report_date": rep["report_date"], "time": rep.get("time", ""),
            "reaction_day": r_day.isoformat(),
            "side": side,
            "eps_actual": rep.get("eps_actual"),
            "eps_consensus": rep.get("eps_consensus"),
            "n_estimates": rep.get("n_estimates"),
            "sue_source": rep.get("sue_source", ""),
            "raw": raw,
            "context": {
                "close_pre": round(close_pre, 4) if close_pre else None,
                "cap_tier": cap_tier(rep.get("mktcap")),
                "mktcap": rep.get("mktcap"),
                "adv_usd": adv_usd(frame, pre_day),
                "next_earn_date": nxt,
                "td_to_next_earn": (len(cal.trading_days(r_day, date.fromisoformat(nxt))) - 1
                                    if nxt else None),
            },
            "flags": {
                "gap_held": gap_held(frame, r_day, side) if side != "mixed" else None,
                "momentum_aligned": ((mom > 0) == (side == "long")
                                     if mom is not None and side != "mixed" else None),
            },
        }
        row["excluded_reason"] = gates(row)
        if row["excluded_reason"] in (None, "mixed", "weak_reaction"):
            ranks, miss = component_ranks(raw, side if side != "mixed" else "long", pool)
            row["ranks"] = ranks
            row["flags"].update(miss)
            row["composite"] = composite(ranks)
            row["rank_pool_n"] = min((len(pool.get(k, [])) for k in POOL_KEYS
                                      if pool.get(k)), default=0)
        else:
            row["ranks"], row["composite"], row["rank_pool_n"] = {}, None, 0
        rows.append(row)

    rows.sort(key=lambda r: (r["composite"] is None, -(r["composite"] or 0)))
    return rows


def update_pool(pool: dict, rows: list[dict], r_day: date) -> dict:
    """Fold a day's raw values into the trailing pool (kept per-day so old days
    can be dropped once outside POOL_WINDOW_TD)."""
    days = dict(pool.get("days", {}))
    days[r_day.isoformat()] = {
        k: [r["raw"].get(k) for r in rows
            if r["raw"].get(k) is not None] for k in POOL_KEYS}
    keep = sorted(days)[-POOL_WINDOW_TD:]
    days = {d: days[d] for d in keep}
    flat = {k: [v for d in keep for v in days[d].get(k, [])] for k in POOL_KEYS}
    return {"days": days, "window_start": keep[0] if keep else None, **flat}
