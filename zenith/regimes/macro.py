"""REGIMES macro data layer: fetch -> point-in-time -> transform -> z-score.

Pipeline, in order (each step matters for why the next one is written the
way it is):

  1. `fetch_raw()` — pull every FRED series the registry needs, with a
     COMMITTED warm-start cache (data/regimes/macro_raw.json) so a fresh CI
     checkout does not re-download ~45 series' full history every night —
     cas/sources/fred.py's own cache is gitignored and would not survive
     across Action runs. Series are grouped by native frequency (daily/
     weekly vs monthly/quarterly) with different TTLs, and fetched through
     the THROTTLED `cas.sources.fred.get_series(sleep=...)` added for this
     feature — a tight loop over ~90 FRED requests got this IP rate-limited
     for over a minute during exploration; a nightly Action doing that would
     silently degrade to partial data every run.

  2. `point_in_time_monthly()` — resample every series onto a MONTHLY grid
     using ONLY data that had actually been published by each month's
     end (each series shifted by its own `series.lag_days` first, then
     `pandas.Series.asof()` — which only ever looks backward — picks the
     latest visible value). This is what makes historical reconstruction not
     look-ahead-biased on TIMING (see regimes/__init__.py's docstring for
     why this does not also fix REVISION look-ahead, which is vintage.py's
     separate, optional job).

  3. `transform()` — turns a point-in-time level series into the stationary
     quantity the classifier actually wants (YoY %, month-over-month diff,
     3-month diff), applied AFTER step 2 so the transform itself never
     straddles the lag boundary.

  4. `zscore_panel()` — z-scores every transformed series against its OWN
     trailing history (config.REGIMES_ZSCORE_WINDOW_MONTHS, min
     config.REGIMES_ZSCORE_MIN_MONTHS) and applies `series.direction` so a
     positive value always means "toward this dimension's positive pole" —
     see series.py's SeriesSpec docstring for exactly what that pole is per
     dimension. The rolling window uses each series' FULL available history
     (which for deep-history series reaches back to the 1940s-1980s) even
     though the reported panel is later sliced to config.REGIMES_HISTORY_START
     — a longer estimation window makes the mean/std estimate more stable,
     it just doesn't get REPORTED for pre-1990 months (see config.py's
     inline rationale for the 1990 cutoff).

Two derived series (not directly on FRED) are computed here, after step 2,
from their own point-in-time monthly inputs — see series.DERIVED_INPUTS:

  * REAL_FFR = DFF (level) - PCEPILFE's OWN yoy transform, in percentage
    points. Needs PCEPILFE's transform to already be computed, so derived
    series are built in a second pass after all direct series are
    transformed.
  * NET_LIQ  = WALCL - RRPONTSYD - WTREGEN, unit-aligned to $ billions
    (WALCL/WTREGEN publish in $ millions, RRPONTSYD in $ billions — see
    series.EXTRA_META) — the standard "Fed net liquidity" construction.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import (REGIMES_ZSCORE_WINDOW_MONTHS, REGIMES_ZSCORE_MIN_MONTHS,
                      REGIMES_HISTORY_START)
from ..cas.sources import fred as cas_fred
from ..cas.sources import prices as cas_prices
from . import save as regimes_save, load as regimes_load
from .series import REGISTRY, BY_ID, ALL_FRED_IDS, DERIVED_INPUTS, freq_of, lag_of


# ------------------------------------------------------------------- fetch --
def fetch_raw(force: bool = False, sleep: float = 0.2) -> tuple[dict[str, list], dict]:
    """Every FRED series the registry needs, id -> [{date,value}, ...].
    Committed-cache warm start; only genuinely stale series hit the network."""
    committed: dict = regimes_load("macro_raw", {})
    today = date.today()

    def _stale(sid: str) -> bool:
        if force or sid not in committed:
            return True
        try:
            fetched_at = date.fromisoformat(committed[sid]["fetched_at"][:10])
        except Exception:
            return True
        ttl_days = 1 if freq_of(sid) in ("D", "W") else 5
        return (today - fetched_at).days >= ttl_days

    todo = [sid for sid in ALL_FRED_IDS if _stale(sid)]
    fetch_status = {"ok": True, "n": 0}
    if todo:
        fetched, fetch_status = cas_fred.get_series(
            todo, max_age_hours=0.0, sleep=sleep, limit=None, cache_key="regimes_scratch")
        for sid, pts in fetched.items():
            committed[sid] = {"points": pts, "fetched_at": today.isoformat()}
        regimes_save("macro_raw", committed, indent=None)

    raw = {sid: committed[sid]["points"] for sid in ALL_FRED_IDS if sid in committed}
    return raw, {"requested": len(ALL_FRED_IDS), "fetched": len(todo),
                "reused": len(ALL_FRED_IDS) - len(todo), "coverage": len(raw),
                "fetch_ok": fetch_status.get("ok", False)}


# -------------------------------------------------------- point-in-time ----
def _pit_series(points: list[dict], lag_days: int) -> pd.Series:
    """points -> a pandas Series indexed by (release date + lag), so it only
    becomes visible to `.asof()` from the date it was actually published."""
    if not points:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([p["date"] for p in points]) + pd.Timedelta(days=lag_days)
    s = pd.Series([p["value"] for p in points], index=idx, dtype=float)
    return s.sort_index()[~s.index.duplicated(keep="last")]


def month_ends(start: str = REGIMES_HISTORY_START, end: date | None = None) -> pd.DatetimeIndex:
    end = end or date.today()
    return pd.date_range(start=start, end=pd.Timestamp(end) + pd.Timedelta(days=31), freq="ME")


def point_in_time_monthly(raw: dict[str, list], ends: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Every raw FRED id -> a point-in-time monthly Series (asof-sampled at
    each month-end in `ends`, using each id's own publication lag). Includes
    the extra inputs (WALCL/RRPONTSYD/WTREGEN) alongside registered series."""
    out: dict[str, pd.Series] = {}
    for sid, points in raw.items():
        pit = _pit_series(points, lag_of(sid))
        if pit.empty:
            out[sid] = pd.Series(index=ends, dtype=float)
            continue
        out[sid] = pit.reindex(pit.index.union(ends)).ffill().reindex(ends)
    return out


# ----------------------------------------------------------------- transform
def _transform(monthly: pd.Series, kind: str) -> pd.Series:
    if kind == "level":
        return monthly
    if kind == "yoy":
        return monthly.pct_change(12, fill_method=None) * 100.0
    if kind == "mom_diff":
        return monthly.diff(1)
    if kind == "chg3m":
        return monthly.diff(3)
    raise ValueError(f"unknown transform {kind!r}")


def build_transformed_panel(raw: dict[str, list], ends: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Every registry series id (direct AND derived) -> its transformed,
    point-in-time monthly Series, ready for z-scoring."""
    pit = point_in_time_monthly(raw, ends)
    out: dict[str, pd.Series] = {}

    for spec in REGISTRY:
        if spec.fred_id is None:
            continue   # derived series, second pass below
        out[spec.id] = _transform(pit.get(spec.fred_id, pd.Series(index=ends, dtype=float)), spec.transform)

    # REAL_FFR = DFF level - PCEPILFE's own yoy (already computed above), pp.
    dff = pit.get("DFF", pd.Series(index=ends, dtype=float))
    pcepilfe_yoy = out.get("PCEPILFE", pd.Series(index=ends, dtype=float))
    out["REAL_FFR"] = dff - pcepilfe_yoy

    # NET_LIQ = WALCL - RRPONTSYD - WTREGEN, unit-aligned to $ billions
    # (WALCL/WTREGEN are $ millions -> /1000; RRPONTSYD is already $ billions).
    walcl = pit.get("WALCL", pd.Series(index=ends, dtype=float)) / 1000.0
    rrp = pit.get("RRPONTSYD", pd.Series(index=ends, dtype=float))
    tga = pit.get("WTREGEN", pd.Series(index=ends, dtype=float)) / 1000.0
    net_liq_level = walcl - rrp - tga
    out["NET_LIQ"] = _transform(net_liq_level, "chg3m")

    return out


# ------------------------------------------------------------------ z-score
def zscore_panel(transformed: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direction-adjusted rolling z-score for every series. Returns
    (z_df, raw_transformed_df) both indexed by month-end, columns = series id.
    `raw_transformed_df` (pre-z, pre-direction) is kept for the "latest value"
    explainability display — z-scores are for combining, not for reading."""
    zcols, rawcols = {}, {}
    for sid, s in transformed.items():
        spec = BY_ID.get(sid)
        direction = spec.direction if spec is not None else 1
        roll = s.rolling(REGIMES_ZSCORE_WINDOW_MONTHS, min_periods=REGIMES_ZSCORE_MIN_MONTHS)
        mu, sd = roll.mean(), roll.std()
        z = (s - mu) / sd.replace(0.0, np.nan)
        zcols[sid] = direction * z
        rawcols[sid] = s
    z_df = pd.DataFrame(zcols)
    raw_df = pd.DataFrame(rawcols)
    return z_df, raw_df


# --------------------------------------------------------- TLT realized vol
def tlt_realized_vol(ends: pd.DatetimeIndex) -> pd.Series:
    """21d realized volatility on TLT, annualized, resampled to month-end —
    a MOVE-index PROXY (VXTYN, the real Treasury-vol index, was discontinued
    2020-05; see series.py's module docstring). Labelled as a proxy wherever
    it is shown, never presented as the real index. Best-effort: returns an
    all-NaN Series (never raises) if the price pull fails."""
    try:
        px, status = cas_prices.get_history(["TLT"], period="max")
        df = px.get("TLT")
        if df is None or df.empty:
            return pd.Series(index=ends, dtype=float)
        ret = df["close"].pct_change().dropna()
        rvol = ret.rolling(21).std() * np.sqrt(252) * 100.0
        rvol.index = pd.to_datetime(rvol.index)
        return rvol.reindex(rvol.index.union(ends)).ffill().reindex(ends)
    except Exception:
        return pd.Series(index=ends, dtype=float)


def build_panel(force: bool = False, sleep: float = 0.2
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The single entry point: fetch -> point-in-time -> transform -> z-score,
    plus the TLT-realized-vol proxy folded in as the "TLT_RVOL" volatility
    series. Returns (z_df, raw_transformed_df, fetch_status)."""
    raw, fetch_status = fetch_raw(force=force, sleep=sleep)
    ends = month_ends()
    transformed = build_transformed_panel(raw, ends)
    transformed["TLT_RVOL"] = tlt_realized_vol(ends)
    z_df, raw_df = zscore_panel(transformed)
    return z_df, raw_df, fetch_status
