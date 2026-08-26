"""REGIMES theme monitors (spec sections 13-19): Dollar, Fiscal/Treasury, Yen/FX
intervention, AI investment, Crypto regulatory, Geopolitical/resource.

Per the user's explicit choice: themes backed by a real time series get a
QUANTITATIVE SIGNAL SCORE (0-100); themes that are genuinely policy/news-
driven get an EVIDENCE BOARD mined from Zenith's own scraper archive
(evidence.py), with NO invented probability — a probability with no series
behind it would be exactly the fabrication spec section 29 warns against.

IMPORTANT: quant themes here output a SIGNAL SCORE, not a "probability".
Only the top-level growth/inflation quadrant (transition.py) has the
historical timeline infrastructure to back a genuine empirical probability
(spec section 8's own choice: base rates, n always shown). Building that
same historical-timeline infrastructure per THEME is out of scope for this
phase — rather than fake a probability for themes, they get an honestly-
named "signal_score" (a composite of the theme's own z-scored inputs) plus
the narrative/evidence context. This naming distinction is deliberate and
carried into the UI.
"""

from __future__ import annotations

import pandas as pd

from ..cas.sources import fred as cas_fred
from ..cas.sources import prices as cas_prices
from ..config import REGIMES_ZSCORE_WINDOW_MONTHS, REGIMES_ZSCORE_MIN_MONTHS
from .macro import month_ends, _pit_series
from . import evidence, load as regimes_load

EXTRA_FRED_IDS = ("DGS30", "THREEFYTP10", "GFDEBTN", "DFII10")
EXTRA_LAG_DAYS = {"DGS30": 0, "THREEFYTP10": 0, "GFDEBTN": 45, "DFII10": 0}


def _fetch_extra(sleep: float = 0.2) -> dict[str, list]:
    raw, _status = cas_fred.get_series(list(EXTRA_FRED_IDS), max_age_hours=20.0, sleep=sleep,
                                       limit=None, cache_key="regimes_themes")
    return raw


def _monthly_z(raw_points: list[dict], lag_days: int, ends: pd.DatetimeIndex) -> pd.Series:
    pit = _pit_series(raw_points, lag_days)
    if pit.empty:
        return pd.Series(index=ends, dtype=float)
    monthly = pit.reindex(pit.index.union(ends)).ffill().reindex(ends)
    roll = monthly.rolling(REGIMES_ZSCORE_WINDOW_MONTHS, min_periods=REGIMES_ZSCORE_MIN_MONTHS)
    z = (monthly - roll.mean()) / roll.std().replace(0.0, pd.NA)
    return z


def _z_to_score(z: float | None) -> float | None:
    """z=0 -> 50, z=+/-4 -> 100/0. A linear, documented mapping — not a
    calibrated probability, see module docstring."""
    if z is None or pd.isna(z):
        return None
    return round(max(0.0, min(100.0, 50.0 + 12.5 * float(z))), 1)


# ------------------------------------------------------------------- dollar
def dollar_theme(z_df: pd.DataFrame, extra_z: dict[str, pd.Series]) -> dict:
    dollar_cols = [c for c in z_df.columns if c in ("DTWEXBGS", "DTWEXAFEGS", "DEXUSEU", "DEXJPUS")]
    composite = z_df[dollar_cols].mean(axis=1, skipna=True) if dollar_cols else pd.Series(dtype=float)
    z_latest = None if composite.empty or pd.isna(composite.iloc[-1]) else float(composite.iloc[-1])
    real_yield_z = extra_z.get("DFII10")
    ry_latest = None if real_yield_z is None or real_yield_z.dropna().empty else float(real_yield_z.dropna().iloc[-1])
    return {
        "signal_score": _z_to_score(z_latest),
        "composite_z": None if z_latest is None else round(z_latest, 3),
        "real_yield_z": None if ry_latest is None else round(ry_latest, 3),
        "context": (
            "The dollar's trade-weighted level and its trend versus real US yield "
            "differentials are the quantitative core of this theme. The Triffin dilemma "
            "argument — that the dollar's reserve-currency role structurally requires "
            "persistent US current-account deficits, which are themselves eventually "
            "destabilizing — is a real and long-standing academic thesis, but BIS and NBER "
            "research has also documented meaningful nuance and counter-evidence: reserve "
            "currency status has proven durable across many predicted 'dollar crisis' "
            "episodes, and simplified Triffin narratives often understate the dollar's "
            "network-effect advantages (deep capital markets, rule of law, no close "
            "substitute) and the difference between a SLOW multi-decade reserve-diversification "
            "trend and an acute depreciation event. This score reflects only the market-based "
            "trend, not a judgment on which narrative is correct."
        ),
    }


# ------------------------------------------------------------------- fiscal
def fiscal_theme(extra_z: dict[str, pd.Series]) -> dict:
    parts = {k: extra_z.get(k) for k in ("DGS30", "THREEFYTP10", "GFDEBTN")}
    latest = {k: (None if v is None or v.dropna().empty else round(float(v.dropna().iloc[-1]), 3))
             for k, v in parts.items()}
    vals = [v for v in latest.values() if v is not None]
    score_z = sum(vals) / len(vals) if vals else None
    return {
        "signal_score": _z_to_score(score_z),
        "components": latest,
        "context": (
            "This theme distinguishes two DIFFERENT things that both increase Treasury "
            "buyback activity: (1) LIQUIDITY OPERATIONS — the Treasury smoothing the maturity "
            "profile of outstanding debt, which is routine market-structure management and "
            "not obviously yield-suppressive; versus (2) actual FINANCIAL REPRESSION — "
            "policy that holds real yields below their market-clearing level to erode the "
            "real value of debt. The August 2026 increase in long-duration buybacks (to at "
            "least $4bn, following the 30y yield reaching a multi-year high) has been read "
            "both ways by market participants; this score does not adjudicate which "
            "interpretation is correct, it tracks the term premium and long-yield trend "
            "components that either interpretation would move."
        ),
    }


# --------------------------------------------------------------------- yen
INTERVENTION_TIERS = ((0.08, "Extreme"), (0.05, "High"), (0.025, "Moderate"), (0.0, "Low"))


def _tier_for(abs_move: float) -> str:
    for threshold, label in INTERVENTION_TIERS:
        if abs_move >= threshold:
            return label
    return INTERVENTION_TIERS[-1][1]


def _usdjpy_pct_move_3m(ends: pd.DatetimeIndex) -> float | None:
    """The intervention-risk tier is defined on a REAL percentage move (large,
    fast % moves have historically preceded MOF/BOJ intervention), not a
    z-score delta — a z-score delta of the same magnitude as this tier's
    thresholds would be a near-nonexistent move, not an extreme one. Pulled
    from the ALREADY-COMMITTED macro_raw cache (DEXJPUS is fetched by the
    main registry) so this needs no extra network call."""
    raw = regimes_load("macro_raw", {})
    points = raw.get("DEXJPUS", {}).get("points", [])
    if not points:
        return None
    pit = _pit_series(points, lag_days=0)
    monthly = pit.reindex(pit.index.union(ends)).ffill().reindex(ends).dropna()
    if len(monthly) < 4:
        return None
    return float(monthly.iloc[-1] / monthly.iloc[-4] - 1.0)


def yen_theme(z_df: pd.DataFrame, extra_z: dict[str, pd.Series]) -> dict:
    usdjpy_z = z_df.get("DEXJPUS")
    if usdjpy_z is None or usdjpy_z.dropna().empty:
        return {"intervention_risk": None, "note": "USD/JPY data unavailable this run."}
    z_latest = float(usdjpy_z.dropna().iloc[-1])
    pct_move_3m = _usdjpy_pct_move_3m(month_ends())
    tier = _tier_for(abs(pct_move_3m)) if pct_move_3m is not None else None
    real_yield_z = extra_z.get("DFII10")
    ry_latest = None if real_yield_z is None or real_yield_z.dropna().empty else float(real_yield_z.dropna().iloc[-1])
    return {
        "intervention_risk": tier,
        "usdjpy_z": round(z_latest, 3), "usdjpy_pct_move_3m": None if pct_move_3m is None else round(pct_move_3m, 4),
        "us_real_yield_z": None if ry_latest is None else round(ry_latest, 3),
        "note": ("Intervention-risk tier is a PROXY based on the magnitude/speed of the "
                "USD/JPY move alone (large, fast one-directional moves have historically "
                "preceded MOF/BOJ intervention) — it is not a probability of intervention. "
                "Japan-side real yield data is not freely available, so only the US side of "
                "the rate differential is shown; this is an acknowledged, one-sided gap, not "
                "a claim the JP side doesn't matter."),
    }


# --------------------------------------------------------------------- AI
def ai_theme() -> dict:
    px, _status = cas_prices.get_history(["SMH", "SPY"], period="1y")
    smh, spy = px.get("SMH"), px.get("SPY")
    rel_strength_3m = None
    if smh is not None and spy is not None and len(smh) > 63 and len(spy) > 63:
        smh_ret = float(smh["close"].iloc[-1] / smh["close"].iloc[-63] - 1.0)
        spy_ret = float(spy["close"].iloc[-1] / spy["close"].iloc[-63] - 1.0)
        rel_strength_3m = round(smh_ret - spy_ret, 4)
    ev = evidence.mine(["AI capex", "data center", "hyperscaler", "GPU demand", "semiconductor cycle",
                        "artificial intelligence spending", "AI bubble"], limit=10)
    return {
        "semis_relative_strength_3m": rel_strength_3m,
        "note": ("Market-based context only (semiconductor sector vs SPY relative strength) — "
                "this repo does not have a free source for hyperscaler capex, GPU shipment, or "
                "data-center construction data, so those go through the evidence board below "
                "rather than being fabricated as a quantitative reading."),
        "evidence": ev,
    }


# ----------------------------------------------------------------- crypto
def crypto_theme() -> dict:
    px, _status = cas_prices.get_history(["BTC-USD"], period="1y")
    btc = px.get("BTC-USD")
    trend_3m = None
    if btc is not None and len(btc) > 63:
        trend_3m = round(float(btc["close"].iloc[-1] / btc["close"].iloc[-63] - 1.0), 4)
    ev = evidence.mine(["CLARITY Act", "crypto regulation", "stablecoin legislation", "SEC crypto",
                        "digital asset market structure", "crypto ETF"], limit=10)
    return {
        "btc_trend_3m": trend_3m,
        "note": "Market reaction context (BTC price trend) only — regulatory developments "
                "themselves are tracked through the evidence board, not scored.",
        "evidence": ev,
    }


# ---------------------------------------------------------------- geopolitical
def geopolitical_theme() -> dict:
    ev = evidence.mine(["reshoring", "tariff", "critical minerals", "rare earth", "trade fragmentation",
                        "China decoupling", "supply chain diversification", "friend-shoring",
                        "Greenland", "resource nationalism"], limit=12)
    return {
        "evidence": ev,
        "framework_note": (
            "Every item below is EVIDENCE (an observable fact or a source's reported claim), not a "
            "conclusion. Where an item implies a strategic motive (for example, characterizations of "
            "US interest in Greenland as partly resource-driven), treat that as one HYPOTHESIS among "
            "several plausible explanations (which can include, and are not limited to, security "
            "policy, territorial claims context, or Arctic shipping-route strategy) — this board "
            "deliberately does not pick a winner among competing explanations for a geopolitical "
            "development; it surfaces the sourced claims so the reader can weigh them."
        ),
    }


def build(z_df: pd.DataFrame, sleep: float = 0.2) -> dict:
    ends = month_ends()
    extra_raw = _fetch_extra(sleep=sleep)
    extra_z = {k: _monthly_z(extra_raw.get(k, []), EXTRA_LAG_DAYS.get(k, 20), ends) for k in EXTRA_FRED_IDS}
    return {
        "dollar": dollar_theme(z_df, extra_z),
        "fiscal": fiscal_theme(extra_z),
        "yen": yen_theme(z_df, extra_z),
        "ai": ai_theme(),
        "crypto": crypto_theme(),
        "geopolitical": geopolitical_theme(),
    }
