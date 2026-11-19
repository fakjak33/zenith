"""Synthetic-chain options backtester (MODELED — no historical chains are free).

Each day an implied-vol surface is rebuilt from free Cboe indices:

- **ATM term structure**: VIX9D / VIX / VIX3M / VIX6M interpolated in total
  variance (flat beyond the ends; VIX alone where the others don't exist yet).
  Non-SPX underlyings use their own Cboe vol index when one exists (VXN, RVX,
  GVZ, OVX, VXEEM), shaped by the VIX term structure; otherwise VIX scaled by
  the ratio of the underlying's realised vol to the S&P 500's.
- **Smile**: σ(K) = σ_atm · exp(−κ·z), z = ln(K/S)/(σ_atm√T) and
  κ = 0.06 · (SKEW − 100)/10, floored/capped — puts richer than calls,
  steeper when Cboe SKEW is high.

Options are priced with THETA's Black-Scholes. Every open/close pays a
half-spread. Results are always stamped MODELED and calibrated against Cboe
PUT and BXM (short side) and PPUT (long side).

Strategies are built from up to a handful of ``Leg``s. Strikes can be picked by
delta, % of spot, ATM, a wing width from another leg, or a target premium; each
leg can have its own expiry (calendars / diagonals). Sizing is fully
collateralised (Cboe style), a premium budget, a max-loss budget or a notional
share. Entries can be gated by VIX level, z-score, change, term structure,
VIX9D/VIX, percentile, a trend filter, weekday and the trade's own credit,
delta and credit/width. ``simulate_0dte`` runs same-day expiries from daily
SPX open/high/low/close and the Cboe VIX1D index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...theta.engine.pricing import greeks, price, strike_for_delta

TENORS = np.array([9, 30, 91, 182]) / 365.0
TERM_COLS = ["VIX9D", "VIX", "VIX3M", "VIX6M"]
# Fitted once on Cboe PUT (2007→) so modeled ATM put-write CAGR matches the index; BXM is the
# out-of-sample check (see tests/test_backtest_multiasset.py and the calibration panel).
ATM_RATIO = 0.87
# One session as a year fraction for 0DTE pricing. Empirically the VIX1D-implied open→close move
# matches SPX's realised one on a one-TRADING-day horizon (1/252); the 6.5-hour calendar-minute
# horizon (390/525,600) understates the realised move by ~2x (see implied_vs_realised_0dte).
SESSION_T = 1 / 252


# ------------------------------------------------------------------ legs ----
@dataclass(frozen=True)
class Leg:
    kind: str            # "put" | "call"
    side: int            # -1 short, +1 long
    rule: str            # "delta" | "pct" | "atm" | "wing" | "premium"
    value: float = 0.0   # 0.30 for 30-delta; 0.95 for 95% of spot; wing width as share of spot; premium as share of spot
    qty: float = 1.0
    ref: int = -1        # "wing": index of the leg this one is measured from
    dte: int = 0         # this leg's own days to expiry (0 = the strategy's DTE) — calendars / diagonals


SIZING = {
    "collateral": "fully collateralised (Cboe style): one unit of underlying notional per contract-unit",
    "premium": "premium budget: spend this % of equity on each trade's net debit",
    "risk": "max-loss budget: size so the worst case at expiry loses this % of equity",
    "notional": "notional: control this % of equity in underlying notional per trade",
}


@dataclass
class OptionStrategy:
    name: str
    legs: list = field(default_factory=list)
    stock: float = 0.0           # units of underlying held (1 = covered call / collar)
    dte: int = 30                # calendar days to target expiry
    roll: str = "monthly"        # "monthly" = 3rd-Friday cycle (Cboe style) | "dte" = reopen at target DTE | "0dte"
    take_profit: float = 0.0     # credit: share of the credit captured · debit: gain as a share of the debit (0 = off)
    stop_mult: float = 0.0       # credit: loss as a multiple of the credit · debit: loss as a share of the debit (0 = off)
    min_vix: float = 0.0         # legacy: only open when VIX ≥ this (same as filters["vix_lo"])
    spread_pct: float = 0.03     # half-spread as a share of option price, per leg, each way
    min_spread_bps: float = 0.5  # floor on the half-spread, in bps of spot
    atm_ratio: float = ATM_RATIO  # VIX is a variance-swap level, above ATM implied vol; ATM ≈ ratio × VIX
    sizing: str = "collateral"   # see SIZING
    size_pct: float = 0.0        # budget for premium / risk / notional sizing (share of equity)
    exit_dte: int = 0            # close when the front leg has this many days left (0 = hold to expiry)
    max_hold: int = 0            # close after this many calendar days (0 = no limit)
    delta_exit: float = 0.0      # close when |net delta| per unit exceeds this (0 = off)
    filters: dict = field(default_factory=dict)   # entry gates, see FILTER_KEYS


FILTER_KEYS = {
    "vix_lo": "VIX at or above", "vix_hi": "VIX at or below",
    "z_lo": "VIX z-score at or above", "z_hi": "VIX z-score at or below", "z_len": "z-score lookback (days)",
    "chg_lo": "VIX N-day change ≥ (σ)", "chg_hi": "VIX N-day change ≤ (σ)", "chg_len": "change length (days)",
    "ts_lo": "VIX / VIX3M ≥", "ts_hi": "VIX / VIX3M ≤",
    "r9_lo": "VIX9D / VIX ≥", "r9_hi": "VIX9D / VIX ≤",
    "pct_lo": "VIX 1y percentile ≥", "pct_hi": "VIX 1y percentile ≤",
    "sma_len": "trend filter length", "sma_side": "price vs its average (above / below)",
    "dow": "weekdays allowed (0 = Mon)",
    "net_lo": "net credit ≥ (% of spot; debit is negative)", "net_hi": "net credit ≤ (% of spot)",
    "delta_lo": "net delta ≥", "delta_hi": "net delta ≤", "cw_min": "credit / width ≥",
}


def _L(kind, side, rule, value=0.0, qty=1.0, ref=-1, dte=0):
    return Leg(kind, side, rule, value, qty, ref, dte)


def _S(name, legs, **kw):
    return OptionStrategy(name, legs, **kw)


_DEBIT = dict(sizing="premium", size_pct=0.02)

# name → (strategy, tags). Tags drive the strategy browser: flow, vol, bias, risk.
CATALOG: dict[str, tuple[OptionStrategy, dict]] = {
    # ---------------------------------------------------------------- short vol / credit
    "Short put (PUT-style)": (_S("Short put", [_L("put", -1, "atm")]),
                              dict(flow="credit", vol="short", bias="bullish", risk="undefined")),
    "Covered call (BXM-style)": (_S("Covered call", [_L("call", -1, "atm")], stock=1.0),
                                 dict(flow="credit", vol="short", bias="bullish", risk="stock")),
    "OTM put write (30Δ)": (_S("30Δ put write", [_L("put", -1, "delta", 0.30)]),
                            dict(flow="credit", vol="short", bias="bullish", risk="undefined")),
    "Short strangle (16Δ)": (_S("Short strangle", [_L("put", -1, "delta", 0.16), _L("call", -1, "delta", 0.16)]),
                             dict(flow="credit", vol="short", bias="neutral", risk="undefined")),
    "Short straddle (ATM)": (_S("Short straddle", [_L("put", -1, "atm"), _L("call", -1, "atm")]),
                             dict(flow="credit", vol="short", bias="neutral", risk="undefined")),
    "Iron condor (16Δ / 5Δ)": (_S("Iron condor", [_L("put", -1, "delta", 0.16), _L("put", 1, "delta", 0.05),
                                                  _L("call", -1, "delta", 0.16), _L("call", 1, "delta", 0.05)]),
                               dict(flow="credit", vol="short", bias="neutral", risk="defined")),
    "Iron fly (ATM, 5% wings)": (_S("Iron fly", [_L("put", -1, "atm"), _L("put", 1, "wing", 0.05, ref=0),
                                                 _L("call", -1, "atm"), _L("call", 1, "wing", 0.05, ref=2)]),
                                 dict(flow="credit", vol="short", bias="neutral", risk="defined")),
    "Put spread (30Δ / 10Δ)": (_S("Put credit spread", [_L("put", -1, "delta", 0.30), _L("put", 1, "delta", 0.10)]),
                               dict(flow="credit", vol="short", bias="bullish", risk="defined")),
    "Call credit spread (30Δ / 10Δ)": (_S("Call credit spread", [_L("call", -1, "delta", 0.30),
                                                                  _L("call", 1, "delta", 0.10)]),
                                       dict(flow="credit", vol="short", bias="bearish", risk="defined")),
    "Jade lizard (20Δ put, 20Δ/5% call spread)": (
        _S("Jade lizard", [_L("put", -1, "delta", 0.20), _L("call", -1, "delta", 0.20),
                           _L("call", 1, "wing", 0.03, ref=1)]),
        dict(flow="credit", vol="short", bias="bullish", risk="undefined")),
    "Put ratio spread (1×30Δ / 2×15Δ)": (_S("Put ratio spread", [_L("put", 1, "delta", 0.30),
                                                                   _L("put", -1, "delta", 0.15, qty=2.0)]),
                                          dict(flow="credit", vol="short", bias="bullish", risk="undefined")),
    "Collar (95% put / 105% call)": (_S("Collar", [_L("put", 1, "pct", 0.95), _L("call", -1, "pct", 1.05)],
                                        stock=1.0),
                                     dict(flow="credit", vol="neutral", bias="bullish", risk="stock")),
    # ----------------------------------------------------------------- long vol / debit
    "Long put hedge (90%)": (_S("Protective put", [_L("put", 1, "pct", 0.90)], stock=1.0),
                             dict(flow="debit", vol="long", bias="bullish", risk="stock")),
    "Protective put (95%, PPUT-style)": (_S("PPUT-style", [_L("put", 1, "pct", 0.95)], stock=1.0),
                                         dict(flow="debit", vol="long", bias="bullish", risk="stock")),
    "Long call (ATM)": (_S("Long call", [_L("call", 1, "atm")], **_DEBIT),
                        dict(flow="debit", vol="long", bias="bullish", risk="defined")),
    "Long call (30Δ)": (_S("Long 30Δ call", [_L("call", 1, "delta", 0.30)], **_DEBIT),
                        dict(flow="debit", vol="long", bias="bullish", risk="defined")),
    "Long put (ATM)": (_S("Long put", [_L("put", 1, "atm")], **_DEBIT),
                       dict(flow="debit", vol="long", bias="bearish", risk="defined")),
    "Long put (25Δ)": (_S("Long 25Δ put", [_L("put", 1, "delta", 0.25)], **_DEBIT),
                       dict(flow="debit", vol="long", bias="bearish", risk="defined")),
    "Long straddle (ATM)": (_S("Long straddle", [_L("put", 1, "atm"), _L("call", 1, "atm")], **_DEBIT),
                            dict(flow="debit", vol="long", bias="neutral", risk="defined")),
    "Long strangle (25Δ)": (_S("Long strangle", [_L("put", 1, "delta", 0.25), _L("call", 1, "delta", 0.25)],
                               **_DEBIT),
                            dict(flow="debit", vol="long", bias="neutral", risk="defined")),
    "Call debit spread (50Δ / 25Δ)": (_S("Call debit spread", [_L("call", 1, "delta", 0.50),
                                                                _L("call", -1, "delta", 0.25)], **_DEBIT),
                                      dict(flow="debit", vol="long", bias="bullish", risk="defined")),
    "Put debit spread (50Δ / 25Δ)": (_S("Put debit spread", [_L("put", 1, "delta", 0.50),
                                                              _L("put", -1, "delta", 0.25)], **_DEBIT),
                                     dict(flow="debit", vol="long", bias="bearish", risk="defined")),
    "Long call butterfly (ATM ± 5%)": (_S("Long butterfly", [_L("call", 1, "pct", 0.95), _L("call", -1, "atm", qty=2.0),
                                                             _L("call", 1, "pct", 1.05)], **_DEBIT),
                                       dict(flow="debit", vol="short", bias="neutral", risk="defined")),
    "Put backspread (1×40Δ short / 2×20Δ long)": (
        _S("Put backspread", [_L("put", -1, "delta", 0.40), _L("put", 1, "delta", 0.20, qty=2.0)],
           sizing="risk", size_pct=0.02),
        dict(flow="debit", vol="long", bias="bearish", risk="defined")),
    "Call backspread (1×40Δ short / 2×20Δ long)": (
        _S("Call backspread", [_L("call", -1, "delta", 0.40), _L("call", 1, "delta", 0.20, qty=2.0)],
           sizing="risk", size_pct=0.02),
        dict(flow="debit", vol="long", bias="bullish", risk="defined")),
    "Tail hedge (5Δ puts, monthly)": (_S("Tail hedge", [_L("put", 1, "delta", 0.05)], sizing="premium",
                                         size_pct=0.005),
                                      dict(flow="debit", vol="long", bias="bearish", risk="defined")),
    "Long calendar (ATM, 30d / 60d)": (_S("Long calendar", [_L("call", -1, "atm"), _L("call", 1, "atm", dte=60)],
                                          roll="dte", **_DEBIT),
                                       dict(flow="debit", vol="long", bias="neutral", risk="defined")),
    "Long diagonal (30d 30Δ / 60d 50Δ calls)": (
        _S("Long diagonal", [_L("call", -1, "delta", 0.30), _L("call", 1, "delta", 0.50, dte=60)], roll="dte",
           **_DEBIT),
        dict(flow="debit", vol="long", bias="bullish", risk="defined")),
    "Risk reversal (short 25Δ put / long 25Δ call)": (
        _S("Risk reversal", [_L("put", -1, "delta", 0.25), _L("call", 1, "delta", 0.25)], sizing="notional",
           size_pct=1.0),
        dict(flow="credit", vol="neutral", bias="bullish", risk="undefined")),
    # ----------------------------------------------------------------------- 0DTE examples
    "0DTE iron condor (10Δ / 5Δ)": (_S("0DTE iron condor", [_L("put", -1, "delta", 0.10), _L("put", 1, "delta", 0.05),
                                                             _L("call", -1, "delta", 0.10), _L("call", 1, "delta", 0.05)],
                                       roll="0dte", dte=0, sizing="risk", size_pct=0.02),
                                    dict(flow="credit", vol="short", bias="neutral", risk="defined")),
    "0DTE put credit spread (15Δ / 5Δ)": (_S("0DTE put spread", [_L("put", -1, "delta", 0.15),
                                                                   _L("put", 1, "delta", 0.05)],
                                             roll="0dte", dte=0, sizing="risk", size_pct=0.02),
                                          dict(flow="credit", vol="short", bias="bullish", risk="defined")),
    "0DTE long straddle (ATM)": (_S("0DTE long straddle", [_L("put", 1, "atm"), _L("call", 1, "atm")],
                                    roll="0dte", dte=0, sizing="premium", size_pct=0.01),
                                 dict(flow="debit", vol="long", bias="neutral", risk="defined")),
}
TEMPLATES: dict[str, OptionStrategy] = {k: v[0] for k, v in CATALOG.items()}
TAGS: dict[str, dict] = {k: v[1] for k, v in CATALOG.items()}


# --------------------------------------------------------------- surface ----
def atm_vol(term_row: np.ndarray, T: float) -> float:
    """ATM vol at maturity T from one day's [VIX9D, VIX, VIX3M, VIX6M] (%, NaN allowed)."""
    v = np.asarray(term_row, float) / 100.0
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    tv = v[ok] ** 2 * TENORS[ok]
    tt = TENORS[ok]
    if T <= tt[0]:
        return float(v[ok][0])
    if T >= tt[-1]:
        return float(v[ok][-1])
    w = np.interp(T, tt, tv)
    return float(np.sqrt(max(w, 1e-8) / T))


def smile(atm: float, S: float, K, T: float, skew_idx: float) -> np.ndarray:
    kappa = float(np.clip(0.06 * (skew_idx - 100.0) / 10.0, 0.02, 0.35)) if np.isfinite(skew_idx) else 0.15
    z = np.log(np.asarray(K, float) / S) / (atm * np.sqrt(max(T, 1e-6)))
    mult = np.clip(np.exp(-kappa * z), 0.6, 3.0)
    return atm * mult


def third_fridays(start, end) -> pd.DatetimeIndex:
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    return pd.DatetimeIndex([m + pd.offsets.WeekOfMonth(week=2, weekday=4) for m in months])


def spx_daily_expiry(d: pd.Timestamp) -> bool:
    """Did SPX list an option expiring on this date? Fridays (weeklys) throughout; Wednesday from
    Feb 2016, Monday from Aug 2016, Tuesday/Thursday from May 2022 (every weekday since)."""
    wd = d.weekday()
    if wd == 4:
        return True
    if d >= pd.Timestamp("2022-05-11"):
        return wd < 5
    if d >= pd.Timestamp("2016-08-15"):
        return wd in (0, 2)
    if d >= pd.Timestamp("2016-02-23"):
        return wd == 2
    return False


# ------------------------------------------------------------- results ----
@dataclass
class OptionResult:
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.DataFrame         # daily net delta / vega per unit notional
    costs: pd.Series
    label: str = "MODELED"
    gate: pd.Series | None = None  # True where the entry filters allowed a new position


# ------------------------------------------------------------- strikes ----
def _strike(leg: Leg, S: float, T: float, sig_atm: float, r: float, skew: float, prior: list | None = None) -> float:
    if leg.rule == "atm":
        return S
    if leg.rule == "pct":
        return S * leg.value
    if leg.rule == "wing":
        base = prior[leg.ref] if prior and 0 <= leg.ref < len(prior) else S
        return base - leg.value * S if leg.kind == "put" else base + leg.value * S
    if leg.rule == "premium":
        # the strike (searched by delta) whose modeled price is closest to value × spot
        best, err = S, np.inf
        for d in np.linspace(0.01, 0.55, 55):
            K = strike_for_delta(S, T, sig_atm, r, d, leg.kind == "call")
            sig = float(smile(sig_atm, S, K, T, skew))
            p = float(price(leg.kind == "call", S, K, T, sig, r))
            e = abs(p - leg.value * S)
            if e < err:
                best, err = K, e
        return best
    # delta strike: iterate once so the strike uses its own smile vol
    K = strike_for_delta(S, T, sig_atm, r, leg.value, leg.kind == "call")
    sig = float(smile(sig_atm, S, K, T, skew))
    return strike_for_delta(S, T, sig, r, leg.value, leg.kind == "call")


def dividend_yield(price_idx: pd.Series, total_return: pd.Series) -> pd.Series:
    """Trailing-year dividend yield from a price index and its total-return twin (lagged a day)."""
    df = pd.concat([price_idx.rename("p"), total_return.rename("t")], axis=1).dropna()
    if len(df) < 260:
        return pd.Series(0.0, index=price_idx.index)
    q = (df["t"] / df["t"].shift(252)) - (df["p"] / df["p"].shift(252))
    return q.shift(1).clip(0.0, 0.08).reindex(price_idx.index).ffill().fillna(0.0)


def expiry_payoff(legs: list[dict], S_grid: np.ndarray, stock: float = 0.0, S0: float = 0.0) -> np.ndarray:
    """Value per unit at the front expiry across ``S_grid`` (back legs at intrinsic — an approximation)."""
    v = np.zeros_like(S_grid, dtype=float)
    for L in legs:
        lg = L["leg"]
        intr = np.maximum(S_grid - L["K"], 0) if lg.kind == "call" else np.maximum(L["K"] - S_grid, 0)
        v += lg.side * lg.qty * intr
    if stock:
        v += stock * (S_grid - S0)
    return v


def max_loss_per_unit(legs: list[dict], book: float, S: float, stock: float = 0.0) -> float:
    """Worst P&L per unit at expiry over 0…3×spot (undefined-risk shorts are capped at that range)."""
    grid = np.linspace(1e-6, 3.0 * S, 601)
    pnl = expiry_payoff(legs, grid, stock, S) - book
    return float(max(-pnl.min(), 1e-6))


def _width(legs: list[dict]) -> float:
    """Widest same-kind strike gap (spread width) — 0 when there is no spread."""
    w = 0.0
    for kind in ("put", "call"):
        ks = [L["K"] for L in legs if L["leg"].kind == kind]
        if len(ks) > 1:
            w = max(w, max(ks) - min(ks))
    return w


# --------------------------------------------------------------- gates ----
def entry_gate(strat: OptionStrategy, idx: pd.DatetimeIndex, S: pd.Series, term: pd.DataFrame) -> pd.Series:
    """Day-level entry filters (everything except the trade-level credit / delta / width checks).
    Every series is known at the close of day t, when the position is opened."""
    f = dict(strat.filters or {})
    if strat.min_vix and "vix_lo" not in f:
        f["vix_lo"] = strat.min_vix
    g = pd.Series(True, index=idx)
    if not f:
        return g
    vix = term["VIX"].reindex(idx).ffill() if "VIX" in term else pd.Series(np.nan, index=idx)

    def band(x: pd.Series, lo, hi):
        ok = pd.Series(True, index=idx)
        if lo is not None:
            ok &= x >= lo
        if hi is not None:
            ok &= x <= hi
        return ok.fillna(False) if (lo is not None or hi is not None) else ok

    g &= band(vix, f.get("vix_lo"), f.get("vix_hi"))
    if f.get("z_lo") is not None or f.get("z_hi") is not None:
        n = int(f.get("z_len", 63))
        z = (vix - vix.rolling(n, min_periods=max(10, n // 2)).mean()) / vix.rolling(n, min_periods=max(10, n // 2)).std()
        g &= band(z, f.get("z_lo"), f.get("z_hi"))
    if f.get("chg_lo") is not None or f.get("chg_hi") is not None:
        k = int(f.get("chg_len", 5))
        sd = vix.diff().rolling(252, min_periods=60).std() * np.sqrt(k)
        g &= band((vix - vix.shift(k)) / sd, f.get("chg_lo"), f.get("chg_hi"))
    if (f.get("ts_lo") is not None or f.get("ts_hi") is not None) and "VIX3M" in term:
        g &= band(vix / term["VIX3M"].reindex(idx).ffill(), f.get("ts_lo"), f.get("ts_hi"))
    if (f.get("r9_lo") is not None or f.get("r9_hi") is not None) and "VIX9D" in term:
        g &= band(term["VIX9D"].reindex(idx).ffill() / vix, f.get("r9_lo"), f.get("r9_hi"))
    if f.get("pct_lo") is not None or f.get("pct_hi") is not None:
        pct = vix.rolling(252, min_periods=60).rank(pct=True) * 100
        g &= band(pct, f.get("pct_lo"), f.get("pct_hi"))
    if f.get("sma_len") and f.get("sma_side") in ("above", "below"):
        sma = S.reindex(idx).rolling(int(f["sma_len"]), min_periods=int(f["sma_len"])).mean()
        side = S.reindex(idx) > sma if f["sma_side"] == "above" else S.reindex(idx) < sma
        g &= side.fillna(False)
    if f.get("dow"):
        g &= pd.Series(idx.weekday.isin(list(f["dow"])), index=idx)
    return g.astype(bool)


def _trade_ok(f: dict, net_pct: float, delta: float, cw: float | None) -> bool:
    if f.get("net_lo") is not None and net_pct < f["net_lo"]:
        return False
    if f.get("net_hi") is not None and net_pct > f["net_hi"]:
        return False
    if f.get("delta_lo") is not None and delta < f["delta_lo"]:
        return False
    if f.get("delta_hi") is not None and delta > f["delta_hi"]:
        return False
    if f.get("cw_min") is not None and cw is not None and cw < f["cw_min"]:
        return False
    return True


def _units(strat: OptionStrategy, V: float, s: float, legs: list[dict], book: float) -> float:
    """Underlying units per contract-unit for a new position under the strategy's sizing."""
    mode, b = strat.sizing, strat.size_pct
    if mode == "premium" and book > 1e-9:
        return b * V / book
    if mode in ("risk", "premium"):
        return b * V / max_loss_per_unit(legs, book, s, strat.stock) if b > 0 else V / s
    if mode == "notional" and b > 0:
        return b * V / s
    return V / s                                           # collateral


# ------------------------------------------------------------- simulator ----
def simulate(strat: OptionStrategy, S: pd.Series, term: pd.DataFrame, skew: pd.Series, rf: pd.Series,
             stock_tr: pd.Series | None = None, vol_scale: pd.Series | None = None,
             q: pd.Series | None = None) -> OptionResult:
    """Daily mark-to-model simulation. ``S`` is the (price) underlying the options settle on;
    ``stock_tr`` its total-return series for any stock leg (defaults to ``S``)."""
    idx = S.dropna().index
    S = S.reindex(idx)
    term = term.reindex(idx).ffill()
    skew = skew.reindex(idx).ffill()
    rf = rf.reindex(idx).ffill().fillna(0.0)                 # annualised decimal
    scale = (vol_scale.reindex(idx).ffill().fillna(1.0) if vol_scale is not None
             else pd.Series(1.0, index=idx))
    q = (q.reindex(idx).ffill().fillna(0.0) if q is not None else pd.Series(0.0, index=idx))
    stock = (stock_tr if stock_tr is not None else S).reindex(idx).ffill()
    stock_ret = stock.pct_change().fillna(0.0)
    expiries = sorted(set(third_fridays(idx[0], idx[-1] + pd.Timedelta(days=400))))
    gate = entry_gate(strat, idx, S, term)
    gate_np = gate.to_numpy()
    filt = dict(strat.filters or {})

    V = 1.0
    open_legs: list[dict] = []
    units = 0.0
    entry_book = 0.0
    opened_at = None
    eq, rets, cost_s, net_delta, net_vega, trades = [], [], [], [], [], []
    prev_val = 0.0
    tv = term.to_numpy(float)
    Sv, skv, rfv, qv, scv = (S.to_numpy(float), skew.to_numpy(float), rf.to_numpy(float), q.to_numpy(float),
                             scale.to_numpy(float))

    def expiry_for(t, dte):
        if strat.roll == "monthly":
            return next((e for e in expiries if e > t + pd.Timedelta(days=max(7, dte - 25))), None)
        return t + pd.Timedelta(days=max(1, dte))

    def leg_values(i, legs):
        s, t = Sv[i], idx[i]
        out = []
        for L in legs:
            T = max((L["expiry"] - t).days, 0) / 365.0
            atm = atm_vol(tv[i], max(T, 1 / 365)) * scv[i] * strat.atm_ratio
            sig = float(smile(atm, s, L["K"], T, skv[i])) if np.isfinite(atm) else L["sig0"]
            out.append(float(price(L["leg"].kind == "call", s * np.exp(-qv[i] * T), L["K"], T, sig, rfv[i])))
        return np.array(out)

    def half_spread(vals, legs, s, alive_only=False, t=None):
        tot = 0.0
        for v, L in zip(vals, legs):
            if alive_only and L["expiry"] <= t:
                continue                                    # settled legs pay no spread
            tot += max(strat.spread_pct * v, strat.min_spread_bps / 1e4 * s) * L["leg"].qty
        return tot

    for i, t in enumerate(idx):
        day_cost = 0.0
        pnl = 0.0
        s = Sv[i]
        # 1) mark existing legs and manage
        if open_legs:
            vals = leg_values(i, open_legs)
            book = float(sum(L["leg"].side * L["leg"].qty * v for L, v in zip(open_legs, vals)))
            pnl += units * (book - prev_val)
            prev_val = book
            front = min(L["expiry"] for L in open_legs)
            gain = book - entry_book                        # P&L per unit since open
            basis = abs(entry_book)                         # credit received or debit paid
            reason = None
            if front <= t:
                reason = "expiry"
            elif strat.take_profit > 0 and basis > 0 and gain >= strat.take_profit * basis:
                reason = "take profit"
            elif strat.stop_mult > 0 and basis > 0 and -gain >= strat.stop_mult * basis:
                reason = "stop"
            elif strat.exit_dte > 0 and (front - t).days <= strat.exit_dte:
                reason = "time exit"
            elif strat.max_hold > 0 and opened_at is not None and (t - opened_at).days >= strat.max_hold:
                reason = "max hold"
            elif strat.delta_exit > 0:
                d = 0.0
                for L in open_legs:
                    T = max((L["expiry"] - t).days, 0) / 365.0
                    g = greeks(L["leg"].kind == "call", s, L["K"], T, L["sig0"], rfv[i])
                    d += float(g["delta"]) * L["leg"].side * L["leg"].qty
                if abs(d + strat.stock) > strat.delta_exit:
                    reason = "delta exit"
            if reason:
                day_cost += half_spread(vals, open_legs, s, alive_only=True, t=t) * units
                trades[-1].update({"close": t.date(), "reason": reason, "days": (t - opened_at).days,
                                   "pnl_per_unit_%": round(gain / Sv[i] * 100, 3),
                                   "pnl_%_equity": round(units * gain / max(V, 1e-12) * 100, 3)})
                open_legs, prev_val = [], 0.0
        # 2) stock leg and collateral interest (on yesterday's equity)
        if i > 0:
            if units and strat.stock:
                pnl += strat.stock * stock_ret.iloc[i] * units * Sv[i - 1]
            cash_rate = rfv[i - 1] / 252.0
            stock_notional = strat.stock * units * Sv[i - 1] if (units and strat.stock) else 0.0
            pnl += cash_rate * (V - stock_notional)
        # 3) open a new cycle
        if not open_legs and gate_np[i] and i < len(idx) - 2:
            exp0 = expiry_for(t, strat.dte)
            if exp0 is not None:
                T0 = (exp0 - t).days / 365.0
                atm0 = atm_vol(tv[i], T0) * scv[i] * strat.atm_ratio
                if np.isfinite(atm0):
                    legs, book, prior = [], 0.0, []
                    for leg in strat.legs:
                        exp = exp0 if not leg.dte else expiry_for(t, leg.dte)
                        if exp is None:
                            break
                        T = max((exp - t).days, 1) / 365.0
                        atm = atm_vol(tv[i], T) * scv[i] * strat.atm_ratio
                        fwd_s = s * np.exp(-qv[i] * T)              # dividend-adjusted spot
                        K = _strike(leg, fwd_s if leg.rule in ("delta", "premium") else s, T, atm, rfv[i], skv[i],
                                    prior)
                        sig = float(smile(atm, s, K, T, skv[i]))
                        v = float(price(leg.kind == "call", fwd_s, K, T, sig, rfv[i]))
                        legs.append({"leg": leg, "K": K, "expiry": exp, "sig0": sig, "v0": v})
                        prior.append(K)
                        book += leg.side * leg.qty * v
                    else:
                        net_pct = -book / s * 100
                        d0 = strat.stock + sum(
                            float(greeks(L["leg"].kind == "call", s, L["K"], (L["expiry"] - t).days / 365.0,
                                         L["sig0"], rfv[i])["delta"]) * L["leg"].side * L["leg"].qty for L in legs)
                        w = _width(legs)
                        cw = (-book / w) if (w > 0 and book < 0) else None
                        if _trade_ok(filt, net_pct, d0, cw):
                            V_now = V + pnl - day_cost
                            units = _units(strat, V_now, s, legs, book)
                            open_legs, prev_val, entry_book, opened_at = legs, book, book, t
                            day_cost += half_spread([L["v0"] for L in legs], legs, s) * units
                            trades.append({"open": t.date(), "expiry": exp0.date(), "spot": round(s, 2),
                                           "strikes": " / ".join(
                                               f"{L['leg'].side * L['leg'].qty:+g}{L['leg'].kind[0].upper()}"
                                               f"{L['K']:.0f}" + (f"@{(L['expiry'] - t).days}d" if L['leg'].dte else "")
                                               for L in legs),
                                           "net_credit_%": round(net_pct, 3), "delta": round(d0, 3),
                                           "atm_iv": round(atm0 * 100, 1),
                                           "size_x_notional": round(units * s / max(V_now, 1e-12), 3)})
        # 4) book the day
        V_new = V + pnl - day_cost
        rets.append(V_new / V - 1.0 if V > 0 else 0.0)
        V = max(V_new, 1e-9)
        eq.append(V)
        cost_s.append(day_cost / (V_new + day_cost) if V_new + day_cost > 0 else 0.0)
        # exposures per $ of equity
        d = strat.stock * units * s / V if units else 0.0
        vg = 0.0
        if open_legs:
            for L in open_legs:
                T = max((L["expiry"] - t).days, 0) / 365.0
                g = greeks(L["leg"].kind == "call", s * np.exp(-qv[i] * T), L["K"], T, L["sig0"], rfv[i])
                d += float(g["delta"]) * L["leg"].side * L["leg"].qty * units * s / V
                vg += float(g["vega"]) * L["leg"].side * L["leg"].qty * units / V
        net_delta.append(d)
        net_vega.append(vg)

    r = pd.Series(rets, index=idx)
    return OptionResult(returns=r, equity=pd.Series(eq, index=idx), trades=pd.DataFrame(trades),
                        exposure=pd.DataFrame({"delta": net_delta, "vega_per_volpt": net_vega}, index=idx),
                        costs=pd.Series(cost_s, index=idx), label=f"{strat.name} · MODELED", gate=gate)


# ------------------------------------------------------------------ 0DTE ----
def simulate_0dte(strat: OptionStrategy, ohlc: pd.DataFrame, iv1d: pd.Series, skew: pd.Series, rf: pd.Series,
                  term: pd.DataFrame | None = None, iv_scale: float = 1.0,
                  actual_expiries: bool = True) -> OptionResult:
    """Same-day-expiry options from daily bars (MODELED).

    Opens at the session open, priced with T = one 6.5-hour session and σ_atm = yesterday's
    close of the 1-day vol index (VIX1D, or the VIX9D-based fill before it existed) × ``iv_scale``.
    Settles at the close at intrinsic value. Stops and targets are checked along an assumed
    intraday path through the day's high and low (low first on up days, high first on down days),
    marked with 2/3 and 1/3 of the session remaining."""
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    idx = ohlc.index
    O, H, Lo, C = (ohlc[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    sig_y = (iv1d.reindex(idx).ffill().shift(1) / 100.0 * iv_scale).to_numpy(float)   # known at the open
    skv = skew.reindex(idx).ffill().shift(1).to_numpy(float)
    rfv = rf.reindex(idx).ffill().fillna(0.0).to_numpy(float)
    term_i = (term if term is not None else pd.DataFrame({"VIX": iv1d})).reindex(idx).ffill().shift(1)
    gate = entry_gate(strat, idx, ohlc["close"].shift(1).reindex(idx), term_i)
    if actual_expiries:
        gate &= pd.Series([spx_daily_expiry(d) for d in idx], index=idx)
    g = gate.to_numpy()
    filt = dict(strat.filters or {})
    T0 = SESSION_T

    def book_at(legs, s, T, sig_atm, sk, r):
        tot = 0.0
        for L in legs:
            sig = float(smile(sig_atm, s, L["K"], T, sk)) if T > 0 else sig_atm
            tot += L["leg"].side * L["leg"].qty * float(price(L["leg"].kind == "call", s, L["K"], T, sig, r))
        return tot

    V = 1.0
    eq, rets, costs, trades = [], [], [], []
    for i, t in enumerate(idx):
        pnl, cost = 0.0, 0.0
        if i > 0:
            pnl += rfv[i - 1] / 252.0 * V
        if g[i] and np.isfinite(sig_y[i]) and sig_y[i] > 0:
            s0, sig, sk, r = O[i], sig_y[i], skv[i] if np.isfinite(skv[i]) else 130.0, rfv[i]
            legs, prior = [], []
            for leg in strat.legs:
                K = _strike(leg, s0, T0, sig, r, sk, prior)
                v = float(price(leg.kind == "call", s0, K, T0, float(smile(sig, s0, K, T0, sk)), r))
                legs.append({"leg": leg, "K": K, "v0": v})
                prior.append(K)
            b0 = sum(L["leg"].side * L["leg"].qty * L["v0"] for L in legs)
            net_pct = -b0 / s0 * 100
            d0 = sum(float(greeks(L["leg"].kind == "call", s0, L["K"], T0, sig, r)["delta"]) * L["leg"].side
                     * L["leg"].qty for L in legs)
            w = _width(legs)
            cw = (-b0 / w) if (w > 0 and b0 < 0) else None
            if _trade_ok(filt, net_pct, d0, cw):
                for L in legs:
                    L["expiry"] = t
                units = _units(strat, V, s0, legs, b0)
                spread = lambda vals: sum(max(strat.spread_pct * v, strat.min_spread_bps / 1e4 * s0) * L["leg"].qty  # noqa: E731
                                          for v, L in zip(vals, legs))
                cost += spread([L["v0"] for L in legs]) * units
                basis = abs(b0)
                stop_amt = strat.stop_mult * basis if strat.stop_mult > 0 else None
                tp_amt = strat.take_profit * basis if strat.take_profit > 0 else None
                # OHLC path rule: an up day is assumed to visit its low first, a down day its high first;
                # the extremes are marked with 2/3 and 1/3 of the session left. First stop/target hit wins.
                path = (Lo[i], H[i]) if C[i] >= O[i] else (H[i], Lo[i])
                reason, gain = "expiry", None
                for x, left in zip(path, (2 / 3, 1 / 3)):
                    pnl_x = book_at(legs, x, T0 * left, sig, sk, r) - b0
                    if stop_amt is not None and -pnl_x >= stop_amt:
                        reason, gain = "stop", -stop_amt
                        break
                    if tp_amt is not None and pnl_x >= tp_amt:
                        reason, gain = "take profit", tp_amt
                        break
                if gain is None:
                    gain = book_at(legs, C[i], 0.0, sig, sk, r) - b0
                if reason != "expiry":
                    cost += spread([abs(v) for v in [L["v0"] for L in legs]]) * units * 0.5
                pnl += units * gain
                trades.append({"date": t.date(), "open": round(s0, 2), "close": round(C[i], 2),
                               "strikes": " / ".join(f"{L['leg'].side * L['leg'].qty:+g}{L['leg'].kind[0].upper()}"
                                                     f"{L['K']:.0f}" for L in legs),
                               "net_credit_%": round(net_pct, 3), "iv_1d": round(sig * 100, 1), "reason": reason,
                               "pnl_%_equity": round(units * gain / V * 100, 3)})
        V_new = V + pnl - cost
        rets.append(V_new / V - 1.0)
        V = max(V_new, 1e-9)
        eq.append(V)
        costs.append(cost / (V_new + cost) if V_new + cost > 0 else 0.0)
    z = np.zeros(len(idx))
    return OptionResult(returns=pd.Series(rets, index=idx), equity=pd.Series(eq, index=idx),
                        trades=pd.DataFrame(trades),
                        exposure=pd.DataFrame({"delta": z, "vega_per_volpt": z}, index=idx),
                        costs=pd.Series(costs, index=idx), label=f"{strat.name} · 0DTE · MODELED", gate=gate)


def implied_vs_realised_0dte(ohlc: pd.DataFrame, iv1d: pd.Series) -> dict:
    """Does the 1-day vol index size the open→close move right? Ratio > 1: options were cheap."""
    df = pd.DataFrame({"o": ohlc["open"], "c": ohlc["close"],
                       "iv": iv1d.reindex(ohlc.index).ffill().shift(1) / 100}).dropna()
    df = df[df["o"] > 0]
    if len(df) < 60:
        return {"ok": False}
    realised = np.log(df["c"] / df["o"])
    implied = df["iv"] * np.sqrt(SESSION_T)
    return {"ok": True, "n": len(df), "rms_realised": float(np.sqrt((realised ** 2).mean())),
            "rms_implied": float(np.sqrt((implied ** 2).mean())),
            "ratio": float(np.sqrt((realised ** 2).mean()) / np.sqrt((implied ** 2).mean()))}


# --------------------------------------------------------------- preview ----
def entry_preview(strat: OptionStrategy, s: float, term_row: np.ndarray, skew_v: float, r: float,
                  vol_scale: float = 1.0, iv1d: float | None = None) -> dict:
    """Legs, prices, greeks and the expiry payoff for opening ``strat`` today."""
    zero = strat.roll == "0dte"
    T0 = SESSION_T if zero else max(strat.dte, 1) / 365.0
    atm0 = (iv1d / 100 if (zero and iv1d) else atm_vol(term_row, T0) * strat.atm_ratio) * vol_scale
    legs, prior, book, net = [], [], 0.0, {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    for leg in strat.legs:
        T = T0 if (zero or not leg.dte) else leg.dte / 365.0
        atm = atm0 if (zero or not leg.dte) else atm_vol(term_row, T) * strat.atm_ratio * vol_scale
        K = _strike(leg, s, T, atm, r, skew_v, prior)
        sig = float(smile(atm, s, K, T, skew_v))
        v = float(price(leg.kind == "call", s, K, T, sig, r))
        g = greeks(leg.kind == "call", s, K, T, sig, r)
        for k in net:
            net[k] += float(g[k]) * leg.side * leg.qty
        legs.append({"leg": leg, "K": K, "price": v, "iv": sig, "T_days": T * 365, "expiry": None})
        prior.append(K)
        book += leg.side * leg.qty * v
    net["delta"] += strat.stock
    grid = np.linspace(s * 0.75, s * 1.25, 201) if not zero else np.linspace(s * 0.95, s * 1.05, 201)
    pay = expiry_payoff(legs, grid, strat.stock, s) - book
    return {"legs": legs, "book": book, "net_credit_%": -book / s * 100, "greeks": net, "grid": grid,
            "payoff": pay, "max_loss": float(-pay.min()), "max_gain": float(pay.max()),
            "breakevens": [float(grid[j]) for j in np.where(np.diff(np.sign(pay)) != 0)[0]]}


# ------------------------------------------------------------ calibration ----
def calibrate(model: pd.Series, actual_level: pd.Series) -> dict:
    """Compare modeled daily returns with a published index level (monthly)."""
    a = actual_level.pct_change()
    m = (1 + model).resample("ME").prod() - 1
    am = (1 + a).resample("ME").prod() - 1
    df = pd.concat([m.rename("model"), am.rename("actual")], axis=1).dropna()
    df = df.iloc[1:]
    if len(df) < 12:
        return {"ok": False, "n": len(df)}
    te = (df["model"] - df["actual"]).std(ddof=1) * np.sqrt(12)
    yrs = len(df) / 12
    cagr = lambda x: (1 + x).prod() ** (1 / yrs) - 1  # noqa: E731
    return {"ok": True, "n": len(df), "corr": float(df["model"].corr(df["actual"])), "tracking_error": float(te),
            "cagr_model": float(cagr(df["model"])), "cagr_actual": float(cagr(df["actual"])),
            "vol_model": float(df["model"].std() * np.sqrt(12)), "vol_actual": float(df["actual"].std() * np.sqrt(12)),
            "months": df}


def calibration_tier(cal: dict) -> tuple[str, str]:
    if not cal.get("ok"):
        return "C", "no overlap with the published index to calibrate against"
    gap = abs(cal["cagr_model"] - cal["cagr_actual"])
    if cal["corr"] > 0.9 and cal["tracking_error"] < 0.04 and gap < 0.02:
        return "A", "model tracks the published index closely"
    if cal["corr"] > 0.75 and cal["tracking_error"] < 0.08:
        return "B", "model tracks the index's shape; levels drift — treat absolute returns as approximate"
    return "C", "model and index diverge — use for relative comparisons only"
