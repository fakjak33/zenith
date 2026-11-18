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

Options are priced with THETA's Black-Scholes; positions are fully
collateralised (one unit of underlying notional per contract-unit, like Cboe's
PUT index), cash earns T-bills, and every open/close pays a half-spread.
Results are always stamped MODELED and calibrated against Cboe PUT and BXM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...theta.engine.pricing import price, strike_for_delta

TENORS = np.array([9, 30, 91, 182]) / 365.0
TERM_COLS = ["VIX9D", "VIX", "VIX3M", "VIX6M"]
# Fitted once on Cboe PUT (2007→) so modeled ATM put-write CAGR matches the index; BXM is the
# out-of-sample check (see tests/test_backtest_options.py and the calibration panel).
ATM_RATIO = 0.87


# ------------------------------------------------------------------ legs ----
@dataclass(frozen=True)
class Leg:
    kind: str            # "put" | "call"
    side: int            # -1 short, +1 long
    rule: str            # "delta" | "pct" | "atm"
    value: float = 0.0   # 0.30 for 30-delta; 0.95 for 95% of spot
    qty: float = 1.0


@dataclass
class OptionStrategy:
    name: str
    legs: list = field(default_factory=list)
    stock: float = 0.0           # units of underlying held (1 = covered call / collar)
    dte: int = 30                # calendar days to target expiry
    roll: str = "monthly"        # "monthly" = 3rd-Friday cycle (Cboe style) | "dte" = reopen at target DTE
    take_profit: float = 0.0     # close when this share of the opening credit is captured (0 = hold)
    stop_mult: float = 0.0       # close when loss reaches this × opening credit (0 = none)
    min_vix: float = 0.0         # only open when VIX ≥ this
    spread_pct: float = 0.03     # half-spread as a share of option price, per leg, each way
    min_spread_bps: float = 0.5  # floor on the half-spread, in bps of spot
    atm_ratio: float = ATM_RATIO  # VIX is a variance-swap level, above ATM implied vol; ATM ≈ ratio × VIX


TEMPLATES: dict[str, OptionStrategy] = {
    "Short put (PUT-style)": OptionStrategy("Short put", [Leg("put", -1, "atm")]),
    "Covered call (BXM-style)": OptionStrategy("Covered call", [Leg("call", -1, "atm")], stock=1.0),
    "OTM put write (30Δ)": OptionStrategy("30Δ put write", [Leg("put", -1, "delta", 0.30)]),
    "Short strangle (16Δ)": OptionStrategy("Short strangle", [Leg("put", -1, "delta", 0.16),
                                                              Leg("call", -1, "delta", 0.16)]),
    "Iron condor (16Δ / 5Δ)": OptionStrategy("Iron condor", [Leg("put", -1, "delta", 0.16), Leg("put", 1, "delta", 0.05),
                                                             Leg("call", -1, "delta", 0.16), Leg("call", 1, "delta", 0.05)]),
    "Put spread (30Δ / 10Δ)": OptionStrategy("Put credit spread", [Leg("put", -1, "delta", 0.30),
                                                                   Leg("put", 1, "delta", 0.10)]),
    "Collar (95% put / 105% call)": OptionStrategy("Collar", [Leg("put", 1, "pct", 0.95), Leg("call", -1, "pct", 1.05)],
                                                   stock=1.0),
    "Long put hedge (90%)": OptionStrategy("Protective put", [Leg("put", 1, "pct", 0.90)], stock=1.0),
}


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
    z = np.log(np.asarray(K, float) / S) / (atm * np.sqrt(max(T, 1 / 365)))
    mult = np.clip(np.exp(-kappa * z), 0.6, 3.0)
    return atm * mult


def third_fridays(start, end) -> pd.DatetimeIndex:
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    return pd.DatetimeIndex([m + pd.offsets.WeekOfMonth(week=2, weekday=4) for m in months])


# ------------------------------------------------------------- simulator ----
@dataclass
class OptionResult:
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.DataFrame         # daily net delta / vega per unit notional
    costs: pd.Series
    label: str = "MODELED"


def _strike(leg: Leg, S: float, T: float, sig_atm: float, r: float, skew: float) -> float:
    if leg.rule == "atm":
        return S
    if leg.rule == "pct":
        return S * leg.value
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
    fridays = set(third_fridays(idx[0], idx[-1] + pd.Timedelta(days=40)))
    expiries = sorted(fridays)

    V = 1.0
    open_legs: list[dict] = []
    units = 0.0                 # underlying notional units per $ of equity at open
    credit = 0.0
    eq, rets, cost_s, net_delta, net_vega, trades = [], [], [], [], [], []
    prev_val = 0.0
    tv = term.to_numpy(float)
    vix_now = term["VIX"].to_numpy(float) if "VIX" in term else np.full(len(idx), np.nan)

    def leg_values(i, legs):
        s, t = S.iloc[i], idx[i]
        out = []
        for L in legs:
            T = max((L["expiry"] - t).days, 0) / 365.0
            atm = atm_vol(tv[i], max(T, 1 / 365)) * scale.iloc[i] * strat.atm_ratio
            sig = float(smile(atm, s, L["K"], T, skew.iloc[i])) if np.isfinite(atm) else L["sig0"]
            out.append(float(price(L["leg"].kind == "call", s * np.exp(-q.iloc[i] * T), L["K"], T, sig, rf.iloc[i])))
        return np.array(out)

    def close_all(i, reason):
        nonlocal open_legs, units, prev_val
        vals = leg_values(i, open_legs)
        sp = sum(max(strat.spread_pct * v, strat.min_spread_bps / 1e4 * S.iloc[i]) * L["leg"].qty
                 for v, L in zip(vals, open_legs))
        trades.append({"close": idx[i].date(), "reason": reason})
        open_legs, prev_val = [], 0.0
        return sp * units

    for i, t in enumerate(idx):
        day_cost = 0.0
        pnl = 0.0
        s = S.iloc[i]
        # 1) mark existing legs
        if open_legs:
            vals = leg_values(i, open_legs)
            book = float(sum(L["leg"].side * L["leg"].qty * v for L, v in zip(open_legs, vals)))
            pnl += units * (book - prev_val)
            prev_val = book
            expired = all(L["expiry"] <= t for L in open_legs)
            gained = credit - (-book)                           # for credit trades: credit captured so far
            if expired:
                trades[-1].update({"close": t.date(), "reason": "expiry", "pnl_per_unit": round(credit + book, 4)})
                open_legs, prev_val = [], 0.0
            elif strat.take_profit > 0 and credit > 0 and gained >= strat.take_profit * credit:
                trades[-1].update({"pnl_per_unit": round(credit + book, 4)})
                day_cost += close_all(i, "take profit")
            elif strat.stop_mult > 0 and credit > 0 and -book - credit >= strat.stop_mult * credit:
                trades[-1].update({"pnl_per_unit": round(credit + book, 4)})
                day_cost += close_all(i, "stop")
        # 2) stock leg and collateral interest (on yesterday's equity)
        if i > 0:
            pnl += strat.stock * stock_ret.iloc[i] * units * S.iloc[i - 1] if units else 0.0
            cash_rate = rf.iloc[i - 1] / 252.0
            pnl += cash_rate * (V - (strat.stock * units * S.iloc[i - 1] if units else 0.0)) \
                if strat.stock else cash_rate * V
        # 3) open a new cycle
        can_open = not open_legs and (strat.min_vix <= 0 or (np.isfinite(vix_now[i]) and vix_now[i] >= strat.min_vix))
        if can_open and i < len(idx) - 2:
            if strat.roll == "monthly":
                exp = next((e for e in expiries if e > t + pd.Timedelta(days=max(7, strat.dte - 25))), None)
            else:
                exp = t + pd.Timedelta(days=strat.dte)
            if exp is not None:
                T = (exp - t).days / 365.0
                atm = atm_vol(tv[i], T) * scale.iloc[i] * strat.atm_ratio
                if np.isfinite(atm):
                    V_now = V + pnl - day_cost
                    units = V_now / s                              # fully collateralised: 1 unit per $ notional
                    legs, book, sp = [], 0.0, 0.0
                    for leg in strat.legs:
                        fwd_s = s * np.exp(-q.iloc[i] * T)          # dividend-adjusted spot
                        K = _strike(leg, s, T, atm, rf.iloc[i], skew.iloc[i]) if leg.rule != "delta" else                             _strike(leg, fwd_s, T, atm, rf.iloc[i], skew.iloc[i])
                        sig = float(smile(atm, s, K, T, skew.iloc[i]))
                        v = float(price(leg.kind == "call", fwd_s, K, T, sig, rf.iloc[i]))
                        legs.append({"leg": leg, "K": K, "expiry": exp, "sig0": sig})
                        book += leg.side * leg.qty * v
                        sp += max(strat.spread_pct * v, strat.min_spread_bps / 1e4 * s) * leg.qty
                    open_legs, prev_val, credit = legs, book, -book
                    day_cost += sp * units
                    trades.append({"open": t.date(), "expiry": exp.date(), "spot": round(s, 2),
                                   "strikes": " / ".join(f"{L['leg'].side:+d}{L['leg'].kind[0].upper()}{L['K']:.0f}"
                                                         for L in legs),
                                   "credit_%": round(credit / s * 100, 3), "atm_iv": round(atm * 100, 1)})
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
            from ...theta.engine.pricing import greeks
            for L in open_legs:
                T = max((L["expiry"] - t).days, 0) / 365.0
                g = greeks(L["leg"].kind == "call", s * np.exp(-q.iloc[i] * T), L["K"], T, L["sig0"], rf.iloc[i])
                d += float(g["delta"]) * L["leg"].side * L["leg"].qty * units * s / V
                vg += float(g["vega"]) * L["leg"].side * L["leg"].qty * units / V
        net_delta.append(d)
        net_vega.append(vg)

    r = pd.Series(rets, index=idx)
    return OptionResult(returns=r, equity=pd.Series(eq, index=idx), trades=pd.DataFrame(trades),
                        exposure=pd.DataFrame({"delta": net_delta, "vega_per_volpt": net_vega}, index=idx),
                        costs=pd.Series(cost_s, index=idx), label=f"{strat.name} · MODELED")


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
