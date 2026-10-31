"""From a priced chain to smiles, a constant-maturity term structure and a surface.

Per listed expiry ("slice"): the OTM smile in log-moneyness k = ln(K/F), ATM
vol read at k = 0, vols at fixed DELTAS (10P..ATM..10C) and on a fixed K/F
grid. Across expiries, constant maturities are interpolated LINEARLY IN TOTAL
VARIANCE w = sigma^2 T (the VIX convention: variance is additive in time, vol
is not), with flat-vol extrapolation flagged at either end.

Earnings: an expiry that spans the report carries the event's variance e^2 on
top of the diffusive variance, w = sigma_ex^2 T + e^2. With a pre-event expiry
the ex-event vol is read straight off it; with two post-event expiries it is
the FORWARD variance between them (both hold the event, so it cancels):
    sigma_ex^2 = (w3 - w2) / (T3 - T2),   e^2 = w2 - sigma_ex^2 T2.
The market's implied earnings move is E|move| = e sqrt(2/pi).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from ..config import VOLSCREEN_DELTAS, VOLSCREEN_MNY, VOLSCREEN_TENORS

MIN_DTE = 1                     # 0DTE vols are dominated by the overnight gap: skipped
MIN_QUOTES = 4


def _interp_inside(x, y, at) -> float:
    """Linear interpolation, NaN outside the observed range (never extrapolate a smile)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return float("nan")
    o = np.argsort(x)
    x, y = x[o], y[o]
    if at < x[0] - 1e-12 or at > x[-1] + 1e-12:
        return float("nan")
    return float(np.interp(at, x, y))


def slices(px: pd.DataFrame) -> list[dict]:
    """One dict per usable expiry: dte, T, F, atm, delta[9], mny[17], n."""
    out = []
    if px.empty:
        return out
    for exp, g in px.groupby("expiry"):
        dte = int(g["dte"].iloc[0])
        if dte < MIN_DTE:
            continue
        o = g[g["otm"] & np.isfinite(g["iv_use"])]
        if len(o) < MIN_QUOTES:
            continue
        k = o["k"].to_numpy(float)
        v = o["iv_use"].to_numpy(float)
        atm = _interp_inside(k, v, 0.0)
        if not np.isfinite(atm):                       # one-sided smile: nearest OTM vol
            atm = float(v[np.argmin(np.abs(k))]) if abs(k[np.argmin(np.abs(k))]) < 0.03 else float("nan")
        if not np.isfinite(atm):
            continue
        puts = o[(o["cp"] == "P") & (o["d"] < 0) & (o["d"] > -0.5)]
        calls = o[(o["cp"] == "C") & (o["d"] > 0) & (o["d"] < 0.5)]
        dvals = []
        for _, dl in VOLSCREEN_DELTAS:
            if dl == 0.5:
                dvals.append(atm)
            elif dl < 0:
                dvals.append(_interp_inside(-puts["d"], puts["iv_use"], -dl))
            else:
                dvals.append(_interp_inside(calls["d"], calls["iv_use"], dl))
        mvals = [_interp_inside(k, v, math.log(m)) for m in VOLSCREEN_MNY]
        out.append({"expiry": pd.Timestamp(exp).date().isoformat(), "dte": dte,
                    "T": float(g["T"].iloc[0]), "F": float(g["F"].iloc[0]), "atm": atm,
                    "delta": dvals, "mny": mvals, "n": int(len(o))})
    return out


def cm_interp(Ts, vols, T: float) -> tuple[float, bool]:
    """Constant-maturity vol at T by linear interpolation in total variance.
    Returns (vol, extrapolated)."""
    Ts = np.asarray(Ts, float)
    v = np.asarray(vols, float)
    m = np.isfinite(Ts) & np.isfinite(v) & (v > 0)
    Ts, v = Ts[m], v[m]
    if not len(Ts):
        return float("nan"), True
    o = np.argsort(Ts)
    Ts, v = Ts[o], v[o]
    if T <= Ts[0]:
        return float(v[0]), T < Ts[0] * 0.97
    if T >= Ts[-1]:
        return float(v[-1]), T > Ts[-1] * 1.03
    w = v * v * Ts
    wt = float(np.interp(T, Ts, w))
    return math.sqrt(max(wt, 1e-12) / T), False


def term(sl: list[dict], tenors=VOLSCREEN_TENORS, field: str = "atm") -> tuple[dict, list[int]]:
    """{tenor_days: vol} and the tenors that had to be extrapolated."""
    Ts = [s["T"] for s in sl]
    vals = [s[field] for s in sl]
    out, ext = {}, []
    for t in tenors:
        v, e = cm_interp(Ts, vals, t / 365.0)
        out[t] = v
        if e:
            ext.append(t)
    return out, ext


def grid(sl: list[dict], key: str, n: int, tenors=VOLSCREEN_TENORS) -> list[list[float]]:
    """tenor x column grid (delta buckets or moneyness), each column in total variance."""
    Ts = [s["T"] for s in sl]
    rows = []
    for t in tenors:
        row = []
        for j in range(n):
            col = [s[key][j] for s in sl]
            v, e = cm_interp(Ts, col, t / 365.0)
            # wings are only trusted where both bracketing expiries quote them
            row.append(v if np.isfinite(v) else float("nan"))
        rows.append(row)
    return rows


def calendar_violations(sl: list[dict], tol: float = 0.02) -> int:
    """Count of expiries whose ATM total variance falls (calendar arbitrage / bad quote)."""
    w = [s["atm"] ** 2 * s["T"] for s in sorted(sl, key=lambda s: s["T"])]
    return int(sum(1 for a, b in zip(w, w[1:]) if b < a * (1 - tol)))


def event_variance(sl: list[dict], event: date | None) -> dict | None:
    """Earnings event variance from the term-structure kink (see module doc)."""
    if not event or len(sl) < 2:
        return None
    ev = pd.Timestamp(event).date()
    srt = sorted(sl, key=lambda s: s["T"])
    pre = [s for s in srt if date.fromisoformat(s["expiry"]) < ev and s["dte"] >= 2]
    post = [s for s in srt if date.fromisoformat(s["expiry"]) >= ev]
    if not post:
        return None
    s2 = post[0]
    w2 = s2["atm"] ** 2 * s2["T"]
    if pre:
        sig_ex, how = pre[-1]["atm"], "pre-event expiry"
    elif len(post) >= 2:
        s3 = post[1]
        fv = (s3["atm"] ** 2 * s3["T"] - w2) / max(s3["T"] - s2["T"], 1e-6)
        if fv <= 0:
            return None
        sig_ex, how = math.sqrt(fv), "forward variance"
    else:
        return None
    e2 = max(w2 - sig_ex ** 2 * s2["T"], 0.0)
    e = math.sqrt(e2)
    return {"e2": e2, "move": e * math.sqrt(2.0 / math.pi), "sigma_ex": sig_ex, "method": how,
            "expiry": s2["expiry"]}


def ex_event_term(tm: dict, ev: dict | None, event_T: float | None) -> dict:
    """Constant-maturity vols with the event variance removed from tenors past it."""
    if not ev or event_T is None or not ev["e2"]:
        return dict(tm)
    out = {}
    for t, v in tm.items():
        T = t / 365.0
        if np.isfinite(v) and T > event_T:
            w = v * v * T - ev["e2"]
            out[t] = math.sqrt(max(w, 0.25 * v * v * T) / T)
        else:
            out[t] = v
    return out


def front_back(sl: list[dict]) -> dict:
    """Listed front (first expiry >= 7 DTE) and back (nearest 90 DTE) ATM vols."""
    if not sl:
        return {}
    f = [s for s in sl if s["dte"] >= 7] or sl
    front = min(f, key=lambda s: s["dte"])
    back = min(sl, key=lambda s: abs(s["dte"] - 90))
    return {"front": front["atm"], "front_dte": front["dte"], "back": back["atm"], "back_dte": back["dte"]}


def skew_metrics(grid_delta: list[list[float]], tenors=VOLSCREEN_TENORS, at: int = 30) -> dict:
    """25-delta put-minus-call skew (put richness) and butterfly at a tenor."""
    labels = [d[0] for d in VOLSCREEN_DELTAS]
    try:
        row = grid_delta[list(tenors).index(at)]
    except ValueError:
        return {}
    p25, c25, atm = row[labels.index("25P")], row[labels.index("25C")], row[labels.index("50")]
    p10 = row[labels.index("10P")]
    return {"skew25": p25 - c25, "bf25": 0.5 * (p25 + c25) - atm, "put_wing": p10 - atm}
