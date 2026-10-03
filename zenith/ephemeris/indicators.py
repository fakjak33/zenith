"""Causal technical indicators (pure numpy), ported from VELA's indicators.ts
and extended.

Contract: every function returns arrays the SAME LENGTH as the input with
NaN during warm-up, and the value at index i depends only on bars 0..i.
That is what makes "compute once on the full fetched series, then slice" safe
-- tests/test_ephemeris_indicators.py truncates the series at many decision
points and asserts no visible value changes.

`compute(specs, ...)` turns a list of indicator specs into chart series:
overlays (drawn on price, rebased with it) and panes (own scale).
"""

from __future__ import annotations

import numpy as np

NAN = np.nan


def _nan(n: int) -> np.ndarray:
    return np.full(n, NAN)


# ------------------------------------------------------------- primitives ----
def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = _nan(len(x))
    if n <= 0 or len(x) < n:
        return out
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded with the SMA of the first n values (charting convention)."""
    x = np.asarray(x, float)
    out = _nan(len(x))
    start = np.argmax(np.isfinite(x)) if np.isfinite(x).any() else len(x)
    if n <= 0 or len(x) - start < n:
        return out
    k = 2.0 / (n + 1)
    prev = x[start:start + n].mean()
    out[start + n - 1] = prev
    for i in range(start + n, len(x)):
        prev = x[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothing (RMA), seeded with the SMA of the first n values."""
    x = np.asarray(x, float)
    out = _nan(len(x))
    start = np.argmax(np.isfinite(x)) if np.isfinite(x).any() else len(x)
    if n <= 0 or len(x) - start < n:
        return out
    prev = x[start:start + n].mean()
    out[start + n - 1] = prev
    for i in range(start + n, len(x)):
        prev = (prev * (n - 1) + x[i]) / n
        out[i] = prev
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Population standard deviation over a rolling window."""
    x = np.asarray(x, float)
    out = _nan(len(x))
    if n <= 0 or len(x) < n:
        return out
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n - 1:] = w.std(axis=1)
    return out


def rolling_max(x, n):
    x = np.asarray(x, float)
    out = _nan(len(x))
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x, n):
    x = np.asarray(x, float)
    out = _nan(len(x))
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n).min(axis=1)
    return out


def true_range(h, l, c) -> np.ndarray:
    h, l, c = (np.asarray(a, float) for a in (h, l, c))
    pc = np.concatenate([[NAN], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n: int = 14) -> np.ndarray:
    return wilder(true_range(h, l, c), n)


def rsi(c, n: int = 14) -> np.ndarray:
    c = np.asarray(c, float)
    out = _nan(len(c))
    if len(c) <= n:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = wilder(up, n), wilder(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ad == 0, 100.0, 100 - 100 / (1 + au / ad))
    r = np.where(np.isfinite(au) & np.isfinite(ad), r, NAN)
    out[1:] = r
    return out


def macd(c, fast=12, slow=26, signal=9):
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)                    # runs over the defined part only
    return line, sig, line - sig


def bollinger(c, n=20, k=2.0):
    m = sma(c, n)
    s = rolling_std(c, n)
    return m + k * s, m, m - k * s


def keltner(h, l, c, n=20, mult=2.0):
    m = ema(c, n)
    a = atr(h, l, c, n)
    return m + mult * a, m, m - mult * a


def donchian(h, l, n=20):
    return rolling_max(h, n), rolling_min(l, n)


def vwap_rolling(h, l, c, v, n=20):
    tp = (np.asarray(h) + np.asarray(l) + np.asarray(c)) / 3.0
    v = np.asarray(v, float)
    num, den = sma(tp * v, n), sma(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, NAN)


def vwap_anchored(h, l, c, v, anchor: int):
    tp = (np.asarray(h) + np.asarray(l) + np.asarray(c)) / 3.0
    v = np.asarray(v, float)
    out = _nan(len(tp))
    if anchor >= len(tp):
        return out
    num = np.cumsum(tp[anchor:] * v[anchor:])
    den = np.cumsum(v[anchor:])
    with np.errstate(divide="ignore", invalid="ignore"):
        out[anchor:] = np.where(den > 0, num / den, NAN)
    return out


def swing_levels(h, l, k=5):
    """Most recent CONFIRMED swing high / low as of each bar. A pivot at j
    (highest high within +-k bars) is only known at j+k -- no lookahead."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    n = len(h)
    hi, lo = _nan(n), _nan(n)
    last_h = last_l = NAN
    for i in range(n):
        j = i - k
        if j - k >= 0:
            win_h = h[j - k:i + 1]
            win_l = l[j - k:i + 1]
            if h[j] == win_h.max():
                last_h = h[j]
            if l[j] == win_l.min():
                last_l = l[j]
        hi[i], lo[i] = last_h, last_l
    return hi, lo


def adx(h, l, c, n=14):
    h, l, c = (np.asarray(a, float) for a in (h, l, c))
    up = np.concatenate([[0.0], np.diff(h)])
    dn = np.concatenate([[0.0], -np.diff(l)])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = wilder(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * wilder(pdm, n) / tr
        mdi = 100 * wilder(mdm, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.where(np.isfinite(dx), dx, NAN)
    return wilder(dx, n), pdi, mdi


def roc(c, n=10):
    c = np.asarray(c, float)
    out = _nan(len(c))
    out[n:] = (c[n:] / c[:-n] - 1) * 100
    return out


def obv(c, v):
    c, v = np.asarray(c, float), np.asarray(v, float)
    sign = np.sign(np.concatenate([[0.0], np.diff(c)]))
    return np.cumsum(sign * v)


def ma_slope(c, n=50, m=20):
    """% change of SMA-n over the last m bars (scale-free trend slope)."""
    s = sma(c, n)
    out = _nan(len(s))
    out[m:] = (s[m:] / s[:-m] - 1) * 100
    return out


# ------------------------------------------------------------- registry ----
# kind: "overlay" (price units, rebased with candles) or "pane" (own scale).
REGISTRY = {
    "sma": {"label": "SMA", "kind": "overlay", "params": {"lengths": "20,50,200"}},
    "ema": {"label": "EMA", "kind": "overlay", "params": {"lengths": "21"}},
    "bb": {"label": "Bollinger Bands", "kind": "overlay", "params": {"n": 20, "k": 2.0}},
    "kc": {"label": "Keltner Channels", "kind": "overlay", "params": {"n": 20, "mult": 2.0}},
    "vwap": {"label": "VWAP (anchored at window start)", "kind": "overlay", "params": {}},
    "rvwap": {"label": "Rolling VWAP", "kind": "overlay", "params": {"n": 20}},
    "donchian": {"label": "Donchian Channel", "kind": "overlay", "params": {"n": 20}},
    "swings": {"label": "Prior swing high / low", "kind": "overlay", "params": {"k": 5}},
    "volume": {"label": "Volume (+MA)", "kind": "pane", "params": {"n": 20}},
    "rsi": {"label": "RSI", "kind": "pane", "params": {"n": 14}},
    "macd": {"label": "MACD", "kind": "pane", "params": {"fast": 12, "slow": 26, "signal": 9}},
    "atr": {"label": "ATR % of price", "kind": "pane", "params": {"n": 14}},
    "slope": {"label": "MA slope", "kind": "pane", "params": {"n": 50, "m": 20}},
    "adx": {"label": "ADX / DI", "kind": "pane", "params": {"n": 14}},
    "roc": {"label": "Rate of change", "kind": "pane", "params": {"n": 10}},
    "obv": {"label": "On-balance volume", "kind": "pane", "params": {}},
}
DEFAULT_SET = [{"id": "sma", "params": {"lengths": "20,50,200"}}, {"id": "volume", "params": {"n": 20}}]
DAILY_FIVE_SET = DEFAULT_SET

_OV_COLORS = ["#ffc857", "#c46b8b", "#2a9bc4", "#7bdcb5", "#ff8c2b", "#b8b8b8"]


def _lengths(s) -> list[int]:
    out = []
    for tok in str(s).replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and 1 <= int(tok) <= 400:
            out.append(int(tok))
    return out[:6]


def params_of(spec: dict) -> dict:
    base = dict(REGISTRY[spec["id"]]["params"])
    base.update(spec.get("params") or {})
    return base


def label_of(spec: dict) -> str:
    p = params_of(spec)
    lab = REGISTRY[spec["id"]]["label"]
    if spec["id"] in ("sma", "ema"):
        return f"{lab} {p['lengths']}"
    return lab + (" (" + ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v)
                                   for v in p.values()) + ")" if p else "")


def compute(specs: list[dict], o, h, l, c, v, anchor: int) -> dict:
    """{"overlays": [{name, color, values, style?}], "panes": [{name, series: [...], levels}]}."""
    overlays, panes = [], []
    ci = 0

    def col():
        nonlocal ci
        ci += 1
        return _OV_COLORS[(ci - 1) % len(_OV_COLORS)]

    for spec in specs or []:
        sid = spec.get("id")
        if sid not in REGISTRY:
            continue
        p = params_of(spec)
        if sid in ("sma", "ema"):
            fn = sma if sid == "sma" else ema
            for n in _lengths(p["lengths"]):
                overlays.append({"name": f"{sid.upper()} {n}", "color": col(), "values": fn(c, n)})
        elif sid in ("bb", "kc"):
            up, mid, lo = (bollinger(c, int(p["n"]), float(p["k"])) if sid == "bb"
                           else keltner(h, l, c, int(p["n"]), float(p["mult"])))
            cc = col()
            nm = "BB" if sid == "bb" else "KC"
            overlays += [{"name": f"{nm} up", "color": cc, "values": up, "style": 2},
                         {"name": f"{nm} mid", "color": cc, "values": mid, "style": 1},
                         {"name": f"{nm} lo", "color": cc, "values": lo, "style": 2}]
        elif sid == "vwap":
            overlays.append({"name": "aVWAP", "color": col(), "values": vwap_anchored(h, l, c, v, anchor)})
        elif sid == "rvwap":
            overlays.append({"name": f"VWAP {int(p['n'])}", "color": col(),
                             "values": vwap_rolling(h, l, c, v, int(p["n"]))})
        elif sid == "donchian":
            hi, lo = donchian(h, l, int(p["n"]))
            cc = col()
            overlays += [{"name": "DC hi", "color": cc, "values": hi, "style": 0},
                         {"name": "DC lo", "color": cc, "values": lo, "style": 0}]
        elif sid == "swings":
            hi, lo = swing_levels(h, l, int(p["k"]))
            overlays += [{"name": "swing hi", "color": "#ff5a3c", "values": hi, "style": 3, "step": True},
                         {"name": "swing lo", "color": "#2ec4b6", "values": lo, "style": 3, "step": True}]
        elif sid == "volume":
            vv = np.asarray(v, float)
            panes.append({"name": "VOL", "series": [
                {"name": "vol", "type": "hist", "values": vv, "updown": True},
                {"name": f"MA {int(p['n'])}", "type": "line", "color": "#ffc857", "values": sma(vv, int(p["n"]))}],
                "levels": [], "volume": True})
        elif sid == "rsi":
            panes.append({"name": f"RSI {int(p['n'])}", "series": [
                {"name": "rsi", "type": "line", "color": "#c46b8b", "values": rsi(c, int(p["n"]))}],
                "levels": [30, 70], "fixed": [0, 100]})
        elif sid == "macd":
            line, sig, hist = macd(c, int(p["fast"]), int(p["slow"]), int(p["signal"]))
            panes.append({"name": "MACD", "series": [
                {"name": "hist", "type": "hist", "values": hist, "signed": True},
                {"name": "macd", "type": "line", "color": "#2a9bc4", "values": line},
                {"name": "signal", "type": "line", "color": "#ffc857", "values": sig}], "levels": [0]})
        elif sid == "atr":
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = atr(h, l, c, int(p["n"])) / np.asarray(c, float) * 100
            panes.append({"name": f"ATR% {int(p['n'])}", "series": [
                {"name": "atr%", "type": "line", "color": "#7bdcb5", "values": vals}], "levels": []})
        elif sid == "slope":
            panes.append({"name": f"SMA{int(p['n'])} slope {int(p['m'])}", "series": [
                {"name": "slope", "type": "hist", "values": ma_slope(c, int(p["n"]), int(p["m"])),
                 "signed": True}], "levels": [0]})
        elif sid == "adx":
            a, pdi, mdi = adx(h, l, c, int(p["n"]))
            panes.append({"name": f"ADX {int(p['n'])}", "series": [
                {"name": "adx", "type": "line", "color": "#ffffff", "values": a},
                {"name": "+DI", "type": "line", "color": "#2ec4b6", "values": pdi},
                {"name": "-DI", "type": "line", "color": "#ff5a3c", "values": mdi}], "levels": [25]})
        elif sid == "roc":
            panes.append({"name": f"ROC {int(p['n'])}", "series": [
                {"name": "roc", "type": "hist", "values": roc(c, int(p["n"])), "signed": True}], "levels": [0]})
        elif sid == "obv":
            panes.append({"name": "OBV", "series": [
                {"name": "obv", "type": "line", "color": "#2a9bc4", "values": obv(c, v)}], "levels": [],
                "volume": True})
    return {"overlays": overlays, "panes": panes[:4]}
