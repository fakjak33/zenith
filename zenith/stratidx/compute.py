"""Nightly STRATEGY INDICES build.

    python -m zenith.stratidx.compute --action auto|force

Cboe benchmark CSVs (free) -> per-index performance, drawdown, Sharpe, trend
signals; family x window "what's working" matrix vs the S&P 500 total return
and vs T-bills; volatility context (VIX term, VVIX, SKEW, implied
correlation, dispersion); S&P 500 breadth (% above 20 / 50 / 200-day SMA,
the S5TW / S5FI / S5TH family, computed from current constituents); and a
verified methodology link for each index.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import load, save
from .fetch import cboe_csv
from .registry import DASHBOARD, FAMILIES, INDICES, VOL_CONTEXT, methodology_url
from .signals import composite, donchian, ema_trend, ewmac, performance, tsmom, xs_ranks

WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(x, n=5):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if math.isfinite(x) else None


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def benchmarks() -> tuple[pd.Series, pd.Series]:
    """S&P 500 total return (Yahoo ^SP500TR) and a T-bill cash index (FRED DTB3)."""
    sptr, cash = pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        from ..cas.sources.prices import get_history
        data, _ = get_history(["^SP500TR"], period="max")
        if "^SP500TR" in data:
            sptr = data["^SP500TR"]["close"].astype(float)
            sptr.index = pd.to_datetime(sptr.index).tz_localize(None)
    except Exception:
        pass
    try:
        from ..cas.sources.fred import get_series
        out, _ = get_series(["DTB3"], limit=None, cache_key="stratidx_fred")
        pts = out.get("DTB3") or []
        if pts:
            y = pd.Series([p["value"] / 100 for p in pts], index=pd.to_datetime([p["date"] for p in pts]))
            idx = pd.bdate_range(y.index[0], pd.Timestamp.today())
            y = y.reindex(idx).ffill()
            cash = (1 + y.shift(1).fillna(0) / 252).cumprod()
    except Exception:
        pass
    return sptr, cash


def window_ret(s: pd.Series, n: int) -> float | None:
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0) if len(s) > n else None


def breadth(period: str = "2y") -> dict:
    """% of S&P 500 members above their 20 / 50 / 200-day SMA, daily."""
    try:
        from ..brief.sources import sp500_constituents
        from ..cas.sources.prices import get_history
        syms = sp500_constituents()
        data, _ = get_history(syms, period=period)
    except Exception as e:
        return {"error": str(e)[:120]}
    closes = pd.DataFrame({t: df["close"] for t, df in data.items() if len(df) > 200})
    if closes.empty:
        return {"error": "no prices"}
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    out = {"n_members": int(closes.shape[1]), "date": [], "s5tw": [], "s5fi": [], "s5th": []}
    pct = {}
    for n, k in ((20, "s5tw"), (50, "s5fi"), (200, "s5th")):
        sma = closes.rolling(n, min_periods=n).mean()
        valid = sma.notna()
        pct[k] = ((closes > sma) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    df = pd.DataFrame(pct).dropna().iloc[-260:]
    out["date"] = [d.date().isoformat() for d in df.index]
    for k in ("s5tw", "s5fi", "s5th"):
        out[k] = [_f(v, 4) for v in df[k]]
    return out


def vol_context() -> dict:
    out = {}
    for n in VOL_CONTEXT:
        s = cboe_csv(n)
        if not len(s):
            continue
        w = s.iloc[-252:]
        out[n] = {"last": _f(s.iloc[-1], 3), "date": s.index[-1].date().isoformat(),
                  "pct1y": _f((w.iloc[:-1] < w.iloc[-1]).mean(), 3), "chg1w": _f(window_ret(s, 5), 4),
                  "spark": [_f(v, 3) for v in s.iloc[-126:].iloc[::3]]}
    return out


def verify_methodology() -> dict:
    out = {}
    try:
        import requests
        from ..config import BROWSER_HEADERS
        sess = requests.Session()
        sess.headers.update({"User-Agent": BROWSER_HEADERS["User-Agent"]})
        for t in INDICES:
            url = methodology_url(t)
            try:
                r = sess.head(url, allow_redirects=True, timeout=20)
                ok = r.status_code == 200 and "pdf" in r.headers.get("content-type", "")
                out[t] = {"url": r.url if ok else url, "ok": ok, "status": r.status_code,
                          "dashboard": DASHBOARD.format(t=t.lower())}
            except Exception as e:
                out[t] = {"url": url, "ok": False, "status": str(e)[:60], "dashboard": DASHBOARD.format(t=t.lower())}
    except Exception:
        pass
    return {"checked": _utc(), "links": out}


def environment(ind: dict, matrix: dict, vc: dict, br: dict) -> list[dict]:
    """Numeric plain-English read of the environment (each line carries its numbers)."""
    lines = []
    def r(t, k):
        return (ind.get(t) or {}).get("perf", {}).get(k)
    put3, cash3, sptr3 = r("PUT", "r3m"), matrix.get("_cash", {}).get("3M"), matrix.get("_sptr", {}).get("3M")
    if put3 is not None and cash3 is not None:
        ok = put3 > cash3
        lines.append({"tone": "pos" if ok else "neg", "text":
                      f"Short volatility {'is' if ok else 'is not'} being paid: PUT {put3 * 100:+.1f}% over 3M vs "
                      f"T-bills {cash3 * 100:+.1f}%."})
    c3 = r("CNDR", "r3m")
    if c3 is not None and put3 is not None:
        lines.append({"tone": "pos" if c3 > put3 else "neu", "text":
                      f"{'Range-bound' if c3 > put3 else 'Directional'} tape: CNDR {c3 * 100:+.1f}% vs PUT "
                      f"{put3 * 100:+.1f}% over 3M ({'both wings short paid' if c3 > put3 else 'the upside leg mattered'})."})
    pp3 = r("PPUT", "r3m")
    if pp3 is not None and sptr3 is not None:
        paid = pp3 > sptr3
        lines.append({"tone": "neg" if paid else "pos", "text":
                      f"Tail protection {'paid' if paid else 'cost'}: PPUT {pp3 * 100:+.1f}% vs S&P 500 TR "
                      f"{sptr3 * 100:+.1f}% over 3M."})
    up = [t for t, v in ind.items() if (v.get("signals") or {}).get("tsmom", {}).get("sign") == 1]
    if ind:
        lines.append({"tone": "pos" if len(up) >= len(ind) / 2 else "neg", "text":
                      f"Trend: {len(up)}/{len(ind)} strategy indices have positive 12-1 momentum."})
    v, v3 = (vc.get("VIX") or {}).get("last"), (vc.get("VIX3M") or {}).get("last")
    if v and v3:
        lines.append({"tone": "pos" if v < v3 else "neg", "text":
                      f"VIX term structure in {'contango' if v < v3 else 'backwardation'}: VIX {v:.1f} / VIX3M "
                      f"{v3:.1f} = {v / v3:.2f}."})
    if br.get("s5fi"):
        x = br["s5fi"][-1]
        prev = br["s5fi"][-6] if len(br["s5fi"]) > 6 else None
        if x is not None:
            lines.append({"tone": "pos" if x >= 0.5 else "neg", "text":
                          f"Breadth: {x * 100:.0f}% of S&P 500 members above their 50-day average"
                          + (f" ({(x - prev) * 100:+.0f} pts in a week)." if prev is not None else ".")})
    sk = vc.get("SKEW") or {}
    if sk.get("pct1y") is not None:
        lines.append({"tone": "neu", "text": f"SKEW {sk['last']:.0f} sits at the {sk['pct1y'] * 100:.0f}th "
                      f"percentile of its year: tail demand {'elevated' if sk['pct1y'] > 0.7 else 'normal'}."})
    return lines


def run(action: str = "auto") -> dict:
    status = load("status", {}) or {}
    today = datetime.now(timezone.utc).date().isoformat()
    if action == "auto" and status.get("run_date") == today and status.get("ok"):
        print("[stratidx] already built today")
        return status
    sptr, cash = benchmarks()
    levels = {t: cboe_csv(t) for t in INDICES}
    levels = {t: s for t, s in levels.items() if len(s) > 60}
    xs = xs_ranks(levels)
    ind = {}
    for t, s in levels.items():
        fam, name, rule, _ = INDICES[t]
        sig = {"tsmom": tsmom(s), "donchian": donchian(s), "ewmac": ewmac(s), "ema": ema_trend(s),
               "xs": xs.get(t, {})}
        sig["composite"] = composite(sig)
        perf = performance(s, cash)
        rel = {}
        if len(sptr):
            for w, n in WINDOWS.items():
                a, b = window_ret(s, n), window_ret(sptr.reindex(s.index).ffill(), n)
                rel[w] = (a - b) if a is not None and b is not None else None
        m = s.resample("ME").last().iloc[-180:]
        ind[t] = {"family": fam, "name": name, "rule": rule, "perf": perf, "signals": sig, "rel_sptr": rel,
                  "spark": [_f(v, 3) for v in s.iloc[-252:].iloc[::5]],
                  "monthly": {"d": [d.date().isoformat() for d in m.index], "v": [_f(v, 4) for v in m]}}
    # family x window matrix: mean member return, minus T-bills and minus SPTR
    matrix = {"_cash": {}, "_sptr": {}}
    for w, n in WINDOWS.items():
        matrix["_cash"][w] = window_ret(cash, n) if len(cash) else None
        matrix["_sptr"][w] = window_ret(sptr, n) if len(sptr) else None
    for fam in FAMILIES:
        mem = [t for t in ind if ind[t]["family"] == fam]
        matrix[fam] = {}
        for w, n in WINDOWS.items():
            rs = [window_ret(levels[t], n) for t in mem]
            rs = [x for x in rs if x is not None]
            avg = float(np.mean(rs)) if rs else None
            matrix[fam][w] = {"ret": avg,
                              "x_cash": (avg - matrix["_cash"][w]) if avg is not None and matrix["_cash"][w] is not None else None,
                              "x_sptr": (avg - matrix["_sptr"][w]) if avg is not None and matrix["_sptr"][w] is not None else None,
                              "members": mem}
    sp = {}
    if len(sptr):
        m = sptr.resample("ME").last().iloc[-180:]
        sp = {"perf": performance(sptr, cash), "monthly": {"d": [d.date().isoformat() for d in m.index],
                                                            "v": [_f(v, 2) for v in m]}}
    vc = vol_context()
    br = breadth()
    meth = verify_methodology()
    as_of = max((v["perf"]["date"] for v in ind.values()), default=today)
    save("indices", _clean({"as_of": as_of, "generated": _utc(), "indices": ind, "sptr": sp}))
    save("regime", _clean({"as_of": as_of, "windows": list(WINDOWS), "families": list(FAMILIES), "matrix": matrix,
                           "environment": environment(ind, matrix, vc, br)}))
    save("vol_context", _clean({"as_of": as_of, "series": vc}))
    save("breadth", _clean({"as_of": as_of, **br}))
    save("methodology", meth)
    st = {"ok": bool(ind), "run_date": today, "run_utc": _utc(), "n_indices": len(ind),
          "missing": [t for t in INDICES if t not in ind], "breadth_members": br.get("n_members"),
          "sptr": bool(len(sptr)), "cash": bool(len(cash))}
    save("status", st)
    print(f"[stratidx] {len(ind)} indices, breadth n={br.get('n_members')}")
    return st


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="zenith.stratidx.compute")
    ap.add_argument("--action", default="auto", choices=["auto", "force"])
    a = ap.parse_args(argv)
    run(a.action)


if __name__ == "__main__":
    main()
