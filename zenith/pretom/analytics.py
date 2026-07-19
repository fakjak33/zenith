"""PRETOM ranking + window statistics (pure functions of prices/dates).

Ranking follows the paper's dispensability logic: distance below the 52-week
high is the dominant sort (stronger than 12-2 momentum), liquidity ranks a
name UP (the selling pressure concentrates in liquid losers where institutions
trade), index weight proxies prevalence across funds, and non-dividend-payers
get a small bonus (less foregone income when liquidated).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Composite score weights (percentile ranks oriented "higher = better short").
W_HIGH, W_ADV, W_WEIGHT, NONPAYER_BONUS = 0.50, 0.30, 0.20, 0.05
MIN_BARS = 200          # eligibility: enough history for a meaningful 52w high


def _ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def _asof_close(close: pd.Series, d: date) -> float | None:
    s = close.loc[:_ts(d)].dropna()
    return float(s.iloc[-1]) if len(s) else None


def metrics_asof(df: pd.DataFrame, lock_date: date) -> dict | None:
    """Ranking metrics for one stock using data up to and including lock_date."""
    sub = df.loc[:_ts(lock_date)].dropna(subset=["close"])
    close = sub["close"]
    if len(close) < MIN_BARS:
        return None
    last = float(close.iloc[-1])
    hi52 = float(close.tail(252).max())
    mom = (float(close.iloc[-21] / close.iloc[-252] - 1.0)
           if len(close) >= 252 else None)
    adv = float((close * sub["volume"]).tail(63).mean())
    return {"pct_below_high": 1.0 - last / hi52 if hi52 > 0 else None,
            "mom_12_2": mom, "adv_dollar": adv, "px_lock": last}


def build_basket(px: dict[str, pd.DataFrame], universe: list[dict],
                 lock_date: date, div_flags: dict | None = None) -> list[dict]:
    """Rank the universe by distance below the 52-week high; return the bottom
    decile (the names furthest below), composite-scored and ranked."""
    div_flags = div_flags or {}
    rows = []
    for u in universe:
        df = px.get(u["ticker"])
        if df is None or df.empty:
            continue
        m = metrics_asof(df, lock_date)
        if m is None or m["pct_below_high"] is None:
            continue
        rows.append({**u, **m, "div_payer": div_flags.get(u["ticker"])})
    if not rows:
        return []
    rows.sort(key=lambda r: r["pct_below_high"], reverse=True)
    decile = rows[:max(1, round(len(rows) / 10))]

    d = pd.DataFrame(decile)
    r_high = d["pct_below_high"].rank(pct=True)
    r_adv = d["adv_dollar"].rank(pct=True)
    w = pd.to_numeric(d["weight_pct"], errors="coerce")
    r_w = w.rank(pct=True).fillna(0.5)      # neutral when weights unavailable
    score = W_HIGH * r_high + W_ADV * r_adv + W_WEIGHT * r_w
    score = score + d["div_payer"].apply(
        lambda v: NONPAYER_BONUS if v is False else 0.0)
    d["score"] = score.clip(0.0, 1.0).round(4)
    d = d.sort_values(["score", "pct_below_high"], ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1

    keep = ["rank", "ticker", "name", "sector", "score", "pct_below_high",
            "mom_12_2", "adv_dollar", "weight_pct", "div_payer", "px_lock"]
    out = d[keep].to_dict("records")
    for r in out:                            # JSON-safe: NaN -> None
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                r[k] = None
    return out


def _cum_path(close: pd.Series, start: date, end: date) -> pd.Series | None:
    """Cumulative return over trading days in [start, end], measured from the
    last close strictly before start (returns earned *during* the window)."""
    close = close.dropna()
    base = close.loc[:_ts(start) - pd.Timedelta(days=1)]
    win = close.loc[_ts(start):_ts(end)]
    if not len(base) or not len(win):
        return None
    return win / float(base.iloc[-1]) - 1.0


def window_stats(names: list[dict], px: dict[str, pd.DataFrame],
                 start: date, end: date, spy: pd.Series) -> dict:
    """EW + cap-weight basket return over [start, end], vs SPY, with per-day
    path and the share of names negative at the end / at any close."""
    paths, weights = {}, {}
    for n in names:
        df = px.get(n["ticker"])
        if df is None or df.empty:
            continue
        p = _cum_path(df["close"], start, end)
        if p is not None and len(p):
            paths[n["ticker"]] = p
            weights[n["ticker"]] = n.get("weight_pct") or 0.0
    spy_path = _cum_path(spy, start, end)
    if not paths or spy_path is None or not len(spy_path):
        return {"n_priced": 0}

    mat = pd.DataFrame(paths).ffill()
    w = pd.Series(weights).reindex(mat.columns).astype(float)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(mat.columns), index=mat.columns)
    ew, cw = mat.mean(axis=1), (mat * w).sum(axis=1)
    spy_path = spy_path.reindex(mat.index).ffill()

    ends, mins = mat.iloc[-1], mat.min()
    daily = [{"d": d.date().isoformat(), "ew": round(float(ew[d]), 6),
              "cw": round(float(cw[d]), 6),
              "spy": round(float(spy_path[d]), 6) if pd.notna(spy_path[d]) else None}
             for d in mat.index]
    ew_ret, cw_ret = float(ew.iloc[-1]), float(cw.iloc[-1])
    spy_ret = float(spy_path.iloc[-1]) if pd.notna(spy_path.iloc[-1]) else None
    return {
        "n_priced": len(mat.columns),
        "ew_ret": round(ew_ret, 6), "cw_ret": round(cw_ret, 6),
        "spy_ret": round(spy_ret, 6) if spy_ret is not None else None,
        "ew_excess": round(ew_ret - spy_ret, 6) if spy_ret is not None else None,
        "cw_excess": round(cw_ret - spy_ret, 6) if spy_ret is not None else None,
        "pct_negative_end": round(float((ends < 0).mean()), 4),
        "pct_negative_any": round(float((mins < 0).mean()), 4),
        "daily": daily,
    }


def post_stats(names: list[dict], px: dict[str, pd.DataFrame],
               post_start: date, post_end: date, spy: pd.Series,
               classic_ew_excess: float | None) -> dict:
    """Post-window [tau-3, tau+3] rebound; the paper finds ~70% of the PreTOM
    loser hit reverses here (reversal_ratio ~= 0.7)."""
    s = window_stats(names, px, post_start, post_end, spy)
    if not s.get("n_priced"):
        return {"n_priced": 0}
    out = {k: s.get(k) for k in ("n_priced", "ew_ret", "cw_ret", "spy_ret",
                                 "ew_excess", "cw_excess")}
    ex = s.get("ew_excess")
    out["reversal_ratio"] = (round(-ex / classic_ew_excess, 4)
                             if ex is not None and classic_ew_excess
                             and abs(classic_ew_excess) > 1e-6 else None)
    return out


def per_name_marks(names: list[dict], px: dict[str, pd.DataFrame],
                   sched: dict) -> dict:
    """Per-ticker window returns: classic [tau-9,tau-4], t1 span [tau-9,tau-3],
    worst mark inside the span, and the post-window [tau-3,tau+3] rebound."""
    out = {}
    for n in names:
        df = px.get(n["ticker"])
        if df is None or df.empty:
            continue
        close = df["close"]
        span = _cum_path(close, sched["win_start"], sched["win_end_t1"])
        post = _cum_path(close, sched["post_start"], sched["post_end"])
        if span is None or not len(span):
            continue
        classic = span.loc[:_ts(sched["win_end_classic"])]
        out[n["ticker"]] = {
            "classic_ret": round(float(classic.iloc[-1]), 6) if len(classic) else None,
            "t1_ret": round(float(span.iloc[-1]), 6),
            "min_ret_in_window": round(float(span.min()), 6),
            "post_ret": round(float(post.iloc[-1]), 6) if post is not None and len(post) else None,
        }
    return out


def tom_market_stats(spy: pd.Series, sched: dict) -> dict:
    """Market-level turn-of-the-month window [tau, tau+3]: SPY cumulative
    return measured from the tau-1 close (Lakonishok & Smidt 1988 window as
    dated by Xu & McConnell 2008)."""
    p = _cum_path(spy, sched["tom_start"], sched["tom_end"])
    if p is None or not len(p):
        return {"n_days": 0}
    return {
        "n_days": int(len(p)),
        "spy_ret": round(float(p.iloc[-1]), 6),
        "daily": [{"d": d.date().isoformat(), "spy": round(float(v), 6)}
                  for d, v in p.items()],
    }


def tom_longrun(spy: pd.Series) -> list[dict]:
    """Per-month TOM decomposition over the full SPY history.

    Sessions come from the price index itself (real trading days, so ad-hoc
    closures can't drift the labels). For each month m the TOM window is the
    last session of m plus the first TOM_END sessions of m+1; every other
    session of calendar month m is "rest of month". rest_ret is the rest-days
    mean daily return compounded over the TOM window length, so the two
    columns compare like-for-like 4-day returns.
    """
    from . import calendar as cal

    close = spy.dropna()
    if len(close) < 40:
        return []
    ret = close.pct_change()
    months = pd.PeriodIndex(close.index, freq="M")
    uniq = list(months.unique())

    # Map month -> its sessions (in order), from the actual index.
    by_month = {m: close.index[months == m] for m in uniq}

    rows: list[dict] = []
    for i, m in enumerate(uniq[:-1]):
        nxt = uniq[i + 1]
        nxt_days = by_month[nxt]
        if len(nxt_days) < cal.TOM_END:
            continue                      # next month incomplete: skip
        tom_days = [by_month[m][-1]] + list(nxt_days[:cal.TOM_END])
        # Rest of month m = its sessions minus the ones inside ANY TOM window
        # (its own tau, and the first TOM_END sessions that belong to m-1).
        rest_days = [d for d in by_month[m][:-1]
                     if d not in set(by_month[m][:cal.TOM_END])]
        tom_r = ret.loc[tom_days].dropna()
        rest_r = ret.loc[rest_days].dropna()
        if len(tom_r) < cal.TOM_END + 1 or len(rest_r) < 5:
            continue                      # needs the full window + a real rest
        tom_ret = float((1.0 + tom_r).prod() - 1.0)
        rest_ret = float((1.0 + rest_r.mean()) ** len(tom_r) - 1.0)
        rows.append({
            "month": str(m),
            "tom_ret": round(tom_ret, 6),
            "rest_ret": round(rest_ret, 6),
            "n_tom_days": int(len(tom_r)),
            "n_rest_days": int(len(rest_r)),
        })
    return rows


def _tom_bucket(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    tom = np.array([r["tom_ret"] for r in rows], dtype=float)
    rest = np.array([r["rest_ret"] for r in rows], dtype=float)
    diff = tom - rest
    n = len(diff)
    t_stat = (float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
              if n > 2 and diff.std(ddof=1) > 0 else None)
    return {
        "n_months": n,
        "tom_avg": round(float(tom.mean()), 6),
        "rest_avg": round(float(rest.mean()), 6),
        "tom_hit": round(float((tom > 0).mean()), 4),
        "tom_minus_rest_avg": round(float(diff.mean()), 6),
        "t_stat": round(t_stat, 2) if t_stat is not None else None,
    }


def summarize_tom(rows: list[dict]) -> dict:
    """Aggregate the per-month TOM decomposition: overall, by decade, and two
    recency slices. t_stat is a simple paired t on (tom_ret - rest_ret)."""
    if not rows:
        return {"overall": None, "by_decade": {}, "since_2010": None,
                "trailing_24m": None}
    decades: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{(int(r['month'][:4]) // 10) * 10}s"
        decades.setdefault(key, []).append(r)
    return {
        "overall": _tom_bucket(rows),
        "by_decade": {k: _tom_bucket(v) for k, v in sorted(decades.items())},
        "since_2010": _tom_bucket([r for r in rows if r["month"] >= "2010"]),
        "trailing_24m": _tom_bucket(rows[-24:]),
    }


def build_panel(px: dict[str, pd.DataFrame], tickers: list[str],
                start: date, end: date) -> dict:
    """Compact committed close panel for [start, end] (powers the deployed
    per-name heatmap; the raw price cache is gitignored)."""
    panel = {}
    for t in tickers:
        df = px.get(t)
        if df is None or df.empty:
            continue
        s = df["close"].loc[_ts(start):_ts(end)].dropna()
        if len(s):
            panel[t] = {"d": [d.date().isoformat() for d in s.index],
                        "c": [round(float(v), 4) for v in s]}
    return panel
