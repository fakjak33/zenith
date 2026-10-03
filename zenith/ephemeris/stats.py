"""Stats engine over a player's guess history (pure pandas/numpy, no scipy).

Definitions -- one place, so the dashboard and the Read report agree:
  hit          the trade made money: trade_ret > 0 (flat counts as a miss)
  base rate    P(UP) for the chart's class x timeframe x horizon, stored per
               guess. The always-long benchmark's expected hit rate on YOUR
               charts is mean(base_rate); edge vs base = hit - that, tested
               with a normal approximation to the Poisson-binomial.
  trend rule   the configurable SMA rule's call on the same chart; compared
               PAIRED, per chart, on return (no-trade = 0 return).
  coin flip    50%, with a Wilson band at your n.
  small n      any cell with n < MIN_N is flagged and never makes a claim.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

MIN_N = 30
Z95 = 1.959964
CONV_PROB = {"Low": 0.55, "Medium": 0.65, "High": 0.75}   # implied P(win) per conviction


# ------------------------------------------------------------ primitives ----
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def norm_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def binom_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p-value (sum of outcomes at most as likely)."""
    if n <= 0:
        return float("nan")
    if n > 1000:                                    # normal approximation
        z = (k - n * p0) / math.sqrt(n * p0 * (1 - p0))
        return min(1.0, 2 * norm_sf(abs(z)))
    pk = [math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)]
    obs = pk[k]
    return min(1.0, sum(x for x in pk if x <= obs * (1 + 1e-9)))


def edge_vs_base(wins: np.ndarray, p: np.ndarray) -> dict:
    """Hit rate vs the per-chart base rates (Poisson-binomial, normal approx)."""
    wins = np.asarray(wins, float)
    p = np.asarray(p, float)
    m = np.isfinite(p)
    wins, p = wins[m], p[m]
    n = len(wins)
    if n == 0:
        return {"n": 0, "edge": float("nan"), "z": float("nan"), "p_value": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "base": float("nan")}
    var = float((p * (1 - p)).sum())
    diff = float(wins.sum() - p.sum())
    z = diff / math.sqrt(var) if var > 0 else float("nan")
    se = math.sqrt(var) / n
    edge = diff / n
    return {"n": n, "edge": edge, "base": float(p.mean()), "z": z,
            "p_value": 2 * norm_sf(abs(z)) if np.isfinite(z) else float("nan"),
            "lo": edge - Z95 * se, "hi": edge + Z95 * se}


def paired_edge(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean of a - b with a t-style CI (normal quantile; fine for n >= 30)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float(d.mean()) if n else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "z": float("nan"), "p_value": float("nan")}
    se = d.std(ddof=1) / math.sqrt(n)
    z = d.mean() / se if se > 0 else float("nan")
    return {"n": n, "mean": float(d.mean()), "lo": float(d.mean() - Z95 * se), "hi": float(d.mean() + Z95 * se),
            "z": float(z), "p_value": 2 * norm_sf(abs(z)) if np.isfinite(z) else float("nan")}


def longest_run(mask) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


# ------------------------------------------------------------ prepare ----
def _json_col(s: pd.Series) -> pd.Series:
    def load(x):
        if isinstance(x, (dict, list)):
            return x
        try:
            return json.loads(x) if x else None
        except Exception:
            return None
    return s.map(load)


def indicator_label(specs) -> str:
    if not specs:
        return "none"
    names = {"sma": "SMA", "ema": "EMA", "bb": "BB", "kc": "KC", "vwap": "aVWAP", "rvwap": "VWAP",
             "donchian": "DC", "swings": "Swings", "volume": "Vol", "rsi": "RSI", "macd": "MACD",
             "atr": "ATR", "slope": "Slope", "adx": "ADX", "roc": "ROC", "obv": "OBV"}
    return "+".join(sorted(names.get(s.get("id"), s.get("id", "?")) for s in specs if isinstance(s, dict)))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Typed, enriched copy of guesses_df output."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    for c in ("market_ret", "trade_ret", "pnl", "stake", "base_rate", "rule_ret", "r_mult", "atr_ret",
              "mfe", "mae", "mfe_full", "mae_full"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    for c in ("win", "direction", "rule_call", "horizon", "candles_held", "ambiguous"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(0).astype(int)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.sort_values(["ts", "id"]).reset_index(drop=True)
    ind = _json_col(d.get("indicators_json", pd.Series([None] * len(d))))
    d["indicator_set"] = ind.map(indicator_label)
    rg = _json_col(d.get("regime_json", pd.Series([None] * len(d)))).map(lambda x: x or {})
    for k in ("trend", "vol", "dist_high", "rsi"):
        d[f"rg_{k}"] = rg.map(lambda x, k=k: x.get(k) or "—")
    d["long"] = d["direction"] > 0
    d["up"] = d["market_ret"] > 0
    d["rule_win"] = (d["rule_call"] * d["market_ret"]) > 0
    d["rule_ret"] = d["rule_ret"].fillna(d["rule_call"] * d["market_ret"])
    d["tfh"] = d["timeframe"].astype(str) + " · " + d["horizon"].astype(str)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["atr_pct"] = np.where((d["atr_ret"].abs() > 0) & d["atr_ret"].notna(), d["trade_ret"] / d["atr_ret"], np.nan)
    return d


def filter_df(d: pd.DataFrame, **f) -> pd.DataFrame:
    if d.empty:
        return d
    m = pd.Series(True, index=d.index)
    for col, key in (("mode", "modes"), ("timeframe", "timeframes"), ("horizon", "horizons"),
                     ("asset_class", "classes"), ("indicator_set", "indicator_sets"),
                     ("conviction", "convictions")):
        vals = f.get(key)
        if vals:
            m &= d[col].isin(list(vals))
    if f.get("start") is not None:
        m &= d["ts"] >= pd.Timestamp(f["start"], tz="UTC")
    if f.get("end") is not None:
        m &= d["ts"] < pd.Timestamp(f["end"], tz="UTC") + pd.Timedelta(days=1)
    return d[m]


# ------------------------------------------------------------ headline ----
def headline(d: pd.DataFrame) -> dict:
    n = len(d)
    if n == 0:
        return {"n": 0}
    k = int(d["win"].sum())
    p, lo, hi = wilson(k, n)
    wins, losses = d.loc[d["pnl"] > 0, "pnl"], d.loc[d["pnl"] < 0, "pnl"]
    rule_trades = d[d["rule_call"] != 0]
    r = d["r_mult"].dropna()
    return {
        "n": n, "wins": k, "hit": p, "hit_lo": lo, "hit_hi": hi,
        "coin": wilson(n // 2, n)[1:] if n else (np.nan, np.nan),
        "coin_p": binom_two_sided(k, n, 0.5),
        "base": edge_vs_base(d["win"].to_numpy(), d["base_rate"].to_numpy()),
        "always_long_hit": float(d["up"].mean()),
        "rule": paired_edge(d["trade_ret"].to_numpy(), d["rule_ret"].to_numpy()),
        "rule_hit": float(rule_trades["rule_win"].mean()) if len(rule_trades) else float("nan"),
        "rule_n": len(rule_trades),
        "expectancy": float(d["pnl"].mean()), "avg_ret": float(d["trade_ret"].mean()),
        "total_pnl": float(d["pnl"].sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf"),
        "avg_r": float(r.mean()) if len(r) else float("nan"), "n_r": len(r),
        "win_loss": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan"),
        "longest_win": longest_run(d["win"].astype(bool)), "longest_loss": longest_run(~d["win"].astype(bool)),
    }


# ------------------------------------------------------------ equity ----
def equity_curve(d: pd.DataFrame, resets: list[dict] | None = None, start: float = 10_000.0) -> pd.DataFrame:
    """Per-trade equity with resets (balance returns to `start`, history kept)."""
    if d.empty:
        return pd.DataFrame(columns=["i", "ts", "equity", "peak", "drawdown", "dd_pct", "reset"])
    cuts = sorted(pd.Timestamp(r["ts"]) if pd.Timestamp(r["ts"]).tzinfo else
                  pd.Timestamp(r["ts"], tz="UTC") for r in (resets or []))
    eq, peak, rows, ci = start, start, [], 0
    for i, row in enumerate(d.itertuples()):
        reset = False
        while ci < len(cuts) and row.ts > cuts[ci]:
            eq = peak = start
            reset = True
            ci += 1
        eq += float(row.pnl or 0)
        peak = max(peak, eq)
        rows.append((i + 1, row.ts, eq, peak, eq - peak, (eq / peak - 1) if peak else 0.0, reset))
    return pd.DataFrame(rows, columns=["i", "ts", "equity", "peak", "drawdown", "dd_pct", "reset"])


def drawdown_stats(eq: pd.DataFrame) -> dict:
    if eq.empty:
        return {"max_dd": 0.0, "max_dd_pct": 0.0, "longest_dd": 0}
    under = eq["drawdown"] < -1e-9
    return {"max_dd": float(eq["drawdown"].min()), "max_dd_pct": float(eq["dd_pct"].min()),
            "longest_dd": longest_run(under)}


# ------------------------------------------------------------ breakdowns ----
def breakdown(d: pd.DataFrame, by) -> pd.DataFrame:
    """n, hit (+Wilson), base, edge vs base (+CI, z), expectancy, rule edge, small-n flag."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    if d.empty:
        return pd.DataFrame()
    for key, g in d.groupby(by, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        k = int(g["win"].sum())
        p, lo, hi = wilson(k, n)
        eb = edge_vs_base(g["win"].to_numpy(), g["base_rate"].to_numpy())
        pr = paired_edge(g["trade_ret"].to_numpy(), g["rule_ret"].to_numpy())
        rows.append({**dict(zip(by, key)), "n": n, "hit": p, "hit_lo": lo, "hit_hi": hi,
                     "base": eb["base"], "edge": eb["edge"], "edge_lo": eb["lo"], "edge_hi": eb["hi"],
                     "z": eb["z"], "exp_ret": float(g["trade_ret"].mean()), "exp_pnl": float(g["pnl"].mean()),
                     "rule_edge": pr["mean"], "long_share": float(g["long"].mean()),
                     "small_n": n < MIN_N})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def long_short(d: pd.DataFrame) -> dict:
    if d.empty:
        return {}
    out = {}
    for side, g in (("long", d[d["long"]]), ("short", d[~d["long"]])):
        n = len(g)
        out[side] = {"n": n, "hit": wilson(int(g["win"].sum()), n) if n else (np.nan,) * 3,
                     "exp_ret": float(g["trade_ret"].mean()) if n else float("nan")}
    out["long_share"] = float(d["long"].mean())
    out["base"] = float(d["base_rate"].mean())
    out["bias"] = out["long_share"] - out["base"]     # >0: you lean long more than the tape does
    return out


# ------------------------------------------------------------ calibration ----
def calibration(d: pd.DataFrame, probs: dict | None = None) -> dict:
    probs = probs or CONV_PROB
    if d.empty:
        return {"table": pd.DataFrame(), "brier": float("nan"), "brier_coin": 0.25, "monotonic": None}
    rows = []
    for c in ("Low", "Medium", "High"):
        g = d[d["conviction"] == c]
        n = len(g)
        p, lo, hi = wilson(int(g["win"].sum()), n) if n else (np.nan,) * 3
        rows.append({"conviction": c, "n": n, "hit": p, "hit_lo": lo, "hit_hi": hi,
                     "implied": probs[c], "small_n": n < MIN_N})
    t = pd.DataFrame(rows)
    imp = d["conviction"].map(probs)
    m = imp.notna()
    brier = float(((imp[m] - d.loc[m, "win"]) ** 2).mean()) if m.any() else float("nan")
    ok = t[~t["small_n"]]
    mono = None if len(ok) < 2 else bool(ok["hit"].is_monotonic_increasing)
    return {"table": t, "brier": brier, "brier_coin": 0.25, "monotonic": mono}


# ------------------------------------------------------------ stops ----
def stops_analysis(d: pd.DataFrame) -> dict:
    out: dict = {}
    if d.empty:
        return out
    stopped = d[d["exit_reason"] == "stop"]
    if len(stopped):
        rev = (stopped["direction"] * stopped["market_ret"]) > 0
        out["stopped_n"] = len(stopped)
        out["stopped_then_reversed"] = float(rev.mean())
    mf = d["mfe_full"].dropna()
    if len(mf):
        out["avg_mfe"] = float(mf.mean())
        out["avg_captured"] = float(d.loc[mf.index, "trade_ret"].mean())
        out["left_on_table"] = out["avg_mfe"] - out["avg_captured"]
    out["ambiguous_n"] = int(d["ambiguous"].sum())
    out["hindsight"] = hindsight_stop(d)
    return out


def hindsight_stop(d: pd.DataFrame, multiples=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, None)) -> pd.DataFrame:
    """HINDSIGHT: replay every call with a k x ATR stop and no target. A stop
    is assumed hit when the horizon's max adverse excursion reached it (the
    order of MFE vs MAE within the horizon is unknown -- an approximation)."""
    g = d.dropna(subset=["atr_pct", "mae_full", "market_ret"])
    g = g[g["atr_pct"] > 0]
    if g.empty:
        return pd.DataFrame()
    held = g["direction"] * g["market_ret"]
    rows = []
    for k in multiples:
        if k is None:
            r = held
        else:
            hit = (-g["mae_full"]) >= k * g["atr_pct"]
            r = np.where(hit, -k * g["atr_pct"], held)
        r = pd.Series(r, index=g.index)
        rows.append({"stop": "none" if k is None else f"{k:g}×ATR", "n": len(g), "exp_ret": float(r.mean()),
                     "hit": float((r > 0).mean()),
                     "stopped": 0.0 if k is None else float(((-g["mae_full"]) >= k * g["atr_pct"]).mean())})
    t = pd.DataFrame(rows)
    t["best"] = t["exp_ret"] == t["exp_ret"].max()
    return t
