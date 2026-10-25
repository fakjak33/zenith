"""repro:earnings_drift — after an outlier earnings move, do most stocks keep
drifting the same way over the next month? (the user's seed hypothesis)

Event = a yfinance earnings date. Reaction = market-excess return (stock minus
SPY) over the two-day window close(day before) -> close(day after the first
session on/after the report date); two days so pre-market and after-hours
reporters are both inside without trusting timestamps. Drift = excess return
from that close to +5 and +21 trading days later. "Continues" = drift has the
reaction's sign.

Outlier variants (each vs the stock's own prior 8 reactions, >= 6 required):
  A  |reaction| > 2x the mean |reaction| of the prior 8
  B  |z| > 1.5, z = (reaction - mean8) / sd8
  C  |reaction| > 5% (absolute, no history; a baseline definition)
"""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from . import artefact, claim, rnd, verdict_share

HORIZONS = (5, 21)
PERIODS = [("2001-2005", 2001, 2005), ("2006-2015", 2006, 2015), ("2016-2026", 2016, 2026)]
N_PRIOR, MIN_PRIOR = 8, 6


def sample_universe(rows: list[dict], n: int = 400, seed: int = 7) -> list[str]:
    """Stratified sample by current index weight: n/3 names from each weight tercile."""
    rows = sorted([r for r in rows if r.get("ticker")], key=lambda r: -(r.get("weight_pct") or 0.0))
    k = len(rows) // 3
    rng = random.Random(seed)
    out = []
    for i, part in enumerate((rows[:k], rows[k:2 * k], rows[2 * k:])):
        want = n // 3 + (1 if i < n % 3 else 0)
        out += [r["ticker"] for r in rng.sample(part, min(len(part), want))]
    return sorted(set(out))


def ticker_events(px: pd.DataFrame, spy: pd.Series, report_dates: list[str], ticker: str = "") -> list[dict]:
    """Reaction and drift rows for one ticker. px has close, volume. Pure."""
    px = px.dropna(subset=["close"])
    j = pd.concat([px["close"].rename("c"), px["volume"].rename("v"), spy.rename("m")],
                  axis=1, join="inner").dropna(subset=["c", "m"])
    if len(j) < 100:
        return []
    idx, c, m = j.index, j["c"].to_numpy(), j["m"].to_numpy()
    dv = (j["c"] * j["v"]).to_numpy()
    out = []
    for rd in sorted(set(report_dates)):
        p0 = int(idx.searchsorted(pd.Timestamp(rd)))
        t1 = p0 + 1
        if p0 < 61 or t1 + max(HORIZONS) >= len(idx):
            continue
        pre = p0 - 1
        rx = c[t1] / c[pre] - m[t1] / m[pre]
        if not math.isfinite(rx) or abs(rx) > 1.0:
            continue
        row = {"ticker": ticker, "report_date": rd, "year": int(idx[p0].year), "reaction": float(rx),
               "dollar_vol": float(np.nanmean(dv[p0 - 61:p0 - 1]))}
        for h in HORIZONS:
            row[f"drift_{h}"] = float(c[t1 + h] / c[t1] - m[t1 + h] / m[t1])
        out.append(row)
    # own-history baselines from the prior reactions only
    for i, row in enumerate(out):
        prior = [r["reaction"] for r in out[max(0, i - N_PRIOR):i]]
        if len(prior) >= MIN_PRIOR:
            a = np.array(prior)
            row["mean_abs8"], row["mean8"], row["sd8"] = float(np.abs(a).mean()), float(a.mean()), float(a.std(ddof=1))
    return out


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has = df["mean_abs8"].notna() if "mean_abs8" in df else pd.Series(False, index=df.index)
    df["out_A"] = has & (df["reaction"].abs() > 2.0 * df.get("mean_abs8", np.nan))
    z = (df["reaction"] - df.get("mean8", np.nan)) / df.get("sd8", np.nan)
    df["out_B"] = has & (z.abs() > 1.5)
    df["out_C"] = df["reaction"].abs() > 0.05
    df["dir"] = np.where(df["reaction"] > 0, "up", "down")
    df["size"] = df.groupby("year")["dollar_vol"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["small", "mid", "large"]) if len(s) >= 3 else "mid")
    df["period"] = pd.cut(df["year"], [2000, 2005, 2015, 2026], labels=[p[0] for p in PERIODS]).astype(str)
    return df


def cont_stats(sub: pd.DataFrame, h: int) -> dict:
    if sub.empty:
        return {"n": 0}
    sgn = np.sign(sub["reaction"])
    d = sub[f"drift_{h}"]
    cont = (np.sign(d) == sgn)
    n, p = len(sub), float(cont.mean())
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return {"n": int(n), "share_cont": rnd(p, 4), "ci_lo": rnd(p - 1.96 * se, 4),
            "ci_hi": rnd(p + 1.96 * se, 4), "mean_signed_drift": rnd((d * sgn).mean(), 5),
            "median_signed_drift": rnd((d * sgn).median(), 5)}


def summarize(df: pd.DataFrame) -> dict:
    df = classify(df)
    out = {"n_events": int(len(df)), "n_tickers": int(df["ticker"].nunique()),
           "first": df["report_date"].min(), "last": df["report_date"].max(), "variants": {}}
    sets = {"A": df[df["out_A"]], "B": df[df["out_B"]], "C": df[df["out_C"]],
            "all": df, "non_outlier_A": df[df["mean_abs8"].notna() & ~df["out_A"]]}
    for k, sub in sets.items():
        v = {f"h{h}": cont_stats(sub, h) for h in HORIZONS}
        for col in ("dir", "size", "period"):
            v[f"by_{col}"] = {str(g): {f"h{h}": cont_stats(s, h) for h in HORIZONS}
                              for g, s in sub.groupby(col, observed=True)}
        v["by_size_period"] = {f"{p}|{s}": cont_stats(g, 21)
                               for (p, s), g in sub.groupby(["period", "size"], observed=True)}
        out["variants"][k] = v
    return out


def tracker_stats(rows: list[dict]) -> dict:
    """Hit rate of zenith's own PEAD tracker (data/pead/history.json): signals gated on
    SUE and a same-signed reaction; 'continues' = excess return > 0. history.py already
    stores excess_ret side-adjusted (sign * (ret - spy_ret)), so it is used as is."""
    out = {}
    for h in ("h5", "h20"):
        vals = [r["horizons"][h]["excess_ret"] for r in rows if (r.get("horizons", {}).get(h) or {}).get("evaluated")
                and r["horizons"][h].get("excess_ret") is not None]
        if vals:
            a = np.array(vals)
            p = float((a > 0).mean())
            se = math.sqrt(p * (1 - p) / len(a))
            out[h] = {"n": len(a), "share_cont": rnd(p, 4), "ci_lo": rnd(p - 1.96 * se, 4),
                      "ci_hi": rnd(p + 1.96 * se, 4), "mean_signed_drift": rnd(a.mean(), 5)}
    dates = sorted(r["reaction_day"] for r in rows if r.get("reaction_day"))
    out["period"] = f"{dates[0]} to {dates[-1]}" if dates else ""
    return out


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "n/a"
    return (f"{s['share_cont']:.1%} [{s['ci_lo']:.1%}, {s['ci_hi']:.1%}], n={s['n']}, "
            f"mean {s['mean_signed_drift'] * 100:+.2f}%")


def build(res: dict, tracker: dict | None = None) -> dict:
    V = res["variants"]
    a21 = V["A"]["h21"]
    v_main = verdict_share(a21["ci_lo"], a21["ci_hi"])
    lg = V["A"]["by_size_period"]
    post_large = [lg.get(f"{p}|large") for p in ("2006-2015", "2016-2026")]
    pre = V["A"]["by_period"].get("2001-2005", {}).get("h21", {})
    post = {p: V["A"]["by_period"].get(p, {}).get("h21", {}) for p in ("2006-2015", "2016-2026")}
    ours = (f"Variant A (|move| > 2x own mean): {_fmt(a21)} kept drifting at +21 trading days "
            f"(B: {_fmt(V['B']['h21'])}; C: {_fmt(V['C']['h21'])}; all reports: {_fmt(V['all']['h21'])}).")
    ours_p = (f"Variant A at +21d by period: 2001-05 {_fmt(pre)}; 2006-15 {_fmt(post['2006-2015'])}; "
              f"2016-26 {_fmt(post['2016-2026'])}. Large-tercile after 2006: "
              + "; ".join(_fmt(x or {}) for x in post_large) + ".")
    pl = [x for x in post_large if x and x.get("n")]
    v_p = (verdict_share(min(x["ci_lo"] for x in pl), max(x["ci_hi"] for x in pl)) if pl else "mixed")
    claims = [
        claim("After an outlier earnings move, more than half of companies keep drifting in the "
              "same direction over the next month (user seed hypothesis)", "seed", ours, v_main,
              ["lit:martineau-2022", "lit:chan-2003", "lit:bernard-thomas-1989"]),
        claim("Large-cap drift after earnings is about zero since 2006 (lit:martineau-2022), so the "
              "continuation rate for large caps after 2006 should be near 50%", "lit:earnings",
              ours_p, {"agrees": "contradicts", "contradicts": "contradicts", "mixed": "agrees"}[v_p],
              ["lit:martineau-2022"]),
    ]
    if tracker and tracker.get("h20"):
        t = tracker["h20"]
        claims.append(claim(
            "zenith's PEAD tracker picks (SUE + same-signed reaction) keep drifting over ~1 month",
            "zenith:pead", f"{tracker['period']}: +20d {_fmt(t)}; +5d {_fmt(tracker.get('h5', {}))}",
            verdict_share(t["ci_lo"], t["ci_hi"]), ["lit:bernard-thomas-1989"]))
    chart = {"variants": [{"variant": k, "horizon": f"+{h}d", **V[k][f"h{h}"]}
                          for k in ("A", "B", "C", "all") for h in HORIZONS],
             "size_period_A21": [{"period": key.split("|")[0], "size": key.split("|")[1], **v}
                                 for key, v in V["A"]["by_size_period"].items()],
             "dir_A": [{"dir": d, "horizon": f"+{h}d", **x[f"h{h}"]} for d, x in V["A"]["by_dir"].items()
                       for h in HORIZONS]}
    return artefact(
        "earnings_drift", claim_text=claims[0]["text"], claims=claims, our_value=ours,
        data=(f"yfinance earnings dates and adjusted daily closes for {res['n_tickers']} current Russell 1000 "
              "names (stratified by VONE weight), SPY as the market; zenith data/pead/history.json"),
        period=f"report dates {res['first']} to {res['last']}",
        method=("Two-day market-excess reaction around each report; drift = excess return from the "
                "second day's close to +5 / +21 trading days; continues = same sign. Outlier variants "
                "A/B/C as in the module docstring. Size = dollar-volume tercile within each year "
                "(60 days before the report). 95% CI = normal approximation."),
        caveats=["Survivorship: today's Russell 1000 members only; names that fell out or were "
                 "delisted after bad news are missing, which flatters down-move continuation less "
                 "than it flatters up-move continuation. Treat 2001-2005 as the weakest period (and the smallest: outliers need 6 prior reports, so it effectively starts in 2003).",
                 "yfinance earnings dates can be off by a day; the two-day window absorbs most of it.",
                 "Events overlap in calendar time (earnings season clustering), so the CIs are too "
                 "narrow; read differences under ~3 points as noise.",
                 "'Size' is dollar-volume within this large-cap universe, not small caps: no "
                 "Russell 2000 names are sampled.",
                 "Excess return vs SPY, not a size/industry-matched benchmark; no costs.",
                 "zenith's tracker covers only 2024-05 onward and gates on EPS surprise too, so it is "
                 "a different selection, not a test of the outlier rule."],
        verdict=v_main, chart_data=chart, literature=["lit:martineau-2022", "lit:chan-2003"],
        stats=res, tracker=tracker)
