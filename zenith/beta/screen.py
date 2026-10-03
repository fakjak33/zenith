"""CLEAN BETA screen: per-ticker metrics -> filters -> Quality-Beta Score.

Pure functions over price frames; compute.py does the I/O. Every filter is
reported pass/fail per row so the table can say WHY a name is out.
"""

from __future__ import annotations

import pandas as pd

from ..config import (BETA_BSWA_DELTA, BETA_BSWA_LAMBDA, BETA_IPO_MIN_BARS,
                      BETA_IVOL_MAX_PCT, BETA_JUMP_SIGMA, BETA_LOOKBACK, BETA_MANUAL_EVENT_FLAGS,
                      BETA_MAX_JUMP_DAYS, BETA_MIN_ADV_USD, BETA_MIN_BARS, BETA_MIN_MKTCAP,
                      BETA_MIN_PRICE, BETA_PCT_ENTER, BETA_PCT_EXIT, BETA_QUALITY_MIN_PCT,
                      BETA_RHO_MIN, BETA_SCORE_WEIGHTS, BETA_SIZE_BUCKETS)
from . import quality as q
from . import r
from .estimators import beta_se2, bswa, ols_stats, pct_rank, vasicek

FILTERS = ("f_beta", "f_rho", "f_ivol", "f_events", "f_quality")
FILTER_LABELS = {"f_liquid": "Liquid", "f_beta": "β pct", "f_rho": "ρ floor", "f_ivol": "IVOL",
                 "f_events": "Events", "f_quality": "Quality"}


def size_bucket(mktcap: float | None) -> str:
    if mktcap is None:
        return "—"
    for name, floor in BETA_SIZE_BUCKETS:
        if mktcap >= floor:
            return name
    return BETA_SIZE_BUCKETS[-1][0]


def returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def ticker_metrics(df: pd.DataFrame, bench_ret: pd.Series) -> dict:
    """Price-derived metrics for one name. `df` has close (and volume)."""
    close = df["close"].dropna()
    bars = len(close)
    out = {"bars": bars, "price": r(close.iloc[-1], 2) if bars else None}
    vol = df["volume"] if "volume" in df else None
    if vol is not None and bars:
        dv = (df["close"] * df["volume"]).tail(63)
        out["adv"] = r(dv.mean(), 0)
    if bars < BETA_MIN_BARS:
        return out
    ri = close.pct_change()
    j = pd.concat([ri, bench_ret], axis=1, join="inner").dropna()
    j.columns = ["ri", "rm"]
    if len(j) < BETA_MIN_BARS:
        return out
    out["bswa"] = bswa(j["ri"].values, j["rm"].values, BETA_BSWA_DELTA, BETA_BSWA_LAMBDA)
    last = j.tail(BETA_LOOKBACK)
    st = ols_stats(last["ri"].values, last["rm"].values, BETA_JUMP_SIGMA)
    if st:
        out.update({"ols": st["beta"], "rho": st["rho"], "r2": st["r2"], "ivol": st["ivol"],
                    "vol": st["vol"], "jumps": st["jumps"]})
        out["_se2"] = beta_se2(last["ri"].values, last["rm"].values)
    if bars >= 253:
        out["mom"] = float(close.iloc[-22] / close.iloc[-253] - 1.0)
    return out


def build_rows(members: list[dict], px: dict, bench_ret: pd.Series, fundamentals: dict,
               earnings_soon: dict[str, str], shares_doc: dict, incumbents: set[str],
               today, optionable: dict[str, bool] | None = None) -> list[dict]:
    """Every member -> one row with metrics, filter flags and score."""
    optionable = optionable or {}
    rows = []
    for m in members:
        t = m["ticker"]
        row = {"ticker": t, "name": m.get("name") or t, "sector": m.get("sector") or "—",
               "industry": m.get("industry") or ""}
        df = px.get(t)
        if df is None or df.empty or "close" not in df:
            row.update({"f_liquid": False, "excluded_reason": "no price data"})
            rows.append(row)
            continue
        met = ticker_metrics(df, bench_ret)
        row.update(met)
        info = fundamentals.get(t) or {}
        mc = m.get("mktcap") or info.get("marketCap")
        row["mktcap"] = float(mc) if mc else None
        row["size"] = size_bucket(row["mktcap"])
        reasons = []
        if row.get("bswa") is None or row.get("rho") is None:
            reasons.append(f"history < {BETA_MIN_BARS} bars")
        if (row.get("price") or 0) < BETA_MIN_PRICE:
            reasons.append(f"price < ${BETA_MIN_PRICE:g}")
        if row["mktcap"] is not None and row["mktcap"] < BETA_MIN_MKTCAP:
            reasons.append(f"mkt cap < ${BETA_MIN_MKTCAP / 1e9:g}B")
        if (row.get("adv") or 0) < BETA_MIN_ADV_USD:
            reasons.append(f"ADV < ${BETA_MIN_ADV_USD / 1e6:g}M")
        if optionable.get(t) is False:
            reasons.append("no listed options")
        row["f_liquid"] = not reasons
        row["excluded_reason"] = "; ".join(reasons) or None
        row["earnings"] = earnings_soon.get(t)
        row["manual_flag"] = BETA_MANUAL_EVENT_FLAGS.get(t)
        row["ipo"] = met.get("bars", 0) < BETA_IPO_MIN_BARS
        row["_raw_q"] = q.raw_components(info, met.get("mom"), q.issuance(shares_doc, t, today))
        row["incumbent"] = t in incumbents
        rows.append(row)
    _cross_section(rows)
    for row in rows:
        row.pop("_raw_q", None)
        row.pop("_se2", None)
        for k in ("bswa", "ols", "vasicek", "rho", "r2", "ivol", "vol", "mom"):
            if k in row:
                row[k] = r(row[k], 4)
    rows.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or 0.0)))
    for i, row in enumerate(rows):
        row["rank"] = i + 1 if row.get("score") is not None else None
    return rows


def _cross_section(rows: list[dict]) -> None:
    liq = [x for x in rows if x.get("f_liquid")]
    if not liq:
        return
    betas = {x["ticker"]: x["ols"] for x in liq if x.get("ols") is not None}
    ses = {x["ticker"]: x.get("_se2") for x in liq}
    vas = vasicek(betas, ses)
    for x in liq:
        x["vasicek"] = vas.get(x["ticker"])
    for key, col in (("bswa", "beta_pct"), ("rho", "rho_pct"), ("ivol", "ivol_pct")):
        for x, p in zip(liq, pct_rank([x.get(key) for x in liq])):
            x[col] = p
    qs = q.scores({x["ticker"]: x["_raw_q"] for x in liq})
    w = BETA_SCORE_WEIGHTS
    for x in liq:
        x.update(qs[x["ticker"]])
        bp, ip = x.get("beta_pct") or 0.0, x.get("ivol_pct")
        x["f_beta"] = bp >= BETA_PCT_ENTER or (x["incumbent"] and bp >= BETA_PCT_EXIT)
        x["f_rho"] = (x.get("rho") or 0.0) >= BETA_RHO_MIN
        x["f_ivol"] = ip is not None and ip < BETA_IVOL_MAX_PCT
        ev_reasons = []
        if (x.get("jumps") or 0) > BETA_MAX_JUMP_DAYS:
            ev_reasons.append(f"{x['jumps']} event days")
        if x.get("earnings"):
            ev_reasons.append(f"earnings {x['earnings']}")
        if x.get("ipo"):
            ev_reasons.append("recent IPO")
        if x.get("manual_flag"):
            ev_reasons.append(x["manual_flag"])
        x["f_events"] = not ev_reasons
        x["f_quality"] = x["quality_imputed"] or x["quality"] >= BETA_QUALITY_MIN_PCT
        x["event_score"] = (0.0 if (x.get("ipo") or x.get("manual_flag"))
                            else max(0.0, 100.0 - 25.0 * (x.get("jumps") or 0)))
        comp = {"beta": bp, "rho": x.get("rho_pct") or 0.0,
                "ivol": 100.0 - (ip if ip is not None else 100.0),
                "quality": x["quality"], "events": x["event_score"]}
        x["score"] = round(sum(w[k] * comp[k] for k in w) / sum(w.values()), 1)
        x["passes"] = all(x[f] for f in FILTERS)
        fails = []
        if not x["f_beta"]:
            fails.append(f"β pct {bp:.0f} < {BETA_PCT_ENTER:g}"
                         + (f" (incumbent exit {BETA_PCT_EXIT:g})" if x["incumbent"] else ""))
        if not x["f_rho"]:
            fails.append(f"ρ {x.get('rho') or 0:.2f} < {BETA_RHO_MIN}")
        if not x["f_ivol"]:
            fails.append("IVOL top tercile")
        fails += ev_reasons
        if not x["f_quality"]:
            fails.append(f"quality {x['quality']:.0f} (worst quintile)")
        x["fail_reasons"] = "; ".join(fails) or None


def earnings_flags(calendar_rows: list[dict]) -> dict[str, str]:
    """Nasdaq upcoming-calendar rows -> {ticker: earliest report date}."""
    out: dict[str, str] = {}
    for rep in calendar_rows:
        t, d = rep.get("ticker"), rep.get("report_date")
        if t and d and (t not in out or d < out[t]):
            out[t] = d
    return out


def returns_panel(px: dict, tickers: list[str], n: int) -> pd.DataFrame:
    """Aligned daily returns (last n rows) for a set of tickers."""
    cols = {t: px[t]["close"].pct_change() for t in tickers if t in px and px[t] is not None}
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).tail(n)

