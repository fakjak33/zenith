"""repro:cboe_short_vol and repro:cboe_tail_hedge — Cboe strategy benchmark
indices vs the S&P 500 total return index, re-checking Phase 03's computed_stats.

Conventions (same as Phase 03): CAGR from daily closes over calendar time; vol,
Sharpe and beta from month-end returns, Sharpe vs ^IRX (13-week T-bill, prior
month-end yield / 12); max drawdown on daily closes; gross of fees and slippage.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import artefact, claim, rnd

SHORT_VOL = ["PUT", "BXM", "BXMD", "WPUT", "CNDR", "BFLY"]
TAIL = ["PPUT", "CLL"]
BENCH = "SPTR"
END = "2026-10-23"          # Phase 03's data end; fixed so the re-check is like for like
WINDOWS = {"1988": "1988-01-04", "2007": "2007-01-03", "2009": "2009-03-09",
           "2016": "2016-12-30"}
STRESS = {"gfc": ("2007-10-01", "2009-06-30"), "feb2018": ("2018-01-20", "2018-04-30"),
          "covid": ("2020-02-15", "2020-04-30"), "y2022": ("2022-01-01", "2022-12-31")}
YEARS = [2008, 2018, 2020, 2022]
# Cboe's PUT file is sparse before 2007-01-03 (Phase 03 note); use it from there on.
FIRST_GOOD = {"PUT": "2007-01-03"}

# Phase 03 computed_stats, transcribed from data/vault/literature.json
# (topics option-selling, tail-hedging, sizing-risk). (stat, window/episode, index): value.
PHASE03 = {
    ("cagr", "2007", "PUT"): 7.24, ("cagr", "2007", "SPTR"): 10.95, ("sharpe", "2007", "PUT"): 0.56,
    ("sharpe", "2007", "SPTR"): 0.65, ("maxdd", "2007", "PUT"): -37.1, ("maxdd", "2007", "SPTR"): -55.3,
    ("beta", "2007", "PUT"): 0.60,
    ("cagr", "2007", "BXM"): 6.03, ("sharpe", "2007", "BXM"): 0.44, ("beta", "2007", "BXM"): 0.62,
    ("cagr", "2007", "BXMD"): 8.44, ("sharpe", "2007", "BXMD"): 0.57, ("beta", "2007", "BXMD"): 0.80,
    ("cagr", "2007", "WPUT"): 4.38, ("sharpe", "2007", "WPUT"): 0.33, ("beta", "2007", "WPUT"): 0.53,
    ("cagr", "2007", "CNDR"): 0.98, ("sharpe", "2007", "CNDR"): -0.06, ("beta", "2007", "CNDR"): 0.10,
    ("cagr", "2007", "BFLY"): -1.99, ("sharpe", "2007", "BFLY"): -0.27,
    ("cagr", "2009", "SPTR"): 16.88, ("sharpe", "2009", "SPTR"): 1.00,
    ("cagr", "2009", "PUT"): 10.37, ("sharpe", "2009", "PUT"): 0.85,
    ("cagr", "2009", "BXM"): 9.51, ("sharpe", "2009", "BXM"): 0.75,
    ("cagr", "2009", "BXMD"): 13.00, ("sharpe", "2009", "BXMD"): 0.91,
    ("cagr", "2009", "WPUT"): 6.05, ("sharpe", "2009", "WPUT"): 0.52,
    ("cagr", "2009", "CNDR"): 0.99, ("sharpe", "2009", "CNDR"): -0.06,
    ("cagr", "2016", "SPTR"): 15.17, ("sharpe", "2016", "SPTR"): 0.83,
    ("cagr", "2016", "PUT"): 8.42, ("sharpe", "2016", "PUT"): 0.62,
    ("cagr", "2016", "BXM"): 7.84, ("sharpe", "2016", "BXM"): 0.54,
    ("cagr", "2016", "WPUT"): 3.63, ("sharpe", "2016", "WPUT"): 0.16,
    ("cagr", "2016", "CNDR"): 1.18, ("sharpe", "2016", "CNDR"): -0.18,
    ("cagr", "1988", "CNDR"): 5.65,
    ("dd", "gfc", "SPTR"): -55.3, ("dd", "gfc", "PUT"): -37.1, ("dd", "gfc", "BXM"): -40.1,
    ("dd", "feb2018", "SPTR"): -10.1, ("dd", "feb2018", "WPUT"): -12.8, ("dd", "feb2018", "PUT"): -7.6,
    ("dd", "feb2018", "BXM"): -7.7, ("dd", "feb2018", "CNDR"): -5.6,
    ("dd", "covid", "SPTR"): -33.8, ("dd", "covid", "PUT"): -28.9, ("dd", "covid", "WPUT"): -25.3,
    ("dd", "covid", "BXM"): -30.3, ("dd", "covid", "BXMD"): -32.3, ("dd", "covid", "CNDR"): -11.2,
    ("dd", "y2022", "SPTR"): -24.5, ("dd", "y2022", "PUT"): -16.0,
    ("year", "2020", "SPTR"): 18.4, ("year", "2020", "PUT"): 2.1, ("year", "2020", "BXM"): -2.8,
    ("year", "2020", "WPUT"): -7.2, ("year", "2020", "CNDR"): -6.1,
    ("year", "2008", "SPTR"): -37.0, ("year", "2008", "PUT"): -26.8, ("year", "2008", "BXM"): -28.7,
    ("year", "2008", "WPUT"): -15.2, ("year", "2008", "CNDR"): -4.0,
    ("worst_month", "2007", "PUT"): -17.7, ("worst_month", "2007", "SPTR"): -16.8,
    ("recovery", "covid", "SPTR"): "2020-08-10", ("recovery", "covid", "PUT"): "2021-01-07",
    ("recovery", "covid", "BXM"): "2021-03-22", ("recovery", "covid", "CNDR"): "2023-05-05",
    # tail hedging
    ("cagr", "1988", "PPUT"): 7.92, ("cagr", "1988", "SPTR"): 11.44,
    ("cagr", "2007", "PPUT"): 8.14, ("cagr", "2016", "PPUT"): 12.20,
    ("sharpe", "1988", "PPUT"): 0.46, ("sharpe", "1988", "SPTR"): 0.62,
    ("sharpe", "2007", "PPUT"): 0.58, ("sharpe", "2016", "PPUT"): 0.79,
    ("maxdd", "2007", "PPUT"): -42.0, ("dd", "covid", "PPUT"): -11.8,
    ("year", "2008", "PPUT"): -20.1, ("year", "2022", "PPUT"): -19.8, ("year", "2022", "SPTR"): -18.1,
    ("year", "2020", "PPUT"): 29.1,
    ("cagr", "2016", "CLL"): 11.62, ("sharpe", "2016", "CLL"): 0.80, ("maxdd", "2016", "CLL"): -17.2,
    ("maxdd", "2016", "SPTR"): -33.8,
}
TOL = {"cagr": 0.15, "sharpe": 0.05, "maxdd": 0.5, "dd": 0.5, "beta": 0.03, "year": 0.3,
       "worst_month": 0.3, "recovery": 3}


# --- pure statistics ---------------------------------------------------------------

def cagr(s: pd.Series) -> float | None:
    s = s.dropna()
    if len(s) < 2:
        return None
    days = (s.index[-1] - s.index[0]).days
    return (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1.0 if days > 0 else None


def max_dd(s: pd.Series) -> float | None:
    s = s.dropna()
    return float((s / s.cummax() - 1.0).min()) if len(s) else None


def monthly_returns(s: pd.Series) -> pd.Series:
    """Month-end returns, the first month measured from the window's first close."""
    s = s.dropna()
    m = s.resample("ME").last()
    m = pd.concat([pd.Series([s.iloc[0]], index=[s.index[0] - pd.Timedelta(days=1)]), m])
    return m.pct_change().dropna()


def rf_monthly(rf: pd.Series, months: pd.DatetimeIndex) -> pd.Series:
    """Monthly T-bill return per month = prior month-end annual yield / 12."""
    m = rf.resample("ME").last().shift(1) / 12.0
    return m.reindex(months).ffill().fillna(0.0)


def cash_index(rf: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily-compounded T-bill wealth index on the given trading days."""
    y = rf.reindex(index).ffill().fillna(0.0)
    dt = pd.Series(index, index=index).diff().dt.days.fillna(0).to_numpy()
    return pd.Series(np.cumprod(1.0 + y.shift(1).fillna(0.0).to_numpy() * dt / 365.0), index=index)


def window_stats(s: pd.Series, bench: pd.Series, rf: pd.Series) -> dict:
    """CAGR, vol, Sharpe, beta, max drawdown, worst month for s (already windowed)."""
    mr = monthly_returns(s)
    rfm = rf_monthly(rf, mr.index)
    ex = mr - rfm
    sd = ex.std()
    out = {"start": s.index[0].date().isoformat(), "end": s.index[-1].date().isoformat(),
           "cagr": cagr(s), "vol": float(mr.std() * math.sqrt(12)),
           "sharpe": float(ex.mean() / sd * math.sqrt(12)) if sd > 0 else None,
           "maxdd": max_dd(s), "worst_month": float(mr.min())}
    bm = monthly_returns(bench.loc[s.index[0]:s.index[-1]])
    j = pd.concat([mr, bm], axis=1, join="inner").dropna()
    out["beta"] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 3 else None
    return out


def year_return(s: pd.Series, y: int) -> float | None:
    a, b = s.loc[:f"{y - 1}-12-31"], s.loc[:f"{y}-12-31"]
    if a.empty or b.empty or b.index[-1].year != y:
        return None
    return float(b.iloc[-1] / a.iloc[-1] - 1.0)


def recovery_date(s: pd.Series, peak_from: str = "2020-01-01", peak_to: str = "2020-02-20") -> str | None:
    """First close at or above the pre-COVID high, after the subsequent trough."""
    pk = s.loc[peak_from:peak_to]
    if pk.empty:
        return None
    hi = pk.max()
    after = s.loc[pk.idxmax():]
    trough = after.idxmin() if not after.empty else None
    rec = after.loc[trough:]
    hit = rec[rec >= hi]
    return hit.index[0].date().isoformat() if len(hit) else None


def matched_mix(target_cagr: float, stock: pd.Series, rf: pd.Series) -> dict:
    """Static stock/T-bill mix (monthly rebalanced) whose CAGR matches target_cagr."""
    mr = monthly_returns(stock)
    rfm = rf_monthly(rf, mr.index)
    best = None
    for w in np.linspace(0.0, 1.0, 201):
        path = (1.0 + w * mr + (1.0 - w) * rfm).cumprod()
        yrs = (mr.index[-1] - stock.index[0]).days / 365.25
        c = path.iloc[-1] ** (1.0 / yrs) - 1.0
        if best is None or abs(c - target_cagr) < abs(best["cagr"] - target_cagr):
            dd = float((path / path.cummax() - 1.0).min())
            best = {"w_stock": float(w), "cagr": float(c), "maxdd_monthly": dd,
                    "worst_month": float((w * mr + (1 - w) * rfm).min())}
    return best


def compute(ix: dict[str, pd.Series], bench: pd.Series, rf: pd.Series, names: list[str],
            end: str = END) -> dict:
    """ix: {name: daily close}; bench: S&P 500 TR daily close; rf: decimal yield. Pure."""
    series = {BENCH: bench.loc[:end]}
    for n in names:
        if n in ix:
            series[n] = ix[n].loc[FIRST_GOOD.get(n, "1900-01-01"):end]
    stats: dict = {}
    for wk, start in WINDOWS.items():
        for n, s in series.items():
            w = s.loc[start:]
            if len(w) < 60 or w.index[0] > pd.Timestamp(start) + pd.Timedelta(days=10):
                continue
            stats[(wk, n)] = window_stats(w, bench, rf)
        cash = cash_index(rf, bench.loc[start:end].index)
        stats[(wk, "TBILL")] = {"cagr": cagr(cash)}
    stress = {(ek, n): max_dd(s.loc[a:b]) for ek, (a, b) in STRESS.items() for n, s in series.items()
              if len(s.loc[a:b]) > 5}
    years = {(str(y), n): year_return(s, y) for y in YEARS for n, s in series.items()}
    recov = {n: recovery_date(s) for n, s in series.items()}
    return {"stats": stats, "stress": stress, "years": years, "recovery": recov,
            "series": series}


def ours_for(res: dict, stat: str, key: str, n: str):
    if stat in ("cagr", "sharpe", "maxdd", "beta", "worst_month"):
        v = res["stats"].get((key, n), {}).get(stat)
    elif stat == "dd":
        v = res["stress"].get((key, n))
    elif stat == "year":
        v = res["years"].get((key, n))
    elif stat == "recovery":
        return res["recovery"].get(n)
    else:
        return None
    if v is None:
        return None
    return v if stat in ("sharpe", "beta") else v * 100.0


def phase03_check(res: dict, names: list[str]) -> dict:
    rows = []
    for (stat, key, n), ref in PHASE03.items():
        if n not in names and n != BENCH:
            continue
        ours = ours_for(res, stat, key, n)
        if ours is None:
            ok = None
        elif stat == "recovery":
            ok = abs((pd.Timestamp(ours) - pd.Timestamp(ref)).days) <= TOL[stat]
        else:
            ok = abs(ours - ref) <= TOL[stat]
        rows.append({"stat": stat, "window": key, "index": n, "phase03": ref,
                     "ours": ours if stat == "recovery" else rnd(ours, 2), "match": ok})
    n_ok = sum(1 for r in rows if r["match"])
    return {"n": len(rows), "matched": n_ok, "tolerance": TOL, "rows": rows}


def _table(res: dict, names: list[str]) -> list[dict]:
    out = []
    for (wk, n), st in res["stats"].items():
        if n in names or n in (BENCH, "TBILL"):
            out.append({"window": wk, "index": n, **{k: (v if k in ("start", "end") else rnd(v, 4))
                                                      for k, v in st.items()}})
    return out


def _chart(res: dict, names: list[str], start: str = "2007-01-03") -> dict:
    growth = []
    for n in [BENCH] + names:
        s = res["series"].get(n)
        if s is None:
            continue
        m = s.loc[start:].resample("ME").last()
        if m.empty:
            continue
        g = m / m.iloc[0]
        growth += [{"month": d.strftime("%Y-%m"), "index": n, "growth": rnd(v, 4)}
                   for d, v in g.items()]
    episodes = [{"episode": ek, "index": n, "drawdown": rnd(v, 4)}
                for (ek, n), v in res["stress"].items() if n in names or n == BENCH]
    bars = [{"index": n, "window": wk, "cagr": rnd(st.get("cagr"), 4), "sharpe": rnd(st.get("sharpe"), 3)}
            for (wk, n), st in res["stats"].items() if (n in names or n in (BENCH, "TBILL"))]
    return {"growth_monthly_2007": growth, "episode_drawdowns": episodes, "window_bars": bars}


def _p(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


COMMON_CAVEATS = [
    "Gross index returns: no commissions, no bid-ask (Cboe indices trade at stated prices), "
    "no taxes. Real implementations lag the index.",
    "Cboe publishes these indices and sponsored most of the early studies of them.",
    "The S&P 500 TR comes from Yahoo ^SP500TR; Sharpe uses month-end returns and ^IRX, so "
    "it can differ by about +/-0.05 from other conventions.",
]


def build_short_vol(res: dict) -> dict:
    st = res["stats"]
    put7, sp7, cash7 = st.get(("2007", "PUT"), {}), st.get(("2007", BENCH), {}), st.get(("2007", "TBILL"), {})
    pput7 = st.get(("2007", "PPUT"), {})
    names09 = [n for n in SHORT_VOL if ("2009", n) in st]
    trail09 = [n for n in names09 if st[("2009", n)]["cagr"] < st[("2009", BENCH)]["cagr"]]
    betas = {n: st[("2007", n)]["beta"] for n in SHORT_VOL if ("2007", n) in st}
    cndr = st.get(("2007", "CNDR"), {})
    # seller = PUT over its T-bill collateral; buyer = PPUT's put overlay vs the bare S&P 500 TR
    sell_edge = put7["cagr"] - cash7["cagr"] if put7 else None
    buy_cost = pput7["cagr"] - sp7["cagr"] if pput7 else None
    v1 = ("agrees" if sell_edge is not None and buy_cost is not None and sell_edge > 0 > buy_cost
          else "mixed")
    v2 = "mixed" if len(trail09) == len(names09) and max(betas.values()) >= 0.5 else "agrees"
    v3 = ("agrees" if cndr and cndr["cagr"] < put7["cagr"] and cndr["cagr"] <= cash7["cagr"] + 0.005
          else "mixed")
    ours1 = (f"2007-2026: selling ATM SPX puts (PUT, {_p(put7.get('cagr'))} CAGR) earned "
             f"{sell_edge * 100:+.2f} pts/yr over its T-bill collateral ({_p(cash7.get('cagr'))}); "
             f"buying 5% OTM puts on the S&P (PPUT, {_p(pput7.get('cagr'))}) cost "
             f"{buy_cost * 100:+.2f} pts/yr vs the S&P 500 TR ({_p(sp7.get('cagr'))}).")
    ours2 = (f"Since 2009-03-09 {len(trail09)}/{len(names09)} short-vol indices trailed the S&P 500 TR "
             f"({_p(st[('2009', BENCH)]['cagr'])}); 2007-2026 betas to the S&P "
             + ", ".join(f"{n} {b:.2f}" for n, b in betas.items()) + ".")
    ours3 = (f"CNDR (20-delta short strangle + 5-delta wings) {_p(cndr.get('cagr'))} CAGR 2007-2026 "
             f"vs PUT {_p(put7.get('cagr'))} and T-bills {_p(cash7.get('cagr'))}.")
    claims = [
        claim("Selling premium beats buying it over time", "lit:option-selling", ours1, v1,
              ["lit:bondarenko-2019"]),
        claim("A short-premium book is a distinct, high-probability return engine for 25-50% of "
              "capital", "lit:option-selling", ours2, v2, ["lit:israelov-nielsen-2015"]),
        claim("Defined-risk wings give up much of the edge", "lit:option-selling", ours3, v3,
              ["lit:moran-2021b"]),
    ]
    chk = phase03_check(res, SHORT_VOL)
    return artefact(
        "cboe_short_vol", claim_text=claims[1]["text"], claims=claims, our_value=ours2,
        data="Cboe index history CSVs (cdn.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv): "
             "PUT, BXM, BXMD, WPUT, CNDR, BFLY (PPUT for comparison); Yahoo ^SP500TR, ^IRX",
        period=f"windows from 1988-01-04 / 2007-01-03 / 2009-03-09 / 2016-12-30 to {END}",
        method="CAGR from daily closes; vol, Sharpe (vs ^IRX) and beta from month-end returns; "
               "max drawdown and episode peak-to-trough on daily closes. Phase 03's computed_stats "
               "are re-derived and compared within stated tolerances (phase03_check).",
        caveats=COMMON_CAVEATS + [
            "PUT's Cboe file is sparse before 2007-01-03, so PUT starts there; BXM's file starts "
            "2002-03-22 and WPUT's 2006-01-31.",
            "Cboe has no undefined-risk strangle index; CNDR has 5-delta wings, so it is not the "
            "tastylive naked strangle (see repro:strangle_mgmt for a model-priced one)."],
        verdict=v2, chart_data=_chart(res, SHORT_VOL), literature=["lit:bondarenko-2019"],
        phase03_check=chk, table=_table(res, SHORT_VOL + ["PPUT"]),
        episodes=[{"episode": e, "index": n, "drawdown": rnd(v, 4)} for (e, n), v in res["stress"].items()],
        calendar_years=[{"year": y, "index": n, "ret": rnd(v, 4)} for (y, n), v in res["years"].items()],
        recovery=res["recovery"])


def build_tail_hedge(res: dict, mix: dict | None = None) -> dict:
    st = res["stats"]
    drag = {wk: (st[(wk, BENCH)]["cagr"] - st[(wk, "PPUT")]["cagr"]) for wk in ("1988", "2007", "2016")
            if (wk, "PPUT") in st and (wk, BENCH) in st}
    covid = (res["stress"].get(("covid", "PPUT")), res["stress"].get(("covid", BENCH)))
    v1 = "agrees" if drag and all(d > 0 for d in drag.values()) else "mixed"
    v2 = ("mixed" if covid[0] is not None and covid[0] - covid[1] > 0.10 and v1 == "agrees"
          else "agrees")
    ours1 = ("PPUT trailed the S&P 500 TR by " + ", ".join(f"{d * 100:.1f} pts/yr from {wk}"
                                                           for wk, d in drag.items()) + ".")
    ours2 = (f"COVID peak-to-trough: PPUT {covid[0] * 100:.1f}% vs S&P 500 TR {covid[1] * 100:.1f}%; "
             f"calendar 2020 PPUT {_p(res['years'].get(('2020', 'PPUT')))} vs "
             f"{_p(res['years'].get(('2020', BENCH)))}; 2022 PPUT {_p(res['years'].get(('2022', 'PPUT')))}.")
    claims = [
        claim("Buying options or protection loses money over time", "lit:tail-hedging", ours1, v1,
              ["lit:israelov-2018", "lit:bondarenko-2019"]),
        claim("Long-volatility positions have no useful role in a portfolio", "lit:tail-hedging",
              ours2, v2, ["lit:harvey-etal-2019"]),
    ]
    if mix:
        pp = st[("1988", "PPUT")]
        pp_mdd = max_dd((1 + monthly_returns(res["series"]["PPUT"].loc["1988-01-04":])).cumprod())
        v3 = "agrees" if mix["maxdd_monthly"] > pp_mdd else "contradicts"
        ours3 = (f"1988-2026, a {mix['w_stock']:.0%} S&P TR / {1 - mix['w_stock']:.0%} T-bill mix "
                 f"matches PPUT's {_p(pp['cagr'])} CAGR with a {mix['maxdd_monthly'] * 100:.1f}% "
                 f"max drawdown (month-end) vs PPUT {pp_mdd * 100:.1f}%; worst month "
                 f"{mix['worst_month'] * 100:.1f}% vs {pp['worst_month'] * 100:.1f}%.")
        claims.append(claim("Manage risk by trading smaller rather than buying puts",
                            "lit:tail-hedging", ours3, v3, ["lit:israelov-2018"], mix=mix))
    chk = phase03_check(res, TAIL)
    return artefact(
        "cboe_tail_hedge", claim_text=claims[0]["text"], claims=claims, our_value=ours1,
        data="Cboe PPUT and CLL history CSVs; Yahoo ^SP500TR, ^IRX",
        period=f"windows from 1988-01-04 / 2007-01-03 / 2016-12-30 to {END}",
        method="As repro:cboe_short_vol. The matched mix is a monthly-rebalanced S&P 500 TR / "
               "T-bill blend with the stock weight chosen (0.5% grid) to match PPUT's 1988-2026 CAGR.",
        caveats=COMMON_CAVEATS + [
            "PPUT buys a 5% OTM one-month put every month: one hedging design among many "
            "(tenor, strike and monetization rules change results a lot).",
            "CLL's file starts 2008-08-26, so it has no full 2007 window.",
            "The matched-mix drawdown is on month-end closes; PPUT's comparison number uses the "
            "same month-end basis."],
        verdict=v1, chart_data=_chart(res, TAIL), literature=["lit:israelov-2018"],
        phase03_check=chk, table=_table(res, TAIL),
        episodes=[{"episode": e, "index": n, "drawdown": rnd(v, 4)} for (e, n), v in res["stress"].items()],
        calendar_years=[{"year": y, "index": n, "ret": rnd(v, 4)} for (y, n), v in res["years"].items()],
        recovery=res["recovery"])
