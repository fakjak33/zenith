"""repro:strangle_mgmt — a model-priced 16-delta SPX short strangle under five
management rules, entered every trading day since 1990.

Pricing: Black-Scholes with sigma = that day's VIX for both strikes (no skew),
rate = ^IRX, no dividends, marked at each close. Strikes solve the target delta
at entry (inverse normal from statistics.NormalDist; the CDF is
zenith.beta.hedge._ncdf). Expiry = last trading day on or before entry + DTE.

  hold            to expiry (cash-settled at the expiry close)
  p50             close when the strangle is worth <= 50% of the credit
  d21             close at the first close with <= 21 calendar days left
  p50_d21         whichever comes first (the tastylive default)
  p50_d21_stop2x  ... or close when the loss reaches 2x the credit

This is ILLUSTRATIVE: no skew and no real quotes understate put credit and tail
losses; closes-only marking understates intraday gaps and stop slippage.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from ...beta.hedge import _ncdf
from . import artefact, claim, rnd

RULES = ["hold", "p50", "d21", "p50_d21", "p50_d21_stop2x"]
DTES = [30, 45, 60]
EPISODES = {"gfc_2008": ("2008-09-01", "2008-12-31"), "volmageddon_2018": ("2018-01-26", "2018-02-28"),
            "covid_2020": ("2020-02-19", "2020-04-30")}
BOOK_NOISE = 0.25          # pts of notional a year; smaller book differences = "no change"
_N = np.vectorize(_ncdf, otypes=[float])
_INV = NormalDist().inv_cdf


# --- pricing ---------------------------------------------------------------------

def bs_price(S, K, T, sig, r, call: bool):
    """Vectorized Black-Scholes (no dividends); T in years, T<=0 -> intrinsic."""
    S, K, T, sig, r = (np.asarray(x, dtype=float) for x in (S, K, T, sig, r))
    intrinsic = np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)
    live = T > 1e-9
    Tl = np.where(live, T, 1.0)
    v = sig * np.sqrt(Tl)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * Tl) / v
    d2 = d1 - v
    disc = np.exp(-r * Tl)
    if call:
        px = S * _N(d1) - K * disc * _N(d2)
    else:
        px = K * disc * _N(-d2) - S * _N(-d1)
    return np.where(live, px, intrinsic)


def strike_for_delta(S: float, T: float, sig: float, r: float, delta: float, call: bool) -> float:
    """Strike whose BS delta is +delta (call) or -delta (put)."""
    d1 = _INV(delta) if call else _INV(1.0 - delta)
    return S * math.exp(-d1 * sig * math.sqrt(T) + (r + 0.5 * sig * sig) * T)


def regt_bpr(S: float, Kp: float, Kc: float, put_px: float, call_px: float) -> float:
    """Approximate Reg T requirement for a naked broad-index strangle: the larger
    leg's max(15% S - OTM, 10% of S or strike) plus its premium, plus the other
    leg's premium."""
    rp = max(0.15 * S - max(S - Kp, 0.0), 0.10 * Kp) + put_px
    rc = max(0.15 * S - max(Kc - S, 0.0), 0.10 * S) + call_px
    return max(rp + call_px, rc + put_px)


# --- simulation --------------------------------------------------------------------

def simulate(S: pd.Series, vix: pd.Series, rf: pd.Series, dte: int = 45, delta: float = 0.16,
             step: int = 1) -> pd.DataFrame:
    """One row per entry day x rule. S close, vix decimal, rf decimal. Pure."""
    j = pd.concat([S.rename("S"), vix.rename("iv"), rf.rename("r")], axis=1).sort_index()
    j["r"] = j["r"].ffill()
    j = j.dropna()
    dates = j.index
    Sa, iva, ra = j["S"].to_numpy(), j["iv"].to_numpy(), j["r"].to_numpy()
    rows = []
    for i in range(0, len(dates), step):
        exp_date = dates[i] + pd.Timedelta(days=dte)
        e = dates.searchsorted(exp_date, side="right") - 1
        if exp_date > dates[-1] or e <= i:
            break
        T0 = dte / 365.0
        Kp = strike_for_delta(Sa[i], T0, iva[i], ra[i], delta, call=False)
        Kc = strike_for_delta(Sa[i], T0, iva[i], ra[i], delta, call=True)
        sl = slice(i, e + 1)
        tau = np.array([(exp_date - d).days for d in dates[sl]], dtype=float)
        tau[-1] = 0.0
        p = bs_price(Sa[sl], Kp, tau / 365.0, iva[sl], ra[sl], call=False)
        c = bs_price(Sa[sl], Kc, tau / 365.0, iva[sl], ra[sl], call=True)
        V = p + c
        credit = V[0]
        if not credit > 0:
            continue
        bpr = regt_bpr(Sa[i], Kp, Kc, p[0], c[0])
        k = np.arange(len(V))
        hit50 = np.where((V <= 0.5 * credit) & (k > 0))[0]
        hit21 = np.where((tau <= 21) & (k > 0))[0]
        hitst = np.where((V >= 3.0 * credit) & (k > 0))[0]
        last = len(V) - 1
        f = lambda a: int(a[0]) if len(a) else last          # noqa: E731
        exits = {"hold": last, "p50": f(hit50), "d21": f(hit21)}
        exits["p50_d21"] = min(exits["p50"], exits["d21"])
        exits["p50_d21_stop2x"] = min(exits["p50_d21"], f(hitst))
        for rule, x in exits.items():
            pnl = credit - V[x]
            rows.append({"entry": dates[i], "exit": dates[i + x], "rule": rule, "dte": dte,
                         "S0": Sa[i], "iv0": iva[i], "credit": credit, "bpr": bpr,
                         "pnl": pnl, "days": (dates[i + x] - dates[i]).days,
                         "touched_put": bool((Sa[sl][:x + 1] < Kp).any()),
                         "touched_call": bool((Sa[sl][:x + 1] > Kc).any())})
    return pd.DataFrame(rows)


def rule_stats(df: pd.DataFrame) -> dict:
    """Trade-level distribution stats; P&L in credits and in % of SPX notional."""
    pc = df["pnl"] / df["credit"]
    pn = df["pnl"] / df["S0"] * 100.0
    q = max(1, int(len(df) * 0.05))
    worst = pn.nsmallest(q)
    worst_c = pc.nsmallest(q)
    days = df["days"].clip(lower=1)
    return {"n": int(len(df)), "win_rate": rnd((df["pnl"] > 0).mean(), 4),
            "mean_credit_pct": rnd((df["credit"] / df["S0"]).mean() * 100, 3),
            "mean_pnl_credits": rnd(pc.mean(), 3), "mean_pnl_pct": rnd(pn.mean(), 4),
            "median_pnl_pct": rnd(pn.median(), 4), "stdev_pct": rnd(pn.std(), 4),
            "cvar5_pct": rnd(worst.mean(), 3), "cvar5_credits": rnd(worst_c.mean(), 2),
            "max_loss_pct": rnd(pn.min(), 3), "max_loss_credits": rnd(pc.min(), 2),
            "mean_days": rnd(days.mean(), 1),
            "pnl_per_day_bp": rnd(pn.mean() / days.mean() * 100.0, 3),
            "mean_roc_bpr": rnd((df["pnl"] / df["bpr"]).mean(), 4)}


def sequential_book(df: pd.DataFrame) -> pd.DataFrame:
    """One strangle at a time: the next opens on the close the previous one exits.
    Returns the trades taken (P&L in % of notional at entry)."""
    df = df.sort_values("entry").reset_index(drop=True)
    entries = df["entry"].to_numpy()
    out, pos = [], 0
    while pos < len(df):
        row = df.iloc[pos]
        out.append({"entry": row["entry"], "exit": row["exit"],
                    "pnl_pct": row["pnl"] / row["S0"] * 100.0})
        nxt = int(entries.searchsorted(row["exit"].to_datetime64(), side="left"))
        pos = max(nxt, pos + 1)
    return pd.DataFrame(out)


def episode_stats(df: pd.DataFrame) -> dict:
    out = {}
    for ek, (a, b) in EPISODES.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        sub = df[(df["entry"] <= b) & (df["exit"] >= a)]
        if sub.empty:
            continue
        pc, pn = sub["pnl"] / sub["credit"], sub["pnl"] / sub["S0"] * 100.0
        out[ek] = {"n": int(len(sub)), "share_losing": rnd((sub["pnl"] < 0).mean(), 3),
                   "mean_pnl_credits": rnd(pc.mean(), 2), "worst_pnl_credits": rnd(pc.min(), 2),
                   "worst_pnl_pct": rnd(pn.min(), 2)}
    return out


def compute(S: pd.Series, vix: pd.Series, rf: pd.Series, step: int = 1) -> dict:
    sims = {d: simulate(S, vix, rf, dte=d, step=step) for d in DTES}
    main = sims[45]
    by_rule, books, episodes, hist = {}, {}, {}, []
    for rule in RULES:
        sub = main[main["rule"] == rule]
        by_rule[rule] = rule_stats(sub)
        bk = sequential_book(sub)
        yrs = (bk["exit"].iloc[-1] - bk["entry"].iloc[0]).days / 365.25
        per_year = bk.groupby(bk["exit"].dt.year)["pnl_pct"].sum()
        books[rule] = {"trades": int(len(bk)), "trades_per_year": rnd(len(bk) / yrs, 1),
                       "pnl_pct_per_year": rnd(bk["pnl_pct"].sum() / yrs, 3),
                       "worst_year_pct": rnd(per_year.min(), 2),
                       "worst_year": int(per_year.idxmin()),
                       "by_year": {int(y): rnd(v, 3) for y, v in per_year.items()},
                       "curve": bk}
        episodes[rule] = episode_stats(sub)
        pc = (sub["pnl"] / sub["credit"]).clip(-10, 1)
        counts = pd.cut(pc, np.arange(-10.0, 1.26, 0.25), right=False).value_counts().sort_index()
        hist += [{"rule": rule, "bin_lo": rnd(iv.left, 2), "share": rnd(c / len(sub), 4)}
                 for iv, c in counts.items() if c]
    by_dte = {d: {rule: rule_stats(s[s["rule"] == rule]) for rule in ("hold", "p50_d21")}
              for d, s in sims.items()}
    for d, s in sims.items():
        for rule in ("hold", "p50_d21"):
            bk = sequential_book(s[s["rule"] == rule])
            yrs = (bk["exit"].iloc[-1] - bk["entry"].iloc[0]).days / 365.25
            by_dte[d][rule]["book_pnl_pct_per_year"] = rnd(bk["pnl_pct"].sum() / yrs, 3)
    return {"by_rule": by_rule, "books": books, "episodes": episodes, "by_dte": by_dte,
            "hist": hist, "start": main["entry"].min().date().isoformat(),
            "end": main["exit"].max().date().isoformat(),
            "touch_put": rnd(main[main["rule"] == "hold"]["touched_put"].mean(), 3),
            "touch_call": rnd(main[main["rule"] == "hold"]["touched_call"].mean(), 3)}


def _cmp(a, b, higher_better=True):
    return (a > b) if higher_better else (a < b)


def build(res: dict) -> dict:
    R, B = res["by_rule"], res["books"]
    h, p, d, pd21, st = R["hold"], R["p50"], R["d21"], R["p50_d21"], R["p50_d21_stop2x"]
    # 1. 50% management
    win_up = p["win_rate"] > h["win_rate"]
    day_up = p["pnl_per_day_bp"] > h["pnl_per_day_bp"]
    trade_up = p["mean_pnl_pct"] > h["mean_pnl_pct"]
    v1 = "agrees" if win_up and day_up and trade_up else "contradicts" if not day_up else "mixed"
    o1 = (f"50% target vs hold: win rate {p['win_rate']:.0%} vs {h['win_rate']:.0%}; mean P&L "
          f"{p['mean_pnl_credits']:+.2f} vs {h['mean_pnl_credits']:+.2f} credits per trade; "
          f"P&L/day {p['pnl_per_day_bp']:+.2f} vs {h['pnl_per_day_bp']:+.2f} bp of notional; "
          f"one-at-a-time book {B['p50']['pnl_pct_per_year']:+.2f}% vs "
          f"{B['hold']['pnl_pct_per_year']:+.2f}% of notional a year.")
    # 2. 21 DTE exit and tails
    cv = _cmp(d["cvar5_pct"], h["cvar5_pct"]); mx = _cmp(d["max_loss_pct"], h["max_loss_pct"])
    v2 = "agrees" if cv and mx else "mixed" if cv or mx else "contradicts"
    o2 = (f"21-DTE exit vs hold: CVaR(5%) {d['cvar5_credits']:+.2f} vs {h['cvar5_credits']:+.2f} "
          f"credits ({d['cvar5_pct']:+.2f}% vs {h['cvar5_pct']:+.2f}% of notional); max loss "
          f"{d['max_loss_credits']:+.1f} vs {h['max_loss_credits']:+.1f} credits; stdev "
          f"{d['stdev_pct']:.2f}% vs {h['stdev_pct']:.2f}%.")
    # 3. rules add edge?
    # book differences under BOOK_NOISE pts/yr are treated as no change
    t_up = pd21["mean_pnl_pct"] > h["mean_pnl_pct"]
    b_up = B["p50_d21"]["pnl_pct_per_year"] > B["hold"]["pnl_pct_per_year"] + BOOK_NOISE
    v3 = "agrees" if t_up and b_up else "contradicts" if not (t_up or b_up) else "mixed"
    o3 = (f"50%-or-21-DTE vs hold: mean P&L per trade {pd21['mean_pnl_pct']:+.3f}% vs "
          f"{h['mean_pnl_pct']:+.3f}% of notional; one-at-a-time book "
          f"{B['p50_d21']['pnl_pct_per_year']:+.2f}% vs {B['hold']['pnl_pct_per_year']:+.2f}% a year "
          f"({B['p50_d21']['trades_per_year']:.0f} vs {B['hold']['trades_per_year']:.0f} trades/yr).")
    # 4. seed strategy
    v4 = "agrees" if pd21["mean_pnl_pct"] > 0 and B["p50_d21"]["pnl_pct_per_year"] > 0 else "contradicts"
    o4 = (f"Managed 16-delta 45-DTE strangle (model): win rate {pd21['win_rate']:.0%}, mean "
          f"{pd21['mean_pnl_credits']:+.2f} credits/trade, mean credit {pd21['mean_credit_pct']:.2f}% "
          f"of notional, CVaR(5%) {pd21['cvar5_credits']:+.2f} credits, worst {pd21['max_loss_credits']:+.1f} "
          f"credits; mean return on Reg-T BPR {pd21['mean_roc_bpr'] * 100:+.2f}% per trade; worst book "
          f"year {B['p50_d21']['worst_year']} ({B['p50_d21']['worst_year_pct']:+.1f}% of notional).")
    # 5. tenor
    dd = res["by_dte"]
    per_day = {k: v["p50_d21"]["pnl_per_day_bp"] for k, v in dd.items()}
    book = {k: v["p50_d21"]["book_pnl_pct_per_year"] for k, v in dd.items()}
    best45 = max(per_day, key=per_day.get) == 45 and max(book, key=book.get) == 45
    v5 = "agrees" if best45 else "mixed"
    o5 = ("Managed P&L/day by entry DTE: " + ", ".join(f"{k} DTE {v:+.2f} bp" for k, v in per_day.items())
          + "; book % per year: " + ", ".join(f"{k} DTE {v:+.2f}" for k, v in book.items())
          + ". VIX (30-day IV) prices every tenor, so this cannot see the term structure.")
    # 6. stops
    s_trade = st["mean_pnl_pct"] < pd21["mean_pnl_pct"]
    s_book = B["p50_d21_stop2x"]["pnl_pct_per_year"] < B["p50_d21"]["pnl_pct_per_year"]
    v6 = "agrees" if s_trade and s_book else "contradicts" if not (s_trade or s_book) else "mixed"
    o6 = (f"Adding a 2x-credit stop: mean {st['mean_pnl_credits']:+.2f} vs {pd21['mean_pnl_credits']:+.2f} "
          f"credits/trade, win rate {st['win_rate']:.0%} vs {pd21['win_rate']:.0%}, CVaR(5%) "
          f"{st['cvar5_credits']:+.2f} vs {pd21['cvar5_credits']:+.2f} credits, book "
          f"{B['p50_d21_stop2x']['pnl_pct_per_year']:+.2f}% vs {B['p50_d21']['pnl_pct_per_year']:+.2f}% a year "
          f"(closes only, so real stop fills would be worse).")
    claims = [
        claim("Sell 16-20 delta strangles at 45-60 DTE and manage at 50% of max profit or ~21 DTE",
              "seed", o4, v4, ["lit:bondarenko-2019"]),
        claim("Managing winners at 50% of max profit improves results", "lit:management-rules", o1, v1,
              ["lit:tastylive-managing-2016"]),
        claim("Exiting at 21 DTE reduces tail losses", "lit:management-rules", o2, v2,
              ["lit:tastylive-managing-2016"]),
        claim("Management rules increase the strategy's expected edge", "lit:management-rules", o3, v3,
              ["lit:williams-1991"]),
        claim("Avoid hard stop-losses on short premium", "lit:management-rules", o6, v6,
              ["lit:kaminski-lo-2014"]),
        claim("About 45 DTE is the best tenor for selling premium", "lit:iv-predictors", o5, v5,
              ["lit:dewbecker-etal-2017", "lit:egloff-etal-2010"]),
    ]
    rule_rows = [{"rule": k, **v} for k, v in R.items()]
    curve = []
    for rule, bk in B.items():
        c = bk["curve"].set_index("exit")["pnl_pct"].cumsum()
        m = c.groupby(c.index.to_period("M")).last()
        curve += [{"month": str(per), "rule": rule, "cum_pnl_pct": rnd(v, 3)} for per, v in m.items()]
    books = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in B.items()}
    ep_rows = [{"rule": rule, "episode": ek, **v} for rule, eps in res["episodes"].items()
               for ek, v in eps.items()]
    return artefact(
        "strangle_mgmt", claim_text=claims[0]["text"], claims=claims, our_value=o4,
        data="Yahoo ^GSPC close, ^VIX close, ^IRX (daily); no option quotes",
        period=f"entries {res['start']} onward, exits to {res['end']}",
        method=("Every trading day: sell a 16-delta put and 16-delta call at 45 DTE (also 30 and 60), "
                "strikes and marks from Black-Scholes with sigma = that day's VIX, r = ^IRX, no "
                "dividends, marked at each close; exits per rule at the close that triggers. P&L per "
                "1x SPX notional; 'credits' = P&L / entry credit. 'Book' = one strangle at a time, "
                "the next opened at the exit close, P&L summed per year in % of notional. BPR = "
                "approximate Reg T (15% of underlying less OTM, min 10%, plus premium)."),
        caveats=["Illustrative only: no skew (real 16-delta puts trade several vol points above VIX "
                 "and calls below), so the put credit and the size of crash losses are understated "
                 "and the call side is overstated.",
                 "No real quotes, no bid-ask, no commissions; marks at closes only, so intraday gaps "
                 "and stop slippage (Feb 2018) are missed.",
                 "One flat VIX for all strikes and remaining tenors; VIX is 30-day, so 60-DTE pricing "
                 "and the tenor comparison are rough.",
                 "Strikes are not rounded to the SPX grid; expiry on the last trading day on/before "
                 "entry + DTE; no early assignment (SPX is European).",
                 "Daily entries overlap heavily; trade counts overstate independent observations.",
                 "No dividends in the forward, which slightly misplaces strikes (~0.2% at 45 DTE)."],
        verdict=v4,
        chart_data={"rules": rule_rows, "pnl_hist_credits": res["hist"], "book_curve": curve,
                    "episodes": ep_rows,
                    "by_dte": [{"dte": k, "rule": rule, **v} for k, rr in res["by_dte"].items()
                               for rule, v in rr.items()]},
        literature=["lit:bondarenko-2019", "lit:tastylive-managing-2016"],
        by_rule=R, books=books, episodes=res["episodes"],
        by_dte={str(k): v for k, v in res["by_dte"].items()},
        touch_rates={"put": res["touch_put"], "call": res["touch_call"]})
