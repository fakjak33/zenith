"""repro:vix_spikes — what follows a VIX spike: forward VIX, SPX return, SPX
drawdown and the realized-vs-implied spread a seller would have faced.

Events (de-clustered: a new event needs 21 trading days since the last one):
  cross30  VIX closes above 30 after closing at or below 30 the day before
  jump50   VIX >= 1.5x its lowest close of the prior 5 trading days
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import artefact, claim, rnd
from .vrp import forward_realized

HORIZONS = (5, 21, 63)
GAP = 21


def events(vix: pd.Series) -> dict[str, list[pd.Timestamp]]:
    v = vix.dropna()
    cross = (v > 0.30) & (v.shift(1) <= 0.30)
    jump = v >= 1.5 * v.shift(1).rolling(5, min_periods=5).min()

    def declust(mask: pd.Series) -> list[pd.Timestamp]:
        out, last = [], -10 ** 9
        pos = {d: i for i, d in enumerate(v.index)}
        for d in v.index[mask.fillna(False).to_numpy()]:
            if pos[d] - last >= GAP:
                out.append(d)
                last = pos[d]
        return out
    return {"cross30": declust(cross), "jump50": declust(jump)}


def forward(S: pd.Series, vix: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame:
    j = pd.concat([S.rename("S"), vix.rename("iv")], axis=1, join="inner").dropna()
    rv = forward_realized(j["S"])
    idx = j.index
    rows = []
    for d in dates:
        if d not in idx:
            continue
        i = idx.get_loc(d)
        row = {"date": d, "vix": j["iv"].iloc[i] * 100, "rv21": rv.iloc[i] * 100 if rv.iloc[i] == rv.iloc[i] else None}
        for h in HORIZONS:
            if i + h < len(idx):
                row[f"spx_{h}"] = j["S"].iloc[i + h] / j["S"].iloc[i] - 1.0
                row[f"vix_{h}"] = (j["iv"].iloc[i + h] - j["iv"].iloc[i]) * 100
        if i + 21 < len(idx):
            path = j["S"].iloc[i:i + 22] / j["S"].iloc[i]
            row["min21"] = float(path.min() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _summ(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    for h in HORIZONS:
        if f"spx_{h}" in df:
            s = df[f"spx_{h}"].dropna()
            out[f"spx_{h}_mean"] = rnd(s.mean(), 4)
            out[f"spx_{h}_median"] = rnd(s.median(), 4)
            out[f"spx_{h}_up"] = rnd((s > 0).mean(), 3)
            out[f"spx_{h}_worst"] = rnd(s.min(), 4)
            out[f"vix_{h}_mean"] = rnd(df[f"vix_{h}"].dropna().mean(), 2)
    if "rv21" in df:
        sp = (df["vix"] - df["rv21"]).dropna()
        out["vrp21_mean"] = rnd(sp.mean(), 2)
        out["vrp21_share_pos"] = rnd((sp > 0).mean(), 3)
        out["vrp21_worst"] = rnd(sp.min(), 2)
    if "min21" in df:
        out["min21_mean"] = rnd(df["min21"].mean(), 4)
        out["min21_share_below_10"] = rnd((df["min21"] < -0.10).mean(), 3)
        out["min21_worst"] = rnd(df["min21"].min(), 4)
    return out


def compute(S: pd.Series, vix: pd.Series) -> dict:
    ev = events(vix)
    allday = forward(S, vix, list(pd.concat([S, vix], axis=1, join="inner").dropna().index))
    out = {"unconditional": _summ(allday), "events": {}}
    for k, ds in ev.items():
        df = forward(S, vix, ds)
        out["events"][k] = {**_summ(df), "list": [
            {"date": r["date"].date().isoformat(), "vix": rnd(r["vix"], 1),
             "spx_21": rnd(r.get("spx_21"), 4), "min21": rnd(r.get("min21"), 4),
             "vrp21": rnd(r["vix"] - r["rv21"], 1) if r.get("rv21") is not None else None}
            for r in df.to_dict("records")]}
    out["start"] = vix.dropna().index[0].date().isoformat()
    out["end"] = vix.dropna().index[-1].date().isoformat()
    return out


def build(res: dict) -> dict:
    u, c, jmp = res["unconditional"], res["events"]["cross30"], res["events"]["jump50"]
    richer = c["vrp21_mean"] > u["vrp21_mean"]
    better = c["spx_21_mean"] > u["spx_21_mean"]
    # tails: how often the month after entry saw a >10% SPX drawdown, vs all days
    tails = c["min21_share_below_10"] > u["min21_share_below_10"]
    v1 = "agrees" if richer and better and not tails else "contradicts" if not (richer or better) else "mixed"
    ours = (f"After VIX first closes above 30 (n={c['n']}): VIX minus next-21d realized averaged "
            f"{c['vrp21_mean']:+.1f} pts vs {u['vrp21_mean']:+.1f} on all days (positive "
            f"{c['vrp21_share_pos']:.0%} vs {u['vrp21_share_pos']:.0%}); SPX +21d mean "
            f"{c['spx_21_mean'] * 100:+.1f}% vs {u['spx_21_mean'] * 100:+.1f}%, up {c['spx_21_up']:.0%}; "
            f"worst +21d {c['spx_21_worst'] * 100:.1f}%, worst intra-month low {c['min21_worst'] * 100:.1f}%, "
            f"worst spread {c['vrp21_worst']:+.1f} pts; a >10% drop within 21 days followed "
            f"{c['min21_share_below_10']:.0%} of events vs {u['min21_share_below_10']:.0%} of all days.")
    ours2 = (f"After a +50% VIX jump in 5 days (n={jmp['n']}): VIX {jmp['vix_21_mean']:+.1f} pts and SPX "
             f"{jmp['spx_21_mean'] * 100:+.1f}% on average 21 days later (up {jmp['spx_21_up']:.0%}); "
             f"spread {jmp['vrp21_mean']:+.1f} pts; worst intra-month low {jmp['min21_worst'] * 100:.1f}%.")
    claims = [
        claim("Allocate more to short premium as VIX rises (sell when volatility is high)",
              "lit:sizing-risk", ours, v1, ["lit:santaclara-saretto-2009", "lit:wysocki-2026"]),
        claim("Volatility spikes mean-revert, so short-vol entries after a spike are paid", "lit:vrp",
              ours2, "agrees" if jmp["vix_21_mean"] < 0 and jmp["vrp21_mean"] > 0 else "mixed",
              ["lit:cheng-2019"]),
    ]
    chart = {"horizons": [{"set": k, "h": h, "spx_mean": s.get(f"spx_{h}_mean"),
                           "spx_up": s.get(f"spx_{h}_up"), "vix_change": s.get(f"vix_{h}_mean")}
                          for k, s in (("all days", u), ("VIX>30", c), ("+50% jump", jmp))
                          for h in HORIZONS],
             "events_cross30": c["list"], "events_jump50": jmp["list"]}
    return artefact(
        "vix_spikes", claim_text=claims[0]["text"], claims=claims, our_value=ours,
        data="Yahoo ^VIX and ^GSPC daily closes", period=f"{res['start']} to {res['end']}",
        method=("Events de-clustered 21 trading days apart. Forward SPX price returns at +5/+21/+63 "
                "trading days, VIX change, the lowest close in the next 21 days, and VIX minus the next "
                "21 days' realized vol (zenith.beta.hedge.realized)."),
        caveats=["Few independent events (dozens), dominated by 2008-09 and 2020; one more crash "
                 "would move the averages.",
                 "Price returns, no dividends; no option P&L: a positive average spread does not mean "
                 "short options at the spike were safe (see the worst cases).",
                 "Margin requirements rise with VIX, so a seller may be forced to cut size exactly here "
                 "(lit:santaclara-saretto-2009)."],
        verdict=v1, chart_data=chart, literature=["lit:santaclara-saretto-2009"], stats=res)
