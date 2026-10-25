"""repro:expected_move — how often does SPX finish (and stay) inside the
VIX-implied 1-sigma move over the next 30 calendar days?

Band at t: S_t x (1 +/- VIX_t x sqrt(30/365)). Outcome at the last close on or
before t + 30 days. 'Touch' = any close outside the band along the way.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import artefact, claim, rnd

DAYS = 30
VIX_BUCKETS = [(0, 15), (15, 20), (20, 30), (30, 200)]


def compute_rows(S: pd.Series, vix: pd.Series, days: int = DAYS) -> pd.DataFrame:
    """One row per start date. vix decimal. Pure."""
    j = pd.concat([S.rename("S"), vix.rename("iv")], axis=1, join="inner").dropna()
    idx, Sa, iv = j.index, j["S"].to_numpy(), j["iv"].to_numpy()
    rows = []
    for i in range(len(idx)):
        end = idx[i] + pd.Timedelta(days=days)
        if end > idx[-1]:
            break
        e = int(idx.searchsorted(end, side="right")) - 1
        if e <= i:
            continue
        em = iv[i] * math.sqrt(days / 365.0)
        path = Sa[i + 1:e + 1] / Sa[i] - 1.0
        rows.append({"date": idx[i], "vix": iv[i] * 100, "em": em, "ret": path[-1],
                     "up_break": path[-1] > em, "down_break": path[-1] < -em,
                     "touch": bool((np.abs(path) > em).any()),
                     "excess_sigma": path[-1] / em})
    return pd.DataFrame(rows)


def _summ(df: pd.DataFrame) -> dict:
    inside = ~(df["up_break"] | df["down_break"])
    up = df[df["up_break"]]
    dn = df[df["down_break"]]
    return {"n": int(len(df)), "inside": rnd(inside.mean(), 4), "touch": rnd(df["touch"].mean(), 4),
            "up_break": rnd(df["up_break"].mean(), 4), "down_break": rnd(df["down_break"].mean(), 4),
            "mean_up_overshoot_sigma": rnd((up["excess_sigma"] - 1).mean(), 3) if len(up) else None,
            "mean_down_overshoot_sigma": rnd((-dn["excess_sigma"] - 1).mean(), 3) if len(dn) else None,
            "worst_sigma": rnd(df["excess_sigma"].min(), 2), "best_sigma": rnd(df["excess_sigma"].max(), 2),
            "realized_sd_sigma": rnd(df["excess_sigma"].std(), 3)}


def compute(S: pd.Series, vix: pd.Series) -> dict:
    df = compute_rows(S, vix)
    by_vix = [{"bucket": f"{lo}-{hi if hi < 200 else '+'}",
               **_summ(df[(df["vix"] >= lo) & (df["vix"] < hi)])}
              for lo, hi in VIX_BUCKETS if len(df[(df["vix"] >= lo) & (df["vix"] < hi)])]
    dec = (df["date"].dt.year // 10) * 10
    by_decade = [{"decade": f"{int(d)}s", **_summ(g)} for d, g in df.groupby(dec)]
    hist = pd.cut(df["excess_sigma"].clip(-4, 4), np.arange(-4.0, 4.01, 0.25), right=False) \
        .value_counts(normalize=True).sort_index()
    return {"overall": _summ(df), "by_vix": by_vix, "by_decade": by_decade,
            "hist": [{"bin_lo": rnd(k.left, 2), "share": rnd(v, 4)} for k, v in hist.items()],
            "start": df["date"].min().date().isoformat(), "end": df["date"].max().date().isoformat()}


def build(res: dict) -> dict:
    o = res["overall"]
    v1 = "agrees" if abs(o["inside"] - 0.683) <= 0.05 else "mixed"
    rarer = o["down_break"] < o["up_break"]
    further = (o["mean_down_overshoot_sigma"] or 0) > (o["mean_up_overshoot_sigma"] or 0)
    v2 = "agrees" if rarer and further else "contradicts" if not (rarer or further) else "mixed"
    ours1 = (f"SPX finished inside the VIX 1-sigma band {o['inside']:.1%} of the time (normal "
             f"theory 68.3%), touched outside it on {o['touch']:.1%} of paths; realized moves had "
             f"{o['realized_sd_sigma']:.2f}x the implied sigma.")
    ours2 = (f"Breaks: up {o['up_break']:.1%} vs down {o['down_break']:.1%}; average overshoot beyond "
             f"the band +{o['mean_up_overshoot_sigma']:.2f} sigma up vs {o['mean_down_overshoot_sigma']:.2f} "
             f"sigma down; worst {o['worst_sigma']:.1f} sigma.")
    claims = [
        claim("The expected move (about 1 standard deviation; 16-delta strikes) contains the "
              "underlying about 68% of the time", "seed", ours1, v1, ["lit:bondarenko-2019"]),
        claim("Breaches are asymmetric: the market breaks the downside less often but further "
              "(crash risk the sellers are paid for)", "lit:vrp", ours2, v2,
              ["lit:bollerslev-todorov-2011"]),
    ]
    return artefact(
        "expected_move", claim_text=claims[0]["text"], claims=claims, our_value=ours1,
        data="Yahoo ^GSPC and ^VIX daily closes", period=f"{res['start']} to {res['end']}",
        method=("Band = S x VIX x sqrt(30/365) around each day's close; finish = last close on or before "
                "t+30 calendar days; touch = any close outside the band in between. Overlapping windows."),
        caveats=["VIX is the variance-swap implied vol (weighted toward OTM puts), slightly above "
                 "ATM IV, so the band is a bit wider than the ATM expected move brokers quote.",
                 "Price only, no dividends; drift over 30 days shifts results by a fraction of a point.",
                 "Touch uses closes; intraday touches are more frequent.",
                 "Overlapping 30-day windows: n overstates independent observations ~21-fold."],
        verdict=v1,
        chart_data={"by_vix": res["by_vix"], "by_decade": res["by_decade"], "sigma_hist": res["hist"]},
        literature=["lit:bondarenko-2019"], stats=res)
