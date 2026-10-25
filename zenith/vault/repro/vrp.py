"""repro:vrp_spx — does VIX overstate the SPX volatility that follows?

VIX_t (30-calendar-day implied) vs realized vol of SPX daily returns over the
next 21 trading days, annualized with zenith.beta.hedge.realized. Split by a
252-day VIX "IV rank" (tastylive's IVR: where VIX sits in its 1-year range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...beta.hedge import realized
from . import artefact, claim, rnd

H = 21
RANK_WIN = 252
BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100.01)]


def forward_realized(close: pd.Series, h: int = H) -> pd.Series:
    """Realized vol over the h trading days after each date (NaN near the end)."""
    ret = close.pct_change()
    vals = ret.to_numpy()
    out = np.full(len(close), np.nan)
    for i in range(len(close) - h):
        v = realized(pd.Series(vals[i + 1:i + 1 + h]), h)
        out[i] = np.nan if v is None else v
    return pd.Series(out, index=close.index)


def iv_rank(iv: pd.Series, win: int = RANK_WIN) -> pd.Series:
    lo = iv.rolling(win, min_periods=win).min()
    hi = iv.rolling(win, min_periods=win).max()
    return ((iv - lo) / (hi - lo) * 100.0).where(hi > lo)


def _row(j: pd.DataFrame) -> dict:
    s = j["iv"] - j["rv"]
    return {"n": int(len(j)), "vix": rnd(j["iv"].mean() * 100, 2),
            "rv": rnd(j["rv"].mean() * 100, 2), "spread": rnd(s.mean() * 100, 2),
            "spread_median": rnd(s.median() * 100, 2),
            "share_iv_gt_rv": rnd((s > 0).mean(), 4),
            "ratio_rv_iv": rnd((j["rv"] / j["iv"]).mean(), 3),
            "spread_p5": rnd(s.quantile(0.05) * 100, 2)}


def compute(vix: pd.Series, spx: pd.Series) -> dict:
    """vix as a decimal (0.20), spx close. Pure."""
    rv = forward_realized(spx)
    j = pd.concat([vix.rename("iv"), rv.rename("rv"), iv_rank(vix).rename("ivr")],
                  axis=1, join="inner").dropna(subset=["iv", "rv"])
    overall = _row(j)
    by_bucket = []
    for lo, hi in BUCKETS:
        sub = j[(j["ivr"] >= lo) & (j["ivr"] < hi)]
        if len(sub):
            by_bucket.append({"bucket": f"{lo}-{int(min(hi, 100))}", **_row(sub)})
    by_year = [{"year": int(y), **_row(g)} for y, g in j.groupby(j.index.year)]
    return {"overall": overall, "by_ivr": by_bucket, "by_year": by_year,
            "start": j.index[0].date().isoformat(), "end": j.index[-1].date().isoformat()}


def build(res: dict) -> dict:
    o, b = res["overall"], res["by_ivr"]
    v1 = "agrees" if o["share_iv_gt_rv"] > 0.5 and o["spread"] > 0 else "contradicts"
    if len(b) < 2:            # flat VIX: no IV-rank range to bucket
        b = b or [{"bucket": "n/a", **o}]
    top, bot = b[-1], b[0]
    pts_up = top["spread"] > bot["spread"]
    share_up = top["share_iv_gt_rv"] > bot["share_iv_gt_rv"]
    v2 = ("mixed" if top is bot else "agrees" if pts_up and share_up
          else "contradicts" if not (pts_up or share_up) else "mixed")
    yrs_neg = [y["year"] for y in res["by_year"] if y["spread"] is not None and y["spread"] < 0]
    ours1 = (f"VIX exceeded the next 21 days' realized vol on {o['share_iv_gt_rv']:.0%} of days; "
             f"mean spread {o['spread']:+.1f} vol pts (median {o['spread_median']:+.1f}); "
             f"realized averaged {o['ratio_rv_iv']:.2f}x VIX. Years with a negative mean spread: "
             f"{', '.join(map(str, yrs_neg)) or 'none'}.")
    ours2 = (f"IV rank {top['bucket']}: spread {top['spread']:+.1f} pts, IV>RV {top['share_iv_gt_rv']:.0%} "
             f"(n={top['n']}); IV rank {bot['bucket']}: {bot['spread']:+.1f} pts, "
             f"{bot['share_iv_gt_rv']:.0%} (n={bot['n']}). Worst 5% of spreads: "
             f"{top['spread_p5']:+.1f} vs {bot['spread_p5']:+.1f} pts.")
    claims = [
        claim("Implied volatility overstates realized volatility most of the time, so selling "
              "premium has positive expected value", "lit:vrp", ours1, v1,
              ["lit:carr-wu-2009", "lit:bondarenko-2019"]),
        claim("Sell more premium when IV rank or IV percentile is high", "lit:iv-predictors",
              ours2, v2, ["lit:fallon-park-2016", "lit:cheng-2019"]),
    ]
    chart = {"by_year": [{"year": y["year"], "vix": y["vix"], "rv": y["rv"], "spread": y["spread"]}
                         for y in res["by_year"]],
             "by_ivr": [{"bucket": x["bucket"], "spread": x["spread"],
                         "share_iv_gt_rv": x["share_iv_gt_rv"], "n": x["n"]} for x in b]}
    return artefact(
        "vrp_spx", claim_text=claims[0]["text"], claims=claims, our_value=ours1,
        data="Yahoo ^VIX close and ^GSPC close (daily)",
        period=f"{res['start']} to {res['end']}",
        method=("For each day: VIX/100 vs the annualized standard deviation of the next 21 "
                "trading days' SPX close-to-close returns (zenith.beta.hedge.realized). "
                "IV rank = VIX position in its trailing 252-day high-low range. Days overlap, "
                "so n overstates independent observations about 21-fold."),
        caveats=["VIX is a 30-calendar-day variance-swap rate and is quoted as vol; comparing "
                 "it to 21-day realized standard deviation is the usual approximation and "
                 "slightly flatters the spread (Jensen gap between variance and vol).",
                 "A positive vol spread is not the seller's P&L: losses scale with gamma and "
                 "jumps, so a short option can lose in a month where the average spread was "
                 "positive.",
                 "Index only; single-stock premia are smaller (lit:carr-wu-2009).",
                 "IV-rank buckets are not a tradable test of IVR timing: high-IVR days cluster "
                 "in a few episodes."],
        verdict=v1, chart_data=chart, literature=["lit:bondarenko-2019", "lit:carr-wu-2009"],
        stats=res)
