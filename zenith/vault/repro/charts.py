"""Altair charts for the reproductions, built only from each artefact's
pre-aggregated ``chart_data`` (so Phase 06's view can reuse them as is).
House rules: THEME colours (teal = profit/up, coral = loss/down), explicit
heights, labelLimit=0 on category axes."""

from __future__ import annotations

import pandas as pd

from ...config import THEME

RULE_ORDER = ["hold", "p50", "d21", "p50_d21", "p50_d21_stop2x"]
PALETTE = [THEME.text, THEME.teal, THEME.mustard, THEME.coral, THEME.mauve, THEME.navy, THEME.mint,
           THEME.orange]


def _sign_color(alt, field: str):
    return alt.condition(f"datum.{field} >= 0", alt.value(THEME.teal), alt.value(THEME.coral))


def vrp_spx(obj: dict, alt):
    df = pd.DataFrame(obj["chart_data"]["by_year"])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelLimit=0, labelAngle=-90)),
        y=alt.Y("spread:Q", title="VIX minus next-21d realized vol (pts)"),
        color=_sign_color(alt, "spread"),
        tooltip=["year", "vix", "rv", "spread"],
    ).properties(height=260, width="container", title="VIX vs realized SPX vol, mean by year")


def _growth(obj: dict, alt, title: str):
    df = pd.DataFrame(obj["chart_data"]["growth_monthly_2007"])
    names = list(dict.fromkeys(df["index"]))
    return alt.Chart(df).mark_line(strokeWidth=1.6).encode(
        x=alt.X("month:T", title=None),
        y=alt.Y("growth:Q", title="growth of $1 (log)", scale=alt.Scale(type="log")),
        color=alt.Color("index:N", sort=names, title=None,
                        scale=alt.Scale(domain=names, range=PALETTE[:len(names)]),
                        legend=alt.Legend(orient="top", labelLimit=0)),
        tooltip=["index", "month", "growth"],
    ).properties(height=300, width="container", title=title)


def cboe_short_vol(obj: dict, alt):
    return _growth(obj, alt, "Cboe short-vol indices vs S&P 500 TR, from 2007-01")


def cboe_tail_hedge(obj: dict, alt):
    df = pd.DataFrame([e for e in obj["episodes"] if e["index"] in ("SPTR", "PPUT", "CLL")
                       and e["drawdown"] is not None])
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("index:N", title=None, sort=["SPTR", "PPUT", "CLL"], axis=alt.Axis(labelLimit=0)),
        x=alt.X("drawdown:Q", title="peak-to-trough in window", axis=alt.Axis(format="%")),
        color=alt.Color("index:N", legend=None,
                        scale=alt.Scale(domain=["SPTR", "PPUT", "CLL"],
                                        range=[THEME.coral, THEME.teal, THEME.mustard])),
        row=alt.Row("episode:N", title=None, sort=["gfc", "feb2018", "covid", "y2022"],
                    header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=["episode", "index", alt.Tooltip("drawdown:Q", format=".1%")],
    ).properties(height=60, width=520, title="Protection by episode: PPUT and CLL vs S&P 500 TR")


def strangle_mgmt(obj: dict, alt):
    df = pd.DataFrame(obj["chart_data"]["rules"])
    base = alt.Chart(df).encode(y=alt.Y("rule:N", sort=RULE_ORDER, title=None,
                                        axis=alt.Axis(labelLimit=0)))
    win = base.mark_bar(color=THEME.teal).encode(
        x=alt.X("win_rate:Q", title="win rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.5, 1])),
        tooltip=["rule", alt.Tooltip("win_rate:Q", format=".1%")]).properties(width=170, height=170)
    mean = base.mark_bar(color=THEME.mustard).encode(
        x=alt.X("pnl_per_day_bp:Q", title="mean P&L/day (bp of notional)"),
        tooltip=["rule", "pnl_per_day_bp", "mean_pnl_credits"]).properties(width=170, height=170)
    cvar = base.mark_bar(color=THEME.coral).encode(
        x=alt.X("cvar5_credits:Q", title="CVaR 5% (x credit)"),
        tooltip=["rule", "cvar5_credits", "max_loss_credits"]).properties(width=170, height=170)
    return alt.hconcat(win, mean, cvar).properties(
        title="16-delta 45-DTE SPX strangle (model-priced): what management does")


def earnings_drift(obj: dict, alt):
    df = pd.DataFrame(obj["chart_data"]["size_period_A21"])
    df = df[df["n"] > 0]
    pts = alt.Chart(df).mark_point(filled=True, size=70).encode(
        x=alt.X("share_cont:Q", title="share still drifting at +21d", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0.35, 0.65])),
        y=alt.Y("size:N", sort=["small", "mid", "large"], title=None, axis=alt.Axis(labelLimit=0)),
        color=alt.Color("period:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(range=[THEME.mustard, THEME.teal, THEME.navy])),
        yOffset="period:N",
        tooltip=["period", "size", "n", alt.Tooltip("share_cont:Q", format=".1%"),
                 alt.Tooltip("mean_signed_drift:Q", format="+.2%")])
    bars = alt.Chart(df).mark_rule(strokeWidth=2).encode(
        x="ci_lo:Q", x2="ci_hi:Q", y=alt.Y("size:N", sort=["small", "mid", "large"]),
        yOffset="period:N", color=alt.Color("period:N", legend=None))
    half = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=THEME.muted, strokeDash=[4, 3]).encode(x="x:Q")
    return (bars + pts + half).properties(
        height=240, width="container",
        title="Outlier earnings moves (variant A): continuation at +21 days by size and period")


def expected_move(obj: dict, alt):
    rows = []
    for b in obj["chart_data"]["by_vix"]:
        rows += [{"bucket": f"VIX {b['bucket']}", "outcome": "inside", "share": b["inside"], "n": b["n"]},
                 {"bucket": f"VIX {b['bucket']}", "outcome": "broke up", "share": b["up_break"], "n": b["n"]},
                 {"bucket": f"VIX {b['bucket']}", "outcome": "broke down", "share": b["down_break"], "n": b["n"]}]
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("bucket:N", title=None, sort=None, axis=alt.Axis(labelLimit=0)),
        x=alt.X("share:Q", stack="normalize", title="30-day outcome vs VIX 1-sigma band",
                axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", sort=["broke down", "inside", "broke up"], title=None,
                        scale=alt.Scale(domain=["broke down", "inside", "broke up"],
                                        range=[THEME.coral, THEME.grid, THEME.teal]),
                        legend=alt.Legend(orient="top")),
        order=alt.Order("outcome:N"),
        tooltip=["bucket", "outcome", alt.Tooltip("share:Q", format=".1%"), "n"],
    ).properties(height=200, width="container", title="Does SPX stay inside the VIX expected move?")


def vix_spikes(obj: dict, alt):
    df = pd.DataFrame(obj["chart_data"]["horizons"])
    df["h"] = "+" + df["h"].astype(str) + "d"
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("set:N", title=None, sort=["all days", "VIX>30", "+50% jump"], axis=alt.Axis(labelLimit=0, labelAngle=0)),
        y=alt.Y("spx_mean:Q", title="mean SPX return", axis=alt.Axis(format="%")),
        color=alt.Color("set:N", legend=None, scale=alt.Scale(
            domain=["all days", "VIX>30", "+50% jump"], range=[THEME.muted, THEME.teal, THEME.mustard])),
        column=alt.Column("h:N", title=None, sort=["+5d", "+21d", "+63d"]),
        tooltip=["set", "h", alt.Tooltip("spx_mean:Q", format="+.2%"), alt.Tooltip("spx_up:Q", format=".0%")],
    ).properties(height=220, width=170, title="SPX after VIX spikes")


BUILDERS = {"vrp_spx": vrp_spx, "cboe_short_vol": cboe_short_vol, "cboe_tail_hedge": cboe_tail_hedge,
            "strangle_mgmt": strangle_mgmt, "earnings_drift": earnings_drift,
            "expected_move": expected_move, "vix_spikes": vix_spikes}


def chart(name: str, obj: dict):
    import altair as alt
    return BUILDERS[name](obj, alt)
