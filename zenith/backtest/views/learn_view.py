"""BACKTEST / LEARN — what each guard rail and test is for, in plain words."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from ...config import THEME as T
from ...ui_theme import key_findings, section

_FINDINGS = [
    {"stat": "Half or more of published trading 'anomalies' fail to replicate once data mining is accounted for.",
     "cite": "Harvey, Liu & Zhu (2016); Hou, Xue & Zhang (2020)"},
    {"stat": "With enough variants tried, a strategy with zero true skill will show an impressive backtest Sharpe "
             "by luck.", "cite": "Bailey, Borwein, López de Prado & Zhu (2014)"},
]

TOPICS = [
    ("GUARD RAILS — always on", 0, [
        ("Next-bar fills", "A signal computed from today's close trades at tomorrow's price. Trading on the same bar "
         "that produced the signal is the most common way backtests cheat without meaning to."),
        ("Causal scaling and sizing", "Every number used to size a position on day t — forecast scalar, volatility, "
         "portfolio weights, diversification multiplier — is estimated from data before t. A unit test jolts the "
         "future and checks that no past position moves."),
        ("Costs on by default", "Commission, slippage and (for futures) a per-contract fee are charged on every trade. "
         "BUILD also shows Sharpe at 0×, 1×, 2× and 3× costs: a rule that dies at 2× has no margin for real fills."),
        ("Locked holdout", "The most recent slice of history is hidden from every step until you reveal it once. The "
         "reveal is logged; change the strategy afterwards and it's flagged CONTAMINATED."),
        ("Trial counter", "Every variant you run on the same data and holdout is counted, including each cell of a "
         "sweep. The deflated Sharpe ratio uses that count to raise the bar a result must clear."),
        ("Benchmark beside every number", "A Sharpe means nothing alone. Each headline sits next to the cheapest "
         "alternative that does the same job — holding the asset, cash, the S&P or a 60/40."),
    ]),
    ("THE RESEARCH STEPS", 1, [
        ("01 Define", "Write down why the edge should exist before you look. If you can't say who is on the other "
         "side of the trade and why they keep losing, a good backtest is more likely luck."),
        ("02 Explore", "Before building anything, bucket a reading and compare the next bar's return across buckets "
         "in two separate halves of the data. Real effects show up in both halves; MemLabs' first test for any idea."),
        ("BUILD › Regimes", "How the strategy behaved in each market state, using labels you could have known at "
         "the time. A regime filter must then beat random filters with the same exposure."),
        ("03 Sensitivity", "Sweep the parameters and look at the shape. A broad plateau of good results is evidence; "
         "a lone spike surrounded by poor results is overfitting. Pick the plateau's centre, not the peak."),
        ("04 Permutation", "Shuffle the bars so the market keeps its return distribution but loses its time "
         "structure, and re-run. If your strategy can't beat most shuffles — especially when each shuffle is allowed "
         "to re-tune too — the 'edge' is something any random series would show."),
        ("04 Noise", "Jitter the prices slightly. A real effect degrades gracefully; a rule keyed to exact prices "
         "collapses."),
        ("04 Synthetic markets", "Known-answer worlds. A trend rule should win where trends exist, lose in "
         "mean-reverting worlds and earn about zero on a random walk."),
        ("04 Walk-forward", "Re-tune on the past, trade the next unseen slice, repeat. The stitched result is what "
         "the tuning process would really have earned."),
        ("04 Overfitting", "PBO, CPCV and the Reality Check run on a whole family of variants: does the in-sample "
         "winner hold up out of sample, and is the best variant better than luck after searching?"),
        ("04 Challenge & stress", "Odds of reaching a target before a loss limit (prop-firm style), crisis replays "
         "and synthetic shocks: vol x2, gaps, triple costs."),
        ("05 Filters", "Each filter is tested by adding and removing it. If its confidence interval spans zero it "
         "isn't adding value — and it still cost you a trial."),
        ("06 Attribution", "Split returns into beta (what holding the benchmark would have given) and what's left. "
         "Check up- vs down-market behaviour, regimes, long vs short, and factor loadings."),
        ("07 Out-of-sample", "One look at data the strategy has never seen. Expect a lower Sharpe; a collapse means "
         "the in-sample result was mostly tuning."),
        ("08 Decay", "After going live: CUSUM, the bootstrap cone, monthly ranks and rolling alpha ask whether a bad "
         "stretch is noise, a bug or the edge disappearing."),
        ("ML LAB", "Machine-learning signals with walk-forward, purged validation and a naive baseline; judge them by "
         "net edge, not accuracy."),
    ]),
    ("READING THE NUMBERS", 2, [
        ("Deflated Sharpe (DSR)", "Probability the true Sharpe is above zero after accounting for how many variants "
         "you tried and the returns' skew and fat tails. Above 90% is strong; below 50% is noise."),
        ("Robustness (sensitivity)", "The average of the best neighbourhood divided by the single best cell. Near 1 "
         "means a flat top; under 0.5 means a spike."),
        ("Return from beta", "Share of the average return that the benchmark exposure alone explains. Above ~80% "
         "the strategy is mostly an expensive way to hold the benchmark."),
        ("Convexity γ", "Positive when the strategy does well in big moves either way (the trend-following 'smile'); "
         "negative for option sellers, who do best in quiet markets and worst in crashes."),
        ("Skew", "Short-volatility strategies have steady small gains and rare large losses. Their Sharpe overstates "
         "how pleasant they are to hold: look at the worst months."),
    ]),
    ("DATA LIMITS — all free sources", 4, [
        ("Survivorship", "Universes are today's members. Companies that failed or were removed are missing, which "
         "flatters stock-selection backtests."),
        ("Futures", "Yahoo's continuous contracts are not roll-adjusted, so rolls can appear as jumps and carry is "
         "missing."),
        ("Options", "No free historical option chains exist. Chains are modeled from Cboe's VIX term structure and "
         "SKEW index and calibrated against Cboe's PUT and BXM indices; results are stamped MODELED."),
        ("Fundamentals", "ETF LAB's fundamental factors are today's snapshot applied to all of history — a "
         "look-ahead, flagged on the page."),
    ]),
]


MODES = ["PATH", "LESSONS", "GLOSSARY", "REFERENCE"]

PATH = [
    ("1 · Have an idea", "LAB › 01 DEFINE", "Write why it should work: who loses money to you, and why do they keep "
     "doing it?"),
    ("2 · Look before you build", "LAB › 02 EXPLORE", "Bucket the signal and check the next return in both halves of "
     "the data. No pattern in both halves: stop here, it cost you nothing."),
    ("3 · Build it", "LAB › 02 SIGNAL", "One rule, a BLEND of rules, or IF/AND/OR CONDITIONS — start from a "
     "template. Hover every ? to see what a setting does."),
    ("4 · Read the result", "LAB › 03 BUILD", "Sharpe next to buy & hold, deflated Sharpe, costs at 2x-3x, and the "
     "REGIMES tab: when does it work?"),
    ("5 · Try to break it", "SENSITIVITY · ROBUSTNESS", "Plateau not peak; beat shuffled markets; walk-forward; PBO "
     "and the Reality Check; stress and crisis replays."),
    ("6 · One honest test", "LAB › OUT-OF-SAMPLE", "Reveal the locked data once. Expect less; a collapse means the "
     "backtest was the tuning."),
    ("7 · Combine and monitor", "PORTFOLIO · DECAY", "Mix different strategies (does any optimiser beat 1/N?), then "
     "watch for decay with a rule you set in advance."),
]


def _glossary() -> pd.DataFrame:
    from .. import glossary as G
    from ..indicators import CATALOG, FAMILIES
    from ..ml.core import CV_SCHEMES, MODELS, TARGETS
    from ..portfolio_opt import METHODS
    from ..regime import DETECTORS
    rows = [{"term": lab, "area": "metric", "meaning": h + (f" ({c})" if c else "")}
            for lab, h, c in G.METRICS.values()]
    rows += [{"term": k, "area": "rule parameter", "meaning": v} for k, v in G.PARAM_HELP.items()]
    rows += [{"term": i.label, "area": "indicator · " + FAMILIES[i.family].lower(),
              "meaning": f"{i.hover} How: {i.how} Evidence: {i.evidence}"} for i in CATALOG.values()]
    rows += [{"term": d.label, "area": "regime", "meaning": f"{d.help} {d.cite}"} for d in DETECTORS.values()]
    rows += [{"term": m.label, "area": "ML model", "meaning": m.help} for m in MODELS.values()]
    rows += [{"term": k.replace("_", " "), "area": "validation", "meaning": v} for k, v in CV_SCHEMES.items()]
    rows += [{"term": k.replace("_", " "), "area": "ML target", "meaning": v} for k, v in TARGETS.items()]
    rows += [{"term": k.replace("_", " "), "area": "portfolio weighting", "meaning": v} for k, v in METHODS.items()]
    for _, _, items in TOPICS:
        rows += [{"term": h, "area": "concept", "meaning": b} for h, b in items]
    return pd.DataFrame(rows)


def _reference() -> None:
    for title, idx, items in TOPICS:
        st.markdown(section(title, idx), unsafe_allow_html=True)
        st.markdown("".join(
            f'<div style="display:flex; gap:1.2rem; border-top:1px solid {T.line}; padding:0.5rem 0;">'
            f'<div style="min-width:13rem; font-size:0.72rem; letter-spacing:0.12em; color:{T.text}; '
            f'text-transform:uppercase;">{html.escape(h)}</div>'
            f'<div style="font-size:0.8rem; line-height:1.55; color:{T.muted};">{html.escape(b)}</div></div>'
            for h, b in items), unsafe_allow_html=True)


def render() -> None:
    st.markdown(key_findings(_FINDINGS), unsafe_allow_html=True)
    mode = st.segmented_control("learn", MODES, default="PATH", key="ln_mode", label_visibility="collapsed") or "PATH"
    if mode == "PATH":
        st.markdown(section("THE PATH — from an idea to a strategy you can trust", 0,
                            help="The order the BACKTEST pages are meant to be used in."), unsafe_allow_html=True)
        st.markdown("".join(
            f'<div style="display:grid; grid-template-columns:15rem 14rem 1fr; gap:1rem; border-top:1px solid '
            f'{T.line}; padding:0.6rem 0;"><div style="font-size:0.78rem; letter-spacing:0.08em; color:{T.text};">'
            f'{html.escape(a)}</div><div style="font-size:0.72rem; color:{T.teal}; letter-spacing:0.1em;">'
            f'{html.escape(b)}</div><div style="font-size:0.8rem; color:{T.muted}; line-height:1.5;">'
            f'{html.escape(c)}</div></div>' for a, b, c in PATH), unsafe_allow_html=True)
        st.caption("New to this? Open LESSONS first — eight two-minute experiments behind the ideas the tools use.")
    elif mode == "LESSONS":
        from .learn_lessons import LESSONS
        names = list(LESSONS)
        pick = st.pills("lesson", names, default=names[0], key="ln_pick", label_visibility="collapsed") or names[0]
        st.markdown(section(pick.upper(), 1), unsafe_allow_html=True)
        LESSONS[pick]()
    elif mode == "GLOSSARY":
        g = _glossary()
        q = st.text_input("search", "", key="ln_q", placeholder="e.g. sharpe, RSI, purged, HRP, regime",
                          help="Searches every term, indicator, metric, regime, model and method on the BACKTEST "
                               "pages.").strip().lower()
        if q:
            g = g[g.apply(lambda r: q in (r["term"] + " " + r["area"] + " " + r["meaning"]).lower(), axis=1)]
        areas = sorted(g["area"].str.split(" · ").str[0].unique())
        chosen = st.multiselect("areas", areas, key="ln_area")
        if chosen:
            g = g[g["area"].str.split(" · ").str[0].isin(chosen)]
        st.caption(f"{len(g)} entries")
        st.dataframe(g, use_container_width=True, hide_index=True, height=560,
                     column_config={"meaning": st.column_config.TextColumn(width="large")})
    else:
        _reference()
    st.caption("Sources: Carver, Systematic Trading (2015) and Advanced Futures Trading Strategies (2023); "
               "Masters, Permutation and Randomization Tests for Trading System Development (2020); "
               "Bailey & López de Prado (2014); López de Prado, Advances in Financial Machine Learning (2018); "
               "SharpeEdge and MemLabs (YouTube) — see docs/research/backtest_sources.md.")
