"""Ready-made CONDITIONS strategies (classic, published rules) to start from."""

from __future__ import annotations


def I(key, **params):  # noqa: E743 — tiny DSL for readability
    return {"src": "ind", "key": key, "params": params}


def N(v):
    return {"src": "num", "value": v}


CLOSE = {"src": "close"}


def g(*rows, join="AND"):
    return {"join": join, "rows": list(rows)}


def r(a, op, b):
    return {"a": a, "op": op, "b": b}


TEMPLATES: dict[str, dict] = {
    "Connors RSI(2) pullback": {
        "why": "Buy sharp short-term dips, but only inside a long-term uptrend; exit on the first bounce. "
               "Connors & Alvarez (2009).",
        "conds": {"long_entry": g(r(I("rsi2", length=2), "<", N(10)), r(CLOSE, ">", I("sma", length=200))),
                  "long_exit": g(r(CLOSE, ">", I("sma", length=5)))},
    },
    "Faber 10-month trend": {
        "why": "Own the market above its ~10-month (210-day) average, step aside below it. Faber (2007).",
        "conds": {"long_entry": g(r(CLOSE, ">", I("sma", length=210))),
                  "long_exit": g(r(CLOSE, "<", I("sma", length=210)))},
    },
    "Golden / death cross": {
        "why": "Long when the 50-day average crosses above the 200-day, out when it crosses back below.",
        "conds": {"long_entry": g(r(I("sma", length=50), "crosses above", I("sma", length=200))),
                  "long_exit": g(r(I("sma", length=50), "crosses below", I("sma", length=200)))},
    },
    "Bollinger reversion (long/short)": {
        "why": "Buy closes below the lower band, short closes above the upper band, exit at the middle. "
               "Mean reversion; fails in strong trends.",
        "conds": {"long_entry": g(r(I("bb_pctb", length=20, mult=2.0), "<", N(0.0))),
                  "long_exit": g(r(I("bb_pctb", length=20, mult=2.0), ">", N(0.5))),
                  "short_entry": g(r(I("bb_pctb", length=20, mult=2.0), ">", N(1.0))),
                  "short_exit": g(r(I("bb_pctb", length=20, mult=2.0), "<", N(0.5)))},
    },
    "Donchian breakout (Turtle-style)": {
        "why": "Buy a 55-bar high, sell a 20-bar low; short the mirror image. Turtle traders; Hurst, Ooi & "
               "Pedersen (2017).",
        "conds": {"long_entry": g(r(I("donchian", length=55), ">", N(0.99))),
                  "long_exit": g(r(I("donchian", length=20), "<", N(-0.99))),
                  "short_entry": g(r(I("donchian", length=55), "<", N(-0.99))),
                  "short_exit": g(r(I("donchian", length=20), ">", N(0.99)))},
    },
    "IBS weak-close bounce": {
        "why": "Buy when the bar closes in its bottom 20% while above the 200-day average; exit on a strong "
               "close. Pagonidis (2014).",
        "conds": {"long_entry": g(r(I("ibs"), "<", N(0.2)), r(CLOSE, ">", I("sma", length=200))),
                  "long_exit": g(r(I("ibs"), ">", N(0.7)))},
    },
    "Trend with ADX confirmation": {
        "why": "Follow the 50/200 trend, but only enter when ADX says a trend is actually present.",
        "conds": {"long_entry": g(r(I("ema", length=50), ">", I("ema", length=200)), r(I("adx", length=14), ">",
                                                                                       N(20))),
                  "long_exit": g(r(I("ema", length=50), "<", I("ema", length=200))),
                  "short_entry": g(r(I("ema", length=50), "<", I("ema", length=200)), r(I("adx", length=14), ">",
                                                                                        N(20))),
                  "short_exit": g(r(I("ema", length=50), ">", I("ema", length=200)))},
    },
}
