"""BACKTEST glossary — the single source of hover text.

Metrics, rule parameters, regimes and indicators all read their info-badge
text from here (indicators carry theirs in ``indicators/catalog.py`` and are
exposed through :func:`indicator_help`). Written for a reader with no stats or
finance background: what the number means, which way is good, one citation.
Modelled on ``zenith/volscreen/glossary.py``.
"""

from __future__ import annotations

# key -> (label, plain-English help, citation)
METRICS: dict[str, tuple[str, str, str]] = {
    "CAGR": ("CAGR", "Compound annual growth rate: the steady yearly return that would turn the start value into "
             "the end value.", ""),
    "Vol": ("Volatility", "How much returns swing, annualised (standard deviation × √bars per year). Higher = a "
            "bumpier ride.", ""),
    "Sharpe": ("Sharpe ratio", "Return per unit of risk: average return ÷ volatility, annualised. 0.5 is decent "
               "for one strategy, above 1 is rare and worth suspecting.", "Sharpe (1966)"),
    "Sortino": ("Sortino ratio", "Like Sharpe but only counts downside swings as risk.", "Sortino & Price (1994)"),
    "MaxDD": ("Max drawdown", "The worst fall from a peak to a later low. The pain you would have had to sit "
              "through.", ""),
    "MaxDD_bars": ("Max drawdown length", "Longest stretch, in bars, spent below a previous high.", ""),
    "Calmar": ("Calmar ratio", "CAGR ÷ |max drawdown|. Return earned per unit of worst pain.", "Young (1991)"),
    "HitRate": ("Hit rate", "Share of bars with a positive return. A low hit rate can still make money if winners "
                "are big — expected value beats win rate.", "MemLabs · Introduction to Quantitative Trading"),
    "ProfitFactor": ("Profit factor", "Sum of gains ÷ sum of losses. Above 1 makes money; 1.3+ is solid.", ""),
    "Skew": ("Skew", "Is the return distribution lopsided? Negative = occasional big losses (selling insurance); "
             "positive = occasional big wins (trend following).", ""),
    "TailRatio": ("Tail ratio", "Size of the best 5% of returns ÷ size of the worst 5%. Above 1 = fatter right "
                  "tail.", ""),
    "Ulcer": ("Ulcer index", "Depth and length of drawdowns combined (root-mean-square drawdown).", "Martin (1987)"),
    "PSR": ("Probabilistic Sharpe", "Probability the true Sharpe is above zero, given how long the test is and how "
            "skewed / fat-tailed returns are.", "Bailey & López de Prado (2012)"),
    "DSR": ("Deflated Sharpe", "PSR again, but the bar is raised for every variant you tried (the TRIALS chip). "
            "Above 90% = probably not luck.", "Bailey & López de Prado (2014)"),
    "Turnover": ("Turnover", "How much of the position is traded per year. High turnover = high costs.", ""),
    "exposure": ("Exposure", "Share of time the strategy holds a position.", ""),
}

# Generic help for the classic rules' parameters (indicator params carry their own).
PARAM_HELP: dict[str, str] = {
    "fast": "Length of the fast average, in days. Shorter = reacts sooner but whipsaws more.",
    "slow": "Length of the slow average, in days. Must be longer than fast.",
    "smooth": "Extra smoothing of the signal, in days. Higher = fewer trades and lower costs.",
    "lookback": "How many days of history the rule looks at.",
    "skip": "Days skipped at the end of the lookback (e.g. skip the last month to avoid short-term reversal).",
    "speed": "Overall speed of the moving-average ribbon: below 1 = slower, above 1 = faster.",
    "entry_lb": "Breakout window: buy a new N-day high, sell a new N-day low.",
    "exit_lb": "Exit window: close when price makes an N-day low (long) or high (short).",
    "atr_period": "Days used to measure the average true range (volatility in price units).",
    "add_atr": "Add another unit every time price moves this many ATRs in your favour (pyramiding).",
    "max_units": "Maximum number of pyramided units.",
    "stop_atr": "Stop distance in ATRs from the last entry.",
    "band": "Dead zone around zero inside which the rule stays flat.",
    "pre": "Trading days before month end to be long.",
    "post": "Trading days after month start to be long.",
    "side": "+1 follows the signal, −1 does the opposite.",
}

REGIME_HELP = ("A regime is a stretch of time when the market behaves in one way (calm vs stressed, trending vs "
               "choppy). Labels here use only data available at each bar — thresholds come from the past, never "
               "the full sample — so you could have known the label on that day.")


def metric_label(key: str) -> str:
    return METRICS.get(key, (key, "", ""))[0]


def metric_help(key: str) -> str:
    lab, h, cite = METRICS.get(key, (key, "", ""))
    return f"{h} ({cite})" if cite else h


def param_help(rule_cls, name: str) -> str:
    own = getattr(rule_cls, "param_help", None) or {}
    return own.get(name) or PARAM_HELP.get(name, "")


def indicator_help(key: str) -> str:
    from .indicators import CATALOG
    i = CATALOG.get(key)
    return i.hover if i else ""
