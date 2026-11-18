"""Ready-made example ETFs (the user's seven designs).

Phase-1 presets are fully backtest-valid (price-derived). Phase-2 presets use
the fundamental snapshot (FCF, margin, sector/industry) — now wired up, but
still carry a look-ahead/survivorship caveat (snapshot held constant across
history), surfaced in the UI via ``screens.spec_is_lookahead``.
"""

from __future__ import annotations

from .screens import ETFSpec, SelectionSpec, FilterRule

PRESETS: list[ETFSpec] = [
    # Lightweight baskets first (fast to load) — universe screens follow.
    ETFSpec(
        name="Equal-weight all sectors",
        selection=SelectionSpec(universe="Sector ETFs"),
        weighting="equal", rebalance="Quarterly", direction="long",
        notes="Equal-weight basket of the sector SPDRs, rebalanced quarterly.",
    ),
    ETFSpec(
        name="QQQ / IEF / GLD / EFA (quarterly)",
        selection=SelectionSpec(explicit=["QQQ", "IEF", "GLD", "EFA"]),
        weighting="equal", rebalance="Quarterly", direction="long",
        notes="Classic multi-asset basket rebalanced quarterly.",
    ),
    ETFSpec(
        name="IWB Top-10 prior-calendar-year (annual)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="calendar_return", top_n=10),
        weighting="equal", rebalance="Annual", direction="long",
        notes="Buy last calendar year's 10 best and hold the next year. Universe "
              "is a current large-cap proxy (survivorship caveat). First load "
              "fetches the universe — give it a moment.",
    ),
    ETFSpec(
        name="IWB Top-10 rolling 12-month momentum",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="trailing_return", rank_lookback=252,
                                top_n=10),
        weighting="equal", rebalance="Monthly", direction="long",
        notes="Top 10 by trailing 12-month return, rebalanced monthly. First load "
              "fetches the universe — give it a moment.",
    ),
    ETFSpec(
        name="Long top-5 beta / short bottom-5 beta (market-neutral)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="beta", top_n=5, bottom_n=5),
        weighting="equal", rebalance="Monthly", direction="long_short",
        notes="Dollar-neutral high-minus-low beta (price-derived, fully valid).",
    ),
    # --- Phase 2 (use the fundamental snapshot — caveated) ---
    ETFSpec(
        name="Long top-5 beta / short 5 least-profitable (market-neutral)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="beta",
                                short_rank_factor="profit_margin",
                                top_n=5, bottom_n=5),
        weighting="equal", rebalance="Monthly", direction="long_short",
        phase=2, enabled=True,
        notes="The user's example 5 in full: long the 5 highest-beta names, short "
              "the 5 lowest profit-margin names. Margin is a current snapshot → "
              "look-ahead/survivorship caveat applies.",
    ),
    ETFSpec(
        name="Free-cash-flow / share ≤ 3 (value)",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                filters=[FilterRule("fcf_per_share", "<=", 3.0)]),
        weighting="equal", rebalance="Quarterly", direction="long",
        phase=2, enabled=True,
        notes="Holds every name with current FCF/share ≤ $3, equal-weighted. "
              "Fundamental snapshot → look-ahead/survivorship caveat applies.",
    ),
    ETFSpec(
        name="Insurance names with 5-yr dividend growth",
        selection=SelectionSpec(universe="IWB (large-cap proxy)",
                                rank_factor="dividend_growth", top_n=20,
                                filters=[FilterRule("industry", "contains", "insurance")]),
        weighting="equal", rebalance="Annual", direction="long",
        phase=2, enabled=True,
        notes="Insurance industry (snapshot) ranked by 5-yr dividend growth "
              "(dividend history is point-in-time valid; the industry tag is a "
              "snapshot → survivorship caveat).",
    ),
]


def by_name(name: str) -> ETFSpec | None:
    return next((p for p in PRESETS if p.name == name), None)
