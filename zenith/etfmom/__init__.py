"""ETF MOMENTUM — the MOMENTUM engine pointed at Zenith's full ETF universe.

This is deliberately NOT a new signal. Every trading day, every ETF in the
curated universe is scored on the SAME six momentum factors as the Russell
1000 engine (see zenith/mom/__init__.py for the full citation list), with the
SAME weights (config.MOM_WEIGHTS), the SAME horizon weights, the SAME moving
averages and the SAME -20..+20 composite scale and state bands. The factor
math is imported from `zenith.mom.factors` / `zenith.mom.engine`, never
copied, so the two tabs can never silently drift apart — an ETF's +12.4 and a
stock's +12.4 mean the same thing.

What differs is only what has to:

  * UNIVERSE — a curated union of three committed lists (cas.universe's
    ~597-name master ETF list, its ~335-name factor-rotation tagged set, and
    the 574-name Morningstar strategic-beta catalog → ~935 unique), gated for
    leveraged/inverse funds. Unlike mom/mvt's own ETF universe this one KEEPS
    near-duplicates: mvt collapses SPY/VOO/IVV/SPLG to one name because a
    pairwise spread between two ~identical funds divides by a near-zero
    spread vol, but a ranked momentum list has no such problem and the user
    should be able to find the fund they actually hold.
  * TAXONOMY — ETFs have no GICS sector/industry. A normalized Morningstar
    category (level 2) and a mechanical asset-class rollup of it (level 1)
    take their place. See universe.py; the rollup is a documented rule list
    over Morningstar's own vocabulary, not an invented taxonomy.
  * THE 6th FACTOR IS READ, NOT RECOMPUTED — mom.yml already scores this ETF
    universe with the Multivariate Trend pairwise engine every night and
    commits data/mom/mvt/etfs_latest.json. etfmom reads it (and inherits a
    score across mvt's own near-duplicate clusters, which is exactly what
    makes keeping the duplicates viable). See mvt_link.py.

Evidence tier B — one notch BELOW the equity engine's B+, and the reason is
stated rather than glossed. Time-series momentum is if anything better
evidenced across asset classes than within the equity cross-section
(Moskowitz, Ooi & Pedersen 2012 test it on 58 futures markets; Asness,
Moskowitz & Pedersen 2013, "Value and Momentum Everywhere"). But two things
genuinely weaken the composite here: (1) the cross-sectional leg pools
instruments that are not comparable — a short-duration Treasury fund and a
single-country equity fund sit in one percentile ranking, whereas
cross-sectional momentum's evidence base is WITHIN an asset class; and (2)
these ETFs are not independent bets — dozens track near-identical indices, so
a breadth reading here is a weaker statement than the same number across the
Russell 1000. Both are surfaced in the UI. Promotion to B+ comes only from
this engine's own out-of-sample IC accruing in picks.json, never from a
backtest fitted after the fact.

Prices are yfinance daily OHLC with auto_adjust=True (splits AND distributions
applied) — the only self-consistent choice for return math and trailing
high/low breakout levels alike. Computed offline in a GitHub Action that
commits JSON under data/etfmom/; views read only those committed artefacts.
"""

from __future__ import annotations

import json

from ..config import ETFMOM_FILES
# The six factors, their labels and the horizon grid are MOMENTUM's, re-exported
# here so callers can `from zenith.etfmom import FACTORS` without needing to
# know that the definition lives one package over. Re-export, never redefine.
from ..mom import FACTORS, FACTOR_LABELS, HORIZONS, HORIZON_LABELS  # noqa: F401

DISCLAIMER = ("ETF MOMENTUM scores Zenith's curated ETF universe on the same six documented "
              "momentum dimensions as the Russell 1000 engine (time-series, breakout, "
              "cross-sectional, trend speed, momentum strength, multivariate trend), blended "
              "with the same weights into the same transparent -20..+20 composite, so the two "
              "tabs are directly comparable. Cross-sectional momentum is weaker evidence across "
              "mixed asset classes than within one, and overlapping funds make breadth readings "
              "softer than they look. Decision-support and a research monitor, not investment "
              "advice.")

UNIVERSE_NOTE = ("This universe is a curated union of three committed lists — the CAS master ETF "
                 "list, the factor-rotation tagged set, and the Morningstar strategic-beta "
                 "catalog — not every US-listed ETF (there are ~3,000+). Leveraged and inverse "
                 "funds are excluded by an explicit list, a name regex, and an empirical "
                 "vol/correlation backstop. Near-duplicate funds (SPY/VOO/IVV) are deliberately "
                 "KEPT so you can find the one you actually hold, which means breadth counts "
                 "overstate how many independent bets are on the list. The universe is not "
                 "survivorship-controlled: a fund that closed simply stops appearing, so "
                 "historical cross-sections tilt toward survivors.")

XSEC_CAVEAT = ("Cross-sectional momentum ranks each ETF against every other ETF here, which pools "
               "instruments that are not really comparable — a short-duration Treasury fund and a "
               "single-country equity fund land in one percentile ranking. Read this factor "
               "within an asset class (the Categories view) rather than across the whole list.")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load(name: str, default=None):
    return _read(ETFMOM_FILES[name], default if default is not None else {})


def save(name: str, obj, indent: int | None = 2) -> None:
    """Write an ETFMOM artefact. Large daily artefacts (scores/detail/picks)
    pass indent=None for the same reason mom.save does: at ~900 rows/day,
    pretty-printing costs hundreds of MB of git history a year for something
    nobody diffs by eye. The small ones keep indent=2 so diffs stay readable."""
    ETFMOM_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    ETFMOM_FILES[name].write_text(json.dumps(obj, indent=indent, ensure_ascii=False),
                                  encoding="utf-8")
