"""THETA RUN — a turn-based options-portfolio speedrun built on tastytrade mechanics.

THE GAME. Pick a scenario (a random window, a hidden-ticker blind run, or a
scripted crisis such as Volmageddon or COVID), screen a universe of liquid ETFs
and large caps by IV, IV rank, expected move and upcoming earnings, open
strategies from tastytrade-style templates (strangles, iron condors, jade
lizards, credit spreads, covered calls ...), then manage the book: advance a
day, a week or straight to the next event (21 DTE, expiry, earnings, profit
target, max loss, buying-power breach) and close, roll or defend. The run is
scored against buy-and-hold SPY and a mechanical tastytrade bot
(45 DTE / 16 delta / 50% / 21 DTE) on the same tape.

DATA. Prices are REAL: split/dividend-adjusted daily bars from the EPHEMERIS
store. Option prices are MODELED -- real historical chains are not free:

  iv_base   stocks/ETFs: a HAR-lite realized-vol forecast (5/21/63-day, earnings
            reaction days excluded) times a volatility-risk-premium multiplier
            taken from the index each day (VIX / SPY's own forecast, smoothed),
            so single names feel market fear on the same day. SPY = VIX itself.
            Single stocks get a damped multiplier: the index premium is larger
            than the single-stock one [lit: Carr & Wu 2009; Bakshi & Kapadia 2003].
  earnings  each expiry that spans a report adds variance em^2 / T, where em is
            the trailing mean absolute two-day reaction of the last 8 reports
            (past events only). After the report the bump is gone: IV crush.
  skew      iv(K) = atm * (1 - s*0.10*m + 0.03*m^2) with m = ln(K/S)/(atm*sqrt(T));
            s is steeper for index ETFs and rises with VIX.

Every feature at date t uses data up to t only (tests perturb the future and
check the past is unchanged). Earnings DATES are known ahead, as in reality.
Survivorship: the universe is today's liquid names; disclosed in the tab.

MECHANICS follow OPTIONS VAULT: Reg-T style buying power, $1/contract to open
and $0 to close, fills at mid minus a liquidity-tier half-spread, equity options
physically settle at the expiry close (a short put becomes stock: the wheel).
The COACH reads data/vault/*.json (cards, decision trees, allocation rules,
edges) and nudges with the source attached; it also says where tastylive's
rules are conventions rather than edges.

Not investment advice: a training game on modeled prices.
"""

from __future__ import annotations

import json

from ..config import THETA_FILES

DISCLAIMER = ("THETA RUN is a training game. Prices are real daily closes (split- and "
              "dividend-adjusted, so older prices sit below the quotes you remember); option prices, IV, "
              "skew and earnings moves are MODELED (no real chains), fills are idealized and early "
              "assignment is not simulated. Results do not predict live trading. Decision-support, "
              "not investment advice.")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load(name: str, default=None):
    """Load a named THETA artefact (see config.THETA_FILES)."""
    return _read(THETA_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    _write(THETA_FILES[name], obj)
