"""VOL SCREENER + VOL SURFACE — a daily premium-selling (and buying) screener on
REAL option chains.

THE FLOW it serves, in the trader's own order:
  1. a basket of LIQUID underlyings (option volume, open interest, tight
     markets; consistent CORE names apart from event-driven SURGE names);
  2. baskets of UNCORRELATED underlyings (clusters of the return-correlation
     matrix, one representative each);
  3. MONITOR the volatility term structure, VRP and price behaviour, scored
     by a per-underlying checklist;
  4. (context only) deploy small, volatility-sized, diversified positions.
     Zenith finds candidates; it does not manage trades.

DATA. Once a day after the close, every chain comes from Cboe's public
delayed-quote JSON (free, no key; yfinance is the fallback). Implied vols are
recomputed from mid quotes against the put-call-parity forward (so dividends
and borrow drop out), interpolated to constant maturities in TOTAL VARIANCE
(the VIX convention), and the earnings event is stripped out of the term
structure from the kink it leaves (ex-earnings IV + implied move).

REALIZED VOL is measured six ways (close-to-close, Parkinson 1980,
Garman-Klass 1980, Rogers-Satchell 1991, Yang-Zhang 2000, EWMA) and FORECAST
two ways (HAR-RV, Corsi 2009; GARCH(1,1) / GJR-GARCH, Bollerslev 1986 /
Glosten-Jagannathan-Runkle 1993 — fitted here in pure numpy). VRP per tenor
is ex-earnings IV minus the GARCH forecast of realized vol over the same
horizon (Carr & Wu 2009; Bollerslev, Tauchen & Zhou 2009).

HISTORY. Free historical implied vol does not exist, so the per-ticker IV
history starts the night this ships. SPX/SPY/QQQ/IWM/DIA/EEM/GLD/USO are
seeded on day one from Cboe's own volatility indices (VIX9D..VIX1Y, VXN, RVX,
VXD, VXEEM, GVZ, OVX); other names fall back to THETA RUN's modeled IV and
carry a MODELED badge until 252 observed days accrue. IV rank / percentile
say "provisional" until then.

Not investment advice.
"""

from __future__ import annotations

import json
import math

from ..config import VOLSCREEN_FILES

DISCLAIMER = ("VOL SCREENER reads real end-of-day option quotes (Cboe delayed data, free) and "
              "publishes derived metrics only. IV rank/percentile need a year of history: names "
              "marked PROVISIONAL or MODELED are not yet on their own observed IV history. The VRP "
              "state machine is a practitioner heuristic, not a published result, and the star "
              "rating uses our thresholds, not tastytrade's formula. Screens find candidates; they "
              "are decision-support, not investment advice.")


def _scrub(obj):
    """NaN / inf -> None, recursively (strict JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def load(name: str, default=None):
    path = VOLSCREEN_FILES[name]
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save(name: str, obj) -> None:
    path = VOLSCREEN_FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(_scrub(obj), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)
