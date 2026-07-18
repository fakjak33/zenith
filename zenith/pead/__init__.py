"""PEAD — post-earnings announcement drift screener for Zenith.

Implements the classic PEAD evidence: after a firm reports earnings, the stock
tends to keep drifting in the direction of the surprise for ~60 trading days
(Bernard & Thomas 1989/90). Two confirmations gate every signal — an EPS
surprise (SUE; Livnat & Mendenhall 2006 analyst-based form) AND a same-signed
market-excess announcement reaction (EAR; Brandt, Kishore, Santa-Clara &
Venkatachalam 2008, whose combination with SUE is the strongest documented
pairing). Three amplifiers complete a 0-100 composite: time-series revenue
surprise (Jegadeesh & Livnat 2006), abnormal announcement volume (Garfinkel &
Sokobin 2006) and 52-week-high anchoring (George, Hwang & Li 2015). Exits are
tracked at +5/+20/+60 trading days and the day before the NEXT report
(Bernard & Thomas 1990: 25-30% of the drift concentrates around it).

Free / best-effort data only (Nasdaq earnings calendar, yfinance, Vanguard
VONE holdings), computed offline in a GitHub Action that commits JSON under
data/pead/; the app is a thin reader. Decision-support, not investment advice.
"""

from __future__ import annotations

import json

from ..config import PEAD_FILES, PEAD_ARCHIVE_DIR

DISCLAIMER = ("PEAD signals are auto-generated from free/best-effort data (Nasdaq earnings "
              "calendar consensus snapshots, yfinance, Vanguard VONE holdings). Revenue "
              "surprise is a time-series proxy, not analyst consensus. Backfilled picks use "
              "current Russell 1000 membership (survivorship bias). Drift has attenuated in "
              "large caps since ~2006 (Martineau 2021); short-side drift is weaker and borrow "
              "costs are not modeled. Decision-support, not investment advice.")


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
    """Load a named PEAD artefact (see config.PEAD_FILES)."""
    return _read(PEAD_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    _write(PEAD_FILES[name], obj)


# --- per-reaction-day signal sheets: data/pead/archive/YYYY-MM-DD.json -------
def archive_day(day: str, obj: dict) -> None:
    _write(PEAD_ARCHIVE_DIR / f"{day}.json", obj)


def archive_days() -> list[str]:
    return sorted((p.stem for p in PEAD_ARCHIVE_DIR.glob("*.json")), reverse=True)


def load_day(day: str) -> dict:
    return _read(PEAD_ARCHIVE_DIR / f"{day}.json", {})
