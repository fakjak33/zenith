"""Scenarios: seeded random windows, blind (hidden-ticker) runs and challenges.

A scenario is a small dict, so it saves and replays exactly:
  {"seed", "start_i", "end_i", "difficulty", "hide", "challenge", "cash", "aliases"}
Difficulty sets the rules of the table:
  easy    tickers and dates visible, coach on, fills at mid
  normal  tickers and dates hidden, coach on, liquidity-tier slippage
  hard    hidden, slippage, coach silent until the debrief
"""

from __future__ import annotations

import random

from ..market import Market

DIFFICULTY = {
    "easy": {"hide": False, "slip": 0.0, "coach": True},
    "normal": {"hide": True, "slip": 1.0, "coach": True},
    "hard": {"hide": True, "slip": 1.0, "coach": False},
}
LENGTHS = {"Quarter (63d)": 63, "Half year (126d)": 126, "Year (252d)": 252}
START_CASH = 100_000.0
WARMUP = 260                       # trading days of history needed before the start (IV rank)


def aliases(tickers: list[str], seed: int) -> dict[str, str]:
    """Stable blind names: ASSET-01.. in a shuffled order (ETFs keep a hint)."""
    rng = random.Random(seed)
    order = list(tickers)
    rng.shuffle(order)
    return {t: f"ASSET-{k + 1:03d}" for k, t in enumerate(order)}


def random_scenario(m: Market, seed: int, length: int, difficulty: str,
                    cash: float = START_CASH) -> dict:
    rng = random.Random(seed)
    lo = WARMUP
    hi = len(m) - length - 1
    start = rng.randint(lo, max(lo, hi))
    d = DIFFICULTY[difficulty]
    return {"seed": seed, "start_i": start, "end_i": min(start + length, len(m) - 1),
            "difficulty": difficulty, "hide": d["hide"], "challenge": None, "cash": cash,
            "aliases": aliases(m.tickers, seed) if d["hide"] else {}}


def challenge_scenario(m: Market, ch: dict, difficulty: str, cash: float = START_CASH) -> dict:
    """Scripted window from data/theta/challenges.json; dates stay visible (the
    point is to live through a known episode) unless the challenge says blind."""
    start, end = m.idx(ch["start"]), min(m.idx(ch["end"]), len(m) - 1)
    d = DIFFICULTY[difficulty]
    hide = bool(ch.get("blind")) and d["hide"]
    return {"seed": int(ch.get("seed", 0)), "start_i": start, "end_i": end,
            "difficulty": difficulty, "hide": hide, "challenge": ch["id"], "cash": cash,
            "aliases": aliases(m.tickers, int(ch.get("seed", 0))) if hide else {}}


def daily_seed(date_iso: str) -> int:
    """Everyone gets the same daily run on a given calendar day."""
    return int(date_iso.replace("-", "")) * 7919 % 2_147_483_647


def label(sc: dict, ticker: str) -> str:
    return sc.get("aliases", {}).get(ticker, ticker) if sc.get("hide") else ticker
