"""Run statistics, XP and achievements."""

from __future__ import annotations

import math

import numpy as np

ACHIEVEMENTS = {
    "first_blood": ("FIRST BLOOD", "Open your first position"),
    "wheel": ("WHEEL TURNER", "Get assigned and sell a call against the stock"),
    "beat_bot": ("BEAT THE BOT", "Finish ahead of the mechanical tastytrade bot"),
    "beat_spy": ("BEAT THE TAPE", "Finish ahead of buy-and-hold SPY"),
    "survivor": ("SURVIVOR", "Finish a crisis challenge with drawdown under 15%"),
    "discipline": ("DISCIPLINE", "Follow at least 80% of the coach's management calls"),
    "small": ("TRADE SMALL", "Never run buying power above 50% of net liq"),
    "occurrences": ("NUMBER OF OCCURRENCES", "Close 20 or more trades in one run"),
    "theta": ("THETA MACHINE", "Average theta of 0.1% of net liq per day or more"),
    "green": ("IN THE GREEN", "Finish with a positive return"),
}


def curve_stats(equity: list[list[float]]) -> dict:
    """Return, max drawdown and an annualized Sharpe-like ratio from [[i, nlv, ...]]."""
    if len(equity) < 2:
        return {"ret": 0.0, "max_dd": 0.0, "sharpe": 0.0, "days": len(equity)}
    v = np.array([e[1] for e in equity], dtype=float)
    r = np.diff(v) / v[:-1]
    peak = np.maximum.accumulate(v)
    dd = float(np.max((peak - v) / peak))
    sd = float(np.std(r))
    sharpe = float(np.mean(r) / sd * math.sqrt(252)) if sd > 0 else 0.0
    return {"ret": float(v[-1] / v[0] - 1.0), "max_dd": dd, "sharpe": sharpe, "days": len(v)}


def buy_hold(closes: np.ndarray) -> list[float]:
    c = np.asarray(closes, dtype=float)
    return list(c / c[0])


def score(player: dict, bot: dict, spy_ret: float, adherence: float | None,
          difficulty: str, crisis: bool, flags: dict) -> dict:
    """XP and achievements for a finished run. player/bot are curve_stats dicts."""
    mult = {"easy": 1.0, "normal": 1.5, "hard": 2.0}.get(difficulty, 1.0)
    xp = 100.0
    xp += max(0.0, player["ret"]) * 1000.0
    xp += max(0.0, player["ret"] - bot["ret"]) * 2000.0
    xp += max(0.0, player["ret"] - spy_ret) * 1000.0
    xp -= player["max_dd"] * 300.0
    if adherence is not None:
        xp += adherence * 100.0
    xp = max(10.0, xp) * mult
    got = []
    if flags.get("trades", 0) >= 1:
        got.append("first_blood")
    if flags.get("wheel"):
        got.append("wheel")
    if player["ret"] > bot["ret"]:
        got.append("beat_bot")
    if player["ret"] > spy_ret:
        got.append("beat_spy")
    if crisis and player["max_dd"] < 0.15:
        got.append("survivor")
    if adherence is not None and adherence >= 0.8:
        got.append("discipline")
    if flags.get("max_bp", 1.0) <= 0.50 and flags.get("trades", 0) >= 1:
        got.append("small")
    if flags.get("closed", 0) >= 20:
        got.append("occurrences")
    if flags.get("theta_nlv", 0.0) >= 0.001:
        got.append("theta")
    if player["ret"] > 0:
        got.append("green")
    return {"xp": int(round(xp)), "achievements": got, "mult": mult}


def level(total_xp: int) -> tuple[int, int, int]:
    """(level, xp into level, xp needed for next) on a gently rising curve."""
    lvl, need, rem = 1, 500, int(total_xp)
    while rem >= need:
        rem -= need
        lvl += 1
        need = int(need * 1.25)
    return lvl, rem, need
