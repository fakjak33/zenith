"""Synthetic guess history for testing the dashboard and the Read.

    python -m zenith.ephemeris.seed --player demo-seed --n 200

Planted structure (so the Read has something true to find): real skill in
Crypto on 4H (~65% right), a weakness in Bonds on Weekly (~38%), coin-flip
elsewhere, a long bias, calibrated-ish conviction, ~25% of calls with
stops, and one account reset. Rows are tagged mode "practice"/"daily".
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from . import stake_for
from .benchmarks import base_rate
from .repo import Repository, get_repo

SETS = [[{"id": "sma", "params": {"lengths": "20,50,200"}}, {"id": "volume", "params": {"n": 20}}],
        [{"id": "ema", "params": {"lengths": "21,50"}}, {"id": "rsi", "params": {"n": 14}}],
        []]
CELLS = [("Crypto", "4H", 5), ("Bonds", "Weekly", 10), ("Russell 1000", "Daily", 10),
         ("US Equity ETFs", "Daily", 20), ("Precious Metals", "Daily", 10), ("Currencies", "4H", 5)]
SKILL = {("Crypto", "4H"): 0.65, ("Bonds", "Weekly"): 0.38}


def seed(repo: Repository, player: str, n: int = 200, rng_seed: int = 7) -> int:
    rng = random.Random(rng_seed)
    nprng = np.random.default_rng(rng_seed)
    p = repo.get_or_create_player(player)
    t0 = datetime.now(timezone.utc) - timedelta(days=60)
    for i in range(n):
        cls, tf, h = rng.choices(CELLS, weights=[3, 3, 3, 2, 1, 1])[0]
        br = (base_rate(cls, tf, h) or {}).get("p", 0.55)
        up = rng.random() < br
        vol = {"4H": 0.02, "Daily": 0.04, "Weekly": 0.06}[tf]
        market = abs(nprng.normal(0.0, vol)) * (1 if up else -1)
        right = rng.random() < SKILL.get((cls, tf), 0.5 + (0.04 if up else 0))
        direction = (1 if up else -1) if right else (-1 if up else 1)
        if rng.random() < 0.15:                       # long bias
            direction = 1
        conv = rng.choices(["Low", "Medium", "High"], weights=[2, 3, 1 + 2 * right])[0]
        stake = stake_for(conv)
        atr_pct = vol / 2
        mae_full = -abs(nprng.normal(0, vol * 0.6))
        mfe_full = abs(nprng.normal(0, vol * 0.8)) + max(0.0, direction * market)
        exit_reason, trade_ret, sl = "horizon", direction * market, None
        if rng.random() < 0.25:                       # with a 1.5 ATR stop
            sl = 1.5 * atr_pct
            if -mae_full >= sl:
                exit_reason, trade_ret = "stop", -sl
        rule = rng.choice([1, -1, 0, 1])
        repo.log_guess({
            "player_id": p["id"], "mode": "daily" if i % 9 == 0 else "practice",
            "ts": (t0 + timedelta(hours=7 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ticker": f"SYN{i % 40:02d}", "name": "synthetic", "asset_class": cls, "sector": "",
            "timeframe": tf, "horizon": h, "lookback": 120, "window_start": "2015-01-01",
            "decision_date": (datetime(2015, 1, 1) + timedelta(days=11 * i)).isoformat(),
            "chart_key": f"SYN|{i}", "indicators_json": SETS[i % 3],
            "blinding_json": {"rebase": True}, "direction": direction, "conviction": conv,
            "stake": stake, "entry": 100.0, "sl": None if sl is None else 100 * (1 - direction * sl),
            "outcome": "UP" if market > 0 else "DOWN", "market_ret": market, "trade_ret": trade_ret,
            "pnl": trade_ret * stake, "win": trade_ret > 0,
            "r_mult": (trade_ret / sl) if sl else None, "atr_ret": trade_ret / atr_pct,
            "mfe": mfe_full, "mae": mae_full, "mfe_full": mfe_full, "mae_full": mae_full,
            "candles_held": h, "exit_reason": exit_reason, "ambiguous": False, "gap_fill": False,
            "base_rate": br, "rule_call": rule, "rule_ret": rule * market,
            "regime_json": {"trend": rng.choice(["above SMA200 ↑", "below SMA200 ↓"]),
                            "vol": rng.choice(["low vol", "mid vol", "high vol"]),
                            "dist_high": rng.choice(["at 52w high", "-2% to -10%", "-10% to -25%"]),
                            "rsi": rng.choice(["RSI 30-50", "RSI 50-70"])},
            "decision_ms": rng.randint(800, 20000),
        })
        if i == n // 2:
            repo._exec("INSERT INTO account_resets (player_id, mode, ts) VALUES (?, ?, ?)",
                       (p["id"], "practice", (t0 + timedelta(hours=7 * i, minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    return p["id"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="demo-seed")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--db-url", default=None, help="Postgres URL; default local SQLite")
    a = ap.parse_args()
    print(json.dumps({"player_id": seed(get_repo(a.db_url), a.player, a.n)}))
