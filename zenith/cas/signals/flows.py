"""Flows & positioning segment.

Free data caps this segment to *proxies* — true fund/ETF/options/dealer flows are
paywalled (EPFR, SpotGamma, …). What we can build for free:
  * COT positioning extremes & momentum (dealer / asset-mgr / leveraged / managed
    money) — the core proxy for dealer/CTA/hedge-fund/managed-futures positioning.
  * ETF volume-trend flow proxy (volume vs its average = participation).
Each signal carries source + confidence so the UI can be honest about it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from ..schema import make_signal
from ..universe import COT_MAP, asset_class_of


def _z_last(values: list[float], lookback: int = 104) -> tuple[float, float]:
    s = pd.Series(values[::-1]).tail(lookback)        # records are newest-first
    if len(s) < 8:
        return 0.0, 0.5
    z = (s.iloc[-1] - s.mean()) / (s.std() + 1e-9)
    pct = float((s <= s.iloc[-1]).mean())
    return float(z), pct


def from_cot(cot: dict[str, list]) -> list[dict]:
    """Positioning signals from COT. Leveraged/managed-money net at a positive
    extreme = crowded long = contrarian sell tilt; asset-manager trend = follow."""
    out: list[dict] = []
    inv = {v: k for k, v in COT_MAP.items()}          # market name -> etf ticker
    for market, recs in cot.items():
        if not recs:
            continue
        etf = inv.get(market, market)
        ac = asset_class_of(etf)
        for field, family, contrarian in (
            ("lev_money_net", "cot_leveraged_positioning", True),
            ("mgd_money_net", "cot_managed_money_positioning", True),
            ("asset_mgr_net", "cot_asset_manager_positioning", False),
            ("dealer_net", "cot_dealer_positioning", False),
        ):
            vals = [r.get(field, 0.0) for r in recs]
            z, pct = _z_last(vals)
            sig = ind.clip1((-z if contrarian else z) / 2.0)
            out.append(make_signal(
                etf, "flows", family, sig, asset_class=ac, horizon="weeks",
                source="cftc", percentile=pct,
                rationale=f"{market}: {field} z={z:+.2f}, pct={pct:.0%} "
                          f"({'contrarian' if contrarian else 'follow'})"))
    return out


def from_volume(data: dict[str, pd.DataFrame]) -> list[dict]:
    """ETF participation proxy: 20d volume vs 100d average, signed by price trend."""
    out: list[dict] = []
    for t, df in data.items():
        vol, close = df["volume"], df["close"]
        v20, v100 = vol.rolling(20).mean(), vol.rolling(100).mean()
        if pd.isna(v20.iloc[-1]) or pd.isna(v100.iloc[-1]) or v100.iloc[-1] == 0:
            continue
        participation = v20.iloc[-1] / v100.iloc[-1] - 1.0
        trend = np.sign(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 21 else 0
        sig = ind.clip1(participation * trend)
        out.append(make_signal(
            t, "flows", "etf_volume_flow_proxy", sig, asset_class=asset_class_of(t),
            horizon="weeks", source="yfinance", confidence="low",
            rationale=f"20d/100d volume {participation:+.0%}, trend {trend:+.0f} "
                      f"(proxy — true ETF flows are paywalled)"))
    return out


# Dynamics with no free feed — surfaced as explicit manual/proxy placeholders so
# the user sees them in the UI and can wire a paid feed or hand-enter values.
MANUAL_DYNAMICS = [
    "dealer_gamma_hedging", "options_positioning", "options_flow",
    "leveraged_etf_flows", "pension_fund_flows", "retail_flows",
    "factor_flows", "market_neutral_positioning", "long_short_positioning",
]


def compute(cot: dict[str, list], data: dict[str, pd.DataFrame]) -> list[dict]:
    return from_cot(cot) + from_volume(data)
