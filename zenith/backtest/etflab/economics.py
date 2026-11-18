"""Fund economics simulator — what the ISSUER earns running the ETF.

Pure functions (no I/O), fully unit-testable. Models per-year AUM, fee revenue,
the recurring cost stack (index licensing, maintenance, rebalance trading) plus
one-time startup cost, and the resulting net profit to the issuer — and the
fee drag the expense ratio imposes on the investor.

All figures are ESTIMATES for planning; real costs vary widely by provider.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FundEconConfig:
    aum: float = 100_000_000.0          # starting assets under management ($)
    expense_ratio: float = 0.0040       # annual, e.g. 0.40%
    expected_aum_growth: float = 0.10   # annual AUM growth assumption
    startup_cost: float = 400_000.0     # one-time: legal, listing, seed, index dev
    annual_maintenance: float = 150_000.0   # admin, custody, audit, index calc (flat)
    index_licensing_bps: float = 3.0    # bps of AUM paid to the index provider
    rebalance_trading_bps: float = 5.0  # bps of AUM in trading cost per year
    num_holdings: int = 50


def economics(cfg: FundEconConfig, years: int = 5) -> pd.DataFrame:
    """Per-year issuer P&L over ``years``."""
    rows = []
    cum = 0.0
    for y in range(1, int(years) + 1):
        aum = cfg.aum * (1.0 + cfg.expected_aum_growth) ** (y - 1)
        revenue = aum * cfg.expense_ratio
        licensing = aum * cfg.index_licensing_bps / 10_000.0
        trading = aum * cfg.rebalance_trading_bps / 10_000.0
        startup = cfg.startup_cost if y == 1 else 0.0
        total_cost = licensing + cfg.annual_maintenance + trading + startup
        net = revenue - total_cost
        cum += net
        rows.append({
            "Year": y, "AUM": aum, "Revenue": revenue,
            "IndexLicensing": licensing, "Maintenance": cfg.annual_maintenance,
            "Trading": trading, "Startup": startup,
            "TotalCost": total_cost, "NetProfit": net, "CumNetProfit": cum,
        })
    return pd.DataFrame(rows).set_index("Year")


def breakeven_aum(cfg: FundEconConfig) -> float:
    """AUM at which fee revenue covers recurring costs (excludes one-time startup).

    revenue = aum·ER ; recurring = maintenance + aum·(licensing+trading)/1e4.
    """
    variable = (cfg.index_licensing_bps + cfg.rebalance_trading_bps) / 10_000.0
    denom = cfg.expense_ratio - variable
    if denom <= 0:
        return float("inf")
    return cfg.annual_maintenance / denom


def investor_fee_drag(gross_return_annual: float, expense_ratio: float,
                      years: int = 10, principal: float = 10_000.0) -> dict:
    """What the expense ratio costs the investor vs a fee-free version."""
    net_return = gross_return_annual - expense_ratio
    gross_fv = principal * (1.0 + gross_return_annual) ** years
    net_fv = principal * (1.0 + net_return) ** years
    return {
        "net_return_annual": net_return,
        "gross_fv": gross_fv,
        "net_fv": net_fv,
        "fee_cost": gross_fv - net_fv,
        "fee_cost_pct": (gross_fv - net_fv) / gross_fv * 100 if gross_fv else 0.0,
    }
