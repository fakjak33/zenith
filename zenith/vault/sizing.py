"""OPTIONS VAULT — sleeve sizing calculator (pure, no streamlit).

Only the verified guidance in data/vault/portfolio.json is used; there is no VIX-tiered
buying-power ladder because none could be confirmed (portfolio.json unverified[]):

  BP in use       25-50% of net liq in short premium, never above 75%
                  [ep:options-jive-2023-12-27@06:56]; about 30% with VIX above 15
                  [lit:tastylive-allocation-2022]
  per trade       about 1% risk (2% for higher tolerance) [page:concepts-risk-management];
                  under 5% of the account [page:concepts-number-of-occurrences]
  stress cap      undefined-risk index BPR of roughly 8-17% of net liq: a 2-3x-BPR crash loss
                  kept below 25-35% of net liq (illustrative arithmetic in portfolio.json)
                  [lit:santaclara-saretto-2009]
  theta           about 0.1% of net liq per day, a single-episode yardstick
                  [ep:from-theory-to-practice-2022-12-21@16:01]

The sleeve (the investor's 25-50% options allocation) caps the BP band: a 30% sleeve cannot run
50% of net liq in short premium.
"""

from __future__ import annotations

import math

ACCOUNTS = {"cash": "Cash account", "ira": "IRA", "reg_t": "Reg T margin", "pm": "Portfolio margin"}
BP_LO, BP_HI, BP_MAX = 0.25, 0.50, 0.75
TRADE_LO, TRADE_HI = 0.01, 0.05
STRESS_LO, STRESS_HI = 0.08, 0.17
THETA_NLV = 0.001
SOURCES = ["ep:options-jive-2023-12-27@06:56", "lit:tastylive-allocation-2022",
           "page:concepts-risk-management", "page:concepts-number-of-occurrences",
           "lit:santaclara-saretto-2009", "ep:from-theory-to-practice-2022-12-21@16:01",
           "lit:tastytrade-ira-2026", "lit:finra-4210-2026", "repro:vix_spikes"]


def calculate(nlv: float, sleeve_pct: float, account: str, vix: float | None) -> dict:
    """sleeve_pct in percent (e.g. 35). Returns dollar ranges plus warnings; never raises on
    out-of-range input -- it says what is wrong instead."""
    warnings: list[str] = []
    try:
        nlv = float(nlv)
        sleeve = float(sleeve_pct) / 100.0
    except (TypeError, ValueError):
        return {"ok": False, "warnings": ["Enter numbers for net liq and sleeve."]}
    if not math.isfinite(nlv) or nlv <= 0:
        return {"ok": False, "warnings": ["Net liquidation value must be above zero."]}
    sleeve = min(max(sleeve, 0.0), 1.0)
    if sleeve <= 0:
        return {"ok": False, "warnings": ["Sleeve is 0%: nothing to size."]}

    lo, hi = min(BP_LO, sleeve), min(BP_HI, sleeve)
    if sleeve < BP_LO:
        warnings.append(f"A {sleeve:.0%} sleeve is below tastylive's 25% floor; the band is capped "
                        f"at the sleeve.")
    if sleeve > BP_MAX:
        warnings.append("A sleeve above 75% of net liq is beyond tastylive's stated maximum.")

    lean = None
    if vix is not None and vix > 0:
        if vix < 15:
            lean = "lean to the lower half of the band (VIX below 15)"
        else:
            lean = "aim near 30% of net liq (tastylive's figure with VIX above 15), within the band"
        if vix >= 30:
            warnings.append("VIX at or above 30: the variance premium has been richer, but a >10% "
                            "drawdown within 21 days followed 11% of such days vs 4% of all days "
                            "[repro:vix_spikes]. Do not size up without a stress limit.")

    acct_ok = account in ACCOUNTS
    if not acct_ok:
        warnings.append(f"Unknown account type {account!r}.")
    if account == "cash":
        warnings.append("Cash account: no naked puts, short calls, strangles or straddles; spreads "
                        "unverified. Cash-secured puts reserve strike x 100, so BP equals notional "
                        "[lit:tastytrade-ira-2026].")
    elif account == "ira":
        warnings.append("IRA: no naked short puts; uncovered calls only at The Works tier with $25k; "
                        "defined-risk spreads allowed [lit:tastytrade-ira-2026].")
    elif account == "pm":
        if nlv < 100_000:
            warnings.append("Portfolio margin usually needs about $100k net liq "
                            "[lit:finra-4210-2026].")
        warnings.append("Portfolio margin stresses indexes only -8%/+6%: posted margin is not the "
                        "maximum loss [lit:finra-4210-2026].")
    if nlv < 25_000:
        warnings.append("Small account: one $1,000 BPR trade is already "
                        f"{1000 / nlv:.0%} of net liq; favour defined-risk spreads "
                        "[page:concepts-number-of-occurrences].")

    per_lo, per_hi = TRADE_LO * nlv, TRADE_HI * nlv
    undefined = account in ("reg_t", "pm")
    return {
        "ok": acct_ok,
        "nlv": nlv, "sleeve": sleeve, "sleeve_usd": sleeve * nlv,
        "bp_range": (lo * nlv, hi * nlv), "bp_pct": (lo, hi), "bp_max": min(BP_MAX, sleeve) * nlv,
        "lean": lean,
        "per_trade": (per_lo, per_hi),
        "positions": (max(1, math.ceil(lo * nlv / per_hi)), max(1, math.floor(hi * nlv / per_lo))),
        "stress_cap": (STRESS_LO * nlv, STRESS_HI * nlv) if undefined else None,
        "theta_day": THETA_NLV * nlv,
        "warnings": warnings,
        "sources": SOURCES,
    }
