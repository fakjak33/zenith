"""The Cboe options-strategy benchmarks tracked, with their methodology documents.

Rules are paraphrased from each index's official Cboe methodology (linked);
the link is re-verified on every build (methodology.json).
"""

from __future__ import annotations

GOV = "https://cdn.cboe.com/resources/indices/governance/"

# ticker -> (family, name, rule summary, methodology file)
INDICES: dict[str, tuple[str, str, str, str]] = {
    "PUT": ("PUT-WRITE", "S&P 500 PutWrite",
            "Sells a one-month at-the-money SPX put each month (3rd Friday), fully collateralised by "
            "one- and three-month T-bills.", "Cboe_SP_500_PutWrite_Indices_Methodology.pdf"),
    "WPUT": ("PUT-WRITE", "S&P 500 One-Week PutWrite",
             "Sells an at-the-money SPX put every week, collateralised by one-month T-bills.",
             "Cboe_PutWrite_Indices_Methodology.pdf"),
    "PUTY": ("PUT-WRITE", "S&P 500 2% OTM PutWrite",
             "Sells a 2% out-of-the-money SPX put monthly, collateralised by one-month T-bills.",
             "Cboe_PutWrite_Indices_Methodology.pdf"),
    "PUTR": ("PUT-WRITE", "Russell 2000 PutWrite",
             "Sells an at-the-money RUT put monthly, collateralised by one-month T-bills.",
             "Cboe_PutWrite_Indices_Methodology.pdf"),
    "BXM": ("BUY-WRITE", "S&P 500 BuyWrite",
            "Long the S&P 500, sells a one-month at-the-money SPX call each month.",
            "Cboe_BuyWrite_Indices_Methodology.pdf"),
    "BXMD": ("BUY-WRITE", "S&P 500 30-Delta BuyWrite",
             "Long the S&P 500, sells the monthly SPX call with delta closest to 0.30.",
             "Cboe_BuyWrite_Indices_Methodology.pdf"),
    "BXY": ("BUY-WRITE", "S&P 500 2% OTM BuyWrite",
            "Long the S&P 500, sells a one-month 2% out-of-the-money SPX call.",
            "Cboe_BuyWrite_Indices_Methodology.pdf"),
    "BXMW": ("BUY-WRITE", "S&P 500 Multi-Week BuyWrite",
             "Four staggered short at-the-money SPX/SPXW calls, one written each week ~4 weeks out.",
             "BXMW_Methodology.pdf"),
    "BXR": ("BUY-WRITE", "Russell 2000 BuyWrite",
            "Long the Russell 2000, sells a one-month at-the-money RUT call.",
            "Cboe_BuyWrite_Indices_Methodology.pdf"),
    "CNDR": ("CONDOR / FLY", "S&P 500 Iron Condor",
             "Monthly SPX (AM-settled): sells ~0.20Δ call and −0.20Δ put, buys ~0.05Δ wings; strikes set "
             "before 11:00 ET on the 3rd-Friday roll; T-bill account of 10× the max loss (≈10% max loss).",
             "CNDR_Methodology.pdf"),
    "BFLY": ("CONDOR / FLY", "S&P 500 Iron Butterfly",
             "Monthly SPX: sells the at-the-money put and call, buys 5% out-of-the-money wings; T-bill "
             "account sized to limit the downside.", "BFLY_Methodology.pdf"),
    "CLL": ("COLLAR", "S&P 500 95-110 Collar",
            "Long the S&P 500, buys quarterly 5% OTM puts, sells monthly 10% OTM calls.",
            "Cboe_Collar_Indices_Methodology.pdf"),
    "CLLZ": ("COLLAR", "S&P 500 Zero-Cost Put Spread Collar",
             "Long the S&P 500, buys a monthly 2.5% OTM put, sells a 5% OTM put, and sells an OTM call "
             "struck to pay for the put spread.", "Cboe_Zero-Cost_Put_Spread_Collar_Indices_Methodology.pdf"),
    "PPUT": ("PROTECTION", "S&P 500 5% Put Protection",
             "Long the S&P 500 plus a monthly 5% out-of-the-money SPX put.",
             "Cboe_SP_500_Put_Protection_Indices_Methodology.pdf"),
    "RXM": ("RISK REVERSAL", "S&P 500 Risk Reversal",
            "Buys a monthly 0.25Δ SPX call and sells a −0.25Δ put, collateralised by T-bills.",
            "RXM_Methodology.pdf"),
    "CMBO": ("COMBO", "S&P 500 Covered Combo",
             "Long the S&P 500, sells a 2% OTM call and an at-the-money put monthly (a covered strangle).",
             "CMBO_Methodology.pdf"),
    "SVRPO": ("VOL PREMIUM", "S&P 500 Market-Neutral VRP Optimized",
              "Writes one-month SPX calls and puts and buys cheaper wings to harvest the volatility risk "
              "premium with low equity beta; T-bills cover the maximum loss.", "SVRPO_Methodology.pdf"),
    "VPD": ("VOL PREMIUM", "VIX Premium Strategy",
            "Short the next monthly VIX future over a T-bill account, sized so a 25-point VIX jump keeps "
            "75% of the portfolio.", "VPD_Methodology.pdf"),
}
FAMILIES = ("PUT-WRITE", "BUY-WRITE", "CONDOR / FLY", "COLLAR", "PROTECTION", "RISK REVERSAL", "COMBO",
            "VOL PREMIUM")
VOL_CONTEXT = ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "SKEW", "COR1M", "COR3M", "DSPX")
VOL_DOCS = {"VIX9D": "Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf",
            "VVIX": "Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf"}
DASHBOARD = "https://www.cboe.com/us/indices/dashboard/{t}/"


def methodology_url(t: str) -> str:
    return GOV + INDICES[t][3]
