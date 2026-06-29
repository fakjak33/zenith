"""Academic-factor backtest layer for the CAS Style-Trend model.

Replicates Gupta & Kelly (2019) "Factor Momentum Everywhere" on long-history,
publicly available academic factor returns (Ken French regional factors, plus
AQR QMJ/BAB if the user caches the xlsx). Validates the live ETF style-trend
model with decades of data the ETFs themselves don't have.

Honest scope: the public factor set (~5 factors x ~6 regions, + optional AQR) is
smaller than the paper's proprietary 65, so we report the autocorrelation / TS
factor-momentum *property* and how many of OUR factors exhibit it — a faithful
replication of the effect, not of the exact 65-factor portfolio.
"""
