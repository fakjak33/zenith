# CLEAN BETA — design decisions log

Every structural or methodological choice made building this tab, with the reason. Thresholds themselves
live in `zenith/config.py` (CLEAN BETA block); this file says *why*.

## Confirmed with Jobe (2026-10-02)

| # | Decision | Reason |
|---|---|---|
| 1 | Universe: Russell 1000 vs SPY now; ~100–150 curated ADRs vs ACWI as a later toggle | Reuses the R1000 loader every other tab uses; free foreign data is patchy |
| 2 | Score = clean-beta metrics + a yfinance fundamental quality/mispricing proxy | LSY 2018: the anomaly runs through *overpriced* high-beta names, so a mispricing layer matters |
| 3 | Put ladder is advisory only (no manual position entry) | Monitor, not an execution engine |
| 4 | [P]/[S]/[E] data-confidence tags scoped to this tab; dated PARALLAX export | Neither existed in ZENITH; scoped so other tabs are untouched |
| 5 | Data: yfinance only, no new dependencies, no paid API | Matches every other tab; paid upgrade path stays open |

## Implementation decisions (2026-10-03)

| # | Decision | Reason |
|---|---|---|
| 6 | One daily job; `auto` re-screens when the stored screen is from an earlier month and rebalances when the basket predates the current quarter-start month | Self-healing: a missed first-of-month run is caught up the next day, never skipped. Replaces the plan's `--action monthly` with `screen` / `rebalance` / `hedge` |
| 7 | bswa uses raw daily returns, not excess returns | Daily rf ≈ 0.016% moves the slope by far less than rounding; avoids a FRED dependency |
| 8 | bswa on ~2y of data; OLS/ρ/R²/IVOL on the last 252 days | Welch's decay makes data older than ~1y nearly weightless; ρ/IVOL per the spec's 252d |
| 9 | Welch parity is tested against an independent loop-based port of his procedure (5 synthetic tickers, 1e-9) | His published R code/files were not fetched in this session; spot-check vs his beta files remains a follow-up |
| 10 | Fundamentals read-only from IDEAS' committed `data/ideas/fundamentals.json`; no refresh here | That file belongs to ideas.yml; a second writer would collide in git. Coverage is ~100% of the R1000 |
| 11 | Accruals proxy = (profit margin × revenue − OCF) / revenue, tagged [E] | `.info` has no total assets; revenue scaling keeps the sign and ranking intent |
| 12 | Net issuance starts from this tab's own monthly `sharesOutstanding` snapshots (`data/beta/shares.json`) and enters the score after 12 months | No free point-in-time share history; never back-filled from today's number |
| 13 | Missing quality (< 3 components) → neutral 50, flagged [E], never a fail | Don't exclude names for a vendor gap |
| 14 | Event score = 100 − 25 per >4σ residual day; 0 for IPO / manual flag; earnings-soon is a filter, not a score input | Earnings timing is transient; jump history is a property of the name |
| 15 | Optionable check only for screen finalists, 90-day TTL cache; lookup failure = assumed optionable | ~1,000 `.options` calls nightly is too slow; R1000 names >$2B are almost all optionable |
| 16 | Basket: greedy, no optimizer — incumbents first (buffer), then size-bucket representation, then score minus correlation penalty | No new dependency; every seat has a stated reason (`why`) |
| 17 | Weights: equal weight water-filled under BOTH the 4% name cap and the 15% sector cap; any remainder is reported uninvested | A first version capped sector *name count* and silently breached the 15% *weight* cap on a short basket (caught by `scripts/screen.py`). Tightening the name count instead spiralled the basket from 19 to 12 names |
| 18 | A basket that falls below 30 names is NOT padded with failing names; it is flagged `short` | Padding would defeat the filters. Spec thresholds kept as defaults (see open question below) |
| 19 | Two diversification numbers: effective bets = (Σλ)²/Σλ² of the members' correlation eigenvalues (headline), and Meucci's variance ENB | Meucci's ENB is ~1.0 for any long-only high-beta basket (the market factor is nearly all of its variance) — true but uninformative as a headline |
| 20 | Trend gate has three states: ON (12-1 and 10m-MA both positive), PARTIAL (split), REDUCED (both negative) → exposure 100/75/50% | The plan's binary gate left the split case undefined |
| 21 | Vol forecast = mean of 1m and 3m realized; VRP realized forecast = equal blend of 5/21/63-day realized (HAR horizons, no fitted coefficients) | Transparent, no estimation risk; tagged [E] |
| 22 | Put budget slides linearly from 1.5%/yr (VRP at 0th pct) to 0.5%/yr (100th pct); coverage = budget ÷ full-coverage ladder carry, capped at 100% | Israelov & Nielsen: size on VRP, not the VIX level |
| 23 | DBMF/KMLM trend is computed from their own prices here, not read from TREND artefacts | Keeps the hedge job independent of trend.yml's run order |
| 24 | Hedge history is one compact `hedge_history.json`, not yearly shards | ~250 small rows a year |
| 25 | The basket's "β" includes the cash remainder; "invested names β" is shown alongside | The put ladder sizes on the dollar beta of the sleeve, which includes cash |
| 26 | **ρ floor lowered from the spec's 0.55 to 0.50** (Jobe, 2026-10-03) | See below |

## Resolved: the correlation floor (2026-10-03)

On the first live run (as of 2026-10-02) only **21 of 977** liquid names passed every filter at the spec's
**ρ ≥ 0.55**. Among the top-quintile-beta names, the median 252-day correlation to SPY was 0.49. Pass counts
by threshold:

| ρ floor | IVOL cut at top tercile | IVOL cut at top quintile |
|---|---|---|
| 0.45 | 47 | 62 |
| 0.50 | 35 | 44 |
| 0.55 (spec) | 21 | 25 |

Jobe chose **0.50** (IVOL cut unchanged). Result: 35 pass. The basket seats 23 of them, 65% invested,
β 1.06 including cash (1.64 for the invested names), effective bets 4.4. It is still below the 30-name floor
because the passing names cluster in 5 sectors and the 15% sector cap seats at most 6 names per sector.
A percentile-based floor (stable basket size across correlation regimes) remains an option if this recurs.
