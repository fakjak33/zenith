# Phase 04 — Free-data reproductions of headline claims
Read 00_master_brief.md and HANDOFF.md first. Needs Phase 02's research_catalog.json. If 03 is
done, use literature.json too.

Goal: re-test 5–8 headline claims on free data, so a card can say: "tastylive found X; on
[data, period] we find Y."

Propose the final list at GATE A, before coding. Draw from the candidates below plus frequently
cited, testable claims from research_catalog.json.

Candidates:
1. VRP on SPX: VIX vs subsequent 21-day realized vol since 1990, overall and by VIX
   "IV-rank" bucket. Reuse beta/hedge.py realized and vrp.
2. CBOE strategy benchmark indices vs the S&P 500: PUT, BXM, BXMD, CNDR, BFLY, WPUT, CLL,
   PPUT, or whichever histories CBOE still publishes (find the current CSV locations).
   Measure CAGR, vol, Sharpe, max drawdown and worst months, and the behaviour in
   2008/2018/2020/2022.
3. Earnings outlier-move continuation (the user's example):
   - outlier = reaction beyond a threshold vs the stock's own history of earnings moves;
     state the definition and test 2–3 variants;
   - measure the share that keep drifting and the mean drift at +5/+21 trading days, by
     direction and size bucket;
   - reuse zenith/pead/earnings.py and PEAD's tracking code.
4. Expected-move hit rate: how often SPX stays inside the VIX-implied ±1σ over 30 calendar
   days, and how asymmetric the breaches are.
5. Mechanical SPX strangle, model-priced:
   - a 16-delta strangle at 45 DTE, priced with Black-Scholes off VIX;
   - compare hold-to-expiry, a 50% target, a 21-DTE exit, and both;
   - report win rate, mean P&L, P&L/day, stdev, CVaR(5%) and max loss.
   Say plainly: no skew and no real quotes, so put credit and tail losses are understated;
   this is illustrative only. The point is to show how management reshapes the distribution.
6. VIX spikes: forward VIX and SPX after VIX > 30, or a +50% jump within 5 days.

Data: FRED via pandas-datareader; yfinance (^GSPC, ^VIX, ^VIX9D, ^VIX3M, SPY); CBOE public
CSVs; zenith's existing data/ artefacts. No paid data, no scipy.

Code:
- zenith/vault/repro/<name>.py — a pure compute function + a thin fetch layer.
- Output data/vault/repro/<name>.json (cited elsewhere as repro:<name>) with:
  - claim, tasty_value + sources, our_value, data, period, method, caveats;
  - verdict (agrees|mixed|contradicts);
  - chart_data: small and pre-aggregated, ready for Altair.
- Offline tests in tests/test_vault.py with synthetic series for every compute function.

GATE B — show:
- a table of claim × tasty × ours × verdict;
- each result's caveats;
- one chart per reproduction.

Then update HANDOFF.md.
