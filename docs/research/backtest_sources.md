# BACKTEST research sources

These are the sources behind the BACKTEST upgrade: two YouTube channels, plus the papers each idea rests on.

- **What this file holds:** short summaries written in our own words. It holds no transcripts.
- **How it was gathered:** titles, descriptions and auto-captions were pulled with `yt-dlp`, which is free and open source.
- **Videos without a transcript:** these are marked *(desc)*. They are summarised from the video description only.

Each feature in the app cites the video it came from in its GUIDE or info badge. The video is listed next to the
paper(s) that support or challenge the idea.

## SharpeEdge (preferred: concise, validation-first)

| Video | Method in one line | Where it lives in Zenith |
|---|---|---|
| How to Properly Backtest Algorithmic Trading Strategies | Signals act on the next bar; costs: fee, swap, spread, slippage; ATR stop; size so one stop costs X% | Engine (already next-bar); cost presets (Phase 3) |
| How to Optimize and Validate Systematic Trading Strategies | Coarse grid, distribution of all combinations, k-means on the top-N, ±k sensitivity, walk-forward, bootstrap + permutation on OOS | SENSITIVITY / ROBUSTNESS; grid views & clustering (Phase 3) |
| How to Use Logistic Regression for Trading *(desc)* | Linear regression of returns fails; logistic regression with a well-defined target, naive baseline, walk-forward, replication across instruments | ML LAB (Phase 4) |
| How to Evaluate Prop Firm Challenges Using Monte Carlo *(desc)* | Block bootstrap of real returns; target / max-loss / daily-limit; pass and ruin probability | Monte Carlo target/stop simulator (Phase 3) |
| When to Stop a Trading Strategy — Alpha Decay *(desc)* | Bootstrap drawdown breaker, MC equity bands, monthly percentile rank, CUSUM, rolling alpha with robust errors | DECAY (Phase 5) |
| Does ML Actually Improve a Trading Strategy? *(desc)* | Meta-labeling with triple-barrier labels; L1 linear SVM; walk-forward with an inner grid search; Wilson CIs; two-proportion z-test | ML LAB meta-labeling (Phase 4) |
| Is the Kalman Filter Just a Fancy Moving Average? *(desc)* | Trend filters seen as kernels at equal lag; Kalman local trend; use filtered, not smoothed, values | Indicators: `kalman`, `reg_slope`, MA family; LEARN kernels lesson |
| How to Manage a Portfolio of Trading Strategies *(desc)* | 1/N, inverse vol, ERC, min-var, max-Sharpe and HRP compared walk-forward against 1/N, with a bootstrap of the Sharpe difference | PORTFOLIO optimisers (Phase 5) |
| Market Regime Detection for Algo Trading *(desc)* | Vol × directionality grid; ER / ADX / Hurst / variance ratio; HMM filtered not smoothed; hysteresis; 1000-random-filter benchmark | `regime/` detectors, BUILD › REGIMES, regime gating (Phase 2) |
| How to Code a Trading Bot in MetaTrader 5 *(desc)* | Triple-EMA plus ATR stop as an MT5 Expert Advisor | Not ported (platform-specific) |

## MemLabs (dense, ML and statistics first)

| Video | Method in one line | Where it lives in Zenith |
|---|---|---|
| Mean Reversion strategy | Group by sign(lag-1 return) → mean, sum and count of the next return; 75/25 time split; −sign(lag1) | EXPLORE › last bar's direction; indicator `lag1` |
| Trend Following strategy | Weekly bars, sign(lag-1); fees charged on entry and exit notional; maker vs taker fees | EXPLORE; `tsmom`; cost presets (Phase 3) |
| Quant Trading for Dummies | Day-of-week seasonality; 100+ samples per bucket; 70/30 split | EXPLORE › day of week (with a snooping warning) |
| Feature Engineering | Model returns, not prices; AR lags, cross lags, ARDL, spread lag | Cross-asset indicators; ML LAB features (Phase 4) |
| Cross Returns *(desc)* | Another asset's lag predicted ETH better than ETH's own history | `ref_lag`, `rel_strength`, `spread_z` |
| Decomposed Returns *(desc)* | E[r] = p·up − (1−p)·down; independence permutation test | ML LAB decomposition (Phase 4) |
| Model Stacking *(desc)* | Combine models and test against the best single model | ML LAB stacking (Phase 4) |
| How to trade with XGBoost *(desc)* | Gradient-boosted trees as a taker and as a maker | ML LAB HistGradientBoosting (Phase 4) |
| How to handle Regime Changes | Moments shift over time; a rolling-mean memory feature; online SGD where weights flip; bandits | ML LAB online learner (Phase 4); regimes |
| Accelerator 4: Time Series | Mean vs median, stationarity, autocorrelation; AR(1) sign = mean reversion or momentum | EXPLORE › serial dependence; LEARN |
| Accelerator 7: Cross Validation | Train-fraction sweep, rolling and expanding windows; score on trade EV, not MSE | ML LAB CV; CPCV (Phase 3) |
| Let's Build a Strategy 2 | Constant vs compounding size, leverage, liquidation price | LEARN leverage lesson (Phase 6) |
| Introduction to Quantitative Trading / RenTech | EV beats win rate; frequency scaling; thresholds on probability | LEARN; ML probability → position mapping |
| Accelerator 1–3, 5, 6, 8, 9, What is Quant Trading, Algothon, Podcast #1 | Python basics, vectorisation, classification, bots, careers | LEARN background only |

## Key academic references used in hovers and guides

- **Backtest overfitting**
  - Bailey & López de Prado (2014) deflated Sharpe
  - Bailey, Borwein, López de Prado & Zhu (2017) PBO
  - Harvey, Liu & Zhu (2016)
  - White (2000)
  - Hansen (2005)
  - Arnott, Harvey & Markowitz (2019)
- **Cross-validation:** López de Prado, *Advances in Financial Machine Learning* (2018): purging, embargo, CPCV, meta-labeling.
- **Trend**
  - Moskowitz, Ooi & Pedersen (2012)
  - Hurst, Ooi & Pedersen (2017)
  - Levine & Pedersen (2016)
  - Faber (2007)
  - Brock, Lakonishok & LeBaron (1992)
- **Reversal and serial dependence**
  - Jegadeesh (1990)
  - Lehmann (1990)
  - Lo & MacKinlay (1988, 1990)
  - Nagel (2012)
- **Regimes**
  - Hamilton (1989)
  - Ang & Bekaert (2002)
  - Kritzman, Page & Turkington (2012)
  - Moreira & Muir (2017)
  - Daniel & Moskowitz (2016)
- **Data snooping:** Sullivan, Timmermann & White (1999, 2001).
- **Prediction out of sample**
  - Welch & Goyal (2008)
  - Campbell & Thompson (2008)
  - Rapach, Strauss & Zhou (2010)
  - Gu, Kelly & Xiu (2020)
- **Portfolios**
  - DeMiguel, Garlappi & Uppal (2009)
  - Maillard, Roncalli & Teiletche (2010)
  - López de Prado (2016) HRP
- **Decay:** McLean & Pontiff (2016); Page (1954) CUSUM.
