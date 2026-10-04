# EPHEMERIS — tape-reading trainer

EPHEMERIS is a blind-chart prediction game inside ZENITH, built for deliberate practice in reading price action.

1. You see a random historical chart. The ticker, dates and price level are hidden.
2. You call UP or DOWN for the next N candles.
3. The future candles replay and your call is scored.
4. Every call feeds a stats engine. That engine answers the only question that matters: **are you beating the base rate and a simple trend rule on the same charts?**

Beating 50% is not skill when the asset class goes up 56% of the time.

---

## How to play

1. **Pick a handle.** Open the EPHEMERIS tab and type any name. That's the whole sign-up.
   - Your handle is saved in the URL as `?player=yourname`, so bookmarking the page logs you back in.
   - The switcher moves between profiles.
   - You can set an optional 4-digit PIN (PIN menu) on shared machines.
2. **PLAY (practice, unlimited).**
   - Read the chart.
   - Optionally set your conviction (Low, Medium or High) and a one-line "why is the opposite move unlikely?" note.
   - Call **▲ UP** or **▼ DOWN**.
   - The future candles animate in with a running P&L. Then the card unblinds the ticker, name, class, dates, real price, move, MFE/MAE, base rate, trend-rule call and regime.
   - **Settings:**
     - candle size (1H, 4H, Daily, Weekly, Monthly);
     - visible window (60, 120, 250 or 500 candles);
     - horizon (1–50 candles, or a custom number);
     - universe classes;
     - era (start from a given year, or crisis periods only);
     - blinding toggles: ticker, dates, rebase price to 100;
     - the trend-rule parameters.
   - **Indicators:** 16 of them, each with editable parameters, and you can save favourite sets.
   - **Drill presets:** one click switches timeframe, horizon and indicators together. There are six built in, and you can save your own.
   - **Stops and targets:** turn on Stop / target to set an SL/TP in % or ATR multiples. Preview lines appear before you submit.
3. **DAILY FIVE.**
   - Five charts a day, the same for every player.
   - Fixed settings: Daily candles, 120 visible, 10-candle horizon, SMA 20/50/200 + volume, no stops.
   - One attempt per chart.
   - You get a streak, a separate paper account, a copyable share block (an emoji grid and P&L, no answers) and a recap of yesterday's charts.
   - A new set appears at 00:00 UTC.
4. **STATS.** The dashboard of everything you have played, with filters (see the definitions below).
5. **READ.** Plain-language strengths, weaknesses and your next drill, downloadable as HTML (print it to PDF from the browser).
6. **HISTORY.** Every call, with export to CSV or Parquet.

**Keys (PLAY and DAILY FIVE):**

| Key | Action |
|---|---|
| ↑ | UP |
| ↓ | DOWN |
| N | Next chart |
| S | Skip the reveal animation |

Shortcuts are ignored while you are typing in a box.

---

## Scoring — exact definitions

- **Entry:** the close of the last visible candle, `close[t]`.
- **No stops:**
  - `market_ret = close[t+N] / close[t] − 1`. The outcome is its sign (UP, DOWN or FLAT).
  - `trade_ret = direction × market_ret`.
  - `pnl = trade_ret × stake`.
- **With SL and/or TP:** the trade walks forward candle by candle on OHLC.
  - A **gap through a level fills at the open**, not at the level. That cuts both ways: a worse fill through a stop, a better one through a target.
  - Otherwise the first level touched exits, at that level.
  - If **both are touched in one candle, the stop is assumed to have hit first** (conservative), and the trade is flagged `ambiguous`.
  - If neither is touched, the trade exits at `close[t+N]`.
  - Levels on the wrong side of entry are rejected.
- **Stake:** $1,000 base × conviction (Low 25%, Medium 50%, High 100%).
- **Paper accounts:** $10,000 per mode (practice and daily). Resets keep your history and are marked on the equity curve.
- **A hit** is `trade_ret > 0`. A flat result counts as a miss.
- **Also recorded per call:**
  - % return and P&L in $;
  - R-multiple (when a stop is set) and ATR-normalised return;
  - MFE/MAE over the bars held and over the full horizon;
  - candles held and exit reason;
  - the ambiguous and gap flags;
  - base rate and trend-rule call/return;
  - regime tags at decision time: trend vs SMA-200 and its slope, ATR-percentile volatility regime, distance from the 52-week high, RSI bucket;
  - indicators and blinding used;
  - your note;
  - time taken to decide.

## Benchmarks — on the same charts

- **Coin flip:** 50%, shown with a Wilson band at your sample size.
- **Always-long base rate:** the share of UP outcomes for the chart's asset class × timeframe × horizon. It is measured from the full price store using non-overlapping windows and rebuilt nightly (`base_rates.json`). Your edge is your hit rate minus the mean base rate of *your* charts, tested with a normal approximation to the Poisson-binomial.
- **Trend rule:** long if close > SMA-N and the SMA has risen over the last m bars, short if both are reversed, otherwise no trade. The defaults are N=50 and m=10, and both are configurable. Comparison is **paired per chart** on return, with no-trade counting as 0.

## Stats & Read — definitions and guardrails

- **Headline:** hit rate with a Wilson 95% CI, edge vs base rate (pp, z, p), edge vs trend rule (paired mean, CI, p), expectancy, profit factor, average R, win/loss size, max drawdown and its longest duration, and longest win/loss streaks.
- **Breakdowns:** timeframe × horizon heatmap, plus asset class, sector (Russell 1000 only), regime tags, indicator set, and long vs short with your long bias against the base rate.
- **Calibration:** hit rate by conviction against the implied probabilities (Low 55%, Medium 65%, High 75%), whether High > Medium > Low holds, and a Brier score against 0.25 for a coin.
- **Stops:**
  - the stopped-then-reversed rate;
  - MFE left on the table;
  - a **hindsight** scan of k×ATR stops. It assumes a stop was hit whenever the horizon's worst point reached it, because the order within the horizon is unknown, so it is labelled as an approximation.
- **Guardrails:** any cell with n < 30 is greyed out and never stated as a strength or weakness. Every claim in the Read carries its n and a 95% interval that excludes zero. The Read is generated by fixed rules, not an LLM.

## Data

- **Universe:** about 1,870 instruments.
  - The current Russell 1000 (Vanguard VONE holdings, as used elsewhere in ZENITH).
  - The ZENITH ETF universe (`etfmom`, Morningstar categories). It is split into US Equity ETFs, International Equity, Bonds, Precious Metals, Commodities, Real Estate, Currencies, Crypto and Alternatives.
  - Spot FX pairs, spot crypto, and 8 deep-history indices.
  - Sampling is **class-balanced**: a class is picked uniformly first, then a ticker within it.
- **Prices:** yfinance with `auto_adjust=True`, meaning **split- and dividend-adjusted** (total-return style). Long-run drift is therefore upward, and the base rates include it.
- **Depth by timeframe:**

  | Timeframe | Source | Depth |
  |---|---|---|
  | Daily | yfinance | full history (indices back to the 1920s–80s; most stocks decades; crypto ETFs ~2015+) |
  | Weekly / Monthly | resampled from Daily | full history; Monthly with 500 visible candles fits only about 280 long-history instruments |
  | 1H | yfinance | about 730 days on first fetch; the nightly job appends, so depth grows |
  | 4H | resampled from 1H | same as 1H. US sessions bin 09:30–13:30 and 13:30–16:00 (a short bar); 24h assets use plain 4-hour blocks |

  Settings hides timeframes with no stored data and warns when fewer than 20 instruments fit your window + horizon.
- **Bad-data filter:** a window is redrawn if it contains any of:
  - a single-bar move beyond a per-class limit (an unadjusted split or bad print);
  - a calendar gap beyond the timeframe's limit;
  - 5 or more stale prints in a row;
  - a zero-volume run (except FX and indices);
  - median dollar volume under $1M a day.
- **No repeats:** a player never gets a chart overlapping one they have already played (same ticker and timeframe, decision within one window length).
- **No lookahead:** indicators are computed once over warm-up + window + horizon and then sliced. `tests/test_ephemeris_indicators.py` truncates the series at many decision points and asserts that no visible value changes, for every indicator.
- **No leaks:** before you call, the chart page receives only the visible candles, rebased, on a synthetic time axis. It gets no ticker, name, dates, raw price or future bars, and volume is normalised to its median. A test enforces this.
- **⚠ Survivorship bias:** the universe is *today's* constituents. Delisted, acquired and bankrupt names are missing, which flatters long calls over long histories. See the limitations below.

## Operations

- **Nightly job:** `.github/workflows/ephemeris.yml` runs at 05:30 UTC, Tuesday to Saturday, and can also be started by hand. It takes about 12–25 minutes and does the following:
  1. Rebuilds the universe.
  2. Downloads daily history.
  3. Downloads 1H history and merges it onto the previous release's hourly file.
  4. Writes `daily.parquet` and `hourly.parquet`, one row group per ticker, zstd-compressed with byte-stream-split encoding.
  5. Publishes them to the rolling GitHub release **`ephemeris-px`**.
  6. Rebuilds `base_rates.json` and appends upcoming Daily Five days to `daily_schedule.json`. Existing days are never rewritten.
  7. Commits only the small JSON files under `data/ephemeris/`.
- **App price store:** the app downloads the release assets once per container into `data/ephemeris/px/` (gitignored), refreshes them after 30 hours, and falls back to a stale copy if GitHub is unreachable.
- **Storage:** Supabase Postgres when `st.secrets["ephemeris_db_url"]` is set, otherwise local SQLite (`data/ephemeris/ephemeris.sqlite3`).
  - Use the Supabase **Session pooler** URI, because Streamlit Cloud has no IPv6.
  - Tables are created automatically, with row-level security on and no policies.
  - If the hosted DB fails, the tab says why, in plain words, and keeps playing on SQLite.
- **Local commands:**
  ```bash
  python -m zenith.ephemeris.prefetch --limit 120        # dev subset (daily + hourly)
  python -m zenith.ephemeris.prefetch --base-rates-only  # rebuild base rates from local Parquet
  python -m zenith.ephemeris.seed --player demo-seed     # 200 synthetic calls for the dashboard
  pytest tests/test_ephemeris*.py tests/test_views.py
  ```

## Known limitations

1. **Survivorship bias.** The universe uses current constituents only. The fix is to replay point-in-time membership (`data/mom/membership.json` already archives R1000 membership) together with a delisted-price source.
2. **yfinance as the only source.** No retries are possible beyond what Yahoo serves. A few recent listings (about 15) return no history and are skipped. Hourly depth starts at about 730 days and only grows from nightly appends.
3. **4H session bins.** US sessions produce a short 13:30–16:00 bar. Non-US exchanges' hourly bars are binned on the same rule.
4. **Approximate base rates.** They are pooled over each class's full history with non-overlapping windows, not conditioned on regime or era. A custom horizon uses the nearest tabulated one (1, 3, 5, 10, 20, 30, 50).
5. **Hindsight stop scan.** It uses horizon-wide MAE, not the true path, so it can't tell whether the target or the stop came first.
6. **No drag-to-set SL/TP on the chart.** That needs a bidirectional custom component. Today you enter % or ATR values.
7. **Fragile keyboard shortcuts.** They work by reaching into the Streamlit page from a component iframe, so they could break on a Streamlit upgrade. The buttons always work.
8. **Brief "Running…" overlays.** Changing profile, or the first load, reruns all of ZENITH (tens of seconds). Everything inside the tab reruns only the tab (about 0.4–0.5 s per chart).
9. **The PIN is light protection** (salted SHA-256 behind ZENITH's shared password), not real authentication.
10. **The Daily Five crowd isn't shown.** There is no "how other players called it" yet.

## Recommended next features

- **Point-in-time universe:** include delisted names, to remove survivorship bias.
- **Regime-conditioned base rates:** for example "class × horizon × above/below SMA-200", so edge is measured against a harder bar.
- **Crowd view for Daily Five:** the share of players calling UP, plus a percentile rank.
- **Spaced-repetition drills:** automatically queue more charts from your weakest well-sampled cell, the Read's next drill on autopilot.
- **Setup tags at sampling time:** breakout, trend, pullback, range, ported from VELA's `classifySetup`, to slice skill by pattern.
- **A drag-handle SL/TP component**, and a replay mode to step through the reveal candle by candle.
- **Weekly email or Slack digest** of the Read.
