# EPHEMERIS — Tape-Reading Trainer tab for ZENITH


> **Status (2026-10-03): all five phases complete.** User guide, scoring definitions, data
> caveats, operations, known limitations and next features: see [EPHEMERIS_README.md](EPHEMERIS_README.md).
> Open questions resolved: Supabase (Session pooler URI in `ephemeris_db_url`), $1,000 base stake,
> RTH-only 4H for US sessions / 4-hour blocks for 24h assets, rolling `ephemeris-px` release.
> Changes vs this plan: the Daily Five note became optional (user request, for flow); Daily Five
> charts are frozen in a committed append-only schedule rather than re-derived from the seed each day.

## Context
You want a blind-chart prediction game inside ZENITH for deliberate practice in tape reading. Each guess is logged per player and feeds an analytics report that rigorously answers one question: am I beating the base rate and a simple trend rule?

VELA (`../vela`, Next.js on Vercel) was the first attempt. Its logic is good, but sign-up was too much friction: claim a unique name, get a recovery code, get a JWT cookie.

EPHEMERIS becomes ZENITH tab #19 and follows ZENITH's existing patterns.

**Decisions you made:**
- New Supabase project for storage, with a SQLite fallback.
- Price store published as a GitHub Release asset (Parquet).
- Game board built on lightweight-charts inside `components.html`; Altair for the dashboards.
- Universe uses derived sub-classes plus spot crypto and FX.

**First implementation step:** commit this plan as `PLAN.md` at the repo root. Then build phase by phase, stopping after each phase for a run, a screenshot and your sign-off.

## What exploration found (reuse, don't reinvent)
- **Tabs:** `app.py:171` has one hardcoded `st.tabs([...])` list. Each feature renders through `render_feature("zenith.<pkg>.view", NAME)`, wrapped in `section(label, idx)`.
  - A new tab also touches: the "What is Zenith?" expander (`app.py:23`), `_BADGE_SOURCES` (~`app.py:203`, needs `today_badge()` that never raises), and `tests/test_views.py` (full-app tab-label test ~L465).
  - The pattern to copy is commit `6907daf` (CLEAN BETA).
- **Theme:**
  - `zenith/config.py:850` `THEME`: bg `#000`, panel `#0b0b0b`, grid `#2c2c2c`, teal `#2ec4b6` (up), coral `#ff5a3c` (down), mustard, mauve, navy, mint. Fonts are VT323 for display and Space Mono for body.
  - CSS lives in `zenith/ui_theme.py` (`CSS`, `section`, `stamp`, `help_badge`, `key_findings`, `evidence_rating`).
  - In-tab helpers live in `zenith/ui_charts.py` (`numeric_slab`, `chip`, `state_banner`, `note_strip`, `render_chart`, `hbar`, `grad_diverging`, `diverging_scale`, `fmt_pct`, `fmt_money`, `colcfg`).
  - ZENITH has no Plotly. The JS chart gets the same tokens injected from `THEME`.
- **Prices:**
  - `zenith/cas/sources/prices.py:17` `get_history` wraps yfinance with `auto_adjust=True` and batches of 50. It has no retries.
  - It also has a cache-clobber bug: it rewrites the whole per-period cache with only the requested tickers. EPHEMERIS will **not** use it at runtime. The prefetch script calls yfinance directly with retries and backoff.
- **Universe:**
  - `zenith/etfmom/universe.py` `ASSET_CLASSES` and `asset_class_of(category)`, with data in `data/etfmom/scores_latest.json` (912 ETFs, Morningstar categories).
  - R1000 comes from `zenith/pretom/universe.py:121 russell1000()` (Vanguard VONE, 1024 names) or `data/pretom/universe.json`.
  - Sector and market cap come from `data/mom/meta.json`.
- **Hosting:** Streamlit Community Cloud. GitHub Actions commit data. Only `st.secrets["app_password"]` exists today (a shared gate in `zenith/auth.py`). No DB, no `st.query_params`. Streamlit is 1.58, and psycopg2 and pyarrow are already in `.venv`.
- **Performance hazard:** every rerun executes all 18 tabs. The whole game loop must live inside `@st.fragment` (precedent: `zenith/trend/view.py:325`).
- **Tests:** offline, seeded synthetic data (`tests/test_beta.py` style), and `AppTest` view tests (`tests/test_views.py::_render`).

### From VELA: port vs drop
- **Port to Python:**
  - `src/lib/indicators.ts` (SMA, EMA seeded with SMA, Bollinger, Wilder RSI and ATR, MACD), checked against its hand-computed test fixtures.
  - `bars.ts:rebase` (rebase price to 100; volume divided by its median).
  - `build-pool.mjs`: `windowIsClean` (per-class max single-bar move, max day gap), the liquidity filters ($5M median dollar volume, $3 minimum price), class-quota round-robin, `classifySetup` (breakout, trend, pullback, range), and the Daily-5 difficulty shape (easy, med, med, med, hard; at most 2 per class; no symbol repeated within 30 days).
  - `gamedate.ts` streak functions.
  - The anti-leak rule: the chart payload never carries ticker, date or raw level, enforced by a regression test like `leak.test.ts`.
- **Drop:**
  - Onboarding flow: unique-name claim, 409 on a taken name, recovery code, JWT cookie, restore endpoint.
  - Leagues.
  - VELA's 2×/5×/10× leverage stakes (spec: 25/50/100% of base stake).
  - Daily bankroll reset.

## Architecture
```
zenith/ephemeris/
  __init__.py      DISCLAIMER, constants (BASE_STAKE, CONVICTION_MAP, HORIZONS, LOOKBACKS, TIMEFRAMES)
  universe.py      build tagged universe: ticker, name, class, subclass, sector, cap_bucket
  prefetch.py      CLI: yfinance → Parquet (daily max, 1h 730d) with retries/backoff; builds meta + base rates
  store_px.py      load Parquet (download Release asset once per container → /tmp, st.cache_resource); resample 4H/W/M
  sampler.py       draw (ticker, end_idx) with class-balanced weights, bounds, bad-data filter, no-repeat, era filter
  indicators.py    pure-numpy causal indicators (overlays + panes)
  scoring.py       score_trade(): no-stop / SL-TP walk, gaps, ambiguity, MFE/MAE, R, ATR-norm
  benchmarks.py    coin-flip band, base-rate table lookup, trend rule (configurable)
  regime.py        regime tags at decision time
  daily.py         Daily Five: deterministic schedule from date seed, share text, streaks
  repo.py          Repository protocol + SqliteRepo + PostgresRepo (Supabase via psycopg2, pooler URL in st.secrets)
  profiles.py      handle ↔ ?player= query param, switcher, optional PIN (sha256+salt)
  stats.py         Wilson CI, binomial/z tests vs base rate, expectancy, PF, DD, streaks, Brier, breakdowns, n<30 flags
  report.py        deterministic "Read" rules → plain-language findings + drills; HTML export
  chart/board.html lightweight-charts template (CDN-pinned), reveal animation, PnL ticker, keyboard bridge
  chart.py         builds the JSON payload (rebased, no identifiers) + renders via components.html
  view.py          render() + today_badge(); sub-views: PLAY · DAILY FIVE · STATS · READ · SETTINGS
.github/workflows/ephemeris.yml   nightly: prefetch → upload Release asset `ephemeris-px`; commit small meta JSON
data/ephemeris/   universe.json, base_rates.json, daily_schedule.json (small, committed)
tests/test_ephemeris_*.py
EPHEMERIS_README.md
```
**Modified:**
- `app.py`: tab label, render block, expander entry, `_BADGE_SOURCES`.
- `zenith/config.py`: an `EPHEMERIS_*` paths block.
- `requirements.txt`: add `psycopg2-binary`. pyarrow already ships with Streamlit.
- `tests/test_views.py`.
- `.gitignore`: add `data/ephemeris/px/`.

## Key design choices
- **Data:**
  - **Daily** bars use full history with `auto_adjust=True` (split and dividend adjusted; stated in the UI).
  - **Weekly and Monthly** are resampled from daily.
  - **1H** uses yfinance's ~730-day cap. **4H** is resampled from 1H, anchored to the session open: 09:30–13:30 and 13:30–16:00, the latter a short bar, documented. The nightly job appends 1H bars so the depth grows over time.
  - A timeframe or lookback combination is hidden when too few tickers can fit window + horizon (computed from the meta).
  - Float32 Parquet, one file per timeframe, partitioned by ticker.
- **Universe classes:** US Equity (R1000), US Equity ETFs, International Equity ETFs, Bonds, Precious Metals, Other Commodities, Real Estate, Currencies (ETFs plus spot `EURUSD=X` and similar, no volume), Crypto (ETFs plus `BTC-USD`, `ETH-USD`, …), and Alternatives.
  - The International and Precious Metals splits come from Morningstar categories.
  - Sampling weight is equal per selected class, then uniform over tickers within a class.
- **Survivorship bias:** the universe is current constituents only. This is flagged in the footer and in PLAN.md. A later path to fix it is the point-in-time archive in `data/mom/membership.json` plus a delisted-price source.
- **Indicators:**
  - Computed on the full fetched series with warm-up bars, then sliced.
  - The causality unit test truncates at every decision index and asserts the visible values are identical.
  - Indicator values for reveal bars ship with the payload so overlays keep drawing during the reveal.
- **Game board:**
  - One `components.html` iframe receives rebased bars plus indicator series. Before the call it receives *only* the visible bars, so future data is not in the DOM before submit.
  - On submit, the fragment scores the trade server-side, re-renders with the future bars and `reveal=true`, and JS animates them at about 60 ms per candle (skippable, with a running PnL ticker). It highlights the SL/TP candle, shows the outcome badge, then the unblind card.
  - SL/TP lines are previewed from inputs as % or ATR multiples. Drag-to-set needs a bidirectional custom component, which is deferred.
  - Keyboard shortcuts: a listener on `window.parent.document` clicks the Streamlit buttons, verified in Phase 5.
- **Scoring:** implemented exactly as specified.
  - Entry is close[t].
  - With no stops, PnL = dir × ret × stake.
  - With SL/TP, walk the OHLC bars forward: the first level touched exits; if both are touched in one bar, the stop wins and the trade is flagged ambiguous; a gap through a level fills at the open.
  - Stake = BASE_STAKE (default $1,000) × conviction % (25/50/100), applied to a $10,000 account per mode. Resets are stored as events and marked on the equity curve.
- **Benchmarks:** computed on the exact same charts.
  - Coin flip: 50% with a Wilson band.
  - Base rate: P(up) by class × timeframe × horizon, precomputed nightly into `base_rates.json`.
  - Trend rule: close > SMA-N and slope > 0 is long, the reverse is short, otherwise no trade. N and slope lookback are configurable.
- **Daily Five:**
  - The nightly job commits `daily_schedule.json`: 365 days of 5 (ticker, end_date) pairs per day, seeded per date, using VELA's difficulty shape.
  - Windows are fixed, so data refreshes can't change a day's charts.
  - Settings are locked to Daily / 10-candle horizon / default indicators / no stops / note on.
  - One attempt per day is enforced by the `daily_results` primary key `(player, date)`.
  - Share text is an emoji grid plus PnL, offered as a copyable code block.
- **Profiles:**
  - On first visit, type a handle and the profile is created immediately.
  - `st.query_params["player"]` is kept in sync, so a bookmark means you are logged in.
  - A selectbox switches between existing profiles.
  - An optional 4-digit PIN is stored salted and hashed.
  - All of this sits behind the existing app-wide `require_password()`.
- **Persistence:**
  - The `Repository` protocol has these methods: `get_or_create_player`, `log_guess`, `guesses_df`, `save_daily`, `daily_for`, `presets`, `save_preset`, `reset_account`.
  - `PostgresRepo` is used when `st.secrets["ephemeris_db_url"]` exists, otherwise `SqliteRepo` (local file in dev, in-memory in tests).
  - The schema is your suggestion plus `player_id` FKs, `account_resets(player, mode, ts)`, `indicators_json`, `blinding_json`, `regime_json`, `setup_tag`, and `chart_key` (a hash of ticker, timeframe and end used for no-repeat).
  - DDL lives in `zenith/ephemeris/schema.sql`. You run it once in the Supabase SQL editor.
- **Stats and Read:**
  - All computation runs in pandas/numpy. Tests and CIs need no scipy: Wilson intervals, a normal approximation and an exact binomial via `math.comb`.
  - Cells with n < 30 are greyed out. No claim is shown without its n and CI.
  - The Read report comes from deterministic templates ranked by significance. It downloads as HTML; PDF means printing the HTML from the browser, so no new dependency.

## Phases (stop after each for run + screenshot + your confirmation)
1. **Core loop.**
   - Includes: universe, prefetch for Daily, Release-asset loader, sampler, a minimal board (candles, NOW marker, horizon shading), UP/DOWN, no-stop scoring, repo (SQLite + Postgres), profiles via `?player=`, and the tab wired into `app.py`.
   - Tests: scoring, sampling bounds, leak payload.
2. **Trade mechanics.** Indicators with causality test, SL/TP with validation, conviction, reveal animation and PnL ticker, unblind and context card (MFE/MAE, base rate, rule call), benchmarks, regime tags.
3. **Modes and timeframes.** Daily Five (schedule, streak, recap, share text, note required), presets, 1H/4H/W/M with depth gating, era filter.
4. **Analytics.** Stats dashboard with filters, the Read report, CSV/Parquet export, and a 200-guess synthetic seed profile.
5. **Polish.** Prefetch the next chart in the background during the reveal (target < 1 s), keyboard shortcuts, visual pass with the `zenith-screen` skill, `EPHEMERIS_README.md`, and a list of known limitations.

## Verification
- `.venv/Scripts/python.exe -m pytest tests/test_ephemeris_*.py tests/test_views.py -q`. Tests cover:
  - scoring fixtures: same-candle ambiguity, gap-through-stop fills at open, no-touch exit at close[t+N];
  - indicator truncation causality;
  - Daily seed determinism;
  - sampler bounds and no-repeat;
  - stats against hand-computed fixtures;
  - the leak test;
  - full-app `AppTest` including the EPHEMERIS label.
- Run the app with the `.claude/launch.json` "zenith" config (port 8601), drive it in the browser pane, and screenshot each phase.
- Phase 1 also covers: play 10 charts on SQLite, confirm `?player=` survives a reload, and run with a Supabase URL in `.streamlit/secrets.toml` to confirm the rows appear.

## Open questions (to confirm before or while I implement)
1. **Supabase project:** you create it and paste the pooler connection string into Cloud secrets as `ephemeris_db_url`. Until then I'll build against SQLite.
2. **Base stake:** OK with $1,000 per trade, so High conviction is 10% of the account?
3. **4H session handling:** RTH-only bars (09:30–16:00) for equities, and 24h for crypto and FX?
4. **Release asset upload:** the nightly workflow uses `GITHUB_TOKEN` with `contents: write`. Is it OK to create a rolling `ephemeris-px` release?

## Data caveats (also shown in the tab footer)
- **Survivorship bias:** the universe is *current* R1000 constituents and currently-listed ETFs only. Names that were delisted, acquired or went bankrupt are missing, which biases long calls upward over long histories. Path to fix: use the point-in-time membership archive in `data/mom/membership.json` and a delisted-price source, added later.
- **Adjustment:** all prices come from yfinance with `auto_adjust=True`, meaning split- and dividend-adjusted (total-return style) OHLC.
- **Depth by timeframe:**

  | Timeframe | Source | Depth |
  |---|---|---|
  | 1H | yfinance | about 730 days, growing through nightly appends |
  | 4H | resampled from 1H | about 730 days, ~2 bars per RTH session for equities |
  | Daily | yfinance | full history (varies by ticker; crypto ETFs from about 2015, spot BTC-USD from 2014) |
  | Weekly / Monthly | resampled from Daily | full history |

  Monthly with a 500-candle lookback needs about 42 years of data, so most combinations like it will be hidden by the depth gate.
