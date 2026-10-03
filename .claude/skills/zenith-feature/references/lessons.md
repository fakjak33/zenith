# Lessons from PRETOM / FMOM / PEAD — read before building

Hard-won gotchas from the three shipped features. Each one cost real time.

## Data & caching

- **datetime64[ms] index poisons JSON caches into 1970.** pandas `to_json`
  on a `datetime64[ms]` index (what openassetpricing's polars→pandas
  conversion returns) serializes epoch MINUTES; version-dependent
  `read_json` axis conversion then collapses every date to 1970. Fix
  (already in `zenith/cas/backtest/factor_data.py`): normalize the index to ns
  before `cache_put`, magnitude-detect the epoch unit (s/ms/us/ns) on
  parse, and self-heal poisoned caches (year < 1980 → re-download).
- **Cached Ken French tables round-trip with an epoch-ms int index** — parse
  with `unit='ms'` (`factor_data._parse_cached_table`) or all dates
  collapse to 1970.
- **French AC/NI sorts use quintile columns** (`Lo 20` / `Hi 20`), not the
  30/40/30 breakpoints most other sorts use.
- **yfinance serves only ~5 quarters of earnings history**, so any
  time-series signal over past quarters (PEAD's RUE) is mostly
  neutral/flagged early — design the cache to ACCUMULATE across runs
  rather than assuming history is available on day one.
- **Nasdaq calendar `time` field is usually `time-not-supplied`.** Infer the
  session slot from the yfinance `get_earnings_dates` timestamp hour
  (< 12 → pre-market); otherwise pre-market reporters get their reaction
  day a day late and the announcement-return signal misses the move.
- **Nasdaq feed sometimes returns absurd market caps** — screen warns, don't
  trust it blindly.
- **Russell 1000 membership: use the Vanguard VONE holdings API**
  (`investor.vanguard.com/.../VONE/portfolio-holding/stock` paginated,
  gives ticker/name/percentWeight). The iShares IWB CSV endpoint is
  bot-blocked (returns HTML) — do not retry it. Wikipedia components are
  the fallback.
- **yfinance `.news` is broken since 2024** — use Google News RSS per
  ticker (`brief/sources.ticker_news`) instead.

## Streamlit & verification

- **Zenith app screenshots time out in the browser pane** — the SMIL logo
  animation never idles. Verify views headlessly with
  `streamlit.testing.v1.AppTest` (e.g. `AppTest.from_string`,
  `at.selectbox(key='...').set_value(...)`) — this worked perfectly for
  both FMOM and PEAD.
- **Selectboxes cannot be driven via JS/form_input** — state reverts on
  rerun. Same answer: `AppTest`.
- **Streamlit does not hot-reload imported feature modules** — restart the
  server after editing `zenith/<feat>/view.py`.
- **pandas Styler `.format()` called twice resets earlier formatters** —
  merge everything into one dict.
- **Altair category axes truncate labels** — `labelLimit=0` is the real fix
  (`labelOverlap=False` alone is not enough).
- **Streamlit Cloud only sees committed files** — anything the view needs
  must be committed under `data/` (gitignored caches gave blank price
  overlays until `price_panel.json` was committed).

## Screening & honesty

- **Excess-return sanity bands must be ±5, not ±2** — MRVL/MSTR-class real
  movers hit ±2.2 and a tight band makes the screen cry wolf.
- **Ranks against a trailing pool**: store the pool (PEAD keeps
  `rank_pool` per day inside `signals_latest.json`) so percentiles are
  reproducible and the pool trims itself.
- **Rebuild launch-week sheets after a backfill** so early ranks use the
  fat pool, not the first thin days.
- **Never ship a survivorship-biased backtest.** Current-membership
  universes (VONE snapshot) are fine for HOLDINGS/screens with a stated
  caveat, but the user explicitly declined backtests built on them
  (FMOM replication layer = holdings only). Published factor series
  (French/AQR/OSAP) are the survivorship-free backtest substrate.
- **Report replication gaps as-is** — FMOM ships "Sharpe 0.75 vs paper
  0.84" and PEAD states Martineau attenuation (raw long edge ~flat,
  terciles still discriminate). Honest beats impressive.

## Actions & git

- **Self-gate `auto` runs** so the cron is a cheap no-op on ineligible days,
  and make appends idempotent so re-runs heal rather than duplicate.
- **`git pull --rebase --autostash` before the bot commit** — two features
  pushing near-simultaneously will otherwise race.
- **Backfills belong behind `workflow_dispatch`** with a longer timeout
  (PEAD: 120 min dispatch vs 30 min cron), resumable so a timeout doesn't
  lose progress.
