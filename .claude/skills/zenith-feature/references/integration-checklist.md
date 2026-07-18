# Zenith integration checklist — adding a research-backed feature `<feat>`

The seven parts every shipped feature (PRETOM, FMOM, PEAD) has. Gold-standard
templates to copy from:

- `zenith/pead/` — daily event-driven feature, the most complete template.
- `zenith/fmom/` — monthly cadence, multi-family (`families/` subpackage),
  committed catalogs, optional backtest.
- `zenith/pretom/` — calendar/state-machine driven; its `calendar.py`
  (pure NYSE trading calendar, τ-relative windows) is REUSABLE — PEAD already
  imports it (`from ..pretom import calendar`). Do not write a second calendar.

## 1. Package `zenith/<feat>/`

- `__init__.py` — docstring is the authoritative citation block: paper
  authors/years, mechanism, data sources, survivorship caveats, and the line
  "Free/best-effort data only… committed JSON under `data/<feat>/`; the app
  is a thin reader. Decision-support, not investment advice." Defines
  `DISCLAIMER` (user-facing one-liner with citation) and `load()/save()`
  plus archive helpers keyed off `config.<FEAT>_FILES` /
  `<FEAT>_ARCHIVE_DIR`. Archive granularity follows cadence: per-month
  `YYYY-MM.json` (PRETOM/FMOM) or per-reaction-day `YYYY-MM-DD.json` (PEAD).
- **Pure-math module** (`signals.py` / `core.py` / `analytics.py`) — all
  formulas, no I/O. Weights/constants as module-level named constants
  (PEAD: `WEIGHTS`, `CONVICTION`, `POOL_MIN`; PRETOM: `W_HIGH/W_ADV/...`).
- **One network module** (`earnings.py` / `universe.py` / `families/*.py`) —
  ALL fetching + caching for the package; cache via `zenith.cas.store_cas`
  patterns. Fallback chains like PRETOM's universe (cache → Vanguard VONE →
  Wikipedia → last snapshot).
- `compute.py` — orchestrator, `python -m zenith.<feat>.compute --action
  auto|backfill [...]`. `auto` must self-gate (cheap no-op on non-eligible
  days), be idempotent, and self-heal missed runs (PEAD: 5-trading-day
  backscan; PRETOM: late-lock flag; FMOM: cron on days 1-4 so extra days
  repair a missed first).
- `history.py` — append-only pick history: `make_row`, `append_rows`
  (dedupe on a natural key, e.g. `(ticker, report_date)` or
  `(month, model)`), `evaluate_pending` (fills realized returns on later
  runs — pending, never fabricated), `summarize`. Sign-adjusted excess
  convention: positive = the pick worked, shorts included. (PRETOM instead
  rebuilds history from archived months in `compute._rebuild_history` —
  fine when months are fully archived.)
- `view.py` — Streamlit tab. `render()` takes no args, reads committed JSON
  only (never computes/fetches), opens with `st.caption(DISCLAIMER)` and a
  "How it works — the research in 60 seconds" `st.expander` quoting the
  paper's actual numbers. Daily features also export `today_badge()`
  returning an HTML chip or `None`.

## 2. `zenith/config.py`

Add `<FEAT>_DIR`, `<FEAT>_ARCHIVE_DIR`, `<FEAT>_FILES` following the block
at `zenith/config.py:47-77` (PRETOM/PEAD/FMOM entries), and append the new
dirs to the auto-mkdir loop just below (~line 79).

## 3. `.github/workflows/<feat>.yml`

Clone from `pead.yml` (daily) or `fmom.yml` (monthly). Shared skeleton:

- `on: schedule` (cron) + `workflow_dispatch` with an `action` input.
- `permissions: contents: write`; `concurrency: group: <feat>,
  cancel-in-progress: false`.
- Steps: checkout → setup-python 3.12 → `pip install -r requirements.txt` →
  `python -m zenith.<feat>.compute --action ${{ github.event.inputs.action
  || 'auto' }}` → commit step.
- Commit step (identical across the trio): configure `zenith-bot`,
  `git pull --rebase --autostash`, `git add data/<feat>/`, commit only if
  `! git diff --cached --quiet` with message
  `"<feat>: $(date -u +%Y-%m-%d) <action>"`, push.
- Stagger the cron so features don't race on the same push window
  (PRETOM 22:45, PEAD 23:15 weekdays; FMOM 13:30 on days 1-4).

## 4. `app.py` wiring

- Add the tab to the `st.tabs([...])` call (~`app.py:98`).
- `with tab_<feat>:` block: `st.markdown(section("<TITLE>", <color_idx>),
  unsafe_allow_html=True)` then
  `from zenith.<feat> import view as <feat>_view; <feat>_view.render()`
  (pattern at `app.py:129-145`).
- Add one numbered bullet to the "What is Zenith?" orientation expander
  (~`app.py:22-41`).
- Daily features: import `today_badge` into the TODAY tab block
  (~`app.py:102-110`, see the pretom/pead imports) and render if non-None.

## 5. UI conventions

`ui_theme.section(label, idx, help)`, `ui_theme.stamp(as_of, page)` ("DATA
AS OF" banner on every tab), `ui_theme.help_badge(text)` for per-column "?"
tooltips; colors from `config.THEME.section_colors`. Charts: altair with
`labelLimit=0` on category axes (labels truncate otherwise). Tables: pandas
Styler — merge all `.format()` calls into one dict (a second call resets
earlier formatters).

## 6. `tests/test_<feat>.py`

Offline, synthetic data only — no network. Model on the existing suites:
`tests/test_pead.py` (~30), `test_fmom.py` (~40), `test_pretom.py` (14).
Must cover: math bounds/monotonicity, no-lookahead, gate/invariant logic per
reason, idempotent `append_rows`, `evaluate_pending` horizons, archive
roundtrip + history schema, and any calendar/timing edges (Friday→Monday,
holidays, pre/after-hours).

## 7. `scripts/screen.py`

Add `screen_<feat>()` (see `screen_pretom` at `scripts/screen.py:121`,
`screen_fmom` :169, `screen_pead` :252) and call it in `main()` (:317).
Standard checks: `status`/`as_of` freshness via `_days_old`, artefacts
present and non-empty, no duplicate keys, values in sane bounds (scores
∈ [0,1], composites ∈ [0,100], excess returns ∈ [−5,5] — NOT ±2, real
movers exceed it), ranks contiguous from 1, model invariants (leg weights
±1, two-confirmation gates), plus one printed "eyeball" line for a human.
Exit non-zero on hard fails; iterate until `SCREEN PASSED`.
