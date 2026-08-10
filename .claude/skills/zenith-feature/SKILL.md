---
name: zenith-feature
description: Turn statistically significant empirical finance research into a new self-updating, monitored Zenith tab — screen and analyze the research, replicate the model as faithfully as free data allows, then integrate it following the PRETOM/FMOM/PEAD pattern (package + Action + tab + history + tests + screen). Use when the user wants a NEW standalone signal/feature built from a paper or an anomaly topic. For a lightweight note or a signal tweak inside the existing CAS monitor, use zenith-research instead.
---

# Zenith research-to-feature pipeline

Codifies the process that built PRETOM (turn-of-month shorts), FMOM (factor
momentum), and PEAD (post-earnings drift): research → faithful replication →
self-updating monitored tab. Two hard approval gates — after research analysis
and after replication testing — because the user makes design calls mid-stream
(FMOM: declined survivorship-biased backtests; PRETOM: track both windows).

Supporting references (read when you reach the relevant phase):
- `references/integration-checklist.md` — the exact seven-part Zenith
  integration inventory with template files to copy from.
- `references/lessons.md` — hard-won gotchas from the three shipped features.
  Read this BEFORE Phase 2; several bugs there cost hours the first time.

Standing conventions (apply to every phase):
- Repo venv is `.venv/Scripts/python.exe` — `python` is not on PATH.
- Free/best-effort data only, so the GitHub Action keeps running forever.
- Decision-support, not investment advice — every view carries a `DISCLAIMER`.
- History is append-only and defers evaluation ("pending") rather than
  fabricating realized returns. Never fabricate; state divergences honestly.

## Phase 0 — Route check

If the request is really "log this paper / nudge a CAS weight / scaffold a
signal inside an existing CAS family", stop and use the `zenith-research`
skill instead. This skill is for a new standalone tab with its own package,
Action, data artefacts, and monitoring.

## Phase 1 — Research screening & compilation

Two entry points:

**(a) Paper provided** (PDF path / URL / pasted text): read it fully — the
methodology and construction sections, not just the abstract. PDF → `Read`
with `pages`; URL → WebFetch (scrape tools if blocked).

**(b) Topic or anomaly named** (e.g. "low-vol anomaly", "insider buying"):
run a literature screen first. Find the canonical paper(s) plus the best
follow-ups, and vet:
- effect size and t-stats (post-2016 bar: |t| > 3 is the credible zone);
- out-of-sample and post-publication evidence — McLean & Pontiff (2016) style
  decay (~58% of in-sample returns survive publication on average), and any
  replication studies (e.g. Chen & Zimmermann's Open Source Asset Pricing,
  Hou-Xue-Zhang replications);
- whether the effect survives in liquid large-caps (Zenith screens are
  Russell 1000 based) and after costs;
- the mechanism — behavioral/institutional stories that explain WHY it
  persists (PRETOM: institutional cash-raising; PEAD: underreaction).

Either way, produce a **research brief** for the user:
1. Papers (authors, year, journal/SSRN) and the mechanism in plain words.
2. Headline statistics to replicate later (Sharpe, monthly premium, hit
   rates, key table numbers — quote them exactly; they seed the in-app
   "research in 60 seconds" expander and the fidelity test).
3. Sample and universe (dates, market-cap band, filters).
4. Signal construction details: formation/holding windows, sorting variable,
   weighting, rebalance timing, breakpoints.
5. Known caveats: decay, crowding, cost sensitivity, small-cap dependence.

Then assess **free-data feasibility** against the proven sources: yfinance
(prices, ~5 quarters of earnings, fundamentals via `.info`/statements),
Nasdaq earnings calendar (`api.nasdaq.com` — carries actual EPS + estimates),
Vanguard VONE holdings API (Russell 1000 membership + weights), Ken French
library / AQR datasets / Open Source Asset Pricing (published factor
returns), FRED (macro). Say plainly what the paper uses that we can't get
(CRSP/Compustat point-in-time, I/B/E/S) and what the honest substitute is.

Pick a **cadence**: daily weekday (event-driven like PEAD/PRETOM) vs monthly
(portfolio-formation like FMOM). Cadence decides Action cron, whether the
feature gets a `today_badge()`, and archive granularity.

## GATE 1 — design approval (AskUserQuestion)

Present the research brief plus a proposed model design: signal math and
weights, universe, data sources, cadence, and honest limitations
(survivorship, publication lag, coverage). Use AskUserQuestion for every
design choice the papers leave open — e.g. which window variant to track,
whether to include a backtest at all, long-only vs long-short display.
Do not start building until the user signs off.

## Phase 2 — Replicate / build the model

Read `references/lessons.md` first. Then, in this order:

1. **Pure-math module** (`signals.py` / `core.py` / `analytics.py`): every
   formula from the paper as pure functions of plain data — no I/O, no
   network, fully unit-testable. Clip/winsorize exactly as the paper does;
   keep the paper's variable names in comments.
2. **One isolated network module** (like `pead/earnings.py`,
   `pretom/universe.py`): ALL fetching + caching lives here, cached via the
   existing store patterns so backfills don't re-download.
3. **Replication check**: where free data allows, reproduce the paper's
   headline stats and compare honestly (FMOM shipped "AQR TSFM Sharpe 0.75
   vs paper 0.84" — that is the standard). Where data doesn't allow it,
   state the divergence in-app and in the package docstring; never
   substitute a survivorship-biased or fabricated number. Holdings-only
   (no backtest) is a legitimate outcome — FMOM's replication layer did
   exactly that at the user's request.

## Phase 3 — Test fidelity

- `tests/test_<feat>.py`, offline with synthetic data only (no network),
  modeled on `tests/test_pead.py` / `test_fmom.py` / `test_pretom.py`:
  math bounds and monotonicity, no-lookahead, gate/invariant logic,
  idempotent history appends, deferred evaluation, schema/archive roundtrip.
- Build a **fidelity table**: replicated numbers vs the paper's published
  numbers, with a one-line explanation for each gap (sample, universe,
  data source).
- Run `.venv/Scripts/python.exe -m pytest tests/` — all green, including the
  pre-existing suites.

## GATE 2 — replication approval (AskUserQuestion)

Present the fidelity table and test results. Confirm the model is faithful
enough to wire into the app — or take direction (adjust construction, drop a
component, change display) before integration.

## Phase 4 — Integrate into Zenith

Follow `references/integration-checklist.md` step by step. Summary of the
seven parts (details, exact paths, and template files are in the checklist):

1. Package `zenith/<feat>/` — `__init__.py` (paper-citing docstring,
   `DISCLAIMER`, load/save + archive helpers), pure math, isolated network
   module, `compute.py` (`--action auto|backfill`, self-gating, idempotent,
   self-healing), `history.py` (append-only, `evaluate_pending`), `view.py`
   (`render()` reading committed JSON only, optional `today_badge()`).
2. Register artefacts in `zenith/config.py`.
3. `.github/workflows/<feat>.yml` cloned from the existing trio.
4. Wire the tab in `app.py` (+ orientation bullet, + TODAY badge if daily).
5. UI conventions: `ui_theme.section/stamp/help_badge`, `st.caption(DISCLAIMER)`,
   a "How it works — the research in 60 seconds" expander quoting the
   paper's actual numbers.
6. Tests (done in Phase 3; extend for integration pieces).
7. `screen_<feat>()` in `scripts/screen.py`, registered in `main()`.

Work on a feature branch. Backfill history survivorship-honestly (current
membership caveat stated wherever it applies), then PR to main — never
commit directly to main.

## Phase 5 — Verify

1. `.venv/Scripts/python.exe -m pytest tests/` — full suite green.
2. `.venv/Scripts/python.exe scripts/screen.py` — iterate until
   `SCREEN PASSED` (fix root causes in code, not data).
3. Verify the view headlessly with `streamlit.testing.v1.AppTest` — NOT
   browser screenshots (the SMIL logo never idles; screenshots time out)
   and NOT JS-driven selectboxes (state reverts on rerun).
4. After merge, prove the Action with a `workflow_dispatch` run and check
   the bot commit landed under `data/<feat>/`.
5. Present the result: what shipped, the fidelity table, and what the user
   should watch during the first live cycle.
