# OPTIONS VAULT — master brief (read first, every phase)

You are working in the zenith repo (Streamlit research app, Python 3.12, Windows, venv at .venv).
This project builds OPTIONS VAULT: a new zenith tab that is an evidence-graded, visual
"rolodex" of options strategies, edges and portfolio-management mechanics, distilled from
tastylive/tastytrade educational content and cross-checked against academic and practitioner
literature.

## Audience and purpose
- A self-directed investor running an options sleeve (~25–50% of the account) alongside a core of
  long stocks and ETFs.
- Each strategy or edge is readable in under a minute (cheat-card front) and defensible in depth
  (evidence back).
- Actionable: every card ends in concrete numbers (DTE, delta/strikes, credit target, profit
  target, time exit, roll/defend rule, loss rule, size). The user executes them in other tools.

## Non-goals
- Not a screener, chain viewer or strategy playground. No live quotes. Link out instead
  (OptionStrat, Market Chameleon).
- Not investment advice. The package has zenith's standard DISCLAIMER
  ("... Decision-support, not investment advice.").

## Phases (one fresh session each; hand off through files)
| # | prompt | main outputs |
|---|--------|--------------|
| 01 | 01_harvest.md | data/vault/manifest.json, coverage.json, transcripts in data/vault/cache/ |
| 02 | 02_extract.md | taxonomy.json, extract/<episode_id>.json, research_catalog.json, rules_index.json |
| 03 | 03_literature.md | literature.json (independent; may run in parallel with 01–02) |
| 04 | 04_reproduce.md | repro/<name>.json |
| 05 | 05_synthesize.md | strategies.json, edges.json, mechanics.json, portfolio.json, glossary.json |
| 06 | 06_build.md | zenith/vault/view.py + registration + tests |

Run order: 01 → 02 → 04 → 05 → 06. Phase 03 can run at any time.

Every phase:
1. Read this brief and prompts/vault/HANDOFF.md before anything else.
2. If a decision is genuinely the user's (scope, money, installs, anything irreversible), ask. Don't guess.
3. Stop at every GATE and show the gate deliverables. Continue only when the user says so.
4. At the end, append a dated section to HANDOFF.md: what was done, counts, decisions,
   deviations from this brief, open questions, exact re-run commands.
5. Never commit unless the user asks. Never write secrets to disk.

## Zenith conventions (read the referenced files before writing code)
- Package: zenith/vault/. Its __init__.py docstring is the design doc (sources, citations,
  evidence tiers). It has load(name)/save(name, obj) helpers like the other packages, plus
  DISCLAIMER.
- Paths: register VAULT_FILES in zenith/config.py. Artefacts are indented JSON under
  data/vault/. Raw and third-party text lives only in data/vault/cache/; add that to
  .gitignore (like data/cas/cache/).
- HTTP: reuse zenith/fetch.py (Zenith UA, retries) and the politeness rules documented in
  zenith/index/podcasts.py: one source at a time, backoff, no hammering.
- UI: Altair only. Reuse zenith/ui_theme.py (section, stamp, key_findings, evidence_rating,
  help_badge) and zenith/ui_charts.py (numeric_slab, chip, state_banner, note_strip,
  render_chart, hbar, fmt_pct, fmt_money). House rules:
  - full-width charts with explicit heights and labelLimit=0 on category axes;
  - THEME colours (teal = profit/up, coral = loss/down);
  - interactive parts inside @st.fragment (see zenith/trend/view.py).
- Maths: no scipy (it crashed the nightly jobs; commit 1da6c83). Reuse:
  - zenith/beta/hedge.py: _ncdf, bs_put_delta, realized, har_forecast, vrp;
  - zenith/brief/sources.py: _bs_gamma;
  - zenith/pead/earnings.py: upcoming_calendar, earnings_history.
- Dependencies: the Streamlit view must run on requirements.txt alone. Research-only deps
  (yt-dlp, anthropic, ...) go in a new requirements-vault.txt. They are imported lazily
  inside harvest/extract/repro entry points, never by view.py or by tests.
- Tests: tests/test_vault.py, offline, synthetic data and fixtures only (style:
  tests/test_pead.py, fixtures in tests/fixtures/vault/). Run:
  .venv/Scripts/python -m pytest tests/ -q

## Content rules
- Copyright:
  - Transcripts and page text are third-party material. They stay in data/vault/cache/:
    never committed, never displayed.
  - Committed artefacts and the app contain only paraphrased, condensed takeaways and
    numbers, each linked to its source (episode URL + timestamp).
  - At most one short quoted phrase (<15 words) per card, attributed.
- Traceability: every number in the app traces to an episode_id@timestamp, a literature id
  or a repro id. No source, no number.
- Three kinds of statement, always labelled:
  - EDGE: a source of expected return (e.g. the variance risk premium).
  - CONVENTION: a management rule that reshapes the P&L distribution (win rate, variance,
    tail) without itself creating return (e.g. exiting at 21 DTE, taking 50% profit).
    Say what it does to the distribution.
  - MECHANIC: how something works (margin, assignment, order entry, taxes).
- History: tastylive guidance evolved between 2011 and 2026. Record when a rule changed.
  The latest guidance is the default; the history sits on the card back.
- Candour: tastylive studies are mostly SPY/SPX, mid-price fills and unpublished
  methodology, often over short windows. Short premium has fat left tails (2008, Feb 2018,
  Mar 2020). Cards carry these caveats.

## Evidence rubric (zenith's A/B/C via ui_theme.evidence_rating)
- A — replicated in peer-reviewed or major practitioner research across periods and markets,
  and consistent with tastylive data and with our reproduction where one exists.
- B — documented but sensitive to period, costs or implementation; or tastylive data plus one
  independent source.
- C — tastylive-only, small sample, period-specific, or contradicted elsewhere.
- CONVENTIONs are labelled "rule of thumb", with the distribution effect, instead of a return tier.

## Canonical schemas (phases may ADD fields, never rename)
Manifest row (data/vault/manifest.json → {"episodes": [...]}):
```json
{"episode_id": "market-measures-2016-03-14", "show": "market-measures", "air_date": "2016-03-14",
 "title": "", "url": "", "youtube_id": "", "duration_s": 0, "match_confidence": 0.0,
 "transcript": {"source": "yt_manual|yt_auto|apify|whisper|none", "chars": 0},
 "extract": {"status": "pending|done|failed|skipped", "version": "", "model": ""}}
```
Episode extraction (data/vault/extract/<episode_id>.json):
```json
{"episode_id": "", "episode_type": "study|mechanics|concept|commentary|trade_review|other",
 "signal": 0, "strategies": [], "topics": [],
 "studies": [{"question": "", "underlyings": [], "period": "", "entry": {"dte": null, "rule": ""},
   "management": "", "sample_n": null,
   "metrics": {"win_rate": null, "avg_pnl": null, "pnl_per_day": null, "max_loss": null,
               "roc": null, "stdev": null},
   "findings": [], "caveats": [], "ts": "mm:ss"}],
 "rules": [{"kind": "entry|manage|exit|roll|defend|size|select|avoid", "text": "",
   "conditions": "", "numbers": {}, "ts": ""}],
 "claims": [{"text": "", "numbers": {}, "ts": ""}],
 "terms": [{"term": "", "definition": ""}], "unmapped": []}
```
Strategy card (data/vault/strategies.json → {"cards": [...]}):
```json
{"slug": "", "name": "", "family": "", "aliases": [], "one_liner": "",
 "short_what": ["volatility", "theta"], "greeks": {"delta": "0", "gamma": "-", "theta": "+", "vega": "-"},
 "risk": "defined|undefined", "accounts": {"cash": false, "ira": false, "reg_t": true, "pm": true},
 "legs": [{"type": "put|call|stock", "side": "short|long", "delta": 0.16, "dte": 45, "qty": 1}],
 "when": {"iv": "", "outlook": "", "events": ""}, "avoid": [],
 "setup": {"dte": "", "strikes": "", "credit_target": "", "underlying": ""},
 "manage": {"profit_target": "", "time_exit": "", "roll": "", "defend": "", "loss_rule": ""},
 "size": {"bpr_formula": "", "per_trade": "", "notes": ""},
 "stats": [{"label": "", "value": "", "conditions": "", "sources": []}],
 "evidence": {"tier": "B", "kind": "EDGE|CONVENTION|MECHANIC", "note": "", "literature": [], "repro": []},
 "history": [{"year": 0, "change": "", "sources": []}],
 "mistakes": [], "example": {}, "related": [], "sources": []}
```
Edge card (data/vault/edges.json → {"edges": [...]}):
```json
{"slug": "", "title": "", "finding": "", "numbers": [{"label": "", "value": "", "sources": []}],
 "conditions": "", "act_with": [], "tasty_sources": [], "literature": [], "repro": null,
 "agreement": "agrees|mixed|contradicts|untested", "tier": "B", "kind": "EDGE|CONVENTION", "caveats": []}
```
Source references use one format everywhere: `ep:<episode_id>@mm:ss`, `lit:<paper_id>`,
`repro:<name>`, `page:<slug>`.

## Seed hypotheses from the user (verify; do not assume they are right)
- Sell premium, mostly directionally neutral: short strangles or straddles, short strikes
  around 16–20 delta, entered at roughly 45–60 DTE. Manage at 50% of max profit or ~21 DTE,
  whichever comes first. Roll or close.
- Post-earnings: after an outlier earnings move, more than half of companies keep drifting in
  the same direction over the next month.
- The user explicitly wants coverage of 0DTE, directional trades, earnings trades, account
  size, position sizing, taking losses, rolling and margin.
- The user also asks directly: in a 25–50% options sleeve next to long stocks/ETFs, is margin
  needed? How big should positions be? Which strategies should be avoided?
- Futures options (/ES, /MES and similar) are covered as a secondary chapter.
