# OPTIONS VAULT — handoff log

Each phase appends a dated section with these headings:
- Done
- Counts
- Decisions (with who made them)
- Deviations from the brief
- Open questions
- Re-run commands

---

## 2026-10-23 — Prompt pack written (planning session)
- Done: wrote prompts/vault/00–06 and this log. No code yet.
- Decisions (user): prompt pack per phase; sources = Market Measures archive, research shows,
  written Learn Center + help center (no live trading shows); free-data reproductions;
  re-runnable manual refresh (no GitHub Action); Claude API Message Batches for extraction;
  new OPTIONS VAULT tab in zenith (zenith/vault/); generic sizing tiers + calculator;
  Streamlit only (no PDF export).
- Decisions (planning default, not vetoed): futures options covered as a secondary chapter.
- Open questions: none yet. Next step is Phase 01 (and optionally Phase 03 in parallel).

---

## 2026-10-24 — Phase 03 literature review (session started 2026-10-23)
- Done: deep-research pass over the 12 topics in 03_literature.md (7 parallel research notes, one
  synthesis), then an independent citation check: every DOI resolved via api.crossref.org and
  compared on first author, year and title; every non-DOI URL fetched (script, then browser/curl
  for bot-blocked regulator/exchange/press sites). Output: data/vault/literature.json.
  Long-form report + research notes (not committed, outside the repo):
  side-session-notes/local_8dfcc8f0-3292-400a-9000-d6fe5c2a8218/reports/Options vault literature review.md
  and .../research_notes/Options vault literature review/*.md.
- Counts: 12 topics; 209 unique papers/sources (119 verified=yes, 90 partial); 148 DOIs match
  Crossref; 61 entries in "unverified" (kept-partial / dropped / needs-check, each with the reason).
  Tiers: A = vrp, tail-hedging, trading-costs, mechanics; B = option-selling, iv-predictors,
  earnings, skew, option-anomalies, sizing-risk, management-rules; C = zero-dte.
  Agreement with tastylive: agrees = vrp, tail-hedging, sizing-risk, mechanics; mixed = the other 8;
  no topic is an outright contradicts, but agreement_detail has claim-level contradictions.
- Decisions (Phase 03 session, not yet reviewed by the user):
  - Schema additions (no renames): topic.kind, tier_note, agreement_note, agreement_detail[],
    computed_stats[]; paper.doi, kind, verified (yes|partial), verify_note, check{result,date};
    top-level meta (rubric + verification method/counts) and unverified[].
  - "partial" = metadata checked but some numbers from secondary summaries or memory; the finding
    text says so. Phase 05 should not put a partial-only number on a card front without re-checking.
  - computed_stats (option-selling, tail-hedging) are Phase 03's own calculations from Cboe/Yahoo
    daily index files, gross of costs. They are NOT lit: sources. Phase 04 should reproduce them as
    repro:<name> before any card cites them.
  - Dropped from papers (dead links on 2026-10-24): Cole 2020 "Dragon portfolio" (404), IBKR
    portfolio-margin page (404).
- Findings Phase 05 needs to carry onto cards:
  - IV-rank timing has no peer-reviewed test (C). Cross-sectional IV-vs-RV and term-structure
    signals are B. The 45-60 DTE preference is longer than where the literature finds the variance
    premium concentrated (inference, not a direct test).
  - Single-stock put skew predicts low stock returns (Xing-Zhang-Zhao 2010): selling puts because
    a name's skew is steep contradicts the evidence. Index skew premium is the same risk as the VRP.
  - Earnings: IV crush is predictable (A) but whether it is overpriced is unsettled; Gao-Xing-Zhang
    (2018) find buying straddles 3 days before earnings paid. Retail buyers lose (agrees).
  - User's drift hypothesis: about a coin flip for large caps since 2006 (Martineau 2022); continues
    after news/analyst-backed moves, reverses after no-news moves. No paper reports the hit rate.
  - Management rules (50%, 21 DTE, rolling, stops) are CONVENTIONs: they reshape the distribution and
    capital turnover, with no evidence they add expected return per trade.
  - Since 2009 every Cboe short-vol index trailed the S&P 500 TR; CNDR (iron condor) ~cash.
  - Mechanics: PDT designation and $25k minimum eliminated (SR-FINRA-2025-017, effective
    2026-06-04, brokers may phase in to 2027-10-20). tastytrade IRA "The Works" allows uncovered calls
    (not just defined-risk). Portfolio margin minimum $100k is in a 4210 Interpretation, not the rule.
- Deviations from the brief:
  - Corrected starting points: Israelov "Pathetic Protection" = J. Alternative Investments 21(3),
    2018 (not 2019 / J. Derivatives); Martineau = Critical Finance Review 2022; George-Hwang-Li and
    Brandt et al. are SSRN working papers (never published); the liquidity-premium paper is
    Christoffersen, Goyenko, Jacobs & Karoui (RFS 2018); "0DTE Asset Pricing" is Almeida, Freire &
    Hizmeri, not Bandi-Fusari-Reno; Bondarenko PUT studies are Cboe white papers.
  - "Spina" resolved to Spina & Kulikov, The Unlucky Investor's Guide to Options Trading (Wiley,
    2022, ISBN 9781119882657); its table values could not be extracted.
  - Research notes and report live in the shared notes folder, not the repo (task constraint).
- Open questions:
  - zenith/pead/__init__.py cites George, Hwang & Li 2015 and Martineau 2021; the checked years are
    SSRN 2014 and CFR 2022. Not edited here (out of scope) — fix in a separate change.
  - Gao-Xing-Zhang: exit timing relative to after-close reports decides how hard it contradicts
    tastylive. Needs a read of the paper.
  - No independent net-of-cost backtest of 16-delta 45-DTE SPX strangles through 2008/2018/2020
    was found: a Phase 04 reproduction target. So is the drift continuation rate by size and
    pre/post-2006 from zenith's PEAD data.
  - 17 "needs-check" items in unverified[] (SPAN internals, /ES option exercise style, ODD text,
    RUT/NDX 1256 status, VIX-tier buying-power ladder, "1-5% per trade").
- Re-run commands (scripts in side-session-notes/local_8dfcc8f0-3292-400a-9000-d6fe5c2a8218/scripts/):
  - rebuild draft:  .venv/Scripts/python -I <scripts>/build_lit.py <draft.json>
  - check citations: .venv/Scripts/python -I <scripts>/verify_lit.py <draft.json> <verify_report.json>
  - write artefact: .venv/Scripts/python -I <scripts>/finalize_lit.py <draft.json> <verify_report.json> data/vault/literature.json
- Gate (user, 2026-10-24): Phase 03 gate accepted as shown — tiers, agreement grades and the
  unverified list stand. data/vault/literature.json is the reference Phase 05 grades against.

---

## 2026-10-24 — Phase 04 free-data reproductions (side session started from the planning session)
- Done: 7 reproductions on free data, built before Phase 02 (Phase 01 was still harvesting).
  Code: zenith/vault/repro/ (data.py fetch layer; vrp.py, cboe.py, strangle.py, earnings.py,
  expected_move.py, vix_spikes.py pure compute + build; charts.py Altair builders from chart_data;
  run.py entry point). Output: data/vault/repro/<name>.json (cite as repro:<name>). Tests:
  tests/test_vault_repro.py (23, offline, synthetic). Raw downloads cached in
  data/vault/cache/repro/ (already gitignored). Gate B page (private artifact):
  https://claude.ai/artifact/2hSVd4c48qRYQHoKqZPzpn
- Counts / headline results (verdicts are on the claim, not on tastylive):
  - vrp_spx (1990-2026): VIX > next-21d realized on 85% of days, mean +4.1 vol pts; only 2008
    negative. Claim "sell more when IV rank is high" -> contradicts: IVR 80-100 spread +3.3 pts vs
    +3.9 at IVR 0-20, with a far worse 5% tail (-38.7 vs -2.6 pts).
  - cboe_short_vol / cboe_tail_hedge: Phase 03 computed_stats reproduced 80/80 and 33/33 within
    tolerance (CAGR 0.15 pt, Sharpe 0.05, drawdowns 0.5 pt, dates 3 days). Phase 05 can cite
    repro:cboe_short_vol / repro:cboe_tail_hedge for those numbers. PUT +5.7 pts/yr over T-bills
    2007-26; PPUT costs 2.8-3.5 pts/yr; a 56/44 stock/T-bill mix matched PPUT's 1988-26 CAGR with a
    smaller drawdown (-31.5% vs -38.9% month-end).
  - strangle_mgmt (model, 9,238 daily entries 1990-2026): 50%-or-21-DTE win rate 87%, +0.26
    credits/trade, CVaR5 -3.0 credits, worst -28.8 credits (COVID); hold-to-expiry +0.64
    credits/trade, CVaR5 -4.7. 50% target raises win rate and P&L/day, lowers P&L/trade (mixed);
    21-DTE exit cuts CVaR5 (agrees); rules add no expected edge: book 5.84 vs 5.83%/yr (contradicts);
    2x-credit stop lowers mean and book (agrees with "avoid stops"); 30/45/60 DTE about equal (mixed).
  - earnings_drift (393 Russell 1000 names, 29,122 reports 2001-10 to 2026-09): after outlier moves
    51.0% [49.5, 52.5] keep drifting at +21d (variant A; B and C the same): a coin flip, the user's
    ">50%" hypothesis is not supported (mixed). Large tercile after 2006 ~50-52%. zenith's own PEAD
    tracker (2024-05 to 2026-10): 47.9% at +20d (contradicts).
  - expected_move: SPX finished inside the VIX 1-sigma band 83.9% of 30-day windows (not 68%);
    breaks are rarer on the downside (7.3% vs 8.8%) but twice as far (agrees with crash-risk story).
  - vix_spikes: after VIX first closes > 30 (n=54) the spread is richer (+6.0 vs +4.1 pts) and SPX
    +21d better (+2.2% vs +0.8%), but a >10% drawdown followed 11% of events vs 4% of days (mixed).
- Decisions:
  - (user, Gate A 2026-10-24) the 7 reproductions above; raw cache in data/vault/cache/repro/;
    400-name stratified sample; 2x-credit stop variant included; no skew variant.
  - (this session) claims come from the user's seed hypotheses and literature.json
    agreement_detail; every tasty_value is null with tasty_note "pending Phase 02". Artefacts
    add a claims[] list (one row per tested claim, each with its own verdict) next to the
    top-level claim/verdict; cboe_* add phase03_check, table, episodes, calendar_years, recovery.
  - (this session) verdict rules are coded in each build(); share claims use the 95% CI vs 50%.
    The strangle "rules add edge" rule treats book differences under 0.25 pt/yr as no change
    (set after seeing the 0.01 pt gap; stated in code).
  - (this session) earnings reaction = two-day window around the report (absorbs pre/after-market
    timing errors); size = dollar-volume tercile within year (no market-cap history).
- Deviations from the brief:
  - Ran without research_catalog.json (Phase 02 not run); tests are in tests/test_vault_repro.py,
    not tests/test_vault.py (another session owns that file).
  - zenith's PEAD artefacts only cover 2024-05 onward with cap_tier mostly null, so the
    pre/post-2006 and size split uses yfinance earnings dates (back to ~2001) instead; the PEAD
    tracker is a separate 2024-26 check.
  - strangle_mgmt and cboe_short_vol JSONs are 200-270 KB because chart_data keeps monthly curves.
- Open questions:
  - VAULT_FILES proposal (not written; zenith/config.py is owned elsewhere):
    "repro_dir": VAULT_DIR / "repro"  # Phase 04 reproductions, one <name>.json each
    (zenith/vault/repro/__init__.py uses VAULT_DIR / "repro" directly until then).
  - Phase 02: fill tasty_value / tasty_sources per claims[] row; then re-grade the verdicts that
    are currently against literature-framed claims.
  - The PEAD tracker's 47.9% hit rate at +20d (2024-26) is worth a look in the PEAD tab itself.
  - tests/test_holdings.py::test_snapshot_builds_and_validates fails on data age (dbmf value_date
    78 days old) — unrelated to the vault.
- Re-run commands:
  - all:     .venv/Scripts/python -m zenith.vault.repro.run            (~8 min cold; earnings fetch)
  - one:     .venv/Scripts/python -m zenith.vault.repro.run --only strangle_mgmt
  - refetch: add --refresh ; sample size: --sample 400
  - tests:   .venv/Scripts/python -m pytest tests/test_vault_repro.py -q
- Gate (user, 2026-10-24): Phase 04 Gate B accepted as shown: the claim x ours x verdict table, the
  caveats and the seven charts stand. data/vault/repro/*.json are the repro: sources Phase 05 cites.

---

## 2026-10-24 — Phase 01 harvest (side session started 2026-10-23)

### Done
- Built `zenith/vault/` (`__init__.py` design doc + load/save + DISCLAIMER, `corpus.py` pure helpers,
  `harvest.py` CLI), `VAULT_FILES`/`VAULT_CACHE_DIR` in `zenith/config.py`, `data/vault/cache/` in
  `.gitignore`, `requirements-vault.txt` (yt-dlp), `tests/test_vault.py` (27 offline tests) with
  fixtures in `tests/fixtures/vault/`.
- Discovered the full tastylive episode inventory + the @tastyliveshow YouTube channel; matched,
  fetched YouTube auto captions, fetched and cleaned written pages.
- Artefacts (committable, metadata only): `data/vault/manifest.json`, `coverage.json`,
  `review.json`, `pages.json`. Third-party text only under `data/vault/cache/`
  (`raw/<episode_id>.json` transcripts, `pages/<slug>.json`, `discovery/`).

### Key finding (changes the plan's assumptions)
- tastylive.com is METADATA-ONLY for the archive. `sitemap-episode.xml` lists every episode
  2012-2026 (title, 150-280 char blurb, tags, date) and the show pages page through exactly the
  same set, but episode pages older than ~Jul 2026 301 to `?expiredRedirect=1` and their media
  (gateway.tastylive.com Brightcove mp4s) 404. No per-episode transcript, no audio.
- YouTube @tastyliveshow is the only channel (~4,530 videos) and carries research shows from
  2023 only. Old per-show podcast feeds no longer exist (the network RSS has 47 recent items);
  Wayback copies add only the full description. So Apify/Whisper fallbacks cannot fill the gap:
  there is no media left to transcribe.
- Learn Center concept pages render server-side; support.tastytrade.com (Salesforce) is
  JS-rendered and needs a real browser.

### Counts
- Manifest 5,515 rows: 4,817 tastylive-only, 607 YouTube-only (incl. 267 `research` rows),
  91 matched.
- Transcripts: 695 (all `yt_auto`), median 8,801 chars; 3 failures (`no_captions`); 0 rate limits.

  | show | episodes | tastylive | YouTube | matched | transcripts | % |
  |---|---|---|---|---|---|---|
  | market-measures (2012-26) | 2,649 | 2,488 | 247 | 86 | 246 | 9.3 |
  | from-theory-to-practice | 1,754 | 1,748 | 9 | 3 | 8 | 0.5 |
  | options-jive (YouTube only) | 56 | 0 | 56 | - | 56 | 100 |
  | skinny-math | 316 | 316 | 0 | - | 0 | 0 |
  | skinny-abstract | 367 | 354 | 15 | 2 | 15 | 4.1 |
  | math-check (YouTube only) | 103 | 0 | 103 | - | 103 | 100 |
  | strange-stats (YouTube only) | 1 | 0 | 1 | - | 1 | 100 |
  | research-team-live | 2 | 2 | 0 | - | 0 | 0 |
  | research (show unattributed) | 267 | 0 | 267 | - | 266 | 99.6 |

  Market Measures has zero YouTube coverage 2012-2022 (1,799 episodes).
- Pages: 673 cached (concepts 168, definitions 247, news research subset 258); 168 definition
  pages are duplicates of concept pages (`duplicate_of`). Help center (245) not fetched.
- Transcript QC (all 695): none truncated (last caption within 15% of runtime), 13.6-18.6
  chars/s (p5-p95), adjacent-trigram repeat rate median 0.04% (rolling captions collapsed).
  Auto captions: no punctuation, crosstalk, occasional mishearings ("false talks" ~ "volatile
  stocks"); numbers arrive as digits.

### Decisions (with who made them)
- User (GATE A, 2026-10-24): approved bulk fetch; keep The Math Check and Strange Stats; drop Best
  Practices, Tasty Bites, Skinny on Options Data Science (pages retired, nothing on YouTube);
  fetch captions for unmatched research-playlist videos too (show `research`, attribute by content
  in Phase 02); news = research-style subset only; help center skipped for now.
- User (GATE A, by not choosing (b)): pre-2023 episodes stay metadata-only (no Wayback pass).
- Session: installed yt-dlp 2026.10.21 into .venv (the prompt pack names it as the research dep).
- Session: matching window +/-2 days, min confidence 0.40; generic tastylive titles ("Market
  Measures") match on date only when unambiguous (conf 0.55); research-playlist videos match any
  show only with title similarity >= 0.75.

### Deviations from the brief
- `transcript.source` is only ever `yt_auto` or `none` (no manual captions exist; apify/whisper
  not usable). Manifest rows carry extra fields: `origin`, `yt_title`, `yt_upload_date`, `yt_rule`,
  `captions`, `tl` {slug, tags, in_sitemap}, and `transcript.error` on failures. Episode
  descriptions are NOT in the manifest (third-party text): they live in
  `data/vault/cache/discovery/tastylive.json`.
- A pseudo-show `research` exists for unattributed research videos; `episode_id` is
  `research-YYYY-MM-DD[-n]`.
- Written pages store `text` (boilerplate stripped by cross-page line frequency) and `text_raw`.
- Gate B shows sample transcripts via a local file (`data/vault/cache/gate_b_samples.txt`), not
  pasted into chat, to keep third-party text out of the conversation.

### Open questions
- Phase 02: for 4,800 metadata-only episodes, how much weight should a title + blurb + tags carry?
  (Suggest: topic/timeline evidence only, never numbers.) Many YouTube-only Market Measures rows
  are likely retitled duplicates of tastylive rows; Phase 02 can link them by content.
- Help center (margin, assignment, account types): render ~60 relevant articles with local
  Playwright (install) or Apify (~$1-2) when Phase 05 needs them.

### Re-run commands
```
.venv/Scripts/python -m pip install -r requirements-vault.txt
.venv/Scripts/python -m zenith.vault.harvest discover            # ~1-2 min + ~5 s per new video
.venv/Scripts/python -m zenith.vault.harvest fetch [--show S] [--since YYYY] [--limit N]   # ~12 s/video, resumable
.venv/Scripts/python -m zenith.vault.harvest pages [--kind concepts|definitions|news] [--all-news]
.venv/Scripts/python -m zenith.vault.harvest report
.venv/Scripts/python -m pytest tests/test_vault.py -q
```
`discover --refresh` re-enumerates YouTube lists (cached in `cache/discovery/yt_flat/`).
Existing episode_ids are preserved across re-runs.

---

## 2026-10-25 — Phase 02 (taxonomy + pilot) and Phase 05 rolodex, in-session (session started 2026-10-24)

### Done
- Taxonomy v1 frozen (user, 2026-10-24): `data/vault/taxonomy.json`, 41 strategies, 48 topics,
  6 episode types, from a 60-transcript stratified sample plus a 15-episode 2023/FTTP top-up.
- `zenith/vault/extract.py`: per-episode extraction (cached system prefix, structured outputs,
  Message Batches path) plus an in-session path (`kit`, `pilot-set`, `ingest`). Checks: schema/slugs,
  grounding of every structured number within +/-60 s (study setup fields 300 s before), an
  "anywhere" flag (misplaced ts vs suspected hallucination), 12-word verbatim check. Caption number
  forms handled ("283" = 2.83, "a dollar sixty", "7030" = 70/30).
- Pilot (20 episodes, Sonnet reader subagents, $0): schema 20/20, structured numbers grounded
  125/130 (96%), 0 numbers absent from the transcript, findings-text numbers 115/118, 0 verbatim.
  Hand audit of 10: 0 missed studies, 1 misattributed number (MM 2023-08-04 435/791 is XLK, not SPY),
  2 missed caveats. Outputs: `data/vault/cache/pilot/inline/` (gitignored).
- Phase 05 built on tastylive's written Trader Resources instead of a full transcript run:
  - `zenith/vault/rolodex.py`: popularity, page-note checks, evidence bundles, assembly,
    account rules, ranking, integrity `check`.
  - `zenith/vault/pop.py`: flat-vol POP/tail model per default setup.
  - Artefacts (all `status: draft`): `page_notes.json` (89 Learn Center pages), `popularity.json`,
    `strategies.json` (30 cards), `mechanics.json` (5 decision trees), `edges.json` (15),
    `portfolio.json` (account matrix, allocation, 7-question sleeve playbook, account tiers,
    taxes, 29 unverified items), `glossary.json` (193 terms), `pop.json`, `ranking.json`.

### Counts
- Popularity (5,515 episode titles/descriptions/tags): strangle 614, iron condor 263, 0DTE 239 (all
  2022+), short put 184, straddle 170. 21-DTE topic: 2 mentions 2012-16 vs 45 in 2017-21.
- Cards: tiers A 2 / B 7 / C 21; 21 flagged thin_evidence (most Learn Center pages give rules without
  study numbers). Edges: A 2 (vrp, long-premium-cost), B 5, C 8; agreement with tastylive: agrees 4,
  mixed 10, contradicts 1 (iv-rank-timing).
- Integrity (`python -m zenith.vault.rolodex check`): 30 cards, 5 trees, 1,839 refs, 0 problems;
  15 edges + playbook + glossary, 1,129 refs, 0 problems.
- Tests: tests/test_vault_extract.py 27, tests/test_vault_rolodex.py 15 (all vault tests 92 pass).

### Decisions (with who made them)
- User: taxonomy v1 after a top-up; no API spend (in-session subagent readers instead of Message
  Batches); then pivot from a 592-episode transcript run to Trader Resources + literature + repro;
  card format accepted at the short-strangle example; gate accepted on cards/trees/ranking.
- Session: `anthropic>=1.17,<2` pinned in requirements-vault.txt (installed with user OK; unused now).
- Session: account permissions derived from legs per tastytrade IRA rules (naked puts: margin only;
  uncovered calls: IRA at The Works with $25k; spreads: IRA yes, cash not verified).
- Session: reliability ranking covers only strategies with a claimed positive edge; hedges and long
  premium are listed separately (their tier A evidence is that they cost money).
- Session: dropped the CLL since-2009 worst month (-20.6% vs SPX -12.4%) as a suspected data error.
- Session: thin cuts: pairs, super-bull, short risk reversal, strangle swap; wheel, covered strangle,
  big/reverse jade lizard and ZEBRA folded in as variants.

### Deviations from the brief
- Phase 02 Steps 3-4 (full batch run, research_catalog.json, rules_index.json) not done: replaced by
  the user's Trader Resources pivot. extract.py remains usable for targeted episode reads.
- Phase 05 run before Phase 02 aggregation; episode evidence is limited to the 20 checked pilot
  extracts. Added fields (no renames): cards `conventions`, `popularity`, `variants`,
  `thin_evidence`, `accounts_note`; edges `tasty_position`.
- `zenith/config.py` VAULT_FILES not extended; new artefact paths are defined in the modules.

### Open questions
- VAULT_FILES proposal: taxonomy, page_notes, popularity, strategies, mechanics, edges, portfolio,
  glossary, pop, ranking, repro_dir (all under VAULT_DIR).
- Phase 04 tasty_value fills applied (user, 2026-10-25): all 21 claims[] rows and each file's
  top-level fields in data/vault/repro/*.json; 18 carry a tastylive position, 3 have none (drift and
  PEAD claims). Only the tasty_* fields changed; 38 refs resolve. The fills live in
  data/vault/repro_tasty_fills.json (matched by claim text) and zenith.vault.repro.save re-applies
  them on every build, so `python -m zenith.vault.repro.run` keeps them; a claim whose wording
  changes falls back to "pending Phase 02". Edit the fills file, not the outputs. Repro tests 25 pass
  (2 new: re-apply on save; committed fills match current claims).
- Needs a source: the strangle Reg-T total ("larger leg + other premium"); cash-account spreads;
  the VIX-tiered BP ladder (still unverified). The CLL series needs a data check (vault-repro).
- Strategy finding worth carrying to Phase 06: in the strangle model the 2x-credit stop cut the
  worst March 2020 trade from -28.8 to -4.7 credits while slightly lowering the mean.

### Re-run commands
```
.venv/Scripts/python -m zenith.vault.rolodex popularity
.venv/Scripts/python -m zenith.vault.rolodex notes <notes_dir>          # page notes -> page_notes.json
.venv/Scripts/python -m zenith.vault.rolodex bundle --out <dir>          # per-card evidence bundles
.venv/Scripts/python -m zenith.vault.pop                                 # pop.json
.venv/Scripts/python -m zenith.vault.rolodex assemble <card_dir>         # strategies.json + ranking.json
.venv/Scripts/python -m zenith.vault.rolodex check
.venv/Scripts/python -m zenith.vault.extract ingest <reader_dir> [--dest pilot|run]
.venv/Scripts/python -m pytest tests/test_vault_rolodex.py tests/test_vault_extract.py -q
```

---

## 2026-10-25 — Phase 06 build: the OPTIONS VAULT tab (side session started from the planning session)

### Done
- `zenith/config.py`: VAULT_FILES extended per the Phase 02/05 proposal (taxonomy, page_notes, popularity,
  literature, strategies, mechanics, edges, portfolio, glossary, pop, ranking, repro_dir) plus `build`.
  `zenith/vault/repro/__init__.py` now takes REPRO_DIR from `VAULT_FILES["repro_dir"]`; rolodex.py/pop.py
  keep their module-local paths (not a one-line swap).
- New modules (pure unless noted): `payoff.py` (legs -> strikes on spot 100, Black-Scholes T+0 / T+21 /
  expiry curves, breakevens, max P/L; reuses pop.py's bs/strike_for_delta and so hedge._ncdf; no scipy),
  `sizing.py` (calculator), `build.py` (validate + pack), `help_text.py` (glossary-fed tooltips),
  `view.py` (Streamlit).
- `python -m zenith.vault.build` writes `data/vault/build.json`: counts, status, and a 293-entry link
  table for every ep:/page:/lit:/repro: reference used by cards, edges, trees, portfolio, glossary, POP
  and repro claims, so the view never loads the 5.8 MB manifest or the cache. It also runs the rolodex
  integrity check when the local page cache exists.
- View: seven sections behind one radio (`vault_sub`): START HERE, STRATEGY ROLODEX, EDGES & RESEARCH,
  MECHANICS, PORTFOLIO & SIZING, GLOSSARY, SOURCES & METHOD. Rolodex, literature search, calculator and
  glossary run in @st.fragment. Every card shows DRAFT and, where set, its thin_evidence reason; the
  strangle card carries the 2x-stop callout computed from repro:strangle_mgmt (worst March 2020 trade
  -28.8 -> -4.7 credits, mean +0.26 -> +0.21). Inline [kind:key@ts] markers render as small links
  (YouTube at the timestamp when the episode has a video). Decision trees use st.graphviz_chart with
  DOT strings: renders with no new dependency (confirmed in the preview).
- Registration: app.py tab "OPTIONS VAULT" (before ARCHIVE), orientation entry 16 (Today/Archive now 17),
  `_BADGE_SOURCES` chip "OPTIONS VAULT · 30 strategies · 15 edges", tests in tests/test_views.py.
- `scripts/screen.py`: new [vault] section (build problems, counts, slugs, tiers, None/NaN values,
  payoff sanity for every setup, POP bounds, calculator default, every repro chart builds, 2x-stop finding).

### Counts
- 30 cards (A 2 / B 7 / C 21, 21 thin), 15 edges, 5 trees, 193 terms, 209 papers (268 topic rows),
  7 repro files, 293 resolved refs, 0 build problems, rolodex check 0 problems.
- Tests: tests/test_vault.py 45 (18 new: payoff vs hand-computed values, open-ended legs, calendar,
  calculator edge cases, build integrity on synthetic data, timeline, DOT), tests/test_views.py +11
  vault tests (no-data render, 7 sections, draft/thin/strangle-stop, cash filter narrows, badge).
  Full suite: 924 passed, 1 failed (the known test_holdings data-age failure).
- zenith-screen: [vault] all ok, 1 advisory warn (a related slug with no card, shown without a button).
  The 8 ERRs elsewhere (pretom basket, eap, edge, nightday, mom, etfmom, index) are in other tabs'
  committed data and were not touched here.

### Decisions (with who made them)
- Session: sections via st.radio (the repo's pre-seedable pattern) rather than nested tabs.
- Session: the BP-vs-VIX chart draws only verified guidance (25-50% band, 75% cap, ~30% with VIX > 15)
  because portfolio.json lists the VIX ladder as unverified. The calculator follows the same rule;
  the theta figure is shown as a single-episode yardstick, not a range.
- Session: payoff diagrams for cards without a pop.py setup use EXTRA_SETUPS whose strikes are ours;
  the caption says so. futures-options has no diagram.
- Session: calculator VIX default = CLEAN BETA hedge monitor's latest VIX (data/beta/hedge.json),
  else the last repro:vrp_spx year.

### Deviations from the brief
- No research_catalog.json: EDGES & RESEARCH shows edges, the repro claims table/charts and a searchable
  literature table, with a note that the per-episode study catalog was not built.
- "Study distribution charts" come from the repro builders (strangle win rate / P&L per day / CVaR by
  rule) rather than tastylive study rows, which do not exist as a catalog.
- The three fixed-width repro charts (strangle_mgmt, cboe_tail_hedge, vix_spikes) render at their
  own widths; the first strangle panel's row labels can be clipped (Phase 04 builder, left as is).

### Open questions
- Cards stay `status: draft`; a line-by-line review would flip them and drop the banner.
- Console: Vega logged "translate(1.797e308)" errors during viewport resizing; none when switching
  vault sections.

### Re-run commands
```
.venv/Scripts/python -m zenith.vault.build                 # after any edit to data/vault/*.json
.venv/Scripts/python -m pytest tests/test_vault.py tests/test_views.py -q -k "vault or full_app"
.venv/Scripts/python scripts/screen.py                     # [vault] section at the end
```

### Files for a commit of OPTIONS VAULT
- Include: `zenith/vault/` (all .py, no __pycache__), `zenith/config.py`, `.gitignore`, `app.py`,
  `scripts/screen.py`, `requirements-vault.txt`, `prompts/vault/`, `tests/test_vault.py`,
  `tests/test_vault_extract.py`, `tests/test_vault_repro.py`, `tests/test_vault_rolodex.py`,
  `tests/fixtures/vault/`, `tests/test_views.py`, and `data/vault/` except `cache/`:
  build.json, coverage.json, edges.json, glossary.json, literature.json, manifest.json (5.6 MB),
  mechanics.json, page_notes.json, pages.json, pop.json, popularity.json, portfolio.json,
  ranking.json, repro_tasty_fills.json, review.json, strategies.json, taxonomy.json, repro/*.json.
- Keep out: `data/vault/cache/` (121 MB of transcripts, pages, discovery, pilot extracts; already
  gitignored), every `__pycache__/`, and the unrelated pretom edits (`tests/test_pretom.py`,
  `zenith/pretom/universe.py`).
