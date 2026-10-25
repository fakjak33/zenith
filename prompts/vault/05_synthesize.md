# Phase 05 — Synthesis: cards, edges, mechanics, portfolio playbook
Read 00_master_brief.md and HANDOFF.md first.

Inputs:
- taxonomy.json, extract/*.json, research_catalog.json, rules_index.json;
- literature.json;
- repro/*.json;
- cache/pages (Learn Center and help-center text).

At the start, propose how to run synthesis and get the user's choice:
- zenith/vault/synthesize.py: Claude API with the strongest model, one request per card, with
  the aggregated evidence for that slug as input (load the claude-api skill first); or
- in-session work.

Either way, review every card yourself before accepting it.

1. Strategy cards → strategies.json (schema in brief). One per strategy with enough evidence.
   Each card has:
   - the current tastylive default setup and management, with numbers, plus how it changed
     over time;
   - when to use and when to avoid (IV environment, outlook, events, liquidity);
   - defined/undefined risk and which account types allow it;
   - the Reg-T BPR formula (note PM);
   - sizing;
   - key stats with sources;
   - common mistakes;
   - a worked example with round generic numbers;
   - the EDGE/CONVENTION split;
   - the tier and a one-line justification combining tastylive, literature and repro.

   Families to cover:
   - neutral short premium;
   - directional (short puts, verticals, lizards, ratios, BWBs, risk reversals);
   - stock-based (covered call/strangle, wheel, collar, PMCC);
   - time spreads;
   - long volatility (when tastylive says it makes sense);
   - earnings;
   - 0DTE;
   - hedges;
   - futures options (secondary).
2. Edge cards → edges.json. The distinct findings worth acting on, for example:
   - earnings outlier drift;
   - IV-rank conditioning;
   - the VRP;
   - managing winners;
   - strangle vs straddle;
   - 0DTE results;
   - VIX-spike entries;
   - defined vs undefined risk.
3. Mechanics → mechanics.json, as decision trees (nodes and edges) the UI can draw:
   - opening: liquidity filters, order at mid, legging;
   - managing: profit target, 21 DTE, rolling for duration or credit, defending the tested
     side, inverting, when NOT to roll;
   - closing: buying back cheap shorts, pin and assignment risk, expiration week, early
     assignment and dividends;
   - taking losses: multiples of credit, max-loss rules.
4. Portfolio playbook → portfolio.json:
   - Account-type matrix: cash, IRA (incl. limited margin), Reg-T, portfolio margin, futures.
     For each: strategies allowed, BPR basis and current minimums. Verify against the
     tastytrade help center and date them.
   - tastylive's BP allocation vs VIX: the exact table and its history. Also per-trade size
     (% NLV and % BP), number of positions, theta/NLV target, beta-weighted delta,
     correlation and diversification.
   - The "options sleeve next to a long stock/ETF core" playbook, answering the user
     directly:
     - does a 25–50% sleeve need margin, by strategy?
     - how should it be sized?
     - how do short puts or covered calls change total portfolio delta/beta and
       concentration?
     - what should be avoided or capped?
     - how can the sleeve hedge the core?
     - what happens in a crash?
   - Account-size tiers (<$25k, $25k–$100k, $100k–$250k, >$250k): what is practical at each
     level, PDT and approval constraints, and contract choice (SPX/XSP/SPY, /ES vs /MES).
   - Taxes at a high level: Section 1256, wash sales with options, IRA considerations. Dated,
     educational, not tax advice.
5. Glossary → glossary.json (feeds help_badge tooltips).
6. Integrity script (python -m zenith.vault.build --check):
   - every source ref (ep:/lit:/repro:/page:) resolves to a manifest episode, a literature
     id, a repro file or a cached page;
   - every referenced slug exists;
   - every card has a tier;
   - the quote rule holds;
   - no NaN or empty required fields.

GATE — show:
- 3 finished cards end to end: short strangle, the earnings-outlier-drift edge, and the
  sleeve playbook;
- counts per family;
- the integrity report;
- thin-evidence cards to cut or merge.

Then update HANDOFF.md.
