# Phase 06 — Build the OPTIONS VAULT tab
Read 00_master_brief.md and HANDOFF.md first. Then read:
- app.py, zenith/ui_theme.py, zenith/ui_charts.py;
- zenith/trend/view.py (fragments) and zenith/pead/view.py;
- PLAN.md "What exploration found" (commit 6907daf is the template for adding a tab);
- tests/test_views.py.

Start in plan mode with a short plan for the user.

Data:
- The view reads only committed JSON in data/vault/ through VAULT_FILES and @st.cache_data.
- No network calls, no research dependencies.
- python -m zenith.vault.build validates and packs the phase outputs into what the view reads.

Registration (do all four):
- the app.py st.tabs list, with label "OPTIONS VAULT";
- a numbered entry in the "What is Zenith?" expander (renumber Today/Archive);
- _BADGE_SOURCES: today_badge() returns a chip such as "OPTIONS VAULT · N strategies ·
  M edges" and must never raise;
- tests/test_views.py.

Sections (segmented control or nested tabs; heavy parts in @st.fragment):
1. START HERE — one screen:
   - where the edge comes from (the VRP);
   - why probabilities and mechanics matter;
   - trade small, trade often;
   - a key_findings strip;
   - a legend for A/B/C and EDGE/CONVENTION/MECHANIC.
2. STRATEGY ROLODEX:
   - filters: outlook, IV environment, defined/undefined risk, account type, complexity;
   - a compact card grid → the selected card;
   - card front: four boxes (When · Setup · Manage · Exit) with the numbers, the payoff
     diagram, Greek glyphs and the evidence badge;
   - card back (expander): sourced stats with episode links, rule history, mistakes, the
     worked example, related cards.
3. EDGES & RESEARCH:
   - edge cards by tier, with "act with" links to strategy cards;
   - reproduction charts (tasty vs ours);
   - a searchable research-catalog table of every extracted study, linking to episodes with
     timestamps.
4. MECHANICS:
   - open / manage / roll / close / take-loss decision trees, using st.graphviz_chart from
     DOT strings (confirm it renders with no new dependencies);
   - a DTE timeline graphic: entry ~45 DTE → 50% target → 21-DTE exit.
5. PORTFOLIO & SIZING:
   - the account-type matrix;
   - the BP-vs-VIX chart;
   - the sleeve playbook;
   - account-size tiers;
   - a sizing calculator (in a fragment):
     - inputs: NLV, sleeve %, account type, VIX (default to the latest value zenith already
       stores, if any);
     - outputs: BP-in-use range, max BPR per trade, suggested position count, theta range,
       warnings.
6. GLOSSARY — searchable; the same definitions feed help_badge tooltips (pattern:
   zenith/cas/help_text.py).
7. SOURCES & METHOD:
   - corpus coverage by show and year;
   - the extraction method and the evidence rubric;
   - limitations and the copyright note;
   - links out to OptionStrat and Market Chameleon;
   - DISCLAIMER.

Visuals (Altair, house rules, THEME colours):
- payoff at expiry plus T+0 and T+21 curves, computed from card legs with Black-Scholes
  (reuse hedge._ncdf) on a normalised underlying of 100;
- study distribution charts (win rate vs mean P&L, P&L/day by management rule);
- an evidence ladder;
- the BP-vs-VIX step chart;
- the earnings-drift chart from repro.

Tests:
- tests/test_vault.py: payoff maths vs hand-computed values, card schema validation, build
  integrity, calculator edge cases;
- test_views.py: renders with and without data, disclaimer shown;
- the full suite passes.

Verify:
- preview_start on the .claude/launch.json server (port 8601);
- walk every section and filter;
- check console errors and narrow-width layout;
- take screenshots;
- run the zenith-screen skill (NaN/None leaks, mislabeled charts, empty sections).

GATE — show:
- screenshots of every section;
- the test output;
- the zenith-screen report.

Then update HANDOFF.md. Don't commit unless asked.
