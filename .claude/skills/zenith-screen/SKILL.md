---
name: zenith-screen
description: Screen the Zenith data artefacts and UI for accuracy/consistency bugs after edits — stale dates, mislabeled or mis-sorted charts, empty sections, NaN/None leaks, duplicate tickers, absurd values (e.g. a wrong market cap), and broken price overlays. Use after changing Zenith's brief/CAS code or before merging, to catch the kind of minor errors that are easy to miss by eye.
---

# Zenith accuracy screen

Run the automated sanity checks and fix anything that fails, so "minor errors"
(like a sector chart implying Utilities lagged when Tech actually did) don't ship.

## Steps

1. **Run the screen** with the repo venv (no network; reads committed artefacts):
   ```
   .venv/Scripts/python.exe scripts/screen.py
   ```
   It prints `ok` / `ERR` / `warn` per check and exits non-zero on any hard failure.

2. **If data looks stale or thin**, regenerate first, then re-screen:
   ```
   .venv/Scripts/python.exe -m zenith.cas.compute --cadence weekly
   .venv/Scripts/python.exe -m zenith.brief.compute
   ```

3. **For each `ERR`**, fix the root cause in code (not the data):
   - *sector/movers sort vs data* → the data is usually right; the bug is the
     CHART (missing/!truncated axis labels). Force all axis labels visible
     (`labelOverlap=False`, height = n×rowheight) in `zenith/brief/view.py` /
     `zenith/cas/view.py`.
   - *empty section / NaN leak* → guard the source function in
     `zenith/brief/sources.py` (degrade gracefully) and the view.
   - *duplicate tickers* → de-dupe in `zenith/cas/universe.py` (`setdefault`).
   - *price panel empty* → ensure `analytics/history.build_price_panel` runs in
     `compute` and writes `data/cas/price_panel.json` (must NOT be gitignored).
   - *label doesn't resolve* → extend `universe.label_of`.

4. **Eyeball the printed leader/laggard** lines against reality (the screen can't
   know ground truth) — if e.g. the sector leader looks wrong, investigate the
   data source freshness and the chart's sort/encoding.

5. **`warn` lines are advisory** (e.g. a feed-quirk market cap, no news this run) —
   note them but they don't block.

6. Re-run the screen until it prints `SCREEN PASSED`. Then run `pytest` and, if you
   changed view code, do a data-transform isolation check before merging.

## What it checks
Brief: fresh `as_of`; all 4 overview groups; 11 sectors + sort consistency; movers
have company names; absurd market caps; null prices. CAS: fresh status; signals
populated; **dynamic confidence reaches 'high'**; factor-rotation populated; no
duplicate FRM tickers; required groups (style/industry/beta); **committed price
panel non-empty**; rotation + multi-model hit-rate artefacts present; label
resolution.
