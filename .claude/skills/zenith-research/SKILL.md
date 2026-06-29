---
name: zenith-research
description: Ingest a research paper or article (URL or PDF) and incorporate it into Zenith's CAS monitor — summarize it, classify which model family it informs, log a registry note, and draft a signal-module scaffold + integration plan for review. Use when the user shares new research/insight they want reflected in Zenith or CAS mode.
---

# Zenith research intake

Turn a piece of research (a URL, a PDF path, or pasted text) into a reviewed
integration in the Zenith CAS monitor. Never auto-merge generated code — always
leave a human review step.

## Inputs
The user gives one of: a URL, a local PDF/file path, or pasted abstract/notes.
If none is provided, ask which.

## Steps

1. **Extract the text.**
   - URL → `WebFetch` (or the bright-data/scrape tools if blocked).
   - PDF → `Read` with `pages`, or the `pdf` skill. If the PDF has embedded CID
     fonts (binary on extraction), fall back to fetching a public HTML/abstract.
   - Pasted text → use as-is.

2. **Summarize** into: thesis, the factor(s)/signal(s) it proposes, the markets &
   horizon, the methodology (look-back, weighting, rebalance), and the headline
   result (Sharpe, hit-rate, t-stats). Keep it tight and faithful — quote numbers.

3. **Classify which CAS family it informs.** Map the idea to an existing registry
   family key, or propose a new one. Current families live in
   `zenith/cas/registry.py` (`DEFAULT_WEIGHTS`) and the segments in
   `zenith/cas/schema.py` (`SEGMENTS`). Rough mapping:
   - trend / time-series or cross-sectional factor momentum → `frm_ts_mom`,
     `frm_cs_region`, `frm_cs_peer`, `frm_composite` (segment `factor_rotation`)
   - per-asset technical/statistical strategies → `strategies` segment families
   - sector/industry/theme rotation → `themes` / `relative_strength`
   - positioning / flows → `flows` families
   - macro / regime → `risk_regime`

4. **Log a registry note** (so it shows in the in-app "Models & notes" tab and can
   nudge a weight). Run with the repo venv:
   ```
   .venv/Scripts/python.exe scripts/add_research_note.py \
     --family <family_key> --title "<paper title>" \
     --source "<url-or-filename>" --abstract "<your summary>" \
     [--weight <0..2>] --status processed
   ```
   If the user had already flagged a `pending-review` note in-app, update it instead.

5. **Draft a signal-module scaffold + integration plan (for review).** If the
   research warrants a new signal, scaffold it following the existing pattern —
   a `compute(prices|data) -> list[dict]` that emits `schema.make_signal` records
   under the right segment, reusing `zenith/cas/signals/indicators.py`. Good
   templates: `zenith/cas/signals/factor_rotation.py` and `strategies151.py`.
   - Put new code in `zenith/cas/signals/<name>.py`.
   - Wire it the same way as existing models: add to `schema.SEGMENTS` if a new
     segment, call it in `zenith/cas/compute.py`, add weights in `registry.py`,
     classify families in `consensus.py` (`PHYSICAL`/`BEHAVIORAL`), and surface in
     `zenith/cas/view.py`.
   - Add tooltips to `zenith/cas/help_text.py` and unit tests to
     `tests/test_cas.py` (no network — pass synthetic prices).

6. **Verify, then hand off for review.** Run `.venv/Scripts/python.exe -m pytest`.
   Summarize what you changed and what the user should review. Do **not** commit or
   push unless the user asks — present the diff and the plan first.

## Notes
- Keep everything free-data / yfinance-compatible so the GitHub Action keeps running.
- Be honest about data limits (paywalled feeds, short ETF history) — mirror the
  existing CAS `DISCLAIMER` tone. Decision-support, not investment advice.
