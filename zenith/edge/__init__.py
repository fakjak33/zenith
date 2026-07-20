"""EDGE SCREENS — four cross-sectional signal screeners for Zenith.

One nightly compute over the Russell 1000, four ranked long/short screens, each
rated on the strength of its empirical support (evidence tiers shown in-app):

  * IV spread (C+) — Cremers & Weinbaum (JFQA 2010): stocks with relatively
    expensive calls (ATM call IV > put IV) beat expensive-put stocks by ~51bp/
    week. BUT Muravyev, Pearson & Pollet (JFE 2025) show ~2/3 of this is a
    borrow-fee proxy concentrated in hard-to-borrow names — "a mirage not
    readily exploitable." So the NEGATIVE spread doubles as a hard-to-borrow /
    short flag more than a clean long signal.

  * Analyst revisions (B) — post-forecast-revision drift: prices underreact to
    analyst EPS revisions (Chen, Narayanamoorthy, Sougiannis & Zhou, JBFA 2020;
    sell-side recommendation drift −9.1%/6mo, Bradshaw). Attenuated post-
    publication (~50%, McLean-Pontiff) but alive.

  * Short interest / days-to-cover (B gross, C net) — Rapach, Ringgenberg &
    Zhou (JFE 2016): aggregate short interest is a top predictor (OOS R²≈13%);
    Hong et al. days-to-cover ≈1.2%/mo cross-sectionally. Muravyev (JF 2025):
    net of borrow fees ≈ 0 — so the short side is framed as an avoid-list with
    a squeeze-risk flag, not a tradable alpha.

  * Lottery / MAX (B+) — Bali, Cakici & Whitelaw (JFE 2011): high max-daily-
    return stocks underperform >1%/mo. Bali, Ince & Ozsoylev (2026): raw MAX is
    subsumed by mispricing factors, but the BETA-PURGED MAXβ survives modern
    models (−0.81%/mo, t=−3.62, incl. large caps). MAXβ is the headline here.

Free / best-effort data only (yfinance option chains, .info short-interest and
estimate fields, Vanguard VONE holdings), computed offline in a GitHub Action
that commits JSON under data/edge/; the app is a thin reader. Decision-support,
not investment advice.
"""

from __future__ import annotations

import json

from ..config import EDGE_FILES

SCREENS = ("ivspread", "revisions", "shortint", "lottery")

DISCLAIMER = ("EDGE screens are auto-generated from free/best-effort data (yfinance option "
              "chains + .info short-interest/estimate fields, Vanguard VONE holdings). Each "
              "screen shows its evidence tier: several are attenuated post-publication or "
              "(IV spread, short interest) partly a borrow-fee proxy that is hard to trade net "
              "of costs (Muravyev et al.). Ranked ideas, not investment advice.")


def _read(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load(name: str, default=None):
    return _read(EDGE_FILES[name], default if default is not None else {})


def save(name: str, obj) -> None:
    _write(EDGE_FILES[name], obj)
