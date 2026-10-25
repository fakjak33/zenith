"""OPTIONS VAULT tests — offline, synthetic data and fixtures only (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

import zenith.vault as vault
from zenith.vault import corpus as c

FIX = Path(__file__).parent / "fixtures" / "vault"


# --- VTT ---------------------------------------------------------------------

def test_parse_vtt_collapses_rolling_auto_captions():
    lines = c.parse_vtt((FIX / "rolling.vtt").read_text(encoding="utf-8"))
    assert [ln["text"] for ln in lines] == [
        "welcome back to the study",
        "we sold sixteen delta strangles",
        "at forty five days & managed early",
        "next slide please",
    ]
    # each line keeps the start of the cue that introduced it (not the bridge cue)
    assert [ln["t"] for ln in lines] == [3.0, 5.5, 8.0, 30.0]


def test_parse_vtt_manual_captions_and_entities():
    lines = c.parse_vtt((FIX / "manual.vtt").read_text(encoding="utf-8"))
    assert [ln["text"] for ln in lines] == ["Today we look at iron condors.",
                                            "The win rate went up.",
                                            "The average P&L went down."]
    assert lines[1]["t"] == pytest.approx(65.2, abs=0.1)


def test_parse_vtt_handles_crlf_and_empty():
    crlf = (FIX / "manual.vtt").read_text(encoding="utf-8").replace("\n", "\r\n")
    assert len(c.parse_vtt(crlf)) == 3
    assert c.parse_vtt("") == []
    assert c.parse_vtt("WEBVTT\n\nnot a cue") == []


def test_segments_merge_by_time_and_length():
    lines = [{"t": 0.0, "text": "a"}, {"t": 5.0, "text": "b"}, {"t": 25.0, "text": "c"},
             {"t": 26.0, "text": "x" * 400}]
    segs = c.segments(lines, max_secs=20, max_chars=320)
    assert segs == [{"t": 0, "text": "a b"}, {"t": 25, "text": "c"}, {"t": 26, "text": "x" * 400}]


def test_fmt_ts():
    assert c.fmt_ts(0) == "00:00"
    assert c.fmt_ts(65.9) == "01:05"
    assert c.fmt_ts(3725) == "62:05"


# --- tastylive URLs and ids --------------------------------------------------

def test_parse_episode_url():
    p = c.parse_episode_url("https://www.tastylive.com/shows/market-measures/episodes/"
                            "spx-management-08-21-2014")
    assert p == {"tl_show": "market-measures", "slug": "spx-management", "air_date": "2014-08-21"}
    dup = c.parse_episode_url("/shows/tasty-extras/episodes/monologues-12-13-2012-1")
    assert dup["air_date"] == "2012-12-13" and dup["slug"] == "monologues"
    assert c.parse_episode_url("https://www.tastylive.com/shows/market-measures") is None
    assert c.parse_episode_url("/shows/x/episodes/bad-13-45-2020") is None


def test_assign_ids_suffixes_and_stability():
    rows = [{"show": "market-measures", "air_date": "2016-03-14", "url": "u/b", "youtube_id": ""},
            {"show": "market-measures", "air_date": "2016-03-14", "url": "u/a", "youtube_id": ""},
            {"show": "options-jive", "air_date": "2016-03-14", "url": "", "youtube_id": "Y1"}]
    c.assign_ids(rows)
    assert [r["episode_id"] for r in rows] == ["market-measures-2016-03-14-2",
                                               "market-measures-2016-03-14",
                                               "options-jive-2016-03-14"]
    # a new same-day episode sorting first must not steal an existing id
    existing = {c.row_key(r): r["episode_id"] for r in rows}
    new = [{"show": "market-measures", "air_date": "2016-03-14", "url": "u/0", "youtube_id": ""}]
    rows2 = [{k: v for k, v in r.items() if k != "episode_id"} for r in rows] + new
    c.assign_ids(rows2, existing)
    assert rows2[1]["episode_id"] == "market-measures-2016-03-14"
    assert rows2[3]["episode_id"] == "market-measures-2016-03-14-3"


def test_every_tl_slug_maps_to_one_show():
    assert c.TL_SLUG_TO_SHOW["market-measures"] == "market-measures"
    assert c.TL_SLUG_TO_SHOW["Skinny-on-Options-Math"] == "skinny-math"
    assert len({s.slug for s in c.SHOWS}) == len(c.SHOWS)


# --- classifier --------------------------------------------------------------

@pytest.mark.parametrize("title,playlists,desc,expect", [
    ("Skewed Iron Condor Breakdown | Market Measures", [], "", ("market-measures", "suffix")),
    ("Iron Condors: My Backtested Strategy | Options Jive", [], "", ("options-jive", "suffix")),
    ("Why Low-Delta Trades Are More Dangerous", ["PLPVve34yolHauPYzRmpV0XV8D9AZaimu6"], "",
     ("market-measures", "playlist")),
    ("Why We Manage at 21 DTE | From Theory to Practice Options", [], "",
     ("from-theory-to-practice", "suffix")),
    ("Market Measures: Strangles in Low IV", [], "", ("market-measures", "title")),
    ("Some Clip", [], "On Market Measures, Tom and Tony test ...", ("market-measures", "description")),
    ("We Studied 15 Years of 21 DTE", ["PLPVve34yolHZ1lcMnUb8LKOm3NyrvkYZr"], "",
     ("research", "research-playlist")),
    ("AI SELLOFF | Thursday Oct 22 Live Trading", [], "", (None, "unclassified")),
])
def test_classify_video(title, playlists, desc, expect):
    assert c.classify_video(title, playlists, desc) == expect


def test_suffix_beats_playlist():
    # a Market Measures clip that also sits in the Options Jive playlist
    assert c.classify_video("X | Market Measures", ["PLPVve34yolHb_AA-bnhxCVsxOW9N_WkHS"]) == \
        ("market-measures", "suffix")


# --- matching ----------------------------------------------------------------

def _ep(title, d, show="market-measures", slug=""):
    return {"show": show, "air_date": d, "title": title, "slug": slug or title.lower().replace(" ", "-")}


def _vid(title, d, show="market-measures"):
    return {"show": show, "upload_date": d, "title": title}


def test_title_similarity():
    assert c.title_similarity("Skewed Iron Condors", "Skewed Iron Condor Breakdown | Market Measures") > 0.6
    assert c.title_similarity("Calendar Spreads", "Tesla Earnings Preview") < 0.35
    assert c.title_similarity("", "x") == 0.0


def test_match_prefers_title_within_window():
    eps = [_ep("Skewed Iron Condors", "2023-04-13"), _ep("Rolling Trades", "2023-04-13")]
    vids = [_vid("We Studied Rolling Trades For 17 Years | Market Measures", "2023-04-14"),
            _vid("Skewed Iron Condor Breakdown | Market Measures", "2023-04-14")]
    m = c.match_episodes(eps, vids)
    assert m[0][0] == 1 and m[1][0] == 0
    assert all(conf >= c.MATCH_MIN_CONFIDENCE for _, conf in m.values())


def test_match_rejects_outside_window_and_other_show():
    eps = [_ep("Skewed Iron Condors", "2023-04-13")]
    assert c.match_episodes(eps, [_vid("Skewed Iron Condors", "2023-04-20")]) == {}
    assert c.match_episodes(eps, [_vid("Skewed Iron Condors", "2023-04-13", show="options-jive")]) == {}


def test_generic_title_matches_only_when_unambiguous():
    ep = _ep("Market Measures", "2026-06-15", slug="market-measures")
    one = c.match_episodes([ep], [_vid("Should You Rebalance Before the Weekend?", "2026-06-15")])
    assert one == {0: (0, 0.55)}
    two = c.match_episodes([ep], [_vid("Clip A", "2026-06-15"), _vid("Clip B", "2026-06-16")])
    assert two == {}


def test_research_bucket_matches_any_show_on_strong_title_only():
    eps = [_ep("Skewed Iron Condors", "2023-04-13"), _ep("Market Measures", "2023-04-13")]
    strong = c.match_episodes(eps, [_vid("Skewed Iron Condor Breakdown", "2023-04-14", show="research")])
    assert list(strong) == [0]
    weak = c.match_episodes(eps, [_vid("Iron Condor Ideas", "2023-04-13", show="research")])
    assert weak == {}


def test_match_is_one_to_one():
    eps = [_ep("Iron Condor Width", "2020-01-02"), _ep("Iron Condor Width Part 2", "2020-01-03")]
    vids = [_vid("Iron Condor Width | Market Measures", "2020-01-02")]
    m = c.match_episodes(eps, vids)
    assert len(m) == 1 and list(m.values())[0][0] == 0


# --- package helpers ---------------------------------------------------------

def test_load_save_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setitem(vault.VAULT_FILES, "manifest", tmp_path / "manifest.json")
    assert vault.load("manifest") == {}
    vault.save("manifest", {"episodes": [{"episode_id": "x"}]})
    assert vault.load("manifest")["episodes"][0]["episode_id"] == "x"
    assert "not investment advice" in vault.DISCLAIMER


def test_clean_html_and_split_title():
    assert c.clean_html("<p>Tom &amp; Tony�s study</p>") == "Tom & Tony-s study"
    assert c.split_title("A | B | Market Measures") == ("A | B", "market measures")
    assert c.split_title("No suffix") == ("No suffix", "")


def test_wanted_page_filters_news_and_help():
    from zenith.vault.harvest import wanted_page
    assert wanted_page({"kind": "concepts", "slug": "concepts-strangle"})
    assert wanted_page({"kind": "news", "slug": "news-why-manage-at-50-not-25-for-short-strangles"})
    assert not wanted_page({"kind": "news", "slug": "news-usd-jpy-hits-two-decade-highs"})
    assert wanted_page({"kind": "news", "slug": "news-usd-jpy-hits-two-decade-highs"}, all_news=True)
    assert not wanted_page({"kind": "help", "slug": "help-43000435310"}, all_news=True)


def test_strip_boilerplate_keeps_article_lines():
    nav, foot = "Beginner Options Strategies\nBear Call Spread", "All rights reserved"
    texts = {f"p{i}": f"{nav}\nArticle {i} line one\nArticle {i} body\n{foot}" for i in range(6)}
    out = c.strip_boilerplate(texts)
    assert out["p3"] == "Article 3 line one\nArticle 3 body"
    assert c.strip_boilerplate({"a": nav}) == {"a": nav}   # too few docs: untouched


# --- Phase 06: payoff maths ----------------------------------------------------
from zenith.vault import build as vb
from zenith.vault import payoff as po
from zenith.vault import sizing as sz


def test_atm_call_price_matches_rule_of_thumb():
    legs = po.price_legs([{"type": "call", "qty": 1, "atm": True, "dte": 45}])
    # 0.4 * S * sigma * sqrt(T) = 2.809 for S=100, 20% IV, 45 days; exact BS is 2.801
    assert legs[0]["price"] == pytest.approx(2.801, abs=0.002)


def test_short_put_payoff_at_expiry_hand_computed():
    legs = po.price_legs([{"type": "put", "qty": -1, "K": 90, "dte": 45}])
    p = legs[0]["price"]
    rows = {(r["S"], r["horizon"]): r["pnl"] for r in po.curves(legs, lo=70, hi=130, n=61)}
    assert rows[(100.0, "expiry")] == pytest.approx(p, abs=1e-4)          # expires worthless: keep credit
    assert rows[(80.0, "expiry")] == pytest.approx(p - 10, abs=1e-4)      # 10 points ITM
    assert rows[(100.0, "T+0")] == pytest.approx(0, abs=1e-9)             # zero at entry
    s = po.summary(legs)
    assert s["net"] == "credit" and s["breakevens"] == [pytest.approx(90 - p, abs=0.05)]
    assert s["max_profit"] == pytest.approx(p, abs=1e-3)
    assert s["max_loss"] == pytest.approx(p - 90 + 0.01, abs=0.01)        # bounded at S = 0


def test_iron_condor_max_loss_is_width_minus_credit():
    legs = po.price_legs([{"type": "put", "qty": 1, "K": 85, "dte": 45},
                          {"type": "put", "qty": -1, "K": 90, "dte": 45},
                          {"type": "call", "qty": -1, "K": 110, "dte": 45},
                          {"type": "call", "qty": 1, "K": 115, "dte": 45}])
    credit = -po.entry_cost(legs)
    s = po.summary(legs)
    assert s["max_profit"] == pytest.approx(credit, abs=1e-3)
    assert s["max_loss"] == pytest.approx(credit - 5, abs=1e-3)
    assert len(s["breakevens"]) == 2


def test_short_call_and_long_call_are_open_ended():
    assert po.summary(po.price_legs([{"type": "call", "qty": -1, "K": 105, "dte": 45}]))["max_loss"] is None
    assert po.summary(po.price_legs([{"type": "call", "qty": 1, "K": 105, "dte": 45}]))["max_profit"] is None


def test_t21_curve_dropped_when_expiry_is_sooner_and_calendar_keeps_back_month():
    short = po.price_legs([{"type": "put", "qty": -1, "K": 95, "dte": 7}])
    assert {r["horizon"] for r in po.curves(short)} == {"T+0", "expiry"}
    cal = po.price_legs(po.setup_for("calendar")["legs"])
    at_front = {r["S"]: r["pnl"] for r in po.curves(cal) if r["horizon"] == "expiry"}
    assert at_front[100.0] > 0                     # long back month still worth more than the debit


def test_every_card_slug_with_legs_has_a_payoff_setup_or_none():
    for slug in list(po.SETUPS) + list(po.EXTRA_SETUPS):
        s = po.setup_for(slug)
        legs = po.price_legs(s["legs"])
        assert all(l["K"] is None or 20 < l["K"] < 300 for l in legs), slug
    assert po.setup_for("futures-options") is None


# --- Phase 06: sizing calculator ------------------------------------------------

def test_calculator_band_and_per_trade():
    r = sz.calculate(100_000, 40, "reg_t", 18)
    assert r["bp_range"] == (25_000, 40_000)                 # capped by the 40% sleeve
    assert r["per_trade"] == (1_000, 5_000)
    assert r["positions"] == (5, 40)
    assert r["stress_cap"] == (8_000, 17_000)
    assert r["theta_day"] == pytest.approx(100)
    assert "30%" in r["lean"]


@pytest.mark.parametrize("nlv", [0, -5, float("nan"), "abc"])
def test_calculator_rejects_bad_nlv(nlv):
    r = sz.calculate(nlv, 30, "reg_t", 20)
    assert r["ok"] is False and r["warnings"]


def test_calculator_edge_cases():
    small = sz.calculate(10_000, 10, "cash", 40)
    assert small["bp_range"] == (1_000, 1_000)              # sleeve below the 25% floor
    joined = " ".join(small["warnings"])
    assert "Cash account" in joined and "VIX at or above 30" in joined and "Small account" in joined
    assert small["stress_cap"] is None                      # no undefined risk in cash
    assert "Portfolio margin usually needs" in " ".join(sz.calculate(50_000, 30, "pm", 15)["warnings"])
    assert sz.calculate(50_000, 0, "ira", 15)["ok"] is False
    assert sz.calculate(50_000, 30, "ira", None)["lean"] is None
    assert "beyond" in " ".join(sz.calculate(50_000, 90, "reg_t", 15)["warnings"])


# --- Phase 06: build integrity --------------------------------------------------

def _synthetic_art(extra_ref: str = "") -> dict:
    card = {k: [] for k in vb.CARD_KEYS}
    card.update({"slug": "short-put", "name": "Short put", "family": "x", "one_liner": "Sell a put "
                 "[page:concepts-short-put] [ep:mm-2024-01-01@01:30] " + extra_ref,
                 "risk": "undefined", "evidence": {"tier": "B", "kind": "EDGE", "note": "",
                                                    "literature": ["lit:p1"], "repro": ["repro:vrp_spx"]}})
    return {"strategies": {"status": "draft", "cards": [card]}, "edges": {"edges": []},
            "mechanics": {"trees": [{"id": "t", "nodes": [{"id": "a"}], "edges": [{"from": "a", "to": "b"}]}]},
            "portfolio": {}, "glossary": {"terms": []}, "pop": {"rows": []}, "ranking": {},
            "literature": {"topics": [{"id": "vrp", "papers": [{"id": "p1", "authors": "A & B",
                                                                "year": 2001, "title": "T", "url": "u"}]}]},
            "manifest": {"episodes": [{"episode_id": "mm-2024-01-01", "show": "market-measures",
                                       "air_date": "2024-01-01", "title": "Ep", "url": "eu",
                                       "youtube_id": "yt1", "duration_s": 600}]},
            "pages": {"pages": [{"slug": "concepts-short-put", "url": "pu"}]},
            "repro": {"vrp_spx": {"claim": "IV > RV"}}}


def test_build_resolves_every_ref_and_counts():
    out = vb.pack(_synthetic_art())
    assert out["problems"] == ["tree t: edge a->b"]
    assert set(out["refs"]) == {"page:concepts-short-put", "ep:mm-2024-01-01", "lit:p1", "repro:vrp_spx"}
    assert out["refs"]["ep:mm-2024-01-01"]["yt"] == "yt1"
    assert out["refs"]["lit:p1"]["label"] == "A & B (2001): T"
    assert out["counts"]["strategies"] == 1 and out["counts"]["tiers"] == {"B": 1}


def test_build_flags_unresolved_and_late_timestamps():
    out = vb.pack(_synthetic_art("[lit:nope] [ep:mm-2024-01-01@20:00]"))
    assert "unresolved lit:nope" in out["problems"]
    assert "ep:mm-2024-01-01@20:00 beyond runtime" in out["problems"]


def test_build_card_schema_flags_missing_keys():
    art = _synthetic_art()
    del art["strategies"]["cards"][0]["manage"]
    assert "card short-put: missing manage" in vb.pack(art)["problems"]


@pytest.mark.skipif(not vault.VAULT_FILES["strategies"].exists(), reason="no committed vault cards")
def test_committed_build_has_no_problems():
    b = vault.load("build", {})
    assert b, "run python -m zenith.vault.build"
    assert b["problems"] == []
    assert b["counts"]["strategies"] == len(vault.load("strategies")["cards"])


def test_dte_timeline_captures_credit_from_zero_to_all():
    from zenith.vault.view import dte_timeline_rows
    rows = dte_timeline_rows()
    caps = [r["captured"] for r in rows]
    assert rows[0]["dte"] == 45 and caps[0] == pytest.approx(0, abs=1e-9)
    assert caps[-1] == pytest.approx(1, abs=1e-6)               # worthless at expiry, price unchanged
    assert all(b >= a - 1e-9 for a, b in zip(caps, caps[1:]))   # decay only accumulates
    assert 0 < caps[24] < 1                                     # 21 DTE: part, not all, captured


def test_tree_dot_is_valid_and_strips_refs():
    from zenith.vault.view import tree_dot
    dot = tree_dot({"nodes": [{"id": "a", "type": "decision", "text": 'Is it "liquid"? [page:x-y]'},
                              {"id": "b", "type": "action", "text": "Close"}],
                    "edges": [{"from": "a", "to": "b", "label": "yes"}]})
    assert dot.startswith("digraph") and dot.rstrip().endswith("}")
    assert "[page:" not in dot and '"a" -> "b"' in dot and "'liquid'" in dot
