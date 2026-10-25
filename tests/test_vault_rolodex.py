"""OPTIONS VAULT Phase 05 rolodex tests — offline, synthetic data only."""

from __future__ import annotations

import pytest

from zenith.vault import pop, rolodex

TAX = {
    "version": "t1",
    "strategies": [{"slug": "short-strangle", "name": "Short strangle", "aliases": ["strangle", "strangel"]},
                   {"slug": "iron-condor", "name": "Iron condor", "aliases": ["ic", "condor"]},
                   {"slug": "zero-dte", "name": "0DTE structures", "aliases": ["0dte"]}],
    "topics": [{"slug": "dte-21-management", "name": "21-DTE management", "aliases": ["21 dte"]},
               {"slug": "taxes", "name": "Taxes", "aliases": ["wash sale"]}],
}


# --- popularity ------------------------------------------------------------------

def test_popularity_matches_aliases_and_periods():
    eps = [{"episode_id": "a", "show": "mm", "year": 2014, "text": "selling strangles in spy"},
           {"episode_id": "b", "show": "mm", "year": 2019, "text": "iron condor vs strangle"},
           {"episode_id": "c", "show": "rs", "year": 2024, "text": "0dte iron condors at 21 dte"},
           {"episode_id": "d", "show": "rs", "year": 2024, "text": "picking a topic"}]
    p = rolodex.popularity(TAX, eps)
    by = {r["slug"]: r for r in p["strategies"]}
    assert by["short-strangle"]["episodes"] == 2           # plural "strangles" counts
    assert by["iron-condor"]["episodes"] == 2 and by["zero-dte"]["by_period"]["2022-2026"] == 1
    assert {r["slug"]: r["episodes"] for r in p["topics"]}["dte-21-management"] == 1
    assert p["periods"] == {"2012-2016": 1, "2017-2021": 1, "2022-2026": 2}


def test_short_aliases_are_ignored_unless_allowed():
    pat = rolodex._alias_pattern({"slug": "iron-condor", "name": "Iron condor", "aliases": ["ic"]})
    assert pat.search("an iron condor") and not pat.search("ic is ambiguous")
    assert rolodex._alias_pattern({"slug": "x", "name": "0dte", "aliases": []}).search("0dte trades")


# --- page notes --------------------------------------------------------------------

def _note(**over):
    n = {"page": "concepts-strangle", "summary": "Sell an OTM put and call.", "strategies": ["short-strangle", "wheel"],
         "topics": ["dte-21-management"], "unmapped": ["condor", "zebra"],
         "strategy_notes": [{"strategy": "short-strangle", "legs": "sell put and call"}],
         "rules": [{"kind": "exit", "text": "take profit at half", "strategies": ["short-strangle"],
                    "numbers": [{"name": "profit target", "value": 50, "unit": "%"}]}],
         "stats": [{"label": "win rate", "value": "83%", "number": 83, "strategies": []}],
         "mechanics": [], "terms": [], "quote": "manage winners at fifty percent"}
    n.update(over)
    return n


PAGE = ("The tastylive approach is to manage winners at fifty percent of max profit. In a study the win rate "
        "was 83% and the target was 50% of the credit received.")


def test_check_note_fixes_slugs_and_maps_unmapped_aliases():
    n = _note()
    c = rolodex.check_note(n, PAGE, TAX)
    assert n["strategies"] == ["short-strangle", "iron-condor"]      # 'condor' alias mapped back
    assert n["unmapped"] == ["zebra", "wheel"]                        # unknown slug moved to unmapped
    assert c["numbers_checked"] == 2 and c["numbers_ok"] == 2
    assert c["quote_ok"] and c["quote_in_page"] and c["verbatim"] == []


def test_check_note_flags_numbers_verbatim_and_quotes():
    n = _note(stats=[{"label": "made up", "number": 97}], quote="not on the page at all")
    n["summary"] = "the tastylive approach is to manage winners at fifty percent of max profit always"
    c = rolodex.check_note(n, PAGE, TAX)
    assert c["number_failures"] == ["made up"]
    assert len(c["verbatim"]) == 1 and not c["quote_in_page"]


# --- integrity check -----------------------------------------------------------------

def _card(**over):
    c = {k: [] for k in rolodex.CARD_KEYS}
    c.update({"slug": "short-strangle", "name": "Short strangle", "family": "neutral", "one_liner": "x",
              "risk": "undefined", "greeks": {}, "accounts": {}, "when": {}, "setup": {}, "manage": {},
              "size": {}, "example": {}, "evidence": {"tier": "B", "note": "VRP [lit:paper-1]"},
              "stats": [{"label": "win", "value": "87%", "sources": ["repro:strangle_mgmt"]}],
              "related": ["iron-condor"]})
    c.update(over)
    return c


@pytest.fixture
def res(monkeypatch):
    monkeypatch.setattr(rolodex, "ref_ok", lambda kind, key, ts, r: key in {"paper-1", "strangle_mgmt", "concepts-strangle"}
                        or (kind == "ep" and key == "mm-1" and ts in (None, "01:00")))
    return {"man": {}, "lit": set()}


def test_check_all_passes_clean_card(res):
    rep = rolodex.check_all([_card()], [], TAX, res)
    assert rep["problems"] == [] and rep["refs"] == 2


def test_check_all_reports_problems(res):
    bad = _card(evidence={"tier": "D", "note": "[lit:ghost]"}, related=["wheel"],
                stats=[{"label": "orphan", "value": "1"}], one_liner="")
    bad.pop("history")
    tree = {"id": "t", "nodes": [{"id": "a", "text": "[ep:mm-1@99:00]"}], "edges": [{"from": "a", "to": "b"}]}
    probs = "\n".join(rolodex.check_all([bad], [tree], TAX, res)["problems"])
    for want in ("missing history", "tier 'D'", "empty one_liner", "related not in taxonomy",
                 "stat without sources", "unresolved lit:ghost", "edge a->b", "unresolved ep:mm-1@99:00"):
        assert want in probs


def test_ref_regex_reads_timestamps_and_brackets():
    refs = rolodex.refs_in({"t": "see [ep:market-measures-2024-06-17@08:59] and lit:coval-shumway-2001."})
    assert ("ep", "market-measures-2024-06-17", "08:59") in refs
    assert ("lit", "coval-shumway-2001", "") in refs


# --- POP model --------------------------------------------------------------------------

def test_strike_for_delta_inverts_delta():
    T, sig = 45 / 365, 0.2
    for kind in ("put", "call"):
        k = pop.strike_for_delta(kind, 0.16, T, sig)
        assert abs(abs(pop.bs_delta(kind, pop.S0, k, T, sig)) - 0.16) < 1e-6
    assert pop.strike_for_delta("put", 0.16, T, sig) < pop.S0 < pop.strike_for_delta("call", 0.16, T, sig)


def test_put_call_parity_and_martingale_pricing():
    T, sig, K = 0.5, 0.25, 105.0
    assert pop.bs("call", 100, K, T, sig) - pop.bs("put", 100, K, T, sig) == pytest.approx(100 - K, abs=1e-9)
    legs = pop.build_legs([{"type": "call", "qty": 1, "atm": True}], 45, 0.2)
    ev = pop.evaluate(legs, 45, 0.2)                     # priced and evaluated at the same vol
    assert ev["mean_units"] == pytest.approx(0.0, abs=0.01)


def test_short_premium_gains_when_realized_below_implied():
    legs = pop.build_legs(pop.SETUPS["short-strangle"]["legs"], 45, 0.2)
    theo, real = pop.evaluate(legs, 45, 0.2), pop.evaluate(legs, 45, 0.2 * 0.78)
    assert real["pop"] > theo["pop"] > 0.68 and real["mean_units"] > 0 and theo["cvar5_units"] < -1
    longs = pop.build_legs(pop.SETUPS["long-strangle"]["legs"], 45, 0.2)
    assert pop.evaluate(longs, 45, 0.2 * 0.78)["cvar5_units"] == pytest.approx(-1.0)   # lose the debit


def test_table_rows_have_units_and_sources():
    t = pop.table()
    assert {r["unit"] for r in t["rows"]} == {"credit", "debit", "position cost"}
    assert all(r["sources"] for r in t["rows"]) and t["rows"][0]["pop_theoretical"] >= t["rows"][-1]["pop_theoretical"]


def test_accounts_follow_the_legs():
    L = lambda t, s, q=1: {"type": t, "side": s, "qty": q}
    acc = lambda legs: rolodex.accounts_for(legs)[0]
    assert acc([L("put", "short"), L("call", "short")]) == {"cash": False, "ira": False, "reg_t": True, "pm": True}
    assert acc([L("put", "short"), L("put", "long")])["ira"] and not acc([L("put", "short"), L("put", "long")])["cash"]
    assert acc([L("put", "long"), L("put", "short", 2)])["ira"] is False          # 1x2 ratio has a naked put
    assert acc([L("stock", "long"), L("call", "short")])["cash"]                     # covered call
    assert acc([L("stock", "long"), L("call", "short"), L("put", "long")])["cash"]   # collar
    assert acc([L("call", "long"), L("call", "short")]) == {"cash": False, "ira": True, "reg_t": True, "pm": True}
    assert acc([L("call", "long")])["cash"]


def test_ranking_separates_earners_from_insurance():
    def card(slug, family, risk, tier, episodes=0, thin=""):
        return {"slug": slug, "name": slug, "family": family, "risk": risk, "thin_evidence": thin,
                "evidence": {"tier": tier, "kind": "EDGE", "note": ""}, "popularity": {"episodes": episodes}}
    cards = [card("short-strangle", "neutral short premium", "undefined", "B", 600),
             card("short-put", "directional short premium", "undefined", "B", 180, thin="thin"),
             card("iron-condor", "neutral short premium", "defined", "B", 260),
             card("butterfly", "ratio and butterfly", "defined", "C", 900),
             card("protective-put", "hedges", "defined", "A", 10),
             card("collar", "stock-based", "defined", "B", 5)]
    pops = [{"slug": "short-strangle", "pop_theoretical": 0.75}, {"slug": "iron-condor", "pop_theoretical": 0.74}]
    r = rolodex.ranking(cards, pops)
    assert [x["slug"] for x in r["most_reliable_defined"]] == ["iron-condor", "butterfly"]   # tier beats popularity
    assert [x["slug"] for x in r["most_reliable_undefined"]] == ["short-strangle", "short-put"]
    assert {x["slug"] for x in r["insurance_and_long_premium"]} == {"protective-put", "collar"}
    assert r["most_popular"][0]["slug"] == "butterfly" and [x["slug"] for x in r["highest_pop"]] == ["short-strangle", "iron-condor"]


def test_check_extras(res):
    edge = {k: [] for k in rolodex.EDGE_KEYS}
    edge.update({"slug": "vrp", "title": "VRP", "finding": "IV > RV [lit:paper-1]", "conditions": "", "repro": None,
                 "agreement": "agrees", "tier": "A", "kind": "EDGE", "act_with": ["short-strangle"],
                 "numbers": [{"label": "spread", "value": "4", "sources": ["repro:strangle_mgmt"]}]})
    port = {k: [] for k in ("account_matrix", "sleeve_playbook", "account_tiers", "taxes", "unverified")}
    port["allocation"] = {"per_trade": {"guidance": "5% [page:concepts-strangle]"}}
    gloss = [{"term": "IV rank", "definition": "where IV sits in its range", "related": ["short-strangle"],
              "sources": ["page:concepts-strangle"]}]
    ok = rolodex.check_extras([edge], port, gloss, TAX, res)
    assert ok["problems"] == [] and ok["refs"] == 4
    bad_edge = {**edge, "tier": "Z", "kind": "MECHANIC", "agreement": "maybe", "act_with": ["wheel"],
                "numbers": [{"label": "x", "value": "1"}]}
    bad_port = {"allocation": {}}
    bad_gloss = [{"term": "x", "definition": "", "related": ["nope"], "sources": ["page:ghost"]}]
    probs = "\n".join(rolodex.check_extras([bad_edge], bad_port, bad_gloss, TAX, res)["problems"])
    for want in ("tier 'Z'", "kind 'MECHANIC'", "agreement 'maybe'", "act_with not in taxonomy", "number without sources",
                 "portfolio: missing account_matrix", "empty definition", "related not in taxonomy", "unresolved page:ghost"):
        assert want in probs


def test_accounts_generic_legs_and_ira_uncovered_calls():
    L = lambda t, s, q=1: {"type": t, "side": s, "qty": q}
    acc = lambda legs: rolodex.accounts_for(legs)
    cal = acc([L("call or put", "short"), L("call or put", "long")])[0]
    assert cal == {"cash": False, "ira": True, "reg_t": True, "pm": True}          # calendar is a spread
    assert acc([L("call or put", "long"), L("call or put", "short", 2), L("call or put", "long")])[0]["ira"]
    a, note = acc([L("call", "short")])
    assert a["ira"] and not a["cash"] and "$25,000" in note                         # IRA uncovered calls at The Works
    assert not acc([L("put", "short")])[0]["ira"]
