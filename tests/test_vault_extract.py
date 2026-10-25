"""OPTIONS VAULT Phase 02 extraction tests — offline, synthetic transcripts only (no anthropic)."""

from __future__ import annotations

import json
import sys

import pytest

from zenith.vault import extract as ex

TAX = {
    "version": "t1",
    "strategies": [{"slug": "short-strangle", "name": "Short strangle", "aliases": ["strangle"]},
                   {"slug": "iron-condor", "name": "Iron condor", "aliases": ["ic"]}],
    "topics": [{"slug": "dte-21-management", "name": "21-DTE management", "aliases": ["21 dte"]},
               {"slug": "profit-target", "name": "Profit targets", "aliases": ["50% profit"]}],
}

RAW = {"segments": [
    {"t": 0, "text": "today we look at strangles in SPY from 2005 to 2024"},
    {"t": 65, "text": "we sold the 16 delta strangle at 45 days to expiration"},
    {"t": 130, "text": "managing at fifty percent the win rate was 83% over 1,250 trades"},
    {"t": 400, "text": "and at twenty one days we close it out"},
]}

ROW = {"episode_id": "market-measures-2024-01-02", "show": "market-measures",
       "air_date": "2024-01-02", "title": "Strangle study", "url": "", "youtube_id": "abc123",
       "transcript": {"source": "yt_auto", "chars": 200}, "tl": {"tags": ["Options"]}}


def _out(**over) -> dict:
    out = {"episode_type": "study", "signal": 3, "strategies": ["short-strangle"],
           "topics": ["profit-target"],
           "studies": [{"question": "manage at 50% vs hold", "strategies": ["short-strangle"],
                        "underlyings": ["SPY"], "period": "2005-2024",
                        "entry": {"dte": 45, "rule": "16 delta strangle"},
                        "management": "50% of max profit", "sample_n": 1250,
                        "metrics": {"win_rate": 83, "avg_pnl": None, "pnl_per_day": None,
                                    "max_loss": None, "roc": None, "stdev": None},
                        "findings": ["managing early raised the win rate"], "caveats": ["SPY only"],
                        "ts": "01:30"}],
           "rules": [{"kind": "exit", "text": "close at 21 DTE", "conditions": "",
                      "strategies": ["short-strangle"], "topics": ["dte-21-management"],
                      "numbers": [{"name": "dte", "value": 21, "unit": "days"}], "ts": "06:40"}],
           "claims": [], "terms": [], "unmapped": []}
    out.update(over)
    return out


# --- prompt ------------------------------------------------------------------

def test_system_prompt_is_cached_and_deterministic():
    a, b = ex.build_system(TAX), ex.build_system(json.loads(json.dumps(TAX)))
    assert a == b and len(a) == 1
    assert a[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "short-strangle: Short strangle | strangle" in a[0]["text"]
    assert ex.build_system(TAX, "triage")[0]["text"].startswith("You triage")


def test_user_turn_has_metadata_and_timestamps():
    u = ex.build_user(ROW, ex.transcript_lines(RAW))
    assert "episode_id: market-measures-2024-01-02" in u and "tastylive tags: Options" in u
    assert "[01:05] we sold the 16 delta strangle" in u
    assert "part" not in u.split("TRANSCRIPT")[0]
    assert "transcript part 2 of 3" in ex.build_user(ROW, ["[00:00] x"], (2, 3))


def test_chunking_only_over_budget():
    lines = ex.transcript_lines(RAW)
    assert ex.chunk_lines(lines) == [lines]
    parts = ex.chunk_lines(lines, max_chars=80)
    assert len(parts) > 1 and sum(parts, []) == lines
    assert ex.chunk_lines([]) == [[]]


def test_episode_url_falls_back_to_youtube():
    assert ex.episode_url(ROW) == "https://www.youtube.com/watch?v=abc123"
    assert ex.episode_url({**ROW, "url": "https://tastylive.com/x"}) == "https://tastylive.com/x"


# --- schema ------------------------------------------------------------------

def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_api_schema_is_strict():
    for tier in ("full", "triage"):
        sch = ex.api_schema(TAX, tier)
        for node in _walk(sch):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
    full = ex.api_schema(TAX)
    assert full["properties"]["strategies"]["items"]["enum"] == ["short-strangle", "iron-condor"]
    assert set(ex.api_schema(TAX, "triage")["properties"]) == {"episode_type", "signal",
                                                               "strategies", "topics"}


def test_canonical_conversion_and_custom_ids():
    rec = ex.to_canonical(_out(), ROW)
    assert rec["rules"][0]["numbers"] == {"dte (days)": 21}
    assert rec["url"].endswith("abc123") and rec["studies"][0]["metrics"]["win_rate"] == 83
    assert ex.custom_id(ROW["episode_id"]) == ROW["episode_id"]
    assert ex.custom_id(ROW["episode_id"], 2, 3).endswith("__p2")


def test_merge_parts_unions_and_concatenates():
    a, b = _out(), _out(signal=2, strategies=["iron-condor", "short-strangle"], studies=[])
    m = ex.merge_parts([a, b])
    assert m["signal"] == 3 and m["strategies"] == ["short-strangle", "iron-condor"]
    assert len(m["studies"]) == 1 and len(m["rules"]) == 2


# --- checks ------------------------------------------------------------------

def test_parse_ts():
    assert ex.parse_ts("02:10") == 130 and ex.parse_ts("1:02:03") == 3723
    assert ex.parse_ts("75:00") == 4500
    assert ex.parse_ts("2m10s") is None and ex.parse_ts(None) is None


def test_validate_moves_unknown_slugs_to_unmapped():
    rec = ex.to_canonical(_out(strategies=["short-strangle", "wheel"]), ROW)
    rec["rules"][0]["topics"].append("made-up")
    assert ex.validate(rec, TAX) == []
    assert rec["strategies"] == ["short-strangle"] and set(rec["unmapped"]) == {"wheel", "made-up"}


def test_validate_flags_bad_records():
    rec = ex.to_canonical(_out(studies=[]), ROW)
    assert "signal 3 without studies" in ex.validate(rec, TAX)
    rec = ex.to_canonical(_out(), ROW)
    rec["rules"][0]["ts"] = "soon"
    assert any("rules[0].ts" in e for e in ex.validate(rec, TAX))
    assert ex.validate({"episode_id": "x"}, TAX)[0].startswith("missing")


def test_numbers_in_digits_and_words():
    assert set(ex.numbers_in("83% over 1,250 trades at .16 and 2.5x")) >= {83, 1250, 0.16, 2.5}
    assert 10000 in ex.numbers_in("a 10k account")
    assert 21 in ex.numbers_in("at twenty one days")
    assert 50 in ex.numbers_in("managing at fifty percent")
    assert 0.5 in ex.numbers_in("take half off")
    assert 300 in ex.numbers_in("three hundred trades")


def test_caption_number_forms():
    assert 2.83 in ex.numbers_in("my credit's at 283")
    assert {70.0, 30.0} <= set(ex.numbers_in("maybe 7030 or 6931"))
    assert 1.6 in ex.numbers_in("if I sold this for a dollar sixty")
    assert 3.6 in ex.numbers_in("max profit would be three dollars and sixty cents")
    assert 1.8 in ex.numbers_in("that's a dollar eighty")
    assert 105 in ex.numbers_in("one hundred and five trades")


def test_study_setup_window_and_findings_measure():
    rec = ex.to_canonical(_out(), ROW)
    rec["studies"][0]["ts"] = "05:00"                   # 45 DTE said at 01:05, within 300 s before
    rec["studies"][0]["metrics"]["win_rate"] = None
    rec["studies"][0]["sample_n"] = None
    rec["studies"][0]["findings"] = ["win rate 83% over 1,250 trades", "returned 99%"]
    g = ex.ground(rec, RAW)
    assert not [f for f in g["failures"] if f["where"].startswith("studies")]
    assert g["findings_numbers"] == {"checked": 3, "passed": 2}


def test_number_matching_tolerances():
    assert ex.number_matches(83.0, [83])
    assert ex.number_matches(0.16, [16])        # fraction vs percent
    assert ex.number_matches(86.7, [87])        # rounding
    assert ex.number_matches(-3.0, [3])         # sign spoken as "lost 3"
    assert not ex.number_matches(72.0, [83, 16, 45])
    assert not ex.number_matches(12.5, [1250])       # no x100 on values > 1


def test_grounding_passes_and_fails():
    rec = ex.to_canonical(_out(), ROW)
    g = ex.ground(rec, RAW)
    assert g["checked"] == 4 and g["failures"] == []   # sample_n, dte, win_rate, rule dte
    bad = ex.to_canonical(_out(), ROW)
    bad["studies"][0]["metrics"]["roc"] = 12.5          # never said
    bad["rules"][0]["ts"] = "00:00"                    # 21 said at 06:40, not near 00:00
    g = ex.ground(bad, RAW)
    assert {f["where"]: f["anywhere"] for f in g["failures"]} == {
        "studies[0].metrics.roc": False,                # nowhere: suspected hallucination
        "rules[0].dte (days)": True}                    # said, but at another time
    assert g["nowhere"] == 1


def test_grounding_window_edges():
    rec = ex.to_canonical(_out(studies=[], rules=[]), ROW)
    rec["claims"] = [{"text": "x", "topics": [], "numbers": {"n": 45}, "ts": "02:05"}]
    assert ex.ground(rec, RAW)["failures"] == []        # 45 said at 01:05, 60 s before
    rec["claims"][0]["ts"] = "02:06"
    assert len(ex.ground(rec, RAW)["failures"]) == 1


def test_verbatim_check():
    rec = ex.to_canonical(_out(), ROW)
    assert ex.verbatim(rec, RAW) == []
    rec["claims"] = [{"text": "Managing at fifty percent the win rate was 83% over 1,250 trades!",
                      "topics": [], "numbers": {}, "ts": "02:10"}]
    assert len(ex.verbatim(rec, RAW, n=8)) == 1


def test_check_bundle():
    c = ex.check(ex.to_canonical(_out(), ROW), RAW, TAX)
    assert c["schema_ok"] and c["grounding"]["passed"] == 4 and c["verbatim"] == []


# --- selection and costs -------------------------------------------------------

def test_needs_extract():
    v = "x1+tax-t1"
    assert ex.needs_extract(ROW, None, v)
    cur = {"extract": {"version": v, "transcript_chars": 200}}
    assert not ex.needs_extract(ROW, cur, v)
    assert ex.needs_extract(ROW, cur, "x2+tax-t1")
    assert ex.needs_extract({**ROW, "transcript": {"source": "yt_auto", "chars": 999}}, cur, v)
    assert not ex.needs_extract({**ROW, "transcript": {"source": "none", "chars": 0}}, None, v)


def test_pilot_set_is_stratified_and_seeded():
    rows = [{"episode_id": f"{s}-{y}-{i:02d}", "show": s, "air_date": f"{y}-01-01"}
            for s in ("market-measures", "research") for y in (2023, 2025) for i in range(30)]
    a, b = ex.pilot_set(rows, 40), ex.pilot_set(rows, 40)
    assert a == b and len(a) == 40 and len(set(a)) == 40
    mm = sum(e.startswith("market-measures") for e in a)
    assert 24 <= mm <= 30                               # MM weighted 2x
    assert {e.split("-")[-2] for e in a} >= {"2023", "2025"}


def test_costs():
    u = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert ex.cost_usd("claude-opus-5-5", u) == pytest.approx(2.0)
    assert ex.cost_usd("claude-opus-5-5", u, batch=False) == pytest.approx(4.0)
    p = ex.project_cost("claude-sonnet-5-5", [1000] * 10, system_tokens=5000, out_tokens=2000)
    want = (5000 * 2 * 2.0 + 9 * 5000 * 0.20 + 10_000 * 2.0 + 20_000 * 10.0) / 1e6 * 0.5
    assert p == pytest.approx(want)
    assert ex.project_cost("claude-sonnet-5-5", [], 5000, 2000) == 0.0


def test_two_tier_summary_counts_missed():
    def full(eid, studies):
        r = ex.to_canonical(_out(studies=[] if not studies else _out()["studies"]), {**ROW, "episode_id": eid})
        r["extract"] = {"usage": {"output_tokens": 100}, "cost_usd": 1.0}
        r["checks"] = ex.check(r, RAW, TAX)
        return r
    fulls = {"a": full("a", True), "b": full("b", True), "c": full("c", False)}
    triage = {e: {"signal": s, "extract": {"cost_usd": 0.1}} for e, s in (("a", 3), ("b", 0), ("c", 0))}
    rep = ex.summarize_two_tier(triage, fulls)
    assert rep["kept"] == 1 and rep["missed_episodes"] == ["b", "c"]   # c still has a rule
    assert rep["cost_usd"] == pytest.approx(0.3 + 1.0)


def test_no_anthropic_import():
    assert "anthropic" not in sys.modules


# --- in-session path -----------------------------------------------------------

def test_reader_kit(tmp_path):
    instr, schema = ex.write_kit(TAX, tmp_path)
    text = instr.read_text(encoding="utf-8")
    assert text.startswith(ex.INSTRUCTIONS[:40]) and "short-strangle: Short strangle" in text
    sch = json.loads(schema.read_text(encoding="utf-8"))
    assert sch["required"][0] == "episode_id" and "studies" in sch["properties"]


def test_numbers_accept_dict_form():
    assert ex._num_dict({"dte": 21, "note": "x"}) == {"dte": 21}


def test_ingest_writes_checked_records(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "load_rows", lambda: [ROW])
    monkeypatch.setattr(ex, "load_raw", lambda eid: RAW)
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    good = {"episode_id": ROW["episode_id"], **_out()}
    (src / "a.json").write_text(json.dumps([good, {"episode_id": "nope", **_out()}]), encoding="utf-8")
    done = ex.ingest(src, dest, TAX)
    assert done == [{"episode_id": ROW["episode_id"], "schema_ok": True, "grounding": "4/4"}]
    rec = json.loads((dest / f"{ROW['episode_id']}.json").read_text(encoding="utf-8"))
    assert rec["extract"]["config"] == "inline" and rec["extract"]["cost_usd"] == 0.0
    assert rec["extract"]["version"] == "x1+tax-t1" and rec["rules"][0]["numbers"] == {"dte (days)": 21}
    assert not (dest / "nope.json").exists()


def test_export_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "load_raw", lambda eid: RAW)
    (path,) = ex.export_transcripts([ROW], tmp_path)
    assert "[02:10] managing at fifty percent" in path.read_text(encoding="utf-8")
