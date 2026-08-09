"""HOLDINGS tests — offline, fixture and synthetic data only (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import zenith.holdings as hold
from zenith.holdings import compare, history, normalize, registry, snapshot
from zenith.holdings.sources import imgp

FIXTURE = Path(__file__).parent / "fixtures" / "dbmf_holdings.html"


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Redirect every artefact path into tmp_path, per fund."""
    def _files(fund: str) -> dict:
        d = tmp_path / fund
        d.mkdir(parents=True, exist_ok=True)
        return {k: d / f"{k}.json"
                for k in ("latest", "history", "changes", "status")}

    def _archive(fund: str):
        d = tmp_path / fund / "archive"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(hold, "holdings_files", _files)
    monkeypatch.setattr(hold, "holdings_archive_dir", _archive)
    monkeypatch.setattr(hold, "HOLDINGS_FUNDS_JSON", tmp_path / "funds.json")
    return tmp_path


# --- number parsing ----------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ("$                    -3,991,787,728.45", -3991787728.45),
    ("$ 7,955,184.80", 7955184.80),
    ("-3,879,200,000", -3879200000.0),
    ("0", 0.0),
    ("-0.96", -0.96),
    ("(1,234.50)", -1234.50),
    ("-", None),
    ("", None),
    ("N/A", None),
    (None, None),
])
def test_to_number(raw, expect):
    assert imgp.to_number(raw) == expect


# --- parsing the published table ---------------------------------------------

def test_parse_fixture_shape(fixture_html):
    rows, meta = imgp.parse(fixture_html)
    # 5 futures + 2 bills + the total row; the header row must be dropped.
    assert len(rows) == 8
    assert meta["skipped_non_data_rows"] == 1
    assert meta["value_date"] == "08/07/2026"
    assert "mixed_value_dates" not in meta


def test_parse_preserves_signs_and_names(fixture_html):
    rows, _ = imgp.parse(fixture_html)
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["TUU6"]["notional"] == -3991787728.45
    assert by_ticker["TUU6"]["qty"] == -3879200000.0
    assert by_ticker["TUU6"]["weight"] == -0.96
    assert by_ticker["ESU6"]["notional"] == 643917937.50
    assert by_ticker["ESU6"]["security_name"] == "S+P500 EMINI FUT SEP26"


def test_parse_empty_html_is_not_a_crash():
    rows, meta = imgp.parse("")
    assert rows == []
    assert meta["value_date"] == ""


# --- contract identity -------------------------------------------------------

@pytest.mark.parametrize("ticker,root,expiry", [
    ("TUU6", "TU", "2026-09"),
    ("TUZ6", "TU", "2026-12"),
    ("ECU6", "EC", "2026-09"),
    ("MESU6", "MES", "2026-09"),
    ("MFSU6", "MFS", "2026-09"),
    ("GCZ6", "GC", "2026-12"),
    ("CLV6", "CL", "2026-10"),
    ("USU6", "US", "2026-09"),
    ("CZ6", "C", "2026-12"),
])
def test_parse_contract(ticker, root, expiry):
    got = normalize.parse_contract(ticker, asof=date(2026, 8, 7))
    assert got["root"] == root
    assert got["expiry"] == expiry
    assert got["is_future"] is True


def test_parse_contract_year_rolls_into_next_decade():
    # A March 2030 contract quoted as "H0" while the snapshot is in 2029.
    got = normalize.parse_contract("TUH0", asof=date(2029, 12, 1))
    assert got["expiry"] == "2030-03"


def test_parse_contract_non_future():
    assert normalize.parse_contract("-")["is_future"] is False
    assert normalize.parse_contract("SPY")["root"] == "SPY"
    assert normalize.parse_contract("SPY")["is_future"] is False


def test_classify_known_and_unknown():
    display, cls, sub, flags = normalize.classify("TU", "US 2YR NOTE (CBT) SEP26")
    assert (display, cls) == ("US 2-Year Note", "rates")
    assert flags == []

    # An unmapped root still lands in the right asset class via the name.
    _, cls, _, flags = normalize.classify("QQ", "SUGAR NO11 (WORLD) MAR27")
    assert cls == "commodity"
    assert any(f.startswith("unmapped_root") for f in flags)

    # Nothing recognisable is flagged rather than guessed at.
    _, cls, _, flags = normalize.classify("ZZ", "SOMETHING ENTIRELY NEW")
    assert cls == "unclassified"
    assert any(f.startswith("unclassified") for f in flags)


def test_classify_flags_a_root_that_stops_matching_its_name():
    _, cls, _, flags = normalize.classify("GC", "COPPER FUTURE DEC26")
    assert cls == "commodity"
    assert "root_name_mismatch:GC" in flags


# --- normalisation -----------------------------------------------------------

def _norm(fixture_html):
    rows, _ = imgp.parse(fixture_html)
    return normalize.normalize_rows(rows, asof=date(2026, 8, 7))


def test_total_net_assets_becomes_nav_not_a_position(fixture_html):
    out = _norm(fixture_html)
    assert out["nav"] == 4154320727.25
    assert all("TOTAL" not in p["raw_name"].upper() for p in out["positions"])


def test_treasury_bills_collapse_to_one_collateral_line(fixture_html):
    out = _norm(fixture_html)
    cash = [p for p in out["positions"] if p["id"] == normalize.CASH_ID]
    assert len(cash) == 1
    assert cash[0]["asset_class"] == "collateral"
    assert cash[0]["n_lots"] == 2
    assert cash[0]["notional"] == pytest.approx(7955184.80 + 2023096279.92, rel=1e-9)


def test_weight_is_recomputed_from_notional_and_nav(fixture_html):
    out = _norm(fixture_html)
    tu = next(p for p in out["positions"] if p["id"] == "TU")
    assert tu["weight"] == pytest.approx(-3991787728.45 / 4154320727.25, abs=1e-6)
    assert tu["weight_published"] == -0.96          # the fund's rounded figure
    assert tu["direction"] == "short"


def test_exposure_summary_excludes_collateral(fixture_html):
    out = _norm(fixture_html)
    s = normalize.exposure_summary(out["positions"])
    assert s["n_positions"] == 5                     # bills are not a position
    assert s["short"] < 0 and s["long"] > 0
    assert s["gross"] == pytest.approx(s["long"] - s["short"], rel=1e-9)
    assert s["net"] == pytest.approx(s["long"] + s["short"], rel=1e-9)
    assert s["collateral"] > 0
    assert s["largest"]["id"] == "TU"


# --- snapshot validation -----------------------------------------------------

def _meta(parsed: dict | None = None, **overrides):
    """Adapter meta for a full-size page.

    The fixture is a trimmed table, so its own `html_bytes` would trip the
    partial-download guard; tests that care about that guard pass it here.
    """
    base = {"kind": "test", "url": "x"}
    base.update(parsed or {})
    base["html_bytes"] = 6_000_000
    base.update(overrides)
    return base


def test_snapshot_builds_and_validates(fixture_html):
    rows, m = imgp.parse(fixture_html)
    snap = snapshot.build("dbmf", rows, _meta(m))
    assert snapshot.is_ok(snap), snap["quality"]["errors"]
    assert snap["as_of"] == "2026-08-07"
    assert snap["n_positions"] == 6                  # 5 futures + collateral


def test_snapshot_rejects_a_parse_break(fixture_html):
    rows, m = imgp.parse(fixture_html)
    snap = snapshot.build("dbmf", rows[:2], _meta(m))
    assert not snapshot.is_ok(snap)
    assert "parse break" in snapshot.reason(snap)


def test_snapshot_rejects_a_missing_total_row(fixture_html):
    rows, m = imgp.parse(fixture_html)
    trimmed = [r for r in rows if "TOTAL" not in r["security_name"].upper()]
    snap = snapshot.build("dbmf", trimmed, _meta(m))
    assert not snapshot.is_ok(snap)
    assert "TOTAL NET ASSETS" in snapshot.reason(snap)


def test_snapshot_rejects_an_implausible_nav_jump(fixture_html):
    rows, m = imgp.parse(fixture_html)
    prev = {"as_of": "2026-08-06", "nav": 1_000_000_000.0, "positions": []}
    snap = snapshot.build("dbmf", rows, _meta(m), previous=prev)
    assert not snapshot.is_ok(snap)
    assert "NAV moved" in snapshot.reason(snap)


def test_snapshot_rejects_a_partial_download(fixture_html):
    rows, m = imgp.parse(fixture_html)
    snap = snapshot.build("dbmf", rows, _meta(m, html_bytes=4_000))
    assert not snapshot.is_ok(snap)
    assert "partial download" in snapshot.reason(snap)


def test_snapshot_rejects_a_stale_live_fetch_but_allows_a_replay(fixture_html):
    rows, m = imgp.parse(fixture_html)
    old = _meta(m, value_date="01/02/2020")
    assert not snapshot.is_ok(snapshot.build("dbmf", rows, old, live=True))
    assert snapshot.is_ok(snapshot.build("dbmf", rows, old, live=False))


def test_snapshot_rejects_a_future_value_date(fixture_html):
    rows, m = imgp.parse(fixture_html)
    snap = snapshot.build("dbmf", rows, _meta(m, value_date="12/31/2099"),
                          live=False)
    assert not snapshot.is_ok(snap)
    assert "future" in snapshot.reason(snap)


# --- change detection --------------------------------------------------------

def _snap(as_of: str, weights: dict[str, float], contracts=None) -> dict:
    contracts = contracts or {}
    positions = [{
        "id": pid, "name": pid, "asset_class": "rates", "sub_class": "x",
        "contract": contracts.get(pid, f"{pid}U6"), "expiry": "2026-09",
        "cusip": "", "instrument": "future", "raw_name": pid,
        "qty": 0.0, "notional": w * 1e9, "weight": w, "n_lots": 1,
        "direction": normalize.direction_of(w),
    } for pid, w in weights.items()]
    return {"as_of": as_of, "nav": 1e9, "positions": positions,
            "summary": normalize.exposure_summary(positions)}


@pytest.mark.parametrize("before,after,expect", [
    (None, 0.20, "new"),
    (0.20, None, "closed"),
    (0.20, 0.30, "increased"),
    (0.30, 0.20, "reduced"),
    (-0.20, -0.30, "increased"),      # a bigger short is a bigger position
    (-0.30, -0.20, "reduced"),
    (0.20, -0.20, "flipped"),
    (0.20, 0.2001, "held"),           # below the noise floor
])
def test_classify_move(before, after, expect):
    assert compare.classify_move(before, after) == expect


def test_diff_reports_direction_and_percentage():
    a = _snap("2026-08-06", {"TU": -0.50, "ES": 0.10})
    b = _snap("2026-08-07", {"TU": -0.75, "GC": 0.05})
    events = {e["id"]: e for e in compare.diff(a, b)}
    assert events["TU"]["type"] == "increased"
    assert events["TU"]["d_weight"] == pytest.approx(-0.25)
    assert events["TU"]["d_abs_weight"] == pytest.approx(0.25)
    assert events["TU"]["pct_change"] == pytest.approx(0.5)
    assert events["ES"]["type"] == "closed"
    assert events["GC"]["type"] == "new"


def test_diff_marks_a_roll_without_calling_it_new():
    a = _snap("2026-08-29", {"TU": -0.50}, {"TU": "TUU6"})
    b = _snap("2026-09-02", {"TU": -0.50}, {"TU": "TUZ6"})
    events = compare.diff(a, b, include_held=True)
    assert len(events) == 1
    assert events[0]["type"] == "held"
    assert events[0]["rolled"] is True


def test_change_cap_tracks_the_funds_own_distribution():
    small = [{"type": "increased", "d_weight": 0.002} for _ in range(20)]
    big = [{"type": "increased", "d_weight": 0.20} for _ in range(20)]
    assert compare.change_cap(big) > compare.change_cap(small)
    assert compare.change_cap([]) == compare.CAP_FLOOR


# --- history -----------------------------------------------------------------

def _three_days():
    return [_snap("2026-08-05", {"TU": -0.50, "ES": 0.10}),
            _snap("2026-08-06", {"TU": -0.60}),
            _snap("2026-08-07", {"TU": -0.55, "GC": 0.05})]


def test_history_marks_absence_as_null_not_zero():
    h = history.build("dbmf", _three_days())
    assert h["dates"] == ["2026-08-05", "2026-08-06", "2026-08-07"]
    assert h["series"]["ES"]["w"] == [0.10, None, None]
    assert h["series"]["GC"]["w"] == [None, None, 0.05]
    assert h["series"]["TU"]["w"] == [-0.50, -0.60, -0.55]


def test_history_is_order_independent():
    snaps = _three_days()
    a = history.build("dbmf", snaps)
    b = history.build("dbmf", list(reversed(snaps)))
    assert a == b


def test_history_dates_stay_iso_strings():
    h = history.build("dbmf", _three_days())
    assert all(isinstance(d, str) and len(d) == 10 for d in h["dates"])


def test_position_stats():
    h = history.build("dbmf", _three_days())
    s = history.position_stats(h, "TU")
    assert s["held"] is True
    assert s["first_seen"] == "2026-08-05"
    assert s["days_observed"] == 3
    assert s["max_weight"]["w"] == -0.60
    assert s["current_direction"] == "short"
    assert s["n_changes"] == 2
    assert history.position_stats(h, "NOPE") == {"id": "NOPE", "held": False}


def test_delta_matrix_labels_openings_and_closings():
    h = history.build("dbmf", _three_days())
    recs = {(r["id"], r["d"]): r for r in history.delta_matrix(h)}
    assert recs[("ES", "2026-08-06")]["event"] == "closed"
    assert recs[("GC", "2026-08-07")]["event"] == "new"
    assert recs[("TU", "2026-08-06")]["event"] == "move"
    assert recs[("ES", "2026-08-07")]["event"] == "absent"


def test_matrix_emits_absent_cells_so_gaps_are_drawable():
    h = history.build("dbmf", _three_days())
    recs = history.matrix(h)
    es = [r for r in recs if r["id"] == "ES"]
    assert len(es) == 3
    assert [r["state"] for r in es] == ["long", "absent", "absent"]


def test_window_changes_admits_when_the_span_is_not_what_was_asked_for():
    # Two snapshots two months apart: a "1 day" window cannot be honoured.
    h = history.build("dbmf", [_snap("2026-06-05", {"TU": -0.50}),
                               _snap("2026-08-07", {"TU": -0.90})])
    w = compare.window_changes(h, 1)
    assert w["exact"] is False
    assert w["requested_days"] == 1
    assert w["actual_days"] == 63
    # A window the data can actually satisfy is marked exact.
    h2 = history.build("dbmf", [_snap("2026-08-06", {"TU": -0.50}),
                                _snap("2026-08-07", {"TU": -0.90})])
    assert compare.window_changes(h2, 5)["exact"] is True


def test_history_records_skipped_sessions_before_each_date():
    h = history.build("dbmf", _three_days())
    assert h["gap_before"] == [0, 0, 0]          # Aug 5, 6, 7 are consecutive
    h2 = history.build("dbmf", [_snap("2026-08-03", {"TU": -0.5}),
                                _snap("2026-08-07", {"TU": -0.5})])
    assert h2["gap_before"] == [0, 3]            # Aug 4, 5, 6 missing


def test_delta_matrix_labels_a_gapped_column():
    h = history.build("dbmf", [_snap("2026-08-03", {"TU": -0.5}),
                               _snap("2026-08-07", {"TU": -0.9})])
    rec = history.delta_matrix(h)[0]
    assert rec["gap"] == 3
    assert "snapshots missing" in rec["spans"]


def test_window_changes_compares_endpoints_not_a_sum_of_days():
    # TU goes out to -0.90 and comes back: the window must report the net move.
    snaps = [_snap("2026-08-05", {"TU": -0.50}),
             _snap("2026-08-06", {"TU": -0.90}),
             _snap("2026-08-07", {"TU": -0.55})]
    h = history.build("dbmf", snaps)
    w = compare.window_changes(h, 5)
    tu = next(r for r in w["increases"] + w["decreases"] if r["id"] == "TU")
    assert tu["d_abs_weight"] == pytest.approx(0.05)


# --- persistence + rebuild ---------------------------------------------------

def test_archive_roundtrip_and_derive_is_reproducible(tmp_store):
    from zenith.holdings import compute

    for s in _three_days():
        hold.archive_day("dbmf", s["as_of"], s)
    assert hold.archive_days("dbmf") == ["2026-08-07", "2026-08-06", "2026-08-05"]

    first = compute.derive("dbmf")
    h1 = hold.load("dbmf", "history")
    c1 = hold.load("dbmf", "changes")
    latest = hold.load("dbmf", "latest")

    assert first["n_dates"] == 3
    assert latest["as_of"] == "2026-08-07"
    assert latest["previous_as_of"] == "2026-08-06"
    assert latest["summary"]["n_added"] == 1          # GC opened
    assert set(c1["rankings"]) == set(hold.WINDOWS)

    # Rebuilding from the archive alone must reproduce the same artefacts.
    compute.derive("dbmf")
    assert hold.load("dbmf", "history") == h1
    assert hold.load("dbmf", "changes") == c1


def test_archiving_the_same_day_twice_is_idempotent(tmp_store):
    from zenith.holdings import compute

    snaps = _three_days()
    for s in snaps:
        hold.archive_day("dbmf", s["as_of"], s)
    a = compute.derive("dbmf")
    hold.archive_day("dbmf", snaps[-1]["as_of"], snaps[-1])
    b = compute.derive("dbmf")
    assert a == b


# --- registry ----------------------------------------------------------------

def test_registry_exposes_dbmf_with_a_usable_adapter():
    f = registry.get("dbmf")
    assert f.ticker == "DBMF"
    assert f.adapter == "imgp"
    assert f.source_url.startswith("https://")
    assert f.caveats                                   # the UI relies on these
    snap = registry.registry_snapshot()
    assert snap["default"] == "dbmf"
    assert snap["funds"][0]["key"] == "dbmf"


def test_registry_rejects_an_unknown_fund():
    with pytest.raises(KeyError):
        registry.get("nope")
