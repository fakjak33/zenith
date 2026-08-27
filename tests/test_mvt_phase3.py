"""Phase 3 tests — history tracking, cross-universe comparison, relative-
strength network. Offline, synthetic data only."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from zenith.mom.mvt import crossuniverse as cu
from zenith.mom.mvt import network as net


# ----------------------------------------------------------------- history --
def test_history_append_and_series_roundtrip(tmp_path, monkeypatch):
    import zenith.mom.mvt.history as h
    monkeypatch.setattr(h, "MOM_MVT_HISTORY_DIR", tmp_path)

    rows = [{"ticker": "AAPL", "normalized_score": 5.5, "raw_score": 6.0},
           {"ticker": "MSFT", "normalized_score": -3.2, "raw_score": -1.0},
           {"ticker": "GOOG", "normalized_score": None}]  # unscored, must be excluded
    today = date(2026, 8, 26)
    added = h.append_history(rows, "equities", today)
    assert added == 2  # GOOG excluded (no normalized_score)

    series = h.series_for("AAPL", "equities", start_year=2026, end_year=2026)
    assert len(series) == 1
    assert series[0]["date"] == "2026-08-26"
    assert series[0]["rank"] == 1  # highest normalized_score
    assert series[0]["pctile"] == pytest.approx(100.0)

    msft_series = h.series_for("MSFT", "equities", start_year=2026, end_year=2026)
    assert msft_series[0]["rank"] == 2
    assert msft_series[0]["pctile"] == pytest.approx(0.0)


def test_history_idempotent_same_day(tmp_path, monkeypatch):
    import zenith.mom.mvt.history as h
    monkeypatch.setattr(h, "MOM_MVT_HISTORY_DIR", tmp_path)
    rows = [{"ticker": "AAPL", "normalized_score": 5.5}]
    today = date(2026, 8, 26)
    assert h.append_history(rows, "equities", today) == 1
    assert h.append_history(rows, "equities", today) == 0  # already recorded


def test_history_missing_ticker_series_is_empty(tmp_path, monkeypatch):
    import zenith.mom.mvt.history as h
    monkeypatch.setattr(h, "MOM_MVT_HISTORY_DIR", tmp_path)
    assert h.series_for("NOPE", "equities", start_year=2026, end_year=2026) == []


# ------------------------------------------------------------ crossuniverse --
def test_build_comparison_matches_by_sector():
    equity_rows = [{"ticker": "AAPL", "normalized_score": 10.0},
                  {"ticker": "XOM", "normalized_score": -5.0}]
    etf_rows = [{"ticker": "XLK", "normalized_score": 4.0},
               {"ticker": "XLE", "normalized_score": -1.0}]
    sector_by_ticker = {"AAPL": "Technology", "XOM": "Energy"}
    result = cu.build_comparison(equity_rows, etf_rows, sector_by_ticker)
    assert len(result) == 2
    aapl = next(r for r in result if r["ticker"] == "AAPL")
    assert aapl["sector_etf"] == "XLK"
    assert aapl["gap"] == pytest.approx(6.0)
    assert aapl["classification"] == "broad/systemic"  # gap < 8 threshold


def test_build_comparison_flags_idiosyncratic_gap():
    equity_rows = [{"ticker": "MJ_LIKE", "normalized_score": 15.0}]
    etf_rows = [{"ticker": "XLK", "normalized_score": -2.0}]
    sector_by_ticker = {"MJ_LIKE": "Technology"}
    result = cu.build_comparison(equity_rows, etf_rows, sector_by_ticker)
    assert result[0]["classification"] == "idiosyncratic"
    assert result[0]["gap"] == pytest.approx(17.0)


def test_build_comparison_skips_unmapped_sector():
    equity_rows = [{"ticker": "X", "normalized_score": 5.0}]
    etf_rows = [{"ticker": "XLK", "normalized_score": 1.0}]
    sector_by_ticker = {"X": "Some Unmapped Sector"}
    assert cu.build_comparison(equity_rows, etf_rows, sector_by_ticker) == []


def test_build_comparison_skips_when_sector_etf_not_scored():
    equity_rows = [{"ticker": "AAPL", "normalized_score": 5.0}]
    etf_rows = [{"ticker": "XLK", "normalized_score": None}]  # unscored sector ETF
    sector_by_ticker = {"AAPL": "Technology"}
    assert cu.build_comparison(equity_rows, etf_rows, sector_by_ticker) == []


def test_sector_breadth_counts_by_classification():
    comparison = [
        {"sector": "Technology", "sector_etf": "XLK", "sector_etf_score": 3.0, "classification": "broad/systemic"},
        {"sector": "Technology", "sector_etf": "XLK", "sector_etf_score": 3.0, "classification": "idiosyncratic"},
        {"sector": "Energy", "sector_etf": "XLE", "sector_etf_score": -1.0, "classification": "broad/systemic"},
    ]
    breadth = cu.sector_breadth(comparison)
    assert breadth["Technology"]["n"] == 2
    assert breadth["Technology"]["n_idiosyncratic"] == 1
    assert breadth["Technology"]["pct_idiosyncratic"] == pytest.approx(0.5)
    assert breadth["Energy"]["n_idiosyncratic"] == 0


# ------------------------------------------------------------------ network --
def _network_df(N=20, seed=0):
    rng = np.random.default_rng(seed)
    horizons = ("1m", "3m", "6m", "9m", "12m", "12_1")
    rows = []
    for i in range(N):
        rows.append({
            "ticker": f"T{i}",
            "normalized_score": float(rng.uniform(-20, 20)),
            "total_return": {h: float(rng.standard_normal() * 0.05) for h in horizons},
            "residual_return": {h: float(rng.standard_normal() * 0.03) for h in horizons},
            "total_var": float(abs(rng.standard_normal()) * 1e-4 + 1e-5),
            "resid_var": float(abs(rng.standard_normal()) * 1e-4 + 1e-5),
        })
    return pd.DataFrame(rows)


def test_build_network_shape_and_bounds():
    df = _network_df(N=20)
    tickers = df["ticker"].tolist()
    result = net.build_network(df, tickers, horizon="6m", layer="residual", edges_per_node=3)
    assert result is not None
    assert result["n"] == 20
    assert len(result["nodes"]) == 20
    assert len(result["edges"]) > 0
    # every edge references a real node
    node_tickers = {n["ticker"] for n in result["nodes"]}
    for e in result["edges"]:
        assert e["source"] in node_tickers and e["target"] in node_tickers
        assert e["source"] != e["target"]


def test_build_network_edges_are_undirected_no_duplicates():
    df = _network_df(N=15, seed=3)
    tickers = df["ticker"].tolist()
    result = net.build_network(df, tickers, horizon="3m", edges_per_node=2)
    pairs = [(e["source"], e["target"]) for e in result["edges"]]
    assert len(pairs) == len(set(pairs))  # no duplicate (source,target)
    reversed_pairs = [(b, a) for a, b in pairs]
    assert not (set(pairs) & set(reversed_pairs))  # no (a,b) AND (b,a) both present


def test_build_network_none_on_too_few_tickers():
    df = _network_df(N=2)
    assert net.build_network(df, df["ticker"].tolist()) is None


def test_force_layout_deterministic():
    weights = np.array([[0, 0.5, 0], [0.5, 0, 0.2], [0, 0.2, 0]])
    pos1 = net._force_layout(weights, iterations=30, seed=0)
    pos2 = net._force_layout(weights, iterations=30, seed=0)
    np.testing.assert_array_equal(pos1, pos2)


def test_force_layout_correlated_nodes_end_up_closer_than_unrelated():
    # nodes 0,1 strongly connected; node 2 isolated (no edges)
    weights = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
    pos = net._force_layout(weights, iterations=200, seed=1)
    d01 = np.linalg.norm(pos[0] - pos[1])
    d02 = np.linalg.norm(pos[0] - pos[2])
    assert d01 < d02
