import numpy as np
import pandas as pd

from zenith.backtest.etflab import portfolio as pf
from zenith.backtest.etflab import factors, weighting, screens, economics, overlap, presets


def _panel(n=520, start="2018-01-01", tickers=("A", "B", "C")):
    idx = pd.bdate_range(start, periods=n)
    cols = {}
    for i, t in enumerate(tickers):
        # distinct deterministic ramps so rankings are well-defined
        cols[t] = pd.Series(np.linspace(100, 100 + 40 * (i + 1), n), index=idx)
    return pd.DataFrame(cols)


def _const_panel(n=300, tickers=("A", "B")):
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({t: pd.Series(100.0, index=idx) for t in tickers})


# --- rebalance scheduling ---------------------------------------------------
def test_rebalance_dates_counts():
    idx = pd.bdate_range("2020-01-01", "2022-12-31")
    assert len(pf.rebalance_dates(idx, "Daily")) == len(idx)
    assert len(pf.rebalance_dates(idx, "Annual")) == 3
    assert len(pf.rebalance_dates(idx, "Quarterly")) == 12
    assert len(pf.rebalance_dates(idx, "Biannual")) == 6
    # monthly ~ 36
    assert 35 <= len(pf.rebalance_dates(idx, "Monthly")) <= 37


# --- portfolio engine -------------------------------------------------------
def test_single_asset_equals_buy_and_hold():
    panel = _panel(tickers=("A", "B"))
    ws = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[panel.index[0]])
    res = pf.run_portfolio(panel, ws, pf.PortfolioConfig(cost_bps=0, slippage_bps=0))
    a_ret = panel["A"].pct_change().fillna(0.0)
    # after the first (effective-next-bar) rebalance, returns track asset A
    assert np.allclose(res.returns.iloc[1:].to_numpy(), a_ret.iloc[1:].to_numpy(), atol=1e-9)


def test_empty_schedule_is_flat():
    panel = _panel()
    res = pf.run_portfolio(panel, pd.DataFrame(columns=panel.columns), pf.PortfolioConfig())
    assert np.allclose(res.equity.to_numpy(), res.equity.iloc[0])


def test_transaction_cost_charged_once():
    panel = _const_panel()
    ws = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[panel.index[0]])
    cfg = pf.PortfolioConfig(cost_bps=10.0, slippage_bps=0.0)  # 10bps on turnover 1.0
    res = pf.run_portfolio(panel, ws, cfg)
    # constant prices -> only the one-time rebalance cost of 10bps shows up
    assert abs(res.equity.iloc[-1] / cfg.capital - (1 - 0.0010)) < 1e-9


def test_expense_ratio_drag():
    panel = _const_panel()
    ws = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[panel.index[0]])
    cfg = pf.PortfolioConfig(cost_bps=0, slippage_bps=0, expense_ratio_annual=0.01)
    res = pf.run_portfolio(panel, ws, cfg)
    n = len(panel)
    expected = (1 - 0.01 / 252) ** n   # 252 trading days (ZENITH convention)
    assert abs(res.equity.iloc[-1] / cfg.capital - expected) < 1e-6


def test_long_short_dollar_neutral_and_gross():
    panel = _panel(tickers=("A", "B", "C", "D"))
    spec = screens.ETFSpec(name="ls", selection=screens.SelectionSpec(
        universe=list(panel.columns), rank_factor="trailing_return", rank_lookback=60,
        top_n=1, bottom_n=1), direction="long_short", weighting="equal")
    rdts = pf.rebalance_dates(panel.index, "Monthly")
    ws = screens.build_weight_schedule(spec, panel, rdts)
    row = ws.iloc[-1]
    assert abs(row.sum()) < 1e-9              # dollar-neutral
    assert abs(row.abs().sum() - 2.0) < 1e-9  # gross 2.0 (100/100)


# --- weighting --------------------------------------------------------------
def test_equal_weight_sums_to_one():
    panel = _panel()
    w = weighting.compute("equal", ["A", "B", "C"], panel)
    assert abs(sum(w.values()) - 1.0) < 1e-9 and len(w) == 3


def test_max_weight_cap():
    panel = _panel(tickers=("A", "B", "C", "D"))
    w = weighting.compute("equal", list(panel.columns), panel, max_weight=0.3)
    assert max(w.values()) <= 0.3 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_inverse_vol_favors_low_vol():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    calm = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 300)), index=idx)
    wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.05, 300)), index=idx)
    panel = pd.DataFrame({"CALM": calm, "WILD": wild})
    w = weighting.compute("inverse_vol", ["CALM", "WILD"], panel)
    assert w["CALM"] > w["WILD"]


# --- factor point-in-time discipline ----------------------------------------
def test_factor_is_point_in_time():
    panel = _panel(n=400, tickers=("A", "B", "C"))
    asof = panel.index[200]
    base = factors.trailing_return(panel, asof, lookback=100)
    # spike a price AFTER asof — must not change the as-of factor value
    panel2 = panel.copy()
    panel2.iloc[300:, panel2.columns.get_loc("A")] *= 5
    after = factors.trailing_return(panel2, asof, lookback=100)
    assert np.isclose(base["A"], after["A"])


# --- economics --------------------------------------------------------------
def test_economics_closed_form():
    cfg = economics.FundEconConfig(aum=100e6, expense_ratio=0.005,
        expected_aum_growth=0.0, startup_cost=0.0, annual_maintenance=0.0,
        index_licensing_bps=0.0, rebalance_trading_bps=0.0)
    df = economics.economics(cfg, years=3)
    assert abs(df.loc[1, "Revenue"] - 100e6 * 0.005) < 1e-6
    assert abs(df.loc[1, "NetProfit"] - 100e6 * 0.005) < 1e-6  # no costs


def test_breakeven_aum():
    cfg = economics.FundEconConfig(expense_ratio=0.005, annual_maintenance=500_000,
        index_licensing_bps=0.0, rebalance_trading_bps=0.0)
    assert abs(economics.breakeven_aum(cfg) - 500_000 / 0.005) < 1.0


def test_investor_fee_drag_positive():
    d = economics.investor_fee_drag(0.08, 0.005, years=10)
    assert d["net_return_annual"] == 0.075 and d["fee_cost"] > 0


# --- overlap ----------------------------------------------------------------
def test_overlap_edges():
    assert overlap.weight_overlap({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}) == 1.0
    assert overlap.weight_overlap({"A": 1.0}, {"B": 1.0}) == 0.0
    assert overlap.jaccard(["A", "B"], ["B", "C"]) == 1 / 3


# --- presets + spec ---------------------------------------------------------
def test_preset_lookahead_matches_phase():
    # Phase-1 presets are price-derived (never look-ahead); Phase-2 presets all
    # touch the fundamental snapshot, so they must flag look-ahead.
    for p in presets.PRESETS:
        if p.phase == 1:
            assert screens.spec_is_lookahead(p) is False, p.name
        else:
            assert screens.spec_is_lookahead(p) is True, p.name


# --- Phase 2: fundamental filters / ranking ---------------------------------
def _patch_snapshot(monkeypatch, table):
    """Stub fundamentals.snapshot with a fixed in-memory table (tickers x fields)."""
    df = pd.DataFrame(table).T
    df["fetched_at"] = 0.0

    def fake_snapshot(tickers, max_age_days=30, fields=None):
        idx = [t.upper() for t in tickers]
        cols = (list(fields) if fields else [c for c in df.columns if c != "fetched_at"])
        return df.reindex(index=idx, columns=cols + ["fetched_at"])

    monkeypatch.setattr(screens.fundamentals, "snapshot", fake_snapshot)


def test_numeric_filter_applies(monkeypatch):
    panel = _panel(tickers=("A", "B", "C"))
    _patch_snapshot(monkeypatch, {
        "A": {"fcf_per_share": 1.0}, "B": {"fcf_per_share": 5.0},
        "C": {"fcf_per_share": 2.0}})
    spec = screens.ETFSpec(name="fcf", selection=screens.SelectionSpec(
        universe=list(panel.columns),
        filters=[screens.FilterRule("fcf_per_share", "<=", 3.0)]))
    longs, shorts = screens.selected_at(spec, panel, panel.index[-1])
    assert set(longs) == {"A", "C"} and shorts == []


def test_string_filter_contains(monkeypatch):
    panel = _panel(tickers=("A", "B", "C"))
    _patch_snapshot(monkeypatch, {
        "A": {"industry": "Insurance—Life"}, "B": {"industry": "Software"},
        "C": {"industry": "Property & Casualty Insurance"}})
    spec = screens.ETFSpec(name="ins", selection=screens.SelectionSpec(
        universe=list(panel.columns),
        filters=[screens.FilterRule("industry", "contains", "insurance")]))
    longs, _ = screens.selected_at(spec, panel, panel.index[-1])
    assert set(longs) == {"A", "C"}


def test_fundamental_rank_value_tilt(monkeypatch):
    panel = _panel(tickers=("A", "B", "C"))
    _patch_snapshot(monkeypatch, {
        "A": {"trailing_pe": 30.0}, "B": {"trailing_pe": 8.0},
        "C": {"trailing_pe": 15.0}})
    spec = screens.ETFSpec(name="val", selection=screens.SelectionSpec(
        universe=list(panel.columns), rank_factor="low_pe", top_n=1))
    longs, _ = screens.selected_at(spec, panel, panel.index[-1])
    assert longs == ["B"]                # lowest P/E ranks first for "low_pe"
    assert screens.spec_is_lookahead(spec) is True


def test_separate_short_leg_factor(monkeypatch):
    panel = _panel(tickers=("A", "B", "C", "D"))
    _patch_snapshot(monkeypatch, {
        t: {"profit_margin": m} for t, m in
        zip("ABCD", [0.30, 0.05, 0.20, 0.10])})
    # long the highest trailing return, short the lowest profit margin
    spec = screens.ETFSpec(name="ex5", selection=screens.SelectionSpec(
        universe=list(panel.columns), rank_factor="trailing_return",
        short_rank_factor="profit_margin", rank_lookback=60, top_n=1, bottom_n=1),
        direction="long_short")
    longs, shorts = screens.selected_at(spec, panel, panel.index[-1])
    assert shorts == ["B"]              # B has the lowest margin
    assert longs and longs[0] not in shorts


def test_fcf_weighting(monkeypatch):
    panel = _panel(tickers=("A", "B"))

    def fake_series(tickers, field, max_age_days=30):
        return pd.Series({"A": 3.0, "B": 1.0}).reindex([t.upper() for t in tickers])

    from zenith.backtest.etflab import fundamentals as fund
    monkeypatch.setattr(fund, "factor_series", fake_series)
    w = weighting.compute("fcf_weighted", ["A", "B"], panel)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["A"] > w["B"]              # higher FCF → higher weight


def test_explicit_basket_equal_weights():
    panel = _panel(tickers=("A", "B", "C", "D"))
    spec = screens.ETFSpec(name="x", selection=screens.SelectionSpec(
        explicit=["A", "B", "C", "D"]), weighting="equal", rebalance="Quarterly")
    rdts = pf.rebalance_dates(panel.index, "Quarterly")
    ws = screens.build_weight_schedule(spec, panel, rdts)
    assert abs(ws.iloc[-1].sum() - 1.0) < 1e-9
    assert np.allclose(ws.iloc[-1].replace(0, np.nan).dropna().to_numpy(), 0.25)
