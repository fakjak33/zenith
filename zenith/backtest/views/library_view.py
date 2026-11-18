"""BACKTEST / LIBRARY — every saved strategy, side by side, plus the research ledger."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ... import ui_plotly as up
from ...config import THEME as T
from ...ui_theme import section
from .. import library
from ..engine import metrics
from . import common as C
from .portfolio_view import _component


def render() -> None:
    store = C.saved()
    st.markdown(section("SAVED STRATEGIES", 0), unsafe_allow_html=True)
    if not store:
        C.banner(T.amber, "EMPTY", "save strategies from LAB › BUILD (the SAVE box) — they appear here")
    else:
        rows = []
        for nm, rec in store.items():
            s = library.spec_of(rec)
            rows.append({"name": nm, "rule": s.strategy().describe(), "market": s.name, "bars": s.tf,
                         "filters": ", ".join(f["key"] for f in s.filters) or "—", "holdout": f"{s.holdout}%",
                         "hash": rec["hash"], "saved": rec["saved_at"], "hypothesis": (rec.get("note") or "")[:80]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        c1, c2 = st.columns([3, 1])
        target = c1.selectbox("open a saved strategy in LAB", list(store), key="lib_open_pick")
        if c2.button("OPEN IN LAB", key="lib_open", use_container_width=True):
            st.session_state["bt_pending"] = C.spec_to_widgets(library.spec_of(store[target]))
            from ... import nav
            st.switch_page(nav.PAGES["backtest-lab"])

        st.markdown(section("COMPARE", 1), unsafe_allow_html=True)
        pick = st.multiselect("strategies", list(store), default=list(store)[:4], key="lib_cmp")
        if pick:
            series, stats = {}, {}
            for nm in pick:
                try:
                    r, p = _component(library.spec_of(store[nm]).to_json())
                    series[nm], stats[nm] = r, metrics.summary(r, periods=p)
                except Exception as exc:  # noqa: BLE001
                    C.banner(T.coral, "SKIPPED", f"{nm}: {exc}")
            fig = go.Figure()
            for i, (nm, r) in enumerate(series.items()):
                eq = (1 + r).cumprod()
                fig.add_trace(go.Scatter(x=eq.index, y=eq, name=nm, line=dict(color=T.cat[i % len(T.cat)], width=1.4)))
            fig.update_yaxes(type="log", title_text="growth of 1")
            up.show(fig, 420, "in-sample equity, each on its own data", key="lib_eq")
            keys = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "Skew", "PSR", "Turnover"]
            st.dataframe(pd.DataFrame({nm: [C.fmt(k, s.get(k, float("nan"))) for k in keys] for nm, s in stats.items()},
                                      index=keys), use_container_width=True)

        st.markdown(section("EXPORT / IMPORT", 2), unsafe_allow_html=True)
        st.download_button("DOWNLOAD ALL AS JSON", json.dumps(list(store.values()), indent=1),
                           file_name="zenith_backtest_specs.json", mime="application/json", key="lib_dl")
    up_file = st.file_uploader("import strategies (JSON from DOWNLOAD)", type="json", key="lib_ul")
    if up_file is not None and st.button("IMPORT", key="lib_imp"):
        try:
            for rec in json.loads(up_file.getvalue().decode("utf-8")):
                library.save(store, rec["name"], library.spec_of(rec), rec.get("note", ""))
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            C.banner(T.coral, "BAD FILE", str(exc))

    led = C.ledger()
    st.markdown(section("RESEARCH LEDGER — trials and reveals", 3), unsafe_allow_html=True)
    tr = [{"data + holdout": fam, "trials": len(h), "revealed": "yes" if fam in led["reveals"] else "—"}
          for fam, h in led["trials"].items()]
    if tr:
        st.dataframe(pd.DataFrame(tr).sort_values("trials", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.caption("No trials recorded yet this session.")
    if led["reveals"]:
        st.dataframe(pd.DataFrame([{"data + holdout": f, "revealed at": r["at"], "spec hash": r["hash"],
                                    "trials before": r["trials_at_reveal"]} for f, r in led["reveals"].items()]),
                     use_container_width=True, hide_index=True)
    st.caption("Trials are every distinct strategy variant run on a dataset + holdout, including sweep cells and "
               "ablation variants. They feed the deflated Sharpe. Locally they persist in "
               "data/backtest/runs/ledger.json; on the cloud they last for the session.")
