"""RESEARCH / FUNDLENS — deterministic, committee-ready fund comparisons.

Ported from the standalone FundLens app. Upload Morningstar Advisor Workstation
exports (licensed data: they are parsed in a temp folder that is deleted at once
and never stored), set priorities, explore the ranking and export colour-scaled
comparison workbooks. ``demo.py`` provides a synthetic export for trying it out.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from ..config import THEME as T
from ..ui_kit import rgba
from . import __version__
from .demo import demo_files
from .dictionary import SCORING_HORIZONS, add_user_metric, load_dictionary, norm_label
from .engine import run
from .ingest import load_dataset
from .profile import PROFILE_DIR, list_profiles, load_profile, to_yaml
from .quality import QualityLog
from .render_xlsx import NEUTRAL, PALETTES, _interp, output_name, write_workbook
from .results import results_dict  # noqa: F401  (kept for parity with the CLI / skill)

# Optional metrics offered per pillar in the Priorities editor (scored only if present in the data).
CANDIDATES = {
    "returns": ["nav_ret", "yield_12m"],
    "risk_adjusted": ["sharpe", "sortino", "capture_spread", "alpha", "info_ratio", "treynor", "best_fit_alpha"],
    "downside": ["down_capture", "std_dev", "bear_market_pct", "worst_year_5y", "worst_3m_eligible", "kurtosis", "skew"],
    "consistency": ["batting_avg", "top_half_pct"],
    "cost": ["net_er", "aum", "avg_daily_vol", "abs_prem_disc", "turnover", "index_te", "manager_tenure", "star_rating"],
    "fit": ["beta", "r_squared", "te_ratio", "corr_est"],
    "stewardship": ["med_mgr_ir_rk", "med_success_rk", "med_mgr_exp_rk", "med_brand_retention_rk", "med_brand_tenure_rk",
                    "med_parent_success", "med_obsolete", "med_fee_rank", "med_pas_te_rk", "med_ap_ir_rk",
                    "med_mgr_ownership", "med_brand_ir_rk", "med_proc_ir_rk", "med_fee_rk"],
}


# --------------------------------------------------------------------------- computation
def _save_uploads(files: list[tuple[str, bytes]], folder: str) -> list[Path]:
    out = []
    for n, b in files:
        p = Path(folder) / Path(n).name
        p.write_bytes(b)
        out.append(p)
    return out


@st.cache_resource(max_entries=12, show_spinner="scoring funds…")
def compute(data: tuple, approved: tuple | None, prof_json: str, shortlist: tuple, require_sol: bool):
    with tempfile.TemporaryDirectory(prefix="fundlens_") as tmp:      # uploads deleted as soon as they are parsed
        paths = _save_uploads(list(data), tmp)
        ap = _save_uploads([approved], tmp)[0] if approved else None
        return run(paths, profile=json.loads(prof_json), approved_path=ap, shortlist=list(shortlist),
                   approved_require_solicitation=require_sol)


@st.cache_resource(max_entries=12)
def inspect(data: tuple):
    log = QualityLog()
    with tempfile.TemporaryDirectory(prefix="fundlens_") as tmp:
        ds = load_dataset(_save_uploads(list(data), tmp), log, load_dictionary())
    return ds, log


# --------------------------------------------------------------------------- styling helpers
def heat_css(v, palette="standard", neutral=False):
    if v is None or pd.isna(v):
        return f"background-color:{T.raised};color:{T.faint}"
    stops = NEUTRAL if neutral else PALETTES[palette]
    return f"background-color:#{_interp(stops, float(v))};color:#000"


def style_ranking(df: pd.DataFrame, score_cols: list[str], palette: str):
    sty = df.style
    for c in score_cols:
        if c in df:
            sty = sty.map(lambda v: heat_css(v, palette), subset=[c])
    if "Coverage" in df:
        sty = sty.map(lambda v: heat_css(None if pd.isna(v) else v * 100, palette), subset=["Coverage"])
    fmt = {c: "{:.1f}" for c in score_cols if c in df}
    fmt.update({"Coverage": "{:.0%}", "Rank": "{:.0f}"})
    return sty.format(fmt, na_rep="—")


def recency_flags(c) -> pd.Series:
    if "recency_skew" not in c.df:
        return pd.Series("", index=c.df.index)
    thr = float(c.prof["recency"]["surge_flag"])
    return c.df["recency_skew"].map(lambda v: "Recent surge" if pd.notna(v) and v >= thr else
                                    ("Recent slump" if pd.notna(v) and v <= -thr else ""))


def render() -> None:
    # --------------------------------------------------------------------------- inputs
    with st.container(border=True):
        st.caption(f"Deterministic fund comparison workbooks · v{__version__} · files are parsed in a temp folder and "
                   "deleted immediately — nothing you upload is stored or sent anywhere.")
        i1, i2, i3 = st.columns([1.3, 1, 1])
        up_data = i1.file_uploader("Morningstar export(s)", type=["xlsx", "xls", "xlsm", "csv"], accept_multiple_files=True,
                                   help="Upload every data-set file for the list at once (Basics, Returns, Risk, Annual Returns, "
                                        "Sectors, Equity Portfolio, Medalist...) - they are merged by ticker. "
                                        "Funds in rows or columns, stacked blocks and several files are all handled.")
        up_appr = i1.file_uploader("Approved funds list (optional)", type=["xlsx", "xls", "xlsm", "csv"],
                                   help="Any sheet with a Ticker column. Footer and disclaimer rows are ignored.")
        use_sample = False
        if not up_data:
            use_sample = i2.toggle("Try it with synthetic demo data", value=True, key="fl_demo",
                                   help="A seeded, made-up Mid-Cap Value ETF export and approved list — no real funds.")
        profs = list_profiles()
        names = {p["id"]: p["name"] for p in profs}
        ids = list(names)
        pid = i2.selectbox("Profile", ids, index=ids.index("us_equity_style") if "us_equity_style" in ids else 0,
                           format_func=lambda k: names[k], key="fl_profile_id",
                           help="Preset weights and directions for the sleeve you are comparing. Adjust on the Priorities tab.")
        i2.caption(next(p["description"] for p in profs if p["id"] == pid))
        up_prof = i3.file_uploader("...or load a saved profile (.yaml)", type=["yaml", "yml"])
        palette = i3.radio("Colour palette", ["standard", "colorblind"], horizontal=True, key="fl_palette",
                           format_func=lambda x: "Red-Yellow-Green" if x == "standard" else "Colour-blind safe")

    if up_data:
        data_files = tuple((f.name, f.getvalue()) for f in up_data)
    elif use_sample:
        demo_data, demo_appr = demo_files()
        data_files = (demo_data,)
    else:
        data_files = ()
    appr_file = (up_appr.name, up_appr.getvalue()) if up_appr else (demo_appr if use_sample else None)

    if not data_files:
        st.info("Upload one or more Morningstar Advisor Workstation exports above to begin, or switch on the demo data.")
        return
    if use_sample:
        st.caption(":orange[DEMO DATA — synthetic funds, for exploring the tool only.]")

    # --------------------------------------------------------------------------- profile state
    base = load_profile(yaml.safe_load(up_prof.getvalue()) if up_prof else pid)
    base_key = f"{base['id']}:{hashlib.sha1(json.dumps(base, sort_keys=True).encode()).hexdigest()[:8]}"
    if st.session_state.get("fl_base_key") != base_key:
        st.session_state["fl_base_key"] = base_key
        st.session_state["fl_prof"] = copy.deepcopy(base)
        st.session_state["fl_shortlist"] = []
    prof = st.session_state["fl_prof"]
    d = load_dictionary()
    ds, ds_log = inspect(data_files)
    present = set(ds.df.columns)


    def metric_available(mid: str) -> bool:
        m = d.by_id.get(mid)
        if m is None:
            return False
        if m.derived:
            return True
        if m.horizons:
            return any(f"{mid}_{h}" in present for h in SCORING_HORIZONS)
        return mid in present


    tab_pri, tab_res, tab_data, tab_exp = st.tabs(["1 · Priorities", "2 · Results", "3 · Data check", "4 · Export"])

    # =========================================================================== 1. Priorities
    with tab_pri:
        k = "fl_" + base_key
        st.subheader("What matters for this comparison?")
        st.caption("Pillar weights are relative (they are normalised). Inside each pillar, tick the metrics to score "
                   "and set their relative weight. Everything here is saved in the workbook's Methodology tab.")
        cols = st.columns(len(prof["pillars"]))
        for c, (pk, p) in zip(cols, prof["pillars"].items()):
            with c:
                p["weight"] = float(st.slider(p["label"], 0, 50, int(round(p["weight"])), key=f"{k}:pw:{pk}"))
        tot = sum(p["weight"] for p in prof["pillars"].values()) or 1
        st.caption("Normalised: " + " · ".join(f"{p['label']} {p['weight'] / tot:.0%}" for p in prof["pillars"].values()))

        st.markdown("##### Metrics inside each pillar")
        user_ids = [m.id for m in d.metrics if m.id not in {x for v in CANDIDATES.values() for x in v}
                    and m.kind == "directional" and metric_available(m.id)]
        pcols = st.columns(3)
        for i, (pk, p) in enumerate(prof["pillars"].items()):
            with pcols[i % 3].expander(f"{p['label']}  ({len(p['metrics'])} scored)", expanded=False):
                cands = list(dict.fromkeys(list(p["metrics"]) + CANDIDATES.get(pk, [])))
                rows = [{"Metric": d.by_id[m].label if m in d.by_id else m, "id": m, "Score it": m in p["metrics"],
                         "Weight": p["metrics"].get(m, 1.0), "In data": metric_available(m)} for m in cands]
                extra = st.multiselect("Add another metric", [u for u in user_ids if u not in cands],
                                       format_func=lambda x: d.by_id[x].label, key=f"{k}:extra:{pk}")
                rows += [{"Metric": d.by_id[m].label, "id": m, "Score it": True, "Weight": 1.0, "In data": True} for m in extra]
                ed = st.data_editor(pd.DataFrame(rows), key=f"{k}:ed:{pk}", hide_index=True, width="stretch",
                                    column_config={"id": None,
                                                   "Weight": st.column_config.NumberColumn(min_value=0.0, max_value=5.0, step=0.25),
                                                   "In data": st.column_config.CheckboxColumn(disabled=True)},
                                    disabled=["Metric", "In data"])
                p["metrics"] = {r["id"]: float(r["Weight"]) for _, r in ed.iterrows() if r["Score it"] and r["Weight"] > 0}

        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            st.markdown("##### Horizon emphasis")
            st.caption("How the 3/5/10/15-year versions of a metric are blended (renormalised when a fund lacks a horizon).")
            hc = st.columns(4)
            for col, h in zip(hc, SCORING_HORIZONS):
                prof["horizon_weights"][h] = col.number_input(h.upper(), 0.0, 1.0, float(prof["horizon_weights"][h]), 0.05,
                                                              key=f"{k}:hw:{h}")
        with c2:
            st.markdown("##### Beta & correlation preference")
            ov = prof["overrides"]
            b = ov.get("beta") or {"direction": "target", "target": 1.0}
            opts = ["target", "lower", "higher", "none"]
            bl = {"target": "Closest to a target", "lower": "Lower is better", "higher": "Higher is better", "none": "Ignore (neutral)"}
            bd = st.radio("Beta", opts, index=opts.index(b.get("direction", "target")), format_func=bl.get, horizontal=True,
                          key=f"{k}:beta")
            bt = st.number_input("Beta target", -1.0, 2.0, float(b.get("target", 1.0) or 0.0), 0.05, key=f"{k}:bt",
                                 disabled=bd != "target")
            ov["beta"] = {"direction": bd, "target": bt} if bd == "target" else {"direction": bd}
            r2 = (ov.get("r_squared") or {}).get("direction", "higher")
            r2d = st.radio("R-squared / correlation to index", ["higher", "lower", "none"], index=["higher", "lower", "none"].index(r2),
                           format_func={"higher": "Higher (want beta exposure)", "lower": "Lower (want diversification)",
                                        "none": "Ignore"}.get, horizontal=True, key=f"{k}:r2")
            ov["r_squared"] = {"direction": r2d}
            ov["corr_est"] = {"direction": r2d}
        with c3:
            st.markdown("##### Screens")
            scr = prof["screens"]
            v = st.number_input("Min AUM ($MM)", 0.0, value=float(scr.get("min_aum_mm") or 0.0), step=25.0, key=f"{k}:aum")
            scr["min_aum_mm"] = v or None
            v = st.number_input("Max net expense (%)", 0.0, 5.0, float(scr.get("max_net_er") or 0.0), 0.05, key=f"{k}:er",
                                help="0 = no limit")
            scr["max_net_er"] = v or None
            v = st.selectbox("Min track record", [None, 3, 5, 10], index=[None, 3, 5, 10].index(scr.get("min_history_years")),
                             format_func=lambda x: "None" if x is None else f"{x} years", key=f"{k}:hist")
            scr["min_history_years"] = v
            scr["exclude_leveraged"] = st.checkbox("Exclude leveraged funds", bool(scr.get("exclude_leveraged", True)),
                                                   key=f"{k}:lev")
            objs = sorted(ds.df["objective"].dropna().unique()) if "objective" in ds.df else []
            scr["exclude_objectives"] = st.multiselect("Exclude objectives", objs,
                                                       [o for o in scr.get("exclude_objectives", []) if o in objs], key=f"{k}:obj")

        st.markdown("##### Recency bias")
        rc = prof.setdefault("recency", {"mode": "off", "surge_flag": 35})
        r1c, r2c = st.columns([3, 1])
        rc["mode"] = r1c.radio(
            "How much should the latest 12 months be allowed to drive the ranking?", ["off", "balanced", "strict"],
            index=["off", "balanced", "strict"].index(rc.get("mode", "off")), horizontal=True, key=f"{k}:rec",
            format_func={"off": "Off - standard trailing returns",
                         "balanced": "Balanced - blend in ex-latest-year returns + penalise one-year surges",
                         "strict": "Strict - ex-latest-year returns only"}.get,
            help="Use when one style (e.g. momentum) had an outsized year and is lifting funds in every category. "
                 "The Recency Check tab in the workbook shows each fund's rank with this off.")
        rc["surge_flag"] = r2c.slider("Flag 'Recent surge' at (pct pts)", 20, 60, int(rc.get("surge_flag", 35)), 5, key=f"{k}:sf")
        t1, t2 = st.columns(2)
        rc["theme_gap"] = t1.slider("Theme Scan: flag groups whose median 1Y return beats the comparison by (pts)", 2.0, 15.0,
                                    float(rc.get("theme_gap", 5.0)), 0.5, key=f"{k}:tg")
        rc["theme_skew"] = t2.slider("...and whose median recency skew is at least (pct pts)", 5.0, 40.0,
                                     float(rc.get("theme_skew", 15.0)), 2.5, key=f"{k}:ts")

        with st.expander("Advanced scoring settings"):
            sc = prof["scoring"]
            a1, a2, a3, a4 = st.columns(4)
            sc["shrinkage_k"] = a1.slider("Short-history shrinkage strength (k)", 0.0, 2.0, float(sc["shrinkage_k"]), 0.1,
                                          key=f"{k}:k", help="Adjusted = 50 + (Raw-50) x Coverage^k. 0 = no shrinkage.")
            sc["breadth_threshold"] = a2.slider("Breadth threshold", 50, 80, int(sc["breadth_threshold"]), 5, key=f"{k}:br")
            sc["min_funds_per_metric"] = a3.number_input("Min funds per metric", 2, 20, int(sc["min_funds_per_metric"]),
                                                         key=f"{k}:mn")
            sc["stress_year"] = a4.number_input("Crisis-tested year (worst 3M)", 2000, 2030, int(sc["stress_year"]), key=f"{k}:sy")
            prof["relative_metrics_basis"] = st.radio(
                "Risk statistics in the export are computed against...", ["standard_index", "prospectus"],
                index=["standard_index", "prospectus"].index(prof.get("relative_metrics_basis", "standard_index")),
                format_func={"standard_index": "Morningstar's standard index for the category (default)",
                             "prospectus": "Each fund's prospectus benchmark"}.get, horizontal=True, key=f"{k}:basis")

        st.markdown("##### Save these settings")
        s1, s2, s3 = st.columns([2, 1, 1])
        new_name = s1.text_input("Profile name", value=prof["name"] if prof["id"] == base["id"] else prof["name"], key=f"{k}:nm")
        out_prof = copy.deepcopy(prof)
        out_prof["name"] = new_name
        out_prof["id"] = re.sub(r"[^a-z0-9]+", "_", new_name.lower()).strip("_")[:40] or "custom"
        if s2.button("Save as profile", width="stretch"):
            (PROFILE_DIR / f"{out_prof['id']}.yaml").write_text(to_yaml(out_prof), encoding="utf-8")
            st.success(f"Saved profiles/{out_prof['id']}.yaml - it now appears in the Profile list "
                       "(on the hosted app this lasts until the next restart; download the .yaml to keep it).")
        s3.download_button("Download .yaml", to_yaml(out_prof), f"{out_prof['id']}.yaml", width="stretch")
        prof_for_run = out_prof

    # =========================================================================== compute
    require_sol = False
    comps = compute(data_files, appr_file, json.dumps(prof_for_run, default=str),  # keep pillar order
                    tuple(st.session_state["fl_shortlist"]), require_sol)
    by_kind = {c.kind: c for c in comps}

    # =========================================================================== 2. Results
    with tab_res:
        kinds = list(by_kind)
        kind = st.radio("Universe", kinds, horizontal=True,
                        format_func={"full": "All funds", "approved": "Approved only (re-ranked)"}.get) if len(kinds) > 1 else kinds[0]
        c = by_kind[kind]
        s, dg = c.scores, c.diag
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Funds ranked", int(s.rank.notna().sum()))
        m2.metric("Screened out", len(c.excluded))
        m3.metric("Tier 1", int((s.tier == "Tier 1").sum()))
        m4.metric("Robust leaders", int(dg.robust.sum()))
        m5.metric("Returns as of", str(c.asof))

        pl = [s.pillar_labels[p] for p in s.pillar_weights]
        tab = pd.DataFrame({
            "Rank": s.rank, "Tier": s.tier, "Ticker": c.df.index, "Fund": c.df["name"].fillna("(name not captured)"),
            **({"Approved": c.approved_flag.map({True: "Yes", False: ""})} if c.approved_flag is not None and kind == "full" else {}),
            "Composite": s.adjusted, "Raw": s.raw, "Coverage": s.coverage,
            **{s.pillar_labels[p]: s.pillar_scores[p] for p in s.pillar_weights},
            "Breadth": dg.breadth, "Robust": dg.robust.map({True: "Yes", False: ""}), "Weakest": dg.weakest,
            "Recency": recency_flags(c),
            "Hot themes": pd.Series({t: ", ".join(c.recency.get("fund_hot", {}).get(t, [])) for t in c.df.index}),
            "Rank (recency off)": c.recency.get("rank_off"),
            "Objective": c.df.get("objective"),
        }).loc[s.order]
        f1, f2 = st.columns([1, 3])
        tiers = f1.multiselect("Tier filter", sorted(tab["Tier"].unique()), key="fl_tierf")
        q = f2.text_input("Search ticker / name", key="fl_q")
        view = tab
        if tiers:
            view = view[view["Tier"].isin(tiers)]
        if q:
            view = view[view["Ticker"].str.contains(q, case=False) | view["Fund"].astype(str).str.contains(q, case=False)]
        st.dataframe(style_ranking(view, ["Composite", "Raw"] + pl, palette), width="stretch", hide_index=True,
                     height=min(38 * (len(view) + 1), 620))

        st.markdown("##### Shortlist")
        sl = st.multiselect("Funds to carry into a Shortlist tab in the export", list(tab["Ticker"]),
                            default=[t for t in st.session_state["fl_shortlist"] if t in set(tab["Ticker"])],
                            key=f"fl_sl:{kind}")
        if sorted(sl) != sorted(st.session_state["fl_shortlist"]):
            st.session_state["fl_shortlist"] = sl
            st.rerun()

        st.markdown("##### Visualise")
        v1, v2 = st.columns([1.3, 1])
        with v1:
            num_cols = [x for x in c.df.columns if any(x.startswith(p) for p in
                        ("nav_ret_", "sharpe_", "sortino_", "std_dev_", "down_capture_", "up_capture_", "alpha_",
                         "beta_", "r_squared_", "capture_spread_")) and c.df[x].notna().sum() >= 5]
            default_x = next((x for x in ("std_dev_5y", "std_dev_3y") if x in num_cols), num_cols[0])
            default_y = next((x for x in ("nav_ret_5y", "nav_ret_3y") if x in num_cols), num_cols[0])
            xa = st.selectbox("X axis", num_cols, num_cols.index(default_x), format_func=d.column_label, key="fl_xa")
            ya = st.selectbox("Y axis", num_cols, num_cols.index(default_y), format_func=d.column_label, key="fl_ya")
            pdf = pd.DataFrame({"Ticker": c.df.index, "Fund": c.df["name"], "x": pd.to_numeric(c.df[xa], errors="coerce"),
                                "y": pd.to_numeric(c.df[ya], errors="coerce"),
                                "AUM": pd.to_numeric(c.df.get("aum"), errors="coerce"),
                                "Composite": s.adjusted, "Rank": s.rank, "Tier": s.tier}).dropna(subset=["x", "y"])
            base_c = alt.Chart(pdf).encode(
                x=alt.X("x:Q", title=d.column_label(xa), scale=alt.Scale(zero=False)),
                y=alt.Y("y:Q", title=d.column_label(ya), scale=alt.Scale(zero=False)),
                tooltip=["Ticker", "Fund", alt.Tooltip("x:Q", title=d.column_label(xa), format=".2f"),
                         alt.Tooltip("y:Q", title=d.column_label(ya), format=".2f"),
                         alt.Tooltip("Composite:Q", format=".1f"), "Rank:Q", "Tier", alt.Tooltip("AUM:Q", format=",.0f")])
            pts = base_c.mark_circle(opacity=0.85, stroke=T.line, strokeWidth=0.4).encode(
                size=alt.Size("AUM:Q", scale=alt.Scale(range=[30, 900]), legend=None),
                color=alt.Color("Composite:Q", scale=alt.Scale(domain=[20, 50, 80], range=[T.coral, T.amber, T.teal]),
                                legend=alt.Legend(title="Composite")))
            lbl = base_c.transform_filter(alt.datum.Rank <= 10).mark_text(dx=9, dy=-7, fontSize=10, color=T.text).encode(text="Ticker")
            st.altair_chart((pts + lbl).properties(height=430), width="stretch")
        with v2:
            top3 = [t for t in s.order if pd.notna(s.rank[t])][:3]
            pick = st.multiselect("Compare pillar profiles", list(tab["Ticker"]), default=sl[:5] or top3, max_selections=6,
                                  key=f"fl_pp:{kind}")
            if pick:
                long = pd.DataFrame([{"Ticker": t, "Pillar": s.pillar_labels[p], "Score": s.pillar_scores.at[t, p]}
                                     for t in pick for p in s.pillar_weights]).dropna()
                ch = alt.Chart(long).mark_bar().encode(
                    y=alt.Y("Pillar:N", sort=pl, title=None), x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100])),
                    yOffset="Ticker:N", color=alt.Color("Ticker:N", scale=alt.Scale(range=list(T.cat))),
                    tooltip=["Ticker", "Pillar", alt.Tooltip("Score:Q", format=".1f")])
                rule = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 3], color=T.muted).encode(x="x:Q")
                st.altair_chart((ch + rule).properties(height=430), width="stretch")

        th = c.recency.get("themes")
        if th is not None and not th.empty:
            hot = th[th["flag"] != ""]
            with st.expander(f"Theme scan - {len(hot)} flagged theme(s) / tilt(s) with an outlier latest year",
                             expanded=bool((th["flag"] == "Hot year").any())):
                st.caption("Groups compared with this comparison's median. 'Hot year' = strong latest year that is unusual for "
                           "the group (may be skewing rankings); 'Persistent leader' = strong now and in the prior 2 years.")
                show = th.rename(columns={"dimension": "Dimension", "group": "Group", "n": "Funds", "flag": "Flag",
                                          "median_1y": "Median 1Y %", "excess_1y": "1Y excess", "excess_ytd": "YTD excess",
                                          "excess_prior": "Prior 2Y excess", "median_skew": "Median skew",
                                          "top_quintile_off": "Top-20% (recency off)", "top_quintile_now": "Top-20% (now)",
                                          "avg_rank_change": "Avg rank chg", "tickers": "Funds in group"})
                def _fl(v):
                    return {"Hot year": f"background-color:{rgba(T.coral, 0.35)}",
                            "Cold year": f"background-color:{rgba(T.cyan, 0.35)}",
                            "Persistent leader": f"background-color:{rgba(T.teal, 0.35)}"}.get(v, "")
                st.dataframe(show.style.map(_fl, subset=["Flag"]).format(
                    {c_: "{:+.1f}" for c_ in ("1Y excess", "YTD excess", "Prior 2Y excess", "Median skew", "Avg rank chg")}
                    | {"Median 1Y %": "{:.1f}"}, na_rep="—"), hide_index=True, width="stretch", height=360)

        with st.expander("Lens leaders (top 3 within each group)"):
            if not c.leaders.empty:
                ldr = c.leaders.copy()
                ldr["Fund"] = ldr["ticker"].map(c.df["name"])
                ldr["Composite"] = ldr["ticker"].map(s.adjusted).round(1)
                ldr["Overall rank"] = ldr["ticker"].map(s.rank)
                st.dataframe(ldr.rename(columns={"lens": "Lens", "group": "Group", "group_n": "n", "within_rank": "Within",
                                                 "ticker": "Ticker"}), hide_index=True, width="stretch")

    # =========================================================================== 3. Data check
    with tab_data:
        c = by_kind["full"]
        st.subheader("What was read")
        a, b = st.columns(2)
        with a:
            st.write("**Files**")
            for n, h in c.meta["files"].items():
                st.write(f"- `{n}` · sha {h}")
            if ds.pages:
                st.dataframe(pd.DataFrame([{"Data set": p["kind"], "File": p["file"], "Print date": p.get("print_date"),
                                            "Funds": len(p["tickers"]), "Benchmark values": p.get("bench", 0)}
                                           for p in ds.pages]), hide_index=True, width="stretch")
            else:
                for lay in ds.layouts:
                    st.write(f"- {lay}")
            st.write(f"**Funds:** {c.meta['n_input']} · **Returns basis:** {c.meta['returns_basis']} · **As of:** {c.asof}")
        with b:
            sev = pd.DataFrame([{"Severity": e.severity, "Area": e.area, "Ticker": e.ticker, "Message": e.message}
                                for e in c.log.sorted()])
            st.write("**Data quality log**")
            st.dataframe(sev, hide_index=True, width="stretch", height=260)

        st.markdown("##### Unmapped labels")
        if not ds.unmapped:
            st.success("Every label in the export matched the metric dictionary.")
        else:
            st.caption("These labels are kept in Original Data but not used. Map one here to make it permanent "
                       "(saved to zenith/fundlens/metrics_user.yaml), or ask the Claude Code skill `/fund-compare` to propose mappings.")
            with st.form("fl_map"):
                lab = st.selectbox("Label", ds.unmapped)
                m1_, m2_, m3_, m4_ = st.columns(4)
                name = m1_.text_input("Display name", value=lab.title())
                kind_ = m2_.selectbox("Type", ["directional", "characteristic", "text"])
                direction = m3_.selectbox("Good direction", ["higher", "lower", "none"])
                fmt = m4_.selectbox("Format", ["ratio", "pct", "int", "money_mm", "num", "text"])
                if st.form_submit_button("Save mapping"):
                    mid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                    add_user_metric({"id": mid, "label": name, "kind": kind_, "direction": direction, "format": fmt,
                                     "patterns": [re.escape(norm_label(lab))],
                                     "description": f"User mapping of Morningstar label '{lab}'."})
                    st.cache_resource.clear()
                    st.success(f"Mapped '{lab}' -> {mid}. Add it to a pillar on the Priorities tab to score it.")
                    st.rerun()

        if c.audit is not None:
            st.markdown("##### Approval audit")
            st.dataframe(c.audit, hide_index=True, width="stretch", height=300)
        if c.excluded:
            st.markdown("##### Screened out")
            st.dataframe(pd.DataFrame(c.excluded), hide_index=True, width="stretch")

    # =========================================================================== 4. Export
    with tab_exp:
        st.subheader("Generate workbooks")
        st.caption("One click writes the full-universe workbook and, when an approved list is loaded, the approved-only "
                   "workbook. Colours are baked in, so they look the same in Google Sheets (upload to Drive -> Open with "
                   "Google Sheets), Excel and LibreOffice.")
        st.write("**Will create:** " + ", ".join(f"`{output_name(c)}`" for c in comps))
        if st.button("Build workbooks", type="primary"):
            with st.spinner("Writing workbooks..."):
                tmp = Path(tempfile.mkdtemp(prefix="fundlens_out_"))
                paths = [write_workbook(c, tmp, palette) for c in comps]
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for p in paths:
                        z.write(p, p.name)
                    z.writestr(f"{prof_for_run['id']}.yaml", to_yaml(prof_for_run))
                st.session_state["fl_export"] = {"zip": buf.getvalue(), "files": [(p.name, p.read_bytes()) for p in paths]}
        exp = st.session_state.get("fl_export")
        if exp:
            st.download_button("Download all (.zip)", exp["zip"], "FundLens workbooks.zip", type="primary")
            for n, b in exp["files"]:
                st.download_button(f"Download {n}", b, n, key=f"fl_dl:{n}")
