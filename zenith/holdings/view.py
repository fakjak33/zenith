"""HOLDINGS tab — what a fund holds today, and how that book is moving.

Reads only committed JSON (data/holdings/<fund>/*). Every heavy calculation —
change events, window rankings, the colour-scale cap — happens offline in
`compute.py`, so this module reshapes and draws, nothing more.

The visual argument the page makes, top to bottom: how big and which way
(numeric slabs) -> what moved lately (change heatmap) -> the whole strategy's
shape over time (position matrix) -> who moved most (rankings) -> one position
in depth (explorer) -> the raw book (table).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .. import ui_charts as uc
from ..config import THEME
from ..ui_theme import evidence_rating, section, stamp
from . import (DISCLAIMER, WINDOWS, archive_days, history, load, load_day,
               load_funds)

# ------------------------------------------------------------------ config --
STALE_AFTER_TRADING_DAYS = 2

STATE_META = {
    "current": (THEME.teal, "CURRENT"),
    "stale": (THEME.mustard, "STALE"),
    "unavailable": (THEME.coral, "DATA UNAVAILABLE"),
    "historical": (THEME.muted, "HISTORICAL"),
    "empty": (THEME.muted, "NO DATA YET"),
}

CLASS_COLORS = {
    "rates": THEME.navy, "equity": THEME.teal, "fx": THEME.mauve,
    "commodity": THEME.mustard, "collateral": THEME.muted,
    "unclassified": THEME.orange,
}

LENSES = {
    "Δ Weight": "dw",
    "Δ Exposure": "dn",
    "New / Closed": "event",
    "Long / Short": "w",
}

MATRIX_WINDOWS = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366, "All": None}

COLS = {
    "Position": "Contract root — the continuing position. A quarterly roll "
                "(Sep to Dec) stays one position rather than a close plus an open.",
    "Contract": "The dated futures contract currently held.",
    "Asset class": "Rates, equity, FX, commodity, or collateral.",
    "Exposure": "Direction of the position: long or short.",
    "Notional": "Signed notional market value as published by the fund — for "
                "futures this is exposure, not mark-to-market profit.",
    "Weight": "Notional as a fraction of net assets. -0.96 means short 96% of NAV.",
    "Δ Weight": "Change in weight since the previous published snapshot. "
                "Positive = more long or less short.",
    "Δ Exposure": "Change in notional dollars since the previous snapshot.",
    "Δ %": "Change in the SIZE of the position, ignoring sign — a short that "
           "got 10% bigger reads +10%.",
    "Change": "new, closed, increased, reduced, flipped or held. Increased and "
              "reduced refer to absolute size, so a growing short is an increase.",
    "First seen": "First snapshot in which this position appears.",
    "Days held": "Number of stored snapshots in which the position appears.",
}


# ----------------------------------------------------------------- helpers --
@st.cache_data(ttl=600, show_spinner=False)
def _artefacts(fund: str, cache_bust: str = "") -> dict:
    """Load a fund's committed artefacts once per rerun-storm.

    Cached because Streamlit executes EVERY tab body on EVERY rerun, so an
    uncached read here would be paid on every click anywhere in the app.
    """
    return {
        "latest": load(fund, "latest", {}),
        "history": load(fund, "history", {}),
        "changes": load(fund, "changes", {}),
        "status": load(fund, "status", {}),
        "days": archive_days(fund),
    }


def _trading_days_since(iso: str) -> int:
    try:
        from ..pretom import calendar as cal
        d = date.fromisoformat(iso)
        return max(0, len([x for x in cal.trading_days(d, date.today()) if x > d]))
    except Exception:
        return 0


def _state_of(art: dict) -> tuple[str, str]:
    """(state key, human message) for the banner."""
    latest, status = art["latest"], art["status"]
    if not latest or not latest.get("as_of"):
        return "empty", "no snapshot stored yet — the daily job has not run"
    if status.get("state") in ("unavailable", "error") or status.get("error"):
        return "unavailable", (status.get("error") or "source unavailable")[:160]
    lag = _trading_days_since(latest["as_of"])
    if lag > STALE_AFTER_TRADING_DAYS:
        return "stale", (f"the fund has not published since {latest['as_of']} "
                         f"— {lag} trading days ago. Showing the last good book.")
    n = latest.get("n_snapshots", 0)
    return "current", (f"holdings as published {latest['as_of']} · "
                       f"{n} snapshot{'s' * (n != 1)} on file since "
                       f"{latest.get('first_snapshot', '?')}")


def _pos_frame(positions: list[dict], events: list[dict]) -> pd.DataFrame:
    ev = {e["id"]: e for e in events}
    rows = []
    for p in positions:
        e = ev.get(p["id"], {})
        rows.append({
            "Position": p["name"],
            "Contract": p.get("contract") or "—",
            "Asset class": p["asset_class"],
            "Exposure": p["direction"],
            "Notional": p["notional"],
            "Weight": p["weight"],
            "Δ Weight": e.get("d_weight"),
            "Δ Exposure": e.get("d_notional"),
            "Δ %": e.get("pct_change"),
            "Change": e.get("type", "held"),
            "_id": p["id"],
        })
    return pd.DataFrame(rows)


def today_badge() -> str | None:
    """TODAY-tab chip: only speaks up when the book actually moved."""
    try:
        funds = (load_funds() or {}).get("funds", [])
        chips = []
        for f in funds:
            if not f.get("enabled"):
                continue
            latest = load(f["key"], "latest", {})
            if not latest.get("as_of"):
                continue
            s = latest.get("summary", {})
            moved = (s.get("n_added", 0) + s.get("n_closed", 0)
                     + s.get("n_flipped", 0))
            big = s.get("largest_change") or {}
            if not moved and not big:
                continue
            bits = []
            if s.get("n_added"):
                bits.append(f"{s['n_added']} new")
            if s.get("n_closed"):
                bits.append(f"{s['n_closed']} closed")
            if s.get("n_flipped"):
                bits.append(f"{s['n_flipped']} flipped")
            if big:
                bits.append(f"{big['name']} {uc.fmt_pct(big['d_weight'])}")
            skipped = latest.get("previous_gap_sessions") or 0
            basis = (latest["as_of"] if not skipped
                     else f"{latest['as_of']} vs {latest.get('previous_as_of')}")
            chips.append(uc.chip(f"{f['ticker']} · " + " · ".join(bits),
                                 THEME.mint, basis))
        return "".join(chips) or None
    except Exception:
        return None


# ------------------------------------------------------------------ blocks --
def _summary(latest: dict, fund_meta: dict) -> None:
    s = latest.get("summary", {})
    largest = s.get("largest") or {}
    big = s.get("largest_change") or {}
    net = s.get("net", 0.0)
    slabs = [
        {"label": "Long exposure", "value": uc.fmt_pct(s.get("long"), 0),
         "sub": f"{s.get('n_long', 0)} positions", "color": THEME.teal},
        {"label": "Short exposure", "value": uc.fmt_pct(s.get("short"), 0),
         "sub": f"{s.get('n_short', 0)} positions", "color": THEME.coral},
        {"label": "Net", "value": uc.fmt_pct(net, 0),
         "sub": "long minus short",
         "color": THEME.teal if net >= 0 else THEME.coral},
        {"label": "Gross", "value": uc.fmt_pct(s.get("gross"), 0),
         "sub": "total notional / NAV", "color": THEME.mustard},
        {"label": "Positions", "value": str(s.get("n_positions", 0)),
         "sub": f"+{uc.fmt_pct(s.get('collateral'), 0, signed=False)} collateral",
         "color": THEME.navy},
    ]
    st.markdown(uc.numeric_slab(slabs), unsafe_allow_html=True)

    prev = latest.get("previous_as_of")
    skipped = latest.get("previous_gap_sessions") or 0
    if not prev:
        since = "no prior snapshot"
    elif skipped:
        # The previous snapshot is not yesterday, so none of these counts are
        # a one-day story. Say so in the label itself.
        since = f"over {skipped + 1} sessions from {prev}"
    else:
        since = f"since {prev}"
    slabs2 = [
        {"label": "Added", "value": str(s.get("n_added", 0)), "sub": since,
         "color": THEME.mint},
        {"label": "Increased", "value": str(s.get("n_increased", 0)),
         "sub": since, "color": THEME.teal},
        {"label": "Reduced", "value": str(s.get("n_reduced", 0)), "sub": since,
         "color": THEME.orange},
        {"label": "Closed", "value": str(s.get("n_closed", 0)), "sub": since,
         "color": THEME.coral},
        {"label": "Largest position", "value": uc.fmt_pct(largest.get("weight"), 0),
         "sub": largest.get("name", "—"), "color": THEME.mauve},
        {"label": "Largest move", "value": uc.fmt_pct(big.get("d_weight"), 1),
         "sub": big.get("name", "—"), "color": THEME.mustard},
    ]
    st.markdown(uc.numeric_slab(slabs2, min_width=140), unsafe_allow_html=True)
    if skipped:
        st.caption(
            f"The previous snapshot on file is {prev}, {skipped} trading "
            f"session{'s' * (skipped != 1)} before this one — the change figures "
            "above are the move across that whole span, not a single day.")


def _exposure_bars(latest: dict) -> None:
    by_class = (latest.get("summary", {}) or {}).get("by_asset_class", [])
    if not by_class:
        return
    rows = []
    for b in by_class:
        for side in ("long", "short"):
            if b[side]:
                rows.append({"Asset class": b["asset_class"].title(),
                             "side": side, "w": b[side]})
    if not rows:
        return
    df = pd.DataFrame(rows)

    def build(alt):
        return (alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("w:Q", title="notional exposure / NAV",
                    axis=alt.Axis(format="+.0%", labelLimit=0)),
            y=alt.Y("Asset class:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=0)),
            color=alt.Color("side:N", legend=None,
                            scale=alt.Scale(domain=["long", "short"],
                                            range=[THEME.teal, THEME.coral])),
            tooltip=["Asset class", "side",
                     alt.Tooltip("w:Q", format="+.2%", title="exposure")],
        ).properties(height=max(150, 40 * df["Asset class"].nunique())))
    uc.render_chart(build, fallback=df)


def _change_heatmap(hist: dict, caps: dict, key: str) -> str | None:
    """Position x recent days, coloured by what changed. Returns a click pick."""
    ids = history.rank_ids(hist)
    if not ids or len(hist.get("dates", [])) < 2:
        st.caption("The change heatmap appears once a second snapshot is stored.")
        return None

    c1, c2 = st.columns([3, 2])
    with c1:
        lens = st.radio("Lens", list(LENSES), horizontal=True,
                        key=f"{key}_lens", label_visibility="collapsed")
    with c2:
        span = st.slider("Sessions shown", 5, max(6, len(hist["dates"])),
                         min(20, len(hist["dates"])), key=f"{key}_span")

    field = LENSES[lens]
    recs = history.delta_matrix(hist, last_n=span)
    if not recs:
        st.caption("Not enough history yet.")
        return None
    df = pd.DataFrame(recs)
    order = [i for i in ids if i in set(df["id"])]
    labels = {r["id"]: r["name"] for r in recs}
    df["label"] = df["id"].map(labels)
    label_order = [labels[i] for i in order]
    # A column whose change spans missing snapshots is not a day's move, and
    # must not be read as one.
    gapped = sorted({r["d"] for r in recs if r.get("gap")})

    if field == "event":
        return _event_heatmap(df, label_order, key, gapped)

    cap = float((caps or {}).get(
        "d_weight" if field in ("dw", "w") else "d_notional") or 0.01)
    if field == "w":
        cap = max(abs(df["w"].dropna()).max() if df["w"].notna().any() else 1.0,
                  0.01)
    fmt = "+.2%" if field in ("dw", "w") else "+,.0f"
    title = {"dw": "change in weight", "dn": "change in notional ($)",
             "w": "weight (long positive, short negative)"}[field]

    live = df[df[field].notna()].copy()
    absent = df[df[field].isna()].copy()

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["id"], on="click",
                                   clear="dblclick", empty=False)
        base = alt.Chart(live).mark_rect(stroke=THEME.bg, strokeWidth=0.5).encode(
            x=alt.X("d:O", title=None, axis=alt.Axis(labelAngle=-60,
                                                     labelLimit=0,
                                                     labelFontSize=9)),
            y=alt.Y("label:N", sort=label_order, title=None,
                    axis=alt.Axis(labelLimit=0)),
            color=alt.Color(f"{field}:Q", title=title,
                            scale=uc.diverging_scale(cap, alt),
                            legend=alt.Legend(format=fmt, labelLimit=0)),
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.92)),
            tooltip=["label", "d", "asset_class", "event",
                     alt.Tooltip("spans:N", title="move covers"),
                     alt.Tooltip(f"{field}:Q", title=title, format=fmt),
                     alt.Tooltip("w:Q", title="weight after", format="+.2%")],
        ).add_params(pick)
        layers = [base]
        if not absent.empty:
            layers.insert(0, alt.Chart(absent).mark_rect(
                color="#141414", stroke=THEME.bg, strokeWidth=0.5).encode(
                    x=alt.X("d:O", title=None),
                    y=alt.Y("label:N", sort=label_order, title=None),
                    tooltip=["label", "d", alt.Tooltip("event:N", title="state")]))
        opened = live[live["event"] == "new"]
        if not opened.empty:
            layers.append(alt.Chart(opened).mark_rect(
                fillOpacity=0, stroke=THEME.mustard, strokeWidth=2).encode(
                    x=alt.X("d:O"), y=alt.Y("label:N", sort=label_order)))
        shut = live[live["event"] == "closed"]
        if not shut.empty:
            layers.append(alt.Chart(shut).mark_rect(
                fillOpacity=0, stroke=THEME.coral, strokeWidth=2,
                strokeDash=[4, 3]).encode(
                    x=alt.X("d:O"), y=alt.Y("label:N", sort=label_order)))
        layers.extend(_gap_layer(alt, gapped, label_order))
        return alt.layer(*layers).properties(
            height=max(240, 26 * max(len(label_order), 1)))

    event = uc.render_chart(build, fallback=df[["label", "d", field]],
                            on_select="rerun", key=f"{key}_hm")
    st.caption("Dark cells are days the position barely moved; teal is a move "
               "toward long, coral toward short. A mustard outline is a position "
               "opened that day, a dashed coral outline one closed. Grey means "
               "not held. Click a cell to load that position below.")
    _gap_caption(gapped)
    return _picked(event)


def _gap_layer(alt, gapped: list[str], label_order: list[str]) -> list:
    """A mustard rule marking every column that spans missing snapshots."""
    if not gapped:
        return []
    g = pd.DataFrame({"d": gapped})
    return [alt.Chart(g).mark_rule(color=THEME.mustard, strokeWidth=2,
                                   strokeDash=[2, 2], opacity=0.9).encode(
        x=alt.X("d:O", title=None))]


def _gap_caption(gapped: list[str]) -> None:
    if not gapped:
        return
    shown = ", ".join(gapped[:6])
    more = f" and {len(gapped) - 6} more" if len(gapped) > 6 else ""
    st.caption(
        f"⚠ {len(gapped)} column{'s' * (len(gapped) != 1)} carry a **dashed "
        f"mustard rule** ({shown}{more}): the fund published nothing in "
        "between, so that change accumulated over more than one session. Hover "
        "a cell to see how many sessions it covers.")


def _event_heatmap(df: pd.DataFrame, label_order: list[str], key: str,
                   gapped: list[str] | None = None):
    """Categorical lens: openings and closings only."""
    dfe = df.copy()
    dfe["state"] = dfe["event"].map({"new": "opened", "closed": "closed",
                                     "move": "held", "absent": "not held"})

    def build(alt):
        pick = alt.selection_point(name="pick", fields=["id"], on="click",
                                   clear="dblclick", empty=False)
        grid = alt.Chart(dfe).mark_rect(stroke=THEME.bg, strokeWidth=0.5).encode(
            x=alt.X("d:O", title=None,
                    axis=alt.Axis(labelAngle=-60, labelLimit=0, labelFontSize=9)),
            y=alt.Y("label:N", sort=label_order, title=None,
                    axis=alt.Axis(labelLimit=0)),
            color=alt.Color("state:N", title=None,
                            scale=alt.Scale(
                                domain=["opened", "closed", "held", "not held"],
                                range=[THEME.mustard, THEME.coral,
                                       "#1d4f4a", "#141414"]),
                            legend=alt.Legend(orient="top", labelLimit=0)),
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.92)),
            tooltip=["label", "d", "state", "asset_class",
                     alt.Tooltip("spans:N", title="covers")],
        ).add_params(pick)
        return alt.layer(grid, *_gap_layer(alt, gapped or [], label_order)
                         ).properties(
            height=max(240, 26 * max(len(label_order), 1)))

    event = uc.render_chart(build, fallback=dfe[["label", "d", "state"]],
                            on_select="rerun", key=f"{key}_hm_evt")
    st.caption("Every opening and closing in the window. Held positions sit "
               "quiet in dark teal so the two events that matter stand out.")
    _gap_caption(gapped or [])
    return _picked(event)


def _picked(event) -> str | None:
    """Pull an id out of an Altair selection payload, tolerating any shape."""
    try:
        rows = (event or {}).get("selection", {}).get("pick", [])
        if rows:
            return rows[0].get("id")
    except Exception:
        pass
    return None


def _matrix(hist: dict, key: str) -> None:
    dates = hist.get("dates", [])
    if not dates:
        return
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        win = st.radio("Window", list(MATRIX_WINDOWS), horizontal=True,
                       index=1, key=f"{key}_mwin", label_visibility="collapsed")
    with c2:
        layout = st.radio("Layout", ["Matrix", "Stacked"], horizontal=True,
                          key=f"{key}_mlayout", label_visibility="collapsed",
                          help="Stacked redraws the same data as one strip per "
                               "position — easier to scroll on a phone.")
    ids = history.rank_ids(hist)
    with c3:
        depth = st.slider("Positions shown", 3, max(4, len(ids)),
                          min(14, len(ids)), key=f"{key}_mdepth")
    ids = ids[:depth]

    days = MATRIX_WINDOWS[win]
    since = None
    if days:
        from datetime import timedelta
        since = (date.fromisoformat(dates[-1]) - timedelta(days=days)).isoformat()
    recs = history.matrix(hist, ids=ids, since=since)
    if not recs:
        st.caption("No snapshots in this window.")
        return
    df = pd.DataFrame(recs)
    labels = {r["id"]: r["name"] for r in recs}
    df["label"] = df["id"].map(labels)
    order = [labels[i] for i in ids if i in labels]
    cap = max(abs(df["w"].dropna()).max() if df["w"].notna().any() else 1.0, 0.05)

    live = df[df["w"].notna()]
    absent = df[df["w"].isna()]

    if layout == "Stacked":
        _matrix_stacked(live, order, cap, key)
    else:
        def build(alt):
            layers = []
            if not absent.empty:
                layers.append(alt.Chart(absent).mark_rect(
                    color="#141414", stroke=THEME.bg, strokeWidth=0.4).encode(
                        x=alt.X("d:O", title=None),
                        y=alt.Y("label:N", sort=order, title=None)))
            layers.append(alt.Chart(live).mark_rect(
                stroke=THEME.bg, strokeWidth=0.4).encode(
                    x=alt.X("d:O", title=None,
                            axis=alt.Axis(labelAngle=-60, labelLimit=0,
                                          labelFontSize=8)),
                    y=alt.Y("label:N", sort=order, title=None,
                            axis=alt.Axis(labelLimit=0)),
                    color=alt.Color("w:Q", title="weight / NAV",
                                    scale=uc.diverging_scale(cap, alt),
                                    legend=alt.Legend(format="+.0%",
                                                      labelLimit=0)),
                    tooltip=["label", "d", "asset_class", "state",
                             alt.Tooltip("w:Q", title="weight", format="+.2%")]))
            return alt.layer(*layers).properties(
                height=max(240, 28 * max(len(order), 1)))
        uc.render_chart(build, fallback=live[["label", "d", "w"]])

    n_snaps = len(df["d"].unique())
    if layout == "Stacked":
        st.caption(f"{n_snaps} published snapshots, one strip per position. "
                   "Each bar is a single snapshot on a real time axis, so a "
                   "stretch where the fund published nothing reads as empty "
                   "space rather than a drawn-through line.")
    else:
        st.caption(f"{n_snaps} published snapshots. Teal is long, coral short, "
                   "intensity is size; grey means the position was not held. "
                   "Columns are publication dates, so a day the fund did not "
                   "publish simply has no column — nothing is interpolated.")


def _matrix_stacked(live: pd.DataFrame, order: list[str], cap: float,
                    key: str) -> None:
    """Small-multiple strips — the mobile-friendly reading of the matrix.

    Bars, not a connected area: an interpolated line would draw a confident
    stroke straight across a stretch where the fund published nothing. Each bar
    is one snapshot, and a gap is simply empty space.
    """
    def build(alt):
        return (alt.Chart(live).mark_bar(size=3).encode(
            x=alt.X("d:T", title=None, axis=alt.Axis(format="%d %b",
                                                     tickCount=7,
                                                     labelLimit=0,
                                                     labelFontSize=8)),
            y=alt.Y("w:Q", title=None, axis=alt.Axis(format="+.0%",
                                                     labelLimit=0, tickCount=3)),
            color=alt.Color("w:Q", scale=uc.diverging_scale(cap, alt),
                            legend=None),
            tooltip=["label", "d", "state",
                     alt.Tooltip("w:Q", title="weight", format="+.2%")],
        ).properties(height=62).facet(
            row=alt.Row("label:N", sort=order, title=None,
                        header=alt.Header(labelAngle=0, labelAlign="left",
                                          labelLimit=0, labelFontSize=11))))
    uc.render_chart(build, fallback=live[["label", "d", "w"]])


def _rankings(changes: dict, key: str) -> None:
    ranks = changes.get("rankings", {})
    if not ranks:
        st.caption("Rankings appear once two snapshots are stored.")
        return
    labels = [w for w in WINDOWS if w in ranks]
    win = st.radio("Timeframe", labels, horizontal=True, index=0,
                   key=f"{key}_rankwin", label_visibility="collapsed")
    r = ranks.get(win, {})
    if not r.get("from"):
        st.caption("Not enough history for this timeframe yet.")
        return
    if r.get("exact", True):
        st.caption(f"Comparing {r['from']} to {r['to']} — "
                   f"{r.get('n_snapshots', 0)} snapshots in range.")
    else:
        st.caption(
            f"⚠ No snapshot sits {r.get('requested_days')} day"
            f"{'s' * (r.get('requested_days') != 1)} back, so this is the "
            f"nearest available comparison: **{r['from']} to {r['to']}**, a span "
            f"of {r.get('actual_days')} calendar days. Read it as that span, "
            f"not as {win}.")

    cap = max([abs(x["d_abs_weight"]) for x in
               (r.get("increases", []) + r.get("decreases", []))] or [0.01])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(section("Largest increases", 0), unsafe_allow_html=True)
        inc = r.get("increases", [])
        if inc:
            df = pd.DataFrame(inc)
            df["label"] = df["name"]
            uc.hbar(df, "d_abs_weight", "label", cap=cap,
                    title="change in position size",
                    tooltip=["label", "asset_class",
                             {"field": "w_from", "type": "quantitative",
                              "format": "+.2%", "title": "from"},
                             {"field": "w_to", "type": "quantitative",
                              "format": "+.2%", "title": "to"}])
        else:
            st.caption("Nothing grew over this window.")
    with c2:
        st.markdown(section("Largest decreases", 2), unsafe_allow_html=True)
        dec = r.get("decreases", [])
        if dec:
            df = pd.DataFrame(dec)
            df["label"] = df["name"]
            uc.hbar(df, "d_abs_weight", "label", cap=cap,
                    title="change in position size",
                    tooltip=["label", "asset_class",
                             {"field": "w_from", "type": "quantitative",
                              "format": "+.2%", "title": "from"},
                             {"field": "w_to", "type": "quantitative",
                              "format": "+.2%", "title": "to"}])
        else:
            st.caption("Nothing shrank over this window.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(section("New positions", 5), unsafe_allow_html=True)
        new = r.get("new", [])
        st.markdown(
            "".join(uc.chip(n["name"], THEME.mint, uc.fmt_pct(n["w_to"]))
                    for n in new) or
            '<span style="color:#888;">None opened in this window.</span>',
            unsafe_allow_html=True)
    with c4:
        st.markdown(section("Closed positions", 2), unsafe_allow_html=True)
        shut = r.get("closed", [])
        st.markdown(
            "".join(uc.chip(n["name"], THEME.coral,
                            f"was {uc.fmt_pct(n['w_from'])}") for n in shut) or
            '<span style="color:#888;">None closed in this window.</span>',
            unsafe_allow_html=True)


def _explorer(hist: dict, key: str, preselect: str | None) -> None:
    ids = history.rank_ids(hist)
    if not ids:
        return
    names = {pid: hist["series"][pid]["name"] for pid in ids}
    sel_key = f"{key}_explore"
    if preselect and preselect in names:
        st.session_state[sel_key] = preselect
    if st.session_state.get(sel_key) not in names:
        st.session_state[sel_key] = ids[0]

    pid = st.selectbox("Position", ids, key=sel_key,
                       format_func=lambda i: names.get(i, i))
    stats = history.position_stats(hist, pid)
    if not stats.get("held"):
        st.caption("No history for this position.")
        return

    colour = THEME.teal if stats["current_direction"] == "long" else THEME.coral
    st.markdown(uc.numeric_slab([
        {"label": "Current weight", "value": uc.fmt_pct(stats["current_weight"], 1),
         "sub": stats["sub_class"], "color": colour},
        {"label": "Exposure", "value": uc.fmt_money(stats["current_notional"]),
         "sub": stats["current_contract"] or "—", "color": colour},
        {"label": "Direction", "value": stats["current_direction"].upper(),
         "sub": f"{stats['n_direction_flips']} flips on record",
         "color": colour},
        {"label": "Days held", "value": str(stats["days_observed"]),
         "sub": f"of {hist.get('n_dates', 0)} snapshots", "color": THEME.navy},
    ]), unsafe_allow_html=True)

    pts = history.series_for(hist, pid)
    df = pd.DataFrame(pts)
    if not df.empty:
        cap = max(abs(df["w"]).max(), 0.02)

        def build(alt):
            base = alt.Chart(df).encode(
                # A temporal axis over sparse snapshots generates a tick every
                # other day; cap it so the labels stay readable.
                x=alt.X("d:T", title=None,
                        axis=alt.Axis(format="%d %b", tickCount=7,
                                      labelLimit=0)))
            area = base.mark_area(interpolate="monotone", opacity=0.35).encode(
                y=alt.Y("w:Q", title="weight / NAV",
                        axis=alt.Axis(format="+.0%", labelLimit=0)),
                color=alt.Color("w:Q", scale=uc.diverging_scale(cap, alt),
                                legend=None))
            line = base.mark_line(interpolate="monotone", strokeWidth=2.2,
                                  color=colour).encode(y="w:Q")
            zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
                color=THEME.grid, strokeDash=[4, 4]).encode(y="y:Q")
            pointed = base.mark_circle(size=34, color=colour).encode(
                y="w:Q",
                tooltip=["d", alt.Tooltip("w:Q", title="weight", format="+.2%"),
                         alt.Tooltip("n:Q", title="notional", format="+,.0f"),
                         "contract", "direction"])
            return (zero + area + line + pointed).properties(height=280)
        uc.render_chart(build, fallback=df[["d", "w"]])

    mx, mn = stats["max_weight"], stats["min_weight"]
    last = stats["last_change"] or {}
    st.markdown(
        uc.chip(f"first seen {stats['first_seen']}", THEME.navy)
        + uc.chip(f"largest {uc.fmt_pct(mx['w'])}", THEME.mauve, mx["d"])
        + uc.chip(f"smallest {uc.fmt_pct(mn['w'])}", THEME.mustard, mn["d"])
        + uc.chip(f"mean size {uc.fmt_pct(stats['mean_abs_weight'], 1, False)}",
                  THEME.teal)
        + uc.chip(f"{stats['n_changes']} changes", THEME.mint)
        + (uc.chip(f"last change {uc.fmt_pct(last.get('d_weight'))}",
                   THEME.orange, last.get("d", "")) if last else "")
        + (uc.chip(f"contracts: {', '.join(stats['contracts_seen'][-4:])}",
                   THEME.muted) if stats["contracts_seen"] else ""),
        unsafe_allow_html=True)

    if stats["recent_changes"]:
        rc = pd.DataFrame(stats["recent_changes"])
        rc.columns = ["Date", "Δ Weight", "Weight"]
        try:
            sty = (rc.style
                   .map(lambda v: uc.grad_diverging(v, 0.05), subset=["Δ Weight"])
                   .format({"Δ Weight": "{:+.2%}", "Weight": "{:+.2%}"}))
            st.dataframe(sty, use_container_width=True, hide_index=True,
                         height=min(360, 45 + 35 * len(rc)))
        except Exception:
            st.dataframe(rc, use_container_width=True, hide_index=True)


def _book_table(positions: list[dict], events: list[dict], key: str,
                hist: dict | None = None) -> None:
    df = _pos_frame(positions, events)
    if df.empty:
        st.caption("No positions in this snapshot.")
        return
    if hist:
        firsts, days = {}, {}
        for pid in df["_id"]:
            s = history.position_stats(hist, pid)
            firsts[pid] = s.get("first_seen", "—")
            days[pid] = s.get("days_observed", 0)
        df["First seen"] = df["_id"].map(firsts)
        df["Days held"] = df["_id"].map(days)
    disp = df.drop(columns=["_id"])
    try:
        sty = (disp.style
               .map(lambda v: uc.grad_diverging(v, 1.0), subset=["Weight"])
               .map(lambda v: uc.grad_diverging(v, 0.05), subset=["Δ Weight"])
               .format({"Weight": "{:+.2%}", "Δ Weight": "{:+.2%}",
                        "Δ %": lambda v: uc.fmt_pct(v, 1),
                        "Notional": lambda v: uc.fmt_money(v),
                        "Δ Exposure": lambda v: uc.fmt_money(v)}))
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(560, 45 + 35 * len(disp)),
                     column_config=uc.colcfg(disp.columns, COLS), key=key)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ render --
def render() -> None:
    st.caption(DISCLAIMER)
    reg = load_funds() or {}
    funds = [f for f in reg.get("funds", []) if f.get("enabled")]
    if not funds:
        st.info("No funds registered yet. Run "
                "`python -m zenith.holdings.compute --action auto`.")
        return

    if len(funds) > 1:
        keys = [f["key"] for f in funds]
        pick = st.radio("Fund", keys, horizontal=True, key="hold_fund",
                        label_visibility="collapsed",
                        format_func=lambda k: next(f["ticker"] for f in funds
                                                   if f["key"] == k))
    else:
        pick = funds[0]["key"]
    meta = next(f for f in funds if f["key"] == pick)
    key = f"hold_{pick}"

    art = _artefacts(pick, cache_bust=str((load(pick, "status", {}) or {})
                                          .get("ran_at", "")))
    latest, hist, changes = art["latest"], art["history"], art["changes"]
    state, message = _state_of(art)
    colour, label = STATE_META[state]

    st.markdown(uc.state_banner(colour, f"{meta['ticker']} · {label}", message),
                unsafe_allow_html=True)
    if state == "empty":
        st.info("No snapshots stored yet. The daily job writes the first one "
                "the next time the fund publishes.")
        return

    st.markdown(evidence_rating(
        "A", "PRIMARY SOURCE — THE FUND'S OWN DAILY DISCLOSURE",
        f"{meta['source_label']}. Parsed directly from the published table; "
        "no third-party interpretation sits in between."), unsafe_allow_html=True)
    st.markdown(stamp(latest.get("as_of", "—"), meta["ticker"]),
                unsafe_allow_html=True)

    with st.expander(f"What {meta['ticker']} is, and how to read this book"):
        st.markdown(f"**{meta['name']}** — {meta['adviser']}, sub-advised by "
                    f"{meta['subadviser']}.\n\n{meta['strategy']}\n\n"
                    f"{meta['reads_as']}\n\n[Source]({meta['source_url']})")

    st.markdown(section(f"{meta['ticker']} positioning", 5,
                        help="Exposure figures are notional as a fraction of "
                             "net assets, collateral excluded."),
                unsafe_allow_html=True)
    _summary(latest, meta)
    st.markdown(uc.note_strip("how to read these numbers",
                              list(meta.get("caveats", []))),
                unsafe_allow_html=True)
    _exposure_bars(latest)

    gaps = latest.get("gaps") or []
    if gaps:
        st.caption(
            f"No snapshot on record for {len(gaps)} trading "
            f"day{'s' * (len(gaps) != 1)} in the last month "
            f"({', '.join(gaps[:6])}{'…' if len(gaps) > 6 else ''}). Holdings "
            "can only be captured on the day they are published, so a day the "
            "tracker was not running is gone for good — those days are simply "
            "absent from the charts rather than filled in.")
    warn = (latest.get("quality") or {}).get("warnings") or []
    if warn:
        st.caption("Data-quality flags on the latest snapshot: " + ", ".join(warn))

    st.markdown(section("What changed", 2,
                        help="Day-over-day movement in every position. "
                             "Click a cell to open that position below."),
                unsafe_allow_html=True)
    picked = _change_heatmap(hist, latest.get("caps") or changes.get("caps"), key)

    st.markdown(section("Position matrix — the strategy's fingerprint", 3,
                        help="Position size over time. Teal long, coral short."),
                unsafe_allow_html=True)
    _matrix(hist, key)

    st.markdown(section("Biggest movers", 4,
                        help="Endpoint-to-endpoint change over the chosen "
                             "window, not a sum of daily moves."),
                unsafe_allow_html=True)
    _rankings(changes, key)

    st.markdown(section("Position explorer", 1,
                        help="One position's full history."),
                unsafe_allow_html=True)
    _explorer(hist, key, picked)

    st.markdown(section("The full book", 0), unsafe_allow_html=True)
    n_dir = (latest.get("summary") or {}).get("n_positions", 0)
    n_all = latest.get("n_positions", 0)
    with st.expander(f"Every row as published {latest.get('as_of', '')} — "
                     f"{n_dir} directional position{'s' * (n_dir != 1)}"
                     f"{f' plus {n_all - n_dir} collateral line' if n_all > n_dir else ''}",
                     expanded=False):
        _book_table(latest.get("positions", []), latest.get("changes", []),
                    f"{key}_book", hist)

    days = art["days"]
    if len(days) > 1:
        st.markdown(section("Open a past snapshot", 5), unsafe_allow_html=True)
        day = st.selectbox("Snapshot date", days, key=f"{key}_archive")
        if day != latest.get("as_of"):
            old = load_day(pick, day)
            if old:
                st.markdown(uc.state_banner(
                    THEME.muted, "HISTORICAL",
                    f"the book as published {day} — not current"),
                    unsafe_allow_html=True)
                _book_table(old.get("positions", []), [], f"{key}_book_{day}")

    st.caption("Holdings are published by the fund on business days, usually "
               "for the prior session. Weights are recomputed as notional / net "
               "assets so the arithmetic on this page is internally consistent; "
               "the fund's own rounded weight column is kept alongside in the "
               "stored snapshots.")
