"""Sidebar widgets for the no-code builder: BLEND parts, CONDITIONS rows,
templates, and REGIME RULES. Each ``*_ui`` returns plain spec data; each
``*_to_widgets`` is its inverse (LIBRARY › open in LAB)."""

from __future__ import annotations

import streamlit as st

from ...config import THEME as T
from .. import glossary as G
from ..builder import OPS, REGIME_USES, SLOT_LABEL, SLOTS, TEMPLATES, sentence
from ..indicators import CATALOG
from ..regime import DETECTORS
from ..rules import REGISTRY
from . import common as C

MODES = {"single": "SINGLE RULE", "blend": "BLEND", "conditions": "CONDITIONS"}
MODE_HELP = ("SINGLE RULE: one rule decides. BLEND: several rules vote, weighted, and their average forecast is "
             "scaled up for diversification (Carver's FDM). CONDITIONS: write IF/AND/OR entry and exit rules, "
             "e.g. 'buy when RSI(2) < 10 AND close > SMA(200)'.")
BLEND_DEFAULTS = ["ema", "i_tsmom", "i_donchian", "i_zscore", "i_rsi"]
MAX_ROWS = 4


def _pkey(prefix: str, rule: str, name: str) -> str:
    return C.param_key(rule, name) if prefix == "bt" else f"{prefix}_p_{rule}_{name}"


def rule_params_ui(prefix: str, rule: str) -> dict:
    cls = REGISTRY[rule]
    out, cols = {}, st.columns(2)
    for i, (pname, spec_t) in enumerate(cls.params.items()):
        with cols[i % 2]:
            out[pname] = C._num(pname, spec_t, _pkey(prefix, rule, pname), help=G.param_help(cls, pname) or None)
    return out


# ------------------------------------------------------------------- blend ----
def blend_ui() -> list[dict]:
    n = int(st.slider("how many signals", 2, 5, 2, key="bt_b_n",
                      help="Mixing signals that disagree (trend + mean reversion) diversifies; mixing near-copies "
                           "adds nothing. The BLEND tab in BUILD shows their correlation."))
    parts = []
    for i in range(n):
        pre = f"bt_b{i}"
        if f"{pre}_rule" not in st.session_state:
            st.session_state[f"{pre}_rule"] = BLEND_DEFAULTS[i]
            st.session_state[f"{pre}_lib"] = C.LIBS[1] if BLEND_DEFAULTS[i].startswith("i_") else C.LIBS[0]
        with st.expander(f"SIGNAL {i + 1} · {REGISTRY[st.session_state[f'{pre}_rule']].label}", expanded=i == 0):
            rule, _ = C.signal_picker(pre)
            params = rule_params_ui(pre, rule)
            if f"{pre}_w" not in st.session_state:
                st.session_state[f"{pre}_w"] = 1.0
            w = st.slider("weight", 0.0, 1.0, step=0.05, key=f"{pre}_w",
                          help="Relative weight. Weights are normalised to sum to 1; 0 drops the signal.")
        parts.append({"rule": rule, "params": params, "weight": float(w)})
    tot = sum(p["weight"] for p in parts) or 1.0
    st.caption("Blend: " + " + ".join(f"{p['weight'] / tot:.0%} {REGISTRY[p['rule']].label}" for p in parts))
    return parts


def blend_to_widgets(parts: list[dict]) -> dict:
    w = {"bt_b_n": max(2, len(parts))}
    for i, p in enumerate(parts):
        pre = f"bt_b{i}"
        w[f"{pre}_rule"] = p["rule"]
        w[f"{pre}_lib"] = C.LIBS[1] if p["rule"].startswith("i_") else C.LIBS[0]
        w[f"{pre}_fam"] = "all"
        w[f"{pre}_w"] = float(p.get("weight", 1.0))
        for k, v in (p.get("params") or {}).items():
            w[_pkey(pre, p["rule"], k)] = v
    return w


# -------------------------------------------------------------- conditions ----
IND_KEYS = sorted(CATALOG, key=lambda k: (CATALOG[k].family, CATALOG[k].label))


def _operand_ui(key: str, allow_num: bool) -> dict:
    kinds = (["indicator", "close", "number"] if allow_num else ["indicator", "close"])
    kind = st.radio("type", kinds, horizontal=True, key=f"{key}_t", label_visibility="collapsed")
    if kind == "close":
        return {"src": "close"}
    if kind == "number":
        return {"src": "num", "value": float(st.number_input("value", value=0.0, step=1.0, key=f"{key}_v",
                                                             label_visibility="collapsed", format="%.3f"))}
    k = st.selectbox("indicator", IND_KEYS, key=f"{key}_i", label_visibility="collapsed",
                     format_func=lambda x: f"{CATALOG[x].label}", help="Type to search.")
    ind = CATALOG[k]
    params = {}
    if ind.params:
        cols = st.columns(len(ind.params))
        for j, (pn, pt) in enumerate(ind.params.items()):
            with cols[j]:
                params[pn] = C._num(pn, pt[:4], f"{key}_p_{k}_{pn}", help=pt[4])
    st.caption(ind.hover)
    return {"src": "ind", "key": k, "params": params}


def conditions_ui() -> dict:
    c1, c2 = st.columns([3, 1])
    tpl = c1.selectbox("start from a template", ["—", *TEMPLATES], key="bt_c_tpl",
                       help="Classic published rules, ready to edit.")
    if c2.button("LOAD", key="bt_c_load", disabled=tpl == "—", use_container_width=True):
        C.queue(**conds_to_widgets(TEMPLATES[tpl]["conds"]))
    if tpl != "—":
        st.caption(TEMPLATES[tpl]["why"])
    conds = {}
    for slot in SLOTS:
        n0 = int(st.session_state.get(f"bt_c_{slot}_n", 1 if slot == "long_entry" else 0))
        with st.expander(f"{SLOT_LABEL[slot].upper()} · {n0} rule{'s' if n0 != 1 else ''}",
                         expanded=slot == "long_entry"):
            c1, c2 = st.columns(2)
            n = int(c1.number_input("rows", 0, MAX_ROWS, n0, 1, key=f"bt_c_{slot}_n",
                                    help="0 rows: for an exit slot, exit as soon as the entry stops being true; "
                                         "for an entry slot, never enter on that side."))
            join = c2.radio("join", ["AND", "OR"], horizontal=True, key=f"bt_c_{slot}_j",
                            help="AND: every row must be true. OR: any one row is enough.")
            rows = []
            for i in range(n):
                k = f"bt_c_{slot}_{i}"
                st.markdown(f"<div style='font-size:0.72rem;opacity:0.7;margin-top:0.4rem'>ROW {i + 1}</div>",
                            unsafe_allow_html=True)
                a = _operand_ui(f"{k}_a", allow_num=False)
                op = st.selectbox("operator", OPS, key=f"{k}_op", label_visibility="collapsed",
                                  help="'crosses above' is true only on the bar where A moves from below B to "
                                       "above it.")
                b = _operand_ui(f"{k}_b", allow_num=True)
                rows.append({"a": a, "op": op, "b": b})
            if rows:
                conds[slot] = {"join": join, "rows": rows}
    lines = sentence(conds)
    st.markdown(f"<div style='font-size:0.78rem;border-left:2px solid {T.teal};padding-left:0.5rem'>"
                + ("<br>".join(C.esc(x) for x in lines) if lines else "No entry rule yet: the strategy stays flat.")
                + "</div>", unsafe_allow_html=True)
    return conds


def _operand_widgets(key: str, ref: dict) -> dict:
    src = ref.get("src", "num")
    if src == "close":
        return {f"{key}_t": "close"}
    if src == "num":
        return {f"{key}_t": "number", f"{key}_v": float(ref.get("value", 0.0))}
    w = {f"{key}_t": "indicator", f"{key}_i": ref["key"]}
    for pn, pv in (ref.get("params") or {}).items():
        w[f"{key}_p_{ref['key']}_{pn}"] = pv
    return w


def conds_to_widgets(conds: dict) -> dict:
    w = {}
    for slot in SLOTS:
        g = conds.get(slot) or {}
        rows = g.get("rows", [])
        w[f"bt_c_{slot}_n"] = len(rows)
        w[f"bt_c_{slot}_j"] = g.get("join", "AND")
        for i, r in enumerate(rows):
            k = f"bt_c_{slot}_{i}"
            w.update(_operand_widgets(f"{k}_a", r["a"]))
            w.update(_operand_widgets(f"{k}_b", r["b"]))
            w[f"{k}_op"] = r["op"]
    return w


# ------------------------------------------------------------ regime rules ----
SWITCH_OPTS = ["main", "main (custom params)", "flat"]


def regime_rule_ui(src: str, main_rule: str | None) -> dict:
    on = st.toggle("condition on market regime", key="bt_ru_on",
                   help="Make the strategy depend on the regime: only trade in some regimes, size up or down by "
                        "regime, or switch to a different rule. " + G.REGIME_HELP)
    if not on:
        return {}
    from . import regime_panel as RP
    det, params, mb = RP.controls("bt_ru", default="vol", src=src)
    d = DETECTORS[det]
    labels = list(d.order) or []
    use = st.radio("use the regime to…", list(REGIME_USES), format_func=lambda u: REGIME_USES[u], key="bt_ru_use",
                   help="SharpeEdge lists four uses of a regime: filter, sizing, parameter adaptation and "
                        "allocation. SWITCH with 'main (custom params)' is parameter adaptation.")
    out = {"on": True, "det": det, "params": params, "min_bars": mb, "use": use}
    if use == "filter":
        default = labels[:-1] if len(labels) > 1 else labels
        out["allow"] = st.multiselect("trade only in", labels, default=default, key=f"bt_ru_allow_{det}",
                                      help="Outside these regimes the position is flat. BUILD › REGIMES then "
                                           "tests the filter against random filters with the same exposure.")
    elif use == "size":
        sc, cols = {}, st.columns(2)
        for i, lab in enumerate(labels):
            with cols[i % 2]:
                sc[lab] = float(st.number_input(f"× in {lab}", 0.0, 2.0, 1.0, 0.25, key=f"bt_ru_sc_{det}_{i}",
                                                help="Multiplier on the forecast in this regime (0 = flat)."))
        out["scale"] = sc
    else:
        sw = {}
        for i, lab in enumerate(labels):
            opts = SWITCH_OPTS + C.classic_keys() + [f"i_{k}" for k in CATALOG]
            pick = st.selectbox(f"in {lab}", opts, key=f"bt_ru_sw_{det}_{i}",
                                format_func=lambda k: k if k in SWITCH_OPTS else C.rule_label(k),
                                help="main = keep the signal you built; flat = no position; any other rule "
                                     "runs with its default parameters.")
            if pick == "main (custom params)" and main_rule in REGISTRY:
                sw[lab] = {"rule": main_rule, "params": rule_params_ui(f"bt_ru_sp_{det}_{i}", main_rule)}
            elif pick in ("main", "flat"):
                sw[lab] = pick
            else:
                sw[lab] = {"rule": pick, "params": {}}
        out["switch"] = sw
    return out


def regime_to_widgets(reg: dict, main_rule: str | None = None) -> dict:
    if not reg or not reg.get("on"):
        return {"bt_ru_on": False}
    det = reg.get("det", "vol")
    w = {"bt_ru_on": True, "bt_ru_det": det, "bt_ru_minbars": int(reg.get("min_bars", 3)),
         "bt_ru_use": reg.get("use", "filter")}
    for pn, pv in (reg.get("params") or {}).items():
        w[f"bt_ru_{det}_{pn}"] = pv
    labels = list(DETECTORS[det].order)
    if "allow" in reg:
        w[f"bt_ru_allow_{det}"] = list(reg["allow"])
    for i, lab in enumerate(labels):
        if lab in (reg.get("scale") or {}):
            w[f"bt_ru_sc_{det}_{i}"] = float(reg["scale"][lab])
        alt = (reg.get("switch") or {}).get(lab)
        if alt is not None:
            if isinstance(alt, str):
                w[f"bt_ru_sw_{det}_{i}"] = alt
            elif alt["rule"] == main_rule and alt.get("params"):
                w[f"bt_ru_sw_{det}_{i}"] = "main (custom params)"
                for pn, pv in alt["params"].items():
                    w[_pkey(f"bt_ru_sp_{det}_{i}", main_rule, pn)] = pv
            else:
                w[f"bt_ru_sw_{det}_{i}"] = alt["rule"]
    return w
