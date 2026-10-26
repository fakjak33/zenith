"""App chrome shared by every page: the sidebar brand and foot, and the top bar
(breadcrumb, live market strip with sparklines, data-as-of)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from . import art
from . import ui_kit as uk
from .config import THEME as T

ROOT = Path(__file__).resolve().parent.parent
LOCKUP = ROOT / "static" / "zenith_lockup.svg"
MARK = ROOT / "static" / "zenith_mark.svg"
STRIP = ("SPY", "QQQ", "IWM", "TLT", "GLD")


@st.cache_resource(show_spinner=False)
def market():
    """The THETA RUN market table (real daily closes) doubles as the app's quote feed."""
    try:
        from .theta.market import Market
        return Market.load()
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def strip_items() -> tuple[list[dict], str]:
    m = market()
    if m is None:
        return [], "—"
    i = len(m) - 1
    out = []
    for t in STRIP:
        if t in m.data:
            c = m.data[t]["close"]
            out.append({"sym": t, "last": float(c[i]), "chg": float(c[i] / c[i - 1] - 1),
                        "spark": [float(x) for x in c[i - 30: i + 1]]})
    v = m.vix
    out.append({"sym": "VIX", "last": float(v[i] * 100), "chg": float(v[i] / v[i - 1] - 1),
                "spark": [float(x) * 100 for x in v[i - 30: i + 1]]})
    return out, str(m.dates[i].date())


def topbar(crumb: str) -> None:
    items, as_of = strip_items()
    st.markdown(uk.topbar(crumb, as_of, uk.ticker_strip(items) if items else ""), unsafe_allow_html=True)


def sidebar() -> None:
    """Logo via st.logo, plus a quiet foot under the navigation."""
    if LOCKUP.exists():
        st.logo(str(LOCKUP), size="large", icon_image=str(MARK) if MARK.exists() else None)
    with st.sidebar:
        items, as_of = strip_items()
        st.markdown(f'<div style="margin-top:1.4rem; opacity:0.9;">{art.orbits(150, seed=4)}</div>'
                    f'<div style="font-size:0.58rem; letter-spacing:0.22em; color:{T.faint}; margin-top:0.6rem;">'
                    f'<span style="color:{T.teal};">●</span> DATA AS OF {as_of}<br>FREE DATA · NOT ADVICE</div>',
                    unsafe_allow_html=True)
