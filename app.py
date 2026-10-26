"""ZENITH — Streamlit viewer.   Run:  streamlit run app.py

One global stylesheet + Altair theme (zenith.ui_theme), a sidebar navigation
built from one registry (zenith.nav) where each page runs only its own view, and
shared chrome (zenith.chrome): the sidebar brand and every page's top bar.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from zenith import chrome, nav
from zenith.auth import require_password
from zenith.ui_theme import CSS

_FAVICON = Path(__file__).parent / "assets" / "favicon.png"
_PAGE_ICON = str(_FAVICON) if _FAVICON.exists() else "▲"
st.set_page_config(page_title="ZENITH", layout="wide", page_icon=_PAGE_ICON,
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
require_password()
page = nav.build()
chrome.sidebar()
page.run()
