"""ZENITH — Streamlit viewer.   Run:  streamlit run app.py

ZENITH//TERM: one global stylesheet + Altair theme (zenith.ui_theme), then a
grouped top navigation (zenith.nav) where each page runs only its own view.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from zenith import nav
from zenith.auth import require_password
from zenith.ui_theme import BANNER, CSS

_FAVICON = Path(__file__).parent / "assets" / "favicon.png"
_PAGE_ICON = str(_FAVICON) if _FAVICON.exists() else "▲"
st.set_page_config(page_title="ZENITH // TERM", layout="wide", page_icon=_PAGE_ICON)
st.markdown(CSS, unsafe_allow_html=True)
require_password()
page = nav.build()
st.markdown(BANNER, unsafe_allow_html=True)
page.run()
