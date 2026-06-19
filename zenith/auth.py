"""Password gate (set `app_password` in Streamlit secrets; open when unset)."""

from __future__ import annotations

import hmac

import streamlit as st


def require_password() -> bool:
    try:
        password = st.secrets.get("app_password", None) or None
    except Exception:
        password = None
    if password is None:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.markdown("### 🔒 Zenith — restricted")
    entered = st.text_input("Password", type="password")
    if not entered:
        st.stop()
    if hmac.compare_digest(entered, str(password)):
        st.session_state["authenticated"] = True
        st.rerun()
    st.error("Incorrect password.")
    st.stop()
    return False
