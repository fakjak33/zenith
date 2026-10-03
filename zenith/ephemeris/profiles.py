"""Low-friction profiles: type a handle -> you are that player.

  * The handle lives in `?player=<handle>` and in session state, so a bookmark
    IS a login. No password, email or sign-up form.
  * Typing an existing handle switches to it (unless it has a PIN).
  * Optional 4-digit PIN per profile, for shared deployments; verified once
    per browser session.
The whole app already sits behind ZENITH's shared `require_password()`.
"""

from __future__ import annotations

import streamlit as st

from .repo import Repository, normalize_handle, valid_handle

_KEY = "eph_player"          # session: the active player's normalized handle
_PIN_OK = "eph_pin_ok"       # session: set of handles whose PIN was verified


def _activate(handle: str) -> None:
    st.session_state[_KEY] = normalize_handle(handle)
    try:
        st.query_params["player"] = handle
    except Exception:
        pass


def _pin_verified(handle: str) -> bool:
    return normalize_handle(handle) in st.session_state.get(_PIN_OK, set())


def _mark_pin(handle: str) -> None:
    st.session_state.setdefault(_PIN_OK, set()).add(normalize_handle(handle))


def current_player(repo: Repository) -> dict | None:
    """The active player, resolving ?player= on first load. Renders the PIN
    prompt inline when a protected profile is requested."""
    handle = st.session_state.get(_KEY)
    if not handle:
        qp = st.query_params.get("player")
        if qp:
            handle = qp
    if not handle:
        return None
    p = repo.player(handle)
    if p is None:                    # bookmarked handle that no longer exists -> recreate (low friction)
        if valid_handle(handle) is None:
            p = repo.get_or_create_player(handle)
        else:
            return None
    if p.get("pin_hash") and not _pin_verified(p["handle"]):
        with st.form("eph_pin_form", clear_on_submit=True):
            pin = st.text_input(f"PIN for {p['display']}", type="password", max_chars=4)
            if st.form_submit_button("UNLOCK"):
                if repo.check_pin(p, pin):
                    _mark_pin(p["handle"])
                    _activate(p["display"])
                    st.rerun()
                st.error("Wrong PIN.")
        return None
    st.session_state[_KEY] = p["handle"]
    if st.query_params.get("player") != p["display"]:
        try:
            st.query_params["player"] = p["display"]
        except Exception:
            pass
    return p


def profile_bar(repo: Repository, player: dict | None) -> None:
    """Handle entry (first visit) or a compact switcher + PIN settings."""
    others = [p for p in repo.players() if not player or p["id"] != player["id"]]
    if player is None:
        c1, c2 = st.columns([2, 1])
        with c1:
            with st.form("eph_handle_form", border=False):
                h = st.text_input("Your handle", placeholder="type any name — that's it",
                                  max_chars=24, label_visibility="collapsed")
                go = st.form_submit_button("PLAY ▸", type="primary")
            if go:
                err = valid_handle(h)
                if err:
                    st.error(f"Handle: {err}.")
                else:
                    p = repo.get_or_create_player(h)
                    _activate(p["display"])
                    st.rerun()
        with c2:
            if others:
                pick = st.selectbox("or continue as", ["—"] + [p["display"] for p in others],
                                    key="eph_pick_first")
                if pick != "—":
                    st.session_state.pop("eph_pick_first", None)
                    _activate(pick)
                    st.rerun()
        st.caption("No sign-up: your handle is saved in the page URL, so bookmark it to come back.")
        return

    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        st.markdown(f"**PLAYER** · {player['display']}"
                    + (" · 🔒" if player.get("pin_hash") else ""))
    with c2:
        pick = st.selectbox("Switch player", ["switch player…"] + [p["display"] for p in others]
                            + ["+ new player"], key="eph_switch", label_visibility="collapsed")
        if pick == "+ new player":
            st.session_state.pop(_KEY, None)
            st.session_state.pop("eph_switch", None)
            st.query_params.pop("player", None)
            st.rerun()
        elif pick != "switch player…":
            st.session_state.pop("eph_switch", None)
            _activate(pick)
            st.rerun()
    with c3:
        with st.popover("PIN"):
            st.caption("Optional 4-digit PIN for shared machines. Leave blank to remove.")
            pin = st.text_input("New PIN", type="password", max_chars=4, key="eph_new_pin")
            if st.button("Save PIN", key="eph_save_pin"):
                if pin and not (pin.isdigit() and len(pin) == 4):
                    st.error("4 digits.")
                else:
                    repo.set_pin(player["id"], pin or None)
                    _mark_pin(player["handle"])
                    st.success("Saved." if pin else "PIN removed.")
