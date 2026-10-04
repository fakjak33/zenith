"""Keyboard shortcuts for EPHEMERIS: ↑ = UP, ↓ = DOWN, N = next chart,
S = skip the reveal animation.

Streamlit has no key bindings, so a zero-height component installs ONE
keydown listener on the parent page (guarded by a flag, so reruns never stack
listeners). It acts only while the EPHEMERIS marker element is visible (the
tab is open) and never while you are typing in an input. The chart iframe
forwards its own key presses up (see board.html), so shortcuts keep working
after clicking the chart.
"""

from __future__ import annotations

import streamlit as st

MARKER_ID = "eph-keys-on"
HINT = "Keys: ↑ UP · ↓ DOWN · N next chart · S skip reveal"

_JS = """
<script>
(function () {
  const P = window.parent;
  if (!P || P.__ephKeys) return;
  P.__ephKeys = true;
  const D = P.document;
  const visible = el => !!el && el.offsetParent !== null;
  const button = labels => [...D.querySelectorAll('button')]
      .filter(visible).find(b => labels.some(l => (b.innerText || '').includes(l)));
  D.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target || {};
    const tag = (t.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || t.isContentEditable) return;
    if (!visible(D.getElementById('%MARKER%'))) return;
    let b = null;
    if (e.key === 'ArrowUp') b = button(['UP / LONG']);
    else if (e.key === 'ArrowDown') b = button(['DOWN / SHORT']);
    else if (e.key === 'n' || e.key === 'N') b = button(['NEXT CHART', "SEE TODAY"]);
    else if (e.key === 's' || e.key === 'S') {
      for (const f of D.querySelectorAll('iframe')) {
        try {
          const sk = f.contentDocument && f.contentDocument.getElementById('skip');
          if (sk && sk.style.display !== 'none') { sk.click(); e.preventDefault(); return; }
        } catch (err) {}
      }
      return;
    }
    if (b) { e.preventDefault(); b.click(); }
  }, true);
})();
</script>
""".replace("%MARKER%", MARKER_ID)


def install() -> None:
    """Render the visibility marker + (once per page) the listener."""
    import streamlit.components.v1 as components
    st.markdown(f'<div id="{MARKER_ID}" style="height:0;overflow:hidden"></div>', unsafe_allow_html=True)
    components.html(_JS, height=0)
