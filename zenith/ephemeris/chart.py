"""Game-board payload + render (TradingView lightweight-charts in an iframe).

Anti-leak rule (VELA's, kept): BEFORE the call the payload carries only the
visible candles -- rebased (when enabled), on a synthetic time axis, with no
ticker, name, dates or future bars. The future and the unblinding arrive
only in the reveal payload, which is built after the call is logged.
tests/test_ephemeris.py::test_decision_payload_leaks_nothing enforces it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import THEME
from .sampler import Chart

_TEMPLATE = Path(__file__).with_name("chart") / "board.html"
HEIGHT = 470


def _fmt_date(ts, tf: str) -> str:
    t = pd.Timestamp(ts)
    return t.strftime("%Y-%m-%d %H:%M") if tf in ("1H", "4H") else t.strftime("%Y-%m-%d")


def _r(x: float) -> float:
    return float(np.round(x, 6))


def payload(ch: Chart, *, rebase: bool = True, hide_dates: bool = True, hide_ticker: bool = True,
            direction: int = 0, sl: float | None = None, tp: float | None = None,
            stake: float = 0.0, result: dict | None = None, animate: bool = True) -> dict:
    """Dict for the board. `sl`/`tp` are RAW prices; result=None -> decision state."""
    f = 100.0 / ch.c[ch.vis0] if rebase else 1.0
    s = slice(ch.vis0, ch.t + 1)
    bars = [[_r(o * f), _r(h * f), _r(l * f), _r(c * f)]
            for o, h, l, c in zip(ch.o[s], ch.h[s], ch.l[s], ch.c[s])]
    header = f"<b>{html.escape(ch.tf.upper())}</b> · {ch.lookback} BARS · CALL {ch.horizon} AHEAD"
    if not hide_ticker:
        header = f"<b>{html.escape(ch.ticker)}</b> · " + header
    p = {"bars": bars, "horizon": ch.horizon, "rebased": rebase, "header": header,
         "direction": direction, "stake": stake,
         "sl": _r(sl * f) if sl is not None else None, "tp": _r(tp * f) if tp is not None else None}
    if not hide_dates:
        p["dates"] = [_fmt_date(x, ch.tf) for x in ch.ts[ch.vis0:ch.t + 1]]
    if result is None:
        return p

    fs = slice(ch.t + 1, ch.t + 1 + ch.horizon)
    p["future"] = [[_r(o * f), _r(h * f), _r(l * f), _r(c * f)]
                   for o, h, l, c in zip(ch.o[fs], ch.h[fs], ch.l[fs], ch.c[fs])]
    p["dates"] = [_fmt_date(x, ch.tf) for x in ch.ts[ch.vis0:ch.t + 1 + ch.horizon]]
    p["animate"] = animate
    p["result"] = {"win": bool(result["win"]), "pnl": float(result["pnl"]),
                   "exit_index": int(result["exit_index"]), "exit_reason": result["exit_reason"],
                   "exit_px": _r(result["exit_price"] * f), "ambiguous": bool(result["ambiguous"])}
    rows = [
        ("CLASS", html.escape(ch.cls + (f" · {ch.sector}" if ch.sector else ""))),
        ("WINDOW", f"{_fmt_date(ch.window_start, ch.tf)} → {_fmt_date(ch.end_date, ch.tf)}"),
        ("PRICE AT CALL", f"{ch.entry:,.4g}" if ch.entry < 10 else f"{ch.entry:,.2f}"),
        ("MOVE OVER HORIZON", f"{result['market_ret'] * 100:+.2f}%  ({result['outcome']})"),
        ("YOUR TRADE", f"{'LONG' if direction > 0 else 'SHORT'} {result['trade_ret'] * 100:+.2f}% · "
                       f"exit {result['exit_reason']} after {result['candles_held']} bar(s)"),
        ("MFE / MAE", f"{result['mfe'] * 100:+.2f}% / {result['mae'] * 100:+.2f}%"),
    ]
    if result.get("r_mult") is not None:
        rows.append(("R MULTIPLE", f"{result['r_mult']:+.2f}R"))
    for k, v in (result.get("context_rows") or []):
        rows.append((k, html.escape(str(v))))
    p["unblind"] = {"ticker": html.escape(ch.ticker), "name": html.escape(ch.name), "rows": rows,
                    "why": ("Same-candle stop/target touch — scored as STOP (conservative)."
                            if result["ambiguous"] else "")}
    return p


def board_html(p: dict, height: int = HEIGHT) -> str:
    tpl = _TEMPLATE.read_text(encoding="utf-8")
    subs = {"__BG__": THEME.bg, "__PANEL__": THEME.panel, "__GRID__": THEME.grid,
            "__TEAL__": THEME.teal, "__CORAL__": THEME.coral, "__MUSTARD__": THEME.mustard,
            "__MUTED__": THEME.muted, "__TEXT__": THEME.text, "__HEIGHT__": str(height),
            "__PAYLOAD__": json.dumps(p, separators=(",", ":")).replace("</", "<\\/")}
    for k, v in subs.items():
        tpl = tpl.replace(k, v)
    return tpl


def render(p: dict, height: int = HEIGHT) -> None:
    import streamlit.components.v1 as components
    components.html(board_html(p, height), height=height + 8, scrolling=False)
