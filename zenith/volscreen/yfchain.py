"""yfinance fallback: rebuild a Cboe-shaped payload so one code path parses both.

Only used when the Cboe CDN refuses us (circuit breaker tripped) — slower
(one request per expiry) and without greeks, so it is capped to the first
MAX_EXP expiries out to ~400 days. IVs are recomputed from mid either way.
"""

from __future__ import annotations

from datetime import date, datetime

MAX_EXP = 10


def payload(ticker: str, max_exp: int = MAX_EXP) -> dict | None:
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
        hist = tk.history(period="5d", auto_adjust=False)
    except Exception:
        return None
    if not exps or hist is None or hist.empty:
        return None
    today = date.today()
    pick = [e for e in exps if 0 <= (date.fromisoformat(e) - today).days <= 400]
    if len(pick) > max_exp:                       # keep the near end dense, thin the far end
        near = pick[: max_exp - 3]
        far = pick[max_exp - 3:]
        pick = near + [far[len(far) // 3], far[2 * len(far) // 3], far[-1]]
    opts = []
    root = ticker.replace("-", "").replace(".", "")
    for e in pick:
        try:
            ch = tk.option_chain(e)
        except Exception:
            continue
        d = date.fromisoformat(e)
        for cp, frame in (("C", ch.calls), ("P", ch.puts)):
            for _, o in frame.iterrows():
                k = float(o.get("strike", 0) or 0)
                occ = f"{root}{d:%y%m%d}{cp}{int(round(k * 1000)):08d}"
                opts.append({"option": occ, "bid": float(o.get("bid") or 0), "ask": float(o.get("ask") or 0),
                             "bid_size": None, "ask_size": None,
                             "iv": float(o.get("impliedVolatility") or 0),
                             "open_interest": float(o.get("openInterest") or 0),
                             "volume": float(o.get("volume") or 0), "delta": None, "gamma": None,
                             "vega": None, "theta": None, "theo": None,
                             "last_trade_price": float(o.get("lastPrice") or 0)})
    if not opts:
        return None
    bar = hist.iloc[-1]
    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return {"timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S"), "symbol": ticker, "source": "yfinance",
            "data": {"symbol": ticker, "security_type": "stock", "current_price": float(bar["Close"]),
                     "open": float(bar["Open"]), "high": float(bar["High"]), "low": float(bar["Low"]),
                     "close": float(bar["Close"]), "prev_day_close": prev, "volume": float(bar["Volume"]),
                     "iv30": None, "options": opts}}
