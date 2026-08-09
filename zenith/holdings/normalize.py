"""Pure normalisation: raw holdings rows -> stable, classified positions.

No network, no I/O. Everything here is unit-testable offline.

The central idea is POSITION IDENTITY. A futures book rolls quarterly, so the
dated contract is the wrong identity: DBMF's 2-year note position appears as
TUU6 in August and TUZ6 in September, and naively diffing those two snapshots
would report a closed position plus a brand-new one when nothing changed. We
therefore key every position on its CONTRACT ROOT (TU, ES, GC, EC …) and treat
the dated contract as an attribute that can change under it.
"""

from __future__ import annotations

import re
from datetime import date

# --- contract roots ---------------------------------------------------------
# Bloomberg-style roots as they appear in this family of custodian files.
# `hint` is a regex the security name should match; a mismatch does not
# override the mapping but raises a flag, so a broker changing a ticker's
# meaning is surfaced rather than silently mislabelled.
#   root: (display name, asset class, sub class, name hint)
ROOTS: dict[str, tuple[str, str, str, str]] = {
    # rates
    "TU": ("US 2-Year Note", "rates", "US front", r"2\s*YR|TWO"),
    "FV": ("US 5-Year Note", "rates", "US belly", r"5\s*YR|FIVE"),
    "TY": ("US 10-Year Note", "rates", "US belly", r"10\s*YR|TEN"),
    "UXY": ("US Ultra 10-Year", "rates", "US belly", r"ULTRA|10Y"),
    "US": ("US Long Bond", "rates", "US long", r"LONG BOND|BOND"),
    "WN": ("US Ultra Bond", "rates", "US long", r"ULTRA"),
    "RX": ("Euro-Bund", "rates", "Europe", r"BUND|EURO.*BUND"),
    "OE": ("Euro-Bobl", "rates", "Europe", r"BOBL"),
    "DU": ("Euro-Schatz", "rates", "Europe", r"SCHATZ"),
    "IK": ("Italian BTP", "rates", "Europe", r"BTP|ITAL"),
    "JB": ("Japan 10-Year JGB", "rates", "Asia", r"JGB|JAPAN"),
    "XM": ("Australia 10-Year", "rates", "Asia-Pacific", r"AUST"),
    "ED": ("Eurodollar", "rates", "US front", r"EURODOLLAR"),
    "SFR": ("SOFR", "rates", "US front", r"SOFR"),
    # equity
    "ES": ("S&P 500 E-mini", "equity", "US large cap", r"S\+?P\s*500|SPX|EMINI"),
    "NQ": ("Nasdaq 100 E-mini", "equity", "US growth", r"NASDAQ|NDX"),
    "RTY": ("Russell 2000 E-mini", "equity", "US small cap", r"RUSSELL|RTY"),
    "DM": ("Dow E-mini", "equity", "US large cap", r"DOW|DJIA"),
    "MFS": ("MSCI EAFE", "equity", "Developed ex-US", r"EAFE"),
    "MES": ("MSCI Emerging Markets", "equity", "Emerging markets", r"EMGMKT|EMERG"),
    "NK": ("Nikkei 225", "equity", "Japan", r"NIKKEI"),
    "VG": ("Euro Stoxx 50", "equity", "Europe", r"STOXX|DJ EURO"),
    "GX": ("DAX", "equity", "Europe", r"DAX"),
    "Z": ("FTSE 100", "equity", "UK", r"FTSE"),
    "PT": ("S&P/TSX 60", "equity", "Canada", r"TSX"),
    # fx (all quoted against USD)
    "EC": ("Euro FX", "fx", "EUR/USD", r"EURO"),
    "JY": ("Japanese Yen", "fx", "JPY/USD", r"YEN"),
    "BP": ("British Pound", "fx", "GBP/USD", r"POUND|BRITISH|STERLING"),
    "AD": ("Australian Dollar", "fx", "AUD/USD", r"AUSTRAL"),
    "CD": ("Canadian Dollar", "fx", "CAD/USD", r"CANAD"),
    "SF": ("Swiss Franc", "fx", "CHF/USD", r"SWISS|FRANC"),
    "NV": ("New Zealand Dollar", "fx", "NZD/USD", r"ZEALAND"),
    "MP": ("Mexican Peso", "fx", "MXN/USD", r"PESO|MEXIC"),
    "BR": ("Brazilian Real", "fx", "BRL/USD", r"REAL|BRAZIL"),
    # commodities
    "GC": ("Gold", "commodity", "Precious metals", r"GOLD"),
    "SI": ("Silver", "commodity", "Precious metals", r"SILVER"),
    "PL": ("Platinum", "commodity", "Precious metals", r"PLATIN"),
    "HG": ("Copper", "commodity", "Base metals", r"COPPER"),
    "CL": ("WTI Crude Oil", "commodity", "Energy", r"WTI|CRUDE"),
    "CO": ("Brent Crude Oil", "commodity", "Energy", r"BRENT"),
    "NG": ("Natural Gas", "commodity", "Energy", r"NAT.*GAS"),
    "HO": ("Heating Oil", "commodity", "Energy", r"HEAT"),
    "XB": ("RBOB Gasoline", "commodity", "Energy", r"RBOB|GASOLINE"),
    "C": ("Corn", "commodity", "Agriculture", r"CORN"),
    "W": ("Wheat", "commodity", "Agriculture", r"WHEAT"),
    "S": ("Soybeans", "commodity", "Agriculture", r"SOYBEAN"),
    "SM": ("Soybean Meal", "commodity", "Agriculture", r"MEAL"),
    "BO": ("Soybean Oil", "commodity", "Agriculture", r"SOYBEAN OIL|BEAN OIL"),
    "SB": ("Sugar", "commodity", "Agriculture", r"SUGAR"),
    "KC": ("Coffee", "commodity", "Agriculture", r"COFFEE"),
    "CC": ("Cocoa", "commodity", "Agriculture", r"COCOA"),
    "CT": ("Cotton", "commodity", "Agriculture", r"COTTON"),
    "LC": ("Live Cattle", "commodity", "Livestock", r"CATTLE"),
    "LH": ("Lean Hogs", "commodity", "Livestock", r"HOG"),
}

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

_CONTRACT_RE = re.compile(rf"^([A-Z0-9]{{1,4}}?)([{''.join(MONTH_CODES)}])(\d)$")

# The synthetic identity every cash/collateral row collapses into. DBMF holds
# 6-8 separate Treasury bill CUSIPs that roll constantly; tracking them
# individually would flood the change feed with noise that says nothing about
# positioning, so they are aggregated into one collateral line.
CASH_ID = "CASH_TBILL"
CASH_NAME = "Treasury bills (collateral)"

# Row that states the fund's net assets rather than a holding.
TOTAL_ROW_RE = re.compile(r"TOTAL\s+NET\s+ASSETS", re.I)

_CASH_RE = re.compile(r"TREASURY\s*BILL|T-?BILL|MONEY\s*MARKET|^CASH\b|"
                      r"REPURCHASE|REPO\b", re.I)

# Fallback classifier, applied to the security name when the root is unknown.
# Order matters: the first match wins.
_NAME_RULES: tuple[tuple[str, str, str], ...] = (
    (r"YEN|EURO FX|POUND|STERLING|FRANC|PESO|REAL|AUSTRAL|CANAD|ZEALAND|"
     r"CURR FUT|KRONA|KRONE|RAND|WON|YUAN|RENMINBI", "fx", "Currency"),
    (r"BOND|NOTE|BUND|BOBL|SCHATZ|GILT|JGB|BTP|OAT|SOFR|EURODOLLAR|"
     r"TREASURY(?!\s*BILL)|YR ", "rates", "Rates"),
    (r"CRUDE|BRENT|GAS|GOLD|SILVER|COPPER|PLATIN|CORN|WHEAT|SOYBEAN|SUGAR|"
     r"COFFEE|COCOA|COTTON|CATTLE|HOG|GASOLINE|RBOB|HEAT|ALUMIN|ZINC|NICKEL",
     "commodity", "Commodity"),
    (r"EMINI|E-MINI|MSCI|INDEX|NIKKEI|DAX|STOXX|FTSE|S\+?P|NASDAQ|RUSSELL|"
     r"TOPIX|HANG SENG|KOSPI|TSX|IBEX|CAC|SMI\b", "equity", "Equity index"),
)

ASSET_CLASSES = ("equity", "rates", "fx", "commodity", "collateral", "unclassified")

# Asset classes that represent a directional view. Collateral does not.
DIRECTIONAL = ("equity", "rates", "fx", "commodity", "unclassified")


def parse_contract(ticker: str, asof: date | None = None) -> dict:
    """Split a futures ticker into root / expiry.

    ``TUU6`` -> ``{"root": "TU", "expiry": "2026-09", "is_future": True}``.

    The single-digit year is resolved against ``asof``: the decade that puts
    the contract on or after the snapshot year (allowing one year of slack for
    a contract that expired very recently).
    """
    t = (ticker or "").strip().upper()
    if not t or t == "-":
        return {"root": "", "expiry": None, "is_future": False}
    m = _CONTRACT_RE.match(t)
    if not m:
        # A plain security ticker (an ETF, say) is its own identity.
        return {"root": t, "expiry": None, "is_future": False}
    root, mcode, digit = m.group(1), m.group(2), int(m.group(3))
    ref = (asof or date.today()).year
    year = ref - (ref % 10) + digit
    while year < ref - 1:
        year += 10
    return {"root": root, "expiry": f"{year:04d}-{MONTH_CODES[mcode]:02d}",
            "is_future": True}


def classify(root: str, name: str) -> tuple[str, str, str, list[str]]:
    """Return ``(display, asset_class, sub_class, flags)`` for one position."""
    nm = (name or "").strip()
    flags: list[str] = []
    if _CASH_RE.search(nm):
        return CASH_NAME, "collateral", "Cash & equivalents", flags

    meta = ROOTS.get(root)
    if meta:
        display, cls, sub, hint = meta
        if hint and not re.search(hint, nm, re.I):
            # Mapping still applies — but say so, loudly, in status.
            flags.append(f"root_name_mismatch:{root}")
        return display, cls, sub, flags

    for pattern, cls, sub in _NAME_RULES:
        if re.search(pattern, nm, re.I):
            flags.append(f"unmapped_root:{root or nm[:20]}")
            return _pretty(nm), cls, sub, flags

    flags.append(f"unclassified:{root or nm[:20]}")
    return _pretty(nm), "unclassified", "Unclassified", flags


def _pretty(name: str) -> str:
    """Tidy a raw custodian security name for display.

    ``"US 2YR NOTE (CBT) SEP26"`` -> ``"US 2yr Note (Cbt)"`` is worse than the
    original, so we only strip the trailing contract month and squeeze spaces.
    """
    s = re.sub(r"\s+", " ", (name or "").strip())
    s = re.sub(r"\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\d{2}$",
               "", s, flags=re.I)
    return s.strip() or "Unnamed position"


def direction_of(notional: float | None) -> str:
    if notional is None or notional == 0:
        return "flat"
    return "long" if notional > 0 else "short"


def normalize_rows(rows: list[dict], asof: date | None = None) -> dict:
    """Normalise parsed rows into positions + NAV + quality flags.

    ``rows`` come from a source adapter and carry the raw strings already
    coerced to numbers: ``value_date, security_name, cusip, ticker, qty,
    notional, weight``.

    Returns ``{"positions": [...], "nav": float|None, "flags": [...],
    "n_rows": int}``. Positions are aggregated by identity, so duplicate rows
    (the Treasury bill sleeve, or a genuine duplicate) are summed rather than
    silently overwriting one another.
    """
    flags: list[str] = []
    nav: float | None = None
    agg: dict[str, dict] = {}
    seen_keys: set[tuple[str, str]] = set()

    for r in rows:
        name = (r.get("security_name") or "").strip()
        if TOTAL_ROW_RE.search(name):
            nav = r.get("notional")
            continue
        if not name:
            flags.append("blank_security_name")
            continue

        ticker = (r.get("ticker") or "").strip().upper()
        cusip = (r.get("cusip") or "").strip()
        contract = parse_contract(ticker, asof)
        display, cls, sub, cflags = classify(contract["root"], name)
        flags.extend(cflags)

        pid = CASH_ID if cls == "collateral" else (contract["root"] or name[:24].upper())

        key = (ticker or cusip or name, cusip)
        if key in seen_keys and cls != "collateral":
            flags.append(f"duplicate_row:{pid}")
        seen_keys.add(key)

        slot = agg.setdefault(pid, {
            "id": pid, "name": display, "asset_class": cls, "sub_class": sub,
            "contract": ticker if contract["is_future"] else "",
            "expiry": contract["expiry"], "cusip": cusip,
            "instrument": "future" if contract["is_future"] else
                          ("cash" if cls == "collateral" else "security"),
            "raw_name": name, "qty": 0.0, "notional": 0.0, "weight": 0.0,
            "n_lots": 0,
        })
        slot["qty"] += float(r.get("qty") or 0.0)
        slot["notional"] += float(r.get("notional") or 0.0)
        slot["weight"] += float(r.get("weight") or 0.0)
        slot["n_lots"] += 1
        # Keep the front contract's identity for a rolled/aggregated line.
        if contract["is_future"] and not slot["contract"]:
            slot["contract"] = ticker
            slot["expiry"] = contract["expiry"]

    positions = []
    for p in agg.values():
        p["direction"] = direction_of(p["notional"])
        p["notional"] = round(p["notional"], 2)
        p["weight"] = round(p["weight"], 6)
        p["qty"] = round(p["qty"], 2)
        positions.append(p)

    # Weight as published can drift from notional/NAV (rounding, and the
    # published column is quoted to 2dp). Recompute so the arithmetic in the
    # UI is internally consistent, and keep the published value alongside.
    if nav:
        for p in positions:
            p["weight_published"] = p["weight"]
            p["weight"] = round(p["notional"] / nav, 6)

    positions.sort(key=lambda p: -abs(p["weight"]))
    return {"positions": positions, "nav": nav, "flags": sorted(set(flags)),
            "n_rows": len(rows)}


def exposure_summary(positions: list[dict]) -> dict:
    """Long / short / net / gross exposure, collateral excluded."""
    live = [p for p in positions if p["asset_class"] in DIRECTIONAL]
    longs = [p for p in live if p["weight"] > 0]
    shorts = [p for p in live if p["weight"] < 0]
    collateral = sum(p["weight"] for p in positions
                     if p["asset_class"] == "collateral")
    long_w = sum(p["weight"] for p in longs)
    short_w = sum(p["weight"] for p in shorts)
    by_class: dict[str, dict] = {}
    for p in live:
        b = by_class.setdefault(p["asset_class"],
                                {"asset_class": p["asset_class"], "long": 0.0,
                                 "short": 0.0, "net": 0.0, "gross": 0.0, "n": 0})
        b["long" if p["weight"] > 0 else "short"] += p["weight"]
        b["net"] += p["weight"]
        b["gross"] += abs(p["weight"])
        b["n"] += 1
    for b in by_class.values():
        for k in ("long", "short", "net", "gross"):
            b[k] = round(b[k], 6)
    biggest = max(live, key=lambda p: abs(p["weight"]), default=None)
    return {
        "long": round(long_w, 6),
        "short": round(short_w, 6),
        "net": round(long_w + short_w, 6),
        "gross": round(long_w - short_w, 6),
        "collateral": round(collateral, 6),
        "n_positions": len(live),
        "n_long": len(longs),
        "n_short": len(shorts),
        "largest": {"id": biggest["id"], "name": biggest["name"],
                    "weight": biggest["weight"],
                    "direction": biggest["direction"]} if biggest else None,
        "by_asset_class": sorted(by_class.values(), key=lambda b: -b["gross"]),
    }
