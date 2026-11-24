"""No-code strategy composition: BLEND, CONDITIONS and REGIME RULES.

All three are plain data on ``StrategySpec`` (JSON round-trips, stable hash)
and compile to the same thing every LAB step consumes: a causal Carver
forecast series.

BLEND       weighted average of several rules' scaled forecasts, scaled back
            up by the forecast diversification multiplier (Carver), capped ±20.
CONDITIONS  IF/AND/OR rows for four slots (long entry, long exit, short entry,
            short exit) → a stateful long/flat/short position → forecast ±10.
            An empty exit slot means "exit when the entry stops being true".
REGIME      after the forecast is built: FILTER (trade only in chosen
            regimes), SIZE (scale the forecast per regime) or SWITCH (a
            different rule per regime — also covers per-regime parameters).

Everything evaluated at bar t uses bars <= t; the engine trades at t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FORECAST_CAP, scale_params_for_tf
from ..core.combine import combine_forecasts, forecast_diversification_multiplier
from ..indicators import CATALOG
from ..rules import REGISTRY
from ..rules.base import Strategy

SLOTS = ("long_entry", "long_exit", "short_entry", "short_exit")
SLOT_LABEL = {"long_entry": "BUY when", "long_exit": "SELL (exit long) when",
              "short_entry": "SHORT when", "short_exit": "COVER (exit short) when"}
OPS = (">", "<", "crosses above", "crosses below")
CONDITION_FORECAST = 10.0


# --------------------------------------------------------------- operands ----
def operand_series(ref: dict, df: pd.DataFrame, tf: str = "Daily") -> pd.Series | float:
    """{"src": "close"} | {"src": "num", "value": x} | {"src": "ind", "key": k, "params": {...}}"""
    src = (ref or {}).get("src", "num")
    if src == "close":
        return df["close"]
    if src == "ind":
        ind = CATALOG[ref["key"]]
        p = scale_params_for_tf({**ind.defaults(), **(ref.get("params") or {})}, tf)
        return ind(df, **p)
    return float((ref or {}).get("value", 0.0))


def operand_text(ref: dict) -> str:
    src = (ref or {}).get("src", "num")
    if src == "close":
        return "Close"
    if src == "ind":
        ind = CATALOG[ref["key"]]
        p = {**ind.defaults(), **(ref.get("params") or {})}
        ps = ",".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in p.values())
        name = ind.key.upper().replace("_", " ")
        return f"{name}({ps})" if ps else name
    return f"{(ref or {}).get('value', 0):g}"


def row_mask(row: dict, df: pd.DataFrame, tf: str = "Daily") -> pd.Series:
    a = operand_series(row.get("a"), df, tf)
    b = operand_series(row.get("b"), df, tf)
    a = a if isinstance(a, pd.Series) else pd.Series(a, index=df.index)
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=df.index)
    op = row.get("op", ">")
    if op == ">":
        m = a > b
    elif op == "<":
        m = a < b
    elif op == "crosses above":
        m = (a > b) & (a.shift(1) <= b.shift(1))
    elif op == "crosses below":
        m = (a < b) & (a.shift(1) >= b.shift(1))
    else:
        raise ValueError(f"unknown op {op!r}")
    valid = a.notna() & b.notna()
    return (m & valid).astype(bool)


def group_mask(group: dict, df: pd.DataFrame, tf: str = "Daily") -> pd.Series | None:
    rows = [r for r in (group or {}).get("rows", []) if r]
    if not rows:
        return None
    masks = [row_mask(r, df, tf) for r in rows]
    out = masks[0]
    for m in masks[1:]:
        out = (out | m) if (group or {}).get("join", "AND") == "OR" else (out & m)
    return out


def conditions_state(conds: dict, df: pd.DataFrame, tf: str = "Daily") -> pd.Series:
    """Stateful position in {-1, 0, +1} from the four condition slots."""
    m = {s: group_mask(conds.get(s), df, tf) for s in SLOTS}
    n = len(df)
    false = np.zeros(n, dtype=bool)
    le = m["long_entry"].to_numpy() if m["long_entry"] is not None else false
    se = m["short_entry"].to_numpy() if m["short_entry"] is not None else false
    lx = m["long_exit"].to_numpy() if m["long_exit"] is not None else ~le
    sx = m["short_exit"].to_numpy() if m["short_exit"] is not None else ~se
    if m["long_entry"] is not None and m["long_exit"] is None:
        lx = ~le
    if m["short_entry"] is not None and m["short_exit"] is None:
        sx = ~se
    out = np.zeros(n)
    pos = 0.0
    for i in range(n):
        if pos > 0 and (lx[i] or se[i]):
            pos = 0.0
        elif pos < 0 and (sx[i] or le[i]):
            pos = 0.0
        if pos == 0:
            if le[i]:
                pos = 1.0
            elif se[i]:
                pos = -1.0
        out[i] = pos
    return pd.Series(out, index=df.index)


def sentence(conds: dict) -> list[str]:
    """Plain-English preview, one line per non-empty slot."""
    out = []
    for s in SLOTS:
        g = conds.get(s) or {}
        rows = [r for r in g.get("rows", []) if r]
        if not rows:
            if s.endswith("exit") and (conds.get(s.replace("exit", "entry")) or {}).get("rows"):
                out.append(f"{SLOT_LABEL[s]} the entry condition stops being true")
            continue
        j = f" {g.get('join', 'AND')} "
        out.append(SLOT_LABEL[s] + " " + j.join(f"{operand_text(r['a'])} {r['op']} {operand_text(r['b'])}"
                                              for r in rows))
    return out


def needs_of(spec) -> tuple:
    """Aux columns used anywhere in the spec (main rule, blend parts, conditions, regime switch)."""
    cols: set = set()
    mode = getattr(spec, "mode", "single")
    if mode == "single":
        cols |= set(getattr(REGISTRY.get(spec.rule), "needs", ()) or ())
    elif mode == "blend":
        for c in spec.blend:
            cols |= set(getattr(REGISTRY.get(c.get("rule")), "needs", ()) or ())
    elif mode == "conditions":
        for g in spec.conditions.values():
            for r in (g or {}).get("rows", []):
                for side in ("a", "b"):
                    ref = r.get(side) or {}
                    if ref.get("src") == "ind":
                        cols |= set(CATALOG[ref["key"]].needs)
    elif mode == "ml":
        from ..ml.strategy import needs as ml_needs
        cols |= ml_needs(getattr(spec, "ml", {}))
    reg = getattr(spec, "regime", {}) or {}
    if reg.get("on"):
        from ..regime import DETECTORS
        d = DETECTORS.get(reg.get("det"))
        if d:
            cols |= set(d.needs)
        for alt in (reg.get("switch") or {}).values():
            if isinstance(alt, dict) and alt.get("rule") in REGISTRY:
                cols |= set(getattr(REGISTRY[alt["rule"]], "needs", ()) or ())
    return tuple(c for c in ("ref_close", "vix", "vix3m") if c in cols)


# ------------------------------------------------------------- strategies ----
class BlendStrategy(Strategy):
    key = "_blend"
    label = "Blend"
    params: dict = {}

    def __init__(self, parts: list[dict], tf: str = "Daily"):
        self.values = {}
        self.parts = [p for p in parts if p.get("rule") in REGISTRY and float(p.get("weight", 0)) > 0]
        self.tf = tf

    def rules(self) -> list[tuple[str, Strategy, float]]:
        out = []
        for i, p in enumerate(self.parts):
            s = REGISTRY[p["rule"]](**scale_params_for_tf(p.get("params") or {}, self.tf))
            out.append((f"{i + 1}·{s.describe()}", s, float(p.get("weight", 1.0))))
        return out

    def components(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: s.forecast(ohlcv) for name, s, _ in self.rules()})

    def forecast(self, ohlcv: pd.DataFrame, scalar: float | None = None) -> pd.Series:
        rs = self.rules()
        if not rs:
            return pd.Series(0.0, index=ohlcv.index)
        fc = {name: s.forecast(ohlcv) for name, s, _ in rs}
        return combine_forecasts(fc, {name: w for name, _, w in rs}).reindex(ohlcv.index).fillna(0.0)

    def fdm(self, ohlcv: pd.DataFrame) -> float:
        rs = self.rules()
        if len(rs) < 2:
            return 1.0
        w = np.array([x[2] for x in rs], dtype=float)
        return forecast_diversification_multiplier(self.components(ohlcv).fillna(0.0), w / w.sum())

    def raw_forecast(self, ohlcv):
        return self.forecast(ohlcv)

    def indicator_lines(self, ohlcv):
        out = {}
        for name, s, _ in self.rules():
            out.update({f"{name.split('·')[0]}·{k}": v for k, v in s.indicator_lines(ohlcv).items()})
        return out

    def describe(self) -> str:
        rs = self.rules()
        tot = sum(w for *_, w in rs) or 1.0
        return "BLEND " + " + ".join(f"{w / tot:.0%} {n.split('·', 1)[1]}" for n, _, w in rs)


class ConditionStrategy(Strategy):
    key = "_conditions"
    label = "Conditions"
    params: dict = {}

    def __init__(self, conds: dict, tf: str = "Daily"):
        self.values = {}
        self.conds = conds or {}
        self.tf = tf

    def forecast(self, ohlcv: pd.DataFrame, scalar: float | None = None) -> pd.Series:
        return conditions_state(self.conds, ohlcv, self.tf) * CONDITION_FORECAST

    def raw_forecast(self, ohlcv):
        return self.forecast(ohlcv)

    def indicator_lines(self, ohlcv):
        out = {}
        for g in self.conds.values():
            for r in (g or {}).get("rows", []):
                for side in ("a", "b"):
                    ref = r.get(side) or {}
                    if ref.get("src") == "ind" and CATALOG[ref["key"]].overlay:
                        out[operand_text(ref)] = operand_series(ref, ohlcv, self.tf)
        return out

    def describe(self) -> str:
        s = sentence(self.conds)
        return "IF " + " · ".join(s) if s else "CONDITIONS (empty)"


# ------------------------------------------------------------- regime use ----
REGIME_USES = {"filter": "FILTER — trade only in chosen regimes",
               "size": "SIZE — scale exposure per regime",
               "switch": "SWITCH — a different rule per regime"}


def regime_labels(reg: dict, ohlcv: pd.DataFrame) -> pd.Series:
    from ..regime import detect
    return detect(reg.get("det", "vol"), ohlcv, min_bars=int(reg.get("min_bars", 3)), **(reg.get("params") or {}))


def apply_regime(reg: dict, fc: pd.Series, ohlcv: pd.DataFrame, tf: str = "Daily") -> pd.Series:
    """Forecast after the regime rule. Labels at t are known at the close of t (engine trades t+1)."""
    if not reg or not reg.get("on"):
        return fc
    lab = regime_labels(reg, ohlcv).reindex(fc.index)
    use = reg.get("use", "filter")
    if use == "filter":
        allow = set(reg.get("allow") or [])
        return fc.where(lab.isin(allow), 0.0)
    if use == "size":
        sc = reg.get("scale") or {}
        mult = lab.map(lambda x: float(sc.get(x, 1.0)) if isinstance(x, str) else 0.0).astype(float)
        return (fc * mult).clip(-FORECAST_CAP, FORECAST_CAP)
    if use == "switch":
        out = fc.copy()
        for label, alt in (reg.get("switch") or {}).items():
            m = lab == label
            if not m.any():
                continue
            if alt in (None, "", "main") or (isinstance(alt, dict) and alt.get("rule") == "main"):
                continue
            if alt == "flat" or (isinstance(alt, dict) and alt.get("rule") == "flat"):
                out = out.where(~m, 0.0)
                continue
            rule = alt["rule"] if isinstance(alt, dict) else alt
            params = alt.get("params", {}) if isinstance(alt, dict) else {}
            if rule not in REGISTRY:
                continue
            alt_fc = REGISTRY[rule](**scale_params_for_tf(params, tf)).forecast(ohlcv).reindex(fc.index)
            out = out.where(~m, alt_fc)
        return out.fillna(0.0)
    return fc


def random_mask_test(base_r: pd.Series, labels: pd.Series, allow: set, n: int = 500, seed: int = 0,
                     periods: int = 252) -> dict:
    """Is the regime FILTER better than luck? Compare it with ``n`` random masks that keep the same exposure
    AND the same spell structure (the real mask, circularly shifted by a random offset).

    Approximation: the base strategy's returns are switched on/off by the mask known at the previous bar, so
    extra trading costs from the filter are ignored (they would only make the real filter look worse)."""
    lab = labels.reindex(base_r.index)
    mask = lab.isin(allow).astype(float).shift(1).fillna(0.0).to_numpy()
    r = base_r.fillna(0.0).to_numpy()

    def sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(periods)) if sd > 1e-12 else 0.0

    actual = sharpe(r * mask)
    rng = np.random.default_rng(seed)
    lo, hi = len(r) // 20, len(r) - len(r) // 20
    shifts = rng.integers(max(lo, 1), max(hi, 2), size=n)
    sims = np.array([sharpe(r * np.roll(mask, int(k))) for k in shifts])
    return {"actual": actual, "sims": sims, "p": float((np.sum(sims >= actual) + 1) / (n + 1)),
            "exposure": float(mask.mean()), "unfiltered": sharpe(r)}
