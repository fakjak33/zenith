"""StrategySpec — the one object every LAB step runs from.

A spec is plain data (JSON round-trips losslessly) and hashes stably, so the
same click-built strategy is the same *trial* everywhere: the trial counter,
the result cache, the OOS reveal log and the saved-spec library all key on
``spec.hash()``. Notes (hypothesis text) are excluded from the hash — editing
your reasoning is not a new trial.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import pandas as pd

from .config import FUTURES, BacktestConfig, RiskConfig, scale_params_for_tf
from .engine.engine import BacktestResult, run_single
from .rules import REGISTRY

# --------------------------------------------------------------- filters ----
# Each filter is a causal gate in [0, 1] applied to the forecast at bar t, using
# data up to and including t (the engine then trades at t+1). Gates never add
# exposure; they only remove it, so a filter can be judged by what it takes away.
FILTERS: dict[str, dict] = {
    "trend": {"label": "Trend filter", "params": {"length": (200, 20, 400, 10)},
              "help": "Longs only above the N-day average, shorts only below it."},
    "volregime": {"label": "Vol regime", "params": {"lookback": (252, 60, 1000, 10), "max_pct": (80, 10, 100, 5)},
                  "help": "Stand aside when 20-day realised vol is above this percentile of its own past."},
    "momconfirm": {"label": "Momentum confirm", "params": {"lookback": (252, 20, 500, 10)},
                   "help": "Only trade in the direction of the trailing N-day return."},
    "vix": {"label": "VIX cap", "params": {"max_level": (30, 12, 80, 1)},
            "help": "No new longs while VIX closes above this level (market data only)."},
}


def _trend_gate(close: pd.Series, fc: pd.Series, length: int) -> pd.Series:
    sma = close.rolling(int(length), min_periods=int(length)).mean()
    up, dn = close > sma, close < sma
    return ((fc > 0) & up) | ((fc < 0) & dn) | (fc == 0)


def _volregime_gate(close: pd.Series, lookback: int, max_pct: float) -> pd.Series:
    rv = close.pct_change().rolling(20, min_periods=20).std()
    pct = rv.rolling(int(lookback), min_periods=60).rank(pct=True) * 100
    return (pct <= max_pct) | pct.isna()


def _mom_gate(close: pd.Series, fc: pd.Series, lookback: int) -> pd.Series:
    past = close / close.shift(int(lookback)) - 1
    return (np.sign(past) == np.sign(fc)) | (fc == 0)


def _vix_gate(index: pd.DatetimeIndex, fc: pd.Series, vix: pd.Series | None, max_level: float) -> pd.Series:
    if vix is None or vix.empty:
        return pd.Series(True, index=index)
    v = vix.reindex(index).ffill()
    return ~((fc > 0) & (v > max_level))


def apply_filters(fc: pd.Series, ohlcv: pd.DataFrame, filters: list[dict],
                  aux: dict | None = None) -> pd.Series:
    """Forecast after every *enabled* filter gate (logical AND)."""
    aux = aux or {}
    close = ohlcv["close"].reindex(fc.index)
    gate = pd.Series(True, index=fc.index)
    for f in filters:
        if not f.get("on", True):
            continue
        p = f.get("params", {})
        k = f["key"]
        if k == "trend":
            g = _trend_gate(close, fc, p.get("length", 200))
        elif k == "volregime":
            g = _volregime_gate(close, p.get("lookback", 252), p.get("max_pct", 80))
        elif k == "momconfirm":
            g = _mom_gate(close, fc, p.get("lookback", 252))
        elif k == "vix":
            g = _vix_gate(fc.index, fc, aux.get("vix"), p.get("max_level", 30))
        else:
            continue
        gate &= g.fillna(False).astype(bool)
    return fc.where(gate, 0.0)


# ------------------------------------------------------------------ spec ----
# Synthetic kinds that learn from a real ticker (see data/synthetic.py).
TICKER_SYNTH = ("bootstrap", "shuffle", "gbm_fit", "regime_fit")
# Fields added after specs were first saved; omitted from the hash at these values.
_LATE_DEFAULTS = {"fixed_pct": 1.0, "risk_pct": 0.01, "scale_by_forecast": True, "max_leverage": 0.0,
                  "seed": 42, "synth_base": "SPY", "ref_ticker": "SPY",
                  "mode": "single", "blend": [], "conditions": {}, "regime": {}, "ml": {}}

# Aux series a rule can ask for (Strategy.needs) → the frame column it is joined as.
AUX_COLUMNS = ("ref_close", "vix", "vix3m")

SIZING_MODES = {
    "fixed_pct": "fixed notional %",
    "fixed_risk": "fixed risk % / trade",
    "vol_target": "vol target",
}


@dataclass
class StrategySpec:
    src: str = "Market"                 # "Market" | "Synthetic"
    name: str = "SPY"
    tf: str = "Daily"
    start_year: int = 2005
    rule: str = "ema"
    params: dict = field(default_factory=dict)      # day-based raw params
    filters: list = field(default_factory=list)     # [{"key", "on", "params"}]
    sizing_mode: str = "vol_target"
    target_vol: float = 0.20
    direction: str = "both"
    use_atr_stop: bool = False
    use_atr_tp: bool = False
    atr_stop_mult: float = 3.0
    atr_tp_mult: float = 6.0
    cost_bps: float = 1.0
    slippage_bps: float = 1.5
    holdout: int = 30                   # % of history locked away
    capital: float = 1_000_000.0
    buffer: float = 0.0                 # no-trade band, share of the average position
    contract_fee: float = 2.5           # $ per futures contract per side
    fixed_pct: float = 1.0              # fixed_pct: notional at forecast 10, share of capital
    risk_pct: float = 0.01              # fixed_risk: capital lost when the ATR stop is hit
    scale_by_forecast: bool = True      # fixed modes: size ∝ forecast/10, or the same size every trade
    max_leverage: float = 0.0           # |notional| / capital cap (0 = none)
    seed: int = 42                      # synthetic data seed
    synth_base: str = "SPY"             # ticker the bootstrap / fitted synthetic kinds learn from
    ref_ticker: str = "SPY"             # reference ticker for cross-asset indicators (ref_close)
    mode: str = "single"                # "single" | "blend" | "conditions"  (see builder/compose.py)
    blend: list = field(default_factory=list)        # [{"rule", "params", "weight"}]
    conditions: dict = field(default_factory=dict)   # {slot: {"join", "rows": [{"a", "op", "b"}]}}
    regime: dict = field(default_factory=dict)       # {"on", "det", "params", "min_bars", "use", "allow", "scale", "switch"}
    ml: dict = field(default_factory=dict)           # ML LAB model config (mode "ml"; see ml/strategy.py)
    notes: str = ""                     # hypothesis / goal — not part of the hash

    # ---- serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "StrategySpec":
        return cls.from_dict(json.loads(s))

    def hash_dict(self) -> dict:
        """What identifies a trial: knobs the chosen sizing mode ignores are dropped (moving an
        unused slider is not a new trial), and fields added after the first release are left out
        at their defaults so older saved specs keep their hashes."""
        d = self.to_dict()
        d.pop("notes", None)
        mode = self.sizing_mode
        if mode != "vol_target":
            d.pop("target_vol", None)
        if mode != "fixed_pct":
            d.pop("fixed_pct", None)
        if mode != "fixed_risk":
            d.pop("risk_pct", None)
        if mode == "vol_target":
            d.pop("scale_by_forecast", None)
        if self.src != "Synthetic":
            d.pop("seed", None)
            d.pop("synth_base", None)
        elif self.name not in TICKER_SYNTH:
            d.pop("synth_base", None)
        if "ref_close" not in self.needs():
            d.pop("ref_ticker", None)
        if self.mode != "blend":
            d.pop("blend", None)
        if self.mode != "conditions":
            d.pop("conditions", None)
        if self.mode != "ml":
            d.pop("ml", None)
        if self.mode != "single":
            d.pop("rule", None)
            d.pop("params", None)
        if not (self.regime or {}).get("on"):
            d.pop("regime", None)
        for k, v in _LATE_DEFAULTS.items():
            if k in d and d[k] == v:
                d.pop(k)
        return d

    def hash(self) -> str:
        d = self.hash_dict()
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def family(self) -> str:
        """Same data + same holdout = same pool of trials and the same OOS."""
        base = f"{self.src}:{self.name}:{self.tf}:{self.start_year}:{self.holdout}"
        if self.src == "Synthetic" and (self.seed != 42 or self.name in TICKER_SYNTH):
            base += f":s{self.seed}" + (f":{self.synth_base}" if self.name in TICKER_SYNTH else "")
        return base

    def with_params(self, **kw) -> "StrategySpec":
        return replace(self, params={**self.params, **kw})

    def with_filters(self, filters: list) -> "StrategySpec":
        return replace(self, filters=filters)

    # ---- engine plumbing
    def contract(self) -> tuple | None:
        return FUTURES.get(self.name.upper()) if self.src == "Market" else None

    def bt(self, cost_mult: float = 1.0, whole_contracts: bool = True) -> BacktestConfig:
        fut = self.contract() if whole_contracts else None
        return BacktestConfig(sizing_mode=self.sizing_mode, target_vol=self.target_vol, direction=self.direction,
                              fixed_pct=self.fixed_pct, risk_pct=self.risk_pct, stop_mult=self.atr_stop_mult,
                              scale_by_forecast=self.scale_by_forecast, max_leverage=self.max_leverage,
                              cost_bps=self.cost_bps * cost_mult, slippage_bps=self.slippage_bps * cost_mult,
                              capital=self.capital, buffer=self.buffer,
                              multiplier=float(fut[1]) if fut else None,
                              fee_per_contract=self.contract_fee * cost_mult if fut else 0.0)

    def risk(self) -> RiskConfig:
        # fixed-risk sizing only means something if the stop it is sized from is live
        stop_on = self.use_atr_stop or self.sizing_mode == "fixed_risk"
        return RiskConfig(use_atr_stop=stop_on, use_atr_tp=self.use_atr_tp,
                          atr_stop_mult=self.atr_stop_mult, atr_tp_mult=self.atr_tp_mult)

    def strategy(self):
        if self.mode == "blend":
            from .builder import BlendStrategy
            return BlendStrategy(self.blend, self.tf)
        if self.mode == "conditions":
            from .builder import ConditionStrategy
            return ConditionStrategy(self.conditions, self.tf)
        if self.mode == "ml":
            from .ml.strategy import MLStrategy
            return MLStrategy(self.ml, self.tf)
        return REGISTRY[self.rule](**scale_params_for_tf(self.params, self.tf))

    def needs(self) -> tuple:
        """Extra frame columns the strategy reads (see AUX_COLUMNS)."""
        from .builder import needs_of
        return needs_of(self)


def with_aux(ohlcv: pd.DataFrame, aux: dict | None, needs: tuple) -> pd.DataFrame:
    """OHLCV plus the aux columns a rule needs, aligned causally (last value known at each bar)."""
    if not needs:
        return ohlcv
    aux = aux or {}
    out = ohlcv.copy()
    for col in needs:
        s = aux.get(col)
        if s is None or len(s) == 0:
            continue
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[col] = s.reindex(s.index.union(out.index)).ffill().reindex(out.index)
    return out


def run_spec(spec: StrategySpec, ohlcv: pd.DataFrame, aux: dict | None = None, n_trials: int = 1,
             cost_mult: float = 1.0, whole_contracts: bool = True) -> BacktestResult:
    """Forecast → filters → sizing/buffer/contracts/exits/costs → result."""
    strat = spec.strategy()
    frame = with_aux(ohlcv, aux, spec.needs())
    fc = strat.forecast(frame)
    if (spec.regime or {}).get("on"):
        from .builder import apply_regime
        fc = apply_regime(spec.regime, fc, frame, spec.tf)
    fc = apply_filters(fc, ohlcv, spec.filters, aux)
    return run_single(fc, ohlcv, bt=spec.bt(cost_mult, whole_contracts), risk=spec.risk(), n_trials=n_trials,
                      label=strat.describe())


def split_holdout(df: pd.DataFrame, pct: float) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """In-sample frame and the first locked date (None when no holdout)."""
    if pct <= 0:
        return df, None
    cut = int(len(df) * (1 - pct / 100))
    return df.iloc[:cut], df.index[cut]
