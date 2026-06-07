"""
Layer 5: Execution, Sizing & Risk

Triple Gate Signal Filter:
  Gate 1 — Regime:   P(S_t = TREND) > 0.65
  Gate 2 — Momentum: |mv_t| > 1.0 (>1σ normalised)
  Gate 3 — Volume:   q_t > 1.3 on entry bar

Kelly-Optimal Position Sizing (corrected):
  f* = μ_edge / σ²_edge          (Full Kelly)
  f_t = 0.5 × f* × P(TREND)      (Half-Kelly, regime-weighted)
  Position size = f_t × Equity / ATR_14
  Hard cap: f_t ≤ 0.02 (2% of equity)

ATR-Based Stops (Make or Break Protocol):
  SL  = Entry ∓ ATR_14 × R_sl   (R_sl = 3.0 TREND, 1.0 MEAN-REV)
  TP  = Entry ± ATR_14 × R_tp   (R_tp = 5.0)
  Trailing stop: activate at +2.0 ATR; trail at 1.0 ATR behind highest close
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------ #
# Constants (fixed by spec — not optimisation parameters)      #
# ------------------------------------------------------------ #
REGIME_CONF_THRESHOLD = float(os.environ.get("HMM_VETO_THRESHOLD", "0.65"))
HALF_KELLY_FRACTION = 0.5
MAX_POSITION_FRACTION = 0.02
ATR_PERIOD = int(os.environ.get("HMM_ATR_PERIOD", "14"))

# Make or Break Widen Stops
R_SL_TREND    = 3.0  # WIDENED from 1.5 to 3.0
R_SL_MEAN_REV = 1.0  
R_TP          = 5.0  # ASYMMETRIC TP set to 5.0


@dataclass
class TradeSignal:
    bar_index: int
    timestamp: object
    direction: int         # +1 = long, -1 = short, 0 = flat
    regime: str
    p_trend: float
    p_mean_rev: float
    p_stress: float
    momentum: float
    volume_ratio: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_fraction: float   # fraction of equity
    atr: float
    gates_passed: dict


@dataclass
class OpenPosition:
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    trail_activated: bool = False
    highest_favorable: float = 0.0
    entry_bar: int = 0

    def __post_init__(self):
        if self.highest_favorable == 0.0:
            self.highest_favorable = self.entry_price


class TripleGate:
    """
    Dynamic Execution Router for Multi-Asset Testing.
    """
    def __init__(self, mode: str = "MOMENTUM"):
        self.mode = mode.upper()
        self.trend_status = 0  
        self.pullback_active = False

    def evaluate(self, p_mean_rev: float, p_trend: float, mvt: float, qt: float, mac_t: float = 0.0, liq_t: float = 1.0, mac_slope: float = 0.0) -> dict:
        direction = 0
        all_pass = False
        regime_pass = False
        momentum_pass = False
        volume_pass = False

        if self.mode == "MOMENTUM":
            regime_pass = p_trend > REGIME_CONF_THRESHOLD
            
            # The HMM should never be allowed to fire a Long signal if the macro trend is down
            long_cond  = mvt >  1.5 and mac_t >= 0.0
            short_cond = mvt < -1.5 and mac_t <= 0.0
            momentum_pass = long_cond or short_cond
            
            # qt now carries the Order Flow Imbalance (f_ofi) which is bounded between -1.0 and 1.0
            if long_cond:
                volume_pass = qt > 0.2
                if volume_pass and regime_pass:
                    direction = 1
                    all_pass = True
            elif short_cond:
                volume_pass = qt < -0.2
                if volume_pass and regime_pass:
                    direction = -1
                    all_pass = True

        elif self.mode == "MEAN_REV":
            regime_pass = p_mean_rev > 0.50
            long_cond  = mvt < -1.5   
            short_cond = mvt >  1.5   
            momentum_pass = long_cond or short_cond
            volume_pass = 0.5 < qt < 2.5

            if regime_pass and momentum_pass and volume_pass:
                all_pass = True
                direction = 1 if long_cond else -1

        elif self.mode == "MEAN_REVERSION_EXHAUSTION":
            # 1. HMM REGIME: Ensure the market is choppy or stressed (NOT trending)
            p_stress = 1.0 - p_trend - p_mean_rev
            regime_pass = (p_mean_rev > 0.50) or (p_stress > 0.50)
            
            long_trigger = False
            short_trigger = False
            
            if regime_pass:
                # 2. DOWNSIDE CAPITULATION (Buy the dip in a Bull Trend)
                # THRESHOLD = 0.4 — matches Pine Script screening standard
                if mac_slope > 0: 
                    if mvt < -0.4 and qt < -0.4:
                        long_trigger = True
                # 3. UPSIDE EXHAUSTION (Short the rally in a Bear Trend)
                elif mac_slope < 0:
                    if mvt > 0.4 and qt > 0.4:
                        short_trigger = True

            if long_trigger:
                direction = 1
                momentum_pass = True
                volume_pass = True
                all_pass = True
            elif short_trigger:
                direction = -1
                momentum_pass = True
                volume_pass = True
                all_pass = True

        elif self.mode == "DAYTRADE_SLINGSHOT":
            # 1. HMM VERIFICATION: Ensure the market is waking up
            regime_pass = p_trend > 0.65
            
            # 2. LIQUIDITY VERIFICATION: Institutional money is entering
            volume_pass = liq_t > 1.0
            
            long_trigger = False
            short_trigger = False
            
            if regime_pass and volume_pass:
                # 3. SECULAR BULL IGNITION (Positive Skew Long)
                if mac_slope > 0 and mvt > 0.3 and qt > 0.1:
                    long_trigger = True
                # 4. SECULAR BEAR IGNITION (Positive Skew Short)
                elif mac_slope < 0 and mvt < -0.3 and qt < -0.1:
                    short_trigger = True

            if long_trigger:
                direction = 1
                momentum_pass = True
                all_pass = True
            elif short_trigger:
                direction = -1
                momentum_pass = True
                all_pass = True

        return {
            "all_pass":      all_pass,
            "direction":     direction,
            "regime_gate":   regime_pass,
            "momentum_gate": momentum_pass,
            "volume_gate":   volume_pass,
            "p_trend":       p_trend,
            "p_mean_rev":    p_mean_rev,
            "mvt":           mvt,
            "qt":            qt,
        }

class KellyPositionSizer:
    def __init__(self, estimation_window: int = 500) -> None:
        self.estimation_window = estimation_window
        self._edge_returns: list[float] = []

    def update_edge(self, trade_return: float) -> None:
        self._edge_returns.append(trade_return)
        if len(self._edge_returns) > self.estimation_window:
            self._edge_returns.pop(0)

    def _estimate_edge(self) -> tuple[float, float]:
        if len(self._edge_returns) < 10:
            return 0.0004, 0.01
        arr = np.array(self._edge_returns)
        return float(arr.mean()), max(float(arr.var()), 1e-6)

    def compute_fraction(self, p_trend: float) -> float:
        mu_edge, sigma2_edge = self._estimate_edge()
        f_star = mu_edge / sigma2_edge if sigma2_edge > 0 else 0.0
        f_t = HALF_KELLY_FRACTION * f_star * p_trend
        return float(np.clip(f_t, 0.0, MAX_POSITION_FRACTION))

    def compute_shares(self, p_trend: float, equity: float, atr: float, price: float) -> tuple[float, int]:
        frac = self.compute_fraction(p_trend)
        if atr <= 0 or price <= 0:
            return frac, 0
        n_shares = int(np.floor((frac * equity) / atr))
        return frac, max(n_shares, 0)

class StopManager:
    @staticmethod
    def initial_stops(direction: int, entry_price: float, atr: float, regime: str = "TREND") -> tuple[float, float]:
        r_sl = R_SL_TREND if regime == "TREND" else R_SL_MEAN_REV
        sl = entry_price - direction * atr * r_sl
        tp = entry_price + direction * atr * R_TP
        return float(sl), float(tp)

    @staticmethod
    def update_trailing_stop(
        position: OpenPosition,
        current_price: float,
        atr: float,
        direction: int,
        pip_size: float = 0.01,
    ) -> OpenPosition:
        pnl_atr = (current_price - position.entry_price) * direction / atr

        # Activate Trail only at +2.0 ATR profit
        if pnl_atr >= 2.0 and not position.trail_activated:
            position.stop_loss = position.entry_price + (direction * 2 * pip_size)
            position.trail_activated = True

        # Trail behind the highest close by 1.0 ATR
        if pnl_atr >= 2.0:
            if direction == 1:
                position.highest_favorable = max(position.highest_favorable, current_price)
                trail_sl = position.highest_favorable - (1.0 * atr)
                position.stop_loss = max(position.stop_loss, trail_sl)
            else:
                position.highest_favorable = min(position.highest_favorable, current_price)
                trail_sl = position.highest_favorable + (1.0 * atr)
                position.stop_loss = min(position.stop_loss, trail_sl)

        return position

class SignalEngine:
    def __init__(self, equity: float = 100_000.0, mode: str = "MOMENTUM") -> None:
        self.equity = equity
        self.mode = mode
        self.gate = TripleGate(mode=self.mode)
        self.sizer = KellyPositionSizer()
        self.stop_mgr = StopManager()

    def run(self, df: pd.DataFrame, features: pd.DataFrame, regime_proba: np.ndarray, atr_series: pd.Series) -> list[TradeSignal]:
        T = len(df)
        signals: list[TradeSignal] = []

        # 🚀 OPTIMIZATION: Strip Pandas overhead by extracting contiguous C-arrays once
        p_mean_rev_arr = regime_proba[:, 0]
        p_trend_arr    = regime_proba[:, 1]
        p_stress_arr   = regime_proba[:, 2]
        
        mvt_arr = features["mvt"].to_numpy()
        qt_arr  = features["qt"].to_numpy()
        atr_arr = atr_series.to_numpy()
        close_arr = df["close"].to_numpy()
        timestamps = df.index.to_numpy()

        for t in range(1, T):
            # 100x faster than df.iloc[t]
            p_mean_rev = float(p_mean_rev_arr[t])
            p_trend    = float(p_trend_arr[t])
            p_stress   = float(p_stress_arr[t])
            
            mvt_t = float(mvt_arr[t]) if not np.isnan(mvt_arr[t]) else 0.0
            qt_t  = float(qt_arr[t])  if not np.isnan(qt_arr[t])  else 1.0
            atr   = float(atr_arr[t]) if not np.isnan(atr_arr[t]) else 0.0
            price = float(close_arr[t])

            gate_result = self.gate.evaluate(p_mean_rev, p_trend, mvt_t, qt_t)

            if gate_result["all_pass"] and gate_result["direction"] != 0 and atr > 0:
                direction = gate_result["direction"]
                regime_label = (
                    "TREND" if p_trend >= max(p_mean_rev, p_stress)
                    else "MEAN_REV" if p_mean_rev >= p_stress
                    else "STRESS"
                )
                
                p_size_basis = p_trend if self.mode in ("MOMENTUM", "DAYTRADE_SLINGSHOT") else p_mean_rev
                frac, _ = self.sizer.compute_shares(p_size_basis, self.equity, atr, price)
                sl, tp = self.stop_mgr.initial_stops(direction, price, atr, regime_label)

                sig = TradeSignal(
                    bar_index=t, timestamp=timestamps[t], direction=direction, regime=regime_label,
                    p_trend=p_trend, p_mean_rev=p_mean_rev, p_stress=p_stress, momentum=mvt_t,
                    volume_ratio=qt_t, entry_price=price, stop_loss=sl, take_profit=tp,
                    position_fraction=frac, atr=atr, gates_passed=gate_result,
                )
                signals.append(sig)
                
        return signals

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename("atr14")

def run_execution_layer(df: pd.DataFrame, features: pd.DataFrame, regime_proba: np.ndarray, equity: float = 100_000.0, mode: str = "MOMENTUM") -> dict:
    atr = compute_atr(df)
    engine = SignalEngine(equity=equity, mode=mode)
    signals = engine.run(df, features, regime_proba, atr)
    return {"signals": signals, "n_signals": len(signals), "atr": atr, "engine": engine}
