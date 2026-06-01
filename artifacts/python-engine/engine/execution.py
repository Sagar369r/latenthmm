"""
Layer 5: Execution, Sizing & Risk

Triple Gate Signal Filter:
  Gate 1 — Regime:   P(S_t = TREND) > 0.65
  Gate 2 — Momentum: |mv_t| > 1.0 (>1σ normalised)
  Gate 3 — Volume:   q_t > 1.3 on entry bar

Kelly-Optimal Position Sizing (corrected):
  f* = μ_edge / σ²_edge          (Full Kelly)
  f_t = 0.5 × f* × P(TREND)     (Half-Kelly, regime-weighted)
  Position size = f_t × Equity / ATR_14
  Hard cap: f_t ≤ 0.02 (2% of equity)

ATR-Based Stops:
  SL  = Entry ∓ ATR_14 × R_sl   (R_sl = 1.5 TREND, 1.0 MEAN-REV)
  TP  = Entry ± ATR_14 × R_tp   (R_tp = 3.0)
  Trailing stop: breakeven at +1 ATR; trail at 1 ATR behind highest close at +2 ATR
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------ #
# Constants (fixed by spec — not optimisation parameters)      #
# ------------------------------------------------------------ #
REGIME_CONF_THRESHOLD = 0.65   # P(TREND) > 0.65
MOMENTUM_THRESHOLD = 1.0       # |mv_t| > 1 σ
VOLUME_THRESHOLD = 1.3         # q_t > 1.3
HALF_KELLY_FRACTION = 0.5
MAX_POSITION_FRACTION = 0.02   # 2% hard cap
ATR_PERIOD = 14
R_SL_TREND = 1.5
R_SL_MEAN_REV = 1.0
R_TP = 3.0


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


class TripleGate:
    """
    Evaluates the three entry gates on each bar.
    Returns a dict of gate results and whether all three pass.
    """

    def evaluate(
        self,
        p_trend: float,
        mv_t: float,
        q_t: float,
    ) -> dict:
        regime_pass = p_trend > REGIME_CONF_THRESHOLD
        momentum_pass = abs(mv_t) > MOMENTUM_THRESHOLD
        volume_pass = q_t > VOLUME_THRESHOLD

        all_pass = regime_pass and momentum_pass and volume_pass
        direction = int(np.sign(mv_t)) if all_pass else 0

        return {
            "all_pass": all_pass,
            "direction": direction,
            "regime_gate": regime_pass,
            "momentum_gate": momentum_pass,
            "volume_gate": volume_pass,
            "p_trend": p_trend,
            "mv_t": mv_t,
            "q_t": q_t,
        }


class KellyPositionSizer:
    """
    Half-Kelly position sizer with ATR denominator and hard 2% cap.

    f* = μ_edge / σ²_edge   (estimated from rolling OOS returns)
    f_t = 0.5 × f* × P(TREND)
    Shares = f_t × Equity / (ATR_14 × Price)
    """

    def __init__(self, estimation_window: int = 500) -> None:
        self.estimation_window = estimation_window
        self._edge_returns: list[float] = []

    def update_edge(self, trade_return: float) -> None:
        """Record a completed trade return (for rolling edge estimation)."""
        self._edge_returns.append(trade_return)
        if len(self._edge_returns) > self.estimation_window:
            self._edge_returns.pop(0)

    def _estimate_edge(self) -> tuple[float, float]:
        """Estimate μ_edge and σ²_edge from observed trade returns."""
        if len(self._edge_returns) < 10:
            return 0.005, 0.01   # conservative default: 0.5% edge, 10% vol²
        arr = np.array(self._edge_returns)
        return float(arr.mean()), max(float(arr.var()), 1e-6)

    def compute_fraction(self, p_trend: float) -> float:
        """
        Compute position fraction f_t.

        f* = μ_edge / σ²_edge
        f_t = 0.5 × f* × P(TREND)
        Hard cap at 0.02.
        """
        mu_edge, sigma2_edge = self._estimate_edge()
        f_star = mu_edge / sigma2_edge if sigma2_edge > 0 else 0.0
        f_t = HALF_KELLY_FRACTION * f_star * p_trend
        return float(np.clip(f_t, 0.0, MAX_POSITION_FRACTION))

    def compute_shares(
        self,
        p_trend: float,
        equity: float,
        atr: float,
        price: float,
    ) -> tuple[float, int]:
        """
        Returns (position_fraction, n_shares).
        position_fraction: fraction of equity at risk
        n_shares: number of shares (floored, minimum 0)
        """
        frac = self.compute_fraction(p_trend)
        if atr <= 0 or price <= 0:
            return frac, 0
        # 1 ATR of adverse move = frac × equity at risk
        n_shares = int(np.floor((frac * equity) / atr))
        return frac, max(n_shares, 0)


class StopManager:
    """ATR-based stop-loss, take-profit, and trailing stop logic."""

    @staticmethod
    def initial_stops(
        direction: int,
        entry_price: float,
        atr: float,
        regime: str = "TREND",
    ) -> tuple[float, float]:
        """
        Compute initial SL and TP.

        SL = Entry ∓ ATR × R_sl
        TP = Entry ± ATR × R_tp
        """
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
    ) -> OpenPosition:
        """
        Trailing stop logic:
        - Once position reaches +1 ATR profit → move stop to breakeven
        - Once position reaches +2 ATR profit → trail at 1 ATR behind highest close
        """
        pnl_atr = (current_price - position.entry_price) * direction / atr

        if pnl_atr >= 1.0 and not position.trail_activated:
            # Move to breakeven
            position.stop_loss = position.entry_price
            position.trail_activated = True

        if pnl_atr >= 2.0:
            # Update highest favorable price
            if direction == 1:
                position.highest_favorable = max(
                    position.highest_favorable, current_price
                )
                trail_sl = position.highest_favorable - atr
                position.stop_loss = max(position.stop_loss, trail_sl)
            else:
                position.highest_favorable = min(
                    position.highest_favorable, current_price
                )
                trail_sl = position.highest_favorable + atr
                position.stop_loss = min(position.stop_loss, trail_sl)

        return position


class SignalEngine:
    """
    Full execution engine combining Triple Gate, Kelly sizing, and stops.
    """

    def __init__(self, equity: float = 100_000.0) -> None:
        self.equity = equity
        self.gate = TripleGate()
        self.sizer = KellyPositionSizer()
        self.stop_mgr = StopManager()

    def run(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        regime_proba: np.ndarray,
        atr_series: pd.Series,
    ) -> list[TradeSignal]:
        """
        Generate trade signals bar-by-bar.

        Parameters
        ----------
        df           : OHLCV DataFrame (aligned with features)
        features     : feature DataFrame with columns including 'mvt', 'qt'
        regime_proba : (T, 3) array from HMM.predict_proba()
        atr_series   : ATR_14 Series

        Returns
        -------
        List of TradeSignal objects where a signal was generated.
        """
        T = len(df)
        signals: list[TradeSignal] = []

        for t in range(1, T):
            p_trend = float(regime_proba[t, 0])
            p_mean_rev = float(regime_proba[t, 1])
            p_stress = float(regime_proba[t, 2])

            mv_t = float(features["mvt"].iloc[t]) if not np.isnan(features["mvt"].iloc[t]) else 0.0
            q_t = float(features["qt"].iloc[t]) if not np.isnan(features["qt"].iloc[t]) else 1.0
            atr = float(atr_series.iloc[t]) if not np.isnan(atr_series.iloc[t]) else 0.0
            price = float(df["close"].iloc[t])

            gate_result = self.gate.evaluate(p_trend, mv_t, q_t)

            if gate_result["all_pass"] and gate_result["direction"] != 0 and atr > 0:
                direction = gate_result["direction"]
                regime_label = (
                    "TREND" if p_trend >= max(p_mean_rev, p_stress)
                    else "MEAN_REV" if p_mean_rev >= p_stress
                    else "STRESS"
                )
                frac, _ = self.sizer.compute_shares(p_trend, self.equity, atr, price)
                sl, tp = self.stop_mgr.initial_stops(direction, price, atr, regime_label)

                sig = TradeSignal(
                    bar_index=t,
                    timestamp=df.index[t],
                    direction=direction,
                    regime=regime_label,
                    p_trend=p_trend,
                    p_mean_rev=p_mean_rev,
                    p_stress=p_stress,
                    momentum=mv_t,
                    volume_ratio=q_t,
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    position_fraction=frac,
                    atr=atr,
                    gates_passed=gate_result,
                )
                signals.append(sig)

        return signals


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Compute ATR_14 for execution."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename("atr14")


def run_execution_layer(
    df: pd.DataFrame,
    features: pd.DataFrame,
    regime_proba: np.ndarray,
    equity: float = 100_000.0,
) -> dict:
    """
    Full Layer 5 pipeline.

    Returns dict with signals and position-level summary.
    """
    atr = compute_atr(df)
    engine = SignalEngine(equity=equity)
    signals = engine.run(df, features, regime_proba, atr)

    return {
        "signals": signals,
        "n_signals": len(signals),
        "atr": atr,
        "engine": engine,
    }
