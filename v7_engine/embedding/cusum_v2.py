"""
CUSUMv2 — Adaptive CUSUM filter using EMA baseline.

FIXES applied vs V6:
  • No prev_val attribute — uses ema_baseline (level detection, not increments)
  • Adaptive threshold via MAD of rolling history
  • Detects three event types: +1 (up shift), -1 (down shift), +2 (vol spike)
"""
from __future__ import annotations
import numpy as np
from collections import deque


class CUSUMv2:
    """
    Adaptive CUSUM filter operating on deviation from an EMA baseline.

    Event codes:
        0  — no event
        1  — positive level shift detected
       -1  — negative level shift detected
        2  — volatility spike detected
    """

    __slots__ = (
        "window", "ema_alpha",
        "c_pos", "c_neg", "c_sq",
        "history_c", "history_sq",
        "ema_baseline",
        "_tick_count", "_h_drift_cache", "_h_vol_cache"
    )

    def __init__(self, window: int = 60, ema_alpha: float = 0.02):
        self.window    = window
        self.ema_alpha = ema_alpha
        self.c_pos     = 0.0
        self.c_neg     = 0.0
        self.c_sq      = 0.0
        self.history_c : deque = deque(maxlen=window)
        self.history_sq: deque = deque(maxlen=window)
        self.ema_baseline: float | None = None
        self._tick_count = 0
        self._h_drift_cache = 1e-4
        self._h_vol_cache = 1e-8
        # NO prev_val — BUG 3.5 fix: operate on deviations from EMA, not increments

    # ── adaptive threshold ────────────────────────────────────────────────────

    def _adaptive_threshold(self, history: deque) -> float:
        if len(history) < 10:
            return 1e-4
        arr = np.array(history)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad < 1e-8:
            mad = 1e-6
        return max(med + 3.0 * mad, 1e-6)

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, val: float) -> int:
        """
        Push one price/feature value. Returns event code.
        On first call, initialises ema_baseline and returns 0.
        """
        if self.ema_baseline is None:
            self.ema_baseline = val
            return 0

        deviation = val - self.ema_baseline
        dev_sq    = deviation ** 2

        # Accumulate CUSUM statistics
        self.c_pos = max(0.0, self.c_pos + deviation)
        self.c_neg = min(0.0, self.c_neg + deviation)
        avg_sq     = float(np.mean(self.history_sq)) if len(self.history_sq) > 0 else 0.0
        self.c_sq  = max(0.0, self.c_sq + dev_sq - avg_sq)

        self.history_c.append(abs(deviation))
        self.history_sq.append(dev_sq)

        self._tick_count += 1
        update_freq = max(1, self.window // 10)
        if self._tick_count % update_freq == 0 or self._tick_count == 10:
            self._h_drift_cache = self._adaptive_threshold(self.history_c)
            self._h_vol_cache   = self._adaptive_threshold(self.history_sq)

        h_drift = self._h_drift_cache
        h_vol   = self._h_vol_cache

        # Scale-relative floor to prevent spurious triggers on near-zero signals
        if abs(self.ema_baseline) > 1e-8:
            rel_floor = abs(self.ema_baseline) * 1e-4
            h_drift = max(h_drift, rel_floor)
            h_vol   = max(h_vol,   rel_floor ** 2)

        # Emit event and reset accumulators independently
        event = 0
        if self.c_pos >= h_drift:
            event = 1
            self.c_pos = 0.0
        elif self.c_neg <= -h_drift:
            event = -1
            self.c_neg = 0.0
        elif self.c_sq >= h_vol:
            event = 2
            self.c_sq = 0.0

        # Update EMA baseline
        self.ema_baseline = self.ema_alpha * val + (1.0 - self.ema_alpha) * self.ema_baseline
        return event

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "ema_alpha": self.ema_alpha,
            "c_pos": self.c_pos,
            "c_neg": self.c_neg,
            "c_sq": self.c_sq,
            "history_c": list(self.history_c),
            "history_sq": list(self.history_sq),
            "ema_baseline": self.ema_baseline,
            "_tick_count": self._tick_count,
            "_h_drift_cache": self._h_drift_cache,
            "_h_vol_cache": self._h_vol_cache
        }
        
    def from_dict(self, state: dict) -> None:
        self.window = int(state["window"])
        self.ema_alpha = float(state["ema_alpha"])
        self.c_pos = float(state["c_pos"])
        self.c_neg = float(state["c_neg"])
        self.c_sq = float(state["c_sq"])
        self.history_c = deque(state["history_c"], maxlen=self.window)
        self.history_sq = deque(state["history_sq"], maxlen=self.window)
        self.ema_baseline = state["ema_baseline"]
        self._tick_count = int(state["_tick_count"])
        self._h_drift_cache = float(state["_h_drift_cache"])
        self._h_vol_cache = float(state["_h_vol_cache"])
