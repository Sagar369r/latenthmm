"""
TIB Engine — Tick Imbalance Bar sign function + cumulative imbalance.
Operates on the resampled 100ms grid, not raw ticks.
"""
from __future__ import annotations
import numpy as np


class TIBEngine:
    """
    Tracks the rolling tick-imbalance bar (TIB) sign.
    sign = +1 if price up, -1 if price down, last sign if flat.
    Window-based cumulative imbalance (CIB) is the primary feature.
    """

    __slots__ = ("window", "_last_sign", "_signs_buf", "_buf_head", "_buf_count", "_running_sum", "_pos_count", "_neg_count")

    def __init__(self, window: int = 500):
        self.window      = window
        self._last_sign  = 0
        self._signs_buf  = np.zeros(window, dtype=np.int8)
        self._buf_head   = 0
        self._buf_count  = 0
        self._running_sum = 0
        self._pos_count   = 0
        self._neg_count   = 0

    def compute_sign(self, price: float, prev_price: float | None) -> int:
        """Lee-Ready tick rule: return +1/−1/last_sign."""
        if prev_price is None:
            return 0
        if price > prev_price:
            self._last_sign = 1
        elif price < prev_price:
            self._last_sign = -1
        # flat → keep previous sign (Lee-Ready rule)
        return self._last_sign

    def push_sign(self, sign: int) -> float:
        """Push sign into rolling window. Returns current CIB."""
        old_sign = int(self._signs_buf[self._buf_head])
        self._signs_buf[self._buf_head] = sign
        self._buf_head  = (self._buf_head + 1) % self.window
        
        if old_sign == 1: self._pos_count -= 1
        elif old_sign == -1: self._neg_count -= 1
        
        if sign == 1: self._pos_count += 1
        elif sign == -1: self._neg_count += 1
        
        if self._buf_count < self.window:
            self._running_sum += sign
            self._buf_count += 1
        else:
            self._running_sum += sign - old_sign
            
        n = self._buf_count
        return float(self._running_sum) / max(n, 1)

    def get_features(self) -> tuple[float, float, float, float]:
        """Return the four TIB features used in the embedding vector."""
        n   = self._buf_count
        cib = float(self._running_sum) / max(n, 1)
        return (
            float(self._running_sum),
            float(self._running_sum) / max(n, 1),
            float(self._pos_count) / max(n, 1),
            float(self._neg_count) / max(n, 1),
        )
