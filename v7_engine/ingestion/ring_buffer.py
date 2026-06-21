"""
Ring Buffer — Fixed-size NumPy ring buffer for tick data.
RAM-resident. Zero disk I/O. O(1) push and O(n) get_latest.
"""
from __future__ import annotations
import numpy as np
import threading
from dataclasses import dataclass

@dataclass(slots=True)
class TickRecord:
    bid:          float
    ask:          float
    timestamp_ns: int   = 0
    delta_t:      float = 0.0
    sign:         int   = 0


class RingBuffer:
    """
    Fixed-capacity NumPy ring buffer.
    push() is O(1). get_latest(n) returns the n most-recent ticks.
    Thread-safe reads are caller's responsibility.
    """

    __slots__ = ("_cap", "_ts", "_bid", "_ask", "_dt", "_sign", "_head", "_count", "_lock")

    def __init__(self, capacity: int = 50_000):
        self._cap   = capacity
        self._ts    = np.zeros(capacity, dtype=np.int64)
        self._bid   = np.zeros(capacity, dtype=np.float64)
        self._ask   = np.zeros(capacity, dtype=np.float64)
        self._dt    = np.zeros(capacity, dtype=np.float64)
        self._sign  = np.zeros(capacity, dtype=np.int8)
        self._head  = 0
        self._count = 0
        self._lock  = threading.Lock()

    # ── write ─────────────────────────────────────────────────────────────────

    def push(self, timestamp_ns: int, bid: float, ask: float, delta_t: float, sign: int) -> None:
        with self._lock:
            i = self._head
            self._ts[i]   = timestamp_ns
            self._bid[i]  = bid
            self._ask[i]  = ask
            self._dt[i]   = delta_t
            self._sign[i] = sign
            self._head    = (i + 1) % self._cap
            if self._count < self._cap:
                self._count += 1

    # ── read ──────────────────────────────────────────────────────────────────

    def get_latest(self, n: int, copy: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the n most-recent ticks in chronological order (ts, bid, ask, dt, sign)."""
        with self._lock:
            n = min(n, self._count)
            if n == 0:
                return (np.array([], dtype=np.int64), np.array([], dtype=np.float64), 
                        np.array([], dtype=np.float64), np.array([], dtype=np.float64), 
                        np.array([], dtype=np.int8))
    
            # Index calculation accounts for wrap-around.
            end   = self._head
            start = (end - n) % self._cap

        if start < end:
            sl = slice(start, end)
            if copy:
                return (
                    self._ts[sl].copy(),
                    self._bid[sl].copy(),
                    self._ask[sl].copy(),
                    self._dt[sl].copy(),
                    self._sign[sl].copy(),
                )
            else:
                return (
                    self._ts[sl],
                    self._bid[sl],
                    self._ask[sl],
                    self._dt[sl],
                    self._sign[sl],
                )
        else:
            # Wrap-around: stitch two slices
            idx = np.concatenate([np.arange(start, self._cap), np.arange(0, end)])
            return (
                self._ts[idx].copy(),
                self._bid[idx].copy(),
                self._ask[idx].copy(),
                self._dt[idx].copy(),
                self._sign[idx].copy(),
            )

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_warm(self) -> bool:
        return self._count >= 100
