"""
Welford Online Normaliser.
Updates running mean/variance on every tick. Never fits once and freezes.
Handles regime shifts (NFP, flash crash) that would break static scalers.
"""
from __future__ import annotations
import numpy as np


class WelfordNormaliser:
    """
    Per-dimension Welford streaming normaliser.
    Each call to update_and_transform() is O(D) — safe in the hot path.
    """

    __slots__ = ("_dim", "_clip", "_min_std", "_warmup",
                 "_n", "_mean", "_M2", "is_warm")

    def __init__(
        self,
        dim: int,
        clip_sigma: float = 5.0,
        min_std: float = 1e-8,
        warmup: int = 1_000,
    ):
        self._dim     = dim
        self._clip    = clip_sigma
        self._min_std = min_std
        self._warmup  = warmup
        self._n       = 0
        self._mean    = np.zeros(dim, dtype=np.float64)
        self._M2      = np.zeros(dim, dtype=np.float64)
        self.is_warm  = False

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, x: np.ndarray) -> None:
        """Ingest one feature vector and update running stats (no output)."""
        x = x.astype(np.float64)
        self._n += 1
        delta  = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2   += delta * delta2
        if not self.is_warm and self._n >= self._warmup:
            self.is_warm = True

    def combine(self, other_state: dict) -> None:
        """Combine another Welford state into this one using parallel variance algorithm."""
        n_B = int(other_state["n"])
        if n_B == 0:
            return
            
        mean_B = other_state["mean"]
        M2_B = other_state["m2"]
        
        if self._n == 0:
            self._n = n_B
            self._mean = mean_B.copy()
            self._M2 = M2_B.copy()
        else:
            n_A = self._n
            n_X = n_A + n_B
            delta = mean_B - self._mean
            
            self._mean = self._mean + delta * n_B / n_X
            self._M2 = self._M2 + M2_B + (delta ** 2) * n_A * n_B / n_X
            self._n = n_X
            
        if not self.is_warm and self._n >= self._warmup:
            self.is_warm = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalise x using current running stats. Returns float32."""
        if self._n < 2:
            return np.zeros(self._dim, dtype=np.float32)
        std = np.sqrt(self._M2 / (self._n - 1)).clip(min=self._min_std)
        z   = (x.astype(np.float64) - self._mean) / std
        return np.clip(z, -self._clip, self._clip).astype(np.float32)

    def update_and_transform(self, x: np.ndarray) -> np.ndarray:
        """Update stats with x, then return normalised x. Most-used entry point."""
        self.update(x)
        return self.transform(x)

    @property
    def mean_(self) -> np.ndarray:
        return self._mean.copy().astype(np.float32)

    @property
    def std_(self) -> np.ndarray:
        if self._n < 2:
            return np.ones(self._dim, dtype=np.float32) * self._min_std
        return np.sqrt(self._M2 / max(self._n - 1, 1)).clip(
            min=self._min_std
        ).astype(np.float32)

    @property
    def feature_stability_index(self) -> float:
        """Mean of all dim stds — rising sharply signals a regime shift."""
        return float(np.mean(self.std_))

    def save_state(self) -> dict:
        return {"mean": self._mean.copy(), "m2": self._M2.copy(), "n": self._n}

    @classmethod
    def from_state(cls, state: dict, dim: int, clip_sigma: float, min_std: float) -> "WelfordNormaliser":
        obj = cls(dim=dim, clip_sigma=clip_sigma, min_std=min_std, warmup=1)
        obj._mean    = state["mean"].copy()
        obj._M2      = state["m2"].copy()
        obj._n       = int(state["n"])
        obj.is_warm  = obj._n >= 1
        return obj
