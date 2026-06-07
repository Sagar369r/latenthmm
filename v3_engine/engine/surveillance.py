"""
Layer 5 (cont.): Wasserstein Distribution Surveillance

Retained unchanged from v2 — one of the most valuable components.

W1(P_live, P_train) = inf_{γ ∈ Γ(P,Q)} E_{(x,y)~γ}[|x - y|]

Implementation: Sinkhorn approximation (O(n log n))
Threshold: W1 > 0.3σ_training → reduce position size to 25% and trigger recalibration.

Uses the POT (Python Optimal Transport) library for Sinkhorn.
"""
from __future__ import annotations

import numpy as np
import warnings

try:
    import ot  # POT: Python Optimal Transport
    _POT_AVAILABLE = True
except ImportError:
    _POT_AVAILABLE = False
    warnings.warn("POT library not available; falling back to 1D Wasserstein approximation.")


def _wasserstein1_sinkhorn(
    X_train: np.ndarray,
    X_live: np.ndarray,
    reg: float = 0.1,
    n_samples: int = 200,
) -> float:
    """
    Compute W1 Earth Mover's Distance using Sinkhorn regularisation.
    Operates on the full D-dimensional feature space.
    """
    # Sub-sample for computational efficiency
    rng = np.random.default_rng(42)
    if len(X_train) > n_samples:
        idx = rng.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[idx]
    if len(X_live) > n_samples:
        idx = rng.choice(len(X_live), n_samples, replace=False)
        X_live = X_live[idx]

    n, m = len(X_train), len(X_live)
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Cost matrix: pairwise L2 distances
    diff = X_train[:, None, :] - X_live[None, :, :]   # (n, m, D)
    M = np.sqrt((diff ** 2).sum(axis=2))                # (n, m)

    if _POT_AVAILABLE:
        try:
            W = ot.sinkhorn2(a, b, M, reg=reg)[0]
            return float(W)
        except Exception:
            pass

    # Fallback: per-dimension 1D Wasserstein sum
    w1_total = 0.0
    for d in range(X_train.shape[1]):
        xs = np.sort(X_train[:, d])
        ys = np.sort(X_live[:, d])
        # Interpolate to common length
        t = np.linspace(0, 1, max(n, m))
        xs_interp = np.interp(t, np.linspace(0, 1, n), xs)
        ys_interp = np.interp(t, np.linspace(0, 1, m), ys)
        w1_total += float(np.mean(np.abs(xs_interp - ys_interp)))
    return w1_total


class WassersteinMonitor:
    """
    Monitors distributional shift between training and live feature distributions.

    Usage:
        monitor = WassersteinMonitor()
        monitor.fit(X_train_whitened)
        result = monitor.check(X_live_window)
        if result["halt"]:
            scale_positions_by(0.25)
    """

    def __init__(
        self,
        window: int = 50,
        w1_threshold_multiplier: float = 0.3,
        position_scale_on_halt: float = 0.25,
    ) -> None:
        self.window = window
        self.threshold_multiplier = w1_threshold_multiplier
        self.position_scale = position_scale_on_halt

        self._train_X: np.ndarray | None = None
        self._sigma_train: float = 1.0
        self._threshold: float = np.inf
        self._fitted = False
        self._history: list[float] = []

    def fit(self, X_train: np.ndarray) -> "WassersteinMonitor":
        """
        Fit the monitor on training data.
        Computes the reference distribution and W1 threshold.

        X_train: (T, D) whitened feature matrix (clean rows only)
        """
        clean = X_train[~np.any(np.isnan(X_train), axis=1)]
        self._train_X = clean

        # Bootstrap estimate of W1 baseline within training set.
        # We compare window×window sub-samples (same size as live check) so
        # the calibration is apples-to-apples with the live comparison.
        if len(clean) >= 100:
            rng = np.random.default_rng(0)
            w1_samples = []
            for _ in range(50):
                idx_a = rng.choice(len(clean), self.window, replace=False)
                idx_b = rng.choice(len(clean), self.window, replace=False)
                w1 = _wasserstein1_sinkhorn(clean[idx_a], clean[idx_b])
                w1_samples.append(w1)
            w1_mean = float(np.mean(w1_samples)) if w1_samples else 1.0
            if w1_mean == 0.0:
                raise ValueError("Zero-Tolerance 3: Wasserstein W1 baseline calibrated to exactly 0.0! Frozen data feed or severe data leakage detected.")
            self._sigma_train = float(np.std(w1_samples))  if w1_samples else 1.0
            # Threshold = baseline × (1 + multiplier).
            # E.g., multiplier=0.3 → halt when live W1 exceeds 130% of the
            # natural in-distribution W1 level.  This is robust against
            # finite-sample W1 noise within the training distribution.
            self._threshold = w1_mean * (1.0 + self.threshold_multiplier)
        else:
            self._sigma_train = 1.0
            self._threshold = 0.3

        self._fitted = True
        return self

    def check(self, X_live: np.ndarray) -> dict:
        """
        Check if live distribution has shifted from training.

        X_live: (window, D) most recent whitened observations

        Returns
        -------
        dict with:
            w1_distance         : float, current W1 distance
            threshold           : float, halt trigger threshold
            halt                : bool, True if recalibration needed
            position_scale      : float, recommended position scale
            sigma_train         : float, training W1 std estimate
        """
        if not self._fitted or self._train_X is None:
            return {
                "w1_distance": 0.0,
                "threshold": np.inf,
                "halt": False,
                "position_scale": 1.0,
                "sigma_train": 1.0,
                "status": "not_fitted",
            }

        clean_live = X_live[~np.any(np.isnan(X_live), axis=1)]
        if len(clean_live) < 5:
            return {
                "w1_distance": 0.0,
                "threshold": self._threshold,
                "halt": False,
                "position_scale": 1.0,
                "sigma_train": self._sigma_train,
                "status": "insufficient_data",
            }

        # Sub-sample training to exactly self.window rows so the comparison is
        # apples-to-apples with the bootstrap calibration (which also uses
        # window × window pairs).  Using the full training set would produce a
        # systematically larger W1 than the bootstrapped threshold, causing
        # false halts on in-distribution live data.
        rng_check = np.random.default_rng(int(len(self._history)) % (2**31))
        if len(self._train_X) > self.window:
            idx_tr = rng_check.choice(len(self._train_X), self.window, replace=False)
            train_sample = self._train_X[idx_tr]
        else:
            train_sample = self._train_X

        # Also cap the live window at self.window for symmetry
        if len(clean_live) > self.window:
            idx_lv = rng_check.choice(len(clean_live), self.window, replace=False)
            clean_live = clean_live[idx_lv]

        w1 = _wasserstein1_sinkhorn(train_sample, clean_live)
        self._history.append(w1)

        halt = w1 > self._threshold
        pos_scale = self.position_scale if halt else 1.0

        return {
            "w1_distance": float(w1),
            "threshold": float(self._threshold),
            "halt": halt,
            "position_scale": float(pos_scale),
            "sigma_train": float(self._sigma_train),
            "status": "halt" if halt else "ok",
            "w1_history": list(self._history[-20:]),
        }

    def run_rolling_surveillance(
        self, X_whitened: np.ndarray, train_end: int
    ) -> pd.DataFrame:
        """
        Run rolling W1 surveillance over the full dataset.

        Parameters
        ----------
        X_whitened : (T, D) full whitened feature matrix
        train_end  : bar index where out-of-sample begins

        Returns DataFrame with W1 distances and halt flags over time.
        """
        import pandas as pd

        T = len(X_whitened)
        self.fit(X_whitened[:train_end])

        records = []
        for t in range(train_end + self.window, T):
            live_window = X_whitened[t - self.window: t]
            result = self.check(live_window)
            records.append({
                "bar": t,
                "w1_distance": result["w1_distance"],
                "threshold": result["threshold"],
                "halt": result["halt"],
                "position_scale": result["position_scale"],
            })

        return pd.DataFrame(records)


import pandas as pd  # noqa: E402
