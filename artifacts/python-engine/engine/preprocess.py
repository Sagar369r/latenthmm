"""
Layer 2 (cont.): Whitening & Normalisation

Apply in order:
  1. Winsorise each feature at 1st/99th percentile (rolling expanding window)
  2. Standardise: z = (X - μ̂_expanding) / σ̂_expanding
  3. Robust PCA whitening to decorrelate the 6 features

All statistics are computed on an expanding window — never in-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from dataclasses import dataclass, field


@dataclass
class PreprocessorState:
    """Serialisable state for online expanding-window preprocessing."""
    n: int = 0
    means: np.ndarray = field(default_factory=lambda: np.array([]))
    M2: np.ndarray = field(default_factory=lambda: np.array([]))   # for Welford's algorithm
    p1: np.ndarray = field(default_factory=lambda: np.array([]))   # 1st percentile buffer
    p99: np.ndarray = field(default_factory=lambda: np.array([]))  # 99th percentile buffer
    pca_components: np.ndarray | None = None
    pca_explained: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    history: list[np.ndarray] = field(default_factory=list)        # for percentile estimation


def expanding_winsorize(X: np.ndarray, state: PreprocessorState) -> np.ndarray:
    """
    Winsorise each feature at expanding 1st and 99th percentiles.
    Returns a copy with outliers clipped.
    """
    T, D = X.shape
    result = X.copy()

    for t in range(T):
        if len(state.history) >= 2:
            history_arr = np.array(state.history)
            p1 = np.nanpercentile(history_arr, 1, axis=0)
            p99 = np.nanpercentile(history_arr, 99, axis=0)
            result[t] = np.clip(X[t], p1, p99)
        state.history.append(result[t].copy())

    return result


def expanding_standardize(X: np.ndarray, state: PreprocessorState) -> np.ndarray:
    """
    Standardise using Welford's online algorithm for expanding mean/variance.
    z_t = (X_t - μ̂_t) / σ̂_t
    """
    T, D = X.shape

    if state.n == 0:
        state.means = np.zeros(D)
        state.M2 = np.zeros(D)

    result = np.full_like(X, np.nan)

    for t in range(T):
        state.n += 1
        delta = X[t] - state.means
        state.means += delta / state.n
        delta2 = X[t] - state.means
        state.M2 += delta * delta2

        if state.n >= 2:
            variance = state.M2 / (state.n - 1)
            std = np.sqrt(np.maximum(variance, 1e-8))
            result[t] = (X[t] - state.means) / std

    return result


def fit_pca_whitening(X_standardized: np.ndarray, n_components: int | None = None) -> dict:
    """
    Fit robust PCA whitening on standardised data.
    Returns a dict with components, mean, and explained variance.
    """
    clean = X_standardized[~np.any(np.isnan(X_standardized), axis=1)]
    if len(clean) < 10:
        D = X_standardized.shape[1]
        return {
            "components": np.eye(D),
            "mean": np.zeros(D),
            "scale": np.ones(D),
            "n_components": D,
        }

    n_comp = n_components or clean.shape[1]
    pca = PCA(n_components=n_comp, whiten=True)
    pca.fit(clean)

    return {
        "components": pca.components_,
        "mean": pca.mean_,
        "scale": np.sqrt(pca.explained_variance_),
        "n_components": n_comp,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def apply_pca_whitening(X: np.ndarray, pca_params: dict) -> np.ndarray:
    """Apply pre-fitted PCA whitening transform."""
    X_centered = X - pca_params["mean"]
    X_white = (X_centered @ pca_params["components"].T) / pca_params["scale"]
    return X_white


class Preprocessor:
    """
    Stateful expanding-window preprocessor:
      winsorise → standardise → PCA whiten

    Usage:
        pp = Preprocessor()
        X_clean = pp.fit_transform(feature_df)   # fits on full history (offline)
        X_live = pp.transform(new_row)            # uses fitted params (online)
    """

    def __init__(self) -> None:
        self._state = PreprocessorState()
        self._pca_params: dict | None = None
        self._fitted = False
        self._feature_names: list[str] = []

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessing on the full DataFrame (expanding window, no look-ahead).
        Returns whitened array of shape (T, D).
        """
        self._feature_names = list(features.columns)
        X = features.values.astype(float)

        # Step 1: expanding winsorise
        self._state = PreprocessorState()
        X_wins = expanding_winsorize(X, self._state)

        # Step 2: expanding standardise
        state2 = PreprocessorState()
        state2.n = 0
        state2.means = np.zeros(X.shape[1])
        state2.M2 = np.zeros(X.shape[1])
        X_std = expanding_standardize(X_wins, state2)

        # Step 3: fit PCA whitening on clean rows
        self._pca_params = fit_pca_whitening(X_std)
        self._fitted = True

        # Apply whitening (replace NaN rows with zeros)
        mask = ~np.any(np.isnan(X_std), axis=1)
        X_white = np.full_like(X_std, np.nan)
        X_white[mask] = apply_pca_whitening(X_std[mask], self._pca_params)

        return X_white

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """Transform new data using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        # Use the fitted PCA params (expanding stats tracked implicitly)
        mask = ~np.any(np.isnan(X_new), axis=1)
        result = np.full_like(X_new, np.nan)
        if mask.any():
            result[mask] = apply_pca_whitening(X_new[mask], self._pca_params)
        return result

    def preprocess_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing returning a DataFrame with same index."""
        X_white = self.fit_transform(features)
        cols = [f"w{name}" for name in self._feature_names]
        return pd.DataFrame(X_white, index=features.index, columns=cols)

    @property
    def pca_params(self) -> dict | None:
        return self._pca_params

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names
