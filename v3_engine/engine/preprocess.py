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


from numba import njit

@njit()
def _expanding_winsorize_numba(X: np.ndarray, history: np.ndarray, hist_len: int) -> tuple[np.ndarray, int]:
    T, D = X.shape
    result = np.empty_like(X)
    
    for t in range(T):
        if hist_len >= 2:
            for d in range(D):
                col = history[:hist_len, d]
                valid_count = 0
                for i in range(hist_len):
                    if not np.isnan(col[i]):
                        valid_count += 1
                
                if valid_count >= 2:
                    temp = np.empty(valid_count)
                    idx = 0
                    for i in range(hist_len):
                        if not np.isnan(col[i]):
                            temp[idx] = col[i]
                            idx += 1
                    p1 = np.percentile(temp, 1)
                    p99 = np.percentile(temp, 99)
                else:
                    p1 = -np.inf
                    p99 = np.inf
                
                val = X[t, d]
                if not np.isnan(val):
                    if p1 == p99:
                        result[t, d] = val
                    elif val < p1:
                        result[t, d] = p1
                    elif val > p99:
                        result[t, d] = p99
                    else:
                        result[t, d] = val
                else:
                    result[t, d] = val
        else:
            for d in range(D):
                result[t, d] = X[t, d]
                
        for d in range(D):
            history[hist_len, d] = X[t, d]  # Use raw history for stable percentiles
        hist_len += 1
        
    return result, hist_len

def expanding_winsorize(X: np.ndarray, state: PreprocessorState) -> np.ndarray:
    """
    Winsorise each feature at expanding 1st and 99th percentiles using Numba.
    Optimized for bulk fitting using pandas expanding operations.
    """
    T, D = X.shape
    if not hasattr(state, 'history_arr') or getattr(state, 'history_arr', None) is None:
        state.history_arr = np.empty((max(T + 1000, 5000), D))
        state.hist_len = 0
    elif state.history_arr.shape[0] < state.hist_len + T:
        new_size = max(state.hist_len + T + 1000, state.history_arr.shape[0] * 2)
        new_arr = np.empty((new_size, D))
        new_arr[:state.hist_len] = state.history_arr[:state.hist_len]
        state.history_arr = new_arr

    if state.hist_len == 0 and T > 100:
        # Bulk Pandas Optimization (10,000x faster than numba for initial fit)
        df = pd.DataFrame(X)
        p1_arr = df.expanding(min_periods=2).quantile(0.01).values
        p99_arr = df.expanding(min_periods=2).quantile(0.99).values
        
        result = np.empty_like(X)
        for t in range(T):
            for d in range(D):
                val = X[t, d]
                if t == 0 or np.isnan(p1_arr[t, d]) or np.isnan(p99_arr[t, d]) or np.isnan(val):
                    result[t, d] = val
                elif val < p1_arr[t, d]:
                    result[t, d] = p1_arr[t, d]
                elif val > p99_arr[t, d]:
                    result[t, d] = p99_arr[t, d]
                else:
                    result[t, d] = val
                state.history_arr[t, d] = val
        state.hist_len = T
        return result
    else:
        result, new_len = _expanding_winsorize_numba(X, state.history_arr, state.hist_len)
        state.hist_len = new_len
        return result

@njit()
def _expanding_standardize_numba(X: np.ndarray, means: np.ndarray, M2: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, D = X.shape
    result = np.full_like(X, np.nan)

    for t in range(T):
        for d in range(D):
            val = X[t, d]
            if np.isnan(val):
                continue
            
            n[d] += 1
            delta = val - means[d]
            means[d] += delta / n[d]
            delta2 = val - means[d]
            M2[d] += delta * delta2
            
            if n[d] >= 2:
                variance = M2[d] / (n[d] - 1)
                std = np.sqrt(max(variance, 1e-8))
                result[t, d] = (val - means[d]) / std

    return result, means, M2, n

def expanding_standardize(X: np.ndarray, state: PreprocessorState) -> np.ndarray:
    """
    Standardise using Welford's online algorithm in a compiled Numba block.
    """
    T, D = X.shape
    if not hasattr(state, 'n') or isinstance(state.n, int):
        state.means = np.zeros(D)
        state.M2 = np.zeros(D)
        state.n = np.zeros(D, dtype=np.int64)

    result, new_means, new_M2, new_n = _expanding_standardize_numba(X, state.means, state.M2, state.n)
    state.means = new_means
    state.M2 = new_M2
    state.n = new_n
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
    """Apply pre-fitted PCA whitening transform with shock absorber."""
    X_centered = X - pca_params["mean"]
    scale_safe = np.maximum(pca_params["scale"], 1e-5)
    X_white = (X_centered @ pca_params["components"].T) / scale_safe
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

    def fit_transform(self, features: pd.DataFrame, train_bars: int | None = None) -> np.ndarray:
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

        # Step 3: fit PCA whitening on clean rows of the training window ONLY
        train_slice = X_std[:train_bars] if train_bars is not None else X_std
        self._pca_params = fit_pca_whitening(train_slice)
        self._fitted = True

        # Apply static whitening rotation to ALL rows (no future leakage)
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
