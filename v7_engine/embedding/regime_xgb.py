"""
XGBoost Regime Classifier — Predicts market regime from the 128-dim feature vector.

Regimes:
  0 = trending_up
  1 = trending_down
  2 = sideways
  3 = volatile

Labels are derived from Kalman velocity, CUSUM events, and wavelet energy
on real Dukascopy data. No synthetic labels.

Output is fed back into the SDE drift context (concatenated to embedding).
"""
from __future__ import annotations
import logging
import os

import numpy as np
from v7_engine.config import (
    EMBED_XGB_CACHE_TICKS, EMBED_COL_KALMAN, EMBED_COL_CUSUM, 
    EMBED_COL_WAVELET, EMBED_REGIME_ROLLING_VAR
)
import pandas as pd

try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    logging.warning("xgboost not installed — RegimeXGBClassifier will be disabled")

from v7_engine.config import (
    XGB_REGIME_N_ESTIMATORS, XGB_REGIME_MAX_DEPTH,
    XGB_REGIME_LR, XGB_REGIME_SUBSAMPLE, XGB_REGIME_COLSAMPLE,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

_N_REGIMES = 4


class RegimeXGBClassifier:
    """
    XGBoost multi-class regime classifier.

    Input:  128-dim feature vector (from TickFeatureVector)
    Output: (4,) probability vector [p_up, p_down, p_sideways, p_volatile]
    """

    def __init__(self):
        if not _XGB_OK:
            raise ImportError("xgboost required: pip install xgboost")
        self._model: xgb.XGBClassifier | None = None

    def fit(
        self,
        X: np.ndarray,   # (N, EMBEDDING_DIM)
        y: np.ndarray,   # (N,) int labels 0-3
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RegimeXGBClassifier":
        """
        Train on real-data feature/label pairs.

        Parameters
        ----------
        X     : feature matrix from TickFeatureVector.compute()
        y     : regime labels (0=up, 1=down, 2=sideways, 3=volatile)
        X_val : optional validation set for early stopping
        y_val : validation labels
        """
        # Remap possibly-sparse labels (e.g. [0,1,3] missing class 2) to dense 0-indexed
        unique_classes = np.unique(y)
        self._label_map = {orig: idx for idx, orig in enumerate(unique_classes)}
        y_mapped = np.array([self._label_map[lbl] for lbl in y], dtype=np.int64)
        n_classes = len(unique_classes)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model = xgb.XGBClassifier(
            n_estimators     = XGB_REGIME_N_ESTIMATORS,
            max_depth        = 4,  # Heavily constrained depth to prevent overfitting
            learning_rate    = XGB_REGIME_LR,
            subsample        = XGB_REGIME_SUBSAMPLE,
            colsample_bytree = XGB_REGIME_COLSAMPLE,
            reg_alpha        = 1.0, # L1 Regularization
            reg_lambda       = 1.0, # L2 Regularization
            objective        = "multi:softprob",
            num_class        = n_classes,
            use_label_encoder= False,
            eval_metric      = "mlogloss",
            early_stopping_rounds = 20 if eval_set else None,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self._model.fit(X, y_mapped, eval_set=eval_set, verbose=False)
        logger.info("RegimeXGBClassifier trained: %d samples, %d features, %d classes %s",
                    len(X), X.shape[1], n_classes, list(unique_classes))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for one feature vector.

        Parameters
        ----------
        x : (EMBEDDING_DIM,) — single feature vector

        Returns
        -------
        probs : (4,) float32
        """
        if self._model is None:
            return np.ones(_N_REGIMES, dtype=np.float32) / _N_REGIMES
        x_2d = x.reshape(1, -1)
        probs = self._model.predict_proba(x_2d)[0]
        return probs.astype(np.float32)

    def predict(self, x: np.ndarray) -> int:
        """Return the argmax regime index."""
        return int(np.argmax(self.predict_proba(x)))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on held-out set."""
        if self._model is None:
            return 0.0
        preds = self._model.predict(X)
        return float((preds == y).mean())

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "models/regime_xgb.json") -> None:
        if self._model is None:
            raise RuntimeError("No model to save — call fit() first")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._model.save_model(path)
        logger.info("RegimeXGBClassifier saved → %s", path)

    @classmethod
    def load(cls, path: str = "models/regime_xgb.json") -> "RegimeXGBClassifier":
        if not _XGB_OK:
            raise ImportError("xgboost required")
        obj = cls.__new__(cls)
        obj._model = xgb.XGBClassifier()
        if os.path.exists(path):
            obj._model.load_model(path)
            logger.info("RegimeXGBClassifier loaded ← %s", path)
        else:
            obj._model = None
            logger.warning("RegimeXGBClassifier checkpoint not found at %s", path)
        return obj


# ── Label generation from real ticks ─────────────────────────────────────────

def label_regimes_from_features(
    feature_matrix: np.ndarray,    # (N, EMBEDDING_DIM) — from real Dukascopy ticks
    kalman_vel_col: int = EMBED_COL_KALMAN,      # Column index of kalman_vel in feature vector
    cusum_col:      int = EMBED_COL_CUSUM,      # Column index of cusum_event
    wavelet_hf_col: int = EMBED_COL_WAVELET,       # Column index of first HF wavelet coeff
) -> np.ndarray:
    """
    Rule-based regime labelling for initial XGBoost training.
    Uses Kalman velocity variance, CUSUM energy, and wavelet HF energy
    from the real feature matrix columns.

    Labels:
      0 = trending_up
      1 = trending_down
      2 = sideways
      3 = volatile

    After training, XGBoost self-improves via backtest feedback.
    """
    n = len(feature_matrix)
    labels = np.zeros(n, dtype=np.int64)

    kalman_vel = feature_matrix[:, kalman_vel_col]
    cusum_ev   = feature_matrix[:, cusum_col]
    wf_energy  = np.abs(feature_matrix[:, wavelet_hf_col])

    # Rolling variance of Kalman velocity (proxy for trend strength)
    # Replaced O(N) list comprehension with instantaneous Pandas C-extension rolling var
    vel_var = pd.Series(kalman_vel).rolling(EMBED_REGIME_ROLLING_VAR, min_periods=1).var().fillna(0.0).to_numpy()

    for i in range(n):
        if wf_energy[i] > 0.3:
            labels[i] = 3   # volatile
        elif vel_var[i] < 1e-6:
            labels[i] = 2   # sideways
        elif kalman_vel[i] > 0:
            labels[i] = 0   # trending up
        else:
            labels[i] = 1   # trending down

    return labels
