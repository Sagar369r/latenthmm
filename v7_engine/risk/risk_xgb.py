"""
XGBoost Risk Classifier — Binary classifier for large loss prediction.

Predicts P(drawdown > $50 in next RISK_XGB_DRAWDOWN_LIMIT ticks) from the last 10 feature vectors.
Acts as a secondary gate: if risk_score > 0.70, block trade even if EBM allows it.

Input:  Last 10 feature vectors concatenated → (10 × EMBEDDING_DIM,) = (1280,)
Output: P(loss > $50) ∈ [0, 1]
"""
from __future__ import annotations
import logging
import os

import numpy as np
from v7_engine.config import (
    RISK_XGB_THRESHOLD, RISK_XGB_BLOCK_THRESHOLD, RISK_XGB_DRAWDOWN_LIMIT, 
    RISK_XGB_ROLLING_VAR
)

try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

from v7_engine.config import (
    XGB_RISK_N_ESTIMATORS, XGB_RISK_MAX_DEPTH, XGB_RISK_LR,
    XGB_RISK_DRAWDOWN_THRESH, EMBEDDING_DIM,
    BACKTEST_INITIAL_EQUITY,
)

logger = logging.getLogger(__name__)

_FEATURE_WINDOW = 10   # Number of consecutive feature vectors to concatenate
_FEATURE_DIM    = _FEATURE_WINDOW * EMBEDDING_DIM


class RiskXGBClassifier:
    """
    Binary XGBoost risk model.

    Predicts P(max_drawdown > XGB_RISK_DRAWDOWN_THRESH in next RISK_XGB_DRAWDOWN_LIMIT ticks).

    Output gate: block trade if predict_risk(x) > HYBRID_RISK_OVERRIDE_THRESH
    """

    def __init__(self):
        if not _XGB_OK:
            raise ImportError("xgboost required: pip install xgboost")
        self._model: xgb.XGBClassifier | None = None

    def fit(
        self,
        X:     np.ndarray,   # (N, _FEATURE_DIM)
        y:     np.ndarray,   # (N,) binary labels 0/1
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RiskXGBClassifier":
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model = xgb.XGBClassifier(
            n_estimators     = XGB_RISK_N_ESTIMATORS,
            max_depth        = 4,  # Heavily constrained depth to prevent overfitting
            learning_rate    = XGB_RISK_LR,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            reg_alpha        = 1.0, # L1 Regularization
            reg_lambda       = 1.0, # L2 Regularization
            scale_pos_weight = 3.0,   # Handle class imbalance (rare large losses)
            objective        = "binary:logistic",
            eval_metric      = "auc",
            early_stopping_rounds = 20 if eval_set else None,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self._model.fit(X, y, eval_set=eval_set, verbose=False)
        logger.info("RiskXGBClassifier trained: %d samples, %d features", len(X), X.shape[1])
        return self

    def predict_risk(self, feature_window: np.ndarray) -> float:
        """
        Predict risk probability from last 10 feature vectors.

        Parameters
        ----------
        feature_window : (10, EMBEDDING_DIM) or (1280,) — last 10 feature vectors

        Returns
        -------
        risk_score ∈ [0, 1]
        """
        if self._model is None:
            return 0.0
        x = feature_window.reshape(1, -1)
        prob = self._model.predict_proba(x)[0, 1]
        return float(prob)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """AUC-ROC on held-out set."""
        if self._model is None:
            return RISK_XGB_THRESHOLD
        from sklearn.metrics import roc_auc_score
        proba = self._model.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, proba))

    def save(self, path: str = "models/risk_xgb.json") -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._model.save_model(path)
        logger.info("RiskXGBClassifier saved → %s", path)

    @classmethod
    def load(cls, path: str = "models/risk_xgb.json") -> "RiskXGBClassifier":
        if not _XGB_OK:
            raise ImportError("xgboost required")
        obj = cls.__new__(cls)
        obj._model = xgb.XGBClassifier()
        if os.path.exists(path):
            obj._model.load_model(path)
            logger.info("RiskXGBClassifier loaded ← %s", path)
        else:
            obj._model = None
            logger.warning("RiskXGBClassifier not found at %s", path)
        return obj


# ── Label generation from real tick data ──────────────────────────────────────

def label_risk_windows(
    tick_sequences:   np.ndarray,   # (N, EMBEDDING_DIM) — feature vectors
    future_rets:      np.ndarray,   # (N,) log returns
    forward_n:        int   = RISK_XGB_DRAWDOWN_LIMIT,
    dd_thresh_usd:    float = XGB_RISK_DRAWDOWN_THRESH,
    equity:           float = BACKTEST_INITIAL_EQUITY,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Label each position as high-risk (1) if max drawdown > dd_thresh_usd
    in the next forward_n ticks.

    Returns
    -------
    X      : (N - forward_n, _FEATURE_DIM) — windowed feature matrices
    labels : (N - forward_n,) binary
    """
    n = len(tick_sequences)
    valid_len = n - _FEATURE_WINDOW - forward_n
    
    if valid_len <= 0:
        return np.zeros((0, _FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)

    # 1. Vectorized Peak Drawdown Calculation (O(1) loops)
    # fr_view shape: (valid_len, forward_n)
    fr_view = np.lib.stride_tricks.sliding_window_view(
        future_rets[_FEATURE_WINDOW:], forward_n
    )
    fr_view = fr_view[:valid_len]
    
    # Calculate localized cumulative sums of log returns
    fr_cumsum = np.cumsum(fr_view, axis=1)
    
    # Prepend a column of zeros so the starting relative price is 1.0 (e^0)
    fr_cumsum_pad = np.pad(fr_cumsum, ((0, 0), (1, 0)), constant_values=0.0)
    fp_view = np.exp(fr_cumsum_pad)

    peaks = np.maximum.accumulate(fp_view, axis=1)
    dd_fracs = np.max((peaks - fp_view) / (peaks + 1e-8), axis=1)
    dd_usds = dd_fracs * equity

    labels = (dd_usds > dd_thresh_usd).astype(np.int64)

    # 2. Vectorized Feature Window Extraction
    # ts_view shape: (valid_len, 1, _FEATURE_WINDOW, EMBEDDING_DIM)
    ts_view = np.lib.stride_tricks.sliding_window_view(
        tick_sequences[:n - forward_n],
        window_shape=(_FEATURE_WINDOW, EMBEDDING_DIM)
    )
    # Reshape to (valid_len, _FEATURE_DIM)
    X = ts_view[:valid_len].reshape(valid_len, _FEATURE_DIM).astype(np.float32)

    # 3. Mathematical proof of causality
    # X[0] is built from tick_sequences[0 : _FEATURE_WINDOW]
    # labels[0] is built from future_rets[_FEATURE_WINDOW : _FEATURE_WINDOW + forward_n]
    # This guarantees zero overlap and zero lookahead bias.
    assert len(X) == len(labels), f"Length mismatch: X={len(X)}, labels={len(labels)}"

    # 4. Memory-safe subsampling to prevent OOM Killer on large datasets
    # 1.29M samples * 1280 features = ~6.6GB. XGBoost doubles this, causing OOMs on 16GB machines.
    MAX_SAMPLES = 250000
    if len(X) > MAX_SAMPLES:
        logger.info(f"Sub-sampling Risk features from {len(X)} down to {MAX_SAMPLES} to prevent RAM OOM.")
        # Use random sampling to avoid skipping clustered tail events, then sort to preserve temporal order
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(X), size=MAX_SAMPLES, replace=False)
        idx.sort()
        X = X[idx]
        labels = labels[idx]

    return X, labels
