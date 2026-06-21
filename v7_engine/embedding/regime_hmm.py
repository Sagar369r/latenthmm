"""
Gaussian HMM Regime Classifier — Predicts market regime using structural time-series memory.

Regimes (Mapped to 4 hidden states)
"""
from __future__ import annotations
import logging
import os
import pickle
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class RegimeHMM:
    """
    GaussianHMM multi-class regime classifier.

    Input:  Raw mid prices buffer
    Output: (4,) probability vector of the hidden state
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([0.0, 0.0])
        self.scaler.scale_ = np.array([1.0, 1.0])
        
        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]

    @classmethod
    def load(cls, path: str) -> "RegimeHMM":
        return cls(path)
        
    def fit(self, bids: np.ndarray, asks: np.ndarray, save_path: str):
        """
        Trains the GaussianHMM on the provided bids and asks and saves it.
        Includes regularization to prevent covariance collapse.
        """
        import hmmlearn.hmm as hmm
        logger.info("Extracting HMM features for fitting...")
        stride = 100
        bids = bids[::stride]
        asks = asks[::stride]
        mid = (bids + asks) / 2.0
        
        returns = np.log(mid[1:] / mid[:-1])
        
        series_ret = pd.Series(returns)
        volatility = series_ret.rolling(50).var().bfill().values
        
        X = np.column_stack([returns, volatility])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Re-instantiate scaler to avoid sklearn _reset() bug
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info("Fitting GaussianHMM (4 states) with Regularization...")
        # Add regularization to prevent overfitting/look-ahead variance collapse
        self.model = hmm.GaussianHMM(
            n_components=4, 
            covariance_type="diag", 
            n_iter=500, 
            tol=1e-3,
            min_covar=1e-3, # Critical to prevent zero variance memorization
            init_params="stmc",
            params="stmc",
            random_state=42
        )
        self.model.fit(X_scaled)
        
        logger.info("HMM Fit successful!")
        
        # Save model and scaler
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"Saved regularized HMM to {save_path}")

    def predict_proba(self, mid_prices: np.ndarray) -> np.ndarray:
        """
        Calculates the HMM state probability from recent mid prices.
        mid_prices: array of recent mid prices (e.g. last 5000 ticks)
        Returns: (4,) probability vector
        """
        if self.model is None or len(mid_prices) < 200:
            return np.ones(4, dtype=np.float32) / 4.0
            
        # Match the stride=100 from training
        stride = 100
        mid_strided = mid_prices[::stride]
        
        if len(mid_strided) < 3:
            return np.ones(4, dtype=np.float32) / 4.0
            
        returns = np.log(mid_strided[1:] / mid_strided[:-1])
        
        # Volatility over a short rolling window
        series_ret = pd.Series(returns)
        # Using min_periods=1 so we don't get all NaNs if window is small
        volatility = series_ret.rolling(10, min_periods=1).var().bfill().values
        
        X = np.column_stack([returns, volatility])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale to match training distribution
        X_scaled = self.scaler.transform(X)
        
        try:
            # Decode the sequence using the transition matrix
            probs = self.model.predict_proba(X_scaled)
            # Return the probabilities of the hidden states at the LAST timestep
            return probs[-1].astype(np.float32)
        except Exception as e:
            logger.debug(f"HMM predict failed: {e}")
            return np.ones(4, dtype=np.float32) / 4.0
