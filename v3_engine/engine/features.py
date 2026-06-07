from __future__ import annotations

import warnings
import numpy as np
import polars as pl
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Fractional Differencing Math (from data.py)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_frac_diff_weights(d: float, size: int) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)

def _apply_frac_diff(values: np.ndarray, d: float) -> np.ndarray:
    T = len(values)
    weights = _compute_frac_diff_weights(d, T)
    threshold = 1e-5
    active = np.abs(weights) > threshold
    max_lag = int(active.sum())
    if max_lag == 0: max_lag = 1

    w = weights[:max_lag]
    res = fftconvolve(values, w, mode='full')[:T]
    res[:max_lag] = np.nan
    return res

# ─────────────────────────────────────────────────────────────────────────────
# f1: f_fd — Fractional Differencing (Stationary Memory)
# ─────────────────────────────────────────────────────────────────────────────
def fractional_differencing(df: pl.DataFrame, d: float = 0.45) -> pl.Series:
    """Applies a fixed fractional differencing filter to log prices."""
    close = df["close"].to_numpy().astype(float)
    log_close = np.log(close + 1e-8)
    fd = _apply_frac_diff(log_close, d)
    return pl.Series(fd).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).alias("f_fd")

# ─────────────────────────────────────────────────────────────────────────────
# f2: f_ofi — Order Flow Imbalance Proxy
# ─────────────────────────────────────────────────────────────────────────────
def order_flow_imbalance(df: pl.DataFrame, lookback: int = 60) -> pl.Series:
    """Matches Pine Script: raw_ofi / max(abs(raw_ofi), 60)"""
    range_hl = (df["high"] - df["low"]).clip(lower_bound=1e-10)
    direction = (df["close"] - df["open"]) / range_hl
    raw_ofi = df["volume"] * direction
    
    ofi_max = raw_ofi.abs().rolling_max(window_size=lookback, min_periods=5).fill_null(1e-8)
    # prevent div by zero
    ofi_max = pl.when(ofi_max == 0).then(1e-8).otherwise(ofi_max)
    
    f_ofi = raw_ofi / ofi_max
    return f_ofi.clip(lower_bound=-1.0, upper_bound=1.0).alias("f_ofi")

# ─────────────────────────────────────────────────────────────────────────────
# f3: f_macro — Macro Trend Alignment Z-Score
# ─────────────────────────────────────────────────────────────────────────────
def macro_alignment(df: pl.DataFrame, sma_window: int = 200, norm_window: int = 288) -> pl.Series:
    """Distance to 200-SMA, normalized as a Z-score."""
    close = df["close"]
    sma200 = close.rolling_mean(window_size=sma_window, min_periods=20).fill_null(strategy="forward") + 1e-8
    dist = (close - sma200) / sma200
    
    dist_std = dist.rolling_std(window_size=norm_window, min_periods=20).clip(lower_bound=1e-8)
    z_score = dist / dist_std
    return z_score.clip(lower_bound=-5.0, upper_bound=5.0).alias("f_macro")

# ─────────────────────────────────────────────────────────────────────────────
# f4: f_vol — Multi-Timeframe Volatility Ratio (GARCH-Lite)
# ─────────────────────────────────────────────────────────────────────────────
def volatility_clustering(df: pl.DataFrame, short_w: int = 20, long_w: int = 100) -> pl.Series:
    """ln(sigma^2_20 / sigma^2_100) - Detects compression to expansion"""
    close = df["close"]
    log_ret = (close / (close.shift(1).fill_null(0.0) + 1e-8)).log()
    
    var_short = log_ret.rolling_var(window_size=short_w, min_periods=5).clip(lower_bound=1e-12)
    var_long = log_ret.rolling_var(window_size=long_w, min_periods=20).clip(lower_bound=1e-12)
    
    vol_ratio = (var_short / var_long).log()
    return vol_ratio.clip(lower_bound=-5.0, upper_bound=5.0).alias("f_vol")

# ─────────────────────────────────────────────────────────────────────────────
# f5: f_mom — Micro-Momentum Velocity
# ─────────────────────────────────────────────────────────────────────────────
def micro_momentum(df: pl.DataFrame, roc_w: int = 5, lookback: int = 60) -> pl.Series:
    """Matches Pine Script: roc5 / max(abs(roc5), 60)"""
    close = df["close"]
    close_shifted = close.shift(roc_w).fill_null(0.0) + 1e-8
    
    roc5 = (close - close_shifted) / close_shifted * 100
    roc_max = roc5.abs().rolling_max(window_size=lookback, min_periods=5).fill_null(1e-8)
    # prevent div by zero
    roc_max = pl.when(roc_max == 0).then(1e-8).otherwise(roc_max)
    
    f_mom = roc5 / roc_max
    return f_mom.clip(lower_bound=-1.0, upper_bound=1.0).alias("f_mom")

# ─────────────────────────────────────────────────────────────────────────────
# f6: f_liq — Relative Liquidity Surge
# ─────────────────────────────────────────────────────────────────────────────
def liquidity_surge(df: pl.DataFrame, short_w: int = 12, long_w: int = 288) -> pl.Series:
    """SMA(Volume, 12) / SMA(Volume, 288) - Institutional footprint tracker"""
    vol = df["volume"]
    sma_short = vol.rolling_mean(window_size=short_w, min_periods=1).clip(lower_bound=1e-8)
    sma_long = vol.rolling_mean(window_size=long_w, min_periods=12).clip(lower_bound=1e-8)
    
    surge = sma_short / sma_long
    return surge.clip(lower_bound=0.0, upper_bound=10.0).alias("f_liq")

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

def compute_feature_tensor(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        df_copy.index.name = "date"
        df_pl = pl.from_pandas(df_copy.reset_index())
    else:
        df_pl = df

    features = df_pl.select([
        fractional_differencing(df_pl, d=0.45),
        order_flow_imbalance(df_pl),
        macro_alignment(df_pl),
        volatility_clustering(df_pl),
        micro_momentum(df_pl),
        liquidity_surge(df_pl),
    ])

    features = features.fill_null(strategy="forward")
    if np.isinf(features.to_numpy().astype(float)).any():
        raise ValueError("CRITICAL: Infinity propagated in feature tensor.")

    if "date" in df_pl.columns:
        features = features.with_columns(df_pl["date"])
    
    return features.to_pandas().set_index("date")
