import polars as pl
import numpy as np

def calculate_atr(df: pl.DataFrame, period: int = 14) -> pl.Series:
    """Calculates Average True Range"""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1).fill_null(df["open"])
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pl.max_horizontal([tr1, tr2, tr3])
    
    # Simple rolling mean for ATR
    atr = tr.rolling_mean(window_size=period, min_samples=1).fill_null(strategy="backward")
    atr = pl.when(atr < 1e-10).then(1e-10).otherwise(atr)
    return atr.alias("atr")

def transform_to_dimensionless(df: pl.DataFrame, window: int = 90) -> pl.DataFrame:
    """
    Phase 1 Data Foundry Transformer:
    Transforms raw OHLCV price data into a strictly dimensionless, stationary Z-score representation.
    This guarantees Bitcoin ($70,000) and EUR/GBP (0.85) are processed identically by the VAE/HMM.
    """
    # Force lowercase columns
    df = df.rename({c: c.lower() for c in df.columns})
    
    # 1. Log Returns
    close = df["close"]
    close_prev = close.shift(1).fill_null(strategy="backward")
    
    # Prevent log errors
    close = pl.when(close <= 0).then(1e-8).otherwise(close)
    close_prev = pl.when(close_prev <= 0).then(1e-8).otherwise(close_prev)
    
    log_ret = (close / close_prev).log().alias("log_return")
    
    # 2. Add ATR (for Stop Loss / Risk sizing later)
    atr = calculate_atr(df, 14)
    
    # 3. Rolling 90-period Z-Score of Log Returns (The Dimensionless Transformer)
    roll_mean = log_ret.rolling_mean(window_size=window, min_samples=5)
    roll_std = log_ret.rolling_std(window_size=window, min_samples=5)
    
    roll_std = pl.when(roll_std < 1e-10).then(1e-10).otherwise(roll_std)
    
    z_score = ((log_ret - roll_mean) / roll_std).clip(lower_bound=-10.0, upper_bound=10.0).alias("z_score")
    
    # 4. Volume Z-Score (Dimensionless Volume)
    vol = df["volume"]
    vol_mean = vol.rolling_mean(window_size=window, min_samples=5)
    vol_std = vol.rolling_std(window_size=window, min_samples=5).clip(lower_bound=1e-10)
    vol_z = ((vol - vol_mean) / vol_std).clip(lower_bound=-10.0, upper_bound=10.0).alias("vol_z_score")
    
    # Construct output
    out = df.with_columns([
        log_ret,
        atr,
        z_score,
        vol_z
    ]).fill_null(0.0)
    
    return out
