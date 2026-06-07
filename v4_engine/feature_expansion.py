import polars as pl
import numpy as np

def generate_20_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates a robust quantitative basket of 20 technical indicators.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df["volume"]
    
    # Fill nulls and avoid division by zero
    close = close.fill_null(strategy="forward")
    vol = vol.fill_null(0.0) + 1e-10
    
    features = []
    
    # === TREND (Distance from SMA) ===
    for w in [10, 20, 50, 100, 200]:
        sma = close.rolling_mean(window_size=w, min_samples=1)
        features.append(((close - sma) / sma).alias(f"trend_sma_{w}"))
        
    # === MOMENTUM (Rate of Change) ===
    for w in [5, 10, 20, 50]:
        prev = close.shift(w).fill_null(close)
        features.append(((close - prev) / prev).alias(f"mom_roc_{w}"))
        
    # MACD (12, 26)
    ema12 = close.ewm_mean(span=12, min_samples=1, adjust=False)
    ema26 = close.ewm_mean(span=26, min_samples=1, adjust=False)
    features.append(((ema12 - ema26) / close).alias("mom_macd"))
    
    # === VOLATILITY ===
    # ATR Approximation
    tr = pl.max_horizontal([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ])
    atr5 = tr.rolling_mean(window_size=5, min_samples=1)
    atr20 = tr.rolling_mean(window_size=20, min_samples=1).clip(lower_bound=1e-10)
    features.append((atr5 / atr20).alias("vol_atr_ratio"))
    
    # Bollinger Widths
    for w in [20, 50]:
        sma = close.rolling_mean(window_size=w, min_samples=1)
        std = close.rolling_std(window_size=w, min_samples=1).clip(lower_bound=1e-10)
        features.append((std / sma).alias(f"vol_bb_width_{w}"))
        
    # Log return variance
    log_ret = (close / close.shift(1).fill_null(close)).log()
    for w in [10, 20]:
        features.append(log_ret.rolling_var(window_size=w, min_samples=1).alias(f"vol_var_{w}"))
        
    # === VOLUME / ORDER FLOW ===
    # Volume surges
    for short_w, long_w in [(5, 20), (10, 50)]:
        v_short = vol.rolling_mean(window_size=short_w, min_samples=1)
        v_long = vol.rolling_mean(window_size=long_w, min_samples=1).clip(lower_bound=1e-10)
        features.append((v_short / v_long).alias(f"volm_surge_{short_w}_{long_w}"))
        
    # Volume ROC
    v_prev = vol.shift(5).fill_null(vol)
    features.append(((vol - v_prev) / v_prev).alias("volm_roc_5"))
    
    # Order Flow Imbalance (Directional Volume)
    range_hl = (high - low).clip(lower_bound=1e-10)
    direction = (close - open_) / range_hl
    raw_ofi = vol * direction
    
    for w in [10, 20]:
        ofi_ema = raw_ofi.ewm_mean(span=w, min_samples=1, adjust=False)
        vol_ema = vol.ewm_mean(span=w, min_samples=1, adjust=False)
        features.append((ofi_ema / vol_ema).alias(f"volm_ofi_{w}"))
        
    old_cols = set(df.columns)
    df_feats = df.with_columns(features)
    
    # Collect the column names of our 20 newly generated indicators
    feat_cols = list(set(df_feats.columns) - old_cols)
    return df_feats, feat_cols


def extract_z_score_tensor(df: pl.DataFrame, window: int = 90) -> pl.DataFrame:
    """
    1. Generates the 20 raw indicators.
    2. Applies the Dimensionless Data Transformer (90-day rolling Z-Score) to ALL of them.
    """
    df, feat_cols = generate_20_indicators(df)
    
    z_features = []
    for col in feat_cols:
        series = df[col].fill_nan(0.0).fill_null(0.0)
        roll_mean = series.rolling_mean(window_size=window, min_samples=5)
        roll_std = series.rolling_std(window_size=window, min_samples=5).clip(lower_bound=1e-10)
        
        z = ((series - roll_mean) / roll_std).clip(lower_bound=-10.0, upper_bound=10.0).alias(f"z_{col}")
        z_features.append(z)
        
    df = df.with_columns(z_features).fill_null(0.0).fill_nan(0.0)
    
    z_cols = [f.name for f in z_features]
    return df, z_cols
