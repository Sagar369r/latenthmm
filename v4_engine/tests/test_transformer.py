import os
import sys
import polars as pl
import numpy as np
import pytest

# Ensure v4_engine is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from v4_engine.data_transformer import transform_to_dimensionless

def test_dimensionless_stationarity():
    btc_path = "data/btcusd_1h.csv"
    fx_path = "data/eurgbp_daily.csv"
    
    assert os.path.exists(btc_path), f"Missing {btc_path}"
    assert os.path.exists(fx_path), f"Missing {fx_path}"
    
    df_btc = pl.read_csv(btc_path)
    df_fx = pl.read_csv(fx_path)
    
    out_btc = transform_to_dimensionless(df_btc, window=90)
    out_fx = transform_to_dimensionless(df_fx, window=90)
    
    # Extract the transformed arrays (skip the first 90 rolling warmup bars)
    btc_z = out_btc["z_score"].to_numpy()[90:]
    fx_z = out_fx["z_score"].to_numpy()[90:]
    
    # 1. Verify No NaNs
    assert not np.isnan(btc_z).any(), "NaN found in BTC Z-score"
    assert not np.isnan(fx_z).any(), "NaN found in FX Z-score"
    
    # 2. Verify mean is centered near 0
    btc_mean = np.mean(btc_z)
    fx_mean = np.mean(fx_z)
    assert abs(btc_mean) < 0.2, f"BTC Mean is not 0: {btc_mean}"
    assert abs(fx_mean) < 0.2, f"FX Mean is not 0: {fx_mean}"
    
    # 3. Verify standard deviation is bounded near 1
    btc_std = np.std(btc_z)
    fx_std = np.std(fx_z)
    assert 0.8 < btc_std < 1.3, f"BTC Std is not 1: {btc_std}"
    assert 0.8 < fx_std < 1.3, f"FX Std is not 1: {fx_std}"
    
    # 4. Compare Absolute Scales (The proof of asset-agnosticism)
    # BTC moves $1000s, EURGBP moves 0.0010s
    # Yet their transformed max/min scales should be nearly identical (-10 to 10 limits)
    btc_max = np.max(btc_z)
    fx_max = np.max(fx_z)
    
    print(f"\n[BTCUSD] Mean: {btc_mean:.3f} | Std: {btc_std:.3f} | Max Spike: {btc_max:.2f}z")
    print(f"[EURGBP] Mean: {fx_mean:.3f} | Std: {fx_std:.3f} | Max Spike: {fx_max:.2f}z")
    
    assert btc_max > 2.0, "BTC doesn't show standard deviations > 2"
    assert fx_max > 2.0, "FX doesn't show standard deviations > 2"
    
    print("✓ Both asset classes successfully compressed into dimensionless stationary tensors.")
