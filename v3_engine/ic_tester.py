import sys
import os
import time
import numpy as np
import pandas as pd
from scipy import stats

# Ensure engine modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.data import load_and_prepare
from engine.features import compute_feature_tensor

def _sep(title: str = "", width: int = 74) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        left = pad // 2
        right = pad - left
        print(f"{'─'*left} {title} {'─'*right}")
    else:
        print("─" * width)

def run_ic_tester(filepath: str):
    print()
    _sep("LATENT DIFFUSION-HMM  —  INFORMATION COEFFICIENT (IC) TESTER")
    print(f"  Target Asset: {filepath}")
    
    t0 = time.time()
    
    # 1. Ingest Base Data
    print("  Ingesting 1-Hour base data...")
    d_layer1 = load_and_prepare(
        filepath=filepath,
        start="2020-01-01",
        end="2024-12-31",
        volume_threshold=0,
        apply_frac_diff=False,
    )
    base_df = d_layer1["bars_df"].copy()
    
    timeframes = ['1h', '4h', '1d']
    k_horizons = [1, 3, 5]
    
    for tf in timeframes:
        print()
        _sep(f"ANALYZING TIMEFRAME: {tf.upper()}")
        
        # Resampling
        if tf == '1h':
            bars_df = base_df.copy()
        else:
            # Pandas resample
            bars_df = base_df.copy()
            # If index is not datetime, it should be set
            if "date" in bars_df.columns:
                bars_df.set_index("date", inplace=True)
            elif not isinstance(bars_df.index, pd.DatetimeIndex):
                bars_df.index = pd.to_datetime(bars_df.index)
                
            bars_df = bars_df.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # The features computation expects specific column structure, sometimes with date
            # compute_feature_tensor resets index internally if it is passed a DataFrame
        
        # 2. Compute Features
        features = compute_feature_tensor(bars_df)
        
        # 3. Calculate Forward Returns
        returns_dict = {}
        close = bars_df["close"].values
        
        for k in k_horizons:
            fwd_ret = np.log(bars_df["close"].shift(-k) / bars_df["close"])
            returns_dict[k] = fwd_ret.values
            
        feature_names = features.columns
        results = {feat: {k: {"ic": 0.0, "p": 0.0} for k in k_horizons} for feat in feature_names}
        
        # 4. Compute Spearman Rank Correlation
        for feat in feature_names:
            feat_vals = features[feat].values
            
            for k in k_horizons:
                ret_vals = returns_dict[k]
                
                valid_mask = ~np.isnan(feat_vals) & ~np.isnan(ret_vals) & ~np.isinf(feat_vals) & ~np.isinf(ret_vals)
                
                if np.sum(valid_mask) > 100:
                    rho, p_val = stats.spearmanr(feat_vals[valid_mask], ret_vals[valid_mask])
                    results[feat][k]["ic"] = float(rho)
                    results[feat][k]["p"]  = float(p_val)
                else:
                    results[feat][k]["ic"] = 0.0
                    results[feat][k]["p"]  = 1.0
        
        # 5. Formatted Output Table
        header = f"  {'Feature':<15} | {'1-Bar (k=1)':<15} | {'3-Bar (k=3)':<15} | {'5-Bar (k=5)':<15}"
        print(header)
        print("  " + "-" * 68)
        
        for feat in feature_names:
            row_str = f"  {feat:<15} |"
            for k in k_horizons:
                ic = results[feat][k]["ic"]
                p  = results[feat][k]["p"]
                
                # Grade formatting
                grade_char = " "
                if abs(ic) > 0.10:
                    grade_char = "★"
                elif abs(ic) > 0.05:
                    grade_char = "✓"
                elif abs(ic) < 0.02:
                    grade_char = "✗"
                    
                cell = f" {ic:+.4f} {grade_char} "
                row_str += f"{cell:<15} |"
                
            print(row_str)
            
        print("  " + "-" * 68)
        
    print(f"\n  Processed all timeframes in {time.time()-t0:.2f}s")
    print("  Legend:  ★ > 0.10 (Holy Grail)  |  ✓ > 0.05 (Edge)  |  ✗ < 0.02 (Noise)")
    _sep()
    print()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "../data/xauusd-h1-bid-2020-01-01-2024-12-31.csv"
    run_ic_tester(target)
