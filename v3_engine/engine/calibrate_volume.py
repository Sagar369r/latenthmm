"""
Microstructure Volume Threshold Calibrator
Scans high-resolution (M1/Tick) CSVs and calculates the optimal Volume-Clock threshold.
"""
import pandas as pd
import sys
import os

def calibrate_asset(csv_path: str, target_bars_per_day: int = 288):
    print(f"\nAnalyzing Microstructure for: {os.path.basename(csv_path)} ...")
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Strip timezone for safety
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_convert(None)  # type: ignore

    # [NEW] Scale Dukascopy Commodity Volume
    if "xauusd" in csv_path.lower() or "wti" in csv_path.lower() or "lightcmdusd" in csv_path.lower():
        df["volume"] = df["volume"] * 1_000_000

    total_volume = df['volume'].sum()
    total_days = (df.index[-1] - df.index[0]).days
    
    if total_days == 0:
        print("Dataset is too small to calibrate.")
        return

    avg_daily_volume = total_volume / total_days
    optimal_threshold = int(avg_daily_volume / target_bars_per_day)

    print(f"  Total Days Analyzed  : {total_days:,}")
    print(f"  Avg Daily Volume     : {avg_daily_volume:,.0f} ticks")
    print(f"  Target Bars/Day      : {target_bars_per_day}")
    print(f"  ===========================================")
    print(f"  ✅ OPTIMAL THRESHOLD   : {optimal_threshold:,}")
    print(f"  ===========================================\n")
    print(f"  → Plug {optimal_threshold} into your PipelineConfig for this asset.")
    
    return optimal_threshold

if __name__ == "__main__":
    # Test all three assets if __name__ == "__main__":
    assets = [
        "/home/suchith/Downloads/Latent-Diffusion-HMM/data/eurusd_2022_2024_1m.csv",
    ]
    
    for asset in assets:
        if os.path.exists(asset):
            calibrate_asset(asset, target_bars_per_day=288)
        else:
            print(f"Skipping {asset} (Not found)")
