#!/usr/bin/env python3
"""
Hyperparameter Grid Search Engine (Two-Stage Tournament Protocol)
Latent Diffusion-HMM v3.0
"""

import os, sys, json, time
import itertools
import numpy as np
import pandas as pd
import concurrent.futures
import subprocess
from datetime import datetime

# ---------------------------------------------------------
# STAGE 1: THE QUALIFIERS (Scale Invariant Assets)
# ---------------------------------------------------------
import glob
import json

# Fallback assets if no ensemble results exist
SCALE_INVARIANT_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "AUDNZD", "AUDCAD", "EURCHF", "GBPCHF", "EURNOK", "EURSEK", "NOKSEK",
    "NZDCAD", "AUDCHF", "NZDCHF", "CADCHF", "EURAUD", "EURCAD", "GBPAUD",
    "CHFJPY", "CADJPY", "GBPCAD", "GBPNZD"
]

# Dynamically load the winning assets from the latest Ensemble run
ensemble_runs = sorted(glob.glob("results/ensemble_results/run_*/ensemble_results.json"))
if ensemble_runs:
    try:
        with open(ensemble_runs[-1], 'r') as f:
            edata = json.load(f)
            winning = []
            for asset in edata.get("per_asset", []):
                if asset["sharpe"] >= 0.40:  # Must clear baseline firewall
                    ticker_clean = asset["ticker"].split("/")[-1].split("_")[0].upper()
                    winning.append(ticker_clean)
            if winning:
                SCALE_INVARIANT_ASSETS = winning
                print(f"Loaded {len(SCALE_INVARIANT_ASSETS)} Winning Assets from Baseline Ensemble!")
    except Exception as e:
        print(f"Warning: Failed to load ensemble winners: {e}")
DATA_DIR = "data"
RESULTS_DIR = "grid_results"
TMP_BASE = os.path.join(RESULTS_DIR, ".tmp_grid")

# Speed settings: reduce EM iterations for grid search, suppress verbose output
os.environ.setdefault("HMM_N_ITER", "30")   # Fast mode: 30 EM iter
os.environ.setdefault("HMM_VERBOSE", "0")    # Suppress per-block prints
os.environ.setdefault("TIMEFRAME", "1H")     # Shift to 1-Hour Timeframe
os.environ.setdefault("HMM_WF_TRAIN_BARS", "1512") # ~3 months of 1H data

# ---------------------------------------------------------
# HYPERPARAMETER GRID
# ---------------------------------------------------------
GRID = {
    "VETO_THRESHOLD": [0.65, 0.70, 0.75],
    "TIME_EXIT_BARS": [8, 12, 24, 48],  # 8 hrs, 12 hrs, 1 day, 2 days
    "STOP_LOSS_ATR":  [3.0, 4.0, 5.0],
    "TAKE_PROFIT_ATR":[6.0, 8.0, 10.0],
    "HMM_WF_OOS_BARS":[120, 240, 504] # ~1 week, 2 weeks, 1 month
}
SLIPPAGE_PENALTY_KEY = "1.5 pip"  # Slippage stress test

def get_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for prod in itertools.product(*values):
        yield dict(zip(keys, prod))

def _sep(title: str = "", width: int = 74) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        left = pad // 2
        right = pad - left
        print(f"{'─'*left} {title} {'─'*right}")
    else:
        print("─" * width)

def process_asset(csv_path, tmp_dir, params):
    import sys, os, numpy as np
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v3_engine"))
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    import forex_tearsheet

    ticker = os.path.basename(csv_path).replace("_daily.csv", "").replace("_1h.csv", "").upper()
    try:
        os.environ["FOREX_RESULTS_PATH"] = os.path.join(tmp_dir, f"{ticker}_results.json")
        
        import io
        from contextlib import redirect_stdout
        with io.StringIO() as buf, redirect_stdout(buf):
            data = forex_tearsheet.run_tearsheet_dynamic(csv_path, params)
        
        
        scenarios = data.get("pip_scenarios", {})
        slip_key = next((k for k in scenarios.keys() if SLIPPAGE_PENALTY_KEY in k), None)
        if not slip_key:
            slip_key = next((k for k in scenarios.keys() if "0.1%" in k or "0.05%" in k), list(scenarios.keys())[-1])
            
        metrics = scenarios.get(slip_key, {})
        returns = np.array(metrics.get("returns_array", []))
        return ticker, True, {
            "returns": returns,
            "sharpe": metrics.get("sharpe_ratio", 0),
            "win_rate": metrics.get("win_rate_pct", 0),
            "trades": metrics.get("n_trades", 0)
        }
    except Exception as e:
        print(f"  [Error in worker] {ticker}: {e}")
        return ticker, False, None

def calculate_dsr(portfolio_returns, n_trials):
    import scipy.stats as stats
    returns = portfolio_returns[~np.isnan(portfolio_returns)]
    T = len(returns)
    if T < 5: return 0.0
    sr = float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))
    gamma_e = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    expected_max = (1 - gamma_e) * z1 + gamma_e * z2
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denominator <= 0 or np.isnan(denominator): return 0.0
    dsr_z = (sr - expected_max) * np.sqrt(T - 1) / denominator
    p_val = stats.norm.cdf(dsr_z)
    return p_val

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_BASE, exist_ok=True)
    
    combinations = list(get_combinations(GRID))
    total_trials = len(combinations)
    
    print("=" * 74)
    print("  LATENT DIFFUSION-HMM v3.0 — HYPERPARAMETER GRID SEARCH (STAGE 1)")
    print(f"  Total Combinations: {total_trials}")
    print(f"  Scale Invariant Assets: {SCALE_INVARIANT_ASSETS}")
    print("=" * 74)
    print()

    csv_files = []
    for asset in SCALE_INVARIANT_ASSETS:
        path = os.path.join(DATA_DIR, f"{asset.lower()}_1h.csv")
        if os.path.exists(path):
            csv_files.append(path)
            
    if not csv_files:
        print("✗ No Stage 1 assets found in data/ directory.")
        return

    leaderboard = []

    # Run combinations sequentially, but assets in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
        for i, params in enumerate(combinations, 1):
            pct = (i / total_trials) * 100
            print(f"  [{i:>3}/{total_trials} | {pct:>5.1f}%] Testing Params: {params}")
            t0 = time.time()
            
            # Parallel asset execution
            portfolio_returns = []
            total_trades = 0
            success_count = 0
            
            futures = [executor.submit(process_asset, csv, TMP_BASE, params) for csv in csv_files]
            for future in concurrent.futures.as_completed(futures):
                ticker, success, payload = future.result()
                if success and payload["trades"] > 0:
                    portfolio_returns.append(payload["returns"])
                    total_trades += payload["trades"]
                    success_count += 1
            
            elapsed = time.time() - t0
        
        if success_count > 0 and total_trades > 10:
            # Aggregate returns (Equal Weight)
            max_len = max(len(r) for r in portfolio_returns)
            agg_rets = np.zeros(max_len)
            for r in portfolio_returns:
                agg_rets[-len(r):] += r
            agg_rets = agg_rets / success_count
            
            # Calculate metrics
            daily_mean = agg_rets.mean()
            daily_vol = agg_rets.std() + 1e-9
            sr = daily_mean / daily_vol * np.sqrt(252)
            
            # Deflated Sharpe Penalty
            dsr_p = calculate_dsr(agg_rets, total_trials)
            
            leaderboard.append({
                "params": params,
                "sharpe": float(sr),
                "dsr_p": float(dsr_p),
                "trades": total_trades
            })
            
            status = "✓" if dsr_p < 0.05 else "✗"
            print(f"         → {status} SR={sr:+.2f}  DSR_p={dsr_p:.4f}  Trades={total_trades}  ({elapsed:.1f}s)")
        else:
            print(f"         → ─ No Trades Generated ({elapsed:.1f}s)")

    # Sort leaderboard
    leaderboard.sort(key=lambda x: x["dsr_p"])  # Lowest p-value is best
    
    print("\n" + "=" * 74)
    print("  🏆 STAGE 1 LEADERBOARD (TOP 10 DISTINCT ROBUST PARAMETERS)")
    print("=" * 74)
    
    top_10 = []
    seen_hash = set()
    
    for entry in leaderboard:
        if len(top_10) >= 10: break
        if entry["dsr_p"] >= 0.05: continue # Failed Deflated Sharpe
        
        # Prevent clustering by ensuring unique structural shapes
        shape_hash = f"{entry['params']['TIME_EXIT_BARS']}-{entry['params']['HMM_WF_OOS_BARS']}"
        if shape_hash in seen_hash and len(top_10) > 2:
            continue # We already have a similar setup, skip for diversity
            
        seen_hash.add(shape_hash)
        top_10.append(entry)
        
    for i, entry in enumerate(top_10, 1):
        print(f"  #{i} | DSR_p={entry['dsr_p']:.4f} | SR={entry['sharpe']:+.2f} | Trades={entry['trades']}")
        print(f"      {entry['params']}\n")
        
    # Export to JSON
    with open(os.path.join(RESULTS_DIR, "stage1_top10.json"), "w") as f:
        json.dump(top_10, f, indent=2)

if __name__ == "__main__":
    main()
