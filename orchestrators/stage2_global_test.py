#!/usr/bin/env python3
"""
Hyperparameter Grid Search Engine (STAGE 2)
Global Stress Test across all assets.
"""

import os, sys, json, glob, time
import numpy as np
import concurrent.futures
import subprocess

DATA_DIR = "data"
RESULTS_DIR = "results/grid_results"
TMP_BASE = os.path.join(RESULTS_DIR, ".tmp_stage2")
STAGE1_FILE = os.path.join(RESULTS_DIR, "stage1_top10.json")
SLIPPAGE_PENALTY_KEY = "1.5 pip"

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
        return ticker, True, {
            "returns": np.array(metrics.get("returns_array", [])),
            "sharpe": metrics.get("sharpe_ratio", 0),
            "win_rate": metrics.get("win_rate_pct", 0),
            "trades": metrics.get("n_trades", 0)
        }
    except Exception as e:
        print(f"  [Error in worker] {ticker}: {e}")
        return ticker, False, None

def main():
    if not os.path.exists(STAGE1_FILE):
        print(f"✗ Stage 1 results not found at {STAGE1_FILE}. Run grid_search.py first.")
        return
        
    with open(STAGE1_FILE, "r") as f:
        top_10 = json.load(f)
        
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_1h.csv")))
    if not csv_files:
        print("✗ No assets found in data/.")
        return
        
    os.makedirs(TMP_BASE, exist_ok=True)
    
    os.environ.setdefault("TIMEFRAME", "1H")
    os.environ.setdefault("HMM_WF_TRAIN_BARS", "1512")
    
    print("=" * 74)
    print("  LATENT DIFFUSION-HMM v3.0 — GLOBAL STRESS TEST (STAGE 2)")
    print(f"  Testing Top {len(top_10)} parameters across all {len(csv_files)} assets.")
    print("=" * 74)
    print()
    
    final_leaderboard = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
        for i, entry in enumerate(top_10, 1):
            params = entry["params"]
            pct = (i / len(top_10)) * 100
            print(f"  [{i:>2}/{len(top_10)} | {pct:>5.1f}%] Testing Params: {params}")
            t0 = time.time()
            
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
        
        if success_count > 0 and total_trades > 0:
            max_len = max(len(r) for r in portfolio_returns)
            agg_rets = np.zeros(max_len)
            for r in portfolio_returns:
                agg_rets[-len(r):] += r
            agg_rets = agg_rets / success_count
            
            daily_mean = agg_rets.mean()
            daily_vol = agg_rets.std() + 1e-9
            sr = daily_mean / daily_vol * np.sqrt(252)
            
            final_leaderboard.append({
                "params": params,
                "global_sharpe": float(sr),
                "global_trades": total_trades,
                "stage1_dsr_p": entry["dsr_p"]
            })
            print(f"         → Global SR={sr:+.2f}  Global Trades={total_trades}  ({elapsed:.1f}s)")
        else:
            print(f"         → ─ No Trades Generated ({elapsed:.1f}s)")

    final_leaderboard.sort(key=lambda x: x["global_sharpe"], reverse=True)
    
    print("\n" + "=" * 74)
    print("  🌍 STAGE 2 GLOBAL LEADERBOARD (THE ULTIMATE SURVIVORS)")
    print("=" * 74)
    
    for i, entry in enumerate(final_leaderboard, 1):
        print(f"  #{i} | Global SR={entry['global_sharpe']:+.2f} | Trades={entry['global_trades']}")
        print(f"      {entry['params']}\n")

if __name__ == "__main__":
    main()
