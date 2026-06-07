import os
import sys
import json
import argparse
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
try:
    import yfinance as yf
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "yfinance"], check=True)
    import yfinance as yf

def fetch_yahoo_data(ticker: str, start_date: str = "2016-01-01", end_date: str = "2024-12-31"):
    print(f"📡 Fetching High-Quality Daily Data for {ticker} from Yahoo Finance...")
    
    # Map common symbols to Yahoo Finance formats
    yf_mapping = {
        "XAUUSD": "GC=F",     # Gold Futures
        "XAGUSD": "SI=F",     # Silver Futures
        "BTCUSD": "BTC-USD",  # Bitcoin
        "ETHUSD": "ETH-USD",  # Ethereum
        "SPX": "^GSPC",       # S&P 500 Index
        "NDX": "^NDX"         # Nasdaq 100 Index
    }
    
    fetch_ticker = yf_mapping.get(ticker, ticker)
    
    # Auto-detect Forex pairs (e.g. EURUSD -> EURUSD=X) if length is exactly 6 and no map matched
    if fetch_ticker == ticker and len(ticker) == 6 and ticker.isalpha() and not ticker.endswith("=X"):
        fetch_ticker = f"{ticker}=X"
        
    df = yf.download(fetch_ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker} ({fetch_ticker}). Invalid symbol or delisted.")
        
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype('int64') // 10**6
    
    if "volume" not in df.columns or df["volume"].isnull().all():
        df["volume"] = 0
        
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df

def run_backtest(csv_path: str, ticker: str):
    print(f"\n⚙️  Running Latent Diffusion Backtest Engine on {ticker}...")
    
    result_path = os.path.join("ensemble_results", f"{ticker.lower()}_dynamic_results.json")
    os.makedirs("ensemble_results", exist_ok=True)
    
    env = os.environ.copy()
    env["HMM_STOP_LOSS_ATR"] = "4.0"
    env["HMM_TAKE_PROFIT_ATR"] = "8.0"
    env["STRATEGY_MODE"] = "MEAN_REVERSION_EXHAUSTION"
    env["TIMEFRAME"] = "1D"
    env["TIME_EXIT_BARS"] = "5"
    env["FOREX_RESULTS_PATH"] = os.path.abspath(result_path)
    
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "../v3_engine/forex_tearsheet.py", csv_path],
        env=env,
        capture_output=True,
        text=True
    )
    
    if proc.returncode != 0:
        print("❌ Backtest Engine Failed!")
        print(proc.stderr)
        return None
        
    print(f"✅ Backtest completed in {time.time() - t0:.1f}s")
    
    if not os.path.exists(result_path):
        print("❌ Results JSON not found.")
        return None
        
    with open(result_path, "r") as f:
        return json.load(f)

def print_tearsheet(data: dict, ticker: str):
    print("\n" + "="*50)
    print(f"  📊 TEAR SHEET: {ticker}")
    print("="*50)
    
    scenarios = data.get("pip_scenarios", {})
    gross_key = next((k for k in scenarios.keys() if "gross" in k), None)
    gross = scenarios.get(gross_key, {}) if gross_key else {}
    trades = gross.get("n_trades", 0)
    win_rate = gross.get("win_rate_pct", 0.0)
    pf = gross.get("profit_factor", 0.0)
    sr = gross.get("sharpe_ratio", 0.0)
    ret = gross.get("total_ret_pct", 0.0)
    
    print(f"Total Trades : {trades}")
    print(f"Win Rate     : {win_rate:.1f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Sharpe Ratio : {sr:+.2f}")
    print(f"Total Return : {ret:+.1f}%")
    
    if trades == 0:
        print("\n⚠️  WARNING: Strategy generated 0 valid setups. Div/0 aborted.")
    elif sr > 0.4:
        print("\n✅ VERDICT: EXCELLENT (ADMITTED)")
    else:
        print("\n❌ VERDICT: POOR EDGE (REJECTED)")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Diffusion Dynamic Tester")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. SPY, EURUSD)")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    import tempfile
    
    try:
        df = fetch_yahoo_data(ticker)
        
        # Use a temporary file so no CSV is permanently saved to the hard drive
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            csv_path = tmp.name
            df.to_csv(csv_path, index=False)
        
        print(f"✅ Downloaded {ticker} securely into memory. No CSV files saved.")
        
        results = run_backtest(csv_path, ticker)
        if results:
            print_tearsheet(results, ticker)
            
        # Clean up the temporary file
        os.remove(csv_path)
            
    except Exception as e:
        print(f"❌ Pipeline Error: {str(e)}")
