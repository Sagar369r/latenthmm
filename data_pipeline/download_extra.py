import yfinance as yf
import pandas as pd
import os

# Tickers to download
tickers = {
    "SPY": "spy",
    "QQQ": "qqq",
    "BTC-USD": "btcusd",
    "EURUSD=X": "eurusd",
    "GBPUSD=X": "gbpusd"
}

os.makedirs("data", exist_ok=True)

for ticker, name in tickers.items():
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start="2016-01-01", end="2024-12-31")
    
    if df.empty:
        print(f"Failed to download {ticker}")
        continue
        
    # Reset index to get Date
    df = df.reset_index()
    
    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # yfinance column names: Date, Open, High, Low, Close, Adj Close, Volume
    # We need: timestamp, open, high, low, close, volume
    
    df = df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Convert timestamp to unix milliseconds
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype('int64') // 10**6
    
    # Select columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    out_path = f"data/{name}_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

print("Done downloading extra assets!")
