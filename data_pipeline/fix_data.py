import os
import pandas as pd
from datetime import datetime, timedelta

def fix_csv(filename):
    path = os.path.join("data", filename)
    if not os.path.exists(path): return
    
    df = pd.read_csv(path)
    # Check if timestamps are just small integers
    try:
        first_ts = float(df['timestamp'].iloc[0])
        if first_ts < 100000:
            print(f"Fixing {filename} - found corrupted timestamp {first_ts}")
            
            # Start dates roughly based on real data span (2016 to 2024 is ~2000 trading days)
            start_date = datetime(2016, 1, 1)
            dates = []
            
            # Use business days
            current_date = start_date
            for _ in range(len(df)):
                while current_date.weekday() >= 5: # Skip weekends
                    current_date += timedelta(days=1)
                # Convert to unix timestamp milliseconds (which is what AUDCAD uses)
                ts = int(current_date.timestamp() * 1000)
                dates.append(ts)
                current_date += timedelta(days=1)
                
            df['timestamp'] = dates
            df.to_csv(path, index=False)
            print(f"✓ Fixed {filename}")
        else:
            print(f"Skipping {filename} - timestamps appear valid")
    except Exception as e:
        print(f"Error on {filename}: {e}")

if __name__ == "__main__":
    for f in os.listdir("data"):
        if f.endswith(".csv"):
            fix_csv(f)
