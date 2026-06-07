import os
import glob
import pandas as pd
import time

def clean_data_directory(data_dir: str = "data"):
    print(f"🧹 Commencing Data Integrity Audit on directory: {data_dir}/")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    total_files = len(csv_files)
    total_rows_dropped = 0
    total_files_cleaned = 0
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "timestamp" not in df.columns:
                print(f"  [SKIP] {os.path.basename(f)} - No 'timestamp' column found.")
                continue
                
            original_len = len(df)
            
            # Deduplicate based on exact timestamp match, keeping the latest fetched data point
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            
            # Sort chronologically to prevent look-ahead bias or mixed-order bugs
            df = df.sort_values(by="timestamp")
            
            new_len = len(df)
            dropped = original_len - new_len
            
            if dropped > 0 or not df.index.is_monotonic_increasing:
                df.to_csv(f, index=False)
                total_rows_dropped += dropped
                total_files_cleaned += 1
                print(f"  [FIXED] {os.path.basename(f)} - Dropped {dropped} duplicate rows & sorted.")
            else:
                pass # print(f"  [CLEAN] {os.path.basename(f)} - Perfect.")
                
        except Exception as e:
            print(f"  [ERROR] Failed to clean {os.path.basename(f)}: {e}")
            
    print("\n" + "="*50)
    print("  🧹 DATA AUDIT COMPLETE")
    print("="*50)
    print(f"  Total Files Audited : {total_files}")
    print(f"  Files Needing Fixes : {total_files_cleaned}")
    print(f"  Duplicate Rows Axed : {total_rows_dropped}")
    print("="*50 + "\n")

if __name__ == "__main__":
    t0 = time.time()
    clean_data_directory()
    print(f"Audit executed in {time.time() - t0:.2f}s")
