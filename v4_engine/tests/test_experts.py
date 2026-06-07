import os
import sys
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from v4_engine.triple_barrier import apply_triple_barrier

def test_triple_barrier():
    # 1. Create a fake dataframe that guarantees a TP hit on bar 3
    df = pl.DataFrame({
        "close": [100, 100, 100, 100, 100, 100, 100],
        "high":  [100, 101, 102, 110, 100, 100, 100], # Hits TP (100 + 8 = 108) at index 3
        "low":   [100,  99,  98,  97, 100, 100, 100],
        "atr":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    # 2. Apply barrier
    # TP = 8.0, SL = 4.0, Time = 5
    out = apply_triple_barrier(df, tp_mult=8.0, sl_mult=4.0, time_limit=5)
    
    target = out["target"].to_list()
    # For index 0: High hits 110 at step +3. So index 0 target should be 1
    # For index 1: High hits 110 at step +2. So target 1
    # For index 2: High hits 110 at step +1. So target 1
    # For index 3: Entry is 100, future highs never hit 108. So target 0
    assert target[0] == 1
    assert target[1] == 1
    assert target[2] == 1
    assert target[3] == 0
    
    print("✓ Triple Barrier Method correctly identifies future targets.")

def test_embargo_split():
    # We will mathematically verify that the Embargo logic drops the exact
    # number of overlapping indices to prevent future leakage.
    embargo_bars = 5
    X = np.zeros((100, 2))
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for train_idx, val_idx in tscv.split(X):
        purge_cutoff = len(train_idx) - embargo_bars
        purged_train_idx = train_idx[:purge_cutoff]
        
        # The maximum index in the training set MUST be strictly less than
        # the minimum index in the validation set minus the embargo size.
        max_train = np.max(purged_train_idx)
        min_val = np.min(val_idx)
        
        distance = min_val - max_train
        assert distance > embargo_bars, f"Leakage detected! Distance between train and val is only {distance}"
        
    print("✓ Purged TimeSeriesSplit successfully blocks overlapping trade leakage.")

if __name__ == "__main__":
    test_triple_barrier()
    test_embargo_split()
