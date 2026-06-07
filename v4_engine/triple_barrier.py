import polars as pl

def apply_triple_barrier(df: pl.DataFrame, tp_mult=2.0, sl_mult=1.0, time_limit=24) -> pl.DataFrame:
    """
    Implements the Triple Barrier Labeling Method vector-wise using Polars.
    Upper Barrier: entry + (tp_mult * atr)
    Lower Barrier: entry - (sl_mult * atr)
    Time Barrier: time_limit
    
    Returns a DataFrame with a new integer column 'target' (1 for hit TP, 0 otherwise).
    """
    close = df["close"]
    atr = df["atr"]
    
    upper_barrier = close + (tp_mult * atr)
    lower_barrier = close - (sl_mult * atr)
    
    # Initialize physical columns in the dataframe
    df = df.with_columns([
        pl.lit(False).alias("hit_tp"),
        pl.lit(False).alias("hit_sl")
    ])
    
    for i in range(1, time_limit + 1):
        future_high = df["high"].shift(-i)
        future_low = df["low"].shift(-i)
        
        # We must read the materialized columns
        current_hit_tp = df["hit_tp"]
        current_hit_sl = df["hit_sl"]
        
        tp_condition = (future_high >= upper_barrier).fill_null(False)
        sl_condition = (future_low <= lower_barrier).fill_null(False)
        
        new_step_tp = tp_condition & ~current_hit_sl & ~sl_condition
        new_step_sl = sl_condition & ~current_hit_tp
        
        # Accumulate and MATERIALIZE immediately to prevent AST explosion
        df = df.with_columns([
            (current_hit_tp | new_step_tp).alias("hit_tp"),
            (current_hit_sl | new_step_sl).alias("hit_sl")
        ])
        
    df = df.with_columns(pl.when(pl.col("hit_tp")).then(1).otherwise(0).cast(pl.Int32).alias("target"))
    # Drop the temporary tracking columns
    df = df.drop(["hit_tp", "hit_sl"])
    return df
