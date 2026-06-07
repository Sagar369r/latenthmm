"""
Layer 1: Data Ingestion
- Dollar-Volume bar construction
- Fractional Differentiation with MMMS (Maximum-Memory Minimal-Stationarity)
"""
from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import yfinance as yf
from numba import njit
from statsmodels.tsa.stattools import adfuller
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore")

def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pl.DataFrame:
    """Fetch OHLCV data. Reads from local CSV if ticker ends in .csv"""
    if ticker.endswith(".csv"):
        df = pl.read_csv(ticker)
        df = df.rename({c: c.lower() for c in df.columns})
        
        if "timestamp" in df.columns:
            df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("date"))
        elif "date" in df.columns:
            # Parse string date to Datetime
            df = df.with_columns(
                pl.coalesce([
                    pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
                ])
            )
            
        start_dt = pl.lit(start).str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
        end_dt = pl.lit(end + " 23:59:59").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        
        df = df.filter((pl.col("date") >= start_dt) & (pl.col("date") <= end_dt))
        df = df.select(["date", "open", "high", "low", "close", "volume"]).fill_null(strategy="forward").drop_nulls()

        if "xauusd" in ticker.lower() or "wti" in ticker.lower() or "lightcmdusd" in ticker.lower():
            df = df.with_columns(pl.col("volume") * 1_000_000)
            
        return df
    else:
        # Fallback to yfinance
        import pandas as pd
        df_pd = yf.download(ticker, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=True)
        if df_pd.empty:
            raise ValueError(f"No data returned for {ticker}")
            
        df_pd.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_pd.columns]
        df_pd.index = pd.to_datetime(df_pd.index, utc=True).tz_localize(None)
        
        df_pd = df_pd.reset_index()
        # Rename the first column (the index) to 'date'
        first_col = df_pd.columns[0]
        df_pd = df_pd.rename(columns={first_col: "date"})
        df_pd = df_pd[["date", "open", "high", "low", "close", "volume"]].ffill().dropna()
        
        return pl.from_pandas(df_pd)


@njit(fastmath=True)
def _build_volume_bars_numba(
    date_arr: np.ndarray, open_arr: np.ndarray, high_arr: np.ndarray, 
    low_arr: np.ndarray, close_arr: np.ndarray, vol_arr: np.ndarray, threshold: float
):
    T = len(open_arr)
    out_date = np.zeros(T, dtype=np.int64)
    out_open = np.zeros(T, dtype=np.float64)
    out_high = np.zeros(T, dtype=np.float64)
    out_low  = np.zeros(T, dtype=np.float64)
    out_close= np.zeros(T, dtype=np.float64)
    out_vol  = np.zeros(T, dtype=np.float64)
    
    count = 0
    cum_vol = 0.0
    
    if T == 0:
        return out_date[:0], out_open[:0], out_high[:0], out_low[:0], out_close[:0], out_vol[:0]
        
    o = open_arr[0]
    h = high_arr[0]
    l = low_arr[0]
    
    for i in range(T):
        cum_vol += vol_arr[i]
        if high_arr[i] > h: h = high_arr[i]
        if low_arr[i] < l: l = low_arr[i]
        
        if cum_vol >= threshold or i == T - 1:
            out_date[count] = date_arr[i]
            out_open[count] = o
            out_high[count] = h
            out_low[count] = l
            out_close[count] = close_arr[i]
            out_vol[count] = cum_vol
            count += 1
            cum_vol = 0.0
            
            if i < T - 1:
                o = open_arr[i + 1]
                h = high_arr[i + 1]
                l = low_arr[i + 1]
                
    return out_date[:count], out_open[:count], out_high[:count], out_low[:count], out_close[:count], out_vol[:count]


def build_volume_bars(df: pl.DataFrame, volume_threshold: int) -> pl.DataFrame:
    """Construct Volume-Synchronized bars from OHLCV data using Numba."""
    if df.height == 0:
        return df

    date_arr = df["date"].dt.timestamp("ms").to_numpy()
    open_arr = df["open"].cast(pl.Float64).to_numpy()
    high_arr = df["high"].cast(pl.Float64).to_numpy()
    low_arr = df["low"].cast(pl.Float64).to_numpy()
    close_arr = df["close"].cast(pl.Float64).to_numpy()
    vol_arr = df["volume"].cast(pl.Float64).to_numpy()

    od, oo, oh, ol, oc, ov = _build_volume_bars_numba(
        date_arr, open_arr, high_arr, low_arr, close_arr, vol_arr, float(volume_threshold)
    )

    return pl.DataFrame({
        "date": pl.Series(od).cast(pl.Datetime("ms")),
        "open": oo,
        "high": oh,
        "low": ol,
        "close": oc,
        "volume": ov
    })


def _compute_frac_diff_weights(d: float, size: int) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def _apply_frac_diff(values: np.ndarray, d: float) -> np.ndarray:
    T = len(values)
    weights = _compute_frac_diff_weights(d, T)
    threshold = 1e-5
    active = np.abs(weights) > threshold
    max_lag = int(active.sum())
    if max_lag == 0: max_lag = 1

    w = weights[:max_lag]
    res = fftconvolve(values, w, mode='full')[:T]
    res[:max_lag] = np.nan
    return res


def _adf_pvalue(series: np.ndarray) -> float:
    clean = series[~np.isnan(series)]
    if len(clean) < 20:
        return 1.0
    try:
        return float(adfuller(clean, maxlag=1, autolag=None)[1])
    except Exception:
        return 1.0


def _find_mmms_d(values: np.ndarray, max_d: float = 1.0, tol: float = 0.01,
                  pvalue_threshold: float = 0.05) -> float:
    values = values[-5000:]
    lo, hi = 0.0, max_d
    for _ in range(25):
        mid = (lo + hi) / 2.0
        if abs(hi - lo) < tol:
            break
        fd = _apply_frac_diff(values, mid)
        pval = _adf_pvalue(fd)
        if pval < pvalue_threshold:
            hi = mid
        else:
            lo = mid
    return hi


def fractional_differentiation(
    values: np.ndarray,
    min_bars: int = 500,
    reestimate_every: int = 252,
    pvalue_threshold: float = 0.05,
    max_d: float = 1.0,
    step: float = 0.05,
) -> tuple[np.ndarray, list[float]]:
    T = len(values)
    result = np.full(T, np.nan)
    d_estimates: list[float] = []

    d_current = 0.40  

    chunk_starts = list(range(min_bars, T, reestimate_every))
    if not chunk_starts:
        fd = _apply_frac_diff(values, d_current)
        return fd, [d_current] * T

    fd_initial = _apply_frac_diff(values[:min_bars], d_current)
    result[:min_bars] = fd_initial
    d_estimates.extend([d_current] * min_bars)

    for i in range(len(chunk_starts)):
        start_idx = chunk_starts[i]
        end_idx = chunk_starts[i+1] if i + 1 < len(chunk_starts) else T
        
        d_current = _find_mmms_d(values[:start_idx], pvalue_threshold=pvalue_threshold)
        fd = _apply_frac_diff(values[:end_idx], d_current)
        
        result[start_idx:end_idx] = fd[start_idx:end_idx]
        d_estimates.extend([d_current] * (end_idx - start_idx))

    return result, d_estimates


def load_and_prepare(
    filepath: str,
    start: str,
    end: str,
    volume_threshold: int = 10000,
    apply_frac_diff: bool = True,
) -> dict:
    print("    [DEBUG] Starting fetch_ohlcv (Polars)...")
    raw_df = fetch_ohlcv(filepath, start, end)
    print("    [DEBUG] fetch_ohlcv complete.")

    if volume_threshold > 0:
        print("    [DEBUG] Starting build_volume_bars (Numba)...")
        bars_df = build_volume_bars(raw_df, volume_threshold)
        print("    [DEBUG] build_volume_bars complete.")
    else:
        bars_df = raw_df.clone()

    fd_close_arr: np.ndarray | None = None
    d_estimates: list[float] = []

    if apply_frac_diff:
        print("    [DEBUG] Starting fractional_differentiation...")
        close_vals = bars_df["close"].to_numpy()
        fd_close_arr, d_estimates = fractional_differentiation(close_vals)
        print("    [DEBUG] fractional_differentiation complete.")

    return {
        "raw_df": raw_df.to_pandas().set_index("date"),
        "bars_df": bars_df.to_pandas().set_index("date"),
        "fd_close": fd_close_arr,
        "d_estimates": d_estimates,
    }
