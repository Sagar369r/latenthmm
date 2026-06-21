"""
Real Toxicity Labeler — Labels real market windows from Dukascopy tick data.

REPLACES: ebm/toxicity_generator.py (GBM synthetic labels — DELETED)

Labels a market state as toxic (1) or clean (0) using:
  • Realised Sortino ratio over the next N ticks
  • Maximum drawdown in the window
  • VaR vs rolling average VaR

Reference: V7 blueprint Section 3.3, Redesign Section 4.3
"""
from __future__ import annotations
import numpy as np
from numba import njit
from v7_engine.config import MAX_DAILY_DRAWDOWN_USD, BACKTEST_INITIAL_EQUITY


@njit(nogil=True)
def label_state_toxicity_jit(
    sortino:    float,
    current_dd: float,
    prop_limit: float,
    var:        float,
    avg_var:    float,
    price:      float,
    kalman_mean: float,
    kalman_std:  float,
) -> int:
    is_toxic = (
        (sortino < -0.5)
        or (current_dd > prop_limit)
        or (var > 1.5 * max(avg_var, 1e-8))
    )
    in_kalman_band = abs(price - kalman_mean) <= 3.0 * max(kalman_std, 1e-8)
    is_clean = (sortino > 0.0) and in_kalman_band

    if is_toxic:
        return 1
    if is_clean:
        return 0
    return -1

@njit(nogil=True)
def fast_label_windows(mid_arr, timestamps, forward_n, stride, equity, prop_limit):
    n = len(mid_arr)
    num_windows = (n - forward_n) // stride + 1
    
    ts_out = np.empty(num_windows, dtype=np.int64)
    lab_out = np.empty(num_windows, dtype=np.int8)
    mid_out = np.empty(num_windows, dtype=np.float64)
    
    var_history = np.zeros(50, dtype=np.float64)
    var_idx = 0
    var_count = 0
    
    out_idx = 0
    
    for i in range(0, n - forward_n + 1, stride):
        window_mid = mid_arr[i : i + forward_n]
        
        # log_rets
        log_rets = np.empty(forward_n - 1, dtype=np.float64)
        for j in range(forward_n - 1):
            val1 = window_mid[j]
            val2 = window_mid[j+1]
            if val1 < 1e-12: val1 = 1e-12
            if val2 < 1e-12: val2 = 1e-12
            log_rets[j] = np.log(val2) - np.log(val1)
            
        # Sortino
        mean_ret = 0.0
        neg_sum = 0.0
        neg_count = 0
        for j in range(forward_n - 1):
            ret = log_rets[j]
            mean_ret += ret
            if ret < 0:
                neg_sum += ret
                neg_count += 1
        mean_ret /= (forward_n - 1)
        
        if neg_count > 0:
            neg_mean = neg_sum / neg_count
            neg_var = 0.0
            for j in range(forward_n - 1):
                if log_rets[j] < 0:
                    neg_var += (log_rets[j] - neg_mean)**2
            down_std = np.sqrt(neg_var / max(neg_count - 1, 1)) + 1e-8
        else:
            down_std = 1e-8
            
        sortino = mean_ret / down_std
        
        # Drawdown
        peak = window_mid[0]
        max_dd_frac = 0.0
        for j in range(forward_n):
            if window_mid[j] > peak:
                peak = window_mid[j]
            dd = (peak - window_mid[j]) / (peak + 1e-8)
            if dd > max_dd_frac:
                max_dd_frac = dd
                
        dd_usd = max_dd_frac * equity
        
        # VaR proxy
        sorted_rets = np.sort(log_rets)
        q_idx = int(0.05 * len(sorted_rets))
        var_now = np.abs(sorted_rets[q_idx]) * equity
        
        var_history[var_idx] = var_now
        var_idx = (var_idx + 1) % 50
        if var_count < 50:
            var_count += 1
            
        avg_var = 0.0
        for j in range(var_count):
            avg_var += var_history[j]
        avg_var /= var_count
        
        # Kalman
        k_mean = 0.0
        for j in range(forward_n):
            k_mean += window_mid[j]
        k_mean /= forward_n
        
        k_var = 0.0
        for j in range(forward_n):
            k_var += (window_mid[j] - k_mean)**2
        k_std = np.sqrt(k_var / forward_n) + 1e-8
        
        label = label_state_toxicity_jit(
            sortino, dd_usd, prop_limit, var_now, avg_var, window_mid[-1], k_mean, k_std
        )
        
        if label != -1:
            ts_out[out_idx] = timestamps[i]
            lab_out[out_idx] = label
            mid_out[out_idx] = mid_arr[i]
            out_idx += 1
            
    return ts_out[:out_idx], lab_out[:out_idx], mid_out[:out_idx]


def label_tick_windows(
    ticks:      dict,
    forward_n:  int   = 100,
    stride:     int   = 50,
    equity:     float = BACKTEST_INITIAL_EQUITY,
    prop_limit: float = MAX_DAILY_DRAWDOWN_USD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replay real Dukascopy tick data and label each window.

    Parameters
    ----------
    ticks      : dict with keys bid, ask, timestamp_ns, delta_t, sign
    forward_n  : number of ticks ahead used to compute realised metrics
    stride     : step size between windows
    equity     : account equity for USD conversion
    prop_limit : prop firm daily drawdown limit USD

    Returns
    -------
    timestamps : (N,) int64 — timestamp of each labelled window
    labels     : (N,) int8  — 1=toxic, 0=clean (-1 excluded)
    mid_prices : (N,) float64 — mid price at each window
    """
    bids = ticks["bid"]
    asks = ticks["ask"]
    n    = len(bids)
    mid  = (bids + asks) / 2.0

    timestamps_arr = ticks.get("timestamp_ns", np.arange(n) * int(1e8)).astype(np.int64)

    return fast_label_windows(
        mid.astype(np.float64), 
        timestamps_arr, 
        int(forward_n), 
        int(stride), 
        float(equity), 
        float(prop_limit)
    )

# Backward-compatible alias for tests
label_state_toxicity = label_state_toxicity_jit

