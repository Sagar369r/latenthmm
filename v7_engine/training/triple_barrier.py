import numpy as np
from numba import njit

@njit(nogil=True)
def fast_triple_barrier(bids, asks, tns, out_tick_idx, out_vars, 
                        pt_mult=2.0, sl_mult=2.0, max_ticks=10000):
    """
    Computes Triple Barrier Target Labels.
    
    Barriers:
      - Upper (Profit Take): Current Price + pt_mult * Rolling Volatility
      - Lower (Stop Loss): Current Price - sl_mult * Rolling Volatility
      - Time (Max Ticks): Reached max_ticks before hitting either barrier
      
    Outputs labels:
       1 if Upper Barrier hit first
      -1 if Lower Barrier hit first
       0 if Time Barrier hit first
       
    Args:
        bids (np.ndarray): Full raw bids array
        asks (np.ndarray): Full raw asks array
        tns (np.ndarray): Full raw timestamps (ns)
        out_tick_idx (np.ndarray): Array mapping feature rows to the tick index they were extracted at
        out_vars (np.ndarray): Realized rolling variance per feature row
        pt_mult (float): Profit-take volatility multiplier
        sl_mult (float): Stop-loss volatility multiplier
        max_ticks (int): Maximum look-forward horizon (Barrier 3)
        
    Returns:
        labels (np.ndarray): Triple barrier labels (-1, 0, 1) mapping to out_tick_idx
    """
    n_features = len(out_tick_idx)
    labels = np.zeros(n_features, dtype=np.float64)
    n_ticks = len(bids)
    
    for i in range(n_features):
        tick_idx = int(out_tick_idx[i])
        
        # Guard against edges
        if tick_idx >= n_ticks - 1:
            labels[i] = 0.0
            continue
            
        current_mid = (bids[tick_idx] + asks[tick_idx]) / 2.0
        
        # Volatility is stored as variance in out_vars. Convert to standard deviation price scale
        # Assuming out_vars is log variance, but actually it is variance of log returns.
        # Approx standard deviation of price = current_price * sqrt(variance)
        return_std = np.sqrt(out_vars[i])

        # Dynamic barrier limits
        upper_barrier = current_mid * (1.0 + pt_mult * return_std)
        lower_barrier = current_mid * (1.0 - sl_mult * return_std)
        
        horizon = min(tick_idx + max_ticks, n_ticks)
        hit = 0.0
        
        # Look forward
        for j in range(tick_idx + 1, horizon):
            mid_fwd = (bids[j] + asks[j]) / 2.0
            
            if mid_fwd >= upper_barrier:
                hit = 1.0
                break
            elif mid_fwd <= lower_barrier:
                hit = -1.0
                break
                
        labels[i] = hit
        
    return labels
