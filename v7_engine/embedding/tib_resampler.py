"""
TIB Resampler — Forward-fill raw ticks onto a fixed-interval grid.
All price-based features (log returns, momentum, spread) are computed on
the 100ms grid, not raw ticks, to prevent variable-frequency bias.
"""
from __future__ import annotations
import numpy as np


def resample_to_grid(
    timestamps: np.ndarray,
    prices:     np.ndarray,
    grid_ms:    int = 100,
) -> tuple[np.ndarray, bool]:
    """
    Forward-fill *prices* onto a uniform grid with spacing *grid_ms*.

    Parameters
    ----------
    timestamps : seconds (float64), monotonically increasing
    prices     : same length as timestamps
    grid_ms    : grid interval in milliseconds (default 100ms)

    Returns
    -------
    grid_prices : float64 array on the uniform grid
    is_stale    : True if the last tick is > 500ms before the last grid point
    """
    if len(timestamps) < 2:
        return prices.copy(), False

    grid_sec = grid_ms / 1000.0
    t_end    = timestamps[-1]
    # Cap the grid to a maximum of 5 minutes (300 seconds) to prevent massive memory
    # spikes when the 64-tick window crosses a weekend gap.
    t_start  = max(timestamps[0], t_end - 300.0)

    n_grid   = max(int((t_end - t_start) / grid_sec) + 1, 2)
    grid_t   = np.linspace(t_start, t_end, n_grid)

    # Vectorized forward-fill
    idx = np.searchsorted(timestamps, grid_t, side='right') - 1
    idx = np.clip(idx, 0, len(prices) - 1)
    grid_prices = prices[idx]

    # Staleness: gap between the last two ticks > 500ms
    is_stale = (timestamps[-1] - timestamps[-2]) > 0.5 if len(timestamps) > 1 else False

    return grid_prices, is_stale
