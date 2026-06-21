import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class TickSanitizer:
    """
    Strict mathematical gating for broker data feeds.
    Protects the ML engine from out-of-order packets, crossed spreads,
    and toxic fat-finger spikes.
    """
    def __init__(self, max_spread_pips: float = 20.0, mad_multiplier: float = 5.0, window_size: int = 60, pip_scale: float = 10000.0):
        self.max_spread = max_spread_pips / pip_scale
        self.mad_multiplier = mad_multiplier
        self.window_size = window_size
        
        self.last_ts = -1.0
        self.last_mid = None
        self.price_history = deque(maxlen=window_size)
        
    def sanitize(self, timestamp: float, bid: float, ask: float) -> bool:
        """
        Returns True if the tick is valid and should be processed.
        Returns False if the tick is toxic and should be dropped.
        """
        # 1. Monotonic Time Enforcement
        if timestamp <= self.last_ts:
            return False
            
        # 2. Bid-Ask Spread Validation (Ensure Ask > Bid and Spread < Max)
        spread = ask - bid
        if spread <= 0 or spread > self.max_spread:
            return False
            
        mid = (bid + ask) / 2.0
        
        # 3. Fat Finger / Spike Filtration
        if len(self.price_history) >= 10 and self.last_mid is not None:
            arr = np.array(self.price_history)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            
            if mad < 1e-6:
                mad = 1e-5
                
            jump = abs(mid - self.last_mid)
            if jump > self.mad_multiplier * mad:
                logger.warning(f"Toxic spike detected: jump={jump:.5f}, mad={mad:.5f}. Dropping tick.")
                return False
                
        # Passed all gates
        self.last_ts = timestamp
        self.last_mid = mid
        self.price_history.append(mid)
        return True
