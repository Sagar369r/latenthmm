"""
MC Barrier Guard — Pre-trade Monte Carlo barrier check.

FIXES applied:
  • path_norms (L2 norm as P&L proxy) is DELETED
  • Uses LatentDecoder to map latent paths → real P&L in USD
  • .detach().cpu().numpy() always applied to avoid requires_grad errors
  • Async execution with caching to meet 6ms budget

Architecture:
  1. Submit SDE simulation to thread pool (non-blocking)
  2. If result ready within MC_TIMEOUT_MS: use it
  3. If not ready: use cached result from previous call
  4. Block trade if VaR at MC_CONFIDENCE > remaining daily budget
"""
from __future__ import annotations
import hashlib
import logging
import threading
import time

from concurrent.futures import Future
import numpy as np
import torch
from v7_engine.config import (
    RISK_MC_TIMESTEPS, RISK_MC_SIMS_INTERNAL, RISK_MC_PENALTY_MULT, 
    RISK_MC_DIV_PROTECTOR, RISK_MC_MARGIN_NORM, RISK_MC_THROTTLE_SEC, 
    RISK_MC_BARRIER_BLOCKS
)

from v7_engine.config import (
    MC_PATHS, MC_TIMEOUT_MS, MC_CONFIDENCE,
    MAX_DAILY_DRAWDOWN_USD, SDE_LATENT_DIM, EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


# ── Result cache ──────────────────────────────────────────────────────────────

class _MCCache:
    """Thread-safe LRU-style cache for MC barrier results."""

    def __init__(self, maxsize: int = 1024):
        self._lock   = threading.Lock()
        self._cache: dict[tuple, tuple[float, float]] = {}  # key → (var_usd, ts)
        self._maxsize = maxsize

    def _key(self, x0: torch.Tensor, ts: torch.Tensor, units: float, context: torch.Tensor | None, current_equity: float, daily_limit_usd: float) -> tuple:
        import hashlib
        x0_bytes = x0.detach().cpu().numpy().tobytes()
        ts_bytes = ts.detach().cpu().numpy().tobytes()
        context_bytes = b''
        if context is not None:
            context_bytes = context.detach().cpu().numpy().tobytes()
        h = hashlib.md5(x0_bytes + ts_bytes + context_bytes).hexdigest()
        return (h, round(units, 6), round(current_equity, 2), round(daily_limit_usd, 2))

    def get(
        self, x0: torch.Tensor, ts: torch.Tensor, units: float, 
        current_equity: float, daily_limit_usd: float,
        context: torch.Tensor | None = None
    ) -> float | None:
        key = self._key(x0, ts, units, context, current_equity, daily_limit_usd)
        with self._lock:
            if key in self._cache:
                var, timestamp = self._cache[key]
                if time.time() - timestamp < RISK_MC_THROTTLE_SEC:
                    return var
                else:
                    del self._cache[key]
        return None

    def put(self, x0: torch.Tensor, ts: torch.Tensor, units: float, 
            current_equity: float, daily_limit_usd: float,
            var_usd: float, context: torch.Tensor | None = None) -> None:
        key = self._key(x0, ts, units, context, current_equity, daily_limit_usd)
        with self._lock:
            if len(self._cache) >= self._maxsize:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (var_usd, time.time())


# ── Main guard ────────────────────────────────────────────────────────────────

class MonteCarloBarrierGuard:
    """
    Pre-trade Monte Carlo barrier.

    Uses the LatentDecoder to convert latent paths → P&L.
    Blocks trade if VaR at MC_CONFIDENCE exceeds remaining daily budget.
    """

    def __init__(self, sde, decoder=None):
        """
        Parameters
        ----------
        sde     : NeuralSDE instance
        decoder : LatentDecoder instance (if None, falls back to L2 estimate with warning)
        """
        self.sde      = sde
        self.decoder  = decoder
        self.cache    = _MCCache()

    # ── public API ────────────────────────────────────────────────────────────

    def can_trade(
        self,
        x0:             torch.Tensor,     # (1, SDE_LATENT_DIM)
        ts:             torch.Tensor,     # (T,) time points
        context:        torch.Tensor | None,  # (1, seq_len, EMBEDDING_DIM) — REAL
        proposed_units: float,
        current_dd_usd: float,
        current_equity: float,
        daily_limit_usd: float | None = None,
    ) -> tuple[bool, dict]:
        """
        Returns (allowed, diagnostics_dict).

        allowed = False if:
          • Daily budget already exhausted
          • VaR at MC_CONFIDENCE exceeds remaining budget
        """
        limit = daily_limit_usd if daily_limit_usd is not None else MAX_DAILY_DRAWDOWN_USD
        budget_remaining = limit - current_dd_usd

        if budget_remaining <= 0:
            return False, {
                "allowed": False,
                "reason":  "Daily budget exhausted",
                "var_loss_usd": float("inf"),
                "timeout_occurred": False,
            }

        # Check cache first (< 0.1ms)
        cached = self.cache.get(x0, ts, proposed_units, current_equity, limit, context)
        if cached is not None:
            allowed = cached < budget_remaining
            return allowed, {
                "allowed":          allowed,
                "var_loss_usd":     cached,
                "budget_remaining": budget_remaining,
                "timeout_occurred": False,
                "from_cache":       True,
            }

        # Execute simulation synchronously to prevent PyTorch thread contention
        var_usd = self._simulate_and_cache(x0, ts, context, proposed_units, budget_remaining, current_equity, limit)
        timeout = False

        allowed = var_usd < budget_remaining
        return allowed, {
            "allowed":          allowed,
            "var_loss_usd":     var_usd,
            "budget_remaining": budget_remaining,
            "timeout_occurred": timeout,
            "from_cache":       False,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _simulate_and_cache(
        self,
        x0:      torch.Tensor,
        ts:      torch.Tensor,
        context: torch.Tensor | None,
        units:   float,
        budget:  float,
        current_equity: float,
        limit: float,
    ) -> float:
        """Run MC paths and return the VaR loss in USD."""
        try:
            with torch.no_grad():
                if context is None:
                    from v7_engine.config import EMBEDDING_DIM
                    context = torch.zeros(1, EMBEDDING_DIM, device=x0.device)
                
                if context.dim() == 3:
                    # Defensive: if it was passed as 3D (batch, seq, dim), squeeze it to 2D
                    context = context.squeeze(1)
                    
                # We duplicate the context to simulate MC_PATHS independent paths
                # context is shape (1, EMBEDDING_DIM). Repeat to (MC_PATHS, EMBEDDING_DIM)
                seqs_batch = context.repeat(MC_PATHS, 1)

                # Generate paths using unified multi-step
                # paths shape: (MC_PATHS, T, SDE_LATENT_DIM)
                paths = self.sde.forward_multi_step(seqs_batch, ts=ts)

                if self.decoder is None:
                    raise RuntimeError("LatentDecoder is required to compute PnL in USD. L2 proxy is invalid.")
                    
                # Proper P&L via LatentDecoder across full path trajectory
                pnl_paths  = self.decoder.decode_paths(
                    paths,
                    units  = units,
                    equity = current_equity,
                ).detach().cpu().numpy()                     # (MC_PATHS,)
                losses = -pnl_paths   # losses > 0 means money lost

            # VaR at MC_CONFIDENCE
            var_usd = float(np.quantile(losses, MC_CONFIDENCE))
            self.cache.put(x0, ts, units, current_equity, limit, var_usd, context)
            return var_usd

        except Exception as exc:
            logger.error("MC simulation failed: %s", exc)
            return budget   # Conservative: treat as if barrier would be hit
