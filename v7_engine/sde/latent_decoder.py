"""
Latent Decoder — Maps SDE latent state → price log-return.

PURPOSE:
  Replaces the L2 norm as MC barrier proxy. Instead of using path norms
  as a fake P&L proxy, we train a small MLP on paired (latent_state,
  future_return) data from real Dukascopy ticks.

TRAINING:
  See training/train_latent_decoder.py
  Input:  latent state z ∈ R^{SDE_LATENT_DIM}
  Target: log(mid_{t+horizon} / mid_t) — actual price change

USAGE in MC barrier:
  decoder = LatentDecoder.load("models/latent_decoder.pth")
  log_ret = decoder.decode(latent_paths)   # (n_paths, horizon)
  pnl_usd = log_ret * position_units * equity
"""
from __future__ import annotations
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from v7_engine.config import SDE_LATENT_DIM

logger = logging.getLogger(__name__)


class LatentDecoder(nn.Module):
    """
    Small MLP: R^{SDE_LATENT_DIM} → R^1 (log price change).

    Architecture chosen to be fast (< 0.1ms inference) so it does not
    blow the 6ms MC barrier budget.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SDE_LATENT_DIM, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden // 2),
            nn.Mish(),
            nn.Linear(hidden // 2, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, SDE_LATENT_DIM) or (n_paths, batch, L)

        Returns
        -------
        log_return : (batch,) or (n_paths, batch)
        """
        orig_shape = z.shape[:-1]
        z_flat = z.reshape(-1, SDE_LATENT_DIM)
        out = self.net(z_flat).squeeze(-1)
        return out.reshape(orig_shape)

    # ── training helpers ──────────────────────────────────────────────────────

    def train_on_pairs(
        self,
        latent_states:  np.ndarray,   # (N, SDE_LATENT_DIM)
        future_returns: np.ndarray,   # (N,) log price changes
        epochs: int = 50,
        lr:     float = 1e-3,
        batch:  int = 256,
    ) -> list[float]:
        """
        Train the decoder on (latent_state, future_return) pairs.
        All data must come from real Dukascopy ticks.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optim  = torch.optim.Adam(self.parameters(), lr=lr)
        X = torch.tensor(latent_states,  dtype=torch.float32)
        y = torch.tensor(future_returns, dtype=torch.float32)
        losses = []
        for ep in range(epochs):
            self.train()
            idx = torch.randperm(len(X))
            ep_loss = 0.0
            for i in range(0, len(X), batch):
                bi = idx[i : i + batch]
                xb, yb = X[bi].to(device), y[bi].to(device)
                pred    = self.forward(xb)
                loss    = nn.functional.mse_loss(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
                ep_loss += loss.item()
            avg = ep_loss / max(len(X) // batch, 1)
            losses.append(avg)
            if ep % 10 == 0:
                logger.info("LatentDecoder epoch %d | MSE %.6f", ep, avg)
        return losses

    def decode_paths(
        self,
        latent_paths: torch.Tensor,   # (n_paths, T, SDE_LATENT_DIM)
        units:        float,
        equity:       float,
        pip_value:    float = 10.0,
    ) -> torch.Tensor:
        """
        Convert latent paths → P&L paths in USD.

        Parameters
        ----------
        latent_paths : SDE output (n_paths, T, L)
        units        : position size in lots
        equity       : current account equity
        pip_value    : USD per pip per lot (default 10 for EURUSD mini)

        Returns
        -------
        pnl_paths : (n_paths,) — cumulative P&L per path
        """
        with torch.no_grad():
            # Decode each time step
            log_rets = self.forward(latent_paths)  # (n_paths, T)
            # Cumulative price change: sum of log returns
            cum_log_ret = log_rets.sum(dim=-1)     # (n_paths,)
            # Convert to USD P&L: P&L = (exp(cum_log_ret) - 1) × equity × position_fraction
            pnl = (torch.exp(cum_log_ret) - 1.0) * equity * units
        return pnl

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "models/latent_decoder.pth") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("LatentDecoder saved → %s", path)

    @classmethod
    def load(cls, path: str = "models/latent_decoder.pth") -> "LatentDecoder":
        model = cls()
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            logger.info("LatentDecoder loaded ← %s", path)
        else:
            logger.warning("LatentDecoder checkpoint not found at %s — using uninitialised weights", path)
        model.eval()
        return model
