"""
RL Actor — PPO policy network for position size + direction.

Inputs:  Latent state from Neural SDE (SDE_LATENT_DIM)
Outputs: (direction logit, Beta distribution params for size)

Training uses REAL context replay buffer from Dukascopy ticks.
No dummy_context, no torch.zeros context.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from v7_engine.config import SDE_LATENT_DIM, MAX_POSITION_FRACTION


class RLActor(nn.Module):
    """
    Actor for PPO.
    Outputs:
      • dir_logit : scalar logit → Bernoulli(sigmoid(logit)) for direction
      • alpha, beta: Beta distribution params for position size ∈ [0, MAX_POSITION_FRACTION]
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(SDE_LATENT_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.dir_head  = nn.Linear(32, 1)
        self.size_head = nn.Linear(32, 2)   # → (alpha, beta)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonal initialization is the golden standard for PPO
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
        # Scale down final layers to prevent early deterministic collapse
        nn.init.orthogonal_(self.dir_head.weight, gain=0.01)
        nn.init.orthogonal_(self.size_head.weight, gain=0.01)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        dir_logit : (batch,)
        alpha     : (batch,) — Beta param > 1
        beta_     : (batch,) — Beta param > 1
        """
        h         = self.backbone(x)
        dir_logit = self.dir_head(h).squeeze(-1)
        ab        = nn.functional.softplus(self.size_head(h)) + 1.0   # α, β > 1
        return dir_logit, ab[:, 0], ab[:, 1]

    def sample_action(self, x: torch.Tensor) -> dict:
        """
        Sample a deterministic (greedy) action for live inference.

        Returns dict with: direction, size_fraction, dir_prob as Tensors of shape (batch,)
        """
        with torch.no_grad():
            dir_logit, alpha, beta_ = self.forward(x)
            dir_prob  = torch.sigmoid(dir_logit)
            direction = torch.where(dir_prob > 0.5, torch.tensor(1, device=x.device), torch.tensor(-1, device=x.device))
            # Use Beta mean for deterministic inference (not a sample)
            size_mean = alpha / (alpha + beta_)
            size_frac = size_mean  # Callers will scale by MAX_POSITION_FRACTION
        return {
            "direction":     direction,
            "size_fraction": size_frac,
            "dir_prob":      dir_prob,
        }


class RLCritic(nn.Module):
    """State value network for PPO critic."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SDE_LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
