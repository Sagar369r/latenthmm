"""
Energy-Based Model (EBM) — E_γ(X_t).

Low energy → clean market state (tradeable).
High energy → toxic market state (blocked).

Trained with Contrastive Loss (BCE) on real toxicity labels from Dukascopy data.
Uses Spectral Normalization to enforce a Lipschitz constant of 1, guaranteeing
a smooth energy manifold and robustness against out-of-distribution spikes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from v7_engine.config import (
    SDE_LATENT_DIM, EBM_ENERGY_THRESHOLD
)


class EnergyModel(nn.Module):
    """
    E_γ : R^{SDE_LATENT_DIM} → R

    Lower energy = cleaner/more-tradeable market state.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(SDE_LATENT_DIM, 128)),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.LayerNorm(128),
            spectral_norm(nn.Linear(128, 64)),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.LayerNorm(64),
            spectral_norm(nn.Linear(64, 32)),
            nn.Mish(),
            nn.Dropout(p=0.2),
            spectral_norm(nn.Linear(32, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar energy per sample. Shape: (batch,)"""
        return self.net(x).squeeze(-1)

    # ── inference ─────────────────────────────────────────────────────────────

    def is_tradeable(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (tradeable, energy_value) as Tensors.
        tradeable = True only if energy <= EBM_ENERGY_THRESHOLD.
        """
        with torch.no_grad():
            energy = self.forward(x)
        return energy <= EBM_ENERGY_THRESHOLD, energy


