"""
Lightning-Fast Native SDE Model
Replaces torchsde with a manual single-step Euler-Maruyama loop over pre-compressed features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v7_engine.config import (
    SDE_LATENT_DIM, SDE_DRIFT_DIM_FF, EMBEDDING_DIM, SDE_DT
)

import torch.nn.utils.spectral_norm as spectral_norm

class DriftMLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = EMBEDDING_DIM
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, SDE_DRIFT_DIM_FF)),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.LayerNorm(SDE_DRIFT_DIM_FF),
            spectral_norm(nn.Linear(SDE_DRIFT_DIM_FF, SDE_DRIFT_DIM_FF)),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.LayerNorm(SDE_DRIFT_DIM_FF),
            spectral_norm(nn.Linear(SDE_DRIFT_DIM_FF, SDE_LATENT_DIM))
        )
        
    def forward(self, x):
        return self.net(x)

class DiffusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = EMBEDDING_DIM
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 64)),
            nn.Mish(),
            nn.Dropout(p=0.2),
            spectral_norm(nn.Linear(64, 1))
        )
        
    def forward(self, x):
        # We enforce strictly positive volatility through softplus later
        return self.net(x)

class NeuralSDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.drift_net = DriftMLP()
        self.diffusion_net = DiffusionMLP()
        
        # Dual-Head Predictors
        self.trend_head = nn.Linear(SDE_LATENT_DIM, 3) # Predicts Triple Barrier (-1, 0, 1)
        
        # Projection from embedding to latent state
        self.proj = nn.Linear(EMBEDDING_DIM, SDE_LATENT_DIM)
        self.dt = SDE_DT

    def forward(self, seqs):
        """
        seqs: (batch, EMBEDDING_DIM) 
        Returns paths, final_state, trend_logits, vol_pred
        """
        # seqs is already a flattened feature vector
        
        # Initial State
        h0 = self.proj(seqs)
        
        # 1. Calculate drift and diffusion parameters from the raw features
        drift = self.drift_net(seqs)
        
        # Enforce positive volatility and add epsilon to prevent NaN. Limit extreme explosions.
        diffusion_raw = self.diffusion_net(seqs).squeeze(-1)
        diffusion = torch.clamp(F.softplus(diffusion_raw), min=1e-5, max=10.0)
        
        # 2. Manual Euler-Maruyama Step
        # h_{t+dt} = h_t + drift * dt + diffusion * sqrt(dt) * Z
        # Z is standard normal noise with shape (batch, latent_dim)
        noise = torch.randn_like(h0)
        
        # Note: diffusion is shape (batch), we need to unsqueeze to multiply with noise (batch, latent_dim)
        diffusion_expanded = diffusion.unsqueeze(-1)
        
        h_next = h0 + (drift * self.dt) + (diffusion_expanded * (self.dt ** 0.5) * noise)
        h_next = torch.nan_to_num(h_next, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Triple Barrier Trend Head
        trend_logits = self.trend_head(h_next)
        
        # To match legacy API expectations (paths, final)
        # We only computed one step, so paths is just [h0, h_next]
        # shape: (batch, 2, latent_dim)
        paths = torch.stack([h0, h_next], dim=1)
        
        return paths, h_next, trend_logits, diffusion

    def forward_multi_step(self, seqs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Roll out Euler-Maruyama integration over ts.
        
        NOTE: This implementation assumes the feature context `seqs` remains constant 
        throughout the integration horizon. This is mathematically sound for conditional 
        generative modeling if the horizon is short (e.g. Monte Carlo barrier simulation),
        but may be unrealistic for longer horizons as market microstructure evolves.
        """
        import logging
        logging.getLogger("sde").debug("forward_multi_step: Assumes constant context (seqs) over integration horizon.")
        h = self.proj(seqs)
        paths = [h]
        
        # Assume ts is 1D (T,)
        dt_seq = (ts[1:] - ts[:-1]).tolist()

        for dt in dt_seq:
            if dt <= 0:
                continue
            drift     = self.drift_net(seqs)
            diffusion_raw = self.diffusion_net(seqs).squeeze(-1)
            diffusion = torch.clamp(F.softplus(diffusion_raw), min=1e-5, max=10.0).unsqueeze(-1)
            noise     = torch.randn_like(h)
            h = h + drift * dt + diffusion * (dt ** 0.5) * noise
            paths.append(h)
        return torch.stack(paths, dim=1)   # (batch, len(dt_seq)+1, SDE_LATENT_DIM)
