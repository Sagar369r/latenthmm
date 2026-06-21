import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class RegimeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
    def predict(self, kalman_vel_var: float, cusum_energy: float, wavelet_hf_energy: float) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[kalman_vel_var, cusum_energy, wavelet_hf_energy]], dtype=torch.float32)
            probs = self.forward(x)
            return probs.squeeze(0).numpy()
def compute_sideways_confidence(kalman_vel_var: float, obs_noise: float = 1e-4) -> float:
    return float(np.clip(1.0 / (1.0 + kalman_vel_var / obs_noise), 0.0, 1.0))

