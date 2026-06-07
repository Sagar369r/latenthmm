import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMetaLearner(nn.Module):
    def __init__(self, input_dim=17):
        super().__init__()
        
        # We use a shallow network with massive regularization because we are 
        # combining highly abstract features (OHE + Latent) to predict a continuous delta.
        # Too many parameters will instantly overfit to the OOF training set.
        
        self.fc1 = nn.Linear(input_dim, 16)
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(16, 8)
        self.drop2 = nn.Dropout(0.5)
        
        # Continuous output (the Delta penalty)
        self.fc_out = nn.Linear(8, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        
        # No activation on the output layer because Delta can be negative or positive
        out = self.fc_out(x)
        return out

def meta_loss_function(pred_delta, true_delta):
    """
    Standard MSE loss for regression.
    """
    return F.mse_loss(pred_delta, true_delta)
