import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim: int = 20, latent_dim: int = 4):
        super().__init__()
        
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, 16)
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.enc_fc2 = nn.Linear(16, 8)
        self.enc_bn2 = nn.BatchNorm1d(8)
        
        # Latent Space
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 8)
        self.dec_bn1 = nn.BatchNorm1d(8)
        self.dec_fc2 = nn.Linear(8, 16)
        self.dec_bn2 = nn.BatchNorm1d(16)
        self.dec_fc3 = nn.Linear(16, input_dim)

    def encode(self, x):
        x = F.relu(self.enc_bn1(self.enc_fc1(x)))
        x = F.relu(self.enc_bn2(self.enc_fc2(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        z = F.relu(self.dec_bn1(self.dec_fc1(z)))
        z = F.relu(self.dec_bn2(self.dec_fc2(z)))
        # No activation on final layer since outputs are Z-scores (-10 to 10)
        return self.dec_fc3(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the VAE loss function.
    KL divergence is weighted by beta.
    """
    # Reconstruction loss (MSE since our inputs are unbounded Z-scores)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld, recon_loss, kld
