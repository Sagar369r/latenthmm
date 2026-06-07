import os
import sys
import polars as pl
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from v4_engine.feature_expansion import extract_z_score_tensor
from v4_engine.vae_model import VAE, vae_loss_function

def test_feature_expansion():
    fx_path = "data/eurgbp_daily.csv"
    assert os.path.exists(fx_path)
    
    df = pl.read_csv(fx_path)
    df.columns = [c.lower() for c in df.columns]
    
    df_feat, z_cols = extract_z_score_tensor(df, window=90)
    
    assert len(z_cols) == 20, f"Expected 20 Z-score features, got {len(z_cols)}"
    
    # Check that they are all mean ~0, std ~1 after warmup
    for col in z_cols:
        arr = df_feat[col].to_numpy()[90:]
        assert not np.isnan(arr).any(), f"NaN in {col}"
        mean = np.mean(arr)
        std = np.std(arr)
        assert abs(mean) < 0.2, f"{col} mean too far from 0: {mean}"
        assert 0.6 < std < 1.4, f"{col} std out of bounds: {std}"
        
    print("✓ Feature Expansion: 20 dimensionless indicators successfully generated.")
    return df_feat, z_cols

def test_vae_architecture():
    # Create fake batch of 20-dim features
    batch_size = 64
    x = torch.randn(batch_size, 20)
    
    model = VAE(input_dim=20, latent_dim=4)
    model.train() # BN requires >1 batch size and training mode
    
    recon_x, mu, logvar = model(x)
    
    assert recon_x.shape == (batch_size, 20), "Reconstruction shape mismatch"
    assert mu.shape == (batch_size, 4), "Latent mean shape mismatch"
    assert logvar.shape == (batch_size, 4), "Latent logvar shape mismatch"
    
    loss, recon, kld = vae_loss_function(recon_x, x, mu, logvar)
    
    # Test gradients
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        
    print("✓ VAE Architecture: 20 -> 4 -> 20 compression successful. Gradients intact.")

if __name__ == "__main__":
    test_feature_expansion()
    test_vae_architecture()
