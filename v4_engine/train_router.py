import os
import sys
import polars as pl
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.feature_expansion import extract_z_score_tensor
from v4_engine.vae_model import VAE
from v4_engine.gmm_hmm import LatentRouter

def train_full_pipeline(csv_paths: list[str]):
    print(f"=== Phase 1 & 2: Generating Latent Space for Multi-Asset Basket ===")
    
    all_latents = []
    all_returns = []
    
    # Load VAE Model
    vae = VAE(input_dim=20, latent_dim=4)
    vae_path = "v4_engine/models/vae_weights.pth"
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, weights_only=True))
    else:
        print("WARNING: Using untrained VAE weights!")
    
    vae.eval()
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_path} - not found.")
            continue
            
        print(f"Processing {csv_path}...")
        df = pl.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        # ── Chronological Firewall (2022-12-31) ──
        UNSUPERVISED_TRAIN_END = 1672444800000
        if "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") <= UNSUPERVISED_TRAIN_END)
        
        # Extract 20 Dimensionless Features
        df_feat, z_cols = extract_z_score_tensor(df, window=90)
        
        # We skip the 90-period warmup where rolling windows contain nulls
        features_np = df_feat.select(z_cols).to_numpy()[90:]
        close_series = df_feat["close"]
        log_return = (close_series / close_series.shift(1).fill_null(strategy="backward")).log()
        returns_np = log_return.to_numpy()[90:]
        
        # Convert to tensor and pass through VAE to get 4 Latent Features
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        
        with torch.no_grad():
            mu, _ = vae.encode(features_tensor)
            latent_X = mu.numpy()
            
        all_latents.append(latent_X)
        all_returns.append(returns_np)
        
    if not all_latents:
        print("No valid CSVs provided.")
        return
        
    combined_latents = np.vstack(all_latents)
    combined_returns = np.concatenate(all_returns)
    
    print(f"Combined Latent Tensor Shape: {combined_latents.shape}")
    
    print("\n=== Phase 3: Training GMM-HMM Router ===")
    router = LatentRouter(n_components=3, n_mix=2)
    router.fit(combined_latents)
    
    # Apply the Heuristic Mapper
    router._apply_heuristic_mapping(combined_latents, combined_returns)
    
    # Save the Frozen Router
    os.makedirs("v4_engine/models", exist_ok=True)
    router.save("v4_engine/models/gmm_router.pkl")
    print("\n✓ Pipeline execution complete. Frozen Router saved to v4_engine/models/gmm_router.pkl")

if __name__ == "__main__":
    target_pairs = [
        "data/audcad_1h.csv",
        "data/chfjpy_1h.csv",
        "data/eurnzd_1h.csv"
    ]
    train_full_pipeline(target_pairs)
